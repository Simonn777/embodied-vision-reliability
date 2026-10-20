"""Fixed TRAIN reference-pair information screen. No work or ML imports on import."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

TRAIN = ("dog-agility", "planes-water", "dogs-scale", "dog-gooses")
ARMS = ("paired", "unpaired_max", "deranged", "native")
OFFSETS = (0, -1, 1, -2, 2)
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
BINDINGS = ("cache_receipt", "cache_outer", "cache_operation", "cache_worker_spec",
            "mapping", "data_index")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def module(name, path):
    loader = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(loader)
    sys.modules[name] = result
    loader.loader.exec_module(result)
    return result


def contract():
    return dict(train=list(TRAIN), anchors_per_parent=12, horizons=[1, 3], events=96,
        feature_level=0, expected_feature_shape_hwc=[256, 256, 32],
        reference_cap=128, current_crossing_cap=64, grid=[8, 8],
        radius="ceil(0.008*hypot(H,W))", offset_radius_multipliers=list(OFFSETS),
        endpoints="center + (offset +/- radius)*outward_axis",
        coordinate_convention="original pixel centers; half-pixel edge centers",
        label_lookup="floor(xy+0.5); reject out-of-bounds, never clip endpoints",
        feature_lookup="align_corners=False bilinear with feature-border replication",
        similarity="float64 cosine; normalize after interpolation; exact zero unavailable",
        sampling="row-major grid round-robin; within cell horizontal then vertical, y then x",
        reference_filter="initial endpoints target/non-target; neither endpoint void 255",
        derangement="SHA256 evr0031-pairing-v1 + sequence + index ordering, one-cycle shift",
        ranking="maximize signed score; exact ties use [0,-r,+r,-2r,+2r] order",
        fallback="fewer than two usable reference pairs or invalid native descriptors -> native",
        gt_lookup="same-orientation scanline crossing; absent crossing has capped loss 1",
        void_policy="current 255 background, counted; initial void endpoints excluded",
        aggregation="equal queries per event; empty event deltas zero; equal 24 events per parent; equal 4 parents",
        scientific_scope="TRAIN candidate-ranking information only; no whole-mask or novelty claim")


def crossings(mask):
    """All internal adjacent-pixel crossings, oriented inside -> outside."""
    import numpy as np
    require(mask.ndim == 2 and mask.dtype == np.bool_, "boolean mask required")
    rows = []
    for axis in (0, 1):
        difference = (mask[:, :-1] != mask[:, 1:]) if axis == 0 else (mask[:-1] != mask[1:])
        for y, x in np.argwhere(difference):
            sign = 1 if bool(mask[y, x]) else -1
            rows.append(dict(axis=axis, sign=sign,
                center=[float(x) + (0.5 if axis == 0 else 0),
                        float(y) + (0.5 if axis == 1 else 0)]))
    return rows


def spread(rows, shape, cap):
    """Geometry-only fixed cap. Never refill after inspecting features or current GT."""
    h, w = shape
    buckets = [[] for _ in range(64)]
    for row in rows:
        x, y = row["center"]
        cell = min(7, int(8*y/h))*8 + min(7, int(8*x/w))
        buckets[cell].append(row)
    selected, depth = [], 0
    while len(selected) < cap:
        layer = [bucket[depth] for bucket in buckets if depth < len(bucket)]
        if not layer:
            break
        selected.extend(layer[:cap-len(selected)])
        depth += 1
    return selected


def endpoints(row, offset, radius, shape):
    import numpy as np
    center = np.array(row["center"], dtype=np.float64)
    direction = np.zeros(2, dtype=np.float64)
    direction[row["axis"]] = row["sign"]
    center = center + offset*direction
    xy = np.stack((center-radius*direction, center+radius*direction))
    h, w = shape
    valid = bool(np.isfinite(xy).all() and (xy >= 0).all()
                 and (xy[:, 0] <= w-1).all() and (xy[:, 1] <= h-1).all())
    return center, xy, valid


def label_pair(label, xy, target):
    import numpy as np
    ij = np.floor(xy+0.5).astype(np.int64)
    values = label[ij[:, 1], ij[:, 0]]
    return [bool(v == target) for v in values], int(np.count_nonzero(values == 255))


def descriptors(feature, xy, image_shape):
    """Features Hf,Wf,C; resize geometry agrees with native align_corners=False."""
    import numpy as np
    h, w = image_shape
    fh, fw, _ = feature.shape
    x = np.clip((xy[:, 0]+0.5)*fw/w-0.5, 0, fw-1)
    y = np.clip((xy[:, 1]+0.5)*fh/h-0.5, 0, fh-1)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0+1, fw-1), np.minimum(y0+1, fh-1)
    wx, wy = (x-x0)[:, None], (y-y0)[:, None]
    values = ((1-wy)*((1-wx)*feature[y0, x0]+wx*feature[y0, x1])
              + wy*((1-wx)*feature[y1, x0]+wx*feature[y1, x1])).astype(np.float64)
    require(np.isfinite(values).all(), "nonfinite feature interpolation")
    scales = np.max(np.abs(values), axis=1)
    valid = scales > 0
    values = values/np.where(valid, scales, 1)[:, None]
    norms = np.sqrt(np.sum(values*values, axis=1))
    values /= np.where(valid, norms, 1)[:, None]
    return values, valid


def derangement(count, sequence):
    require(count >= 2, "derangement requires two pairs")
    order = sorted(range(count), key=lambda i: (
        hashlib.sha256(f"evr0031-pairing-v1\0{sequence}\0{i}".encode()).hexdigest(), i))
    result = [None]*count
    for position, index in enumerate(order):
        result[index] = order[(position+1) % count]
    return result


def reference_bank(initial, feature, target, sequence, radius):
    import numpy as np
    geometries, rejected = [], 0
    raw = crossings(initial == target)
    for row in raw:
        _, xy, valid = endpoints(row, 0, radius, initial.shape)
        labels, void = label_pair(initial, xy, target) if valid else (None, None)
        if valid and labels == [True, False] and void == 0:
            geometries.append(row)
        else:
            rejected += 1
    selected = spread(geometries, initial.shape, 128)
    used, vectors, invalid = [], [], []
    for index, row in enumerate(selected):
        _, xy, _ = endpoints(row, 0, radius, initial.shape)
        values, valid = descriptors(feature, xy, initial.shape)
        if bool(valid.all()):
            used.append(dict(selection_index=index, **row)); vectors.append(values)
        else:
            invalid.append(index)
    array = np.stack(vectors) if vectors else np.empty((0, 2, feature.shape[-1]))
    permutation = derangement(len(used), sequence) if len(used) >= 2 else None
    metadata = dict(raw_crossings=len(raw), geometry_eligible=len(geometries),
        geometry_rejected=rejected, selected_count=len(selected), usable_count=len(used),
        zero_descriptor_selection_indices=invalid, selected=selected, used=used,
        outside_permutation=permutation,
        normalized_descriptors_sha256=hashlib.sha256(array.astype("<f8").tobytes()).hexdigest())
    return array, permutation, metadata


def scores(xy_descriptors, bank, permutation):
    import numpy as np
    x, y = xy_descriptors
    u, v = bank[:, 0], bank[:, 1]
    xu, xv, yu, yv = u@x, v@x, u@y, v@y
    plus, minus = xu+yv, xv+yu
    pp, pm = float(np.max(plus)), float(np.max(minus))
    up, um = float(np.max(xu)+np.max(yv)), float(np.max(xv)+np.max(yu))
    dp, dm = xu+yv[permutation], xv[permutation]+yu
    return dict(paired=pp-pm, unpaired_max=up-um,
        deranged=float(np.max(dp)-np.max(dm))), dict(
        forward=pp, reverse=pm, forward_gap=up-pp, reverse_gap=um-pm,
        forward_index=int(np.argmax(plus)), reverse_index=int(np.argmax(minus)))


def rank_frame(prediction, feature, bank, permutation, radius):
    raw = crossings(prediction)
    sampled = spread(raw, prediction.shape, 64)
    queries = []
    for index, row in enumerate(sampled):
        candidates = []
        for multiplier in OFFSETS:
            offset = multiplier*radius
            center, xy, valid = endpoints(row, offset, radius, prediction.shape)
            candidate = dict(offset_pixels=offset, center=center.tolist(), endpoints=xy.tolist(),
                             status="unavailable", reason="out_of_image", scores=None)
            if valid:
                vectors, good = descriptors(feature, xy, prediction.shape)
                candidate["reason"] = "zero_descriptor"
                if bool(good.all()):
                    candidate.update(status="completed", reason=None)
                    if len(bank) >= 2:
                        candidate["scores"], candidate["paired_components"] = scores(vectors, bank, permutation)
            candidates.append(candidate)
        fallback = ("reference_bank_below_two" if len(bank) < 2 else
                    "native_candidate_unavailable" if candidates[0]["status"] != "completed" else None)
        selection = {}
        for arm in ARMS:
            chosen = 0 if arm == "native" or fallback else min(
                (i for i, c in enumerate(candidates) if c["status"] == "completed"),
                key=lambda i: (-candidates[i]["scores"][arm], i))
            selection[arm] = dict(index=chosen, fallback_reason=fallback if arm != "native" else None,
                                  displacement_pixels=candidates[chosen]["offset_pixels"])
        queries.append(dict(index=index, **row, candidates=candidates, selection=selection))
    return dict(raw_crossings=len(raw), sampled_crossings=len(sampled), queries=queries)


def candidate_metric(query, candidate, label, target, radius, gt_lines):
    import numpy as np
    center = candidate["center"]
    axis, sign = query["axis"], query["sign"]
    line = int(query["center"][1-axis])
    possibilities = gt_lines.get((axis, sign, line), [])
    nearest = min(possibilities, key=lambda value: (abs(center[axis]-value), value)) if possibilities else None
    signed = sign*(center[axis]-nearest) if nearest is not None else None
    localization = min(abs(signed)/(2*radius), 1.0) if signed is not None else 1.0
    _, xy, valid = endpoints(query, candidate["offset_pixels"], radius, label.shape)
    pair, void = label_pair(label, xy, target) if valid else (None, 0)
    state = ("out_of_image" if not valid else "correct" if pair == [True, False] else
             "reversed" if pair == [False, True] else "both_target" if pair == [True, True]
             else "both_background")
    return dict(endpoint_class=state, correct=float(state == "correct"),
        noncrossing=float(state in ("both_target", "both_background")),
        reversed=float(state == "reversed"), out_of_image=float(not valid),
        void_endpoints=void, signed_error_pixels=signed,
        signed_error_capped_pixels=float(np.clip(signed, -2*radius, 2*radius)) if signed is not None else None,
        nearest_gt_crossing=nearest, missing_gt_crossing=nearest is None,
        localization_loss=localization,
        joint_loss=localization if state == "correct" else 1.0,
        displacement_pixels=candidate["offset_pixels"],
        absolute_displacement_pixels=abs(candidate["offset_pixels"]))


METRICS = ("correct", "noncrossing", "reversed", "out_of_image",
           "localization_loss", "joint_loss", "absolute_displacement_pixels")


def measure_frame(rank, label, target, radius):
    lines = {}
    for row in crossings(label == target):
        key = (row["axis"], row["sign"], int(row["center"][1-row["axis"]]))
        lines.setdefault(key, []).append(row["center"][row["axis"]])
    accum = {arm: [] for arm in ARMS}
    for query in rank["queries"]:
        for candidate in query["candidates"]:
            candidate["metric"] = candidate_metric(query, candidate, label, target, radius, lines)
        for arm in ARMS:
            accum[arm].append(query["candidates"][query["selection"][arm]["index"]]["metric"])
    n = len(rank["queries"])
    means = {arm: {key: math.fsum(r[key] for r in values)/n if n else None
                   for key in METRICS} for arm, values in accum.items()}
    deltas = {arm: {key: means[arm][key]-means["native"][key] if n else 0.0
                    for key in METRICS} for arm in ARMS}
    return dict(crossing_count=n, means=means, delta_from_native=deltas,
        no_crossing_event=n == 0,
        fallback_counts={arm: sum(q["selection"][arm]["fallback_reason"] is not None
                                 for q in rank["queries"]) for arm in ARMS},
        missing_gt_crossings={arm: sum(r["missing_gt_crossing"] for r in values)
                              for arm, values in accum.items()})


def aggregate(rows):
    parents = []
    for parent in TRAIN:
        group = [r for r in rows if r["sequence"] == parent]
        require(len(group) == 24, "24 fixed events required per TRAIN parent")
        parents.append(dict(parent_id=parent, events=len(group),
            empty_events=sum(r["summary"]["no_crossing_event"] for r in group),
            crossings=sum(r["summary"]["crossing_count"] for r in group),
            event_mean_delta={arm: {key: math.fsum(
                r["summary"]["delta_from_native"][arm][key] for r in group)/24
                for key in METRICS} for arm in ARMS}))
    count = sum(r["summary"]["crossing_count"] for r in rows)
    pooled = {arm: {key: math.fsum(
        r["summary"]["means"][arm][key]*r["summary"]["crossing_count"]
        for r in rows if r["summary"]["crossing_count"])/count if count else None
        for key in METRICS} for arm in ARMS}
    mean = {arm: {key: math.fsum(p["event_mean_delta"][arm][key] for p in parents)/4
                  for key in METRICS} for arm in ARMS}
    return dict(parents=parents, equal_parent_event_delta_from_native=mean,
                pooled_crossing_means=pooled, pooled_crossings=count,
                unique_frames=len({(r["sequence"], r["frame"]) for r in rows}),
                interpretation="no automated research admission; crossing events are correlated")


def main(spec_path):
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text())
    out = Path(spec["output"])
    out.mkdir(exist_ok=False)
    began = time.perf_counter()
    record = dict(schema="evr0031.train-pair-information.v1", status="started",
        stage="admission", spec_sha256=sha(spec_path), source_sha256=spec["source_sha256"],
        contract=contract(), model_calls=0, final_test_access=False, rows=[], outputs=[],
        offline_attempts=[], compatibility_probes=[])
    fingerprints = dict(spec["source_sha256"])
    fingerprints[str(spec_path)] = record["spec_sha256"]

    def checkpoint():
        record["wall_seconds"] = time.perf_counter()-began
        save(out/"receipt.json", record)

    def descriptor(value):
        path = Path(value["path"])
        require(path.stat().st_size == value["bytes"] and sha(path) == value["sha256"], "input bytes changed")
        fingerprints[str(path)] = value["sha256"]
        return path

    def output(path):
        value = dict(path=str(path.relative_to(out)), bytes=path.stat().st_size, sha256=sha(path))
        record["outputs"].append(value)
        return value

    checkpoint()
    try:
        evaluation = Path(spec["evaluation_root"])
        guard_path = Path(spec["guard_source"])
        required = [Path(__file__).resolve(), Path(__file__).with_name("SCOPE.md").resolve(),
                    Path(__file__).with_name("ADMISSION.md").resolve(), guard_path]
        required += [evaluation/name for name in ("evaluate_jf.py", "evaluate_baseline.py", "evaluate_rollout.py")]
        require(all(spec["source_sha256"].get(str(p)) == sha(p) for p in required), "source binding incomplete")
        require(spec["contract"] == contract(), "fixed contract changed")
        for path, digest in fingerprints.items():
            require(sha(path) == digest, "source/spec changed")
        require(sys.flags.no_site == 1 and sys.version_info[:3] == (3, 12, 13)
                and Path(sys.executable).absolute() == Path(spec["qualified_python"]).absolute()
                and all(os.environ.get(k) == "4" for k in THREADS), "qualified runtime envelope")
        guard = module("pair_screen_guard", guard_path)
        guard.install_guard(record["offline_attempts"], record["compatibility_probes"])
        import site
        site.main()
        import numpy as np
        import torch
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        sys.path.insert(0, str(evaluation))
        admission = module("pair_screen_admission", evaluation/"evaluate_rollout.py")
        e = admission.e
        e.cv2.setNumThreads(0)
        record["runtime"] = dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
                                torch_threads=torch.get_num_threads(), model_imported=False)
        require(set(spec["bindings"]) == set(BINDINGS), "exact cache/mapping bindings required")
        binding = {k: descriptor(d) for k, d in spec["bindings"].items()}
        cache, cache_worker = admission.admit(binding, "cache", fingerprints, legacy=True)
        mapping = json.loads(binding["mapping"].read_text())
        index = json.loads(binding["data_index"].read_text())
        require(cache["completed_frames"] == mapping["member_count"] == 386
                and cache["completed_sequences"] == 6 and mapping["status"] == "completed"
                and cache["data_index"]["sha256"] == sha(binding["data_index"])
                and not cache["offline_attempts"]
                and cache["model_initial_hash"] == cache["model_final_hash"], "complete unchanged cache")
        root = binding["cache_receipt"].parent
        inventory = {d["path"]: d for d in cache["outputs"]}
        require(len(inventory) == len(cache["outputs"]), "duplicate cache inventory")
        bases = {r["sequence"]: r for r in cache["rows"]}
        sources = {r["sequence"]: r for r in index["sequences"]}
        labels = {(r["sequence"], r["frame"]): r for r in mapping["mappings"]}
        require(len(bases) == len(cache["rows"]) == len(sources) == len(index["sequences"]) == 6
                and len(labels) == len(mapping["mappings"]) == 386, "duplicate/incomplete metadata identities")
        plans = []

        def cache_file(relative):
            d = inventory[relative]
            path = root/relative
            require(path.resolve().is_relative_to(root.resolve()), "cache path escapes root")
            return descriptor(dict(d, path=str(path)))

        # All selected feature/prediction/annotation bytes before the first scientific decode.
        for sequence in TRAIN:
            meta, source = bases[sequence], sources[sequence]
            n = meta["frames"]
            anchors = [1+i*(n-5)//11 for i in range(12)]
            require(meta["status"] == "completed" and meta["role"] == source["role"] == "TRAIN"
                    and meta["parent_id"] == source["parent_id"] == sequence and meta["anchors"] == anchors
                    and len(set(anchors)) == 12 and anchors[-1]+3 < n
                    and [p["frame"] for p in meta["predictions"]] == list(range(n)), "fixed TRAIN identities")
            frames = sorted({0} | {a+h for a in anchors for h in (1, 3)})
            paths = {}
            for frame in frames:
                label = labels[sequence, frame]
                require(label["role"] == "TRAIN" and all(label[k] == source["labels"][frame][k]
                        for k in ("path", "bytes", "sha256")), "TRAIN label mapping")
                lp = descriptor(label)
                fp = cache_file(f"{sequence}/frame-{frame:05d}.pt")
                pred = meta["predictions"][frame]
                pp = cache_file(pred["path"])
                require(inventory[pred["path"]] == pred, "prediction inventory differs")
                paths[frame] = dict(label=lp, features=fp, prediction=pp)
            shared = cache_file(sequence+"/shared.pt")
            plans.append((sequence, meta, anchors, paths, shared))
        record.update(stage="ranking", input_sha256=dict(fingerprints), bindings=spec["bindings"],
                      cached_model_hash=cache["model_final_hash"], source_cache_sha256=sha(binding["cache_receipt"]))
        checkpoint()

        def feature(path, sizes):
            obj = torch.load(path, map_location="cpu", weights_only=True)
            tensor = obj["vision"][0]
            fh, fw = sizes[0]
            require(tuple(tensor.shape) == (fh*fw, 1, 32) and (fh, fw) == (256, 256)
                    and tensor.device.type == "cpu" and bool(torch.isfinite(tensor).all()),
                    "finest native feature layout/finite check")
            return tensor[:, 0].to(torch.float64).numpy().reshape(fh, fw, 32).copy()

        scheduled = []
        for sequence, meta, anchors, paths, shared_path in plans:
            shared = torch.load(shared_path, map_location="cpu", weights_only=True)
            sizes = shared["sizes"]
            require(len(sizes) == 3 and tuple(sizes[0]) == (256, 256), "native feature pyramid")
            initial = e._labels(paths[0]["label"])
            require(list(initial.shape) == meta["shape"], "initial dimensions")
            ids = np.unique(initial)
            ids = ids[(ids > 0) & (ids != 255)]
            require(ids.size > 0 and int(ids.min()) == meta["target_id"], "original smallest-ID target")
            radius = int(math.ceil(.008*math.hypot(*initial.shape)))
            reference = feature(paths[0]["features"], sizes)
            bank, permutation, bank_meta = reference_bank(initial, reference, meta["target_id"], sequence, radius)
            del reference, initial
            bank_path = out/f"{sequence}-reference.json"
            save(bank_path, dict(sequence=sequence, radius=radius, **bank_meta)); output(bank_path)
            for anchor in anchors:
                for horizon in (1, 3):
                    frame = anchor+horizon
                    identity = dict(sequence=sequence, parent_id=sequence, anchor=anchor,
                                    horizon=horizon, frame=frame, target_id=meta["target_id"])
                    record["pending"] = identity; checkpoint()
                    tick = time.perf_counter()
                    prediction = e._prediction(paths[frame]["prediction"], [meta["target_id"]], tuple(meta["shape"]))[0]
                    fmap = feature(paths[frame]["features"], sizes)
                    ranked = rank_frame(prediction, fmap, bank, permutation, radius)
                    path = out/f"{sequence}-a{anchor:05d}-h{horizon}-ranking.json"
                    save(path, dict(**identity, radius=radius, **ranked))
                    d = output(path)
                    scheduled.append((identity, path, paths[frame]["label"], radius))
                    record["rows"].append(dict(**identity, status="ranked", ranking=d,
                                               ranking_seconds=time.perf_counter()-tick))
                    del fmap, prediction, ranked
                    checkpoint()
        require(len(scheduled) == 96, "all 96 ranks required before future-label decoding")
        rank_manifest = dict(schema="evr0031.ranking-seal.v1", count=96,
            spec_sha256=record["spec_sha256"], source_sha256=record["source_sha256"],
            outputs=list(record["outputs"]))
        save(out/"ranking-seal.json", rank_manifest)
        record["ranking_seal"] = output(out/"ranking-seal.json")
        for d in rank_manifest["outputs"]:
            require(sha(out/d["path"]) == d["sha256"], "ranking changed before metric stage")
        record["stage"] = "TRAIN_label_diagnostic"; checkpoint()
        for i, (identity, rank_path, label_path, radius) in enumerate(scheduled):
            record["pending"] = identity; checkpoint()
            ranked = json.loads(rank_path.read_text())
            label = e._labels(label_path)
            require(list(label.shape) == bases[identity["sequence"]]["shape"], "future label dimensions")
            summary = measure_frame(ranked, label, identity["target_id"], radius)
            path = out/rank_path.name.replace("-ranking.json", "-diagnostic.json")
            save(path, dict(**ranked, summary=summary))
            record["rows"][i].update(status="completed", diagnostic=output(path), summary=summary)
            checkpoint()
        report = dict(schema=record["schema"], scope=contract()["scientific_scope"],
            aggregation=aggregate(record["rows"]), rows=record["rows"], contract=contract(),
            no_new_encoder=True, no_training=True, no_mask_repair=True, no_final_test=True,
            feature_read_and_ranking_costs_charged=True, baseline_history_cost_not_remeasured=True)
        save(out/"analysis.json", report); output(out/"analysis.json")
        for path, digest in fingerprints.items():
            require(sha(path) == digest, "source/input changed during screen")
        require(not record["offline_attempts"], "offline violation")
        record.update(status="completed", stage="complete", pending=None,
                      completed_events=96, completed_parents=4, input_sha256=fingerprints)
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        checkpoint()


if __name__ == "__main__":
    main(sys.argv[1])
