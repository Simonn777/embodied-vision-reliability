"""Saved crop predictions only: fixed TRAIN events and full DEV target-object J/F."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

TRAIN = ("dog-agility", "planes-water", "dogs-scale", "dog-gooses")
DEV = ("cat-girl", "upside-down")
OLD_KEYS = {"cache_receipt", "cache_outer", "cache_operation", "cache_worker_spec",
            "mapping", "data_index"}
NEW_KEYS = {"refinement_receipt", "refinement_outer", "refinement_operation",
            "refinement_worker_spec"}
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def module(name, path):
    loader = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(loader)
    sys.modules[name] = value
    loader.loader.exec_module(value)
    return value


def run(spec_path):
    spec_path = Path(spec_path).resolve()
    preliminary = json.loads(spec_path.read_text())
    common_path = Path(preliminary["campaign_root"]) / "common.py"
    if preliminary["source_sha256"].get(str(common_path)) != hashlib.sha256(common_path.read_bytes()).hexdigest():
        raise ValueError("changed serialization helper")
    c = module("crop_metric_common", common_path)
    spec, out, record, began = c.start(spec_path, "crop-detail-evaluation")
    record.update(schema="evr0032.crop-detail-evaluation.v1", model_calls=0,
        image_encoder_calls=0, native_calls=0, final_test_access=False,
        label_decode_attempted=0, label_decodes=0, predictions_sealed_before_gt=False)
    fingerprints = dict(spec["source_sha256"])
    fingerprints[str(spec_path)] = record["spec_sha256"]
    try:
        here, evaluation = Path(__file__).resolve().parent, Path(spec["evaluation_root"])
        required = {str(here / name) for name in ("evaluate.py", "METRIC_SCOPE.md", "ADMISSION.md",
                    "refine.py", "SCOPE.md")}
        required |= {str(common_path), str(Path(spec["campaign_root"]) / "cache_baselines.py")}
        required |= {str(evaluation / name) for name in
                     ("evaluate_jf.py", "evaluate_baseline.py", "evaluate_rollout.py")}
        c.require(set(spec["source_sha256"]) == required, "exact metric source set")
        c.require(sys.version_info[:3] == (3, 12, 13) and
            Path(sys.executable).absolute() == Path(spec["python"]).absolute(), "qualified interpreter")
        c.require(all(os.environ.get(k) == "4" for k in THREAD_KEYS), "four-thread environment")
        for path, expected in fingerprints.items():
            c.require(c.sha(path) == expected, "source/spec changed: " + path)
        sys.path.insert(0, str(evaluation))
        e = module("evaluate_jf", evaluation / "evaluate_jf.py")
        admission = module("crop_metric_admission", evaluation / "evaluate_rollout.py")
        e.cv2.setNumThreads(0)

        def check(d):
            path = e._descriptor(d)
            if str(path) in fingerprints:
                c.require(fingerprints[str(path)] == d["sha256"], "conflicting file identity")
            fingerprints[str(path)] = d["sha256"]
            return path

        def relative(root, d):
            rel = Path(d["path"])
            path = (root / rel).resolve()
            c.require(not rel.is_absolute() and ".." not in rel.parts and
                      path.is_relative_to(root.resolve()), "output escapes operation directory")
            return dict(d, path=str(path))

        c.require(set(spec["bindings"]) == OLD_KEYS | NEW_KEYS, "exact ten metadata bindings")
        b = {k: check(d) for k, d in spec["bindings"].items()}
        baseline_path = check(spec["baseline_reference"])
        cache, _ = admission.admit(b, "cache", fingerprints, legacy=True)
        refined, worker = admission.admit(b, "refinement", fingerprints)
        c.require(cache["completed_frames"] == 386 and cache["completed_sequences"] == 6 and
            not cache["offline_attempts"] and cache["model_initial_hash"] == cache["model_final_hash"],
            "complete immutable native cache")
        c.require(refined["schema"] == "evr0032.crop-detail.v1" and
            refined["completed_frames"] == 241 and refined["train_unique_frames"] == 87 and
            refined["dev_frames"] == 154 and refined["current_label_decodes"] == 0 and
            not refined["offline_attempts"] and
            refined["model_initial_hash"] == refined["model_final_hash"] == cache["model_final_hash"],
            "complete GT-free immutable refinement")
        old_bindings = {k: spec["bindings"][k] for k in OLD_KEYS}
        c.require(refined["bindings"] == worker["bindings"] == old_bindings,
                  "refinement used different baseline/annotation metadata")
        c.require(worker["source_sha256"].get(str(here / "refine.py")) ==
            spec["source_sha256"][str(here / "refine.py")], "refinement source mismatch")
        operation = json.loads(b["refinement_operation"].read_text())
        c.require(Path(operation["command"][-2]).resolve() == here / "refine.py",
                  "refinement command script")
        for path, digest in worker["source_sha256"].items():
            c.require(c.sha(path) == digest, "refinement dependency changed")
            if path in fingerprints:
                c.require(fingerprints[path] == digest, "source identity conflict")
            fingerprints[path] = digest
        refine_root, cache_root = b["refinement_receipt"].parent, b["cache_receipt"].parent
        seal_path = refine_root / "seal.json"
        seal = json.loads(seal_path.read_text())
        fingerprints[str(seal_path)] = c.sha(seal_path)
        c.require(seal["schema"] == "evr0032.crop-detail-seal.v1" and seal["complete"] is True and
            seal["receipt_sha256"] == c.sha(b["refinement_receipt"]) and seal["frame_count"] == 241 and
            seal["source_sha256"] == refined["source_sha256"] and
            seal["spec_sha256"] == refined["spec_sha256"] and
            seal["outputs"] == refined["outputs"] ==
                [p for row in refined["rows"] for p in row["predictions"]] and
            seal["diagnostics"] == refined["diagnostics"], "complete sealed predictions required before GT")
        descriptors = refined["outputs"] + refined["diagnostics"]
        c.require(len({d["path"] for d in descriptors}) == len(descriptors), "duplicate refinement file")
        for d in descriptors:
            check(relative(refine_root, d))

        mapping, index = (json.loads(b[k].read_text()) for k in ("mapping", "data_index"))
        c.require(mapping["status"] == "completed" and mapping["member_count"] == 386 and
            cache["data_index"]["sha256"] == c.sha(b["data_index"]), "complete mapping/cache identity")
        rows = {r["sequence"]: r for r in refined["rows"]}
        bases = {r["sequence"]: r for r in cache["rows"]}
        inputs = {r["sequence"]: r for r in index["sequences"]}
        labels = {(r["sequence"], r["frame"]): r for r in mapping["mappings"]}
        c.require([r["sequence"] for r in refined["rows"]] == list(TRAIN + DEV) and
            len(bases) == len(cache["rows"]) == len(inputs) == len(index["sequences"]) == 6 and
            set(bases) == set(inputs) == set(TRAIN + DEV) and
            len(labels) == len(mapping["mappings"]) == 386, "fixed complete six-parent identities")
        cache_files = {d["path"]: d for d in cache["outputs"]}
        c.require(len(cache_files) == len(cache["outputs"]), "duplicate baseline output")
        events, selected = [], {}
        for name in TRAIN + DEV:
            row, base, source = rows[name], bases[name], inputs[name]
            n, role = base["frames"], "TRAIN" if name in TRAIN else "DEV"
            anchors = [1 + i * (n - 5) // 11 for i in range(12)] if role == "TRAIN" else []
            c.require(base["status"] == "completed" and row["parent_id"] == source["parent_id"] == name and
                row["role"] == base["role"] == source["role"] == role and
                all(row[k] == base[k] for k in ("frames", "shape", "target_id")) and
                row["anchors"] == base["anchors"] == anchors and
                [p["frame"] for p in base["predictions"]] == [p["frame"] for p in source["labels"]] ==
                list(range(n)), "parent metadata or full baseline coverage")
            shape, target = tuple(row["shape"]), row["target_id"]
            c.require(len(shape) == 2 and min(shape) == 480 and type(target) is int and
                      0 < target < 255, "original-resolution single target")
            if role == "TRAIN":
                ee = [(name, a, h, a + h) for a in anchors for h in (1, 3)]
                events.extend(ee); frames = sorted({f for _, _, _, f in ee})
            else:
                frames = list(range(n))
            c.require([p["frame"] for p in row["predictions"]] == frames, "complete requested frame set")
            for pred, frame in zip(row["predictions"], frames):
                native, truth = base["predictions"][frame], labels[name, frame]
                c.require(native == cache_files[native["path"]] and native["sequence"] == name and
                    truth["role"] == role and all(truth[k] == source["labels"][frame][k]
                    for k in ("path", "sha256", "bytes")), "native/GT frame identity")
                selected[name, frame] = (
                    check(relative(refine_root, pred)), check(relative(cache_root, native)), check(truth))
        c.require(len(events) == 96 and len({(n, f) for n, _, _, f in events}) == 87 and
                  len(selected) == 241 and sum(rows[n]["frames"] for n in DEV) == 154,
                  "full 96/87 TRAIN plus154 DEV coverage")
        # All 241 refined/native/GT encoded hashes and complete producer seal checked above.
        # Preflight every output's array contract and the two preserved initial outputs before GT.
        for (name, frame), (pp, bp, _) in selected.items():
            row = rows[name]; args = ([row["target_id"]], tuple(row["shape"]))
            pm, bm = e._prediction(pp, *args), e._prediction(bp, *args)
            if name in DEV and frame == 0:
                c.require(e.np.array_equal(pm, bm), "DEV initial mask must remain native")
        record.update(predictions_sealed_before_gt=True, prediction_seal_sha256=c.sha(seal_path),
                      verified_unique_frames=241, bindings=spec["bindings"],
                      baseline_reference=spec["baseline_reference"])
        c.save(out / "receipt.json", record)

        metrics = {}
        for (name, frame), (pp, bp, gp) in selected.items():
            row = rows[name]; shape, target = tuple(row["shape"]), row["target_id"]
            record["label_decode_attempted"] += 1
            c.save(out / "receipt.json", record)
            truth = e._labels(gp)
            c.require(tuple(truth.shape) == shape, "GT shape changed")
            record["label_decodes"] += 1
            entry = dict(sequence=name, parent_id=name, frame=frame, target_id=target,
                final_frame=frame == row["frames"] - 1,
                void_pixels_as_background=int(e.np.count_nonzero(truth == 255)))
            for arm, path in (("native", bp), ("refined", pp)):
                j, f = e.jf(e._prediction(path, [target], shape)[0], truth == target)
                entry[arm] = dict(J=j, F=f, JF=(j + f) / 2)
            entry["delta"] = {k: entry["refined"][k] - entry["native"][k] for k in ("J", "F", "JF")}
            metrics[name, frame] = entry

        def summary(rr, names):
            parents = []
            for name in names:
                chosen = [r for r in rr if r["parent_id"] == name and r["included"]]
                parent = dict(parent_id=name, included_count=len(chosen),
                              **{arm: e._means([r[arm] for r in chosen]) for arm in ("native", "refined")})
                parent["delta"] = {k: parent["refined"][k] - parent["native"][k] for k in ("J", "F", "JF")}
                parents.append(parent)
            answer = dict(rows=rr, parents=parents,
                          **{arm: e._means([p[arm] for p in parents]) for arm in ("native", "refined")})
            answer["delta"] = {k: answer["refined"][k] - answer["native"][k] for k in ("J", "F", "JF")}
            return answer

        train_rows = [dict(metrics[n, f], anchor=a, horizon=h, included=True) for n, a, h, f in events]
        dev_rows = [dict(metrics[n, f], included=0 < f < rows[n]["frames"] - 1)
                    for n in DEV for f in range(rows[n]["frames"])]
        train, dev = summary(train_rows, TRAIN), summary(dev_rows, DEV)
        reference = json.loads(baseline_path.read_text())
        c.require(reference["schema"] == "evr0030.development-jf.cohort.v1" and
                  [p["parent_id"] for p in reference["parents"]] == list(DEV), "native reference identity")
        for observed, expected in [(dev["native"], reference["mean"])] + [
                (p["native"], r["mean"]) for p, r in zip(dev["parents"], reference["parents"])]:
            c.require(all(math.isclose(observed[k], expected[k], abs_tol=1e-12, rel_tol=0)
                          for k in ("J", "F", "JF")), "published native DEV baseline mismatch")
        clauses = dict(dev_JF_gain_at_least_1pp=dev["delta"]["JF"] >= .01,
            both_dev_parents_positive=all(p["delta"]["JF"] > 0 for p in dev["parents"]),
            dev_J_loss_at_most_half_pp=dev["delta"]["J"] >= -.005,
            dev_F_loss_at_most_half_pp=dev["delta"]["F"] >= -.005)
        report = dict(schema="evr0032.crop-detail-metrics.v1", official_benchmark_result=False,
            scope="target-object development J/F; known crop baseline, not innovation or paper gate",
            metric_scale="[0,1]; deltas times100 are absolute percentage points",
            reference_commit=e.REFERENCE, boundary_fraction=e.BOUNDARY_FRACTION,
            void_policy="official_background", target_policy="same native initial target ID; no union",
            train=dict(train, event_count=96, unique_future_frames=87,
                weighting="24 horizon events per parent, duplicates retained, four equal parents",
                endpoint_policy="all fixed horizon events including4 final-frame events"),
            dev=dict(dev, raw_frames=154, included_frames=150,
                endpoint_policy="exclude first/last from means; retain all raw metrics",
                weighting="equal two parents; equal interior frames within parent",
                published_native_parity_atol=1e-12),
            continuation=dict(clauses=clauses, passed=all(clauses.values()),
                meaning="fixed known-baseline development continuation only"),
            producer_costs={k: {x: json.loads(b[k + "_outer"].read_text())[x]
                for x in ("wall_seconds", "peak_group_rss_kib")} for k in ("cache", "refinement")},
            cost_note="Native cache and added refinement costs separate; no common-budget speed claim.")
        c.require(record["label_decodes"] == 241 and sum(r["included"] for r in dev_rows) == 150 and
                  sum(r["final_frame"] for r in train_rows) == 4, "complete metric accounting")
        c.save(out / "evaluation.json", report)
        for path, digest in fingerprints.items():
            c.require(c.sha(path) == digest, "source/input changed during metrics: " + path)
        record.update(status="completed", input_sha256=fingerprints,
            evaluation_sha256=c.sha(out / "evaluation.json"), train_events=96,
            train_unique_frames=87, dev_raw_frames=154, dev_interior_frames=150,
            continuation_passed=report["continuation"]["passed"])
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record["wall_seconds"] = time.perf_counter() - began
        c.save(out / "receipt.json", record)


if __name__ == "__main__":
    run(sys.argv[1])
