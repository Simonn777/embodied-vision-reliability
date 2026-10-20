"""Saved native DEV masks only: boundary/region error location, no model imports."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

DEV = (("cat-girl", 89), ("upside-down", 65))
OLD_BINDINGS = ("cache_receipt", "cache_outer", "cache_operation", "cache_worker_spec",
                "rollout_receipt", "rollout_outer", "rollout_operation", "rollout_worker_spec",
                "mapping", "data_index")
REGIONS = ("within_r", "between_r_and_2r", "beyond_2r", "undefined_no_gt_boundary")
FLAGS = ("truth_empty", "prediction_empty", "both_empty", "complete_target_miss",
         "empty_prediction_with_target", "nonempty_disjoint", "false_positive_when_target_absent",
         "exact_mask")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def dilation(boundary, radius, e):
    axis = e.np.arange(-radius, radius + 1, dtype=e.np.int64)
    disk = (axis[:, None] ** 2 + axis[None, :] ** 2 <= radius ** 2).astype(e.np.uint8)
    return e.cv2.dilate(boundary.astype(e.np.uint8), disk,
        borderType=e.cv2.BORDER_CONSTANT, borderValue=0) != 0


def boundary_statistics(pb, tb, gt_dilated, radius, e):
    """Official empty-boundary P/R cases; 2r is descriptive, not official F."""
    n_pred, n_gt = int(pb.sum()), int(tb.sum())
    p_match = int(e.np.count_nonzero(pb & gt_dilated))
    t_match = int(e.np.count_nonzero(tb & dilation(pb, radius, e)))
    if n_pred == 0 and n_gt > 0:
        precision, recall = 1.0, 0.0
    elif n_pred > 0 and n_gt == 0:
        precision, recall = 0.0, 1.0
    elif n_pred == 0 and n_gt == 0:
        precision, recall = 1.0, 1.0
    else:
        precision, recall = p_match / n_pred, t_match / n_gt
    f = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(radius_pixels=radius, precision=precision, recall=recall, F=f,
                predicted_boundary_pixels=n_pred, gt_boundary_pixels=n_gt,
                matched_predicted_boundary_pixels=p_match, matched_gt_boundary_pixels=t_match)


def frame_statistics(prediction, truth, e):
    np = e.np
    require(prediction.dtype == truth.dtype == np.bool_ and prediction.shape == truth.shape,
            "boolean masks with identical shapes required")
    fp, fn = prediction & ~truth, truth & ~prediction
    tp = int(np.count_nonzero(prediction & truth))
    nfp, nfn = int(fp.sum()), int(fn.sum())
    gt_area, pred_area = int(truth.sum()), int(prediction.sum())
    pb, tb = e.boundary_map(prediction), e.boundary_map(truth)
    radius = e.boundary_radius(truth.shape)
    near, near2 = dilation(tb, radius, e), dilation(tb, 2 * radius, e)
    empty_boundary = not bool(tb.any())
    empty = np.zeros_like(truth)
    regions = dict(within_r=near, between_r_and_2r=near2 & ~near,
                   beyond_2r=~near2 if not empty_boundary else empty,
                   undefined_no_gt_boundary=np.ones_like(truth) if empty_boundary else empty)
    bins = {}
    for name, region in regions.items():
        a, b = int(np.count_nonzero(fp & region)), int(np.count_nonzero(fn & region))
        bins[name] = dict(FP=a, FN=b, error_pixels=a+b,
                         error_mass_fraction=(a+b)/(nfp+nfn) if nfp+nfn else None)
    require(sum(r["FP"] for r in bins.values()) == nfp
            and sum(r["FN"] for r in bins.values()) == nfn, "error partition lost pixels")
    j, f = e.jf(prediction, truth)
    at_r = boundary_statistics(pb, tb, near, radius, e)
    at_2r = boundary_statistics(pb, tb, near2, 2 * radius, e)
    require(at_r["F"] == f, "official boundary F and precision/recall reconstruction differ")
    return dict(J=j, F=f, JF=(j+f)/2,
        counts=dict(image_pixels=int(truth.size), gt_pixels=gt_area, predicted_pixels=pred_area,
                    TP=tp, FP=nfp, FN=nfn, TN=int(truth.size)-tp-nfp-nfn, error_pixels=nfp+nfn),
        error_location=bins, gt_boundary_empty=empty_boundary, official_radius_pixels=radius,
        boundary_at_r=at_r, boundary_at_2r=at_2r,
        flags=dict(truth_empty=gt_area == 0, prediction_empty=pred_area == 0,
                   both_empty=gt_area == pred_area == 0,
                   complete_target_miss=gt_area > 0 and tp == 0,
                   empty_prediction_with_target=gt_area > 0 and pred_area == 0,
                   nonempty_disjoint=gt_area > 0 and pred_area > 0 and tp == 0,
                   false_positive_when_target_absent=gt_area == 0 and pred_area > 0,
                   exact_mask=nfp+nfn == 0))


def summarize(rows, e):
    require(bool(rows), "empty summary")
    counts = {k: sum(r["counts"][k] for r in rows) for k in rows[0]["counts"]}
    bins = {name: {k: sum(r["error_location"][name][k] for r in rows)
                  for k in ("FP", "FN", "error_pixels")} for name in REGIONS}
    for value in bins.values():
        value["error_mass_fraction"] = value["error_pixels"]/counts["error_pixels"] \
            if counts["error_pixels"] else None
    boundary = {}
    for key in ("boundary_at_r", "boundary_at_2r"):
        boundary[key] = dict(
            frame_mean={k: math.fsum(r[key][k] for r in rows)/len(rows)
                        for k in ("precision", "recall", "F")},
            summed_counts={k: sum(r[key][k] for r in rows) for k in
                ("predicted_boundary_pixels", "gt_boundary_pixels",
                 "matched_predicted_boundary_pixels", "matched_gt_boundary_pixels")})
    return dict(frames=len(rows), mean=e._means(rows), pixel_sums=counts,
        error_location_pixel_sums=bins, **boundary,
        flag_frame_counts={k: sum(r["flags"][k] for r in rows) for k in FLAGS},
        gt_boundary_empty_frames=sum(r["gt_boundary_empty"] for r in rows),
        zero_error_frames=sum(r["counts"]["error_pixels"] == 0 for r in rows))


def main(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    directory = Path(spec["evaluation_root"])
    needed = [Path(__file__).resolve()] + [directory / name for name in
        ("evaluate_jf.py", "evaluate_baseline.py", "evaluate_rollout.py")]
    for path in needed:
        require(spec["source_sha256"].get(str(path)) == sha(path), "missing/changed source")
    # Only the already qualified NumPy/cv2/PIL metric modules; never campaign/model modules.
    sys.path.insert(0, str(directory))
    loader = importlib.util.spec_from_file_location("error_analysis_admission", directory / "evaluate_rollout.py")
    admission = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(admission)
    e = admission.e
    e.cv2.setNumThreads(0)
    began = time.perf_counter()
    out = Path(spec["output"]); out.mkdir(exist_ok=False)
    record = dict(schema="evr0030.native-output-error-analysis.v1", status="started",
        model_calls=0, final_test_access=False, scientific_gain_claim=False,
        spec_sha256=sha(spec_path), source_sha256=spec["source_sha256"], rows=[])
    fingerprints = dict(spec["source_sha256"])
    fingerprints[str(Path(spec_path).resolve())] = record["spec_sha256"]

    def descriptor(d):
        e._descriptor(d)
        fingerprints[d["path"]] = d["sha256"]
        return Path(d["path"])

    def checkpoint():
        record["wall_seconds"] = time.perf_counter()-began
        save(out / "receipt.json", record)

    checkpoint()
    try:
        require(set(spec["bindings"]) == set(OLD_BINDINGS) | {
            "evaluation_receipt", "evaluation_outer", "evaluation_operation",
            "evaluation_worker_spec", "evaluation"}, "exact native metric/rollout bindings")
        bindings = {k: descriptor(v) for k, v in spec["bindings"].items()}
        cache, _ = admission.admit(bindings, "cache", fingerprints, legacy=True)
        rollout, worker = admission.admit(bindings, "rollout", fingerprints)
        metric, metric_spec = admission.admit(bindings, "evaluation", fingerprints)
        require(metric["schema"] == "evr0030.rollout-evaluation.v1" and metric["arm"] == "native"
                and metric["raw_frames"] == 154 and metric["included_frames"] == 150
                and bindings["evaluation"] == bindings["evaluation_receipt"].parent / "evaluation.json"
                and metric["evaluation_sha256"] == sha(bindings["evaluation"])
                and metric["bindings"] == metric_spec["bindings"] ==
                    {k: spec["bindings"][k] for k in OLD_BINDINGS}, "completed native evaluation binding")
        require(rollout["schema"] == "evr0030.development.own-rollout.v1"
                and rollout["arm"] == worker["arm"] == "native" and worker.get("checkpoint") is None
                and rollout["frames"] == 154 and rollout["zero_parity_checks"] == 152
                and rollout["changed_writes"] == 0 and not rollout["offline_attempts"]
                and rollout["model_initial_hash"] == rollout["model_final_hash"] == cache["model_final_hash"]
                and worker["cache_receipt_sha256"] == sha(bindings["cache_receipt"]),
                "complete unchanged native rollout")
        mapping = json.loads(bindings["mapping"].read_text())
        index = json.loads(bindings["data_index"].read_text())
        saved = json.loads(bindings["evaluation"].read_text())
        require(mapping["status"] == "completed" and mapping["member_count"] == 386
                and cache["completed_frames"] == 386 and cache["completed_sequences"] == 6
                and cache["data_index"]["sha256"] == sha(bindings["data_index"]), "complete input metadata")
        names = [name for name, _ in DEV]
        require([r["sequence"] for r in rollout["rows"]] == worker["dev_sequences"] == names
                and [r["sequence"] for r in saved["sequences"]] == names
                and rollout["outputs"] == [p for r in rollout["rows"] for p in r["predictions"]],
                "full native cohort/inventory")
        sources = {r["sequence"]: r for r in index["sequences"]}
        bases = {r["sequence"]: r for r in cache["rows"]}
        plans = []
        for (name, n), row, previous in zip(DEV, rollout["rows"], saved["sequences"]):
            source, base = sources[name], bases[name]
            labels = [m for m in mapping["mappings"] if m["sequence"] == name]
            require(source["role"] == "DEV" and source["parent_id"] == row["parent_id"] == name
                    and row["frames"] == n and all(row[k] == base[k] for k in ("target_id", "shape", "frames"))
                    and not row["writes"] and
                    [p["frame"] for p in row["predictions"]] == [m["frame"] for m in labels] == list(range(n))
                    and all(m["role"] == "DEV" and all(m[k] == source["labels"][i][k]
                        for k in ("path", "sha256", "bytes")) for i, m in enumerate(labels)),
                    "complete DEV prediction/annotation identities")
            require(previous["parent_id"] == name and previous["frame_count"] == n
                    and previous["included_frame_count"] == n-2 and previous["shape_hw"] == row["shape"]
                    and previous["object_ids"] == [row["target_id"]] and len(previous["objects"]) == 1
                    and previous["objects"][0]["object_id"] == row["target_id"]
                    and previous["reference_commit"] == e.REFERENCE
                    and previous["endpoint_policy"] == "exclude_first_and_last"
                    and previous["void_policy"] == "official_background"
                    and previous["boundary_fraction"] == e.BOUNDARY_FRACTION, "saved metric identity/policy")
            old_rows = previous["objects"][0]["frames"]
            require([r["index"] for r in old_rows] == list(range(n))
                    and all(r["frame_id"] == f"{i:05d}" and r["included"] == (0 < i < n-1)
                            for i, r in enumerate(old_rows)), "saved metric complete frame coverage")
            predictions = [dict(p, path=str(bindings["rollout_receipt"].parent / p["path"]))
                           for p in row["predictions"]]
            for d in predictions + labels:
                descriptor(d)
            plans.append((row, predictions, labels, old_rows, previous))
        # All 154 saved prediction and 154 annotation hashes checked before first image/array decode.
        parents = []
        for meta, predictions, labels, old_rows, previous in plans:
            rows = []
            for frame in range(1, meta["frames"]-1):
                record["pending_frame"] = dict(sequence=meta["sequence"], frame=frame)
                prediction = e._prediction(Path(predictions[frame]["path"]), [meta["target_id"]],
                                             tuple(meta["shape"]))[0]
                label = e._labels(Path(labels[frame]["path"]))
                require(tuple(label.shape) == tuple(meta["shape"]), "original annotation dimensions")
                result = frame_statistics(prediction, label == meta["target_id"], e)
                require(all(result[k] == old_rows[frame][k] for k in ("J", "F", "JF")),
                        "saved native per-frame J/F consistency failure")
                row = dict(sequence=meta["sequence"], parent_id=meta["parent_id"], frame=frame,
                    frame_id=f"{frame:05d}", target_id=meta["target_id"], included=True,
                    void_pixels_as_background=int(e.np.count_nonzero(label == 255)),
                    prediction=predictions[frame], annotation=labels[frame], **result)
                rows.append(row); record["rows"].append(row)
                record["pending_frame"] = None
                checkpoint()
            summary = summarize(rows, e)
            require(summary["mean"] == previous["mean"] == previous["objects"][0]["mean"],
                    "saved native parent J/F consistency failure")
            parents.append(dict(parent_id=meta["parent_id"], **summary))
        require(len(record["rows"]) == 150 and [p["frames"] for p in parents] == [87, 63],
                "all 150 interior frames retained")
        mean = e._means([p["mean"] for p in parents])
        require(mean == saved["mean"], "saved native equal-parent J/F consistency failure")
        report = dict(schema=record["schema"], model_calls=0, scientific_gain_claim=False,
            scope="descriptive error location, fixed native target-object DEV rollout only",
            endpoint_policy="exclude first/last; hash all 154, decode all 150 interiors",
            reference_commit=e.REFERENCE, boundary_fraction=e.BOUNDARY_FRACTION,
            distance_reference="discrete official GT boundary; exact Euclidean disks r and 2*r",
            empty_boundary_policy="distance undefined, retained in separate error bin",
            precision_recall_empty_policy="pred-empty:(1,0); GT-empty:(0,1); both-empty:(1,1)",
            aggregation="official mean equal parents; region pixel sums descriptive, frame/parent means separate",
            void_policy="official_background", mean=mean, parents=parents,
            all_frame_descriptive_summary=summarize(record["rows"], e), rows=record["rows"],
            exact_saved_native_metric_consistency=True)
        save(out / "analysis.json", report)
        for path, digest in fingerprints.items():
            require(sha(path) == digest, "source/input changed during analysis")
        record.update(status="completed", included_frames=150, raw_frames_bound=154,
            analysis_sha256=sha(out / "analysis.json"), bindings=spec["bindings"],
            input_sha256=fingerprints, exact_saved_native_metric_consistency=True)
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        checkpoint()


if __name__ == "__main__":
    main(sys.argv[1])
