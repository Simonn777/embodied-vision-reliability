"""One sealed 0011 decode, then one GT evaluation; no model imports or calls."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

import numpy as np

GROUPS = ("g-036", "g-002")
FRAMES = tuple(range(0, 241, 16))
VIEWS = ("original", "reference", "oblique", "axis_control")
METHODS = ("original_native", "reference_native", "oblique_native",
           "axis_control_native", "reference_independent", "coordinate_average",
           "entropy_select", "product_joint", "mixture_joint",
           "oblique_coupling", "axis_coupling")
CANDIDATE = "oblique_coupling"
THRESHOLDS = np.array([1., 2., 4., 8., 16.])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def record(path):
    return {"bytes": Path(path).stat().st_size, "sha256": sha(path)}


def verify(path, expected):
    if Path(path).is_symlink() or record(path) != expected:
        raise ValueError("file binding differs: " + str(path))


def begin(output, phase):
    output = Path(output)
    output.mkdir(exist_ok=False)
    write_json(output / "started.json", {"phase": phase,
               "started_utc": datetime.now(timezone.utc).isoformat()})
    return output


def numerical_checks(diagnostics):
    """Reconstruct every solver/mean gate rather than trusting boolean summaries."""
    def bounded(value, limit):
        return type(value) in (int, float) and np.isfinite(value) and 0 <= value <= limit
    checks = {"all_couplings_converged": True, "fixed_global_means": True, "axis_joint_invariant": True}
    for row in diagnostics:
        if set(row["solvers"]) != {"oblique_coupling", "axis_coupling"}:
            raise ValueError("missing solver record")
        for method, solver in row["solvers"].items():
            checks["all_couplings_converged"] &= (solver["converged"] is True and
                type(solver["iterations"]) is int and 1 <= solver["iterations"] <= 1000 and
                all(bounded(solver[k], 1e-9) for k in ("row_error", "column_error", "residual")) and
                solver["residual"] == max(solver["row_error"], solver["column_error"]))
            mean = row["global_means"][method]
            checks["fixed_global_means"] &= bounded(mean["reference_max_abs_difference"], 5e-5)
        checks["axis_joint_invariant"] &= bounded(row["axis_joint_max_abs_difference"], 1e-7)
    return {k: bool(v) for k,v in checks.items()}


def derive(prediction_root, preparation, output, *, fusion=None):
    """GT paths never opened here. Each native output is verified before decode."""
    if fusion is None:
        from oblique_fusion import fuse
        fusion = fuse
    output = begin(output, "derive")
    started = time.monotonic()
    try:
        preparation = Path(preparation)
        prep = load_json(preparation / "receipt.json")
        if prep["status"] != "completed_preparation":
            raise ValueError("preparation is incomplete")
        rows, bindings = [], {"preparation_receipt": record(preparation / "receipt.json")}
        diagnostics = []
        for group in GROUPS:
            folder = Path(prediction_root) / group
            receipt = load_json(folder / "receipt.json")
            if receipt["status"] != "completed_predictions" or receipt["group_id"] != group:
                raise ValueError("wrong or incomplete group predictions")
            if receipt["forwards_attempted"] != 64 or len(receipt["frames"]) != 64:
                raise ValueError("must retain exactly 64 forwards per group")
            predictor_name = group + ".predictor.npz"
            verify(preparation / predictor_name, prep["files"][predictor_name])
            if receipt["input_sha256"] != prep["files"][predictor_name]["sha256"]:
                raise ValueError("prediction input differs from preparation")
            with np.load(preparation / predictor_name, allow_pickle=False) as inputs:
                point_ids = inputs["point_ids"].tolist()
                if inputs["frame_ids"].tolist() != list(FRAMES) or len(point_ids) != 8:
                    raise ValueError("input identities differ")
            bindings[group] = {"receipt": record(folder / "receipt.json"), "frames": {}}
            frame_records = {}
            seen_files = set()
            for item in receipt["frames"]:
                key = (item["view"], item["observation_index"])
                if key in frame_records or key[0] not in VIEWS or not 0 <= key[1] < 16:
                    raise ValueError("duplicate or invalid view/frame record")
                if item["frame_index"] != FRAMES[key[1]]:
                    raise ValueError("original frame identity differs")
                name = item["output_file"]
                if (Path(name).name != name or name in seen_files or (folder/name).is_symlink() or
                    sha(folder / name) != item["output_sha256"]):
                    raise ValueError("unsafe or changed raw output")
                seen_files.add(name)
                bindings[group]["frames"][name] = record(folder / name)
                frame_records[key] = item
            for observation, frame_id in enumerate(FRAMES):
                raw = {}
                for view in VIEWS:
                    item = frame_records[(view, observation)]
                    with np.load(folder / item["output_file"], allow_pickle=False) as f:
                        if set(f.files) != {"coordinate_logits", "positions_xy", "visibility_logits"}:
                            raise ValueError("unexpected raw output keys")
                        raw[view] = {name: f[name].copy() for name in f.files}
                    for name, shape in (("coordinate_logits", (8,512)), ("positions_xy", (8,2)),
                                        ("visibility_logits", (8,))):
                        a = raw[view][name]
                        if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
                            raise ValueError("invalid saved native array")
                for point_index, point_id in enumerate(point_ids):
                    result = fusion({v: raw[v]["coordinate_logits"][point_index] for v in VIEWS},
                                    {v: raw[v]["positions_xy"][point_index] for v in VIEWS})
                    if result["status"] != "completed":
                        raise ValueError("fusion did not complete: " + str(result))
                    if set(result["methods"]) != set(METHODS):
                        raise ValueError("incomplete method set")
                    predictions = {name: np.asarray(result["methods"][name], dtype=np.float64)
                                   for name in METHODS}
                    if any(p.shape != (2,) or not np.isfinite(p).all() for p in predictions.values()):
                        raise ValueError("invalid decoded point")
                    identity = {"group_id": group, "frame_id": frame_id, "point_id": int(point_id),
                                "observation_index": observation, "point_index": point_index}
                    rows.append(dict(identity, predictions={k: v.tolist() for k,v in predictions.items()},
                                     visibility_logits={v: float(raw[v]["visibility_logits"][point_index]) for v in VIEWS}))
                    diagnostic = dict(identity, solvers=result["solvers"], **result["diagnostics"])
                    if not all(numerical_checks([diagnostic]).values()):
                        raise ValueError("failed numerical diagnostic")
                    diagnostics.append(diagnostic)
        write_json(output / "predictions.json", {"methods": list(METHODS), "rows": rows})
        write_json(output / "diagnostics.json", diagnostics)
        write_json(output / "seal.json", {"status": "sealed", "created_utc": datetime.now(timezone.utc).isoformat(),
                   "rows": len(rows), "inputs": bindings, "files": {name: record(output / name)
                   for name in ("predictions.json", "diagnostics.json")},
                   "wall_seconds": time.monotonic() - started, "model_calls": 0, "GT_opened": False})
    except BaseException as exc:
        write_json(output / "failure.json", {"type": type(exc).__name__, "error": str(exc),
                   "wall_seconds": time.monotonic() - started})
        raise
    return load_json(output / "seal.json")


def summarize_cases(cases):
    """Pure scoring from per-case errors. No resampling of dependent video rows."""
    results = {}
    for group in GROUPS:
        relevant = [r for r in cases if r["group_id"] == group and r["frame_id"] != 0]
        visible = [r for r in relevant if not r["occluded"]]
        if not visible:
            raise ValueError("no eligible rows in " + group)
        metrics = {}
        for method in METHODS:
            e = np.array([r["errors"][method] for r in visible])
            all_e = np.array([r["errors"][method] for r in relevant])
            occ_e = np.array([r["errors"][method] for r in relevant if r["occluded"]])
            if not np.isfinite(all_e).all() or np.any(all_e < 0):
                raise ValueError("invalid error")
            hits = e[:,None] < THRESHOLDS[None,:]
            metrics[method] = {"quality": float(hits.mean()), "hit_counts": hits.sum(axis=0).tolist(),
                               "eligible": len(e), "coverage": 1.0, "mean_epe": float(e.mean()),
                               "median_epe": float(np.median(e)), "all_point_epe": float(all_e.mean()),
                               "occluded_epe": float(occ_e.mean()) if len(occ_e) else None}
        paired = {}
        ce = np.array([r["errors"][CANDIDATE] for r in visible])
        for method in METHODS:
            if method == CANDIDATE:
                continue
            be = np.array([r["errors"][method] for r in visible])
            paired[method] = {"corrections_4px": int(((ce < 4) & (be >= 4)).sum()),
                              "harms_4px": int(((ce >= 4) & (be < 4)).sum())}
        results[group] = {"methods": metrics, "paired": paired}
    macro = {m: {k: float(np.mean([results[g]["methods"][m][k] for g in GROUPS]))
                 for k in ("quality", "mean_epe")} for m in METHODS}
    controls = [m for m in METHODS if m != CANDIDATE]
    exact = {g: {m: Fraction(sum(results[g]["methods"][m]["hit_counts"]),
                               5 * results[g]["methods"][m]["eligible"]) for m in METHODS} for g in GROUPS}
    exact_macro = {m: sum(exact[g][m] for g in GROUPS) / len(GROUPS) for m in METHODS}
    best = max(controls, key=lambda m: exact_macro[m])
    gain = exact_macro[CANDIDATE] - exact_macro[best]
    group_gains = {g: exact[g][CANDIDATE] - max(exact[g][m] for m in controls) for g in GROUPS}
    gates = {"macro_gain_at_least_0.05": gain >= Fraction(1,20),
             "each_group_above_every_control": all(x > 0 for x in group_gains.values()),
             "macro_epe_not_worse_than_best_quality_control": macro[CANDIDATE]["mean_epe"] <= macro[best]["mean_epe"]}
    return {"groups": results, "macro": macro, "strongest_control": best, "macro_quality_gain": float(gain),
            "macro_quality_gain_exact": str(gain),
            "group_quality_gains_over_strongest": {g: float(v) for g,v in group_gains.items()}, "quality_gates": gates}


def score(derived, preparation, output):
    output = begin(output, "score")
    started = time.monotonic()
    try:
        derived, preparation = Path(derived), Path(preparation)
        seal = load_json(derived / "seal.json")
        if seal["status"] != "sealed" or seal["rows"] != 256:
            raise ValueError("incomplete prediction seal")
        for name in ("predictions.json", "diagnostics.json"):
            verify(derived / name, seal["files"][name])
        verify(preparation / "receipt.json", seal["inputs"]["preparation_receipt"])
        prep = load_json(preparation / "receipt.json")
        rows = load_json(derived / "predictions.json")["rows"]
        diagnostics = load_json(derived / "diagnostics.json")
        if len(rows) != 256:
            raise ValueError("wrong row count")
        if len(diagnostics) != len(rows):
            raise ValueError("incomplete numerical diagnostics")
        for row, diagnostic in zip(rows, diagnostics):
            if any(row[k] != diagnostic[k] for k in ("group_id", "frame_id", "point_id", "observation_index", "point_index")):
                raise ValueError("diagnostic row identity differs")
        cases, seen = [], set()
        for group in GROUPS:
            name = group + ".ground_truth.npz"
            verify(preparation / name, prep["files"][name])
            with np.load(preparation / name, allow_pickle=False) as f:
                if set(f.files) != {"points", "occluded", "frame_ids", "point_ids"}:
                    raise ValueError("unexpected GT keys")
                gt, occluded = f["points"], f["occluded"]
                point_array, frame_array = f["point_ids"], f["frame_ids"]
                if (point_array.shape != (8,) or point_array.dtype != np.int64 or
                    frame_array.shape != (16,) or frame_array.dtype != np.int64):
                    raise ValueError("invalid GT schema: identity dtype/shape differs")
                point_ids, frame_ids = point_array.tolist(), frame_array.tolist()
            if (gt.shape != (16,8,2) or gt.dtype != np.float64 or not np.isfinite(gt).all() or
                occluded.shape != (16,8) or occluded.dtype != np.bool_ or frame_ids != list(FRAMES) or
                len(point_ids) != 8 or any(type(q) is not int or not 0 <= q < 30 for q in point_ids) or
                point_ids != sorted(set(point_ids))):
                raise ValueError("invalid GT schema")
            for row in (r for r in rows if r["group_id"] == group):
                t, q = row["observation_index"], row["point_index"]
                if type(t) is not int or type(q) is not int or not 0 <= t < 16 or not 0 <= q < 8:
                    raise ValueError("invalid row indices")
                key = (group,t,q)
                if key in seen or row["frame_id"] != frame_ids[t] or row["point_id"] != point_ids[q]:
                    raise ValueError("duplicated or mismatched identity")
                seen.add(key)
                if set(row["predictions"]) != set(METHODS) or any(
                    np.asarray(v).shape != (2,) or not np.isfinite(np.asarray(v, dtype=np.float64)).all()
                    for v in row["predictions"].values()):
                    raise ValueError("invalid decoded method/coordinate schema")
                errors = {m: float(np.linalg.norm(np.asarray(row["predictions"][m])-gt[t,q])) for m in METHODS}
                cases.append({"group_id": group, "frame_id": row["frame_id"], "point_id": row["point_id"],
                              "occluded": bool(occluded[t,q]), "errors": errors})
        if len(seen) != 256:
            raise ValueError("incomplete scoring identities")
        numerical = numerical_checks(diagnostics)
        numerical["reference_decoder_parity"] = max(float(np.max(np.abs(
                     np.asarray(r["predictions"]["reference_native"]) -
                     np.asarray(r["predictions"]["reference_independent"])))) for r in rows) <= 1e-3
        result = summarize_cases(cases)
        result.update(numerical_gates=numerical, model_calls=0, status="completed_evaluation",
                      decision="GO_development_only" if all(numerical.values()) and all(result["quality_gates"].values()) else "NO_GO",
                      seal_sha256=sha(derived / "seal.json"), wall_seconds=time.monotonic()-started)
        write_json(output / "case-errors.json", cases)
        write_json(output / "results.json", result)
        write_json(output / "receipt.json", {"status": "completed_evaluation", "seal": record(derived / "seal.json"),
                   "files": {n: record(output / n) for n in ("case-errors.json", "results.json")},
                   "wall_seconds": time.monotonic()-started, "model_calls": 0})
    except BaseException as exc:
        write_json(output / "failure.json", {"type": type(exc).__name__, "error": str(exc)})
        raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("derive", "score"))
    parser.add_argument("--input", required=True)
    parser.add_argument("--preparation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = (derive if args.phase == "derive" else score)(args.input, args.preparation, args.output)
    print(json.dumps({k: v for k,v in result.items() if k in ("status", "decision", "rows", "wall_seconds")}, allow_nan=False))


if __name__ == "__main__":
    main()
