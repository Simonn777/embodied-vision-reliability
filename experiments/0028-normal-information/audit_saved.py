"""Independent, saved-only 0028 pose/metric audit.

Only stdlib and the source/file-admission helpers are imported. No metrics,
optimizer, model, array decoder, or historical truth loader is used. Opaque
NPZ bytes are hashed, never decoded. Evaluation truth_rows are read only after
the complete frontend/prediction/evaluation inventories and outer receipts.
"""
import argparse
import importlib.util
import math
from pathlib import Path
import time

_spec = importlib.util.spec_from_file_location(
    "_evr0028_audit_common", Path(__file__).with_name("common.py"))
c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c)

ARMS = ("E", "A", "B", "C", "LS", "IRLS", "ANG", "FULL",
        "NORMAL", "NORMAL_REFINE", "AFFINE")
SCENES = ("s00513", "s00475", "s00478")
KEYS = tuple((s, "B"+str(i)) for s in SCENES for i in range(1, 9))
TOLERANCE = 1e-9
HISTORICAL_TRUTH_SHA256 = "092aea6e4612d8b1d9d76e891a56db042fe9a5c9179f3e01107dc43a818919b0"


def finite(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError("not a finite real")
    return float(x)


def pose(value):
    """Proper rotation and nonzero directed t; no projection or sign repair."""
    R, t = value["R"], value["t"]
    c.require(isinstance(R, list) and len(R) == 3
              and all(isinstance(r, list) and len(r) == 3 for r in R),
              "rotation shape")
    R = [[finite(v) for v in r] for r in R]
    c.require(max(abs(x) for r in R for x in r) <= 1.00001, "rotation magnitude")
    for i in range(3):
        for j in range(3):
            c.require(abs(math.fsum(R[i][k]*R[j][k] for k in range(3))
                          - (1 if i == j else 0)) <= 1e-5, "rotation orthogonality")
    det = (R[0][0]*(R[1][1]*R[2][2]-R[1][2]*R[2][1])
           - R[0][1]*(R[1][0]*R[2][2]-R[1][2]*R[2][0])
           + R[0][2]*(R[1][0]*R[2][1]-R[1][1]*R[2][0]))
    c.require(abs(det-1) <= 1e-5, "rotation orientation")
    c.require(isinstance(t, list) and len(t) == 3, "translation shape")
    t = [finite(v) for v in t]
    scale = max(map(abs, t))
    c.require(scale > 0, "undefined translation")
    t = [v/scale for v in t]
    length = math.hypot(*t)
    return R, [v/length for v in t]


def angles(value, truth):
    R, t = pose(value)
    G, g = pose(truth)
    # Independently implemented cosine convention, matching the frozen metric
    # definition (not an atan2 replacement with different endpoint roundoff).
    cos_r = (math.fsum(R[i][j]*G[i][j] for i in range(3) for j in range(3))-1)/2
    cos_t = math.fsum(t[i]*g[i] for i in range(3))
    r = math.degrees(math.acos(min(1., max(-1., cos_r))))
    d = math.degrees(math.acos(min(1., max(-1., cos_t))))
    return dict(rotation_degrees=r, translation_degrees=d, max_degrees=max(r, d))


def auc(errors):
    """DMS polygon integral, excluding exact-threshold events before clipping."""
    values = sorted(finite(e) for e in errors)
    c.require(values and 0 <= values[0] <= values[-1] <= 180, "error population")
    result = {}
    for threshold in (5, 10, 20):
        vertices = [(0., 0.)]
        for ordinal, error in enumerate(values, 1):
            if error >= threshold:
                break
            vertices.append((error, ordinal/len(values)))
        vertices.append((float(threshold), vertices[-1][1]))
        area = math.fsum((right[0]-left[0])*(right[1]+left[1])/2
                         for left, right in zip(vertices[:-1], vertices[1:]))
        result[str(threshold)] = area*100/threshold
    return result


def summaries(rows):
    result = {}
    for arm in ARMS:
        outcomes = [row["arms"][arm] for row in rows]
        result[arm] = dict(
            denominator=len(outcomes), auc_percent=auc([o["max_degrees"] for o in outcomes]),
            status_counts={s: sum(o["status"] == s for o in outcomes)
                           for s in ("completed", "fallback", "unavailable")},
            valid_pose_count=sum(o["pose_valid"] for o in outcomes),
            genuine_completed=sum(o["genuine_completed"] for o in outcomes),
            penalized_count=sum(o["penalized"] for o in outcomes))
    return result


def gate(pooled, scenes, comparator_valid):
    controls = [a for a in ARMS if a != "ANG"]
    # Scan in declared order and replace only on strict improvement.
    best = {}
    for threshold in ("5", "20"):
        best[threshold] = controls[0]
        for arm in controls[1:]:
            if pooled[arm]["auc_percent"][threshold] > pooled[best[threshold]]["auc_percent"][threshold]:
                best[threshold] = arm
    margins = {a: {k: pooled["ANG"]["auc_percent"][k]-pooled[a]["auc_percent"][k]
                   for k in ("5", "10", "20")} for a in controls}
    scene_delta = {s: scenes[s]["ANG"]["auc_percent"]["5"]
                  - scenes[s][best["5"]]["auc_percent"]["5"] for s in SCENES}
    positive = sum(v > 0 for v in scene_delta.values())
    count = pooled["ANG"]["genuine_completed"]
    m5, m20 = margins[best["5"]]["5"], margins[best["20"]]["20"]
    clauses = dict(
        trustworthy_comparators=comparator_valid,
        auc5_gain_and_coverage=(m5 >= 2 and count >= 20),
        scene_consistency=(positive >= 2 and all(v >= -2 for v in scene_delta.values())),
        auc20_noninferiority=(m20 >= -.5))
    return dict(passed=all(clauses.values()), clauses=clauses, comparator_valid=comparator_valid,
                strongest_control_auc5=best["5"], strongest_control_auc20=best["20"],
                auc5_margin_pp=m5, auc20_margin_pp=m20, ang_genuine_completed=count,
                positive_scene_count=positive, scene_auc5_margins_pp=scene_delta,
                all_control_margins_pp=margins, manuscript_admission=False)


def recompute(predictions, truth_rows, comparator_valid):
    c.require(type(comparator_valid) is bool, "comparator boolean")
    c.require(len(predictions) == len(truth_rows) == 24, "all24 required")
    c.require([(r["scene"], r["query"]) for r in predictions] == list(KEYS)
              and [(r["scene"], r["query"]) for r in truth_rows] == list(KEYS),
              "fixed query ordering")
    rows = []
    for source, truth in zip(predictions, truth_rows):
        identity = {k: source[k] for k in ("scene", "query", "anchor", "target")}
        c.require(all(truth[k] == v for k, v in identity.items()), "truth identity")
        pose(truth)  # Invalid truth is an integrity error, not a method failure.
        c.require(set(source["arms"]) == set(ARMS), "complete eleven-arm population")
        row = dict(identity, arms={})
        for arm in ARMS:
            value = source["arms"][arm]
            status = value["status"]
            c.require(status in ("completed", "fallback", "unavailable"), "arm status")
            valid, error = False, dict(rotation_degrees=180., translation_degrees=180., max_degrees=180.)
            if status != "unavailable":
                try:
                    error = angles(value, truth)
                    valid = True
                except (ValueError, TypeError, KeyError, OverflowError):
                    pass
            row["arms"][arm] = dict(status=status, pose_valid=valid, penalized=not valid,
                                   genuine_completed=status == "completed" and valid, **error)
        rows.append(row)
    pooled = summaries(rows)
    scenes = {s: summaries([r for r in rows if r["scene"] == s]) for s in SCENES}
    return dict(rows=rows, pooled=pooled, scenes=scenes,
                gate=gate(pooled, scenes, comparator_valid))


class Checks:
    def __init__(self):
        self.count = 0
        self.errors = []

    def compare(self, expected, actual, label):
        """Compare every independently produced field; allow descriptive extras."""
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                return self.check(False, label+": mapping")
            for key, value in expected.items():
                if key not in actual:
                    self.check(False, label+"/"+key+": missing")
                else:
                    self.compare(value, actual[key], label+"/"+key)
        elif isinstance(expected, list):
            if not isinstance(actual, list) or len(actual) != len(expected):
                return self.check(False, label+": sequence length")
            for i, (a, b) in enumerate(zip(expected, actual)):
                self.compare(a, b, label+"/"+str(i))
        elif isinstance(expected, float):
            okay = type(actual) in (float, int) and math.isfinite(actual) and abs(expected-actual) <= TOLERANCE
            self.check(okay, label, expected, actual)
        else:
            self.check(type(expected) is type(actual) and expected == actual, label, expected, actual)

    def check(self, okay, label, expected=None, actual=None):
        self.count += 1
        if not okay:
            self.errors.append(dict(field=label, expected=expected, actual=actual))


def inventory(root, stage, job_path, commit, sources):
    """Hash *all* sealed files before parsing any scientific JSON."""
    root = Path(root)
    c.require(not root.is_symlink() and not (root/"seal.json").is_symlink(), "unsafe output root")
    seal = c.read(root/"seal.json")
    c.require(seal["schema"] == "evr0028."+stage+"-seal.v1"
              and seal["status"] == "completed" and seal["expected_commit"] == commit
              and seal["source_sha256"] == sources and seal["job_sha256"] == c.sha(job_path)
              and seal["query_count"] == 24 and seal["image_count"] == 27, "stage seal binding")
    files = seal["files"]
    c.require(isinstance(files, dict) and "receipt.json" in files and "seal.json" not in files,
              "stage file inventory")
    for name, digest in files.items():
        p = root/name
        c.require(Path(name).name == name and name not in (".", "..")
                  and p.is_file() and not p.is_symlink() and c.sha(p) == digest,
                  "sealed file changed: "+name)
    c.require({p.name for p in root.iterdir()} == set(files) | {"seal.json"}, "unsealed extra/missing file")
    receipt = c.read(root/"receipt.json")
    c.require(c.sha(root/"receipt.json") == seal["receipt_sha256"]
              and receipt["status"] == "completed"
              and receipt["expected_commit"] == commit and receipt["source_sha256"] == sources
              and receipt["job_sha256"] == c.sha(job_path), "stage receipt binding")
    return seal, receipt


def verify_saved(job_path, commit, sources):
    job = c.read(job_path)
    c.require(job["expected_commit"] == commit, "audit job freeze")
    roots = {k: Path(job[k]) for k in ("raw_frontend", "predictions", "evaluation")}
    outers, seals, receipts = {}, {}, {}
    for key, stage, script, outer_key in (
            ("raw_frontend", "frontend", "frontend.py", "raw_frontend_outer"),
            ("predictions", "prediction", "predict.py", "predictions_outer"),
            ("evaluation", "evaluation", "evaluate.py", "evaluation_outer")):
        outers[key] = c.verify_outer(job[outer_key], roots[key]/"seal.json", commit, script, job_path)
        seals[key], receipts[key] = inventory(roots[key], stage, job_path, commit, sources)
    front, pred = receipts["raw_frontend"], receipts["predictions"]
    c.require(front["schema"] == "evr0028.frontend.v1"
              and front["ground_truth_access"] is False and not front["offline_attempts"]
              and front["raw_forwards_attempted"] == front["raw_forwards_completed"] == 27
              and seals["raw_frontend"]["raw_forward_count"] == 27, "frontend complete/no-GT")
    c.require(pred["schema"] == "evr0028.prediction.v1" and pred["ground_truth_access"] is False
              and seals["predictions"]["ground_truth_access"] is False
              and not pred["offline_attempts"] and seals["predictions"]["arms"] == list(ARMS),
              "prediction complete/no-GT")
    c.require(len(front["images"]) == len(pred["images"]) == 27
              and len(front["selected_edges"]) == len(pred["queries"]) == 24, "image/query coverage")
    c.require([r["filename"] for r in front["images"]] == front["selected_images"]
              == [r["filename"] for r in pred["images"]]
              and len(set(front["selected_images"])) == 27, "image identities/order")
    front_names, pred_names = {"receipt.json"}, {"receipt.json"}
    by_name = {}
    for raw, processed in zip(front["images"], pred["images"]):
        i = raw["image_index"]
        c.require(type(i) is int and processed["image_index"] == i and raw["status"] == "completed"
                  and processed["status"] == "completed", "image completion")
        for mode in ("A", "B", "C"):
            c.require(raw["controls"][mode]["status"] == "completed", "camera control absent")
        for record, field, filename, root_key in (
                (raw, "raw", "raw-%03d.npz"%i, "raw_frontend"),
                (raw, "processed", "processed-%03d.npz"%i, "raw_frontend"),
                (processed, "samples_file", "samples-%03d.npz"%i, "predictions")):
            meta = record[field]
            c.require(meta["filename"] == filename and meta["sha256"] == seals[root_key]["files"][filename]
                      and meta["bytes"] == (roots[root_key]/filename).stat().st_size, "array byte binding")
        front_names.update(("raw-%03d.npz"%i, "processed-%03d.npz"%i, "image-%03d.json"%i))
        pred_names.update(("samples-%03d.npz"%i, "image-%03d.json"%i))
        c.require(c.read(roots["raw_frontend"]/("image-%03d.json"%i)) == raw
                  and c.read(roots["predictions"]/("image-%03d.json"%i)) == processed, "image JSON duplicate")
        c.require(type(processed["comparator_valid"]) is bool
                  and processed["comparator_valid"] == (processed["full"]["status"] == "completed"),
                  "FULL image validity")
        by_name[raw["filename"]] = processed
    c.require(pred["raw_frontend_binding"] == dict(outer=outers["raw_frontend"],
              seal_sha256=c.sha(roots["raw_frontend"]/"seal.json")), "prediction/frontend linkage")
    rows = []
    for i, (item, edge) in enumerate(zip(pred["queries"], front["selected_edges"])):
        name = "pair-%02d.json"%i
        c.require(item["pair_index"] == i and item["filename"] == name
                  and item["sha256"] == seals["predictions"]["files"][name], "pair file linkage")
        row = c.read(roots["predictions"]/name)
        c.require(row["schema"] == "evr0028.pair.v1" and row["status"] == "completed"
                  and row["phase"] == "complete" and row["pair_index"] == i
                  and (row["scene"], row["query"]) == KEYS[i], "pair identity/completion")
        for key in ("scene", "query", "anchor", "target", "edge_index", "original_match_indices", "selected_match_ids"):
            c.require(row[key] == item[key], "pair duplicate identity")
        c.require((row["scene"], row["anchor"], row["target"], row["edge_index"])
                  == (edge["scene"], edge["source"], edge["target"], edge["edge_index"]), "fixed edge identity")
        for arm, outcome in row["arms"].items():
            if outcome["status"] == "fallback":
                E = row["arms"]["E"]
                c.require(arm != "E" and E["status"] == "completed"
                          and outcome["R"] == E["R"] and outcome["t"] == E["t"], "fallback changed E")
        full_attempts = [v["result"] for v in row["similarity_attempts"] if v["arm"] == "FULL"]
        c.require(len(full_attempts) <= 1, "duplicate FULL attempt")
        validity = (by_name[row["anchor"]]["comparator_valid"]
                    and by_name[row["target"]]["comparator_valid"]
                    and all(v["status"] == "completed" for v in full_attempts))
        c.require(type(row["comparator_valid"]) is bool and row["comparator_valid"] == validity,
                  "pair comparator validity")
        input_name, partial_name = "pair-%02d-input.npz"%i, "pair-%02d.partial.json"%i
        c.require(row["inputs"]["filename"] == input_name
                  and row["inputs"]["sha256"] == seals["predictions"]["files"][input_name]
                  and row["inputs"]["bytes"] == (roots["predictions"]/input_name).stat().st_size,
                  "pair input binding")
        partial = c.read(roots["predictions"]/partial_name)
        c.require(partial["progress"]["status"] == "completed"
                  and partial["progress"]["phase"] == "complete"
                  and partial["progress"]["arms"] == row["arms"], "final partial outcome")
        pred_names.update((name, input_name, partial_name))
        rows.append(row)
    comparator = all(r["comparator_valid"] for r in pred["images"]+rows)
    c.require(pred["comparator_valid"] is comparator
              and seals["predictions"]["comparator_valid"] is comparator, "aggregate comparator validity")
    c.require(set(seals["raw_frontend"]["files"]) == front_names
              and set(seals["predictions"]["files"]) == pred_names
              and set(seals["evaluation"]["files"]) == {"evaluation.json", "receipt.json"},
              "exact stage inventories")
    c.require(receipts["evaluation"]["schema"] == "evr0028.evaluation.v1"
              and receipts["evaluation"]["eligible"] is True
              and receipts["evaluation"]["ground_truth_access"] is True
              and seals["evaluation"]["ground_truth_access"] is True
              and receipts["evaluation"]["query_count"] == 24
              and receipts["evaluation"]["image_count"] == 27
              and receipts["evaluation"]["evaluation_sha256"] == seals["evaluation"]["files"]["evaluation.json"],
              "evaluation receipt")
    return job, roots, seals, receipts, outers, rows, comparator


def expected_bindings(job, roots, seals, receipts, outers):
    front, pred = receipts["raw_frontend"], receipts["predictions"]
    c.require(front["input_bindings"]["frontend"] == pred["old_frontend_binding"]
              and front["input_bindings"]["plan_sha256"] == c.PLAN_SHA
              and front["input_bindings"]["cache_file_sha256"] == pred["source_input_files"],
              "inherited input binding disagreement")
    native = {}
    for name in ("native_library", "native_receipt"):
        path = Path(job[name])
        c.require(not path.is_symlink() and path.is_file()
                  and c.sha(path) == job[name+"_sha256"], "native asset changed")
        native[name+"_sha256"] = c.sha(path)
    c.require(pred["runtime"]["tc_library_sha256"] == native["native_library_sha256"],
              "loaded native digest")
    return dict(
        prediction_outer=outers["predictions"],
        prediction_seal_sha256=c.sha(roots["predictions"]/"seal.json"),
        prediction_receipt_sha256=c.sha(roots["predictions"]/"receipt.json"),
        prediction_files=seals["predictions"]["files"],
        raw_frontend=dict(outer=outers["raw_frontend"],
                          seal_sha256=c.sha(roots["raw_frontend"]/"seal.json")),
        raw_frontend_files=seals["raw_frontend"]["files"],
        old_frontend=pred["old_frontend_binding"], source_input_files=pred["source_input_files"],
        model=front["input_bindings"]["model"], native=pred["native_binding"], **native)


def run(args):
    started = time.perf_counter()
    sources = c.verify_source(args.expected_commit)
    job_hash = c.sha(args.job)
    output = Path(args.output)
    c.require(not output.resolve().is_relative_to(c.ROOT), "private audit output required")
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema="evr0028.audit.v1", status="started", expected_commit=args.expected_commit,
                   source_sha256=sources, job_sha256=job_hash, checks=0, errors=[],
                   evaluation_values_read=False, numeric_tolerance_absolute=TOLERANCE)
    c.write(output/"receipt.json", receipt)
    try:
        job, roots, seals, receipts, outers, rows, comparator = verify_saved(args.job, args.expected_commit, sources)
        bindings = expected_bindings(job, roots, seals, receipts, outers)
        c.require(receipts["evaluation"]["prediction_bindings"] == bindings
                  and job["ground_truth_evaluation_sha256"] == HISTORICAL_TRUTH_SHA256,
                  "evaluation receipt/declared truth binding")
        receipt["evaluation_values_read"] = True
        c.write(output/"receipt.json", receipt, replace=True)
        saved = c.read(roots["evaluation"]/"evaluation.json")
        c.require(saved["schema"] == "evr0028.metrics.v1"
                  and saved["expected_commit"] == args.expected_commit
                  and saved["source_sha256"] == sources and saved["job_sha256"] == job_hash
                  and saved["bindings"] == bindings
                  and saved["ground_truth_evaluation_sha256"] == HISTORICAL_TRUTH_SHA256,
                  "evaluation source/job")
        independent = recompute(rows, saved["truth_rows"], comparator)
        checks = Checks()
        checks.compare(dict(query_count=24, scene_count=3, arm_order=list(ARMS), scene_order=list(SCENES),
                            primary_error="max(rotation_angle,directed_translation_angle)",
                            failure_penalty_degrees=180., auc_convention="DMS_left_search_trapezoid_percent",
                            manuscript_admission=False, **independent), saved, "evaluation")
        checks.compare(independent["gate"]["passed"], receipts["evaluation"]["gate_passed"], "receipt/gate")
        # Repeat all opaque hashes and process checks after metric calculation.
        post = verify_saved(args.job, args.expected_commit, sources)
        c.require(expected_bindings(post[0], post[1], post[2], post[3], post[4]) == bindings,
                  "post-audit binding changed")
        c.require(c.source_hashes() == sources and c.sha(args.job) == job_hash, "audit source/job changed")
        receipt.update(status="completed" if not checks.errors else "failed",
                       passed=not checks.errors, checks=checks.count, errors=checks.errors,
                       error_count=len(checks.errors), query_count=24, pose_outcomes=24*len(ARMS),
                       independent=independent, bindings=dict(outers=outers,
                       seal_sha256={k:c.sha(v/"seal.json") for k,v in roots.items()}),
                       scope=dict(recomputed="directed pose errors, DMS trapezoids, coverage, all four gate clauses",
                                  not_recomputed=["native/MSAC scores", "depth corrections", "model outputs",
                                                  "historical GT extraction", "dispatcher spec bytes"],
                                  array_access="opaque file hashing only",
                                  inherited_models_and_cache="receipt linkage; evaluator owns full original-chain admission"),
                       wall_seconds=time.perf_counter()-started)
        c.write(output/"receipt.json", receipt, replace=True)
        if not checks.errors:
            c.write(output/"seal.json", dict(
                schema="evr0028.audit-seal.v1", status="completed",
                expected_commit=args.expected_commit, source_sha256=sources, job_sha256=job_hash,
                query_count=24, image_count=27, receipt_sha256=c.sha(output/"receipt.json"),
                files={"receipt.json":c.sha(output/"receipt.json")}))
        print(dict(status=receipt["status"], checks=checks.count, errors=len(checks.errors)))
        return 0 if not checks.errors else 1
    except BaseException as exc:
        receipt.update(status="failed", passed=False, error_type=type(exc).__name__, error=str(exc),
                       wall_seconds=time.perf_counter()-started)
        c.write(output/"receipt.json", receipt, replace=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    raise SystemExit(run(parser.parse_args()))
