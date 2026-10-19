"""Evaluate all sealed 0023 predictions, then and only then acquire three GT files.

This custom three-place development diagnostic is not an official Map-free
benchmark submission. No estimator or model is called by this module.
"""
import argparse
from bisect import bisect_left
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCENES = ("s00513", "s00475", "s00478")
METHODS = ("native_E", "geometry_multiview", "joint_field", "hessian", "scalar_deletion",
           "residual_scalar", "directional")
THRESHOLDS = (5, 10, 20)
QUATERNION_NORM_TOLERANCE = 1e-3


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


supervisor = module("_evr0023_evaluation_supervisor", HERE / "supervise.py")
ranges = module("_evr0023_evaluation_ranges", HERE / "range_inputs.py")
planner = module("_evr0023_evaluation_planner", HERE / "propose_plan.py")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return supervisor.sha256(path)


def finite(value, name="number"):
    require(type(value) in (int, float) and math.isfinite(value), "nonfinite/invalid " + name)
    return float(value)


def read_json(path, limit=128 * 1024**2):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit,
            "bounded regular JSON required")
    def bad_constant(value):
        raise ValueError("nonfinite JSON: " + value)
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(path.read_text(), parse_constant=bad_constant, object_pairs_hook=unique_object,
                      parse_float=lambda value: finite(float(value), "JSON number"))


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def vector(value):
    require(isinstance(value, (list, tuple)) and len(value) == 3, "three-vector required")
    return [finite(x) for x in value]


def mm(a, b):
    return [[math.fsum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def transpose(a):
    return [list(row) for row in zip(*a)]


def proper_rotation(value):
    require(isinstance(value, (list, tuple)) and len(value) == 3, "three-by-three rotation required")
    r = [vector(row) for row in value]
    gram = mm(r, transpose(r))
    determinant = (r[0][0] * (r[1][1]*r[2][2] - r[1][2]*r[2][1])
                   - r[0][1] * (r[1][0]*r[2][2] - r[1][2]*r[2][0])
                   + r[0][2] * (r[1][0]*r[2][1] - r[1][1]*r[2][0]))
    require(all(math.isfinite(v) for row in gram for v in row)
            and max(abs(gram[i][j] - (i == j)) for i in range(3) for j in range(3)) <= 1e-5
            and math.isfinite(determinant) and abs(determinant - 1) <= 1e-5,
            "invalid proper rotation")
    return r


def unit_translation(value):
    t = vector(value)
    norm = math.hypot(*t)
    require(math.isfinite(norm) and norm > 0, "undefined translation direction")
    return [v / norm for v in t], norm


def checked_pose(value):
    require(isinstance(value, (list, tuple)) and len(value) == 2, "pose must be [R,t]")
    r = proper_rotation(value[0]); t = vector(value[1]); unit_translation(t)
    return [r, t]


def angle(cosine):
    return math.degrees(math.acos(max(-1.0, min(1.0, finite(cosine)))))


def pose_error(pose, truth):
    r, t = checked_pose(pose); rg, tg = checked_pose(truth)
    rotation = angle((math.fsum(r[i][j]*rg[i][j] for i in range(3) for j in range(3)) - 1) / 2)
    u, _ = unit_translation(t); ug, _ = unit_translation(tg)
    directed = angle(math.fsum(x*y for x, y in zip(u, ug)))
    return dict(rotation_degrees=rotation, translation_degrees=directed,
                max_degrees=max(rotation, directed),
                signfolded_max_degrees=max(rotation, min(directed, 180-directed)))


def dms_auc(errors, thresholds=THRESHOLDS):
    """Equivalent to frozen 0013 DMS left-search/trapezoid fraction convention."""
    require(isinstance(errors, (list, tuple)) and len(errors) > 0, "nonempty errors required")
    values = [finite(x, "error") for x in errors]
    require(min(values) >= 0, "negative error")
    x = [0.0] + sorted(values)
    recall = [i / len(values) for i in range(len(values) + 1)]
    result = {}
    for threshold in thresholds:
        require(finite(threshold) > 0, "positive threshold required")
        stop = bisect_left(x, threshold)
        xx = x[:stop] + [threshold]; yy = recall[:stop] + [recall[stop-1]]
        area = math.fsum((xx[i+1]-xx[i])*(yy[i+1]+yy[i])*.5 for i in range(len(xx)-1))
        result[str(threshold)] = area / threshold
    return result


def plan_queries(plan):
    require(plan.get("schema") == "evr0023.proposed-plan.v1"
            and tuple(s["scene"] for s in plan["scenes"]) == SCENES, "fixed three-place plan required")
    result = []
    for scene in plan["scenes"]:
        prefix = "val/" + scene["scene"] + "/"
        require(scene["anchor"] == prefix + "seq0/frame_00000.jpg", "anchor identity")
        selected = scene["selected_seq1_chronological"]
        require(len(selected) == 16, "sixteen chronological rows required")
        queries = []
        for k, row in enumerate(selected, 1):
            require(row["selected_index_one_based"] == k
                    and row["role"] == ("B_query" if k % 2 else "C_reference")
                    and re.fullmatch(re.escape(prefix) + r"seq1/frame_[0-9]{5}\.jpg", row["filename"]),
                    "chronological query identity")
            if k % 2:
                queries.append(row["filename"])
        require(scene["query_images"] == queries and len(set(queries)) == 8, "eight distinct queries required")
        for k, query in enumerate(queries, 1):
            result.append(dict(scene=scene["scene"], query="B" + str(k),
                               anchor=scene["anchor"], target=query))
    return result


def prepare_predictions(plan, scene_records):
    """Validate the entire 24 x 7 population before any GT byte is permitted."""
    identities = plan_queries(plan)
    require(set(scene_records) == set(SCENES), "all three scene outputs required")
    result = []
    for scene in SCENES:
        record = scene_records[scene]
        require(record.get("schema") == "evr0023.scene-prediction.v1"
                and record.get("scene") == scene and record.get("status") == "completed"
                and record.get("ground_truth_access") is False, "complete GT-free scene output required")
        queries = record.get("queries")
        require(isinstance(queries, list) and len(queries) == 8
                and [row.get("query") for row in queries] == ["B"+str(k) for k in range(1, 9)],
                "all eight ordered query outcomes required")
        for row in queries:
            require(row.get("scene") == scene and set(row.get("arms", {})) == set(METHODS[1:]),
                    "all six method arms required")
            outcomes = {"native_E": row["native_E"], **row["arms"]}
            for method, outcome in outcomes.items():
                require(outcome.get("status") in ("completed", "fallback", "failed"), "nonterminal method status")
                require(method != "native_E" or outcome["status"] != "fallback", "native cannot fall back")
                if outcome["status"] == "failed":
                    require(outcome.get("pose") is None, "failed prediction has a pose")
                else:
                    checked_pose(outcome.get("pose"))
                    if outcome["status"] == "fallback":
                        require(outcome.get("fallback") == "native_E_reference_filtered_query"
                                and outcomes["native_E"].get("status") == "completed"
                                and outcome["pose"] == outcomes["native_E"]["pose"], "fallback identity mismatch")
            identity = next(v for v in identities if v["scene"] == scene and v["query"] == row["query"])
            result.append(dict(identity=identity, outcomes=outcomes))
    require(len(result) == 24, "all queries retained")
    return result


def quaternion_rotation(raw):
    require(len(raw) == 4, "quaternion requires qw qx qy qz")
    q = [finite(x, "quaternion") for x in raw]
    norm = math.hypot(*q)
    require(math.isfinite(norm) and norm > 0 and abs(norm-1) <= QUATERNION_NORM_TOLERANCE,
            "GT quaternion norm outside fixed tolerance")
    w, x, y, z = [v / norm for v in q]
    rotation = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    return proper_rotation(rotation), dict(raw=q, norm=norm, normalized=[w,x,y,z],
                                          renormalized=(norm != 1.0))


def parse_poses(text, selected_names):
    """Official W2C format: frame_path qw qx qy qz tx ty tz; no unit guessing."""
    wanted = set(selected_names); found = {}; seen = set(); normalized_count = 0
    require(len(wanted) == len(selected_names), "duplicate requested GT image")
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        require(re.fullmatch(r"seq[01]/frame_[0-9]{5}\.jpg", fields[0]),
                "invalid authored pose row at line " + str(line_number))
        name = fields[0]
        require(name not in seen, "duplicate authored GT filename")
        seen.add(name)
        if name not in wanted:
            continue  # Only filenames/duplicates are inspected outside selected A/B.
        require(len(fields) == 8, "selected pose must have eight authored columns")
        try:
            numbers = [float(x) for x in fields[1:]]
        except ValueError as exc:
            raise ValueError("nonnumeric authored GT row") from exc
        require(all(math.isfinite(x) for x in numbers), "nonfinite authored GT row")
        rotation, q = quaternion_rotation(numbers[:4])
        normalized_count += q["renormalized"]
        found[name] = dict(R=rotation, t=vector(numbers[4:]), quaternion=q, source_line=line_number)
    require(set(found) == wanted, "missing selected GT pose")
    return found, dict(filename_rows_scanned=len(seen), numeric_rows_parsed=len(found), selected_rows=len(found),
                      quaternion_norm_tolerance=QUATERNION_NORM_TOLERANCE,
                      renormalized_selected_rows=normalized_count, convention="world-to-camera qw,qx,qy,qz,t")


def relative_pose(anchor, query):
    r = mm(query["R"], transpose(anchor["R"]))
    t = [query["t"][i] - math.fsum(r[i][j]*anchor["t"][j] for j in range(3)) for i in range(3)]
    return checked_pose([r, t])


def summarize(rows):
    errors = [r["error"]["max_degrees"] for r in rows]
    folded = [r["error"]["signfolded_max_degrees"] for r in rows]
    counts = Counter(r["status"] for r in rows)
    middle = len(errors)//2
    median = ((sorted(errors)[middle-1]+sorted(errors)[middle])/2 if len(errors)%2 == 0 else sorted(errors)[middle])
    return dict(count=len(rows), completed_count=counts["completed"], fallback_count=counts["fallback"],
                failed_count=counts["failed"], auc_fraction=dms_auc(errors),
                auc_percent={k:100*v for k,v in dms_auc(errors).items()},
                signfolded_auc_fraction=dms_auc(folded),
                median_max_error_degrees=median)


def evaluate_predictions(prepared, gt):
    rows = []
    for prediction in prepared:
        identity = prediction["identity"]; scene = identity["scene"]
        prefix = "val/" + scene + "/"
        anchor = gt[scene][identity["anchor"][len(prefix):]]
        target = gt[scene][identity["target"][len(prefix):]]
        truth = relative_pose(anchor, target)
        outcomes = {}
        for method in METHODS:
            outcome = prediction["outcomes"][method]
            failed = outcome["status"] == "failed"
            error = (dict(rotation_degrees=180.0, translation_degrees=180.0, max_degrees=180.0,
                          signfolded_max_degrees=180.0) if failed else pose_error(outcome["pose"], truth))
            outcomes[method] = dict(status=outcome["status"], pose=outcome.get("pose"), error=error,
                                    reason=outcome.get("reason"), fallback=outcome.get("fallback"),
                                    failure_penalty=failed)
        rows.append(dict(**identity, ground_truth=dict(anchor=anchor, target=target, relative=truth), methods=outcomes))
    require(len(rows) == 24 and Counter(r["scene"] for r in rows) == Counter({s:8 for s in SCENES}),
            "fixed 24-query denominator required")
    overall = {m:summarize([r["methods"][m] for r in rows]) for m in METHODS}
    scenes = {s:{m:summarize([r["methods"][m] for r in rows if r["scene"] == s]) for m in METHODS} for s in SCENES}
    controls = METHODS[:-1]
    margins = {m:{k:overall["directional"]["auc_percent"][k]-overall[m]["auc_percent"][k]
                  for k in ("5","10","20")} for m in controls}
    scene_margins = {s:{m:scenes[s]["directional"]["auc_percent"]["5"]-scenes[s][m]["auc_percent"]["5"]
                         for m in controls} for s in SCENES}
    best5 = max(controls, key=lambda m: (overall[m]["auc_fraction"]["5"], -METHODS.index(m)))
    best20 = max(controls, key=lambda m: (overall[m]["auc_fraction"]["20"], -METHODS.index(m)))
    worst_scene_margin = {s:min(scene_margins[s].values()) for s in SCENES}
    components = dict(auc5_gain_at_least_2pp=margins[best5]["5"] >= 2.0,
                      positive_in_at_least_two_scenes=sum(v > 0 for v in worst_scene_margin.values()) >= 2,
                      no_scene_loss_beyond_1pp=min(worst_scene_margin.values()) >= -1.0,
                      auc20_loss_no_more_than_half_pp=margins[best20]["20"] >= -.5)
    return dict(schema="evr0023.evaluation.v1", query_count=24, method_names=list(METHODS),
                method_display_names={m:("native_E_reference_filtered_query" if m=="native_E" else m) for m in METHODS}, rows=rows,
                overall=overall, scenes=scenes,
                gate=dict(primary="directional", controls=list(controls), best_fixed_auc5_control=best5,
                          best_fixed_auc20_control=best20, margins_pp=margins, scene_margins_auc5_pp=scene_margins,
                          worst_control_margin_by_scene_pp=worst_scene_margin, components=components,
                          passed=all(components.values()), auc20_requirement=True),
                interpretation="Three development places; no bootstrap confidence/SOTA/official Map-free track claim. All24queries retained. Failed method poses receive an explicit180degree coverage penalty; fallbacks retain nativeE poses. Directed translation is primary; signfolded is secondary.")


def expand(value):
    path = Path(value.replace("$REPOSITORY", str(ROOT)).replace("$PRIVATE", str(ROOT.parent)))
    return (path if path.is_absolute() else ROOT / path).resolve()


def verify_prediction_seal(args):
    """No labels or HTTP. All predicted states are checked before returning."""
    own = [Path(__file__), HERE / "predict.py", HERE / "supervise.py", HERE / "range_inputs.py",
           HERE / "propose_plan.py", args.protocol]
    frozen = supervisor.verify_frozen(args.expected_commit, own)
    # This source was just checked against Git; its import is stdlib-only and
    # source_paths performs no prediction, native call, or asset access.
    predictor = module("_evr0023_evaluation_predictor", HERE / "predict.py")
    prediction_sources = supervisor.verify_frozen(args.expected_commit, predictor.source_paths(args.protocol))
    require(re.fullmatch(r"[0-9a-f]{64}", args.plan_sha256)
            and sha(args.plan) == args.plan_sha256 and args.plan_sha256 in args.protocol.read_text(),
            "frozen protocol/private plan binding")
    plan = read_json(args.plan, 1024**2)
    index = ranges.load_index(args.members_index, args.central_directory, args.index_receipt,
                              plan["source"]["cached_index_sha256"])
    expected = planner.build_plan(index, plan["prior_resource_draft_sha256"])
    expected["planning_source_sha256"] = {name:sha(HERE / name) for name in ("propose_plan.py", "range_inputs.py")}
    require(plan == expected, "deterministic adopted plan mismatch")
    root = args.predictions
    require(root.is_dir() and not root.is_symlink(), "regular prediction directory required")
    receipt = read_json(root / "receipt.json"); seal = read_json(root / "seal.json")
    require(receipt.get("schema") == "evr0023.prediction.v1" and receipt.get("status") == "completed"
            and receipt.get("eligible") is True and receipt.get("expected_commit") == args.expected_commit
            and receipt.get("plan_sha256") == args.plan_sha256 and receipt.get("query_count") == 24
            and receipt.get("method_names") == list(METHODS) and receipt.get("ground_truth_access") is False
            and receipt.get("offline_attempts") == [] and receipt.get("source_sha256") == prediction_sources,
            "complete frozen GT-free prediction receipt required")
    require(0 <= finite(receipt.get("wall_seconds"), "prediction wall") < 1800, "prediction wall limit")
    require(seal.get("schema") == "evr0023.prediction-seal.v1" and seal.get("complete") is True
            and seal.get("query_count") == 24 and seal.get("receipt_sha256") == sha(root / "receipt.json")
            and seal.get("outputs") == receipt.get("outputs"), "prediction seal mismatch")
    expected_names = {"selection.json", *("scene-"+scene+".json" for scene in SCENES)}
    require(set(receipt["outputs"]) == expected_names, "exact four prediction outputs required")
    saved = {}
    for name in sorted(expected_names):
        meta = receipt["outputs"][name]; path = root / name
        require(set(meta) == {"sha256", "bytes"} and type(meta["bytes"]) is int and meta["bytes"] > 0
                and path.is_file() and not path.is_symlink() and path.stat().st_size == meta["bytes"]
                and sha(path) == meta["sha256"], "sealed output hash/size mismatch")
        saved[name] = read_json(path)
    prepared = prepare_predictions(plan, {scene:saved["scene-"+scene+".json"] for scene in SCENES})
    outer = read_json(args.prediction_supervision)
    require(outer.get("schema") == "evr0023.supervision.v1" and outer.get("operation") == "prediction"
            and outer.get("status") == "completed" and outer.get("exit_code") == 0
            and outer.get("expected_commit") == args.expected_commit and outer.get("termination") is None
            and outer.get("remaining_descendant_pids") == [] and outer.get("cleanup_errors") == [],
            "successful outer prediction supervision required")
    outer_paths = [HERE / "supervise.py", HERE / "predict.py"] + [HERE / name for name in supervisor.HELPERS["prediction"]]
    require(outer.get("source_sha256") == {p.relative_to(ROOT).as_posix():prediction_sources[p.relative_to(ROOT).as_posix()]
                                            for p in outer_paths}, "prediction supervisor source map")
    require(0 <= finite(outer.get("wall_seconds"), "outer wall") < 1800
            and outer.get("hard_timeout_seconds") == 1800 and outer.get("group_rss_limit_kib") == 8 * 1024**2
            and type(outer.get("peak_group_rss_kib")) is int and 0 <= outer["peak_group_rss_kib"] <= 8 * 1024**2
            and outer.get("thread_environment") == {key:"4" for key in supervisor.THREAD_KEYS},
            "prediction supervision resource limits")
    logs = {}
    for kind in ("stdout", "stderr"):
        logs[kind] = sha(args.prediction_supervision.parent / (kind+".log"))
        require(logs[kind] == outer.get(kind+"_sha256"), "prediction supervision log hash")
    command = outer.get("command")
    require(isinstance(command, list) and command[:2] == ["$PYTHON", "$REPOSITORY/experiments/0023-mapfree/predict.py"]
            and len(command) >= 4 and len(command[2:])%2 == 0, "prediction command")
    flags = dict(zip(command[2::2], command[3::2]))
    require(len(flags)*2 == len(command[2:]) and set(flags) == {"--protocol", "--plan", "--frontend-cache",
            "--frontend-supervision", "--native-runtime", "--output-dir", "--expected-commit", "--plan-sha256"}
            and flags["--expected-commit"] == args.expected_commit and flags["--plan-sha256"] == args.plan_sha256,
            "prediction command flags")
    for flag, path in (("--protocol", args.protocol), ("--plan", args.plan), ("--output-dir", root)):
        require(expand(flags[flag]) == path.resolve(), "prediction command path binding")
    require(sha(expand(flags["--frontend-cache"]) / "receipt.json") == receipt.get("frontend_receipt_sha256")
            and sha(expand(flags["--frontend-supervision"])) == receipt.get("frontend_supervision_sha256"),
            "prediction frontend input bindings")
    # A hash alone cannot establish an eligible frontend/native input. Reuse
    # the frozen predictor's complete admission proof before any GT request.
    # These functions hash opaque files and metadata; they load no arrays,
    # images, models, or native library and do not install the offline guard.
    predictor_args = argparse.Namespace(expected_commit=args.expected_commit, plan_sha256=args.plan_sha256,
        **{name:expand(flags["--"+name.replace("_","-")]) for name in
           ("protocol","plan","frontend_cache","frontend_supervision","native_runtime","output_dir")})
    front, _, frontend_record, frontend_bindings, frontend_sources = predictor.frontend_admission(predictor_args)
    require(all(receipt.get(key) == value for key,value in frontend_bindings.items())
            and receipt.get("installation_binding") == frontend_record.get("frontend_bindings")
            and receipt.get("frontend_source_sha256") == frontend_record.get("source_sha256")
            and receipt.get("frontend_freeze") == frontend_record.get("frontend_freeze")
            and receipt.get("frontend_complete_wall_seconds") == frontend_record.get("wall_seconds"),
            "prediction/frontend eligibility or provenance mismatch")
    require(predictor.native_binding(predictor_args.native_runtime, front.q) == receipt.get("native_binding"),
            "prediction native source/binary binding mismatch")
    lines = (args.prediction_supervision.parent / "stdout.log").read_text().splitlines()
    require(bool(lines) and json.loads(lines[-1]) == {key:receipt[key] for key in ("status", "eligible", "wall_seconds")},
            "prediction stdout/receipt summary mismatch")
    return dict(barrier_complete=True, plan=plan, index=index, prepared=prepared,
                source_sha256={**prediction_sources, **frozen, **frontend_sources},
                bindings=dict(plan_sha256=args.plan_sha256, prediction_receipt_sha256=sha(root / "receipt.json"),
                              prediction_seal_sha256=sha(root / "seal.json"), prediction_outputs=receipt["outputs"],
                              prediction_supervision_sha256=sha(args.prediction_supervision),
                              prediction_supervision_log_sha256=logs,
                              frontend_receipt_sha256=receipt["frontend_receipt_sha256"]))


def acquire_ground_truth(admission, output_dir, receipt_path, opener=None):
    """Explicit GT-only phase. Does not relax range_inputs.acquire's allowlist."""
    require(admission.get("barrier_complete") is True and len(admission["prepared"]) == 24,
            "complete prediction barrier required before any GT request")
    index = admission["index"]
    names = ["val/"+scene+"/poses.txt" for scene in SCENES]
    require(admission["plan"]["future_evaluation_only"]["pose_member_names"] == names,
            "exact future-evaluation pose member list required")
    entries = [index.entries[name] for name in names]
    limits = ranges.Limits(body_bytes=2 * 1024**2, output_bytes=sum(e["uncompressed_bytes"] for e in entries),
                           member_bytes=max(e["uncompressed_bytes"] for e in entries), requests=9,
                           wall_seconds=1800, request_seconds=30)
    limits.validate()
    require(sum(e["compressed_bytes"]+30+len(e["filename"])+65535+3 for e in entries) <= limits.body_bytes,
            "GT metadata worst-case body exceeds fixed cap")
    require(not output_dir.exists() and not output_dir.is_symlink()
            and not receipt_path.exists() and not receipt_path.is_symlink(), "fresh GT output and receipt required")
    record = dict(schema="evr0023.gt-range-acquisition.v1", status="started", phase="prepare",
                  prediction_barrier=admission["bindings"], url=index.url, etag=index.etag,
                  archive_bytes=index.archive_bytes, index_sha256=index.fingerprints,
                  source_sha256=sha(Path(__file__)), range_helper_sha256=sha(HERE / "range_inputs.py"),
                  selected_names=names, limits=vars(limits), requests=[], members=[], response_body_bytes=0,
                  response_body_bytes_exact=True, requested_body_upper_bound=0, output_bytes=0,
                  hidden_retries=False, full_zip_fallback=False,
                  checksum_semantics="Observed local member SHA; fixed ETag, ZIP metadata and CRC. No publisher member SHA claim.")
    write_new(receipt_path, record)
    def save():
        pending = receipt_path.with_suffix(".pending")
        with pending.open("w") as stream:
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
        os.replace(pending, receipt_path)
    began = time.monotonic()
    try:
        output_dir.mkdir()
        reader = ranges.RangeReader(index, limits, record, save, opener)
        for entry in entries:
            record["phase"] = entry["filename"]; save()
            data, metadata = ranges.extract_member(reader, entry)
            require(len(data) == entry["uncompressed_bytes"] and len(data) <= limits.member_bytes,
                    "GT member output bound")
            if time.monotonic()-began > limits.wall_seconds:
                raise TimeoutError("GT acquisition deadline before output")
            target = output_dir / entry["filename"]
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(data)
            record["members"].append(dict(entry, **metadata))
            record["output_bytes"] += len(data); save()
        require(len(record["requests"]) == 9 and record["output_bytes"] == limits.output_bytes
                and record["response_body_bytes_exact"] is True, "incomplete GT range acquisition")
        if time.monotonic()-began > limits.wall_seconds:
            raise TimeoutError("GT acquisition deadline after output")
        record.update(status="completed", phase="completed")
    except BaseException as exc:
        record.update(status="failed", error=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        record["wall_seconds"] = time.monotonic()-began
        save()
    return record


def run(args):
    output = args.output_dir
    require(not output.exists() and not output.is_symlink(), "fresh evaluation output required")
    require(not output.resolve().is_relative_to(ROOT), "raw GT payloads must remain private outside repository")
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    record = dict(schema="evr0023.evaluation-operation.v1", status="started", eligible=False, stage="seal_verification",
                  expected_commit=args.expected_commit, plan_sha256=args.plan_sha256, gt_access_started=False)
    def save():
        pending = output / "receipt.pending"
        with pending.open("w") as stream:
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
        os.replace(pending, output / "receipt.json")
    save()
    try:
        admission = verify_prediction_seal(args)
        record.update(source_sha256=admission["source_sha256"], prediction_bindings=admission["bindings"],
                      stage="ground_truth_acquisition", gt_access_started=True)
        save()
        acquired = acquire_ground_truth(admission, output / "gt", output / "gt-acquisition.json")
        record["gt_acquisition_sha256"] = sha(output / "gt-acquisition.json")
        record["stage"] = "ground_truth_parsing"; save()
        truth, parsing = {}, {}
        for scene in SCENES:
            prefix = "val/"+scene+"/"
            identities = [row["identity"] for row in admission["prepared"] if row["identity"]["scene"] == scene]
            names = [identities[0]["anchor"][len(prefix):]] + [r["target"][len(prefix):] for r in identities]
            path = output / "gt" / prefix / "poses.txt"
            expected = next(row for row in acquired["members"] if row["filename"] == prefix+"poses.txt")
            require(path.stat().st_size == expected["uncompressed_bytes"] and sha(path) == expected["sha256"],
                    "GT saved output changed")
            truth[scene], parsing[scene] = parse_poses(path.read_text(encoding="utf-8"), names)
        record["stage"] = "scoring"; save()
        result = evaluate_predictions(admission["prepared"], truth)
        result.update(expected_commit=args.expected_commit, source_sha256=admission["source_sha256"],
                      bindings=admission["bindings"], gt_acquisition_sha256=record["gt_acquisition_sha256"], gt_parsing=parsing)
        # Recheck all original sealed predictions and sources after evaluation;
        # this is an evaluation-only process with no frontend offline guard.
        final = verify_prediction_seal(args)
        require(final["bindings"] == admission["bindings"] and final["source_sha256"] == admission["source_sha256"],
                "prediction/source bindings changed during evaluation")
        require(time.monotonic()-began < 1800, "evaluation operation deadline")
        write_new(output / "evaluation.json", result)
        record.update(status="completed", eligible=True, stage="completed", query_count=24,
                      evaluation_sha256=sha(output / "evaluation.json"), development_gate_passed=result["gate"]["passed"])
    except BaseException as exc:
        record.update(status="failed", eligible=False, failure_stage=record["stage"],
                      error=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        if (output / "gt-acquisition.json").is_file():
            record["gt_acquisition_sha256"] = sha(output / "gt-acquisition.json")
        record["wall_seconds"] = time.monotonic()-began
        save()
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "plan", "predictions", "prediction-supervision", "members-index",
                 "central-directory", "index-receipt", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--plan-sha256", required=True)
    result = run(parser.parse_args(argv))
    print(json.dumps({key:result[key] for key in ("status", "eligible", "wall_seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
