"""Saved-output review only: no production arithmetic imports, model or media.

CLI --spec JSON: schema evr0037.independent-review-spec.v1, output (fresh),
self_sha256, expected_commit, source_sha256 (worker's80), supervisor_python;
descriptors {path,sha256,bytes(optional)}: worker_spec, worker_operation,
worker_receipt, worker_outer, report, d35_result, guard, supervisor, monitor.
Requires the qualified Python -B -S -W error and a separate60s supervised run.
Only the existing offline guard is imported. NumPy reads derived X/count/action
archives; no source RGB, annotations, native banks, checkpoints or Torch.
Scalar tolerance1e-12; integer/action comparisons and decision thresholds exact.
This validates sealed actions and fit metadata, not a replay of fitted networks.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

COMMIT = "be9ccb5f10101639dc29a86ae0bca24b15fd83f4"
COHORT = (("4691_cut_guava", 86), ("817_cut_onion", 90),
          ("9675_fold_clothes", 201), ("1934_tear_film", 69))
FIXED = ("B", "B+M1", "B+M2", "B+M1+M2")
LEARNED = ("ordinary_mlp", "no_ancestry_mlp", "negative_gate_mlp")
LOCAL = ("fixed_negative", "positive_required") + LEARNED
ARMS = FIXED + LOCAL
POLICIES = ("author_background", "void_ignore")
REPORT_KEYS = ("primary", "void_ignore_sensitivity")
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
SUPERVISOR = "8f7bcbb8b7d31031c709f72ea2f49da1913ced7230154e45e0d165d00457c9a8"
MONITOR = "4b8b0efa4fd713a032967c9b34539eac128184ef6db4e8535e83d785c6bbebcd"
D35_RESULT = "7fe9592298d1cced864eaa6f75faf973e98c10ea3ce8c91100a678cc6cd60fe3"
D35_SPEC = "325961b52bed72c8710fee107265b02545002f9b7591fdd4119095841611a68d"
FEATURE_NAMES = (
    "rgb_r_mean", "rgb_g_mean", "rgb_b_mean", "rgb_r_std", "rgb_g_std", "rgb_b_std",
    "logit_B_mean", "logit_M1_mean", "logit_M2_mean",
    "logit_B_std", "logit_M1_std", "logit_M2_std",
    "support_B", "support_BM1", "support_BM2", "support_union",
    "cell_x_center", "cell_y_center", "foreground_fraction", "background_fraction",
    "unknown_fraction", "valid_fraction", "fb_clipped_div10", "photo_div255",
    "backward_dx_div512", "backward_dy_div512", "frame_age_div200")
CONFIG = dict(dtype="float32", device="cpu", hidden=32, activation="relu",
              epochs=20, batch_size=8192, seed=0, optimizer="Adam",
              learning_rate=.01, betas=[.9, .999], eps=1e-8, weight_decay=1e-4,
              selection="terminal_epoch_only", weight_sum_roundoff_tolerance=16*2.0**-23,
              weight_renormalization=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    def finite_float(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("nonfinite JSON number")
        return value
    return json.loads(Path(path).read_text(),
        parse_float=finite_float,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class Review:
    def __init__(self):
        self.checks, self.errors, self.maximum_difference = 0, [], 0.0
        self.bindings = {}

    def check(self, condition, label, fatal=False):
        self.checks += 1
        if not bool(condition):
            self.errors.append(label)
            if fatal:
                raise ValueError(label)

    def number(self, actual, expected, label):
        valid = (type(actual) in (int, float) and math.isfinite(actual)
                 and math.isfinite(expected))
        difference = abs(actual - expected) if valid else None
        if difference is not None:
            self.maximum_difference = max(self.maximum_difference, difference)
        self.check(valid and difference <= 1e-12, label)

    def bound(self, desc):
        p = Path(desc["path"])
        self.check(p.is_absolute() and p.is_file() and not p.is_symlink(),
                   "regular bound file: " + str(p), True)
        digest = sha(p)
        self.check(digest == desc["sha256"] and
                   ("bytes" not in desc or p.stat().st_size == desc["bytes"]),
                   "file binding: " + str(p), True)
        self.bindings[str(p)] = digest
        return p

    def json(self, desc):
        return read_json(self.bound(desc))


def relative(root, name):
    p = Path(name)
    if p.is_absolute() or not p.parts or any(v in (".", "..") for v in p.parts):
        raise ValueError("unsafe artifact path")
    target = root / p
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("artifact escapes output root")
    return target


def review(spec, audit, result):
    worker = audit.json(spec["worker_spec"])
    operation = audit.json(spec["worker_operation"])
    receipt = audit.json(spec["worker_receipt"])
    outer = audit.json(spec["worker_outer"])
    report = audit.json(spec["report"])
    old = audit.json(spec["d35_result"])
    source_map = spec["source_sha256"]
    audit.check(spec["expected_commit"] == COMMIT == operation["commit"] == outer["expected_commit"],
                "exact frozen commit", True)
    audit.check(len(source_map) == 80 and source_map == worker["source_sha256"] ==
                receipt["source_sha256"], "one80-source closure", True)
    for p, digest in source_map.items():
        audit.bound({"path": p, "sha256": digest})
    for key, expected in (("supervisor", SUPERVISOR), ("monitor", MONITOR)):
        audit.check(spec[key]["sha256"] == expected, key + " identity", True)
        audit.bound(spec[key])
    audit.check(spec["d35_result"]["sha256"] == D35_RESULT ==
                worker["d35_result"]["sha256"], "D35 reference identity", True)
    audit.check(worker["d35_spec"]["sha256"] == D35_SPEC, "original D35 admission identity", True)
    audit.bound(worker["d35_spec"])
    old_spec = read_json(worker["d35_spec"]["path"])
    audit.check(all(source_map.get(p) == h for p, h in old_spec["source_sha256"].items()),
                "inherited computational sources unchanged", True)
    audit.check(receipt["status"] == "completed" and receipt["phase"] == "complete"
                and receipt["spec_sha256"] == spec["worker_spec"]["sha256"]
                and receipt["source_input_postcheck"] is True
                and not receipt["offline_attempts"], "completed worker barrier", True)
    root = Path(worker["output"])
    audit.check(Path(spec["worker_receipt"]["path"]) == root / "receipt.json"
                and Path(spec["report"]["path"]) == root / "report.json"
                and operation["result_file"] == str(root / "receipt.json")
                and operation["operation_output"] == str(Path(spec["worker_outer"]["path"]).parent),
                "receipt/report/outer destinations", True)
    audit.check(outer["schema"] == "evr0029.exit-checked-operation.v1"
                and outer["status"] == "completed" and outer["exit_code"] == 0
                and outer["termination"] is None and not outer["remaining_descendant_pids"]
                and not outer["cleanup_errors"] and not outer["signal_attempts"]
                and outer["post_exit_snapshot"]["rows"] == []
                and outer["cleanup_snapshots"][-1]["rows"] == []
                and outer["output_sha256"] == spec["worker_receipt"]["sha256"]
                and outer["spec_sha256"] == spec["worker_operation"]["sha256"],
                "clean fresh-exit outer", True)
    audit.check(outer["supervisor_sha256"] == SUPERVISOR and outer["monitor_sha256"] == MONITOR
                and outer["thread_environment"] == {k: "4" for k in THREADS}
                and 0 <= outer["wall_seconds"] < outer["hard_timeout_seconds"] <= 1800
                and 0 <= outer["peak_group_rss_kib"] <= outer["group_rss_limit_kib"] == 8*1024*1024,
                "outer resource/source envelope", True)
    repo = Path(operation["repo"])
    shown = [v.replace(spec["supervisor_python"], "$PYTHON").replace(str(repo), "$REPOSITORY")
             .replace(str(repo.parent), "$PRIVATE") for v in operation["command"]]
    audit.check(outer["command"] == shown and operation["command"][1:5] == ["-B", "-S", "-W", "error"]
                and operation["command"][-2:] == ["--spec", spec["worker_spec"]["path"]]
                and source_map.get(operation["command"][5]) is not None,
                "exact source/spec command", True)
    audit.check(set(outer["log_sha256"]) == {"stdout.log", "stderr.log"}, "exact outer logs", True)
    for name, digest in outer["log_sha256"].items():
        audit.bound({"path": str(Path(spec["worker_outer"]["path"]).parent/name), "sha256": digest})
    for key in ("completed_feature_frames", "completed_target_frames",
                "completed_prediction_frames", "completed_metric_frames"):
        audit.check(receipt[key] == 446, key)
    audit.check(receipt["completed_fits"] == 12 and receipt["flow_calls"] == 884
                and receipt["model_calls"] == 0 and receipt["initial_label_decodes"] == 8
                and receipt["noninitial_label_decodes"] == 442, "complete operation counters")
    audit.check(receipt["feature_names"] == list(FEATURE_NAMES) and
                receipt["tracking_columns"] == list(range(18, 27)), "exact causal feature coordinates", True)
    audit.check(report["schema"] == "evr0037.local-comparison.v1" and
                report["all_four_clips_exposed_development"] is True and
                report["recording_independence_established"] is False and
                report["novelty_established"] is False and
                report["final_paper_gate_met"] is False, "development-only claim scope")

    inventory = {}
    for desc in receipt["outputs"]:
        name = desc["path"]
        audit.check(name not in inventory, "unique artifact: " + name, True)
        audit.bound(dict(desc, path=str(relative(root, name))))
        inventory[name] = desc
    audit.check(len(inventory) == 1366 and sum(d["bytes"] for d in inventory.values()) ==
                receipt["bytes_written"] <= 3*1024**3, "complete artifact inventory/bytes", True)
    audit.check(inventory["report.json"]["sha256"] == spec["report"]["sha256"], "report inventory binding", True)
    rows = read_json(root/"frame-inventory.json")
    audit.check([(r["sequence"], r["frame"]) for r in rows] ==
                [(name, i) for name, n in COHORT for i in range(n)], "ordered446 population", True)
    for r in rows:
        for key in ("feature", "targets", "prediction"):
            audit.check(inventory.get(r[key]["path"]) == r[key], "row inventory binding " + key, True)
    fs, ps = read_json(root/"feature-seal.json"), read_json(root/"prediction-seal.json")
    audit.check(fs["complete"] is True and ps["complete"] is True and fs["frames"] == ps["frames"] == 446
                and fs["source_sha256"] == ps["source_sha256"] == source_map
                and fs["noninitial_label_decodes"] == 0 and ps["own_fold_targets_passed_to_predict"] is False
                and fs["features"] == [r["feature"] for r in rows]
                and ps["predictions"] == [r["prediction"] for r in rows]
                and ps["arms"] == list(LOCAL)
                and 0 <= fs["monotonic_seconds"] < ps["monotonic_seconds"] < receipt["wall_seconds"],
                "complete feature-before-label and prediction seals", True)

    # Only provenance/runtime guard reuse; no production metric or model imports.
    guard_path = audit.bound(spec["guard"])
    audit.check(source_map.get(str(guard_path)) == spec["guard"]["sha256"], "bound offline guard", True)
    loader = importlib.util.spec_from_file_location("saved_review_guard", guard_path)
    guard = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(guard)
    guard.install_guard(result["offline_attempts"], result["compatibility_probes"])
    audit.check(sys.flags.no_site == 1 and sys.version_info[:3] == (3, 12, 13)
                and str(Path(sys.executable).absolute()) == operation["command"][0]
                and all(os.environ.get(k) == "4" for k in THREADS), "qualified isolated reviewer", True)
    import site
    site.main()
    import numpy as np
    audit.check(np.__version__ == "2.2.6" and Path(np.__file__).resolve().is_relative_to(
        Path(operation["command"][0]).parent.parent), "qualified NumPy only", True)

    def npz(desc, keys):
        with np.load(root/desc["path"], allow_pickle=False) as f:
            audit.check(set(f.files) == set(keys), "NPZ exact keys: " + desc["path"], True)
            return {k: f[k] for k in keys}

    counts_keys = ("total_gt", "union_tp", "union_fp", "native_tp", "native_fp",
                   "extra_tp", "extra_fp", "fixed_tp", "fixed_fp")
    scores = {p: {s: {a: [] for a in ARMS} for s, _ in COHORT} for p in POLICIES}
    samples = {s: {"n": 0, "active": 0, "frames": 0} for s, _ in COHORT}
    removed = report["removals"]
    audit.check(len(removed) == 892, "all446x2 removal rows", True)
    for ri, row in enumerate(rows):
        name, frame = row["sequence"], row["frame"]
        feat = npz(row["feature"], ("X", "union_counts"))
        x, area = feat["X"], feat["union_counts"]
        n = len(x)
        audit.check(x.shape == (n, 27) and x.dtype == np.float32 and np.isfinite(x).all()
                    and area.shape == (n,) and area.dtype == np.int64 and np.all(area >= 0),
                    f"{name}/{frame}: feature schema", True)
        small = row["small_shape"]
        audit.check(n == ((small[0]+3)//4)*((small[1]+3)//4), "cell count/partial edges")
        audit.check(np.all((x[:, 18:22] >= 0) & (x[:, 18:22] <= 1))
                    and np.allclose(x[:, 18:21].sum(axis=1), 1, rtol=0, atol=2e-7),
                    "identity fraction ranges/partition")
        audit.check(row["metadata"]["feature_names"] == receipt["feature_names"]
                    and row["metadata"]["union_pixels"] == int(area.sum()), "feature metadata")
        if frame:
            live = np.flatnonzero(area)
            k = min(512, len(live))
            selected = live[np.linspace(0, len(live)-1, k).astype(np.int64)] if k else live
            samples[name]["n"] += k
            samples[name]["active"] += int((x[selected, 19] >= .75).sum())
            samples[name]["frames"] += int(k > 0)
        keep = npz(row["prediction"], ("keep",))["keep"]
        audit.check(keep.dtype == np.bool_ and keep.shape == (5, 2, n), "complete boolean actions", True)
        if frame == 0:
            audit.check(keep.all(), "initial native-bank preservation")
        else:
            audit.check(np.array_equal(keep[0, 0], x[:, 19] < .75) and
                        np.array_equal(keep[0, 1], x[:, 19] < .75), "fixed negative action")
            audit.check(keep[1, 0].all() and np.array_equal(keep[1, 1], x[:, 18] >= .75),
                        "positive-required action")
            audit.check(all(np.array_equal(keep[i, 0], keep[i, 1]) for i in (2, 3, 4))
                        and keep[4, :, x[:, 19] < .75].all(), "learned action sharing/constrained unknown retention")
        packed = npz(row["targets"], [p+"__"+k for p in POLICIES for k in counts_keys])
        for pi, policy in enumerate(POLICIES):
            c = {k: packed[policy+"__"+k] for k in counts_keys}
            for key, value in c.items():
                expected_shape = () if key == "total_gt" else ((4,) if key.startswith("fixed") else (n,))
                audit.check(value.shape == expected_shape and value.dtype.kind in "iu" and np.all(value >= 0),
                            "integer nonnegative counts: " + key, True)
            gt = int(c["total_gt"])
            audit.check(all(np.array_equal(c["native_"+k]+c["extra_"+k], c["union_"+k])
                            for k in ("tp", "fp")), "native/extra partition")
            audit.check(int(c["union_tp"].sum()) <= gt and
                        (np.array_equal(c["union_tp"]+c["union_fp"], area) if pi == 0 else
                         np.all(c["union_tp"]+c["union_fp"] <= area)), "union coverage/GT bounds")
            audit.check(all(int(c["native_"+k].sum()) == int(c["fixed_"+k][0]) and
                            int(c["union_"+k].sum()) == int(c["fixed_"+k][3]) for k in ("tp", "fp")),
                        "fixed/native/union count agreement")
            if pi:
                audit.check(gt == int(packed["author_background__total_gt"]) and
                            all(np.array_equal(c[k], packed["author_background__"+k])
                                for k in ("union_tp", "native_tp", "extra_tp", "fixed_tp")) and
                            all(np.all(c[k] <= packed["author_background__"+k])
                                for k in ("union_fp", "native_fp", "extra_fp", "fixed_fp")),
                            "void affects only false-positive counting")
            oldseq = next(s for s in old[REPORT_KEYS[pi]]["per_sequence"] if s["sequence"] == name)
            for ai, arm in enumerate(FIXED):
                tp, fp = int(c["fixed_tp"][ai]), int(c["fixed_fp"][ai])
                value = tp/(gt+fp) if gt+fp else 1.0
                scores[policy][name][arm].append(value)
                audit.check(value == oldseq["methods"][arm]["per_object"][0]["frame_J"][frame],
                            "exact original D35 fixed parity")
            rr = removed[ri*2+pi]
            audit.check((rr["sequence"], rr["frame"], rr["policy"]) == (name, frame, policy)
                        and set(rr["arms"]) == set(LOCAL), "removal row identity", True)
            for ai, arm in enumerate(LOCAL):
                totals = {k: sum(int(c[part+"_"+k][keep[ai, j]].sum())
                                for j, part in enumerate(("native", "extra"))) for k in ("tp", "fp")}
                tp, fp = totals["tp"], totals["fp"]
                value = tp/(gt+fp) if gt+fp else 1.0
                scores[policy][name][arm].append(value)
                expected = dict(J=value, tp=tp, fp=fp)
                for j, part in enumerate(("native", "extra")):
                    for k in ("tp", "fp"):
                        expected["removed_"+part+"_"+k] = int(c[part+"_"+k][~keep[ai, j]].sum())
                audit.check(rr["arms"][arm] == expected, "exact local J/removal counts")

    result["computed"] = {}
    for pi, policy in enumerate(POLICIES):
        reported = report[REPORT_KEYS[pi]]
        audit.check(reported["sequence_count"] == reported["object_count"] == 4 and
                    reported["void_policy"] == policy and set(reported["methods"]) == set(ARMS),
                    "complete four-object nine-arm aggregate")
        audit.check([(s["sequence"], s["frame_count"]) for s in reported["per_sequence"]] ==
                    list(COHORT), "exact distinct ordered clip summaries", True)
        clips = {}
        for name, count in COHORT:
            s = next(r for r in reported["per_sequence"] if r["sequence"] == name)
            ids = list(range(1, count-1))
            intervals = {"J": ids, "Jtr": ids[3*len(ids)//4:]}
            audit.check(s["frame_count"] == count and s["object_ids"] == [1]
                        and s["frame_positions"] == intervals and set(s["methods"]) == set(ARMS),
                        "complete raw/interior/tail positions", True)
            clips[name] = {}
            for arm in ARMS:
                values = scores[policy][name][arm]
                saved = s["methods"][arm]
                audit.check(len(values) == count and len(saved["per_object"]) == 1 and
                            saved["per_object"][0]["object_id"] == 1 and
                            saved["per_object"][0]["frame_J"] == values, "exact raw scores")
                clips[name][arm] = {}
                for metric, positions in intervals.items():
                    average = math.fsum(values[i] for i in positions)/len(positions)
                    clips[name][arm][metric] = average
                    audit.number(saved[metric], average, "clip mean")
                    audit.number(saved["per_object"][0][metric], average, "object mean")
        pooled = {a: {m: math.fsum(clips[s][a][m] for s, _ in COHORT)/4
                      for m in ("J", "Jtr")} for a in ARMS}
        for a in ARMS:
            for m in ("J", "Jtr"):
                audit.number(reported["methods"][a][m], pooled[a][m], "equal-clip mean")
        cand = "negative_gate_mlp"
        fixed_gain = {m: pooled[cand][m]-max(pooled[a][m] for a in FIXED) for m in ("J", "Jtr")}
        control_gain = {a: {m: pooled[cand][m]-pooled[a][m] for m in ("J", "Jtr")}
                        for a in ("fixed_negative", "ordinary_mlp", "no_ancestry_mlp")}
        per_clip = {s: {a: {m: clips[s][cand][m]-clips[s][a][m] for m in ("J", "Jtr")}
                       for a in ("B", FIXED[-1])} for s, _ in COHORT}
        clauses = {"over_fixed_3pp_"+m: fixed_gain[m] >= .03 for m in ("J", "Jtr")}
        clauses.update({"over_"+a+"_0.5pp_"+m: control_gain[a][m] >= .005
                        for a in control_gain for m in ("J", "Jtr")})
        clauses["three_clips_positive_Jtr_over_union"] = sum(per_clip[s][FIXED[-1]]["Jtr"] > 0 for s, _ in COHORT) >= 3
        clauses["no_clip_native_loss_over_0_5pp"] = all(per_clip[s]["B"][m] >= -.005 for s, _ in COHORT for m in ("J", "Jtr"))
        decision = report["continuation" if pi == 0 else "sensitivity_continuation"]
        audit.check(decision["clauses"] == clauses and decision["numerical_passed"] == all(clauses.values())
                    and decision["gains_over_strongest_fixed"] == fixed_gain
                    and decision["gains_over_controls"] == control_gain
                    and decision["per_clip_gains"] == per_clip, "unchanged exact continuation clauses")
        result["computed"][policy] = dict(methods=pooled, per_clip=clips, clauses=clauses,
                                          numerical_passed=all(clauses.values()))
    audit.check(sum(n-2 for _, n in COHORT) == 438 and
                sum((n-2)-(3*(n-2)//4) for _, n in COHORT) == 110, "438/110 populations")
    audit.check(receipt["numerical_passed"] == result["computed"]["author_background"]["numerical_passed"],
                "receipt/independent decision")

    # Check recorded fit state only: no optimizer or network evaluation.
    for held, _ in COHORT:
        source_names = [s for s, _ in COHORT if s != held]
        expected_n = sum(samples[s]["n"] for s in source_names)
        common_initial = common_orders = None
        for arm in LEARNED:
            model = read_json(root/"models"/f"{held}-{arm}-model.json")
            trace = read_json(root/"models"/f"{held}-{arm}-trace.json")
            for item in (model, trace):
                audit.check(item["excluded_clip"] == held and item["training_clips"] == source_names
                            and item["arm"] == arm and item["config"] == CONFIG,
                            "exact excluded fold/config")
            expected_active = sum(samples[s]["active"] for s in source_names) if arm == LEARNED[-1] else expected_n
            audit.check(trace["status"] == "completed" and trace["n"] == expected_n and
                        trace["active_count"] == expected_active and trace["input_dim"] == model["input_dim"] == 27
                        and model["feature_names"] == receipt["feature_names"]
                        and model["selected_frame_counts"] == {s: samples[s]["frames"] for s in source_names},
                        "complete fit sample provenance")
            expected_weight = math.fsum(samples[s]["frames"]/(dict(COHORT)[s]-1)
                                        for s in source_names)/3
            audit.number(trace["input_weight_sum"], expected_weight, "unrenormalized fold weight sum")
            audit.check(trace["positive_effective_weight_count"] == expected_active and
                        0 <= trace["input_weight_sum"] <= 1 + CONFIG["weight_sum_roundoff_tolerance"]
                        and trace["weight_sum_roundoff_admitted"] == (trace["input_weight_sum"] > 1)
                        and trace["input_weight_excess_over_one"] == max(0, trace["input_weight_sum"]-1),
                        "recorded weight admission without renormalization")
            if model["kind"] == "constant_keep":
                audit.check(expected_active == 0 and model["terminal_epoch"] == 0
                            and model["state_dict"] is None and model["constant_probability"] == 1
                            and trace["actual_fit"] is False and trace["completed_epochs"] == 0
                            and trace["steps"] == 0 and trace["epochs"] == []
                            and trace["effective_float32_weight_sum"] == 0, "legal empty-fit branch")
            else:
                epoch_steps = (expected_n+8191)//8192
                audit.check(model["kind"] == "mlp" and model["terminal_epoch"] == 20
                            and trace["completed_epochs"] == 20 and trace["actual_fit"] is True
                            and len(trace["epochs"]) == 20
                            and trace["steps"] == trace["batch_attempts"] == 20*epoch_steps,
                            "terminal20-epoch fit")
                for i, e in enumerate(trace["epochs"], 1):
                    audit.check(e["epoch"] == i and e["steps"] == epoch_steps and e["rows"] == expected_n
                                and len(e["order_sha256"]) == 64, "complete epoch/order metadata")
                initial = trace["initial_state_dict"]
                orders = [e["order_sha256"] for e in trace["epochs"]]
                if common_initial is None:
                    common_initial, common_orders = initial, orders
                audit.check(initial == common_initial and orders == common_orders, "same initialization/orders")
                shapes = {"0.weight": (32, 27), "0.bias": (32,), "2.weight": (1, 32), "2.bias": (1,)}
                for state in (initial, model["state_dict"]):
                    audit.check(set(state) == set(shapes), "exact serialized layer keys", True)
                    for key, shape in shapes.items():
                        array = np.asarray(state[key])
                        audit.check(array.shape == shape and np.isfinite(array).all(), "finite state shape")
                audit.check(math.isfinite(trace["terminal_weighted_bce"]), "finite terminal training diagnostic")
    expected_artifacts = {"feature-seal.json", "prediction-seal.json", "report.json", "frame-inventory.json"}
    expected_artifacts.update(r[key]["path"] for r in rows for key in ("feature", "targets", "prediction"))
    expected_artifacts.update(f"models/{s}-{a}-{kind}.json" for s, _ in COHORT for a in LEARNED for kind in ("model", "trace"))
    audit.check(set(inventory) == expected_artifacts, "exact complete artifact inventory")
    audit.check(not result["offline_attempts"] and "torch" not in sys.modules, "no network/child/Torch execution")
    result.update(raw_frames=446, interior_frames=438, tail_frames=110, policies=2,
                  arms=list(ARMS), fits=12, artifact_inventory=receipt["outputs"],
                  scope="Saved counts/actions and fit provenance only; no flow, MLP, fit, pixel GT, native-model or media replay.")


def run(path):
    spec = read_json(path)
    out = Path(spec["output"])
    if out.exists():
        raise ValueError("fresh review output required; never overwrite failures")
    out.mkdir(parents=True)
    began, audit = time.monotonic(), Review()
    result = dict(schema="evr0037.independent-review.v1", status="started",
                  offline_attempts=[], compatibility_probes=[])
    try:
        audit.check(spec["schema"] == "evr0037.independent-review-spec.v1", "review spec schema", True)
        audit.bound({"path": str(Path(__file__).resolve()), "sha256": spec["self_sha256"]})
        audit.bound({"path": str(Path(path).resolve()), "sha256": sha(path)})
        review(spec, audit, result)
    except BaseException as exc:
        audit.errors.append("exception: " + type(exc).__name__ + ": " + str(exc))
    finally:
        for p, digest in audit.bindings.items():
            try:
                audit.check(sha(p) == digest, "unchanged after review: " + p)
            except Exception as exc:
                audit.errors.append("postcheck: " + type(exc).__name__ + ": " + str(exc))
        result.update(status="completed" if not audit.errors else "failed",
                      passed=not audit.errors, checks=audit.checks, errors=audit.errors,
                      maximum_scalar_difference=audit.maximum_difference,
                      scalar_absolute_tolerance=1e-12, wall_seconds=time.monotonic()-began,
                      source_and_input_hashes=audit.bindings)
        save(out/"review.json", result)
        save(out/"receipt.json", dict(schema="evr0037.independent-review-receipt.v1",
             status=result["status"], passed=result["passed"], checks=audit.checks,
             errors=audit.errors, wall_seconds=result["wall_seconds"],
             source_sha256=spec["self_sha256"], spec_sha256=sha(path),
             review_sha256=sha(out/"review.json")))
    print(json.dumps({k: result[k] for k in ("status", "passed", "checks", "wall_seconds")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    raise SystemExit(run(parser.parse_args().spec))
