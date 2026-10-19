"""Independent saved-only 0011 review. No production imports, model, or downloads.

Use --root STUDY --repo FROZEN_REPO --output NEW.json after scoring. --self-test
uses only artificial errors. PASS means agreement/provenance checks, not a
scientific GO. Source-commit declarations are checked; Git history is not replayed.
Full transport joints and actual backend calls are not independently replayed.
"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import time

START = time.monotonic()
import numpy as np

FREEZE_SHA = "f45cfe56f99cdb8c4cb2ad6376a3c0813693666e2798941b9627d535c7a61e46"
SOURCE_COMMIT = "a288f65bfb9d9e0d70dee72c0d83b020a0339458"
GROUPS = ("g-036", "g-002")
FRAMES = list(range(0, 241, 16))
VIEWS = ("original", "reference", "oblique", "axis_control")
METHODS = ("original_native", "reference_native", "oblique_native", "axis_control_native",
           "reference_independent", "coordinate_average", "entropy_select", "product_joint",
           "mixture_joint", "oblique_coupling", "axis_coupling")
CANDIDATE = "oblique_coupling"
PHASES = ("prepare", "g-036", "g-002", "derive", "score")
LIMITS = (300, 900, 900, 1800, 1800)
TAU = np.array([1., 2., 4., 8., 16.])


class Incomplete(Exception):
    pass


class Invalid(Exception):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Invalid("duplicate JSON key")
        result[key] = value
    return result


def finite_json(value):
    raise Invalid("nonfinite JSON constant")


class Audit:
    def __init__(self, root=None, repo=None):
        self.bases = {"root": Path(root), "repo": Path(repo)} if root is not None else {}
        self.checks = 0
        self.numeric_checks = 0
        self.max_abs = 0.
        self.max_scaled = 0.
        self.failures = []
        self.inputs = {}

    def check(self, passed, name):
        self.checks += 1
        if not bool(passed):
            self.failures.append(name)
        return bool(passed)

    def require(self, passed, name):
        if not self.check(passed, name):
            raise Invalid(name)

    def close(self, expected, actual, name):
        self.numeric_checks += 1
        if (isinstance(actual, bool) or not isinstance(actual, (int, float)) or
                not math.isfinite(actual) or not math.isfinite(expected)):
            return self.check(False, name + ": finite numeric scalar")
        delta = abs(float(expected) - float(actual))
        scaled = delta / max(1., abs(float(expected)))
        self.max_abs = max(self.max_abs, delta)
        self.max_scaled = max(self.max_scaled, scaled)
        return self.check(delta <= 1e-12 + 1e-10 * abs(float(expected)), name)

    def compare(self, expected, actual, name):
        if isinstance(expected, dict):
            self.require(type(actual) is dict and set(actual) == set(expected), name + ": keys")
            for key in expected:
                self.compare(expected[key], actual[key], name + "/" + key)
        elif isinstance(expected, list):
            self.require(type(actual) is list and len(expected) == len(actual), name + ": length")
            for i, (left, right) in enumerate(zip(expected, actual)):
                self.compare(left, right, name + "/" + str(i))
        elif isinstance(expected, float):
            self.close(expected, actual, name)
        else:
            self.check(type(expected) is type(actual) and expected == actual, name)

    def path(self, base, relative):
        p = PurePosixPath(relative)
        self.require(not p.is_absolute() and ".." not in p.parts and bool(p.parts), "safe input path")
        path = self.bases[base]
        for part in p.parts:
            path = path / part
            self.require(not path.is_symlink(), base + "/" + relative + ": no symlink")
        return path

    def blob(self, base, relative, expected=None):
        name = base + "/" + relative
        path = self.path(base, relative)
        if not path.is_file():
            raise Incomplete("missing " + name)
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            self.require(stat.S_ISREG(before.st_mode), name + ": regular")
            data = stream.read()
            after = os.fstat(stream.fileno())
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        self.require(all(getattr(before, key) == getattr(after, key) for key in fields), name + ": stable read")
        rec = {"bytes": len(data), "sha256": digest(data)}
        if name in self.inputs:
            self.require(self.inputs[name] == rec, name + ": unchanged since first read")
        self.inputs[name] = rec
        if expected is not None:
            self.require(rec == expected, name + ": binding")
        return data

    def json(self, base, relative, expected=None):
        return json.loads(self.blob(base, relative, expected), object_pairs_hook=no_duplicates,
                          parse_constant=finite_json)

    def npz(self, relative, expected, wanted=None):
        blob = self.blob("root", relative, expected)
        with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
            self.require(len(archive.files) == len(set(archive.files)), relative + ": unique NPZ keys")
            arrays = {key: archive[key].copy() for key in (wanted or archive.files)}
            return arrays, set(archive.files)


def compute_metrics(errors, occluded):
    """Independent vectorized arithmetic: arrays [group,time,point,method]."""
    if errors.shape != (2, 16, 8, 11) or occluded.shape != (2, 16, 8):
        raise Invalid("wrong metric dimensions")
    if not np.isfinite(errors).all() or np.any(errors < 0) or occluded.dtype != np.bool_:
        raise Invalid("invalid metric inputs")
    candidate = METHODS.index(CANDIDATE)
    controls = [i for i in range(11) if i != candidate]
    groups, exact, means = {}, [], []
    for g, group in enumerate(GROUPS):
        visible = ~occluded[g, 1:]
        observed = errors[g, 1:]
        selected = observed[visible]  # [eligible,method], no predicted-visibility access.
        if len(selected) == 0:
            raise Invalid("no eligible observations in " + group)
        hit = selected[..., None] < TAU
        counts = hit.sum(axis=0)
        rational = [Fraction(int(counts[m].sum()), 5 * len(selected)) for m in range(11)]
        exact.append(rational)
        means.append(selected.mean(axis=0))
        methods = {}
        for m, method in enumerate(METHODS):
            hidden = observed[~visible, m]
            methods[method] = {"quality": float(rational[m]), "hit_counts": counts[m].tolist(),
                               "eligible": len(selected), "coverage": 1.,
                               "mean_epe": float(selected[:, m].mean()),
                               "median_epe": float(np.median(selected[:, m])),
                               "all_point_epe": float(observed[..., m].mean()),
                               "occluded_epe": None if not len(hidden) else float(hidden.mean())}
        correct = selected < 4.
        pairs = {METHODS[m]: {"corrections_4px": int(np.count_nonzero(correct[:, candidate] & ~correct[:, m])),
                             "harms_4px": int(np.count_nonzero(~correct[:, candidate] & correct[:, m]))}
                 for m in controls}
        groups[group] = {"methods": methods, "paired": pairs}
    q = [(exact[0][m] + exact[1][m]) / 2 for m in range(11)]
    epe = (means[0] + means[1]) / 2
    best = controls[0]
    for m in controls[1:]:
        if q[m] > q[best]:  # Preserve first method on exact quality tie.
            best = m
    gain = q[candidate] - q[best]
    group_gains = [exact[g][candidate] - max(exact[g][m] for m in controls) for g in range(2)]
    gates = {"macro_gain_at_least_0.05": gain >= Fraction(1, 20),
             "each_group_above_every_control": all(value > 0 for value in group_gains),
             "macro_epe_not_worse_than_best_quality_control": bool(epe[candidate] <= epe[best])}
    return {"groups": groups, "macro": {m: {"quality": float(q[i]), "mean_epe": float(epe[i])}
                                        for i, m in enumerate(METHODS)},
            "strongest_control": METHODS[best], "macro_quality_gain": float(gain),
            "macro_quality_gain_exact": str(gain),
            "group_quality_gains_over_strongest": {g: float(group_gains[i]) for i, g in enumerate(GROUPS)},
            "quality_gates": gates}


def finite_real(value):
    return type(value) in (int, float) and math.isfinite(value)


def numerical_gates(diagnostics, predictions, audit):
    converged, means, axis = True, True, True
    for i, row in enumerate(diagnostics):
        audit.require(set(row["solvers"]) == {"oblique_coupling", "axis_coupling"}, "solver identities")
        for method in ("oblique_coupling", "axis_coupling"):
            solver = row["solvers"][method]
            values = [solver[k] for k in ("row_error", "column_error", "residual")]
            audit.require(all(finite_real(v) for v in values), "finite solver metrics")
            residual_matches = values[2] == max(values[:2])
            audit.check(residual_matches, "solver residual equals max: " + str(i) + "/" + method)
            converged &= (solver["converged"] is True and type(solver["iterations"]) is int and
                          1 <= solver["iterations"] <= 1000 and residual_matches and
                          all(0 <= v <= 1e-9 for v in values))
            value = row["global_means"][method]["reference_max_abs_difference"]
            audit.require(finite_real(value), "finite fixed-mean metric")
            means &= 0 <= value <= 5e-5
        value = row["axis_joint_max_abs_difference"]
        audit.require(finite_real(value), "finite axis metric")
        axis &= 0 <= value <= 1e-7
    a, b = METHODS.index("reference_native"), METHODS.index("reference_independent")
    parity = np.abs(predictions[..., a, :] - predictions[..., b, :]).max()
    return {"all_couplings_converged": bool(converged), "fixed_global_means": bool(means),
            "axis_joint_invariant": bool(axis), "reference_decoder_parity": bool(parity <= 1e-3)}


def verify_sources_and_phases(audit, freeze_sha):
    freeze_name = "protocols/0011-oblique-development-freeze.json"
    freeze = audit.json("repo", freeze_name)
    audit.require(audit.inputs["repo/" + freeze_name]["sha256"] == freeze_sha == FREEZE_SHA, "canonical freeze SHA")
    audit.require(freeze["source_commit"] == SOURCE_COMMIT and len(freeze["files"]) == 24, "24-source freeze identity")
    for name, expected in freeze["files"].items():
        audit.blob("repo", name)
        audit.require(audit.inputs["repo/" + name]["sha256"] == expected, "frozen source: " + name)
    config = audit.json("repo", "configs/0011-oblique-development.json")
    executions = {}
    for phase, limit in zip(PHASES, LIMITS):
        stem = "supervision/" + phase + "/"
        execution = audit.json("root", stem + "execution.json")
        binding = audit.json("root", stem + "source-binding.json")
        audit.require(binding == {"freeze_sha256": freeze_sha, "source_commit": SOURCE_COMMIT,
                                   "execution_sha256": audit.inputs["root/" + stem + "execution.json"]["sha256"]},
                      "phase source binding: " + phase)
        audit.require(execution["wall_limit_seconds"] == limit and execution["memory_limit_bytes"] == 8 * 1024**3
                      and execution["threads"] == 4 and len(execution["cpu_affinity"]) == 4, "phase limits: " + phase)
        for log in ("stdout.log", "stderr.log"):
            audit.blob("root", stem + log, execution["logs"][log])
        executions[phase] = {k: execution.get(k) for k in ("child_exit_code", "supervisor_exit_code", "timed_out",
                                                         "wall_seconds", "max_rss_kib")}
        if execution["child_exit_code"] != 0 or execution["supervisor_exit_code"] != 0 or execution["timed_out"]:
            raise Incomplete("phase did not complete: " + phase)
        audit.require(execution.get("launch_error") is None and execution.get("interrupted") is None,
                      "clean phase completion: " + phase)
        audit.require(finite_real(execution["wall_seconds"]) and execution["wall_seconds"] > 0, "positive phase wall")
        argv = execution["argv"]
        audit.require(argv[0] == config["python"] and Path(execution["cwd"]).resolve() == audit.bases["repo"].resolve(),
                      "phase interpreter and checkout")
        expected_script = ("prepare_oblique_development.py" if phase == "prepare" else
                           "run_oblique_predictions.py" if phase in GROUPS else "evaluate_oblique_saved.py")
        audit.require(Path(argv[1]).resolve() == audit.bases["repo"].resolve() / "scripts" / expected_script,
                      "phase executable identity")
        audit.require("--output" in argv, "phase output argument")
        out = Path(argv[argv.index("--output") + 1]).resolve()
        suffix = ("prepared" if phase == "prepare" else "predictions/" + phase if phase in GROUPS else
                  "derived" if phase == "derive" else "evaluation")
        audit.require(out == audit.bases["root"].resolve() / suffix, "phase output route")
        def option(flag):
            audit.require(argv.count(flag) == 1 and argv.index(flag) + 1 < len(argv), "unique argument: " + flag)
            return argv[argv.index(flag) + 1]
        if phase in ("derive", "score"):
            audit.require(argv[2] == phase, "derive versus score phase argument")
            expected_input = audit.bases["root"].resolve() / ("predictions" if phase == "derive" else "derived")
            audit.require(Path(option("--input")).resolve() == expected_input and
                          Path(option("--preparation")).resolve() == audit.bases["root"].resolve() / "prepared",
                          "decode/score input routes")
        elif phase in GROUPS:
            audit.require(option("--group-id") == phase and Path(option("--prepared-input")).resolve() ==
                          audit.bases["root"].resolve() / "prepared" / (phase + ".predictor.npz"),
                          "prediction group/input route")
            audit.require(option("--protocol-sha256") == freeze["files"]["protocols/0011-oblique-coupling-development.md"],
                          "prediction protocol argument")
        else:
            audit.require(Path(option("--manifest")).resolve() == audit.bases["repo"].resolve() /
                          "results/0002-preflight/identity/manifest.json", "preparation manifest route")
        env = execution["environment"]
        audit.require(all(env.get(k) == "4" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                                       "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")), "four-thread environment")
    return freeze, config, executions


def inspect_complete(audit, freeze_sha):
    freeze, config, executions = verify_sources_and_phases(audit, freeze_sha)
    prep = audit.json("root", "prepared/receipt.json")
    audit.require(prep["status"] == "completed_preparation", "completed preparation")
    names = {g + suffix for g in GROUPS for suffix in (".predictor.npz", ".ground_truth.npz")}
    audit.require(set(prep["files"]) == names, "exact four prepared outputs")
    audit.require(prep["archive"] == {"bytes": 187291581, "sha256": "ba0cec640deb65ab510bc3920cdaf8b7a2e756fce9fb339199c557ade436b0cc"},
                  "original archive declaration")
    identity = audit.json("repo", "results/0002-preflight/identity/manifest.json")
    audit.require(prep["original_manifest"] == audit.inputs["repo/results/0002-preflight/identity/manifest.json"],
                  "original manifest binding")
    audit.require(prep["new_network_bytes"] == 0, "no new acquisition declaration")
    audit.blob("root", "prepared/started.json")
    audit.require(prep["marker_sha256"] == audit.inputs["root/prepared/started.json"]["sha256"], "preparation marker")
    audit.require([(r["group_id"], r["archive_index"]) for r in prep["groups"]] == [("g-036", 36), ("g-002", 2)],
                  "reserve group allocation")
    gt_all, occ_all, point_lists, raw_bindings, raw_visibility = [], [], [], {}, {}
    for group, index in (("g-036", 36), ("g-002", 2)):
        meta = next(r for r in prep["groups"] if r["group_id"] == group)
        old = next(r for r in identity["identities"] if r["archive_index"] == index)
        audit.require(meta["input_arrays"] == old["arrays"], "original array declaration: " + group)
        predictor_name = group + ".predictor.npz"
        inp, keys = audit.npz("prepared/" + predictor_name, prep["files"][predictor_name],
                             wanted=("query_xy", "point_ids", "frame_ids"))
        audit.require(keys == {"video", "query_xy", "point_ids", "frame_ids"}, "predictor keys")
        for name, shape, dtype in (("query_xy", (8, 2), np.float32), ("point_ids", (8,), np.int64), ("frame_ids", (16,), np.int64)):
            audit.require(inp[name].shape == shape and inp[name].dtype == dtype, "predictor identity shape/dtype")
        ids = inp["point_ids"].tolist()
        audit.require(ids == sorted(set(ids)) and len(ids) == 8 and min(ids) >= 0 and max(ids) < 30, "eight ordered IDs")
        audit.require(inp["frame_ids"].tolist() == FRAMES and meta["point_ids"] == ids and meta["frame_ids"] == FRAMES,
                      "prepared identities")
        truth_name = group + ".ground_truth.npz"
        gt, keys = audit.npz("prepared/" + truth_name, prep["files"][truth_name])
        audit.require(keys == {"points", "occluded", "point_ids", "frame_ids"}, "GT keys")
        for name, shape, dtype in (("points", (16, 8, 2), np.float64), ("occluded", (16, 8), np.bool_),
                                   ("point_ids", (8,), np.int64), ("frame_ids", (16,), np.int64)):
            audit.require(gt[name].shape == shape and gt[name].dtype == dtype, "GT shape/dtype")
        audit.require(np.isfinite(gt["points"]).all(), "finite full GT; no clipping")
        audit.require(np.array_equal(gt["point_ids"], inp["point_ids"]) and np.array_equal(gt["frame_ids"], inp["frame_ids"]),
                      "predictor/GT identities")
        audit.require(np.array_equal(gt["points"][0].astype(np.float32), inp["query_xy"]) and not gt["occluded"][0].any(),
                      "initial query coordinates and visibility")
        gt_all.append(gt["points"]); occ_all.append(gt["occluded"]); point_lists.append(ids)
        folder = "predictions/" + group + "/"
        receipt = audit.json("root", folder + "receipt.json")
        audit.require(receipt["status"] == "completed_predictions" and receipt["group_id"] == group and
                      receipt["forwards_attempted"] == receipt["completed_predictions"] == len(receipt["frames"]) == 64,
                      "recorded 64 unique forwards: " + group)
        audit.require("active_call" not in receipt and receipt["point_ids"] == ids and receipt["frame_ids"] == FRAMES,
                      "finished group identities")
        audit.require(receipt["input_sha256"] == prep["files"][predictor_name]["sha256"] and
                      receipt["protocol_sha256"] == freeze["files"]["protocols/0011-oblique-coupling-development.md"] and
                      receipt["runner_sha256"] == freeze["files"]["scripts/run_oblique_predictions.py"] and
                      receipt["geometry_sha256"] == freeze["files"]["scripts/oblique_view_geometry.py"], "prediction source/data bindings")
        backend = receipt["backend"]
        for source_key, config_key in (("source_revision", "model_source_revision"),
                                       ("checkpoint_revision", "checkpoint_revision"),
                                       ("source_sha256", "source_sha256"), ("checkpoint_sha256", "checkpoint_sha256"),
                                       ("checkpoint_bytes", "checkpoint_bytes"), ("packages", "packages"),
                                       ("asset_receipt_sha256", "asset_receipt_sha256")):
            audit.require(backend[source_key] == config[config_key], "recorded backend identity: " + source_key)
        audit.require(backend["threads"] == [4, 4] and backend["autocast"] is False and backend["compile"] is False and
                      backend["seed"] == 20261018, "recorded frozen execution options")
        completed = audit.json("root", folder + "completed.json")
        audit.require(completed["receipt_sha256"] == audit.inputs["root/" + folder + "receipt.json"]["sha256"] and
                      completed["completed_predictions"] == 64, "completed group marker")
        audit.json("root", folder + "started.json")
        seen, files, bindings = set(), set(), {}
        for entry in receipt["frames"]:
            view, t = entry["view"], entry["observation_index"]
            audit.require(view in VIEWS and type(t) is int and 0 <= t < 16 and (view, t) not in seen, "raw view/time identity")
            seen.add((view, t))
            audit.require(entry["frame_index"] == FRAMES[t] and entry["state_step"] == t + 1, "raw causal observation step")
            name = entry["output_file"]
            audit.require(name == "view-" + view + "-frame-%03d.npz" % t and name not in files, "raw unique filename")
            files.add(name)
            data, keys = audit.npz(folder + name, None)
            rec = audit.inputs["root/" + folder + name]
            audit.require(rec["sha256"] == entry["output_sha256"], "raw file SHA")
            bindings[name] = rec
            audit.require(keys == {"coordinate_logits", "positions_xy", "visibility_logits"}, "raw NPZ keys")
            for key, shape in (("coordinate_logits", (8, 512)), ("positions_xy", (8, 2)), ("visibility_logits", (8,))):
                audit.require(data[key].shape == shape and data[key].dtype == np.float32 and np.isfinite(data[key]).all(),
                              "raw array contract: " + key)
            raw_visibility[group, t, view] = data["visibility_logits"]
            for key in ("forward_seconds", "frame_seconds"):
                audit.require(finite_real(entry[key]) and entry[key] > 0, "positive raw timing")
        audit.require(seen == {(v, t) for v in VIEWS for t in range(16)}, "complete recorded forward grid")
        observed = {p.name for p in audit.path("root", folder).iterdir()}
        audit.require(observed == files | {"receipt.json", "started.json", "completed.json"}, "no extra prediction files/calls recorded")
        raw_bindings[group] = {"receipt": audit.inputs["root/" + folder + "receipt.json"], "frames": bindings}
    seal = audit.json("root", "derived/seal.json")
    audit.require(seal["status"] == "sealed" and seal["rows"] == 256 and seal["model_calls"] == 0 and seal["GT_opened"] is False,
                  "complete saved decode declaration")
    expected_inputs = dict(raw_bindings, preparation_receipt=audit.inputs["root/prepared/receipt.json"])
    audit.require(seal["inputs"] == expected_inputs, "seal input chain")
    document = audit.json("root", "derived/predictions.json", seal["files"]["predictions.json"])
    diagnostics = audit.json("root", "derived/diagnostics.json", seal["files"]["diagnostics.json"])
    rows = document["rows"]
    audit.require(document["methods"] == list(METHODS) and len(rows) == len(diagnostics) == 256, "256 rows and eleven methods")
    predictions = np.empty((2, 16, 8, 11, 2), dtype=np.float64)
    expected_keys = [(g, t, p) for g in range(2) for t in range(16) for p in range(8)]
    for row, d, (g, t, p) in zip(rows, diagnostics, expected_keys):
        ident = {"group_id": GROUPS[g], "frame_id": FRAMES[t], "point_id": point_lists[g][p],
                 "observation_index": t, "point_index": p}
        for key, value in ident.items():
            audit.require(type(row[key]) is type(value) and row[key] == value and type(d[key]) is type(value) and d[key] == value,
                          "exact ordered row/diagnostic identity")
        audit.require(set(row["predictions"]) == set(METHODS) and set(row["visibility_logits"]) == set(VIEWS), "saved method/view coverage")
        for m, method in enumerate(METHODS):
            xy = np.asarray(row["predictions"][method])
            audit.require(xy.dtype.kind in "fiu" and xy.shape == (2,) and np.isfinite(xy).all(), "finite saved coordinate")
            predictions[g, t, p, m] = xy
        for view in VIEWS:
            audit.close(float(raw_visibility[GROUPS[g], t, view][p]), row["visibility_logits"][view], "retained visibility logit")
    ground_truth = np.stack(gt_all)
    occluded = np.stack(occ_all)
    delta = predictions - ground_truth[..., None, :]
    errors = np.sqrt(np.sum(delta * delta, axis=-1, dtype=np.float64))
    expected_cases = [{"group_id": GROUPS[g], "frame_id": FRAMES[t], "point_id": point_lists[g][p],
                       "occluded": bool(occluded[g, t, p]),
                       "errors": {m: float(errors[g, t, p, i]) for i, m in enumerate(METHODS)}} for g, t, p in expected_keys]
    result = compute_metrics(errors, occluded)
    numerical = numerical_gates(diagnostics, predictions, audit)
    result.update(numerical_gates=numerical, model_calls=0, status="completed_evaluation",
                  decision="GO_development_only" if all(numerical.values()) and all(result["quality_gates"].values()) else "NO_GO",
                  seal_sha256=audit.inputs["root/derived/seal.json"]["sha256"])
    evaluation_receipt = audit.json("root", "evaluation/receipt.json")
    audit.require(evaluation_receipt["status"] == "completed_evaluation" and evaluation_receipt["model_calls"] == 0 and
                  evaluation_receipt["seal"] == audit.inputs["root/derived/seal.json"], "evaluation receipt seal binding")
    saved_cases = audit.json("root", "evaluation/case-errors.json", evaluation_receipt["files"]["case-errors.json"])
    saved_result = audit.json("root", "evaluation/results.json", evaluation_receipt["files"]["results.json"])
    audit.require(finite_real(saved_result["wall_seconds"]) and saved_result["wall_seconds"] > 0, "saved scoring wall")
    del saved_result["wall_seconds"]
    audit.compare(expected_cases, saved_cases, "all coordinate errors")
    audit.compare(result, saved_result, "all metrics")
    return {"decision": result["decision"], "quality_gates": result["quality_gates"], "numerical_gates": numerical,
            "strongest_control": result["strongest_control"], "macro_quality_gain_exact": result["macro_quality_gain_exact"],
            "macro": result["macro"], "rows": 256, "methods": 11, "recorded_forwards": 128,
            "phase_executions": executions}


def self_test(audit):
    e = np.ones((2, 16, 8, 11), np.float64)
    occ = np.zeros((2, 16, 8), np.bool_)
    c = METHODS.index(CANDIDATE)
    for g in range(2):
        e[g, 1:].reshape(120, 11)[:30, c] = 0.
    r = compute_metrics(e, occ)
    audit.require(r["macro_quality_gain_exact"] == "1/20" and all(r["quality_gates"].values()), "selftest exact 5pp boundary")
    e[0, 1, 0, c] = 1.
    audit.require(not compute_metrics(e, occ)["quality_gates"]["macro_gain_at_least_0.05"], "selftest below boundary")
    e.fill(100.)
    e[0, ..., c] = 0.
    occ[0] = True; occ[0, 1, 0] = False
    audit.close(.5, compute_metrics(e, occ)["macro"][CANDIDATE]["quality"], "selftest group balance")
    occ.fill(False)
    e.fill(100.)
    for g in range(2):
        e[g, 1:, :, :] = np.tile(np.array([1., 2., 4., 8., 16.]), 24).reshape(15, 8, 1)
    r = compute_metrics(e, occ)
    audit.require(r["groups"][GROUPS[0]]["methods"][CANDIDATE]["hit_counts"] == [0, 24, 48, 72, 96], "selftest strict thresholds/query exclusion")
    e.fill(100.)
    e[..., c] = 0.; e[..., METHODS.index("axis_coupling")] = 0.
    r = compute_metrics(e, occ)
    audit.require(r["strongest_control"] == "axis_coupling" and not r["quality_gates"]["each_group_above_every_control"], "selftest negative control participates")
    e.fill(100.)
    e[..., 0] = 2.1; e[..., 1] = 3.9
    e[:, 1:8, :, c] = 0.; e[:, 8:, :, c] = 5.8
    r = compute_metrics(e, occ)
    audit.require(r["strongest_control"] == "original_native" and not r["quality_gates"]["macro_epe_not_worse_than_best_quality_control"], "selftest tie and EPE gate")
    e.fill(100.)
    e[:, 1, 0, c] = 3.9; e[:, 1, 0, 0] = 4.
    e[:, 1, 1, c] = 4.; e[:, 1, 1, 0] = 3.9
    audit.require(compute_metrics(e, occ)["groups"][GROUPS[1]]["paired"]["original_native"] ==
                  {"corrections_4px": 1, "harms_4px": 1}, "selftest paired strict 4px")
    occ[0] = True
    try:
        compute_metrics(e, occ)
    except Invalid:
        audit.check(True, "selftest empty group refuses")
    else:
        audit.check(False, "selftest empty group refuses")
    return {"selftest_only": True, "artificial_groups": 8, "science_executed": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path)
    p.add_argument("--repo", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--freeze-sha256", default=FREEZE_SHA)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if not args.self_test and (args.root is None or args.repo is None):
        p.error("--root and --repo are required outside self-test")
    with args.output.open("x") as output:  # Exclusive reservation; never overwrite a prior review.
        audit = Audit() if args.self_test else Audit(args.root, args.repo)
        report = {"schema": "0011-independent-saved-metrics-v1", "status": "FAIL",
                  "selftest_only": args.self_test, "source_sha256": digest(Path(__file__).read_bytes()),
                  "numpy_version": np.__version__, "started_utc": datetime.now(timezone.utc).isoformat(),
                  "scope": "saved arithmetic and file bindings; no model/encoder/joint recomputation; private data remain local",
                  "limits": ["declared recorded forwards, not independent backend observation",
                             "24 current frozen-file hashes and declared source commit; no Git-history replay",
                             "selected prepared identities checked; original 50-video decoding/selection not rerun",
                             "diagnostics checked from saved summaries; no full joint reconstruction"]}
        try:
            report["summary"] = self_test(audit) if args.self_test else inspect_complete(audit, args.freeze_sha256)
            report["status"] = "PASS" if not audit.failures else "FAIL"
        except Incomplete as exc:
            report.update(status="INCOMPLETE", reason=str(exc))
        except Exception as exc:
            message = str(exc)
            for root in audit.bases.values():
                message = message.replace(str(root), "<input-root>")
            report.update(status="FAIL", error_type=type(exc).__name__, reason=message)
        report.update(checks=audit.checks, numeric_checks=audit.numeric_checks,
                      max_absolute_difference=audit.max_abs, max_scaled_difference=audit.max_scaled,
                      failure_count=len(audit.failures), failures=audit.failures[:50], inputs=audit.inputs,
                      checker_wall_seconds=time.monotonic()-START,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "checks", "numeric_checks", "failure_count", "checker_wall_seconds")}))
    return 0 if report["status"] == "PASS" else 2 if report["status"] == "INCOMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
