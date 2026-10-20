"""Saved-text audit only: no production imports, arrays, images, models or network."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

COHORT = [("4691_cut_guava", 86), ("817_cut_onion", 90),
          ("9675_fold_clothes", 201), ("1934_tear_film", 69)]
REAL = ["B", "B+M1", "B+M2", "B+M1+M2"]
METHODS = REAL + ["O_B", "O"]
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
SUPERVISOR = "8f7bcbb8b7d31031c709f72ea2f49da1913ced7230154e45e0d165d00457c9a8"
MONITOR = "4b8b0efa4fd713a032967c9b34539eac128184ef6db4e8535e83d785c6bbebcd"
TOL = 1e-12  # Aggregate comparison only; decision inequalities have no tolerance.


class Audit:
    def __init__(self):
        self.checks, self.errors, self.inputs, self.max_difference = 0, [], {}, 0.0
        self.costs = {}

    def check(self, ok, label):
        self.checks += 1
        if not ok:
            self.errors.append(label)

    def close(self, actual, expected, label):
        valid = type(actual) in (int, float) and math.isfinite(actual)
        difference = abs(actual - expected) if valid else float("inf")
        if math.isfinite(difference):
            self.max_difference = max(self.max_difference, difference)
        self.check(valid and difference <= TOL, label)

    def raw(self, path):
        p = Path(path).resolve()
        if p.suffix.lower() not in (".json", ".py", ".md", ".txt", ".log", ".jsonl"):
            raise ValueError("Non-text input forbidden: " + str(p))
        if not p.is_file() or p.stat().st_size > 64 * 1024**2:
            raise ValueError("Missing/oversized text input: " + str(p))
        raw = p.read_bytes()
        raw.decode("utf-8")
        self.inputs[str(p)] = hashlib.sha256(raw).hexdigest()
        return raw

    def load(self, path):
        return json.loads(self.raw(path), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

    def bound(self, desc):
        value = self.load(desc["path"])
        self.check(self.inputs[str(Path(desc["path"]).resolve())] == desc["sha256"],
                   "descriptor hash: " + desc["path"])
        return value

    def digest(self, path):
        self.raw(path)
        return self.inputs[str(Path(path).resolve())]

    def outer(self, desc, expected_output, name, spec_path=None):
        o = self.bound(desc)
        self.check(o["schema"] == "evr0029.exit-checked-operation.v1"
                   and o["supervisor_sha256"] == SUPERVISOR and o["monitor_sha256"] == MONITOR,
                   name + ": supervisor identity")
        self.check(o["status"] == "completed" and o["exit_code"] == 0 and o["termination"] is None
                   and not o["remaining_descendant_pids"] and not o["cleanup_errors"],
                   name + ": clean completion")
        self.check(o["thread_environment"] == {k: "4" for k in THREADS}
                   and 0 <= o["wall_seconds"] < o["hard_timeout_seconds"] <= 1800
                   and 0 <= o["peak_group_rss_kib"] <= o["group_rss_limit_kib"] == 8388608,
                   name + ": resources")
        self.check(o["output_sha256"] == expected_output, name + ": result hash")
        self.check(set(o["log_sha256"]) == {"stdout.log", "stderr.log"}, name + ": log set")
        for log, digest in o["log_sha256"].items():
            self.check(self.digest(Path(desc["path"]).parent / log) == digest, name + ": " + log)
        self.check(not o["post_exit_snapshot"]["rows"] and not o["cleanup_snapshots"][-1]["rows"],
                   name + ": fresh/final process samples")
        if spec_path:
            s = self.load(spec_path)
            self.check(self.digest(spec_path) == o["spec_sha256"], name + ": operation spec")
            self.check(s["commit"] == o["expected_commit"] and s["timeout"] == o["hard_timeout_seconds"],
                       name + ": commit/deadline")
            command = [str(v).replace(s["command"][0], "$PYTHON")
                       .replace(s["repo"], "$REPOSITORY")
                       .replace(str(Path(s["repo"]).parent), "$PRIVATE") for v in s["command"]]
            self.check(command == o["command"], name + ": exact command")
        self.costs[name] = {k: o[k] for k in ("wall_seconds", "peak_group_rss_kib", "exit_code")}
        return o


def mean(values):
    values = list(values)
    return float(sum((Fraction(v) for v in values), Fraction()) / len(values))


def decompose(methods):
    return {m: {"total_O_minus_B": methods["O"][m] - methods["B"][m],
                "native_cleanup_and_ownership_O_B_minus_B": methods["O_B"][m] - methods["B"][m],
                "additional_support_O_minus_O_B": methods["O"][m] - methods["O_B"][m]}
            for m in ("J", "Jtr")}


def aggregate(a, report, policy):
    a.check(report["void_policy"] == policy and report["aggregation"] == "equal_sequence_object_means",
            policy + ": policy/weighting")
    sequences = report["per_sequence"]
    a.check([(s["sequence"], s["frame_count"]) for s in sequences] == COHORT, policy + ": full cohort")
    objects, by_sequence, populations = {m: [] for m in METHODS}, {}, []
    for s, (name, n) in zip(sequences, COHORT):
        ids = s["object_ids"]
        a.check(ids == list(range(1, len(ids) + 1)) and ids, name + ": contiguous supplied IDs")
        a.check(set(s["methods"]) == set(METHODS), name + ": complete method set")
        positions = {"J": list(range(1, n - 1)), "Jtr": list(range(1 + 3 * (n - 2) // 4, n - 1))}
        a.check(s["frame_positions"] == positions, name + ": exact endpoint/tail populations")
        a.check(len(s["original_255_pixels"]) == n and all(type(v) is int and v >= 0
                for v in s["original_255_pixels"]), name + ": all-frame void counts")
        populations.append(dict(sequence=name, frames=n, object_ids=ids,
                                interior=len(positions["J"]), tail=len(positions["Jtr"])))
        current = {}
        for method in METHODS:
            saved = s["methods"][method]
            a.check([o["object_id"] for o in saved["per_object"]] == ids, name + method + ": all IDs")
            calculated = []
            for o in saved["per_object"]:
                raw = o["frame_J"]
                a.check(len(raw) == n, name + method + ": every raw frame")
                for frame, value in enumerate(raw):
                    a.check(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                            f"{policy}/{name}/{method}/{o['object_id']}/{frame}: finite raw J")
                item = {metric: mean(raw[i] for i in use) for metric, use in positions.items()}
                for metric, value in item.items():
                    a.close(o[metric], value, name + method + ": object " + metric)
                calculated.append(item)
            objects[method].extend(calculated)
            current[method] = {m: mean(o[m] for o in calculated) for m in positions}
            for m, value in current[method].items():
                a.close(saved[m], value, name + method + ": sequence " + m)
        for object_index in range(len(ids)):
            for frame in range(n):
                r = {m: s["methods"][m]["per_object"][object_index]["frame_J"][frame] for m in METHODS}
                a.check(r["B"] <= r["O_B"] <= r["O"] and all(r[m] <= r["O"] for m in REAL),
                        f"{policy}/{name}/{frame}: raw ceiling inequalities")
        dec = decompose(current)
        for m, values in dec.items():
            for k, value in values.items():
                a.close(s["oracle_decomposition"][m][k], value, name + ": " + m + k)
        by_sequence[name] = dict(methods=current, oracle_decomposition=dec)
    methods = {method: {m: mean(o[m] for o in objects[method]) for m in ("J", "Jtr")} for method in METHODS}
    for method, values in methods.items():
        for metric, value in values.items():
            a.close(report["methods"][method][metric], value, policy + ": aggregate " + method + metric)
    decomposition = decompose(methods)
    for m, values in decomposition.items():
        for k, value in values.items():
            a.close(report["oracle_decomposition"][m][k], value, policy + ": aggregate " + m + k)
    a.check(report["sequence_count"] == 4 and report["object_count"] == sum(len(x["object_ids"]) for x in populations),
            policy + ": aggregate object count")
    a.check(sum(p["frames"] for p in populations) == 446
            and sum(p["interior"] for p in populations) == 438
            and sum(p["tail"] for p in populations) == 110, policy + ": 446/438/110")
    return dict(methods=methods, oracle_decomposition=decomposition, per_sequence=by_sequence,
                populations=populations)


def decision(report):
    m, d = report["methods"], report["oracle_decomposition"]
    best = {k: max(m[name][k] for name in REAL) for k in ("J", "Jtr")}
    margins = {k: m["O"][k] - best[k] for k in best}
    clips = {k: r["oracle_decomposition"]["Jtr"]["additional_support_O_minus_O_B"]
             for k, r in report["per_sequence"].items()}
    clauses = {
        "total_ceiling_J_at_least_5pp": d["J"]["total_O_minus_B"] >= .05,
        "total_ceiling_Jtr_at_least_5pp": d["Jtr"]["total_O_minus_B"] >= .05,
        "extra_support_J_at_least_2pp": d["J"]["additional_support_O_minus_O_B"] >= .02,
        "extra_support_Jtr_at_least_3pp": d["Jtr"]["additional_support_O_minus_O_B"] >= .03,
        "ceiling_over_best_real_J_at_least_3pp": margins["J"] >= .03,
        "ceiling_over_best_real_Jtr_at_least_3pp": margins["Jtr"] >= .03,
        "extra_Jtr_at_least_1pp_in_three_clips": sum(v >= .01 for v in clips.values()) >= 3}
    return dict(numerical_clauses=clauses, numerical_passed=all(clauses.values()),
                strongest_real_by_metric=best, ceiling_over_best_real=margins,
                per_clip_extra_support_Jtr=clips)


def inspect(a, d):
    spec = a.load(d / "evaluate-spec-01.json")
    report, receipt, seal = (a.load(d / "evaluation-01" / n) for n in ("evaluation.json", "receipt.json", "seal.json"))
    closure, proof = a.load(d / "source-closure-02.json"), a.bound(spec["prediction_proof"])
    sources = closure["source_sha256"]
    a.check(len(sources) == 70 and all(v["source_sha256"] == sources for v in (spec, report, receipt, seal, proof)),
            "one unchanged 70-source scientific closure")
    for path, digest in sources.items():
        a.check(a.digest(path) == digest, "current scientific source: " + path)
    for v in (report, receipt, seal):
        a.check(v["spec_sha256"] == a.digest(d / "evaluate-spec-01.json"), "evaluation spec binding")
    a.check(seal["complete"] is True and seal["receipt_sha256"] == a.digest(d / "evaluation-01/receipt.json")
            and seal["outputs"] == receipt["outputs"], "evaluation seal/receipt")
    for item in receipt["outputs"]:
        p = d / "evaluation-01" / item["path"]
        a.check(a.digest(p) == item["sha256"] and p.stat().st_size == item["bytes"], "evaluation output")
    a.outer(dict(path=str(d / "evaluation-supervision-01/receipt.json"),
                 sha256=a.digest(d / "evaluation-supervision-01/receipt.json")),
            a.digest(d / "evaluation-01/receipt.json"), "evaluation", d / "evaluate-outer-spec-01.json")
    for key in ("completed_frames", "parity_frames", "label_decode_attempts", "label_decodes"):
        a.check(receipt[key] == 446, "evaluation " + key)
    a.check(receipt["completed_sequences"] == 4 and receipt["status"] == "completed"
            and receipt["sealed_predictions_admitted_before_gt"] is True
            and all(receipt[k] == 0 for k in ("model_calls", "image_encoder_calls", "native_calls"))
            and not receipt["offline_attempts"], "evaluation completion/admission/no models")
    values = {k: a.bound(v) for k, v in proof.items() if isinstance(v, dict) and "path" in v}
    for name, output in (("qualification", "qualification_receipt"), ("predict", "predict_receipt"),
                         ("initial", "initial_acquisition"), ("rgb", "rgb_acquisition"),
                         ("initial_inspection", "initial_inspection")):
        a.outer(proof[name + "_outer"], proof[output]["sha256"], name)
    pred, pred_seal = values["predict_receipt"], values["predict_seal"]
    a.check(pred["status"] == "completed" and pred["scientific_eligible"] is True
            and pred["completed_frames"] == 446 and pred["completed_sequences"] == 4
            and len(pred["initial_label_decodes"]) == 4 and pred["noninitial_label_decodes"] == 0,
            "complete prediction before later labels")
    a.check(pred["source_sha256"] == pred_seal["source_sha256"] == values["predict_spec"]["source_sha256"]
            == values["qualification_receipt"]["source_sha256"] == values["initial_inspection"]["source_sha256"]
            == sources, "prediction/qualification/inspection same 70 sources")
    a.check(pred_seal["complete"] is True and pred_seal["receipt_sha256"] == proof["predict_receipt"]["sha256"]
            and pred_seal["spec_sha256"] == pred["spec_sha256"] == proof["predict_spec"]["sha256"]
            and pred_seal["outputs"] == pred["outputs"] and len(pred["outputs"]) == 446,
            "prediction text seal bindings")
    a.check(pred["model_initial_hash"] == pred["model_final_hash"] and not pred["offline_attempts"],
            "recorded unchanged model/no network")
    a.check(all(v == 446 for v in pred["calls"].values()), "recorded native calls")
    a.check([(r["sequence"], r["frames"]) for r in pred["rows"]] == COHORT, "prediction exact cohort")
    a.check(pred["outputs"] == [p for r in pred["rows"] for p in r["predictions"]], "flat prediction inventory")
    for r, (name, n) in zip(pred["rows"], COHORT):
        a.check(r["status"] == "completed" and r["completed_frames"] == n
                and [f["frame"] for f in r["frame_records"]] == list(range(n))
                and [f["frame"] for f in r["predictions"]] == list(range(n)), name + ": complete frame metadata")
        for f, output in zip(r["frame_records"], r["predictions"]):
            a.check(f["selected_low_exact"] is True and f["selected_original_exact"] is True
                    and f["output"] == output and f["sequence"] == name,
                    name + ": recorded parity/descriptor " + str(f["frame"]))
            a.check(f["bank_count"] == (1 if f["frame"] == 0 else 3), name + ": complete native bank")
            for q, winner in zip(f["predicted_quality"], f["selected_slots"]):
                a.check(all(math.isfinite(x) for x in q) and winner == max(range(len(q)), key=lambda j: q[j]),
                        name + ": recorded quality selection")
    qualification = values["qualification_receipt"]
    a.check(qualification["eligible"] and qualification["full_state_parity"]
            and len(qualification["native_output_parities"]) == 7
            and all(x["exact"] for x in qualification["native_output_parities"]), "manufactured qualification parity")
    metric = a.bound(spec["metric_acquisition"])
    a.outer(spec["metric_outer"], spec["metric_acquisition"]["sha256"], "metric acquisition")
    expected_members = [r for r in values["manifest"]["members"] if r["phase"] == "metric"]
    a.check(metric["status"] == "completed" and metric["phase"] == "metric"
            and len(metric["members"]) == len(expected_members) == 442
            and all(all(saved[k] == v for k, v in expected.items())
                    for saved, expected in zip(metric["members"], expected_members)),
            "all 442 metric-member metadata; payload hashes deliberately not reread")
    gt = metric["gt_admission"]
    a.check(gt["eligible"] is True and gt["outputs_verified"] is True and gt["output_count"] == 446
            and gt["prediction_proof_sha256"] == spec["prediction_proof"]["sha256"]
            and gt["prediction_receipt_sha256"] == proof["predict_receipt"]["sha256"]
            and gt["prediction_seal_sha256"] == proof["predict_seal"]["sha256"], "recorded GT proof chain")
    adapter = metric["transport_adapter"]
    a.check(adapter["status"] == "completed" and adapter["no_retry"] and adapter["connection_closed"]
            and adapter["full_population_restart"] and adapter["reused_members"] == 0
            and not adapter["finalization_errors"] and len(metric["requests"]) == 1326,
            "complete no-retry replacement metric transport")
    for path, digest in metric["source_sha256"].items():
        a.check(a.digest(Path(spec["repo"]) / path) == digest, "current metric transport source")
    parent = {k: a.bound(v) for k, v in adapter["parent_failure"].items()}
    a.check(parent["receipt"]["status"] == parent["outer"]["status"] == "failed"
            and len(parent["receipt"]["members"]) == 45, "original metric failure preserved")
    outputs = {}
    for key, policy in (("primary", "author_background"), ("void_ignore_sensitivity", "void_ignore")):
        outputs[key] = aggregate(a, report[key], policy)
        outputs[key]["decision_recomputed"] = decision(outputs[key])
    new_decision = outputs["primary"]["decision_recomputed"]
    for key, value in new_decision.items():
        saved = report["decision"][key]
        if isinstance(value, dict) and key != "numerical_clauses":
            for k, number in value.items():
                a.close(saved[k], number, "decision " + key + k)
        else:
            a.check(saved == value, "decision " + key)
    a.check(receipt["numeric_gate_passed"] == new_decision["numerical_passed"], "inner decision consistency")
    a.check(report["official_benchmark_result"] is False and report["deployed_oracle"] is False
            and report["F_computed"] is False, "no official benchmark/deployable oracle/F claim")
    return outputs


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--directory", required=True)
    cli.add_argument("--output", required=True)
    args = cli.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(target)
    began, a, result = time.monotonic(), Audit(), {}
    try:
        a.check(all(os.environ.get(k) == "4" for k in THREADS), "four-thread environment")
        result = inspect(a, Path(args.directory))
        # Rehash only the exact text sources/records already admitted by raw().
        for path, digest in list(a.inputs.items()):
            a.check(a.digest(path) == digest, "text unchanged throughout audit: " + path)
    except BaseException as exc:
        a.errors.append("audit exception: " + type(exc).__name__ + ": " + str(exc))
        traceback.print_exc()
    report = dict(schema="evr0035.independent-saved-text-audit.v1", passed=not a.errors,
                  checks=a.checks, errors=a.errors, max_aggregate_absolute_difference=a.max_difference,
                  aggregate_tolerance=TOL, decision_tolerance=0, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  wall_seconds=time.monotonic() - began, input_text_sha256=a.inputs,
                  original_outer_costs=a.costs, recomputed=result,
                  scope="Independent saved scalar aggregation/provenance; no pixel J, model, mask parity or payload hash recomputation.",
                  method="Exact Fraction means of saved binary floats, then float; no metrics/evaluator import.",
                  limitations=["Four permanently exposed TRAIN clips; original-recording independence unresolved.",
                               "O/O_B use GT and are privileged pixel-support ceilings, not realizable whole-mask selectors.",
                               "Recorded native/resize parities and annotation decodes are checked as receipt assertions only.",
                               "No novel method, held-out generalization, official benchmark or final two-source JF gate established."])
    with target.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ("passed", "checks", "errors", "max_aggregate_absolute_difference", "wall_seconds")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
