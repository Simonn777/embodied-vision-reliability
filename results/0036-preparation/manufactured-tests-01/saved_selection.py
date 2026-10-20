"""Privileged action-granularity diagnostic from saved scalar scores only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

INPUT_SHA = "7fe9592298d1cced864eaa6f75faf973e98c10ea3ce8c91100a678cc6cd60fe3"
COHORT = (("4691_cut_guava", 86), ("817_cut_onion", 90),
          ("9675_fold_clothes", 201), ("1934_tear_film", 69))
ARMS = ("B", "B+M1", "B+M2", "B+M1+M2")
ALL_ARMS = ARMS + ("O_B", "O")


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 32 * 1024**2,
            "bounded regular input required")
    raw = path.read_bytes()
    require(len(raw) <= 32 * 1024**2, "input grew beyond limit")
    return raw


def decode(raw):
    return json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def mean(values):
    return math.fsum(values) / len(values)


def policy_report(report, policy):
    require(report["void_policy"] == policy and report["aggregation"] == "equal_sequence_object_means"
            and report["sequence_count"] == report["object_count"] == 4, "policy/population mismatch")
    sequences = report["per_sequence"]
    require([(s["sequence"], s["frame_count"]) for s in sequences] == list(COHORT), "fixed complete clips")
    raw_rows, sequence_means, best_means, constant = [], [], [], {"J": [], "Jtr": []}
    for s, (name, n) in zip(sequences, COHORT):
        require(s["object_ids"] == [1] and set(s["methods"]) == set(ALL_ARMS), "one supplied ID/all six arms")
        positions = {"J": list(range(1, n - 1)), "Jtr": list(range(1 + 3 * (n - 2) // 4, n - 1))}
        require(s["frame_positions"] == positions, "fixed interior/tail positions")
        raw = {}
        for arm in ALL_ARMS:
            objects = s["methods"][arm]["per_object"]
            require(len(objects) == 1 and objects[0]["object_id"] == 1, "one complete object")
            raw[arm] = objects[0]["frame_J"]
            require(len(raw[arm]) == n and all(type(v) in (int, float) and math.isfinite(v)
                    and 0 <= v <= 1 for v in raw[arm]), "finite all-frame scores required")
        choices = [max(ARMS, key=lambda arm: raw[arm][i]) for i in range(n)]
        best = [raw[arm][i] for i, arm in enumerate(choices)]
        require(all(best[i] <= raw["O"][i] for i in range(n)), "whole-arm oracle exceeds pixel O")
        scores = {arm: {m: mean([raw[arm][i] for i in use]) for m, use in positions.items()}
                  for arm in ALL_ARMS}
        sequence_means.append(scores)
        best_means.append({m: mean([best[i] for i in use]) for m, use in positions.items()})
        constant_choices = {}
        for metric in positions:
            winner = max(ARMS, key=lambda arm: scores[arm][metric])
            constant_choices[metric] = winner
            constant[metric].append(dict(sequence=name, arm=winner, value=scores[winner][metric]))
        raw_rows.append(dict(sequence=name, object_id=1, frame_count=n, positions=positions,
                             frame_J=raw, best_arm_by_frame=choices, best_J_by_frame=best,
                             constant_arm_by_metric=constant_choices, fixed_scores=scores,
                             per_frame_oracle=best_means[-1]))
    fixed = {arm: {m: mean([s[arm][m] for s in sequence_means]) for m in ("J", "Jtr")} for arm in ALL_ARMS}
    global_best = {m: max(ARMS, key=lambda arm: fixed[arm][m]) for m in ("J", "Jtr")}
    oracle = {m: mean([s[m] for s in best_means]) for m in ("J", "Jtr")}
    gain = {m: oracle[m] - fixed[global_best[m]][m] for m in oracle}
    for row in raw_rows:
        row["per_frame_oracle_gain_over_strongest_global_fixed"] = {
            m: row["per_frame_oracle"][m] - fixed[global_best[m]][m] for m in oracle}
        row["per_frame_oracle_gap_to_pixel_O"] = {
            m: row["fixed_scores"]["O"][m] - row["per_frame_oracle"][m] for m in oracle}
        row["constant_gain_over_strongest_global_fixed"] = {
            m: row["fixed_scores"][row["constant_arm_by_metric"][m]][m] - fixed[global_best[m]][m] for m in oracle}
    constants = {m: dict(choices=constant[m], value=mean([r["value"] for r in constant[m]]))
                 for m in oracle}
    for m, value in constants.items():
        value.update(gain_over_strongest_global_fixed=value["value"] - fixed[global_best[m]][m],
                     gap_to_pixel_O=fixed["O"][m] - value["value"])
    clauses = {m + "_gain_at_least_3pp": gain[m] >= .03 for m in oracle}
    return dict(void_policy=policy, counts=dict(frames=446, interior=438, tail=110, objects=4),
                fixed_methods=fixed, strongest_global_fixed={m: dict(arm=global_best[m],
                value=fixed[global_best[m]][m]) for m in oracle}, per_frame_oracle=dict(values=oracle,
                gain_over_strongest_global_fixed=gain, gap_to_pixel_O={m: fixed["O"][m]-oracle[m] for m in oracle}),
                per_clip_constant_oracle=constants, raw_sequences=raw_rows,
                necessary_clauses=clauses, whole_arm_support_sufficient=all(clauses.values()))


def analyze(data):
    require(data["schema"] == "evr0035.metrics.v1", "saved evaluation schema")
    policies = {key: policy_report(data[key], policy) for key, policy in
                (("primary", "author_background"), ("void_ignore_sensitivity", "void_ignore"))}
    return dict(schema="evr0036.saved-selection.v1", **policies, tie_order=list(ARMS),
                primary_whole_arm_support_sufficient=policies["primary"]["whole_arm_support_sufficient"],
                interpretation="GT-privileged action-class diagnostic only; no deployable selector, novelty, learning or generalization.",
                original_D35_gate_unchanged=True, all_four_clips_exposed_development=True,
                recording_independence_established=False, final_paper_gate_met=False)


def run(spec_path):
    spec_raw = read(spec_path)
    spec = decode(spec_raw)
    require(spec["schema"] == "evr0036.saved-selection-spec.v1", "spec schema")
    output = Path(spec["output"]).absolute()
    require(not output.exists() and not output.is_symlink(), "fresh output required")
    output.mkdir(parents=True)
    began = time.monotonic()
    receipt = dict(schema="evr0036.saved-selection-receipt.v1", status="started",
                   spec_sha256=digest(spec_raw), input=spec["input"], source_sha256=spec["source_sha256"],
                   media_reads=0, model_calls=0, metric_pixel_recomputations=0)
    try:
        source = read(__file__)
        require(digest(source) == spec["source_sha256"], "executing source hash mismatch")
        require(spec["input"]["sha256"] == INPUT_SHA, "exact saved D35 result required")
        raw = read(spec["input"]["path"])
        require(digest(raw) == INPUT_SHA, "saved input hash mismatch")
        report = analyze(decode(raw))
        require(read(__file__) == source and read(spec_path) == spec_raw
                and read(spec["input"]["path"]) == raw, "source/spec/input changed")
        save(output / "report.json", report)
        receipt.update(status="completed", report_sha256=digest(read(output / "report.json")),
                       input_source_verified_before_after=True,
                       primary_whole_arm_support_sufficient=report["primary_whole_arm_support_sufficient"])
    except BaseException as exc:
        receipt.update(status="failed", error=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        receipt["wall_seconds"] = time.monotonic() - began
        save(output / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    print(json.dumps(run(parser.parse_args().spec)))
