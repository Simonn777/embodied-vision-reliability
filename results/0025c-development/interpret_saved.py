"""Descriptive aggregation only: no solver/model imports or matrix refitting."""
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


HERE = Path(__file__).resolve().parent
INPUT = HERE / "noise-01"
OUTPUT = HERE / "interpretation-01"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(values):
    return math.sqrt(sum(float(x) ** 2 for x in values))


def dist(values):
    a = sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not a:
        return {"count": 0}
    # Descriptive nearest-rank tail; explicitly distinct from scientific metrics.
    return dict(count=len(a), median=statistics.median(a),
                p90_nearest_rank=a[math.ceil(.9 * len(a)) - 1], minimum=a[0], maximum=a[-1])


def main():
    started = time.monotonic()
    OUTPUT.mkdir(exist_ok=False)
    seal = json.loads((INPUT / "seal.json").read_text())
    groups = collections.defaultdict(list)
    pairing = collections.defaultdict(dict)
    read_hashes = {}
    for entry in seal["records"]:
        name = entry["path"]
        if not name.startswith("case-"):
            continue
        if Path(name).name != name:
            raise ValueError("unsafe sealed case path")
        path = INPUT / name
        if digest(path) != entry["sha256"]:
            raise ValueError("case hash mismatch")
        read_hashes[name] = entry["sha256"]
        record = json.loads(path.read_text())
        inp = record["input"]
        true_t_norm = norm(inp["truth"]["t"])
        true_speed = max(norm(inp["truth"]["v1"]), norm(inp["truth"]["v2"]))
        for row in record["solves"]:
            solver = row["solver"]
            parameters = solver.get("parameters")
            chir = solver.get("cheirality") or {}
            task_null = solver.get("task_null") or {}
            nulls = {name: value.get("translation_direction_frobenius")
                     for name, value in task_null.items() if name in ("numerical", "weak")}
            speed = t_ratio = motion_ratio = None
            if parameters:
                speed = max(norm(parameters["v1"]), norm(parameters["v2"]))
                tn = norm(parameters["t"])
                t_ratio = tn / true_t_norm
                motion_ratio = .04 * speed / tn if tn else None
            compact = dict(record_index=row["record_index"], indices=row["indices"],
                           status=row["status"], reason=row["reason"], error=row["error_degrees"],
                           raw_angle=row["raw_angle_degrees"], rank=row["rank"], weak_rank=row["weak_rank"],
                           condition=solver.get("condition"), chirality=chir,
                           residual_norm=(solver.get("residuals") or {}).get("norm"),
                           max_speed=speed, t_norm_ratio=t_ratio, readout_motion_to_t=motion_ratio,
                           true_speed=true_speed, true_t_norm=true_t_norm,
                           true_readout_motion_to_t=.04 * true_speed / true_t_norm,
                           task_null=nulls,
                           clean_truth_task_null=solver.get("clean_truth_task_null_diagnostic"))
            key = (inp["case"], inp["regime"], row["arm"])
            groups[key].append(compact)
            if row["arm"] in ("D8", "N8", "A8"):
                pairing[(inp["case"], inp["regime"], inp["realization"])][row["arm"]] = row["error_degrees"]
    if len(read_hashes) != 132 or sum(len(v) for v in groups.values()) != 13332:
        raise ValueError("incomplete saved coverage")
    summaries = []
    for key, rows in sorted(groups.items()):
        valid = [r for r in rows if r["status"] == "valid"]
        useful = [r for r in valid if r["error"] <= 5.]
        finite_raw = [r for r in rows if r["raw_angle"] is not None]
        rejected_useful = [r for r in rows if r["status"] != "valid" and r["raw_angle"] is not None and r["raw_angle"] <= 5.]
        aff = [r for r in rows if r["chirality"].get("rule") == "positive_scale_and_all_pool_corrected_depths"]
        extra_pool_rejections = [r for r in aff if r["chirality"]["scale_positive"]
                                and r["chirality"]["fitting_positive_count"] == r["chirality"]["fitting_count"]
                                and r["chirality"]["positive_count"] < r["chirality"]["pool_size"]]
        clean_null = [r["clean_truth_task_null"] for r in rows if r["clean_truth_task_null"]]
        summary = dict(case=key[0], regime=key[1], arm=key[2], attempted=len(rows),
                       valid=len(valid), useful=len(useful), valid_but_error_over5=len(valid) - len(useful),
                       reasons=dict(collections.Counter(r["reason"] or "valid" for r in rows)),
                       rank_counts=dict(collections.Counter(str(r["rank"]) for r in rows)),
                       weak_rank_counts=dict(collections.Counter(str(r["weak_rank"]) for r in rows)),
                       errors_all=dist(r["error"] for r in rows), errors_valid=dist(r["error"] for r in valid),
                       raw_angle_useful_including_invalid=sum(r["raw_angle"] <= 5. for r in finite_raw),
                       invalid_but_raw_angle_useful=len(rejected_useful),
                       extra_pool_positivity_rejections=len(extra_pool_rejections),
                       scale_nonpositive=sum(not r["chirality"]["scale_positive"] for r in aff),
                       condition_all=dist(r["condition"] for r in rows),
                       residual_norm_valid=dist(r["residual_norm"] for r in valid),
                       speed_valid=dist(r["max_speed"] for r in valid),
                       translation_norm_ratio_valid=dist(r["t_norm_ratio"] for r in valid),
                       readout_motion_to_t_valid=dist(r["readout_motion_to_t"] for r in valid),
                       true_speed=rows[0]["true_speed"], true_t_norm=rows[0]["true_t_norm"],
                       true_readout_motion_to_t=rows[0]["true_readout_motion_to_t"],
                       valid_error_over90=sum(r["error"] > 90. for r in valid),
                       numerical_task_null=dist(r["task_null"].get("numerical") for r in rows),
                       weak_task_null=dist(r["task_null"].get("weak") for r in rows),
                       clean_truth_null_numerical=dist(x["numerical"]["frobenius"] for x in clean_null),
                       clean_truth_null_weak=dist(x["weak"]["frobenius"] for x in clean_null))
        summaries.append(summary)
    pairs = []
    for case in ("G", "N", "C", "E"):
        for regime in ("P", "S"):
            rows = [v for (c, r, _), v in pairing.items() if c == case and r == regime]
            assert len(rows) == 16
            for comparator in ("A8", "N8"):
                difference = [v["D8"] - v[comparator] for v in rows]
                pairs.append(dict(case=case, regime=regime, comparator=comparator,
                                  d8_better=sum(x < 0 for x in difference),
                                  tied=sum(x == 0 for x in difference),
                                  d8_worse=sum(x > 0 for x in difference),
                                  delta_degrees=dist(difference)))
    result = dict(schema="evr0025c.saved-interpretation.v1", scope="saved scalar/vector aggregation only; no fits or new data",
                  seal_sha256=digest(INPUT / "seal.json"), aggregate_sha256=digest(INPUT / "aggregate.json"),
                  case_sha256=read_hashes, groups=summaries, paired_d8_errors=pairs)
    path = OUTPUT / "summary.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    receipt = dict(status="completed", wall_seconds=time.monotonic() - started,
                   source_sha256=digest(Path(__file__)), output_sha256=digest(path),
                   consumed_case_files=132, consumed_solves=13332, production_imports=0,
                   fits=0, generated_perturbations=0, model_calls=0)
    (OUTPUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
