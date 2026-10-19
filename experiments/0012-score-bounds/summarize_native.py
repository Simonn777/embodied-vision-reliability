"""All-pair/all-mode native result summary, without dataset access."""
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
directory = root / "results/0012c-native"
records = [json.loads(line) for line in (directory / "diagnostic.jsonl").read_text().splitlines()]
pairs = [r for r in records if r["type"] == "pair"]
assert len(pairs) == 16 and records[-1]["type"] == "complete"
rows = []
for i, r in enumerate(pairs, 1):
    assert r["status"] == "completed" and len(r["timed_calls"]) == 15
    for mode in range(5):
        calls = [c for c in r["timed_calls"] if c["mode"] == mode]
        assert len(calls) == 3 and all(c["winner_agrees"] and c["score_agrees"] for c in calls)
        for field in ("residuals", "nodes", "refinements", "pruned"):
            assert len({c[field] for c in calls}) == 1
        rows.append(dict(pair=i, key=r["key"], mode=mode, name=calls[0]["name"],
            **r["medians"][str(mode)],
            **{k: calls[0][k] for k in ("residuals", "nodes", "refinements", "pruned")}))
with (directory / "all-pairs-and-modes.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
totals = {str(mode): {key: sum(r[key] for r in rows if r["mode"] == mode)
    for key in rows[0] if key not in ("pair", "key", "mode", "name")} for mode in range(5)}
ratios = {str(mode): totals["4"]["primary_total_seconds"] / totals[str(mode)]["primary_total_seconds"]
          for mode in range(4)}
summary = dict(pairs=16, timed_calls=240, source_revision=records[0]["git_head"], totals=totals,
    candidate_ratios=ratios, numerical_speed_gate_passed=all(ratios[str(m)] <= 2/3 for m in (0, 1, 3)),
    max_reference_score_difference=max(r["max_score_difference"] for r in pairs),
    containment_violations=sum(r["containment"]["violations"] for r in pairs),
    initial_cell_checks=sum(r["containment"]["cell_checks"] for r in pairs),
    aggregate_checks=sum(r["containment"]["aggregate_checks"] for r in pairs),
    pose_accuracy_evaluated=False, confirmation=False)
selected = [sorted((c for c in r["timed_calls"] if c["mode"] == 4),
    key=lambda c: (c["primary_total_seconds"], c["repetition"]))[1] for r in pairs]
summary["primary_candidate_selected_median_call_breakdown"] = {
    "setup_seconds": sum(c["setup_seconds"] for c in selected),
    "scoring_seconds": sum(c["scoring_seconds"] for c in selected),
    "outer_gap_seconds": sum(c["call_seconds"] - c["native_total_seconds"] for c in selected),
    "shuffle_seconds": sum(c["charged_shuffle_seconds"] for c in selected)}
assert summary["numerical_speed_gate_passed"] == records[-1]["numerical_speed_gate_passed"]
(directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
