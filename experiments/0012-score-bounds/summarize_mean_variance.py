"""Summarize every exploratory pair without opening match data."""
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
directory = root / "results/0012b-mean-variance"
records = [json.loads(line) for line in (directory / "diagnostic.jsonl").read_text().splitlines()]
pairs = [r for r in records if r["type"] == "pair"]
assert records[-1]["type"] == "complete" and records[-1]["completed_pairs"] == 16
assert len(pairs) == 16 and all(r["status"] == "completed" for r in pairs)
rows = []
for index, r in enumerate(pairs, 1):
    a, p = r["mean_variance"], r["partial_exact"]
    rows.append(dict(pair=index, key=r["key"], candidate_residuals=a["exact_residuals"],
        partial_residuals=p["exact_residuals"], full_residuals=r["historical"]["direct_residuals"],
        original_moment_residuals=r["historical"]["rankings"]["moments"]["exact_residuals"],
        candidate_pruned=a["pruned_models"], partial_pruned=p["pruned_models"],
        candidate_winner_correct=a["index"] == r["exact_winner"],
        partial_winner_correct=p["index"] == r["exact_winner"],
        cells_lifted=r["lower_lift"]["cells_strictly_lifted"],
        cells_checked=r["lower_lift"]["cells_checked"],
        bound_violations=len(r["bound_violations"]),
        candidate_seconds=r["times_seconds"]["mean_variance_including_setup"],
        partial_seconds=r["times_seconds"]["partial_including_setup"]))
with (directory / "all-pairs.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
totals = {key: sum(r[key] for r in rows) for key in rows[0] if key not in ("pair", "key")}
totals.update(pairs=len(rows), source_revision=records[0]["git_head"], confirmation=False,
    candidate_fraction_of_partial=totals["candidate_residuals"] / totals["partial_residuals"],
    candidate_fraction_of_full=totals["candidate_residuals"] / totals["full_residuals"],
    pose_accuracy_evaluated=False)
totals["numerical_gate_passed"] = (
    totals["candidate_fraction_of_partial"] <= 0.8 and totals["candidate_fraction_of_full"] <= 0.5
    and totals["candidate_winner_correct"] == 16 and totals["partial_winner_correct"] == 16
    and totals["bound_violations"] == 0)
(directory / "summary.json").write_text(json.dumps(totals, indent=2) + "\n")
print(json.dumps(totals, indent=2))
