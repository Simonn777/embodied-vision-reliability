"""Derive the complete numerical-study table; no HDF5 access."""
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
directory = root / "results/0012-development"
records = [json.loads(line) for line in (directory / "diagnostic.jsonl").read_text().splitlines()]
pairs = [r for r in records if r["type"] == "pair"]
assert records[-1] == {"type": "complete", "completed_pairs": 16, "pose_accuracy_evaluated": False}
assert len(pairs) == 16 and all(r["status"] == "completed" for r in pairs)
rows = []
for i, r in enumerate(pairs, 1):
    m, c, t = r["rankings"]["moments"], r["rankings"]["culling"], r["times_seconds"]
    rows.append(dict(pair=i, key=r["key"], matches=r["matches"],
        direct_residuals=r["direct_residuals"], culling_residuals=c["exact_residuals"],
        moment_residuals=m["exact_residuals"], saved_vs_culling=r["residuals_saved_vs_culling"],
        moment_pruned=m["pruned_models"], culling_pruned=c["pruned_models"],
        moment_winner_correct=m["agrees_with_exact_winner"],
        culling_winner_correct=c["agrees_with_exact_winner"],
        dms_form_winner_correct=r["dms_form_agrees_with_exact_winner"],
        dms_form_exact_cost_excess=r["exact_scores"][r["dms_form_winner"]] - min(r["exact_scores"]),
        direct_seconds=t["direct_score"], moment_seconds=t["moments_including_setup"],
        culling_seconds=t["culling_including_setup"], dms_form_seconds=t["dms_including_setup"]))
with (directory / "all-pairs.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
totals = {key: sum(r[key] for r in rows) for key in (
    "direct_residuals", "culling_residuals", "moment_residuals", "saved_vs_culling",
    "moment_pruned", "culling_pruned", "moment_winner_correct", "culling_winner_correct",
    "dms_form_winner_correct", "direct_seconds", "moment_seconds", "culling_seconds", "dms_form_seconds")}
totals.update(pairs=len(rows), source_revision=records[0]["git_head"],
    saved_fraction_vs_culling=totals["saved_vs_culling"] / totals["culling_residuals"],
    bound_violations=sum(len(r["bound_violations"]) for r in pairs),
    initial_cell_checks_per_mode=sum(r["containment_cell_checks_per_mode"] for r in pairs),
    pairs_with_savings=sum(r["saved_vs_culling"] > 0 for r in rows),
    pairs_with_extra_work=sum(r["saved_vs_culling"] < 0 for r in rows),
    pose_accuracy_evaluated=False,
    practical_decision="insufficient_computational_benefit_no_accuracy_or_speed_claim")
(directory / "summary.json").write_text(json.dumps(totals, indent=2) + "\n")
print(json.dumps(totals, indent=2))
