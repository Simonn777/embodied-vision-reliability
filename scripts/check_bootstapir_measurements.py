"""Recompute published development adapter checks from scalar CSV, no torch."""
import csv
import json
from pathlib import Path
import numpy as np
from evr.bootstapir_dev import compare, sha256


def load_runs(path):
    rows = list(csv.DictReader(open(path)))
    result = {}
    for name in sorted({r["run"] for r in rows}):
        subset = [r for r in rows if r["run"] == name]
        shape = (max(int(r["frame"]) for r in subset) + 1,
                 max(int(r["query"]) for r in subset) + 1, 4)
        arr = np.full(shape, np.nan, np.float32)
        for r in subset:
            if int(r["active"]):
                arr[int(r["frame"]), int(r["query"])] = [float(r[k]) for k in
                    ["x", "y", "occlusion_logit", "expected_dist_logit"]]
        result[name] = arr
    return result


def main():
    root = Path("results/0002-development")
    runs = load_runs(root / "validated/raw-outputs.csv")
    base, late = runs["single_16"], runs["late_query_16"]
    checks = {
        "future_image_prefix": compare(base[:8], runs["future_images_changed"][:8]),
        "truncated_prefix": compare(base[:8], runs["prefix_8"]),
        "single_vs_batch": compare(base[:, 0], runs["batch_16"][:, 0]),
        "future_query_past": compare(base[:4, 0], late[:4, 0]),
        "existing_query_after_arrival": compare(base[:, 0], late[:, 0]),
        "late_query_vs_independent_crop": compare(late[4:, 1], runs["independent_late_12"][:, 0]),
        "repeat": compare(base, runs["repeat_single_16"]),
        "prequery_mask": {"passed": bool(np.isnan(late[:4, 1]).all())},
    }
    receipt = json.loads((root / "validated/measure-receipt.json").read_text())
    assert checks == receipt["checks"] and receipt["status"] == "completed"
    other = load_runs(root / "independent-repeat/raw-outputs.csv")
    checks["independent_process"] = compare(runs["cold_4"], other["cold_4"])
    for name, item in json.loads((root / "raw-artifact-index.json").read_text()).items():
        assert sha256(root / name / "raw-outputs.csv") == item["published_csv_sha256"]
    assert all(v["passed"] for v in checks.values())
    print(json.dumps({"all_checks_passed": True, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
