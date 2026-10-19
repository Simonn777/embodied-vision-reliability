"""Replay accounting tests on artificial arrays only."""
import os
import copy
import unittest
import numpy as np
from grid_backend import GridScorer
import run_grid as replay


class GridDriverTests(unittest.TestCase):
    def test_artificial_admission_requires_complete_recomputed_evidence(self):
        report = {"schema": "evr0012d.synthetic.v1", "status": "completed", "admission_passed": True,
            "library_sha256": "same", "original_library_sha256": "same",
            "validation": {"violations": 0, "cell_checks": 128, "aggregate_checks": 128},
            "warmups": [{} for _ in range(5)], "timed_calls": [],
            "medians": {str(i): 0.5 if i == 4 else 1.0 for i in range(5)},
            "candidate_ratios": {str(i): 0.5 for i in range(4)}}
        for repetition in range(9):
            for position in range(5):
                mode = (repetition + position) % 5
                report["timed_calls"].append({"repetition": repetition, "position": position, "mode": mode,
                    "winner_agrees": True, "score_agrees": True, "primary_total_seconds": 0.5 if mode == 4 else 1.0})
        replay.validate_artificial_admission(report, "same")
        for field, value in (("timed_calls", report["timed_calls"][:-1]), ("status", "error"),
                             ("library_sha256", "different"), ("medians", {})):
            damaged = copy.deepcopy(report)
            damaged[field] = value
            with self.assertRaises(ValueError):
                replay.validate_artificial_admission(damaged, "same")

    def test_empty_eligible_partition_and_all_mode_accounting(self):
        backend = GridScorer(os.environ["GRID_LIBRARY"])
        x = np.column_stack((np.arange(40) * 300.0, np.ones(40)))
        y = x + 1.0
        models = np.zeros((64, 3, 3))
        models[:, 2, 2] = 1.0
        reference = backend.scores(x, y, models, 1.0)
        saved = {"key": "artificial", "matches": 40,
            "proposals": [{"valid_index": i, "matrix": m.tolist()} for i, m in enumerate(models)],
            "exact_scores": reference.tolist()}
        result = replay.run_pair(backend, saved, x, y, 0, np.random.default_rng(replay.SHUFFLE_SEED))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["containment"]["cell_checks"], 0)
        self.assertEqual(result["containment"]["aggregate_checks"], 128)
        self.assertEqual(len(result["timed_calls"]), 15)
        for entry in result["timed_calls"]:
            self.assertEqual(entry["charged_shuffle_seconds"], result["shuffle_seconds"] if entry["mode"] == 1 else 0.0)
            self.assertEqual(entry["primary_total_seconds"], entry["call_seconds"] + entry["charged_shuffle_seconds"])
            self.assertEqual(entry["winner"], 0)
            self.assertEqual(entry["moment_cells"], 0)

    def test_reference_winner_mismatch_stops_before_timing(self):
        backend = GridScorer(os.environ["GRID_LIBRARY"])
        x = np.zeros((65, 2))
        models = np.zeros((64, 3, 3))
        reference = np.ones(64) * 65
        reference[3] = 64.0
        saved = {"key": "artificial", "matches": 65,
            "proposals": [{"valid_index": i, "matrix": m.tolist()} for i, m in enumerate(models)],
            "exact_scores": reference.tolist()}
        result = replay.run_pair(backend, saved, x, x, 0, np.random.default_rng(0))
        self.assertEqual(result["status"], "stop_native_reference_mismatch")
        self.assertEqual(result["timed_calls"], [])


if __name__ == "__main__":
    unittest.main()
