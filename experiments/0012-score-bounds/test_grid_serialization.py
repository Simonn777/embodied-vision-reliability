"""Exercise the complete artificial driver's persistence using a fake backend."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import run_grid_synthetic as driver


class FakeBackend:
    def __init__(self, library):
        pass

    def scores(self, x, y, matrices, tau2):
        return np.r_[5000.0, np.full(63, 10000.0)]

    def validate(self, x, y, matrices, tau2):
        return {"violations": 0, "cell_checks": 128, "aggregate_checks": 128}

    def select(self, x, y, matrices, tau2, order, mode):
        return {"winner": 0, "score": 5000.0, "residuals": 1}


class SerializationTests(unittest.TestCase):
    def test_all_numpy_comparisons_persist_in_complete_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            receipt = root / "build.json"
            output = root / "output.json"
            receipt.write_text(json.dumps({"source_sha256": "fake", "included_source_sha256": {"native_score.cpp": "fake"},
                "library": "fake-library", "library_sha256": "fake", "compiler": "fake", "command": ["cc", "-O3", "source", "-o", "library"]}))
            arguments = ["run_grid_synthetic.py", "--build-receipt", str(receipt), "--output", str(output), "--original-library-sha256", "fake"]
            with patch.object(driver, "GridScorer", FakeBackend), patch.object(driver, "sha256", return_value="fake"), \
                    patch.object(driver.subprocess, "check_output", return_value="fake-head"), \
                    patch.object(driver.platform, "platform", return_value="fake-platform"), patch.object(driver.sys, "argv", arguments):
                self.assertEqual(driver.main(), 0)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(result["timed_calls"]), 45)
            self.assertEqual(len(result["warmups"]), 5)
            self.assertTrue(all(type(entry["score_agrees"]) is bool for entry in result["timed_calls"]))
            self.assertTrue(all(type(entry["winner_agrees"]) is bool for entry in result["timed_calls"]))


if __name__ == "__main__":
    unittest.main()
