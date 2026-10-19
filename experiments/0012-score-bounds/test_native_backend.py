"""Synthetic-only native parity tests; require an already-built local library.

Run native_backend.py --build first, then pass its build receipt via
EVR_NATIVE_BUILD_RECEIPT. No dataset or scientific result file is opened.
"""
import json
import os
from pathlib import Path
import unittest

import run_native as driver
from native_backend import NativeScorer, sha256
import score_bounds as python_reference

np = driver.np


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        default = Path(__file__).resolve().parents[3] / "native/build.json"
        receipt = json.loads(Path(os.environ.get("EVR_NATIVE_BUILD_RECEIPT", default)).read_text())
        assert receipt["source_sha256"] == sha256(Path(__file__).with_name("native_score.cpp"))
        assert receipt["library_sha256"] == sha256(receipt["library"])
        cls.backend = NativeScorer(receipt["library"])

    def check_case(self, x, y, matrices):
        reference = np.array([python_reference.exact_msac(m, x, y, 1.) for m in matrices])
        scores = self.backend.scores(x, y, matrices, 1.)
        self.assertTrue(np.all(np.abs(scores - reference) <= 1e-8 + 1e-12 * np.abs(reference)))
        self.assertEqual(int(np.argmin(scores)), int(np.argmin(reference)))
        validation = self.backend.validate(x, y, matrices, 1.)
        self.assertEqual(validation["violations"], 0)  # no containment tolerance
        self.assertEqual(validation["aggregate_checks"], len(matrices))
        self.assertGreaterEqual(validation["cell_checks"], len(matrices))
        permutation = np.random.default_rng(20261020).permutation(len(x)).astype(np.uint64)
        results = []
        for mode in range(5):
            result = self.backend.select(x, y, matrices, 1., permutation, mode)
            self.assertEqual(result["winner"], int(np.argmin(reference)), "mode %d" % mode)
            self.assertTrue(driver.score_matches(result["score"], reference[result["winner"]]))
            self.assertGreaterEqual(result["residuals"], len(x))
            results.append(result)
        self.assertEqual(results[0]["residuals"], len(x) * len(matrices))
        return results

    def test_random_multilevel_and_large_pixel_coordinates(self):
        rng = np.random.default_rng(31126)
        for n, scale in ((19, 1.), (513, 2.), (4101, 1600.)):
            x, y = rng.normal(size=(n, 2)) * scale, rng.normal(size=(n, 2)) * scale
            matrices = rng.normal(size=(9, 3, 3))
            self.check_case(x, y, matrices)

    def test_zero_denominators_model_scales_and_ties(self):
        rng = np.random.default_rng(711)
        x, y = rng.normal(size=(75, 2)), rng.normal(size=(75, 2))
        zero = np.zeros((3, 3))
        self.check_case(x, y, np.array([zero, zero.copy()]))
        base = rng.normal(size=(3, 3))
        self.check_case(x, y, np.array([base, base.copy()]))
        for scale in (1e-200, 1e200):
            self.check_case(x, y, np.array([base * scale]))

    def test_partial_pruning_counts_and_exact_tie(self):
        rng = np.random.default_rng(2401)
        x, y = rng.normal(size=(700, 2)), rng.normal(size=(700, 2))
        y[:, 1] = x[:, 1]
        good = np.array([[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
        bad = good.copy()
        bad[2, 2] = 100
        results = self.check_case(x, y, np.array([good, good.copy(), bad]))
        self.assertEqual(results[1]["winner"], 0)
        self.assertEqual(results[1]["residuals"], 700 + 700 + 256)
        self.assertEqual(results[1]["pruned"], 1)

    def test_invalid_inputs_and_native_error_propagation(self):
        x = np.zeros((8, 2))
        matrix = np.eye(3)[None]
        with self.assertRaises(ValueError):
            self.backend.scores(x, x, matrix * np.nan, 1.)
        with self.assertRaises(RuntimeError):
            self.backend.select(x, x, matrix, 1., np.zeros(8), 1)
        self.assertEqual(self.backend.select(x, x, matrix, 1., None, 0)["winner"], 0)
        self.assertEqual(self.backend.select(x, x, matrix, 1., None, 2)["winner"], 0)
        with self.assertRaises(RuntimeError):
            self.backend.select(x, x, matrix, 1., np.arange(8), 99)

    def test_public_receipt_preserves_hashes_without_workspace_paths(self):
        source = Path(__file__).with_name("native_score.cpp").resolve()
        library = source.parents[3] / "native/libnative_score.dylib"
        receipt = {"source": str(source), "library": str(library), "source_sha256": "source-hash",
                   "library_sha256": "library-hash", "command": ["clang++", "-O3", str(source), "-o", str(library)]}
        public = driver.public_build_receipt(receipt)
        self.assertNotIn(str(source.parents[3]), json.dumps(public))
        self.assertEqual(public["source_sha256"], receipt["source_sha256"])
        self.assertEqual(public["library_sha256"], receipt["library_sha256"])
        self.assertEqual(public["command"][1], "-O3")
        self.assertEqual(receipt["source"], str(source))

    def test_driver_schedule_and_synthetic_pair_schema(self):
        rng = np.random.default_rng(77)
        x, y = rng.normal(size=(40, 2)), rng.normal(size=(40, 2))
        matrices = rng.normal(size=(64, 3, 3))
        reference = [python_reference.exact_msac(m, x, y, 1.) for m in matrices]
        saved = {"key": "synthetic_only", "matches": len(x), "exact_scores": reference,
            "proposals": [{"valid_index": i, "matrix": m.tolist()} for i, m in enumerate(matrices)]}
        result = driver.run_pair(self.backend, saved, x, y, 2, np.random.default_rng(20261020))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["timed_calls"]), 15)
        self.assertEqual([r["mode"] for r in result["timed_calls"]],
                         [m for repeat in range(3) for m in driver.mode_order(2, repeat)])
        for entry in result["timed_calls"]:
            self.assertTrue(entry["winner_agrees"] and entry["score_agrees"])
            charged = result["shuffle_seconds"] if entry["mode"] in (1, 3, 4) else 0.
            self.assertEqual(entry["charged_shuffle_seconds"], charged)
            self.assertEqual(entry["primary_total_seconds"], entry["call_seconds"] + charged)
        self.assertEqual(set(result["medians"]), set("01234"))
        self.assertEqual(set(result["primary_candidate_ratios"]), set("0123"))


if __name__ == "__main__":
    unittest.main()
