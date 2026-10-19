"""Synthetic tests only; no HDF5 or saved scientific outputs are read."""
import unittest

import run_mean_variance as replay

np = replay.np


class ReplayTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2401)
        self.x = rng.normal(size=(700, 2))
        self.y = rng.normal(size=(700, 2))
        self.y[:, 1] = self.x[:, 1]
        self.good = np.array([[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
        self.bad = self.good.copy()
        self.bad[2, 2] = 100
        self.order = np.random.default_rng(replay.SHUFFLE_SEED).permutation(len(self.x))

    def test_partial_pruning_tie_counts_and_determinism(self):
        matrices = [self.good, self.good.copy(), self.bad]
        result = replay.partial_exact(matrices, self.x, self.y, self.order)
        again = replay.partial_exact(matrices, self.x, self.y, self.order)
        self.assertEqual(result, again)
        self.assertEqual(result["index"], 0)
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["models"][1]["score"], 0)
        self.assertEqual(result["models"][2]["residuals"], 256)
        self.assertEqual(result["exact_residuals"], 700 + 700 + 256)
        self.assertEqual(result["early_pruned_models"], 1)
        self.assertGreater(result["models"][2]["partial_sum"], result["guard"])

    def test_completed_sum_uses_original_point_order(self):
        n = 1025
        x, y = np.zeros((n, 2)), np.zeros((n, 2))
        y[:, 1] = np.sqrt(2 * np.r_[1., np.full(n - 1, 2. ** -54)])
        order = np.arange(n - 1, -1, -1)
        costs = replay.residuals(self.good, x, y)
        self.assertNotEqual(float(np.sum(costs)), float(np.sum(costs[order])))
        result = replay.partial_exact([self.bad, self.good], x, y, order)
        self.assertEqual(result["index"], 1)
        self.assertEqual(result["score"], replay.core.exact_msac(self.good, x, y, 1.0))
        self.assertEqual(result["score"], float(np.sum(costs)))
        self.assertEqual(result["exact_residuals"], n * 2)

    def test_injected_selector_and_restoration_on_success_and_exception(self):
        original = replay.core.cell_bounds
        tree = replay.core.MatchTree(self.x, self.y, leaf_size=32)
        with replay.added_bound():
            self.assertIsNot(replay.core.cell_bounds, original)
            result = tree.best_hypothesis([self.good, self.good.copy(), self.bad], 1.,
                initial_cells=128, max_refinements=128, mode="moments")
            self.assertEqual(result.index, 0)
        self.assertIs(replay.core.cell_bounds, original)
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with replay.added_bound():
                raise RuntimeError("synthetic")
        self.assertIs(replay.core.cell_bounds, original)

    def test_replay_schema_and_containment_without_real_reference(self):
        x, y = self.x[:40], self.y[:40]
        matrices = [self.good if i % 2 == 0 else self.bad for i in range(64)]
        truth = [replay.core.exact_msac(m, x, y, 1.) for m in matrices]
        saved = {"key": "synthetic", "matches": 40,
            "proposals": [{"valid_index": i, "matrix": m.tolist()} for i, m in enumerate(matrices)],
            "exact_scores": truth, "rankings": {}, "times_seconds": {},
            "initial_bounds": [[0, 40]] * 64, "direct_residuals": 40 * 64}
        result = replay.replay_pair(saved, x, y, np.random.default_rng(replay.SHUFFLE_SEED))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["bound_violations"], [])
        self.assertEqual(result["mean_variance"]["index"], result["exact_winner"])
        self.assertEqual(result["partial_exact"]["index"], result["exact_winner"])
        self.assertEqual(result["validation_residuals"], 40 * 64)
        self.assertEqual(result["historical"]["rankings"], {})


if __name__ == "__main__":
    unittest.main()
