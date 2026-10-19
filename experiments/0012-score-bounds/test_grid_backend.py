"""Artificial numerical qualification only; no dataset reads."""
import json
import os
from pathlib import Path
import unittest
import numpy as np
from grid_backend import GridScorer


def favorable_fixture():
    s = np.linspace(0.0, 240.0, 10000)
    x = np.column_stack((np.zeros_like(s), s))
    y = np.column_stack((np.zeros_like(s), s + 10.0))
    models = np.zeros((64, 3, 3))
    models[:, 1, 2], models[:, 2, 1] = -1.0, 1.0
    models[0, 2, 2] = 9.0
    return x, y, models


class GridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = GridScorer(os.environ["GRID_LIBRARY"])

    def check_case(self, x, y, models, tau2=1.0):
        exact = self.backend.scores(x, y, models, tau2)
        validation = self.backend.validate(x, y, models, tau2)
        self.assertEqual(validation["violations"], 0, validation)
        order = np.random.default_rng(77).permutation(len(x)).astype(np.uint64)
        for mode in range(5):
            result = self.backend.select(x, y, models, tau2, order, mode)
            self.assertEqual(result["winner"], int(np.argmin(exact)), (mode, result))
            self.assertEqual(result["score"], exact[result["winner"]])
        return validation

    def test_favorable_and_lazy_summary(self):
        x, y, models = favorable_fixture()
        validation = self.check_case(x, y, models)
        self.assertEqual(validation["aggregate_checks"], 128)
        order = np.arange(len(x), dtype=np.uint64)
        candidate = self.backend.select(x, y, models, 1.0, order, 4)
        control = self.backend.select(x, y, models, 1.0, order, 3)
        self.assertEqual(candidate["grid_cells"], 1)
        self.assertEqual(candidate["moment_cells"], 1)
        self.assertEqual(candidate["moment_points"], 10000)
        self.assertEqual(candidate["residuals"], 10000)
        self.assertLess(candidate["residuals"], control["residuals"])

    def test_random_negative_and_boundary_coordinates(self):
        rng = np.random.default_rng(338)
        x, y = rng.uniform(-255.0, -1.0, (512, 2)), rng.uniform(-255.0, -1.0, (512, 2))
        x[:4] = [[-256, 0], [0, 256], [-512, -256], [256, -512]]
        self.check_case(x, y, rng.normal(size=(12, 3, 3)))

    def test_small_cells_and_ties(self):
        x = np.column_stack((np.arange(100) * 300.0, np.ones(100)))
        y = x + 1
        models = np.zeros((4, 3, 3))
        validation = self.check_case(x, y, models)
        self.assertEqual(validation["cell_checks"], 0)
        result = self.backend.select(x, y, models, 1.0, None, 4)
        self.assertEqual(result["moment_cells"], 0)
        self.assertEqual(result["winner"], 0)

    def test_zero_denominator_and_identical_models(self):
        rng = np.random.default_rng(339)
        x, y = rng.uniform(1, 100, (300, 2)), rng.uniform(1, 100, (300, 2))
        models = np.zeros((3, 3, 3))
        models[:, 2, 2] = 1.0
        self.check_case(x, y, models)

    def test_extreme_finite_scales_and_key_range(self):
        rng = np.random.default_rng(340)
        for scale in (1e-80, 1e50):
            x, y = rng.normal(size=(128, 2)) * scale, rng.normal(size=(128, 2)) * scale
            # Residual squares remain finite; grid keys may exceed int64.
            self.check_case(x, y, rng.normal(size=(3, 3, 3)))

    def test_invalid_inputs_rejected(self):
        x, y, models = favorable_fixture()
        with self.assertRaises((ValueError, RuntimeError)):
            self.backend.select(x, y, models, 1.0, None, 99)
        with self.assertRaises((ValueError, RuntimeError)):
            self.backend.select(x, y, models, 1.0, np.zeros(len(x), dtype=np.uint64), 1)
        x[0, 0] = np.nan
        with self.assertRaises((ValueError, RuntimeError)):
            self.backend.select(x, y, models, 1.0, None, 4)


if __name__ == "__main__":
    unittest.main()
