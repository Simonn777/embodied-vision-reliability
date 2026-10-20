"""Manufactured tensor arithmetic only; no model, checkpoint or media access."""
import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("d38_operators", Path(__file__).with_name("operators.py"))
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)


class OperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.t = torch

    def context(self, spatial=(1., 2.), pointer=(3.,), dtype=None):
        t = self.t
        dtype = dtype or t.float32
        values = tuple(spatial) + tuple(pointer)
        return dict(curr=[t.zeros(2, 1, 2, dtype=dtype)],
                    curr_pos=[t.ones(2, 1, 2, dtype=dtype)],
                    memory=t.tensor(values, dtype=dtype).reshape(-1, 1, 1),
                    memory_pos=t.arange(len(values), dtype=dtype).reshape(-1, 1, 1),
                    num_obj_ptr_tokens=len(pointer))

    def write(self, value, pointer):
        t = self.t
        return dict(maskmem_features=t.full((1, 2, 2, 2), value, dtype=t.bfloat16),
                    obj_ptr=t.full((1, 4), pointer, dtype=t.float32),
                    maskmem_pos_enc=[t.ones(1, 2, 2, 2)])

    def attention(self, memory):
        # Independent scalar-query attention: last channel is V, first is K.
        t = self.t
        weights = t.softmax(memory[:, 0, 0].double(), dim=0)
        return (weights * memory[:, 0, 1].double()).sum()

    def test_mean_write_native_dtype_projected_pointer_and_no_mutation(self):
        t = self.t
        rows = [self.write(v, p) for v, p in ((1, 4), (3, 10), (8, 16))]
        out, info = ops.mean_write(rows, torch=t)
        self.assertEqual(out["maskmem_features"].dtype, t.bfloat16)
        self.assertTrue(t.equal(out["maskmem_features"], t.full((1, 2, 2, 2), 4., dtype=t.bfloat16)))
        self.assertTrue(t.equal(out["obj_ptr"], t.full((1, 4), 10.)))
        self.assertFalse(info["identity_fastpath_used"])
        out["maskmem_features"].zero_()
        out["obj_ptr"].zero_()
        out["maskmem_pos_enc"][0].zero_()
        self.assertEqual(rows[0]["maskmem_features"][0, 0, 0, 0].item(), 1)
        self.assertEqual(rows[0]["obj_ptr"][0, 0].item(), 4)
        self.assertEqual(rows[0]["maskmem_pos_enc"][0][0, 0, 0, 0].item(), 1)

    def test_flat_pointer_tail_and_metadata_layout(self):
        t = self.t
        rows = [self.context((1, 2), (3,)), self.context((4, 5), (6,)), self.context((7, 8), (9,))]
        out, info = ops.flatten_contexts(rows, torch=t)
        self.assertEqual(out["memory"].flatten().tolist(), [1, 2, 4, 5, 7, 8, 3, 6, 9])
        self.assertEqual(out["memory_pos"].flatten().tolist(), [0, 1, 0, 1, 0, 1, 2, 2, 2])
        self.assertEqual(out["num_obj_ptr_tokens"], 3)
        self.assertEqual(info["spatial_tokens"], 6)
        out["curr"][0].fill_(9)
        out["memory"].zero_()
        self.assertTrue(t.equal(rows[0]["curr"][0], t.zeros(2, 1, 2)))
        self.assertEqual(rows[0]["memory"].flatten().tolist(), [1, 2, 3])

    def test_zero_pointer_context(self):
        rows = [self.context((1, 2), ()) for _ in range(3)]
        out, _ = ops.flatten_contexts(rows, torch=self.t)
        self.assertEqual(out["memory"].flatten().tolist(), [1, 2, 1, 2, 1, 2])
        self.assertEqual(out["num_obj_ptr_tokens"], 0)

    def test_identity_fastpath_requires_exact_complete_inputs(self):
        t = self.t
        rows = [self.context() for _ in range(3)]
        self.assertTrue(ops.contexts_identical(rows, torch=t))
        out, info = ops.flatten_contexts(rows, torch=t, identity_fastpath=True)
        self.assertTrue(info["identity_fastpath_used"])
        self.assertTrue(t.equal(out["memory"], rows[0]["memory"]))
        rows[2]["memory"][-1] += 0.0001
        self.assertFalse(ops.contexts_identical(rows, torch=t))
        out, info = ops.flatten_contexts(rows, torch=t, identity_fastpath=True)
        self.assertFalse(info["identity_fastpath_used"])
        self.assertEqual(out["memory"].shape[0], 9)
        writes = [self.write(2, 3) for _ in range(3)]
        _, info = ops.mean_write(writes, torch=t, identity_fastpath=True)
        self.assertTrue(info["identity_fastpath_used"])

    def test_equal_candidates_keep_mass_only_with_complete_histories(self):
        t = self.t
        context = self.context((0, 0), (), t.float64)
        context["memory"] = t.tensor([[[0., 0.]], [[0., 1.]]], dtype=t.float64)
        context["memory_pos"] = t.zeros_like(context["memory"])
        flat, _ = ops.flatten_contexts([context, context, context], torch=t)
        self.assertAlmostEqual(self.attention(context["memory"]).item(), .5, places=14)
        self.assertAlmostEqual(self.attention(flat["memory"]).item(), .5, places=14)
        naive = t.tensor([[[0., 0.]], [[0., 1.]], [[0., 1.]], [[0., 1.]]], dtype=t.float64)
        self.assertAlmostEqual(self.attention(naive).item(), .75, places=14)

    def counterexample_contexts(self):
        t = self.t
        rows = []
        for key in (0., 0., math.log(9.)):
            c = self.context((0, 0), (), t.float64)
            c["memory"] = t.tensor([[[0., 0.]], [[key, 1.]]], dtype=t.float64)
            c["memory_pos"] = t.zeros_like(c["memory"])
            rows.append(c)
        return rows

    def test_independent_normalization_differs_from_early_mean(self):
        t = self.t
        rows = self.counterexample_contexts()
        mixture = sum(self.attention(c["memory"]).item() for c in rows) / 3
        early = self.attention(t.stack([c["memory"] for c in rows]).mean(0)).item()
        self.assertAlmostEqual(mixture, 19 / 30, places=14)
        self.assertGreater(abs(early - mixture), .03)

    def test_complete_flat_pooling_differs_from_uniform_branch_mixture(self):
        t = self.t
        rows = self.counterexample_contexts()
        flat, _ = ops.flatten_contexts(rows, torch=t)
        mixture = sum(self.attention(c["memory"]).item() for c in rows) / 3
        self.assertAlmostEqual(self.attention(flat["memory"]).item(), 11 / 14, places=14)
        self.assertGreater(abs(self.attention(flat["memory"]).item() - mixture), .1)

    def test_stack_mean_float32_and_no_mutation(self):
        t = self.t
        rows = [t.full((2, 1, 4), v, dtype=t.bfloat16) for v in (1., 3., 8.)]
        out, info = ops.mean_stack_outputs(rows, torch=t)
        self.assertEqual(out.dtype, t.float32)
        self.assertTrue(t.equal(out, t.full((2, 1, 4), 4.)))
        out.zero_()
        self.assertEqual(rows[0][0, 0, 0].item(), 1)
        self.assertFalse(info["identity_fastpath_used"])
        _, info = ops.mean_stack_outputs([rows[0]] * 3, torch=t, identity_fastpath=True)
        self.assertTrue(info["identity_fastpath_used"])

    def test_invalid_layout_positions_and_nonfinite_rejected(self):
        t = self.t
        rows = [self.context() for _ in range(3)]
        rows[2]["memory_pos"][0] += 1
        with self.assertRaises(ValueError):
            ops.flatten_contexts(rows, torch=t)
        rows = [self.context() for _ in range(3)]
        rows[2]["num_obj_ptr_tokens"] = 3
        with self.assertRaises(ValueError):
            ops.contexts_identical(rows, torch=t)
        writes = [self.write(1, 2) for _ in range(3)]
        writes[1]["obj_ptr"][0, 0] = float("nan")
        with self.assertRaises(ValueError):
            ops.mean_write(writes, torch=t)
        with self.assertRaises(ValueError):
            ops.mean_stack_outputs([t.ones(1, 1, 1)] * 2, torch=t)


if __name__ == "__main__":
    unittest.main()
