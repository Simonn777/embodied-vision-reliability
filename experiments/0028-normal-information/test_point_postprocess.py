"""Manufactured arithmetic only; optional local-author smoke is opt-in.

No model construction, weights, image files or scientific arrays are used.
The parent supervisor executes this file; writing it does not qualify it.
"""
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_evr0028_postprocess_tested", HERE / "point_postprocess.py")
pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pp)
try:
    import torch
except ImportError:
    torch = None


def fake_provider():
    calls = []
    def recover(points, mask, focal=None):
        calls.append(("recover", points.clone(), mask.clone(), None if focal is None else focal.clone()))
        points.add_(999)  # A provider cannot corrupt the raw or other arms.
        return torch.tensor([1.25]) if focal is None else focal, torch.tensor([-0.25])
    def solve(rays, points, focal):
        calls.append(("solve", rays.copy(), points.copy(), focal))
        rays[:] = 999
        points[:] = 999
        return 0.5
    def intrinsics(fx, fy, cx, cy):
        cx, cy = cx.expand_as(fx), cy.expand_as(fy)
        z, o = torch.zeros_like(fx), torch.ones_like(fx)
        return torch.stack((fx, z, cx, z, fy, cy, z, z, o), dim=-1).reshape(-1, 3, 3)
    def sliding(x, window_size, stride=1, dim=(-2, -1)):
        d0, d1 = (i % x.ndim for i in dim)
        return x.unfold(d0, window_size, stride).unfold(d1, window_size, stride)
    namespace = SimpleNamespace(sliding_window=sliding)
    def refine(d, n, K, **options):
        calls.append(("refine", d.clone(), n.clone(), K.clone(), options))
        d.clamp_min_(1e-12)
        n.zero_()
        K.zero_()
        return d + 1
    return SimpleNamespace(torch=torch, recover_focal_shift=recover, solve_optimal_shift=solve,
                           intrinsics_from_focal_center=intrinsics, refine_depth_with_normal=refine,
                           window_namespace=namespace, window_compatibility=False,
                           provenance={"kind": "manufactured_fake_provider"}, calls=calls)


def manufactured(height=5, width=7):
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    points = torch.stack((x, y, x * 0 + 2), dim=-1).float().unsqueeze(0)
    normal = torch.zeros_like(points)
    normal[..., 2] = -1
    probability = torch.ones((1, height, width))
    probability[0, 0, 0] = 0.5
    raw = dict(points=points, normal=normal, mask=probability, metric_scale=torch.tensor([3.]))
    K = torch.tensor([[13., 0., 1.25], [0., 17., 2.75], [0., 0., 1.]])
    return raw, K


@unittest.skipIf(torch is None, "Torch unavailable; supervised existing-runtime qualification required")
class ArithmeticTests(unittest.TestCase):
    def test_inferred_output_mask_metric_scale_and_no_mutation(self):
        raw, K = manufactured()
        raw["points"][0, 1, 1, 2] = 0.125
        original = {k: v.clone() for k, v in raw.items()}
        out = pp.postprocess(raw, K, "A", provider=fake_provider())
        self.assertEqual(out["status"], "completed")
        self.assertTrue(torch.equal(out["depth"], (original["points"][0, ..., 2] - .25) * 3))
        self.assertFalse(bool(out["mask"][0, 0]))  # strict >.5
        self.assertFalse(bool(out["mask"][1, 1]))  # negative shifted z
        self.assertTrue(torch.equal(out["normal"], original["normal"][0]))
        for key in raw:
            self.assertTrue(torch.equal(raw[key], original[key]), key)
        self.assertEqual(out["recovery"]["camera_model"], "official_centered_square_pixels")

    def test_known_horizontal_fov_does_not_claim_full_authored_camera(self):
        raw, K = manufactured()
        provider = fake_provider()
        out = pp.postprocess(raw, K, "B", provider=provider)
        expected = 2 * 13 / math.hypot(7, 5)
        self.assertAlmostEqual(float(provider.calls[0][3][0]), expected, places=6)
        self.assertAlmostEqual(out["recovery"]["supplied_horizontal_fov_degrees"],
                               math.degrees(2 * math.atan(7 / 26)), places=12)
        self.assertAlmostEqual(float(out["K_pixels"][0, 0]), 13, places=5)
        self.assertAlmostEqual(float(out["K_pixels"][1, 1]), 13, places=5)
        self.assertEqual(float(out["K_pixels"][0, 2]), 3)
        self.assertEqual(float(out["K_pixels"][1, 2]), 2)
        self.assertFalse(torch.equal(out["K_pixels"], K))

    def test_general_camera_exact_nearest_indices_and_pixel_rays(self):
        raw, K = manufactured(65, 67)
        provider = fake_provider()
        out = pp.postprocess(raw, K, "C", provider=provider)
        self.assertEqual(out["status"], "completed")
        event = provider.calls[0]
        self.assertEqual(event[0], "solve")
        self.assertEqual(event[3], 1)
        rows = [j * 65 // 64 for j in range(64)]
        cols = [j * 67 // 64 for j in range(64)]
        expected = [(y, x) for y in rows for x in cols if (y, x) != (0, 0)]
        self.assertEqual(event[1].shape, (len(expected), 2))
        for index in (0, 30, len(expected) - 1):
            y, x = expected[index]
            self.assertAlmostEqual(float(event[1][index, 0]), (x - 1.25) / 13, places=6)
            self.assertAlmostEqual(float(event[1][index, 1]), (y - 2.75) / 17, places=6)
            self.assertEqual(event[2][index].tolist(), [x, y, 2])
        self.assertEqual(out["recovery"]["sample_rows"], rows)
        self.assertEqual(out["recovery"]["sample_columns"], cols)
        self.assertTrue(torch.equal(out["K_pixels"], K))
        self.assertTrue(torch.equal(out["depth"], torch.full((65, 67), 7.5)))
        self.assertEqual(float(raw["points"][0, 0, 0, 0]), 0)

    def test_nearest_upsampling_and_empty_support_fallback(self):
        expected = [j * 3 // 64 for j in range(64)]
        self.assertEqual(pp.nearest_indices(3, torch_module=torch).tolist(), expected)
        raw, K = manufactured()
        raw["mask"].zero_()
        provider = fake_provider()
        out = pp.postprocess(raw, K, "C", provider=provider)
        self.assertEqual(provider.calls, [])
        self.assertTrue(out["recovery"]["insufficient_support_fallback"])
        self.assertEqual(out["recovery"]["shift_raw_point_units"], 0)
        self.assertEqual(out["validity"]["mask_count"], 0)

    def test_numpy_raw_inputs_and_nonfinite_output_are_preserved(self):
        raw, K = manufactured()
        raw["points"][0, 2, 2, 2] = float("nan")
        provider = fake_provider()
        out = pp.postprocess({k: v.numpy() for k, v in raw.items()}, K.numpy(), "A", provider=provider)
        self.assertEqual(out["status"], "numerical_failure")
        self.assertTrue(bool(torch.isnan(out["depth"][2, 2])))
        self.assertEqual(out["validity"]["depth_nonfinite_count"], 1)
        metadata = {k: v for k, v in out.items() if not isinstance(v, torch.Tensor)}
        json.dumps(metadata, allow_nan=False)

    def test_nonfinite_recovery_scalar_metadata_is_json_safe(self):
        raw, K = manufactured()
        provider = fake_provider()
        provider.recover_focal_shift = lambda *a, **kw: (torch.tensor([float("nan")]), torch.tensor([float("inf")]))
        out = pp.postprocess(raw, K, "A", provider=provider)
        self.assertEqual(out["status"], "numerical_failure")
        self.assertIsNone(out["recovery"]["shift_raw_point_units"])
        self.assertIsNone(out["recovery"]["focal_relative_to_half_diagonal"])
        json.dumps(out["recovery"], allow_nan=False)

    def test_failures_retain_warnings_without_trying_other_helper(self):
        raw, K = manufactured()
        provider = fake_provider()
        def fail(*args, **kwargs):
            warnings.warn("manufactured helper warning", RuntimeWarning)
            raise RuntimeError("manufactured failure")
        provider.recover_focal_shift = fail
        out = pp.postprocess(raw, K, "A", provider=provider)
        self.assertEqual(out["status"], "helper_error")
        self.assertEqual(out["error"]["message"], "manufactured failure")
        self.assertEqual(out["warnings"][0]["message"], "manufactured helper warning")
        self.assertEqual(provider.calls, [])

    def test_camera_and_shape_faults_rejected_before_helper(self):
        raw, K = manufactured()
        p = fake_provider()
        bad = K.clone()
        bad[0, 1] = 0.01
        with self.assertRaises(pp.ContractError):
            pp.postprocess(raw, bad, "A", provider=p)
        raw["normal"] = raw["normal"][0, :-1]
        with self.assertRaises(pp.ContractError):
            pp.postprocess(raw, K, "C", provider=p)
        self.assertEqual(p.calls, [])

    def test_normal_helper_half_pixel_intrinsics_defaults_and_copies(self):
        raw, K = manufactured(7, 9)
        depth = raw["points"][0, ..., 2].clone()
        depth[0, 0] = -1
        normal = raw["normal"][0].clone()
        d0, n0, K0 = depth.clone(), normal.clone(), K.clone()
        p = fake_provider()
        out = pp.refine_with_normal(depth.numpy(), normal.numpy(), K.numpy(), provider=p)
        self.assertEqual(out["status"], "completed")
        event = p.calls[0]
        self.assertEqual(event[-1], {"iterations": 10, "damp": .001, "eps": 1e-12, "kernel_size": 5})
        self.assertAlmostEqual(float(event[3][0, 2]), 1.75 / 9, places=7)
        self.assertAlmostEqual(float(event[3][1, 2]), 3.25 / 7, places=7)
        self.assertTrue(torch.equal(depth, d0))
        self.assertTrue(torch.equal(normal, n0))
        self.assertTrue(torch.equal(K, K0))
        self.assertEqual(out["input_depth_nonpositive_count"], 1)

    def test_window_alias_matches_explicit_stacked_windows(self):
        p = fake_provider()
        x = torch.arange(2 * 6 * 7).reshape(2, 6, 7)
        actual = pp.last_two_axis_window(p.window_namespace.sliding_window, x, 3, stride=2)
        expected = torch.stack([torch.stack([
            torch.stack([x[b, i:i + 3, j:j + 3] for j in (0, 2, 4)])
            for i in (0, 2)]) for b in range(2)])
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.untyped_storage().data_ptr(), x.untyped_storage().data_ptr())
        with self.assertRaises(pp.ContractError):
            pp.last_two_axis_window(p.window_namespace.sliding_window, x, 3, dim=(0, 1))

    def test_scoped_alias_restored_even_when_helper_fails(self):
        raw, K = manufactured(7, 9)
        p = fake_provider()
        def fail(d, n, intrinsic, **kw):
            self.assertTrue(hasattr(p.window_namespace, "sliding_window_2d"))
            p.window_namespace.sliding_window_2d(d, window_size=5, stride=1, dim=(-2, -1))
            warnings.warn("alias fixture", UserWarning)
            raise RuntimeError("after alias")
        p.refine_depth_with_normal = fail
        out = pp.refine_with_normal(raw["points"][0, ..., 2], raw["normal"][0], K,
                                    provider=pp.with_window_compatibility(p))
        self.assertEqual(out["status"], "helper_error")
        self.assertEqual(out["warnings"][0]["message"], "alias fixture")
        self.assertTrue(out["compatibility"]["applied"])
        self.assertTrue(out["compatibility"]["restored"])
        self.assertFalse(hasattr(p.window_namespace, "sliding_window_2d"))
        json.dumps(out["compatibility"], allow_nan=False)

    def test_source_hash_mismatch_stops_provider_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "author.py"
            path.write_text("# wrong source\n")
            with patch.object(pp.importlib, "import_module", return_value=SimpleNamespace(__file__=str(path))):
                with self.assertRaisesRegex(pp.ContractError, "source_hash_mismatch"):
                    pp.load_provider()


@unittest.skipUnless(torch is not None and os.environ.get("EVR0028_MOGE_SOURCE"),
                     "optional local author-helper smoke not requested")
class LocalAuthorSmoke(unittest.TestCase):
    def test_original_failure_then_declared_alias_full_helper(self):
        source = Path(os.environ["EVR0028_MOGE_SOURCE"]).resolve()
        installation = source.parent.parent
        old_path = list(sys.path)
        sys.path[:0] = [str(source), str(source.parent / "utils3d")]
        try:
            qpath = HERE.parent / "0022-frontend" / "qualify.py"
            qspec = importlib.util.spec_from_file_location("_evr0028_typing_qualification", qpath)
            q = importlib.util.module_from_spec(qspec)
            qspec.loader.exec_module(q)
            q.prepare_utils3d_typing(installation)
            p = pp.load_provider()
            raw, K = manufactured(9, 11)
            depth = raw["points"][0, ..., 2].clone()
            normal = raw["normal"][0].clone()
            failed = pp.refine_with_normal(depth, normal, K, provider=p)
            # This preserves the expected missing API evidence without a
            # unittest expectedFailure that could hide an unrelated failure.
            print("AUTHOR_ORIGINAL_API_PROBE " + json.dumps(
                {k: failed[k] for k in ("status", "error", "warnings", "compatibility")}, sort_keys=True))
            self.assertEqual(failed["status"], "helper_error")
            self.assertEqual(failed["error"]["type"], "AttributeError")
            self.assertIn("sliding_window_2d", failed["error"]["message"])
            fixed = pp.refine_with_normal(depth, normal, K, provider=pp.with_window_compatibility(p))
            print("AUTHOR_DECLARED_ALIAS_PROBE " + json.dumps(
                {k: fixed[k] for k in ("status", "warnings", "compatibility")}, sort_keys=True))
            self.assertEqual(fixed["status"], "completed", fixed.get("error"))
            self.assertTrue(fixed["finite"] and fixed["positive"])
            self.assertTrue(torch.allclose(fixed["depth"], depth, atol=2e-6, rtol=2e-6))
            self.assertTrue(torch.equal(depth, raw["points"][0, ..., 2]))
            self.assertTrue(torch.equal(normal, raw["normal"][0]))
            self.assertFalse(hasattr(p.window_namespace, "sliding_window_2d"))
        finally:
            sys.path[:] = old_path


if __name__ == "__main__":
    unittest.main()
