"""Artificial-only input, geometry, gauge, and accounting tests for 0017."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location("common0017_test", Path(__file__).with_name("common.py"))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def fixture(n=12):
    rng = np.random.default_rng(17)
    X = np.column_stack((rng.uniform(-1, 1, n), rng.uniform(-.8, .8, n), rng.uniform(3, 7, n)))
    t = np.array([.4, .03, .08])
    Y = X + t
    cal = {}
    coords = []
    for i, (P, params) in enumerate(((X, [500., 510., 320., 240.]), (Y, [600., 620., 300., 250.])), 1):
        cal["camera"+str(i)] = {"model": "PINHOLE", "width": 640, "height": 480, "params": params}
        fx, fy, cx, cy = params
        cal["K"+str(i)] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
        coords.append(P[:, :2]/P[:, 2, None]*params[:2]+params[2:])
    return coords[0], coords[1], cal, X[:, 2], Y[:, 2], {"R": np.eye(3), "t": t, "scale": 1., "shift1": 0., "shift2": 0.}


class Dataset:
    def __init__(self, value, name, audit):
        self.value, self.name, self.audit = value, name, audit

    def __getitem__(self, index):
        if index != ():
            raise AssertionError("Unexpected dataset slicing")
        self.audit.append(self.name)
        return self.value


class Group:
    def __init__(self, items):
        self.items = items

    def __getitem__(self, key):
        if key not in self.items:
            raise AssertionError("Forbidden field access: " + key)
        return self.items[key]

    def __iter__(self):
        raise AssertionError("Group enumeration forbidden")


def handle_fixture():
    x1, x2, cal, d1, d2, pose = fixture()
    d1[0] = -3.
    d2[1] = 0.
    audit = []
    values = {"x1": x1, "x2": x2, "K1": cal["K1"], "K2": cal["K2"], "depth1": d1, "depth2": d2}
    items = {k: Dataset(v, k, audit) for k, v in values.items()}
    for name in ("camera1", "camera2"):
        camera = cal[name]
        items[name] = Group({k: Dataset(np.asarray([v] if k != "params" else v), name+"/"+k, audit)
                             for k, v in camera.items()})
    return Group({"pair": Group(items)}), values, audit, pose


class InputTests(unittest.TestCase):
    def test_exact_allowlist_no_labels_and_no_depth_filter(self):
        handle, values, audit, _ = handle_fixture()
        result = c.read_pair(handle, "pair")
        self.assertEqual(len(audit), 14)
        self.assertEqual(set(audit), {"x1", "x2", "K1", "K2", "depth1", "depth2"} |
            {name+"/"+k for name in ("camera1", "camera2") for k in ("model", "width", "height", "params")})
        np.testing.assert_array_equal(result["depth1"], values["depth1"])
        np.testing.assert_array_equal(result["depth2"], values["depth2"])
        self.assertEqual(result["input_diagnostics"]["depth1_nonpositive_count"], 1)
        self.assertEqual(result["input_diagnostics"]["depth2_zero_count"], 1)
        self.assertEqual(len(result["x1"]), len(values["x1"]))

    def test_nonfinite_depth_and_shape_refusal(self):
        for bad in (np.full(12, np.nan), np.full(12, np.inf), np.ones((12, 1))):
            handle, _, _, _ = handle_fixture()
            handle.items["pair"].items["depth1"].value = bad
            with self.assertRaises(ValueError):
                c.read_pair(handle, "pair")

    def test_unsupported_camera_and_K_disagreement(self):
        handle, _, _, _ = handle_fixture()
        handle.items["pair"].items["camera1"].items["model"].value = np.array(["SIMPLE_PINHOLE"])
        with self.assertRaises(ValueError):
            c.read_pair(handle, "pair")
        handle, values, _, _ = handle_fixture()
        values["K1"][0, 1] = 2
        with self.assertRaises(ValueError):
            c.read_pair(handle, "pair")

    def test_masks_and_json_are_strict(self):
        for mask in ([True], [0, 2], [0., np.nan], ["0", "1"]):
            with self.assertRaises(ValueError):
                c.validate_mask(mask, 2)
        np.testing.assert_array_equal(c.validate_mask([0., 1.], 2), [False, True])
        for value in (float("inf"), np.array([float("nan")])):
            with self.assertRaises(ValueError):
                c.json_safe(value)


class GeometryTests(unittest.TestCase):
    def test_perfect_projection_and_common_score(self):
        x1, x2, cal, d1, d2, pose = fixture()
        g = c.prepare_geometry(x1, x2, cal)
        result = c.final_scores(pose, g)
        self.assertLess(result["pixel_msac"], 1e-20)
        self.assertEqual(result["cheiral_count"], len(x1))
        self.assertEqual(result["pixel_threshold"], 2.)
        diag = c.depth_diagnostics(pose, g, d1, d2, np.ones(len(x1), bool))
        self.assertLess(diag["all"]["fixed_reprojection_cost"], 1e-20)
        self.assertEqual(diag["all"]["inactive_transformed_direction_count"], 0)

    def test_negative_translation_cheirality_and_degeneracy(self):
        x1, x2, cal, _, _, pose = fixture()
        g = c.prepare_geometry(x1, x2, cal)
        pose["t"] *= -1
        self.assertEqual(c.final_scores(pose, g)["pixel_msac"], float(len(x1)))
        axis1 = np.array([cal["camera1"]["params"][2:]])
        axis2 = np.array([cal["camera2"]["params"][2:]])
        g = c.prepare_geometry(axis1, axis2, cal)
        result = c.final_scores({"R": np.eye(3), "t": [0, 0, 1]}, g)
        self.assertEqual(result["pixel_msac"], 1.)
        self.assertEqual(result["degenerate_count"], 1)

    def test_pixel_score_matches_explicit_two_pixel_formula(self):
        x1, x2, cal, _, _, pose = fixture()
        x2[:, 1] += np.linspace(.5, 4., len(x2))
        result = c.final_scores(pose, c.prepare_geometry(x1, x2, cal))
        t = pose["t"]
        tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
        F = np.linalg.inv(cal["K2"]).T @ tx @ np.linalg.inv(cal["K1"])
        expected = 0.
        for a, b in zip(x1, x2):
            a, b = np.r_[a, 1.], np.r_[b, 1.]
            l1, l2 = F.T @ b, F @ a
            expected += min(1., float(b @ F @ a)**2 / (float(l1[:2] @ l1[:2])+float(l2[:2] @ l2[:2]))/4.)
        self.assertAlmostEqual(result["pixel_msac"], expected, places=10)

    def test_large_artificial_arrays_do_not_use_accelerate_matmul(self):
        args = fixture(10000)
        g = c.prepare_geometry(args[0], args[1], args[2])
        with np.errstate(all="raise", under="ignore"):
            score = c.final_scores(args[5], g)
            diag = c.depth_diagnostics(args[5], g, args[3], args[4], np.ones(10000, bool))
        self.assertLess(score["pixel_msac"], 1e-18)
        self.assertLess(diag["all"]["fixed_reprojection_cost"], 1e-20)

    def test_invalid_pose_refusal(self):
        for pose in ({"R": np.eye(3), "t": [0, 0, 0]}, {"R": np.diag([1, 1, -1]), "t": [1, 0, 0]},
                     {"R": np.eye(3), "t": [np.nan, 0, 0]}):
            with self.assertRaises(ValueError):
                c.validate_pose(pose)


class DepthTests(unittest.TestCase):
    def test_raw_gauge_invariance_and_wrong_normalization_detected(self):
        x1, x2, cal, d1, d2, pose = fixture()
        g = c.prepare_geometry(x1, x2, cal)
        mask = np.ones(len(x1), bool)
        scaled = {**pose, "t": pose["t"]*7}
        diag = c.depth_diagnostics(scaled, g, d1*7, d2*7, mask)
        self.assertLess(diag["all"]["fixed_reprojection_cost"], 1e-20)
        wrong = {**scaled, "t": scaled["t"]/np.linalg.norm(scaled["t"])}
        wrong_diag = c.depth_diagnostics(wrong, g, d1*7, d2*7, mask)
        self.assertGreater(wrong_diag["all"]["fixed_reprojection_cost"], .001)
        np.testing.assert_array_equal(scaled["t"], pose["t"]*7)

    def test_shift_scale_reconstruction(self):
        x1, x2, cal, d1, d2, pose = fixture()
        pose.update(scale=2., shift1=3., shift2=2.)
        diag = c.depth_diagnostics(pose, c.prepare_geometry(x1, x2, cal), d1-3, d2/2-2, np.ones(len(x1), bool))
        self.assertLess(diag["all"]["fixed_reprojection_cost"], 1e-20)
        self.assertGreater(diag["all"]["nonpositive_counts"]["corrected_source_depth"], -1)

    def test_fixed_denominator_and_empty_mask(self):
        x1, x2, cal, d1, d2, pose = fixture()
        d1[:] = -10
        g = c.prepare_geometry(x1, x2, cal)
        mask = np.zeros(len(x1), bool)
        diag = c.depth_diagnostics(pose, g, d1, d2, mask)
        all_rows = diag["all"]
        self.assertEqual(all_rows["forward"]["cost_sum"], len(x1))
        self.assertEqual(all_rows["forward"]["valid_projection_count"], 0)
        self.assertEqual(all_rows["denominator"], len(x1))
        self.assertGreaterEqual(all_rows["fixed_reprojection_cost"], .5)
        self.assertAlmostEqual(all_rows["fixed_reprojection_cost"]-all_rows["omission_same_denominator_cost"], all_rows["invalid_direction_fraction"])
        self.assertIsNone(diag["returned_mask"]["fixed_reprojection_cost"])
        self.assertEqual(diag["returned_mask"]["status"], "unavailable_empty_support")

    def test_all_and_returned_mask_counts_have_distinct_denominators(self):
        x1, x2, cal, d1, d2, pose = fixture()
        d1[:3] = -10
        mask = np.zeros(len(x1), bool)
        mask[0] = True
        diag = c.depth_diagnostics(pose, c.prepare_geometry(x1, x2, cal), d1, d2, mask)
        self.assertEqual(diag["all"]["nonpositive_counts"]["corrected_source_depth"], 3)
        self.assertEqual(diag["returned_mask"]["nonpositive_counts"]["corrected_source_depth"], 1)
        self.assertEqual(diag["returned_mask"]["denominator"], 1)

    def test_overflow_is_explicit_invalid_direction_not_json_nan(self):
        x1, x2, cal, d1, d2, pose = fixture()
        pose["shift1"] = np.finfo(float).max
        d1[:] = np.finfo(float).max
        diag = c.depth_diagnostics(pose, c.prepare_geometry(x1, x2, cal), d1, d2, np.ones(len(x1), bool))
        self.assertEqual(diag["all"]["nonfinite_counts"]["corrected_source_depth"], len(x1))
        self.assertEqual(diag["all"]["forward"]["nonfinite_transformed_count"], len(x1))
        self.assertEqual(diag["all"]["forward"]["cost_sum"], len(x1))
        json.dumps(c.json_safe(diag), allow_nan=False)

    def test_negative_scale_and_zero_t_remain_diagnostic_inputs(self):
        x1, x2, cal, d1, d2, pose = fixture()
        pose.update(scale=-1., t=np.zeros(3))
        diag = c.depth_diagnostics(pose, c.prepare_geometry(x1, x2, cal), d1, d2, np.ones(len(x1), bool))
        self.assertEqual(diag["all"]["nonpositive_counts"]["scaled_target_depth"], len(x1))
        self.assertEqual(diag["raw_translation_norm"], 0.)

    def test_native_positive_infinity_branch_is_not_valid_projection(self):
        _, _, cal, _, _, _ = fixture(1)
        x1 = np.array([cal["camera1"]["params"][2:]])
        x2 = np.array([cal["camera2"]["params"][2:]])
        pose = {"R": np.eye(3), "t": [0., 0., 1e308], "scale": 1., "shift1": 0., "shift2": 0.}
        result = c.depth_diagnostics(pose, c.prepare_geometry(x1, x2, cal), [1e308], [1.], [True])["all"]
        self.assertEqual(result["native_branch_positive_nonfinite_z_count"], 1)
        self.assertEqual(result["native_branch_inactive_count"], 1)
        self.assertEqual(result["inactive_transformed_direction_count"], 2)
        self.assertEqual(result["fixed_reprojection_cost"], 1.)

    def test_negative_corrected_source_is_not_itself_native_omission(self):
        _, _, cal, _, _, _ = fixture(1)
        x1 = np.array([cal["camera1"]["params"][2:]])
        x2 = np.array([cal["camera2"]["params"][2:]])
        pose = {"R": np.eye(3), "t": [0., 0., 3.], "scale": 1., "shift1": 0., "shift2": 0.}
        result = c.depth_diagnostics(pose, c.prepare_geometry(x1, x2, cal), [-1.], [4.], [True])["all"]
        self.assertEqual(result["nonpositive_counts"]["corrected_source_depth"], 1)
        self.assertEqual(result["native_branch_inactive_count"], 0)
        self.assertEqual(result["fixed_reprojection_cost"], 0.)


class BindingTests(unittest.TestCase):
    def test_options_are_explicit_and_independent(self):
        self.assertEqual(c.options("e_norm")["max_error"], 2.)
        self.assertEqual(c.options("depth_affine")["max_errors"], [16., 2.])
        self.assertTrue(c.options("depth_affine")["estimate_shift"])
        self.assertFalse(c.options("depth_scale")["estimate_shift"])
        opt = c.options("e_ts")
        opt["ransac"]["seed"] = 2
        self.assertEqual(c.options("e_ts")["ransac"]["seed"], 0)

    def test_write_exclusive_and_preserve_on_nonfinite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"saved.json"
            c.write_new(path, {"x": 1})
            with self.assertRaises(FileExistsError):
                c.write_new(path, {"x": 2})
            with path.open("r+") as stream:
                with self.assertRaises(ValueError):
                    c.write_record(stream, {"x": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"x": 1})

    def test_manifest_allocation_and_source_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            here = root/"experiments/0017-depth-information"
            here.mkdir(parents=True)
            asset_protocol = root/"protocols/0017-depth-assets.md"
            asset_protocol.parent.mkdir()
            asset_protocol.write_text("artificial protocol")
            (here/"acquire_and_allocate.py").write_text("# artificial allocator\n")
            with patch.multiple(c, ROOT=root, HERE=here, ASSET_PROTOCOL=asset_protocol):
                manifest = {"schema": "evr0017.development.v1", "selection_salt": c.SALT,
                    "assets_sha256": c.ASSETS_SHA256, "coordinate_convention": "authored_raw_pixels",
                    "development_forever": list(c.SCENES), "scenes": [], "selected_pairs": [],
                    "source_sha256": {str(p.relative_to(root)): c.sha256(p) for p in (here/"acquire_and_allocate.py", asset_protocol)}}
                for scene, size in zip(c.SCENES, c.SCENE_BYTES):
                    keys = ["pair-%02d" % i for i in range(20)]
                    selected = sorted(keys, key=lambda k: (hashlib.sha256((c.SALT+scene+"\0"+k).encode()).hexdigest(), k))[:16]
                    filename = scene+"_roma_moge.h5"
                    manifest["scenes"].append({"scene": scene, "filename": filename,
                        "url": c.BASE_URL+filename, "bytes": size, "sha256": "a"*64,
                        "root_keys": keys, "selected_pair_keys": selected, "pairs_metadata": []})
                    for key in selected:
                        manifest["selected_pairs"].append({"pair_index": len(manifest["selected_pairs"]),
                            "scene": scene, "key": key, "filename": filename, "dataset_sha256": "a"*64, "dataset_bytes": size})
                path = root/"manifest.json"
                path.write_text(json.dumps(manifest))
                self.assertEqual(len(c.load_manifest(path)["selected_pairs"]), 80)
                tampered = copy.deepcopy(manifest)
                tampered["selected_pairs"][0]["key"] = "replacement"
                path.write_text(json.dumps(tampered))
                with self.assertRaises(ValueError):
                    c.load_manifest(path)
                path.write_text(json.dumps(manifest))
                asset_protocol.write_text("changed after allocation")
                with self.assertRaises(ValueError):
                    c.load_manifest(path)

    def test_library_is_bound_without_import_or_native_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            init = folder/"__init__.py"
            binary = folder/"_core.fake"
            init.write_text("# synthetic module\n")
            binary.write_bytes(b"not a real native library")
            backend = SimpleNamespace(__file__=str(init), __version__="3.0.0", _core=SimpleNamespace(__file__=str(binary)))
            expected = c.library_metadata(backend)
            self.assertEqual(c.verify_library(backend, expected), expected)
            binary.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                c.verify_library(backend, expected)


if __name__ == "__main__":
    unittest.main()
