"""Manufactured admission/ordering regressions; no real media, GT or model."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent


def load(name, path):
    if not path.exists():
        path = HERE.parent / path.name
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


r = load("evaluation_operation_under_test", HERE / "run_evaluation.py")
helpers = load("manufactured_pure_evaluator_fixtures", HERE / "test_evaluate_queries.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


class Helper:
    sha = staticmethod(sha)
    read_json = staticmethod(lambda path: json.loads(Path(path).read_text()))

    @staticmethod
    def require(condition, message):
        if not condition:
            raise ValueError(message)


class EvaluationOperationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.h = Helper()

    def video_cost_context(self):
        spec, outer = self.base / "spec.json", self.base / "outer/receipt.json"
        write(spec, {"operation_output": str(outer.parent)})
        write(outer, {"wall_seconds": 12.5, "peak_group_rss_kib": 1234})
        # The real outer_contract returns binding hashes, not resource fields.
        binding = {"outer_sha256": sha(outer), "spec_sha256": sha(spec),
                   "receipt_sha256": "a" * 64, "file_sha256": {}}
        return {"videos": [{"spec": spec, "sequence": "manufactured",
                            "outer": binding,
                            "record": {"native_video": {"calls": {"baseline_frames": 25}}}}]}, outer

    def test_complete_costs_use_bound_raw_outer_not_binding_resource_fields(self):
        context, _ = self.video_cost_context()
        self.assertEqual(r.complete_video_costs(self.h, context), [{
            "sequence": "manufactured", "wall_seconds": 12.5,
            "peak_group_rss_kib": 1234, "calls": {"baseline_frames": 25}}])

    def test_complete_costs_reject_changed_raw_outer_before_reporting(self):
        context, outer = self.video_cost_context()
        write(outer, {"wall_seconds": 0.1, "peak_group_rss_kib": 1})
        with self.assertRaisesRegex(ValueError, "outer receipt changed"):
            r.complete_video_costs(self.h, context)

    def test_new_gt_outer_is_required_before_any_gt_receipt_read(self):
        root = self.base / "repo"
        old = SimpleNamespace(outer_contract=mock.Mock(side_effect=AssertionError("legacy outer used")))
        new = mock.Mock(side_effect=ValueError("unclean replication operation"))
        context = {"installation": self.base / "installation", "new_outer_contract": new}
        with mock.patch.object(self.h, "read_json", side_effect=AssertionError("GT receipt read")):
            with self.assertRaisesRegex(ValueError, "unclean replication operation"):
                r.gt_contract(self.h, old, root, "a" * 40, self.base / "proof.json",
                              self.base / "gt", self.base / "gt-spec.json", {}, context)
        new.assert_called_once()
        old.outer_contract.assert_not_called()

    def array_context(self, nonfinite_last=False):
        rows, _ = helpers.fixture()
        sources, directories = [], []
        for i, row in enumerate(rows):
            names = list(row["candidates"])
            array = np.stack([np.where(row["candidates"][n], 1., -1.) for n in names])[None]
            if nonfinite_last and i == 47:
                array[0, -1, -1, -1] = np.nan
            path = self.base / ("manufactured-%02d.npz" % i)
            np.savez(path, logits=array)
            descriptor = {"filename": path.name, "bytes": path.stat().st_size,
                "sha256": sha(path), "arrays": {"logits": {
                    "shape": list(array.shape), "dtype": str(array.dtype),
                    "raw_c_order_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}}}
            source = copy.deepcopy(row)
            source["candidates"] = {name: {"output": descriptor, "array_key": "logits",
                                           "channel": j} for j, name in enumerate(names)}
            sources.append(source)
            directories.append(self.base)
        return {"rows": sources, "query_prediction_dirs": directories}

    def test_all48_numeric_archives_and_choices_preflight(self):
        rows = r.load_predictions(self.h, self.array_context(), np)
        self.assertEqual(len(rows), 48)
        for row in rows:
            helpers.e.validate_completed(row, helpers.core)
        self.assertEqual(rows[-1]["candidates"]["R"].shape, (4, 4))

    def test_finite_first47_cannot_hide_last_archive_nan(self):
        with self.assertRaisesRegex(ValueError, "numeric array changed"):
            r.load_predictions(self.h, self.array_context(nonfinite_last=True), np)

    def test_last_choice_rejection_precedes_any_gt_decoder(self):
        rows, _ = helpers.fixture()
        rows[-1]["selection"]["choices"]["signed_deletion"] = "M2"
        root, installation = self.base / "repo", self.base / "installation"
        runtime = installation / "venv/lib/python3.12/site-packages"
        self.h.frozen = lambda *args: {
            r.OWN + "run_evaluation.py": sha(HERE / "run_evaluation.py")}
        context = {"h": self.h, "old_admission": SimpleNamespace(),
                   "installation": installation, "sources": {}}
        seals = SimpleNamespace(verify_queries=mock.Mock(return_value=({}, context)),
                                require_runtime=mock.Mock())
        guard = SimpleNamespace(install_guard=mock.Mock())
        image = SimpleNamespace(__file__=str(runtime / "PIL/Image.py"),
                                open=mock.Mock(side_effect=AssertionError("GT decoder reached")))
        dispatch = {"seal_queries.py": seals, "qualify.py": guard,
                    "diagnostic_core.py": helpers.core, "evaluate_queries.py": helpers.e}
        args = SimpleNamespace(repo_root=str(root), proof=str(self.base / "proof.json"),
            output_dir=str(self.base / "evaluation"), expected_commit="a" * 40,
            gt_root=str(self.base / "gt"), gt_spec=str(self.base / "gt-spec.json"))
        masks = {key: self.base / ("unused-%d.png" % i)
                 for i, key in enumerate(helpers.core.QUERY_IDENTITIES)}
        with mock.patch.object(r, "module", side_effect=lambda name, path: dispatch[Path(path).name]), \
             mock.patch.object(r, "gt_contract", return_value=(masks, {"file_sha256": {}})), \
             mock.patch.object(r, "load_predictions", return_value=rows), \
             mock.patch.dict(sys.modules, {"site": SimpleNamespace(main=mock.Mock()),
                                           "PIL": SimpleNamespace(Image=image)}), \
             mock.patch.object(np, "__file__", str(runtime / "numpy/__init__.py")), \
             mock.patch.object(sys, "prefix", str(installation / "venv")), \
             mock.patch.object(sys, "dont_write_bytecode", True), \
             mock.patch("sys.stdout", new=io.StringIO()):
            with self.assertRaisesRegex(ValueError, "escaped"):
                r.run(args)
        image.open.assert_not_called()
        record = json.loads((Path(args.output_dir) / "receipt.json").read_text())
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["query_masks_decoded"], 0)
        self.assertEqual(record["model_calls"], 0)
        self.assertNotIn("prediction_numeric_preflight_complete", record)


if __name__ == "__main__":
    unittest.main()
