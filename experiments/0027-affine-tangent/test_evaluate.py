"""Artificial saved-record tests only; no manufactured science generator imports."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SOURCE = Path(__file__).with_name("evaluate.py")
SPEC = importlib.util.spec_from_file_location("_evr0027_evaluate_test_target", SOURCE)
e = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e)
I = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]


def candidate(angle=0., score=.01, index=0):
    theta = math.radians(angle)
    return {"R": copy.deepcopy(I), "t": [math.cos(theta), math.sin(theta), 0.],
            "a": 1., "b": 0., "c": 0., "pose_usable": True,
            "status": "completed", "reason": None, "score": score,
            "generation_index": index, "solver_candidate": {"artificial": True}}


def arm(angle=0.):
    c = candidate(angle)
    return {"candidates": [c], "selected_index": 0, "selected": copy.deepcopy(c),
            "attempts": [], "status": "completed",
            "timings": {"solver_seconds": 0., "scoring_seconds": 0., "total_seconds": 0.}}


def observation(mode):
    xy = [[319.5+x, 239.5+y] for y in (-16., 0., 16.) for x in (-16., 0., 16.)]
    return {"K": [[500., 0., 319.5], [0., 500., 239.5], [0., 0., 1.]],
            "xy1": xy, "xy2": copy.deepcopy(xy), "d1": [1.]*9, "d2": [1.]*9,
            "center_index": 4,
            "derivatives": {"status": "completed",
                "mode": "analytic" if mode == "exact" else "center_fixed_ols",
                "A": [[1., 0.], [0., 1.]], "g1": [.001, .002], "g2": [.001, .002],
                "fit": None if mode == "exact" else {"test_receipt": True},
                "elapsed_seconds": 0., "reason": None}}


class FakeCohort:
    """138 schema records with invented constant arrays, not physical fixtures."""
    def __init__(self, root):
        self.root = Path(root)
        self.sources = self.root/"sources"
        self.sources.mkdir()
        (self.sources/"evaluate.py").write_bytes(SOURCE.read_bytes())
        (self.sources/"protocol.md").write_text("Artificial binding fixture only.\n")
        self.source_hashes = {p.name: e.sha256(p.read_bytes()) for p in self.sources.iterdir()}
        self.commit = "a"*40
        (self.root/"job.json").write_bytes(e.canonical({
            "expected_commit": self.commit, "source_sha256": self.source_hashes,
            "artificial": True}))
        self.inputs, self.predictions, self.truths = [], [], []
        groups = []
        for case in ("G", "M"):
            for calibration in ("scale_only", "shifted"):
                for mode in ("exact", "fitted"):
                    groups.append((case, calibration, mode, "clean", None))
        for case in ("G", "M"):
            for calibration in ("scale_only", "shifted"):
                for realization in range(16):
                    for regime in ("L", "H"):
                        groups.append((case, calibration, "fitted", regime, realization))
        groups += [("P", "shifted", "exact", "clean", None),
                   ("F", "shifted", "exact", "clean", None)]
        for row_id, (case, calibration, mode, regime, realization) in enumerate(groups):
            obs = observation(mode)
            arms = {name: arm() for name in ("T", "S", "V", "D8")}
            if calibration == "shifted" and regime == "H":
                for name, cutoff in (("S", 12), ("V", 14), ("D8", 14)):
                    if realization >= cutoff:
                        arms[name] = arm(10.)
            self.inputs.append({"row_id": row_id, "observation": obs})
            self.predictions.append({"row_id": row_id,
                "input_sha256": e.sha256(e.canonical(obs)),
                "result": {"schema": "evr0027.study-row.v1", "status": "completed",
                    "arms": arms, "common_timing": {
                        "derivative_fit_seconds": 0., "validation_seconds": 0.,
                        "process_seconds": 0., "total_with_derivatives_seconds": 0.},
                    "call_counts": {"native_attempted": 12, "native_executed": 12},
                    "warnings": []}})
            truth = {"R": copy.deepcopy(I), "t": [1., 0., 0.],
                     "a": 1., "b": 0., "c": 0.}
            if case == "F":
                truth["calibration_family"] = [{"k": .8}, {"k": 1.}, {"k": 1.2}]
            self.truths.append({"row_id": row_id, "truth": truth,
                "provenance": {"case": case, "calibration": calibration,
                    "derivative_mode": mode, "regime": regime, "realization": realization}})
        self.paths = {name: self.root/(name+".jsonl") for name in ("inputs", "predictions", "truth")}
        self.paths["seal"] = self.root/"seal.json"
        self.save()

    def save(self, rebind_observations=True):
        # Artificial process receipts exercise the exact public completion
        # contract. They never invoke an algorithm or generate physical points.
        names = {"T": ["quartic"], "S": ["svd", "ordered_frame"],
                 "V": ["epsilon_0.25", "epsilon_1.0", "epsilon_4.0", "epsilon_16.0"],
                 "D8": ["triangle_"+str(i) for i in range(8)]}
        for pred in self.predictions:
            for name, record in pred["result"]["arms"].items():
                if not record["attempts"]:
                    record["attempts"] = [{"name": title, "status": "completed",
                        "solver_called": True, "native_called": name in ("V", "D8"),
                        "elapsed_seconds": 0., "scoring_seconds": 0., "result": {}}
                        for title in names[name]]
        if rebind_observations:
            for inp, pred in zip(self.inputs, self.predictions):
                pred["input_sha256"] = e.sha256(e.canonical(inp["observation"]))
        for name, rows in (("inputs", self.inputs), ("predictions", self.predictions),
                           ("truth", self.truths)):
            self.paths[name].write_bytes(b"".join(e.canonical(row)+b"\n" for row in rows))
        seal = {"schema": "evr0027.seal.v1", "status": "completed", "row_count": 138,
                "source_sha256": self.source_hashes, "expected_commit": self.commit,
                "job_sha256": e.sha256((self.root/"job.json").read_bytes())}
        seal.update({name+"_sha256": e.sha256(self.paths[name].read_bytes())
                     for name in ("inputs", "predictions", "truth")})
        self.paths["seal"].write_bytes(e.canonical(seal))

    def evaluate(self):
        return e.evaluate_saved(self.paths["predictions"], self.paths["inputs"],
                                self.paths["truth"], self.paths["seal"],
                                _source_root=self.sources)


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.c = FakeCohort(self.temp.name)

    def assert_blocked_before_truth(self):
        with mock.patch.object(e, "decode_truth", side_effect=AssertionError("GT barrier crossed")) as decoder:
            with self.assertRaises(e.EvaluationError):
                self.c.evaluate()
            decoder.assert_not_called()

    def test_complete_fake_cohort_gate_and_independent_utility(self):
        report = self.c.evaluate()
        self.assertEqual(report["row_count"], 138)
        self.assertEqual(report["noise_rows"]["T"]["count"], 128)
        self.assertTrue(report["gates"]["pass"])
        self.assertEqual(report["groups"]["G/shifted/H"]["S"]["useful_count"], 12)
        self.assertEqual(report["groups"]["G/shifted/H"]["D8"]["useful_count"], 14)
        self.assertEqual(e.summary([0., 2.5, 5., 180.])["step_cdf_auc5"], 37.5)
        self.assertEqual(e.summary([0., 2.5, 5., 180.])["useful_count"], 3)

    def test_source_and_job_tamper_prevent_truth_decode(self):
        source = self.c.sources/"protocol.md"
        source.write_bytes(source.read_bytes()+b"tamper")
        self.assert_blocked_before_truth()
        self.c.source_hashes["protocol.md"] = e.sha256(source.read_bytes())
        self.c.save()
        # Updating the seal's source map cannot bypass the still-bound job map.
        self.assert_blocked_before_truth()

    def test_raw_artifact_tamper_prevents_truth_decode(self):
        for name in ("inputs", "predictions", "truth"):
            with self.subTest(name=name):
                original = self.c.paths[name].read_bytes()
                self.c.paths[name].write_bytes(original+b" ")
                self.assert_blocked_before_truth()
                self.c.paths[name].write_bytes(original)

    def test_canonical_observation_binding_and_forbidden_nested_truth(self):
        self.c.predictions[0]["input_sha256"] = "0"*64
        self.c.save(rebind_observations=False)
        self.assert_blocked_before_truth()
        self.c.inputs[0]["observation"]["derivatives"]["fit"] = {"truth": {"R": I}}
        self.c.save()
        self.assert_blocked_before_truth()

    def test_coverage_order_and_incomplete_arm_prevent_truth_decode(self):
        original = copy.deepcopy(self.c.predictions)
        self.c.predictions.pop()
        self.c.save()
        self.assert_blocked_before_truth()
        self.c.predictions = copy.deepcopy(original)
        self.c.predictions[0]["result"]["arms"]["D8"]["attempts"].pop()
        self.c.save()
        self.assert_blocked_before_truth()
        self.c.predictions = copy.deepcopy(original)
        self.c.predictions[0], self.c.predictions[1] = self.c.predictions[1], self.c.predictions[0]
        self.c.save()
        self.assert_blocked_before_truth()
        self.c.predictions = copy.deepcopy(original)
        del self.c.predictions[0]["result"]["arms"]["D8"]
        self.c.save()
        self.assert_blocked_before_truth()

    def test_selector_recomputed_including_exact_ties(self):
        row = self.c.predictions[0]["result"]["arms"]["T"]
        row["candidates"] = [candidate(10., .01, 0), candidate(0., .01, 1)]
        row["selected_index"], row["selected"] = 1, copy.deepcopy(row["candidates"][1])
        self.c.save()
        self.assert_blocked_before_truth()
        row["selected_index"], row["selected"] = 0, copy.deepcopy(row["candidates"][0])
        self.c.save()
        report = self.c.evaluate()
        self.assertGreater(report["rows"][0]["arms"]["T"]["task_error"], 5.)
        self.assertEqual(report["rows"][0]["arms"]["T"]["oracle"]["task_error"], 0.)
        self.assertFalse(report["gates"]["clauses"]["1_clean_exact"]["pass"])
        self.assertFalse(report["gates"]["pass"])

    def test_antipodal_translation_unavailable_and_oracle_cannot_rescue(self):
        truth = {"R": I, "t": [1., 0., 0.]}
        error = e.pose_error({"R": I, "t": [-1., 0., 0.]}, truth)
        self.assertEqual(error["translation_error"], 180.)
        row = self.c.predictions[0]["result"]["arms"]["T"]
        row.update(candidates=[], selected_index=None, selected=None, status="unavailable")
        self.c.save()
        report = self.c.evaluate()
        self.assertEqual(report["rows"][0]["arms"]["T"]["task_error"], 180.)
        self.assertEqual(report["rows"][0]["arms"]["T"]["oracle"]["task_error"], 180.)
        self.assertEqual(report["all_rows"]["T"]["count"], 138)
        self.assertFalse(report["gates"]["pass"])

    def test_eligible_full_pool_depth_and_rotation_integrity(self):
        row = self.c.predictions[0]["result"]["arms"]["T"]
        row["candidates"][0]["b"] = -2.
        row["selected"] = copy.deepcopy(row["candidates"][0])
        self.c.save()
        self.assert_blocked_before_truth()
        row["candidates"][0]["b"] = 0.
        row["candidates"][0]["R"][0][0] = -1.
        row["selected"] = copy.deepcopy(row["candidates"][0])
        self.c.save()
        self.assert_blocked_before_truth()

    def test_fixed_provenance_and_boundary_gate_are_not_pooled(self):
        selected = [r["row_id"] for r in self.c.truths if (
            r["provenance"]["case"], r["provenance"]["calibration"],
            r["provenance"]["regime"]) == ("G", "shifted", "L")]
        for index in selected[:5]:
            self.c.predictions[index]["result"]["arms"]["T"] = arm(10.)
        self.c.save()
        report = self.c.evaluate()
        self.assertEqual(report["groups"]["G/shifted/L"]["T"]["useful_count"], 11)
        self.assertFalse(report["gates"]["clauses"]["2_shifted_yield"]["pass"])
        self.c.truths[-1]["provenance"] = copy.deepcopy(self.c.truths[-2]["provenance"])
        self.c.save()
        with self.assertRaisesRegex(e.EvaluationError, "provenance"):
            self.c.evaluate()

    def test_nonfinite_duplicate_keys_and_output_exclusivity(self):
        for raw in (b'{"x":NaN}', b'{"x":1e999}', b'{"x":1,"x":2}'):
            with self.subTest(raw=raw), self.assertRaises(e.EvaluationError):
                e.decode_json(raw)
        output = Path(self.temp.name)/"evaluation.json"
        args = []
        for name in ("predictions", "inputs", "truth", "seal"):
            args += ["--"+name, str(self.c.paths[name])]
        args += ["--output", str(output)]
        with mock.patch.object(e, "evaluate_saved", side_effect=RuntimeError("artificial failure")):
            with self.assertRaisesRegex(RuntimeError, "artificial"):
                e.main(args)
        saved = output.read_bytes()
        self.assertEqual(json.loads(saved)["status"], "failed")
        with self.assertRaises(FileExistsError):
            e.main(args)
        self.assertEqual(output.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
