"""Artificial saved-record audits; no real cache, scientific poses, or models.

The complete 24-row file fixture has identity poses and analytically known
100-percent AUC. NPZ-named fixtures are opaque bytes, deliberately not arrays.
"""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

_spec = importlib.util.spec_from_file_location(
    "_evr0028_test_audit", Path(__file__).with_name("audit_saved.py"))
a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a)
c = a.c
COMMIT = "a"*40
I = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
T = [1., 0., 0.]


def put(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, allow_nan=False)+"\n")


def identity_rows():
    rows, truths = [], []
    for i, (scene, query) in enumerate(a.KEYS):
        identity = dict(scene=scene, query=query, anchor=scene+"/A", target=scene+"/"+query)
        rows.append(dict(identity, arms={arm:dict(status="completed", R=copy.deepcopy(I), t=T[:])
                                        for arm in a.ARMS}))
        truths.append(dict(identity, R=copy.deepcopy(I), t=T[:]))
    return rows, truths


def perfect_report(rows, truths):
    """Hand-written exact expectation, independent of auditor calculations."""
    def summary(n):
        return {arm:dict(denominator=n, auc_percent={"5":100., "10":100., "20":100.},
                         status_counts=dict(completed=n, fallback=0, unavailable=0),
                         valid_pose_count=n, genuine_completed=n, penalized_count=0)
                for arm in a.ARMS}
    outputs = []
    for row in rows:
        outputs.append({**{k:row[k] for k in ("scene", "query", "anchor", "target")},
                        "arms":{arm:dict(status="completed", pose_valid=True, penalized=False,
                                         genuine_completed=True, penalty_reason=None,
                                         rotation_degrees=0., translation_degrees=0., max_degrees=0.)
                                for arm in a.ARMS}})
    return dict(
        schema="evr0028.metrics.v1", query_count=24, scene_count=3,
        arm_order=list(a.ARMS), scene_order=list(a.SCENES),
        primary_error="max(rotation_angle,directed_translation_angle)",
        failure_penalty_degrees=180., auc_convention="DMS_left_search_trapezoid_percent",
        manuscript_admission=False, rows=outputs, truth_rows=truths,
        pooled=summary(24), scenes={s:summary(8) for s in a.SCENES},
        gate=dict(passed=False, clauses=dict(trustworthy_comparators=True,
                                            auc5_gain_and_coverage=False,
                                            scene_consistency=False, auc20_noninferiority=True),
                  comparator_valid=True, strongest_control_auc5="E", strongest_control_auc20="E",
                  auc5_margin_pp=0., auc20_margin_pp=0., ang_genuine_completed=24,
                  positive_scene_count=0, scene_auc5_margins_pp={s:0. for s in a.SCENES},
                  all_control_margins_pp={arm:{"5":0., "10":0., "20":0.}
                                          for arm in a.ARMS if arm != "ANG"},
                  manuscript_admission=False))


class FakeSaved:
    def __init__(self, root):
        self.root = root
        self.repo = root/"repo"
        for name in ("0027-affine-tangent", "0025c-rs-noise"):
            p = self.repo/"experiments"/name/"supervise.py"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("# manufactured source\n")
        self.sources = {"artificial.py":"b"*64}
        self.job_path = root/"job.json"
        self.dirs = {k:root/k for k in ("raw_frontend", "predictions", "evaluation")}
        for d in self.dirs.values():
            d.mkdir()
        self.job = dict(expected_commit=COMMIT, **{k:str(v) for k,v in self.dirs.items()})
        for k in self.dirs:
            p = root/(k+"-outer")
            p.mkdir()
            self.job[k+"_outer"] = str(p/"receipt.json")
        for name in ("native_library", "native_receipt"):
            p = root/name
            p.write_bytes(b"artificial non-executable dependency")
            self.job[name], self.job[name+"_sha256"] = str(p), c.sha(p)
        self.job["ground_truth_evaluation_sha256"] = a.HISTORICAL_TRUTH_SHA256
        put(self.job_path, self.job)
        self.header = dict(expected_commit=COMMIT, source_sha256=self.sources,
                           job_sha256=c.sha(self.job_path), status="completed")
        self.front = dict(self.header, schema="evr0028.frontend.v1", ground_truth_access=False,
                          offline_attempts=[], raw_forwards_attempted=27, raw_forwards_completed=27,
                          images=[], selected_images=[], selected_edges=[])
        self.pred = dict(self.header, schema="evr0028.prediction.v1", ground_truth_access=False,
                         offline_attempts=[], images=[], queries=[], comparator_valid=True,
                         old_frontend_binding={"old":"test"}, source_input_files={"test":"f"*64},
                         native_binding={"native":"test"},
                         runtime={"tc_library_sha256":self.job["native_library_sha256"]})
        self.front["input_bindings"] = dict(
            frontend=self.pred["old_frontend_binding"], plan_sha256=c.PLAN_SHA,
            cache_file_sha256=self.pred["source_input_files"], model={"model":"test"})
        self.rows, self.truths = identity_rows()
        for sidx, scene in enumerate(a.SCENES):
            for j in range(9):
                index = 9*sidx+j
                filename = scene+("/A" if j == 0 else "/B"+str(j))
                raw = dict(image_index=index, filename=filename, status="completed",
                           raw=self.blob("raw_frontend", "raw-%03d.npz"%index),
                           processed=self.blob("raw_frontend", "processed-%03d.npz"%index),
                           controls={mode:dict(status="completed") for mode in ("A","B","C")})
                image = dict(image_index=index, filename=filename, status="completed",
                             full=dict(status="completed"), comparator_valid=True,
                             samples_file=self.blob("predictions", "samples-%03d.npz"%index))
                self.front["images"].append(raw)
                self.front["selected_images"].append(filename)
                self.pred["images"].append(image)
                put(self.dirs["raw_frontend"]/("image-%03d.json"%index), raw)
                put(self.dirs["predictions"]/("image-%03d.json"%index), image)
        for i, row in enumerate(self.rows):
            row.update(schema="evr0028.pair.v1", status="completed", phase="complete",
                       pair_index=i, edge_index=i, original_match_indices=[0,1,2,3],
                       selected_match_ids=[[j,j] for j in range(4)], comparator_valid=True,
                       similarity_attempts=[dict(arm="FULL", result=dict(status="completed"))],
                       inputs=self.blob("predictions", "pair-%02d-input.npz"%i))
            edge = dict(scene=row["scene"], source=row["anchor"], target=row["target"], edge_index=i)
            self.front["selected_edges"].append(edge)
            name = "pair-%02d.json"%i
            put(self.dirs["predictions"]/name, row)
            put(self.dirs["predictions"]/("pair-%02d.partial.json"%i),
                dict(progress=dict(status="completed", phase="complete", arms=copy.deepcopy(row["arms"]))))
            item = {k:row[k] for k in ("pair_index","scene","query","anchor","target","edge_index",
                                       "original_match_indices","selected_match_ids")}
            self.pred["queries"].append(dict(item, filename=name, sha256=c.sha(self.dirs["predictions"]/name)))
        self.saved = perfect_report(self.rows, self.truths)
        self.saved.update({k:v for k,v in self.header.items() if k != "status"})
        self.refresh()

    def blob(self, stage, filename):
        path = self.dirs[stage]/filename
        path.write_bytes(b"opaque artificial bytes: "+filename.encode())
        return dict(filename=filename, sha256=c.sha(path), bytes=path.stat().st_size, arrays={})

    def seal(self, stage, schema, receipt, **extra):
        root = self.dirs[stage]
        put(root/"receipt.json", receipt)
        files = {p.name:c.sha(p) for p in root.iterdir() if p.name != "seal.json"}
        seal = dict(self.header, schema="evr0028."+schema+"-seal.v1", query_count=24,
                    image_count=27, receipt_sha256=c.sha(root/"receipt.json"), files=files, **extra)
        put(root/"seal.json", seal)
        outer_path = Path(self.job[stage+"_outer"])
        for name in ("stdout.log","stderr.log"):
            (outer_path.parent/name).write_text("artificial\n")
        script = dict(raw_frontend="frontend.py", predictions="predict.py", evaluation="evaluate.py")[stage]
        outer = dict(
            schema="evr0026.operation.v1", status="completed", exit_code=0, termination=None,
            remaining_descendant_pids=[], cleanup_errors=[], expected_commit=COMMIT,
            hard_timeout_seconds=1800, wall_seconds=.1, group_rss_limit_kib=8388608,
            peak_group_rss_kib=10, thread_environment={k:"4" for k in c.THREAD_KEYS},
            supervisor_sha256=c.sha(self.repo/"experiments/0027-affine-tangent/supervise.py"),
            monitor_sha256=c.sha(self.repo/"experiments/0025c-rs-noise/supervise.py"),
            log_sha256={n:c.sha(outer_path.parent/n) for n in ("stdout.log","stderr.log")},
            output_sha256=c.sha(root/"seal.json"), spec_sha256="d"*64,
            command=["$PYTHON","-W","error","$REPOSITORY/experiments/0028-normal-information/"+script,
                     "--job", c.display(self.job_path.resolve()), "--expected-commit", COMMIT,
                     "--output",c.display(root.resolve())])
        put(outer_path, outer)

    def refresh(self):
        self.seal("raw_frontend", "frontend", self.front, raw_forward_count=27, ground_truth_access=False)
        outer = c.verify_outer(self.job["raw_frontend_outer"], self.dirs["raw_frontend"]/"seal.json",
                               COMMIT, "frontend.py", self.job_path)
        self.pred["raw_frontend_binding"] = dict(outer=outer, seal_sha256=c.sha(self.dirs["raw_frontend"]/"seal.json"))
        self.seal("predictions", "prediction", self.pred, arms=list(a.ARMS),
                  ground_truth_access=False, comparator_valid=self.pred["comparator_valid"])
        binding = dict(
            prediction_outer=c.verify_outer(self.job["predictions_outer"],
                self.dirs["predictions"]/"seal.json", COMMIT, "predict.py", self.job_path),
            prediction_seal_sha256=c.sha(self.dirs["predictions"]/"seal.json"),
            prediction_receipt_sha256=c.sha(self.dirs["predictions"]/"receipt.json"),
            prediction_files=c.read(self.dirs["predictions"]/"seal.json")["files"],
            raw_frontend=self.pred["raw_frontend_binding"],
            raw_frontend_files=c.read(self.dirs["raw_frontend"]/"seal.json")["files"],
            old_frontend=self.pred["old_frontend_binding"],
            source_input_files=self.pred["source_input_files"],
            model=self.front["input_bindings"]["model"], native=self.pred["native_binding"],
            native_library_sha256=self.job["native_library_sha256"],
            native_receipt_sha256=self.job["native_receipt_sha256"])
        self.saved.update(bindings=binding, ground_truth_evaluation_sha256=a.HISTORICAL_TRUTH_SHA256)
        put(self.dirs["evaluation"]/"evaluation.json", self.saved)
        er = dict(self.header, schema="evr0028.evaluation.v1", eligible=True,
                  query_count=24, image_count=27, gate_passed=self.saved["gate"]["passed"],
                  ground_truth_access=True, prediction_bindings=binding,
                  evaluation_sha256=c.sha(self.dirs["evaluation"]/"evaluation.json"))
        self.seal("evaluation", "evaluation", er, ground_truth_access=True)


class AuditSavedTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="evr0028-audit-artificial-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def fixture(self):
        self.addCleanup(patch.stopall)
        patch.object(c, "ROOT", self.root/"repo").start()
        f = FakeSaved(self.root)
        patch.object(c, "verify_source", return_value=f.sources).start()
        patch.object(c, "source_hashes", return_value=f.sources).start()
        return f

    def run_audit(self, f):
        args = SimpleNamespace(job=f.job_path, expected_commit=COMMIT, output=self.root/"audit")
        return a.run(args), c.read(args.output/"receipt.json")

    def test_directed_angles_reject_reflection_and_accept_translation_scale(self):
        truth = dict(R=I, t=T)
        self.assertEqual(a.angles(dict(R=I, t=[-2.,0.,0.]), truth)["translation_degrees"], 180.)
        self.assertEqual(a.angles(dict(R=I, t=[1e200,0.,0.]), truth)["max_degrees"], 0.)
        R = [[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]
        self.assertAlmostEqual(a.angles(dict(R=R,t=T), truth)["rotation_degrees"], 90.)
        with self.assertRaises(ValueError):
            a.pose(dict(R=[[-1.,0.,0.],[0.,1.,0.],[0.,0.,1.]], t=T))

    def test_trapezoid_exact_threshold_and_duplicate_zeros_are_not_step_auc(self):
        self.assertEqual(a.auc([0.,5.,10.,180.]), {"5":25., "10":43.75, "20":62.5})
        self.assertEqual(a.auc([0.,0.]), {"5":100., "10":100., "20":100.})
        self.assertEqual(a.auc([5.,5.])["5"], 0.)

    def test_failures_and_fallbacks_keep_denominator_and_genuine_coverage(self):
        rows, truths = identity_rows()
        rows[0]["arms"]["ANG"] = dict(status="unavailable",R=I,t=T)
        rows[1]["arms"]["ANG"] = dict(status="fallback",R=I,t=[-1.,0.,0.])
        rows[2]["arms"]["ANG"] = dict(status="completed",R=I,t=[0.,0.,0.])
        result = a.recompute(rows, truths, True)
        summary = result["pooled"]["ANG"]
        self.assertEqual((summary["denominator"],summary["genuine_completed"],summary["penalized_count"]), (24,21,2))
        self.assertEqual([r["arms"]["ANG"]["max_degrees"] for r in result["rows"][:3]], [180.]*3)
        self.assertTrue(result["rows"][1]["arms"]["ANG"]["pose_valid"])
        with self.assertRaises(ValueError):
            a.recompute(rows[:-1], truths, True)

    def test_gate_uses_pooled_winner_for_scenes_and_independent_auc20_winner(self):
        rows, truths = identity_rows()
        report = perfect_report(rows, truths)
        pooled, scenes = report["pooled"], report["scenes"]
        for arm in a.ARMS:
            pooled[arm]["auc_percent"] = {"5":50., "10":50., "20":50.}
            for scene in a.SCENES:
                scenes[scene][arm]["auc_percent"]["5"] = 50.
        pooled["ANG"]["auc_percent"] = {"5":54., "10":55., "20":60.}
        pooled["C"]["auc_percent"]["5"] = 52.
        pooled["FULL"]["auc_percent"]["20"] = 60.5
        for scene, delta in zip(a.SCENES, (1.,1.,-2.)):
            scenes[scene]["C"]["auc_percent"]["5"] = 51.
            scenes[scene]["ANG"]["auc_percent"]["5"] = 51.+delta
            scenes[scene]["E"]["auc_percent"]["5"] = 99.
        got = a.gate(pooled, scenes, True)
        self.assertTrue(got["passed"])
        self.assertEqual((got["strongest_control_auc5"],got["strongest_control_auc20"]),("C","FULL"))
        self.assertFalse(a.gate(pooled, scenes, False)["passed"])
        pooled["ANG"]["genuine_completed"] = 19
        self.assertFalse(a.gate(pooled, scenes, True)["clauses"]["auc5_gain_and_coverage"])

    def test_complete_fake_chain_audits_all264_outcomes_without_array_decoding(self):
        f = self.fixture()
        code, report = self.run_audit(f)
        self.assertEqual(code, 0)
        self.assertTrue(report["passed"])
        self.assertEqual((report["pose_outcomes"],report["error_count"]), (264,0))
        self.assertGreaterEqual(report["checks"], 264 * 7)  # Seven audited outcome fields.
        self.assertEqual(len(report["independent"]["rows"]), 24)
        self.assertTrue(all(set(row["arms"]) == set(a.ARMS)
                            for row in report["independent"]["rows"]))
        self.assertFalse(report["independent"]["gate"]["passed"])
        seal = c.read(self.root/"audit"/"seal.json")
        self.assertEqual(seal["files"], {"receipt.json":c.sha(self.root/"audit"/"receipt.json")})

    def test_tampered_bytes_block_before_any_evaluation_truth_decode(self):
        f = self.fixture()
        (f.dirs["predictions"]/"pair-00-input.npz").write_bytes(b"tamper")
        original = c.read
        def guarded(path):
            if Path(path).name == "evaluation.json":
                self.fail("evaluation truth decoded before complete barrier")
            return original(path)
        with patch.object(c, "read", side_effect=guarded), self.assertRaises(ValueError):
            self.run_audit(f)
        report = original(self.root/"audit"/"receipt.json")
        self.assertFalse(report["evaluation_values_read"])
        self.assertEqual(report["status"], "failed")

    def test_unclean_outer_or_wrong_command_job_blocks_saved_truth(self):
        f = self.fixture()
        p = Path(f.job["predictions_outer"])
        receipt = c.read(p)
        receipt["exit_code"] = 1
        put(p, receipt)
        with self.assertRaisesRegex(ValueError, "unclean"):
            a.verify_saved(f.job_path, COMMIT, f.sources)
        receipt["exit_code"] = 0
        receipt["command"][5] = "$PRIVATE/unrelated-job.json"
        put(p, receipt)
        with self.assertRaisesRegex(ValueError, "linkage"):
            a.verify_saved(f.job_path, COMMIT, f.sources)

    def test_resealed_numeric_and_gate_falsification_is_detected(self):
        f = self.fixture()
        f.saved["rows"][0]["arms"]["E"]["translation_degrees"] = 1e-5
        f.saved["pooled"]["ANG"]["auc_percent"]["5"] = 99.
        f.saved["gate"]["passed"] = True
        f.refresh()
        code, report = self.run_audit(f)
        self.assertEqual(code, 1)
        self.assertGreaterEqual(report["error_count"], 3)
        self.assertTrue(any("translation_degrees" in r["field"] for r in report["errors"]))
        self.assertFalse((self.root/"audit"/"seal.json").exists())

    def test_missing_pair_or_unsafe_inventory_is_not_success_subset(self):
        f = self.fixture()
        f.pred["queries"].pop()
        f.refresh()
        with self.assertRaisesRegex(ValueError, "coverage"):
            a.verify_saved(f.job_path, COMMIT, f.sources)
        p = f.dirs["evaluation"]/"seal.json"
        seal = c.read(p)
        seal["files"]["../outside"] = "f"*64
        put(p, seal)
        with self.assertRaises(ValueError):
            a.inventory(f.dirs["evaluation"], "evaluation", f.job_path, COMMIT, f.sources)

    def test_changed_fallback_E_and_full_validity_cannot_hide_in_seal(self):
        f = self.fixture()
        row = f.rows[0]
        row["arms"]["ANG"] = dict(status="fallback", R=I, t=[-1.,0.,0.])
        put(f.dirs["predictions"]/"pair-00.json", row)
        f.pred["queries"][0]["sha256"] = c.sha(f.dirs["predictions"]/"pair-00.json")
        f.refresh()
        with self.assertRaisesRegex(ValueError, "fallback changed E"):
            a.verify_saved(f.job_path, COMMIT, f.sources)
        row["arms"]["ANG"] = dict(status="completed",R=I,t=T)
        row["comparator_valid"] = False
        put(f.dirs["predictions"]/"pair-00.json", row)
        f.pred["queries"][0]["sha256"] = c.sha(f.dirs["predictions"]/"pair-00.json")
        f.refresh()
        with self.assertRaisesRegex(ValueError, "comparator validity"):
            a.verify_saved(f.job_path, COMMIT, f.sources)

    def test_source_mismatch_precedes_saved_reads(self):
        with patch.object(c, "verify_source", side_effect=ValueError("source changed")), \
                patch.object(c, "read", side_effect=AssertionError("saved read")), \
                self.assertRaisesRegex(ValueError, "source changed"):
            a.run(SimpleNamespace(expected_commit=COMMIT, job=self.root/"absent", output=self.root/"audit"))


if __name__ == "__main__":
    unittest.main()
