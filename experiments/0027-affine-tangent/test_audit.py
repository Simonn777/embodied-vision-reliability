"""Prepared fake-artifact/formula checks; no fixture, solver or native imports."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import audit_saved as audit


def observation():
    return dict(K=[[500., 0., 319.5], [0., 500., 239.5], [0., 0., 1.]],
        xy1=[[319.5, 239.5]]*9, xy2=[[369.5, 239.5]]*9, d1=[1.]*9, d2=[1.]*9,
        center_index=4, derivatives=dict(status="unavailable", mode="center_fixed_ols",
            A=None, g1=None, g2=None, fit=None, elapsed_seconds=0., reason="fake_rank_failure"))


def candidate():
    return dict(R=np.eye(3).tolist(), t=[.1, 0., 0.], a=1., b=0., c=0.)


def provenance():
    records = []
    for case in ("G", "M"):
        for calibration in ("scale_only", "shifted"):
            for mode in ("exact", "fitted"):
                records.append(dict(case=case, calibration=calibration, derivative_mode=mode,
                                    regime="clean", realization=None))
            for regime in ("L", "H"):
                for i in range(16):
                    records.append(dict(case=case, calibration=calibration, derivative_mode="fitted",
                                        regime=regime, realization=i))
    records += [dict(case=c, calibration="shifted", derivative_mode="exact",
                     regime="clean", realization=None) for c in ("P", "F")]
    return records


def fake_package(folder, eligible=False):
    folder = Path(folder)
    source = folder/"source"
    source.mkdir()
    sources = {}
    for name in ("audit_saved.py", "evaluate.py", "pipeline.py", "fixtures.py",
                 "study_core.py", "controls.py", "tangent_solver.py", "protocol.md"):
        data = ("# invented test source "+name+"\n").encode()
        (source/name).write_bytes(data)
        sources[name] = audit.digest(data)
    def write(name, value):
        path = folder/name
        path.write_bytes(audit.canonical(value)+b"\n")
        return path
    def jsonl(name, values):
        path = folder/name
        path.write_bytes(b"".join(audit.canonical(v)+b"\n" for v in values))
        return path
    obs = observation()
    inputs, predictions, truths, evaluation_rows, metric_rows = [], [], [], [], []
    for i, p in enumerate(provenance()):
        inputs.append(dict(row_id=i, observation=copy.deepcopy(obs)))
        banks = {}
        for arm, names in {
            "T": ["quartic"], "S": ["svd", "ordered_frame"],
            "V": ["epsilon_"+str(v) for v in (.25, 1., 4., 16.)],
            "D8": ["triangle_"+str(v) for v in range(8)]}.items():
            banks[arm] = dict(status="unavailable", candidates=[], selected_index=None, selected=None,
                attempts=[dict(name=n, status="unavailable", solver_called=arm == "D8",
                    native_called=arm == "D8", elapsed_seconds=0., scoring_seconds=0.) for n in names],
                timings=dict(solver_seconds=0., scoring_seconds=0., total_seconds=0.))
        if eligible:
            raw = candidate()
            normalized = dict(raw, generation_index=0, pose_usable=True, score=0.,
                status="completed", reason=None, forward=dict(terms=[0.]*9), backward=dict(terms=[0.]*9))
            banks["D8"].update(status="completed", candidates=[normalized],
                               selected_index=0, selected=copy.deepcopy(normalized))
        predictions.append(dict(row_id=i, input_sha256=audit.digest(audit.canonical(obs)),
            result=dict(schema="evr0027.study-row.v1", status="completed", arms=banks,
                        call_counts=dict(native_attempted=12, native_executed=8))))
        truths.append(dict(row_id=i, truth=dict(R=np.eye(3).tolist(), t=[1., 0., 0.]), provenance=p))
        errors = {a: dict(rotation_error=180., translation_error=180., task_error=180.)
                  for a in audit.ARMS}
        if eligible:
            errors["D8"] = dict(rotation_error=0., translation_error=0., task_error=0.)
        metric_rows.append(dict(row_id=i, provenance=p, arms=copy.deepcopy(errors)))
        evaluation_rows.append(dict(row_id=i, provenance=p, arms={
            a: dict(errors[a], selected_index=0 if eligible and a == "D8" else None,
                unavailable=not (eligible and a == "D8"), useful=eligible and a == "D8",
                candidate_count=int(eligible and a == "D8"), eligible_count=int(eligible and a == "D8"),
                oracle=dict(label="oracle_upper_bound_never_used_for_gates",
                    candidate_index=0 if eligible and a == "D8" else None,
                    task_error=0. if eligible and a == "D8" else 180.)) for a in audit.ARMS}))
    paths = dict(inputs=jsonl("inputs.jsonl", inputs), predictions=jsonl("predictions.jsonl", predictions),
                 truth=jsonl("truth.jsonl", truths))
    hashes = {k: audit.digest(p.read_bytes()) for k, p in paths.items()}
    commit = "a"*40
    job = dict(schema="evr0027.prediction-job.v1", source_sha256=sources,
               expected_commit=commit, native_sha256=audit.NATIVE_SHA,
               inputs_sha256=hashes["inputs"], truth_sha256=hashes["truth"])
    jobpath = write("job.json", job)
    jobhash = audit.digest(jobpath.read_bytes())
    seal = dict(schema="evr0027.seal.v1", status="completed", row_count=138,
        expected_commit=commit, source_sha256=sources, job_sha256=jobhash,
        **{k+"_sha256": v for k, v in hashes.items()})
    paths["seal"] = write("seal.json", seal)
    hashes["seal"] = audit.digest(paths["seal"].read_bytes())
    summary = audit.aggregate(metric_rows)
    evaluation = dict(schema="evr0027.evaluation.v1", status="completed", row_count=138,
        bindings=dict(hashes, job=jobhash, source_sha256=sources, expected_commit=commit),
        rows=evaluation_rows, all_rows=summary["all_rows"], noise_rows=summary["noise_rows"],
        groups=summary["groups"], gates=dict(pass_=False))
    evaluation["gates"] = {"pass": False, "clauses": {
        k: {"pass": v} for k, v in summary["clause_pass"].items()}}
    paths["evaluation"] = write("evaluation.json", evaluation)
    write("stdout.log", seal)
    (folder/"stderr.log").write_bytes(b"")
    receipt = dict(schema="evr0026.operation.v1", status="completed", exit_code=0,
        termination=None, remaining_descendant_pids=[], cleanup_errors=[],
        expected_commit=commit, output_sha256=hashes["seal"],
        spec_sha256="b"*64, supervisor_sha256="c"*64, monitor_sha256="d"*64,
        hard_timeout_seconds=30, wall_seconds=.1, group_rss_limit_kib=8388608,
        peak_group_rss_kib=100, thread_environment={"OMP_NUM_THREADS": "4"},
        command=["python", "/fake/pipeline.py", "predict", "--expected-commit", commit],
        log_sha256={n: audit.digest((folder/n).read_bytes()) for n in ("stdout.log", "stderr.log")})
    paths["prediction_receipt"] = write("receipt.json", receipt)
    return source, paths


class AuditTests(unittest.TestCase):
    def test_directed_angles_and_tiny_norm(self):
        estimate = dict(R=np.eye(3).tolist(), t=[1e-300, 0., 0.])
        truth = dict(R=np.eye(3).tolist(), t=[-1., 0., 0.])
        self.assertEqual(audit.pose_errors(estimate, truth)["translation_error"], 180.)
        truth["t"] = [1., 0., 0.]
        self.assertEqual(audit.pose_errors(estimate, truth)["task_error"], 0.)

    def test_pixel_score_exact_and_invalid_projection_mass(self):
        self.assertEqual(audit.pixel_score(candidate(), observation())[0], 0.)
        obs = observation()
        obs["xy2"] = copy.deepcopy(obs["xy1"])
        c = candidate()
        c["t"] = [0., 0., -2.]
        score, terms = audit.pixel_score(c, obs)
        self.assertEqual(terms[:9], [1.]*9)
        self.assertEqual(terms[9:], [0.]*9)
        self.assertEqual(score, .5)
        c["b"] = -1.
        with self.assertRaisesRegex(ValueError, "positivity"):
            audit.pixel_score(c, obs)

    def test_step_auc_fixed_full_denominator(self):
        result = audit.stats([0., 2.5, 5., 180.])
        self.assertEqual(result["step_cdf_auc5"], 37.5)
        self.assertEqual(result["useful_count"], 3)

    def test_all_four_gate_conditions(self):
        result = []
        for i, p in enumerate(provenance()):
            errors = {}
            for arm in audit.ARMS:
                if p["regime"] == "clean":
                    error = 0.
                else:
                    count = 16 if p["calibration"] == "scale_only" else (
                        12 if arm == "T" else 10)
                    error = 0. if p["realization"] < count else 180.
                errors[arm] = dict(task_error=error)
            result.append(dict(row_id=i, provenance=p, arms=errors))
        self.assertTrue(audit.aggregate(result)["gate_pass"])
        # One L group with 11 useful T poses fails the necessary absolute clause.
        row = next(r for r in result if r["provenance"] == dict(
            case="G", calibration="shifted", derivative_mode="fitted", regime="L", realization=11))
        row["arms"]["T"]["task_error"] = 180.
        self.assertFalse(audit.aggregate(result)["clause_pass"]["2_shifted_yield"])

    def test_complete_fake_package_all_unavailable_stays_in_denominator(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, paths = fake_package(temporary)
            result = audit.audit(paths, source_root=source)
            self.assertTrue(result["passed"], result.get("failure"))
            self.assertEqual(result["arm_rows"], 552)
            self.assertFalse(result["scientific_gate_pass"])
            self.assertEqual(result["summaries"]["all_rows"]["T"]["count"], 138)
            self.assertEqual(result["summaries"]["all_rows"]["T"]["step_cdf_auc5"], 0.)

    def test_complete_fake_eligible_score_selection_and_oracle(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, paths = fake_package(temporary, eligible=True)
            result = audit.audit(paths, source_root=source)
            self.assertTrue(result["passed"], result.get("failure"))
            self.assertEqual(result["eligible_candidates_checked"], 138)
            self.assertEqual(result["summaries"]["all_rows"]["D8"]["step_cdf_auc5"], 100.)
            first = result["recomputed_rows"][0]["arms"]["D8"]
            self.assertEqual(first["independent_oracle_index"], 0)
            self.assertEqual(first["candidate_pose_errors"][0]["task_error"], 0.)

    def test_bad_outer_and_source_block_before_truth(self):
        for mode in ("outer", "source"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                source, paths = fake_package(temporary)
                if mode == "outer":
                    r = json.loads(paths["prediction_receipt"].read_text())
                    r["exit_code"] = 1
                    paths["prediction_receipt"].write_bytes(audit.canonical(r))
                else:
                    (source/"pipeline.py").write_text("changed\n")
                with patch.object(audit, "decode_truth", side_effect=AssertionError("truth reached")) as decoder:
                    result = audit.audit(paths, source_root=source)
                self.assertFalse(result["passed"])
                decoder.assert_not_called()

    def test_saved_metric_mismatch_is_retained_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, paths = fake_package(temporary)
            saved = json.loads(paths["evaluation"].read_text())
            saved["all_rows"]["T"]["step_cdf_auc5"] = 1.
            paths["evaluation"].write_bytes(audit.canonical(saved))
            result = audit.audit(paths, source_root=source)
            self.assertFalse(result["passed"])
            self.assertTrue(any("step_cdf_auc5" in x["check"] for x in result["failures"]))

    def test_cli_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/"preserved.json"
            output.write_text("original")
            args = [item for name in ("inputs", "predictions", "truth", "seal", "evaluation",
                                     "prediction-receipt")
                    for item in ("--"+name, str(Path(temporary)/(name+".json")))]
            with self.assertRaises(FileExistsError):
                audit.main(args+["--output", str(output)])
            self.assertEqual(output.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
