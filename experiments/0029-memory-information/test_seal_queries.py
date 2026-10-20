"""Artificial contracts only: inert files, fake native admission and transport.

The real six-video sealer, output-inventory checks and acquisition callback
ordering run. Previously qualified native/evidence chains and interpreter
identity are injected; no model, NumPy, media decoder or socket is imported.
"""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
COMMIT = "a" * 40


def load(name, filename):
    path = HERE / filename
    if not path.exists():
        path = HERE.parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


s = load("evr0029_seal_under_test", "seal_queries.py")
g = load("evr0029_gt_under_test", "acquire_query_gt.py")
v = load("evr0029_video_contract_under_test", "run_video.py")
old = load("evr0029_outer_contract_under_test", "run_evidence.py")
acq = load("evr0029_acquisition_callback_under_test", "acquire_davis.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class Helper:
    THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
    sha = staticmethod(sha)
    read_json = staticmethod(lambda p: json.loads(Path(p).read_text()))
    inventory = staticmethod(lambda _: {"artificial_runtime": "unchanged"})
    @staticmethod
    def require(value, message):
        if not value:
            raise ValueError(message)
    @staticmethod
    def number(value, low, high):
        return type(value) in (int, float) and math.isfinite(value) and low <= value <= high
    @staticmethod
    def git_bytes(root, commit, relative):
        if commit not in (COMMIT, v.QUALIFICATION_COMMIT):
            raise ValueError("fake commit mismatch")
        return (root / relative).read_bytes()
    @staticmethod
    def frozen(root, commit, names):
        if commit not in (COMMIT, v.QUALIFICATION_COMMIT):
            raise ValueError("fake source freeze mismatch")
        return {name: sha(root / name) for name in names}


def expected_selection(sequence, frame, actionable, *, tied=False):
    digest = hashlib.sha256(("\0".join((s.RANDOM_SALT, sequence, "1", str(frame)))).encode()).hexdigest()
    branch = ("D1", "D3")[int(digest[-1], 16) & 1]
    choices = {name: "B" for name in s.METHODS}
    if actionable:
        choices.update(max_predicted_iou="B", max_unsigned_change="D1",
                       always_D1="D1", always_D3="D3", random_deletion=branch,
                       prompt_direct="P", reset_direct="R")
        if not tied:
            choices.update(signed_deletion="D1", random_on_signed=branch,
                           native_bank="M1", prompt_signed="P")
    losses = ({"B": 1., "D1": 1., "D3": 1.} if tied else {"B": 2., "D1": .5, "D3": .5})
    return dict(actionable=actionable, choices=choices, signed_action=actionable and not tied,
                random={"sha256": digest, "branch": branch}, bank_losses=losses if actionable else None)


def scalar_row(sequence="dogs-scale", frame=4, actionable=True, tied=False):
    losses = {"B": 2., "D1": .5, "D3": .5, "M0": .4, "M1": .1, "M2": .1, "P": .05, "R": 3.}
    names = ("B", "D1", "D3", "M0", "M1", "M2") + (("P", "R") if actionable else ())
    candidates = {name: dict(status="completed", predicted_iou=.8,
        loss=(1. if tied else losses[name]) if actionable else None) for name in names}
    for name in ("B", "D1", "D3"):
        candidates[name]["unsigned_change"] = 0. if name == "B" else .3
    return dict(sequence=sequence, frame=frame, target_id=1, status="completed",
                actionable=actionable, candidates=candidates,
                selection=expected_selection(sequence, frame, actionable, tied=tied))


class Limits:
    def __init__(self, body_bytes, output_bytes, member_bytes, *rest):
        self.body_bytes, self.output_bytes, self.member_bytes = body_bytes, output_bytes, member_bytes
    def validate(self):
        pass


class SealContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.root, self.installation = self.base / "repository", self.base / "installation"
        self.h = Helper()
        self.http_calls = 0
        self.mutate_after_transport = False
        self.proof_path = self.base / "proof.json"
        self.sources = set(v.SOURCES) | {s.SELF, s.GT_WRAPPER}
        for name in self.sources:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((HERE / Path(name).name).read_bytes()
                             if name in (s.SELF, s.GT_WRAPPER) else name.encode())
        self.source_hashes = self.h.frozen(self.root, COMMIT, v.SOURCES)
        self.plan = {"cohorts": [], "archive": {"artificial": True}, "source_bindings": {}, "members": []}
        self.index = SimpleNamespace(fingerprints={"manifest": sha(self.root / v.MANIFEST)}, entries={})
        self.verified = {"predictive_rgb": {}, "initial_supervision": {}}
        self.witnesses = {}
        self.proof = dict(schema="evr0029.prediction-proof.v2", expected_commit=COMMIT,
                          evidence_commit=v.EVIDENCE_COMMIT, qualification_commit=v.QUALIFICATION_COMMIT,
                          replication=v.REPLICATION, videos=[])
        self.outputs, self.outer_paths = [], []
        self.evidence_binding = {"file_sha256": {}, "coverage": {"pass": True}}
        self.qualification_binding = {"file_sha256": {}, "receipt_sha256": "c" * 64}
        for sequence, count in s.COHORT:
            queries = [4 + (2*j*(count-5)+7)//14 for j in range(8)]
            self.plan["cohorts"].append(dict(sequence=sequence, frame_count=count,
                frame_stems=["%05d" % f for f in range(count)], query_frame_indices=queries))
            for frame in range(count):
                path = self.base / "rgb" / sequence / ("%05d.jpg" % frame)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"not a real image")
                self.verified["predictive_rgb"][sequence, frame] = self.media(path, frame)
            initial = self.base / "initial" / sequence / "00000.png"
            initial.parent.mkdir(parents=True, exist_ok=True)
            initial.write_bytes(b"not an annotation")
            self.verified["initial_supervision"][sequence, 0] = self.media(initial, 0)
            for frame in queries:
                path = self.base / "witnesses" / sequence / ("%05d.json" % frame)
                write(path, dict(sequence=sequence, frame=frame, target_id=1, actionable=True))
                self.witnesses[sequence, frame] = self.media(path, frame)
                name = sequence + "/" + ("%05d.png" % frame)
                self.index.entries[name] = dict(stage="evaluator_gt", filename=name,
                                               output_relative="evaluator_gt/" + name)
            self.make_video(sequence, count, queries)
        write(self.proof_path, self.proof)
        self.admitted = dict(plan=self.plan, verified=self.verified, witnesses=self.witnesses,
            installation=self.installation, installed={"installed_inventory": self.h.inventory(None)},
            prior={"metadata_root": self.base / "metadata", "native_root": self.base / "native"},
            binding=self.evidence_binding)
        self.fake_v = SimpleNamespace(**{name: getattr(v, name) for name in
            ("SOURCES", "EVIDENCE_COMMIT", "QUALIFICATION_COMMIT", "REPLICATION", "MANIFEST",
             "video_inputs", "prediction_inventory", "unchanged_scientific_sources", "new_outer_contract")})
        self.fake_v.admit_evidence = mock.Mock(side_effect=lambda *a: self.admitted)
        self.fake_v.qualification_contract = mock.Mock(return_value=self.qualification_binding)
        self.fake_v.evidence_arguments = lambda *a: (self.admitted["prior"], [])
        self.helper = SimpleNamespace(Limits=Limits, RangeReader=lambda *a: SimpleNamespace())
        self.fake_acq = SimpleNamespace(HELPER=acq.HELPER, MANIFEST=acq.MANIFEST,
            PHASE_CAPS=acq.PHASE_CAPS, PHASE_COUNTS=acq.PHASE_COUNTS,
            load_helper=lambda _: self.helper, load_plan=lambda *a: (self.plan, self.index),
            acquire_phase=acq.acquire_phase)
        self.fake_old = SimpleNamespace(outer_contract=old.outer_contract, snapshot=old.snapshot,
            unchanged=old.unchanged, module=lambda *a: self.fake_acq)
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(s, "module", side_effect=self.dispatch).start()
        mock.patch.object(s, "require_runtime", return_value=None).start()
        mock.patch.object(g, "module", return_value=s).start()
        mock.patch.object(acq, "extract_bounded", side_effect=self.fake_transport).start()

    def dispatch(self, name, path):
        return {"install_runtime.py": self.h, "run_video.py": self.fake_v,
                "run_evidence.py": self.fake_old}[Path(path).name]

    def media(self, path, frame):
        return dict(path=path, frame=frame, bytes=path.stat().st_size, sha256=sha(path))

    def inert_array(self, directory, name):
        path = directory / name
        path.write_bytes(b"opaque artificial bytes; deliberately not NPZ")
        return dict(filename=name, bytes=path.stat().st_size, sha256=sha(path))

    def make_video(self, sequence, count, queries):
        out = self.base / ("prediction-" + sequence)
        prediction = out / "predictions"
        prediction.mkdir(parents=True)
        self.outputs.append(out)
        worker = dict(schema="evr0029.native-video.v1", status="completed", complete=True,
            source_sha256=self.source_hashes, sequence=sequence, frame_count=count, query_count=8,
            noninitial_annotation_access=False, actionable_count=8, queries=[], outputs=[],
            method_names=list(s.METHODS), descriptive_controls=["ungated_D1", "ungated_D3"],
            baseline=[{"frame": f} for f in range(count)],
            calls=dict(image_encoder_intent=count, image_encoder_completed=count, baseline_frames=count,
                track_step_intent=count+40, track_step_completed=count+40,
                verification_intent=8, verification_completed=8, deletion_intent=16,
                deletion_completed=16, prompt_intent=16, prompt_completed=16, native_bank_extra_calls=0))
        for frame in queries:
            row = scalar_row(sequence, frame)
            witness = self.witnesses[sequence, frame]
            row.update(input={**witness, "path": str(witness["path"])},
                       witnesses=self.h.read_json(witness["path"]))
            descriptors = {}
            for name, candidate in row["candidates"].items():
                filename = "query-%05d-%s.npz" % (frame, "native-bank" if name.startswith("M") else name)
                if filename not in descriptors:
                    descriptors[filename] = self.inert_array(prediction, filename)
                    worker["outputs"].append(descriptors[filename])
                candidate.update(output=descriptors[filename], array_key="logits",
                                 channel=int(name[1:]) if name.startswith("M") else 0)
            worker["queries"].append(row)
        write(prediction / "video-record.json", worker)
        inventory = {name: {"sha256": sha(prediction / name), "bytes": (prediction / name).stat().st_size}
                     for name in sorted(p.name for p in prediction.iterdir())}
        record = dict(schema="evr0029.video-operation.v2", status="completed", eligible=True,
            expected_commit=COMMIT, evidence_commit=v.EVIDENCE_COMMIT, sequence=sequence,
            qualification_commit=v.QUALIFICATION_COMMIT, replication=v.REPLICATION,
            scientific_source_sha256=self.h.frozen(self.root, v.QUALIFICATION_COMMIT, v.SCIENTIFIC_SOURCES),
            frame_count=count, query_count=8, noninitial_annotation_access=False,
            sam_sequence_invocations=1, offline_attempts=[], source_sha256=self.source_hashes,
            evidence_binding=self.evidence_binding, qualification_binding=self.qualification_binding,
            query_frames=queries, outputs=inventory, native_video=worker)
        write(out / "receipt.json", record)
        write(out / "seal.json", dict(schema="evr0029.video-seal.v2", status="completed",
            expected_commit=COMMIT, evidence_commit=v.EVIDENCE_COMMIT,
            qualification_commit=v.QUALIFICATION_COMMIT, replication=v.REPLICATION,
            source_sha256=self.source_hashes, sequence=sequence, frame_count=count, query_count=8,
            receipt_sha256=sha(out / "receipt.json"),
            files={"predictions/" + name: item for name, item in inventory.items()}))
        flags = {"repo-root": str(self.root), "expected-commit": COMMIT,
            "evidence-root": str(self.base / "evidence"), "evidence-spec": str(self.base / "evidence-spec.json"),
            "evidence-commit": v.EVIDENCE_COMMIT, "qualification-root": str(self.base / "qualification"),
            "qualification-spec": str(self.base / "qualification-spec.json"),
            "qualification-commit": v.QUALIFICATION_COMMIT,
            "sequence": sequence, "output-dir": str(out)}
        command = [str(self.installation / "venv/bin/python"), "-S", str(self.root / v.OWN / "run_video.py")]
        for name, value in flags.items():
            command.extend(("--" + name, value))
        spec = self.base / ("spec-" + sequence + ".json")
        outerdir = self.base / ("outer-" + sequence)
        outerdir.mkdir()
        write(spec, dict(repo=str(self.root), commit=COMMIT, result_file=str(out / "seal.json"),
                         operation_output=str(outerdir), command=command, timeout=1800))
        for name in ("stdout.log", "stderr.log"):
            (outerdir / name).write_bytes(b"manufactured clean operation")
        def snapshot(label, start, end, rows):
            return dict(label=label, started_seconds=start, finished_seconds=end,
                        known_groups=[123], discovered_groups=[], rows=rows)
        outer = dict(schema="evr0029.exit-checked-operation.v1", status="completed", exit_code=0, termination=None,
            remaining_descendant_pids=[], cleanup_errors=[], expected_commit=COMMIT,
            signal_attempts=[], child_pid=123, declared_child_pgid=123, observed_child_pgid=123,
            pgid_observation_error=None, exit_observed_at_seconds=.02, triggering_snapshot=None,
            pre_poll_snapshot=snapshot("before_poll", .001, .01, [[123, 1, 123, 1], [124, 123, 123, 0]]),
            post_exit_snapshot=snapshot("after_exit_observed", .03, .04, []),
            cleanup_snapshots=[snapshot("final", .05, .06, [])],
            command=[x.replace(str(self.root), "$REPOSITORY").replace(str(self.root.parent), "$PRIVATE")
                     for x in command], spec_sha256=sha(spec), output_sha256=sha(out / "seal.json"),
            hard_timeout_seconds=1800, group_rss_limit_kib=8388608, wall_seconds=.1,
            peak_group_rss_kib=1, thread_environment={k: "4" for k in self.h.THREAD_KEYS},
            supervisor_sha256=sha(self.root / v.EXIT_CHECKED_SUPERVISOR),
            monitor_sha256=sha(self.root / old.MONITOR),
            log_sha256={name: sha(outerdir / name) for name in ("stdout.log", "stderr.log")})
        write(outerdir / "receipt.json", outer)
        self.outer_paths.append(outerdir / "receipt.json")
        self.proof["videos"].append({"sequence": sequence, "spec_path": str(spec)})

    def fake_transport(self, helper, reader, entry):
        self.http_calls += 1
        if self.mutate_after_transport and self.http_calls == 48:
            self.proof_path.write_bytes(self.proof_path.read_bytes() + b" ")
        return b"inert fake PNG body", {"sha256": hashlib.sha256(b"inert fake PNG body").hexdigest()}

    def acquire(self, name="gt-operation"):
        return g.run(SimpleNamespace(repo_root=str(self.root), expected_commit=COMMIT,
                                    proof=str(self.proof_path), output_dir=str(self.base / name)))

    def test_all_six_opaque_predictions_admit_exactly_48_stub_transfers(self):
        result = self.acquire()
        self.assertTrue(result["eligible"])
        self.assertEqual(self.http_calls, 48)
        admission = result["prediction_admission"]
        self.assertEqual((admission["video_count"], admission["frame_count"], admission["query_count"]), (6, 386, 48))
        self.assertEqual(admission["calls"]["track_step_completed"], 626)
        self.assertEqual(admission["qualification_commit"], v.QUALIFICATION_COMMIT)
        self.assertEqual(self.fake_v.qualification_contract.call_args.args[3], v.QUALIFICATION_COMMIT)
        self.assertFalse(result["query_gt_decoded"])

    def test_missing_video_blocks_before_transport(self):
        self.proof["videos"].pop()
        write(self.proof_path, self.proof)
        with self.assertRaisesRegex(ValueError, "six-video"):
            self.acquire()
        self.assertEqual(self.http_calls, 0)

    def test_sixth_video_failed_outer_blocks_before_any_transport(self):
        path = self.outer_paths[-1]
        outer = self.h.read_json(path)
        outer.update(status="failed", exit_code=1)
        write(path, outer)
        with self.assertRaisesRegex(ValueError, "unclean.*operation"):
            self.acquire()
        self.assertEqual(self.http_calls, 0)

    def test_actual_video_command_identity_mismatch_blocks_transport(self):
        path = Path(self.proof["videos"][-1]["spec_path"])
        spec = self.h.read_json(path)
        position = spec["command"].index("--sequence") + 1
        spec["command"][position] = "dogs-scale"
        write(path, spec)
        with self.assertRaisesRegex(ValueError, "argument identity"):
            self.acquire()
        self.assertEqual(self.http_calls, 0)

    def test_original_attempt_commit_cannot_mix_into_new_six(self):
        path = Path(self.proof["videos"][2]["spec_path"])
        spec = self.h.read_json(path)
        spec["command"][spec["command"].index("--expected-commit")+1] = v.QUALIFICATION_COMMIT
        write(path, spec)
        with self.assertRaisesRegex(ValueError, "argument identity"):
            self.acquire()
        self.assertEqual(self.http_calls, 0)

    def test_wrong_qualification_freeze_rejected_before_transport(self):
        self.proof["qualification_commit"] = COMMIT
        write(self.proof_path, self.proof)
        with self.assertRaisesRegex(ValueError, "six-video"):
            self.acquire()
        self.assertEqual(self.http_calls, 0)

    def test_post_exit_descendant_blocks_even_with_empty_final_remaining(self):
        path = self.outer_paths[-1]
        outer = self.h.read_json(path)
        outer["post_exit_snapshot"]["rows"] = [[124, 1, 123, 0]]
        write(path, outer)
        with self.assertRaisesRegex(ValueError, "post-exit"):
            self.acquire()
        self.assertEqual(self.http_calls, 0)

    def test_all_ties_preserve_bank_order_and_noop_margin(self):
        row = scalar_row(tied=True)
        s.check_choices(self.h, row)
        self.assertEqual(row["selection"]["choices"]["signed_deletion"], "B")
        changed = copy.deepcopy(row)
        changed["selection"]["choices"]["max_unsigned_change"] = "D3"
        with self.assertRaisesRegex(ValueError, "branch choices"):
            s.check_choices(self.h, changed)

    def test_native_prompt_and_reset_use_their_own_signed_comparisons(self):
        row = scalar_row()
        s.check_choices(self.h, row)
        self.assertEqual(row["selection"]["choices"]["native_bank"], "M1")
        self.assertEqual(row["selection"]["choices"]["prompt_signed"], "P")
        self.assertEqual(row["selection"]["choices"]["reset_signed"], "B")
        row["selection"]["choices"]["native_bank"] = "M2"
        with self.assertRaisesRegex(ValueError, "branch choices"):
            s.check_choices(self.h, row)

    def test_nonactionable_cannot_force_deletion_or_prompt(self):
        row = scalar_row(actionable=False)
        s.check_choices(self.h, row)
        row["selection"]["choices"]["always_D1"] = "D1"
        with self.assertRaisesRegex(ValueError, "branch choices"):
            s.check_choices(self.h, row)

    def test_post_transport_changed_proof_preserves_failure(self):
        self.mutate_after_transport = True
        with self.assertRaisesRegex(ValueError, "admitted file changed"):
            self.acquire()
        self.assertEqual(self.http_calls, 48)
        outer = self.h.read_json(self.base / "gt-operation/receipt.json")
        child = self.h.read_json(self.base / "gt-operation/acquisition/receipt.json")
        self.assertEqual((outer["status"], child["status"]), ("failed", "failed"))
        self.assertFalse(outer["eligible"])
        self.assertEqual(len(child["members"]), 48)
        self.assertFalse(child["media_decoded"])


if __name__ == "__main__":
    unittest.main()
