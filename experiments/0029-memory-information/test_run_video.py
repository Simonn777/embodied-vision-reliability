"""Manufactured stdlib-only admission checks; no model, array decoder or network."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


r = load("video_admission_under_test", HERE / "run_video.py")
q = load("manufactured_qualification_checks", HERE / "qualify_native_video.py")
COHORT = (("dogs-scale", 83), ("upside-down", 65), ("dog-agility", 25),
          ("cat-girl", 89), ("planes-water", 38), ("dog-gooses", 86))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


class Helper:
    sha = staticmethod(sha)
    read_json = staticmethod(lambda path: json.loads(Path(path).read_text()))

    @staticmethod
    def require(condition, message):
        if not condition:
            raise ValueError(message)


class VideoAdmission(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.root = self.base / "repo"
        self.directory = self.base / "evidence"
        self.directory.mkdir()
        self.h = Helper()

    def artifact(self, directory, name, content=b"inert, not an encoded array"):
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return dict(filename=name, bytes=len(content), sha256=sha(path))

    def evidence_fixture(self):
        plan = {"cohorts": []}
        verified = {"predictive_rgb": {}, "initial_supervision": {}}
        result = dict(schema="evr0029.frontend-evidence.v1", status="completed", complete=True,
            sam_calls=0, noninitial_annotation_access=False, sam_sequence_admitted=False,
            calls=dict(superpoint_intent=54, superpoint_completed=54,
                       lightglue_intent=48, lightglue_completed=48),
            images=[], queries=[], initial_masks=[])
        for sequence, count in COHORT:
            queries = [4 + (2 * j * (count - 5) + 7) // 14 for j in range(8)]
            plan["cohorts"].append(dict(sequence=sequence, frame_count=count,
                query_frame_indices=queries, frame_stems=["%05d" % f for f in range(count)]))
            mask = dict(path=self.base / "initial" / sequence / "00000.png",
                        bytes=4, sha256="a" * 64)
            verified["initial_supervision"][(sequence, 0)] = mask
            result["initial_masks"].append(dict(sequence=sequence, frame=0, status="completed",
                input={**mask, "path": str(mask["path"])}, target_id=1, shape=[20, 30]))
            for frame in [0, *queries]:
                media = dict(frame=frame, path=self.base / "rgb" / sequence / ("%05d.jpg" % frame),
                             bytes=4, sha256="b" * 64)
                verified["predictive_rgb"][(sequence, frame)] = media
                result["images"].append(dict(sequence=sequence, frame=frame, status="completed",
                    input={**media, "path": str(media["path"])}, shape=[20, 30, 3],
                    output=self.artifact(self.directory, "features-%s-%05d.npz" % (sequence, frame))))
            for frame in queries:
                witness = dict(sequence=sequence, frame=frame, target_id=1,
                    initial_shape=[20, 30], current_shape=[20, 30], actionable=True,
                    positive=[{"label": 1}] * 4, negative=[{"label": 0}] * 4,
                    counts={"positive_retained": 4, "negative_retained": 4})
                filename = "witnesses-%s-%05d.json" % (sequence, frame)
                descriptor = self.artifact(self.directory, filename, json.dumps(witness).encode())
                result["queries"].append(dict(sequence=sequence, frame=frame, status="completed",
                    target_id=1, actionable=True, witness_counts=witness["counts"], witnesses=descriptor,
                    matches=self.artifact(self.directory, "matches-%s-%05d.npz" % (sequence, frame))))
        result["coverage"] = dict(queries=48, videos=6, actionable_queries=48,
            actionable_queries_by_video={name: 8 for name, _ in COHORT},
            videos_with_evidence=6, **{"pass": True})
        write(self.directory / "evidence-record.json", result)
        return plan, result, verified

    def test_complete_evidence_and_opaque_artifact_tamper(self):
        plan, result, verified = self.evidence_fixture()
        witnesses, coverage, _ = r.evidence_inventory(self.h, plan, self.directory, result, verified)
        self.assertEqual(len(witnesses), 48)
        self.assertTrue(coverage["pass"])
        # The files deliberately are not NPZs. Admission must only hash them.
        (self.directory / result["queries"][0]["matches"]["filename"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            r.evidence_inventory(self.h, plan, self.directory, result, verified)

    def test_self_consistent_insufficient_coverage_is_not_admitted(self):
        plan, result, verified = self.evidence_fixture()
        for row in result["queries"][24:]:
            filename = row["witnesses"]["filename"]
            item = self.h.read_json(self.directory / filename)
            item.update(negative=[], actionable=False)
            item["counts"]["negative_retained"] = 0
            row.update(actionable=False, witness_counts=item["counts"],
                       witnesses=self.artifact(self.directory, filename, json.dumps(item).encode()))
        counts = {name: (8 if index < 3 else 0) for index, (name, _) in enumerate(COHORT)}
        result["coverage"].update(actionable_queries=24, actionable_queries_by_video=counts,
                                  videos_with_evidence=3, **{"pass": False})
        write(self.directory / "evidence-record.json", result)
        with self.assertRaisesRegex(ValueError, "coverage does not admit"):
            r.evidence_inventory(self.h, plan, self.directory, result, verified)

    def test_actual_argument_layout_and_duplicate_flag_rejection(self):
        evidence = self.base / "output"
        native = self.base / "native"
        flags = {"repo-root": str(self.root), "expected-commit": r.EVIDENCE_COMMIT,
            "native-root": str(native), "native-commit": "a" * 40,
            "metadata-root": str(self.base / "metadata"), "rgb-root": str(self.base / "rgb"),
            "rgb-spec": str(self.base / "rgb-spec.json"), "initial-root": str(self.base / "initial"),
            "initial-spec": str(self.base / "initial-spec.json"), "output-dir": str(evidence)}
        command = [str(native / "installation/venv/bin/python"), "-S",
                   str(self.root / r.OWN / "run_evidence.py")]
        for key, value in flags.items():
            command.extend(("--" + key, value))
        spec = self.base / "spec.json"
        write(spec, {"command": command})
        parsed, observed = r.evidence_arguments(self.h, self.root, r.EVIDENCE_COMMIT, evidence, spec)
        self.assertEqual(len(observed), 23)
        self.assertEqual(parsed["native_root"], native)
        command[5] = command[3]
        write(spec, {"command": command})
        with self.assertRaisesRegex(ValueError, "duplicate evidence flags"):
            r.evidence_arguments(self.h, self.root, r.EVIDENCE_COMMIT, evidence, spec)

    def test_full_video_inputs_reject_extra_file(self):
        sequence, count = "dog-agility", 25
        folder = self.base / "frames"
        folder.mkdir()
        rgb = {}
        for frame in range(count):
            path = folder / ("%05d.jpg" % frame)
            path.write_bytes(b"not an image")
            rgb[(sequence, frame)] = dict(frame=frame, path=path, bytes=path.stat().st_size, sha256=sha(path))
        initial = dict(path=self.base / "initial.png", bytes=4, sha256="a" * 64)
        admitted = dict(plan={"cohorts": [dict(sequence=sequence, frame_count=count,
            frame_stems=["%05d" % f for f in range(count)], query_frame_indices=list(q.QUERY_FRAMES))]},
            verified={"predictive_rgb": rgb, "initial_supervision": {(sequence, 0): initial}},
            witnesses={(sequence, f): dict(frame=f, path=self.base / ("%d.json" % f),
                        sha256="c" * 64, bytes=1) for f in q.QUERY_FRAMES},
            installation=self.base / "installation")
        inputs = r.video_inputs(self.h, self.root, {"source": "hash"}, admitted, sequence)
        self.assertEqual(inputs["video"]["frame_count"], 25)
        self.assertEqual(len(inputs["video"]["witnesses"]), 8)
        (folder / "extra.jpg").write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "full video directory"):
            r.video_inputs(self.h, self.root, {}, admitted, sequence)

    def prediction_fixture(self):
        directory = self.base / "predictions"
        directory.mkdir()
        result = dict(schema="evr0029.native-video.v1", status="completed", complete=True,
            sequence="dog-agility", frame_count=25, query_count=8, noninitial_annotation_access=False,
            baseline=[{"frame": f} for f in range(25)], queries=[], outputs=[], actionable_count=7,
            method_names=["baseline"], calls=dict(image_encoder_intent=25, image_encoder_completed=25,
                baseline_frames=25, track_step_intent=63, track_step_completed=63,
                verification_intent=8, verification_completed=8, deletion_intent=16,
                deletion_completed=16, prompt_intent=14, prompt_completed=14, native_bank_extra_calls=0))
        witnesses = []
        for index, frame in enumerate(q.QUERY_FRAMES):
            path = self.base / ("%d-witness.json" % frame)
            item = dict(target_id=1, actionable=index != 0)
            write(path, item)
            media = dict(frame=frame, path=path, bytes=path.stat().st_size, sha256=sha(path))
            witnesses.append(media)
            names = ["B", "D1", "D3", "M0", "M1", "M2"] + (["P", "R"] if index else [])
            descriptors, candidates = {}, {}
            for name in names:
                filename = "query-%05d-%s.npz" % (frame, "native-bank" if name.startswith("M") else name)
                if filename not in descriptors:
                    descriptors[filename] = self.artifact(directory, filename)
                    result["outputs"].append(descriptors[filename])
                candidates[name] = dict(status="completed", output=descriptors[filename],
                    array_key="logits", channel=int(name[1:]) if name.startswith("M") else 0)
            result["queries"].append(dict(sequence="dog-agility", frame=frame, status="completed",
                target_id=1, input={**media, "path": str(path)}, witnesses=item,
                actionable=index != 0, candidates=candidates,
                selection={"actionable": index != 0, "choices": {"baseline": "B"}}))
        write(directory / "video-record.json", result)
        video = dict(sequence="dog-agility", frame_count=25, witnesses=witnesses)
        return directory, result, video

    def test_prediction_inventory_requires_all_candidates_and_charged_calls(self):
        directory, result, video = self.prediction_fixture()
        files = r.prediction_inventory(self.h, directory, result, video)
        self.assertEqual(len(files), 47)  # 4 files for first query, 6 for each remaining query, record.
        changed = copy.deepcopy(result)
        del changed["queries"][1]["candidates"]["R"]
        write(directory / "video-record.json", changed)
        with self.assertRaisesRegex(ValueError, "candidate bank"):
            r.prediction_inventory(self.h, directory, changed, video)
        changed = copy.deepcopy(result)
        changed["calls"]["track_step_completed"] = 62
        write(directory / "video-record.json", changed)
        with self.assertRaisesRegex(ValueError, "call completeness"):
            r.prediction_inventory(self.h, directory, changed, video)

    def test_qualification_rejection_precedes_guard_and_worker_loading(self):
        out = self.base / "fresh-operation"
        args = SimpleNamespace(repo_root=str(self.root), expected_commit="b" * 40,
            evidence_commit=r.EVIDENCE_COMMIT, sequence="dog-agility",
            evidence_root=str(self.base / "old"), evidence_spec=str(self.base / "old-spec.json"),
            qualification_root=str(self.base / "qualification"),
            qualification_spec=str(self.base / "qualification-spec.json"), output_dir=str(out))
        h = Helper()
        h.frozen = mock.Mock(return_value={r.OWN + "run_video.py": sha(HERE / "run_video.py")})
        admitted = {"binding": {"inert": True}}
        loaded = []

        def fake_module(_name, path):
            loaded.append(Path(path).name)
            if Path(path).name == "install_runtime.py":
                return h
            if Path(path).name == "run_evidence.py":
                return SimpleNamespace()
            raise AssertionError("guard/model/other source must not load after failed qualification")

        with mock.patch.object(r, "module", side_effect=fake_module), \
             mock.patch.object(r, "admit_evidence", return_value=admitted), \
             mock.patch.object(r, "qualification_contract", side_effect=ValueError("qualification not admitted")), \
             mock.patch.object(r, "video_inputs", side_effect=AssertionError("too late")) as video:
            with self.assertRaisesRegex(ValueError, "qualification not admitted"):
                r.run(args)
        self.assertEqual(loaded, ["install_runtime.py", "run_evidence.py"])
        video.assert_not_called()
        receipt = self.h.read_json(out / "receipt.json")
        self.assertEqual(receipt["status"], "failed")
        self.assertFalse(receipt["eligible"])
        self.assertEqual(receipt["sam_sequence_invocations"], 0)


if __name__ == "__main__":
    unittest.main()
