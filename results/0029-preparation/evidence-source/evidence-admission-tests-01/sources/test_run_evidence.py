"""Manufactured admission tests: inert bytes only, no ML/runtime/network imports."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import site
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


r = load("evr0029_evidence_under_test", HERE / "run_evidence.py")
a = load("evr0029_acquisition_contract", HERE / "acquire_davis.py")
COMMIT = "a" * 40


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


class Helper:
    """Only trust/store APIs are mocked; production admission comparisons run."""
    THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
    MANIFESTS = ("configs/0029-sam2-assets.json", "configs/0029-runtime-assets.json")
    OLD_MANIFESTS = ("configs/0022-source-model-assets.json", "configs/0022-wheel-assets.json")
    sha = staticmethod(sha)
    read_json = staticmethod(lambda path: json.loads(Path(path).read_text()))

    @staticmethod
    def require(value, message):
        if not value:
            raise ValueError(message)

    @staticmethod
    def number(value, low, high):
        return type(value) in (int, float) and math.isfinite(value) and low <= value <= high

    @staticmethod
    def git_bytes(root, commit, relative):
        if commit != COMMIT:
            raise ValueError("wrong fake commit")
        return (root / relative).read_bytes()

    @staticmethod
    def frozen(root, commit, names):
        if commit != COMMIT:
            raise ValueError("wrong fake commit")
        return {name: sha(root / name) for name in names}

    @staticmethod
    def inventory(installation):
        return {"artificial": "unchanged"}


class EvidenceContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.root = self.base / "repository"
        self.native = self.base / "native"
        self.installation = self.native / "installation"
        self.metadata = self.base / "metadata"
        self.h = Helper()
        self.plan = {"archive": {"url": a.URL, "archive_bytes": a.ARCHIVE_BYTES,
                                  "etag": a.ETAG}, "members": [], "cohorts": []}
        for sequence, count in a.COHORT:
            queries = [4 + (2*j*(count-5)+7)//14 for j in range(8)]
            self.plan["cohorts"].append({"sequence": sequence, "frame_count": count,
                                         "query_frame_indices": queries})
            for phase, frames in (("predictive_rgb", range(count)),
                                  ("initial_supervision", [0])):
                for frame in frames:
                    ext = ".jpg" if phase == "predictive_rgb" else ".png"
                    name = sequence + "/" + str(frame).zfill(5) + ext
                    self.plan["members"].append({
                        "filename": "synthetic/" + name, "stage": phase,
                        "sequence": sequence, "frame_index": frame,
                        "output_relative": phase + "/" + name,
                        "uncompressed_bytes": 4})
        sources = {r.OWN + name for name in ("run_evidence.py", "frontend_evidence.py",
            "diagnostic_core.py", "acquire_davis.py", "qualify.py", "native_fixture.py",
            "install_runtime.py")}
        sources.update((r.PROTOCOL, r.MANIFEST, r.SUPERVISOR, r.MONITOR, a.HELPER,
                        "protocols/0029-native-runtime.md",
                        "configs/0029-sdist-build-review.json",
                        "experiments/0022-frontend/qualify.py",
                        *self.h.MANIFESTS, *self.h.OLD_MANIFESTS))
        for name in sources:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((HERE / "run_evidence.py").read_bytes()
                             if name == r.OWN + "run_evidence.py" else name.encode())
        write(self.root / r.MANIFEST, self.plan)
        self.media = {}
        for phase in ("predictive_rgb", "initial_supervision"):
            self.media[phase] = self.make_media(phase)

    def make_outer(self, receipt_path, command, label):
        spec_path = self.base / (label + "-spec.json")
        operation = self.base / (label + "-outer")
        operation.mkdir(exist_ok=True)
        for name in ("stdout.log", "stderr.log"):
            (operation / name).write_bytes(b"artificial log\n")
        spec = {"commit": COMMIT, "repo": str(self.root), "result_file": str(receipt_path),
                "operation_output": str(operation), "command": command, "timeout": 1800}
        write(spec_path, spec)
        displayed = [x.replace(str(self.root), "$REPOSITORY")
                      .replace(str(self.root.parent), "$PRIVATE") for x in command]
        outer = {"schema": "evr0026.operation.v1", "status": "completed", "exit_code": 0,
                 "termination": None, "remaining_descendant_pids": [], "cleanup_errors": [],
                 "command": displayed, "expected_commit": COMMIT,
                 "spec_sha256": sha(spec_path), "output_sha256": sha(receipt_path),
                 "hard_timeout_seconds": 1800, "group_rss_limit_kib": 8388608,
                 "wall_seconds": 1.0, "peak_group_rss_kib": 64,
                 "thread_environment": {key: "4" for key in self.h.THREAD_KEYS},
                 "supervisor_sha256": sha(self.root / r.SUPERVISOR),
                 "monitor_sha256": sha(self.root / r.MONITOR),
                 "log_sha256": {name: sha(operation / name) for name in ("stdout.log", "stderr.log")}}
        write(operation / "receipt.json", outer)
        return spec_path, operation / "receipt.json"

    def make_media(self, phase):
        output = self.base / phase
        output.mkdir()
        entries = [row for row in self.plan["members"] if row["stage"] == phase]
        members = []
        for entry in entries:
            rel = "members/" + entry["output_relative"]
            path = output / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake")
            members.append({**entry, "saved_relative": rel, "bytes": 4, "sha256": sha(path)})
        receipt = {"schema": a.SCHEMA, "status": "completed", "phase": phase,
                   "expected_commit": COMMIT, "manifest_sha256": sha(self.root / r.MANIFEST),
                   "media_decoded": False, "gt_admission": None, "hidden_retries": False,
                   "full_zip_fallback": False,
                   "source_sha256": self.h.frozen(self.root, COMMIT,
                       [a.SELF, a.HELPER, r.MANIFEST, r.PROTOCOL]),
                   "archive": self.plan["archive"], "members": members,
                   "selected_names": [e["filename"] for e in entries],
                   "response_body_bytes_exact": True, "response_body_bytes": 4 * len(entries),
                   "output_bytes": 4 * len(entries)}
        write(output / "receipt.json", receipt)
        command = ["/opt/homebrew/bin/python3.12", "-B", str(self.root / a.SELF),
                   "--repo-root", str(self.root), "--expected-commit", COMMIT,
                   "--metadata-root", str(self.metadata), "--phase", phase,
                   "--output-dir", str(output)]
        spec, outer = self.make_outer(output / "receipt.json", command, phase)
        return output, spec, outer

    def media_contract(self, phase, plan=None):
        output, spec, _ = self.media[phase]
        with mock.patch.object(r, "module", return_value=a):
            return r.media_contract(self.h, self.root, COMMIT, plan or self.plan,
                                    phase, output, spec, self.metadata)

    def refresh_outer_result(self, phase):
        output, _, outer_path = self.media[phase]
        outer = self.h.read_json(outer_path)
        outer["output_sha256"] = sha(output / "receipt.json")
        write(outer_path, outer)

    def test_complete_392_inputs_accepted_without_decode(self):
        rgb, rgb_binding = self.media_contract("predictive_rgb")
        initial, initial_binding = self.media_contract("initial_supervision")
        self.assertEqual((len(rgb), len(initial)), (386, 6))
        self.assertEqual(len(set(rgb) | set(initial)), 386)
        self.assertEqual(sum(v["bytes"] for v in rgb.values()), 1544)
        self.assertNotEqual(rgb_binding["receipt_sha256"], initial_binding["receipt_sha256"])
        for _, row in rgb.items():
            self.assertEqual(row["sha256"], hashlib.sha256(b"fake").hexdigest())

    def test_nonquery_rgb_corruption_blocks_before_worker(self):
        output, _, _ = self.media["predictive_rgb"]
        # Frame1 is not one of the 0 + 8 input frames the evidence worker decodes.
        path = output / "members/predictive_rgb/dogs-scale/00001.jpg"
        path.write_bytes(b"evil")
        with self.assertRaisesRegex(ValueError, "media payload changed"):
            self.media_contract("predictive_rgb")

    def test_self_consistent_partial_phase_is_rejected(self):
        plan = copy.deepcopy(self.plan)
        row = next(x for x in plan["members"] if x["stage"] == "predictive_rgb")
        plan["members"].remove(row)
        with self.assertRaisesRegex(ValueError, "fixed acquisition phase coverage"):
            self.media_contract("predictive_rgb", plan)

    def test_failed_outer_cannot_admit_completed_inner(self):
        _, _, outer_path = self.media["initial_supervision"]
        outer = self.h.read_json(outer_path)
        outer.update(status="failed", exit_code=0, termination="group_rss_limit")
        write(outer_path, outer)
        with self.assertRaisesRegex(ValueError, "unclean operation"):
            self.media_contract("initial_supervision")

    def test_actual_command_mutation_and_source_hash_rejected(self):
        output, spec_path, outer_path = self.media["initial_supervision"]
        original_spec = self.h.read_json(spec_path)
        spec = copy.deepcopy(original_spec)
        spec["command"][-1] = str(self.base / "other-output")
        write(spec_path, spec)
        outer = self.h.read_json(outer_path)
        outer["spec_sha256"] = sha(spec_path)
        write(outer_path, outer)
        with self.assertRaisesRegex(ValueError, "specification identity"):
            self.media_contract("initial_supervision")
        write(spec_path, original_spec)
        outer["spec_sha256"] = sha(spec_path)
        write(outer_path, outer)
        receipt = self.h.read_json(output / "receipt.json")
        receipt["source_sha256"][a.HELPER] = "0" * 64
        write(output / "receipt.json", receipt)
        self.refresh_outer_result("initial_supervision")
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.media_contract("initial_supervision")

    def test_same_or_nested_roots_refused(self):
        rgb = self.base / "rgb"
        for other in (rgb, rgb / "initial", rgb.parent):
            with self.subTest(other=other):
                with self.assertRaisesRegex(ValueError, "disjoint roots"):
                    r.distinct_media_roots(self.h, rgb, other)
        r.distinct_media_roots(self.h, rgb, self.base / "initial")

    def test_log_changed_after_admission_rejected(self):
        _, binding = self.media_contract("initial_supervision")
        _, _, outer = self.media["initial_supervision"]
        (outer.parent / "stderr.log").write_text("changed after admission")
        with self.assertRaisesRegex(ValueError, "admitted file changed"):
            r.unchanged(self.h, binding["file_sha256"])

    def test_native_incomplete_fixture_and_wrong_sources_refused(self):
        installation_binding = {"receipt_sha256": "a" * 64}
        qualify = SimpleNamespace(PROTOCOL="protocols/0029-native-runtime.md",
            REVIEW="configs/0029-sdist-build-review.json",
            OLD_HELPER="experiments/0022-frontend/qualify.py",
            installation_contract=lambda *args: ({}, installation_binding))
        paths = [r.OWN + p for p in ("qualify.py", "native_fixture.py", "install_runtime.py")]
        paths += [qualify.PROTOCOL, qualify.REVIEW, qualify.OLD_HELPER,
                  *self.h.MANIFESTS, *self.h.OLD_MANIFESTS, r.SUPERVISOR, r.MONITOR]
        record = {"schema": "evr0029.native-qualification.v1", "status": "completed",
                  "eligible": True, "expected_commit": COMMIT, "scientific_input_access": False,
                  "offline_attempts": [], "installation_bindings": installation_binding,
                  "source_sha256": self.h.frozen(self.root, COMMIT, paths),
                  "native_fixture": {"status": "started", "manufactured_only": True}}
        path = self.native / "qualification/receipt.json"
        write(path, record)
        with mock.patch.object(r, "module", return_value=qualify), \
                mock.patch.object(r, "outer_contract") as outer:
            with self.assertRaisesRegex(ValueError, "interface checks incomplete"):
                r.native_contract(self.h, self.root, self.native, COMMIT)
            outer.assert_not_called()
            record["source_sha256"][qualify.OLD_HELPER] = "0" * 64
            write(path, record)
            with self.assertRaisesRegex(ValueError, "native source changed"):
                r.native_contract(self.h, self.root, self.native, COMMIT)
            record.update(status="failed", eligible=False)
            write(path, record)
            with self.assertRaisesRegex(ValueError, "not eligible"):
                r.native_contract(self.h, self.root, self.native, COMMIT)

    def run_fake(self, worker):
        args = SimpleNamespace(repo_root=str(self.root), output_dir=str(self.base / "output"),
            expected_commit=COMMIT, native_root=str(self.native), native_commit=COMMIT,
            metadata_root=str(self.metadata),
            rgb_root=str(self.media["predictive_rgb"][0]),
            rgb_spec=str(self.media["predictive_rgb"][1]),
            initial_root=str(self.media["initial_supervision"][0]),
            initial_spec=str(self.media["initial_supervision"][1]))
        acquired = SimpleNamespace(**{key: getattr(a, key) for key in
            ("SCHEMA", "SELF", "HELPER", "PHASE_CAPS", "PHASE_COUNTS")})
        acquired.load_helper = lambda path: None
        acquired.load_plan = lambda *args: (self.plan, None)
        fake_sys = SimpleNamespace(version_info=(3, 12, 13), flags=SimpleNamespace(no_site=1),
            executable=str(self.installation / "venv/bin/python"),
            prefix=str(self.installation / "venv"), path=[], dont_write_bytecode=False)
        loads = []
        guard = mock.Mock()
        def modules(name, path):
            loads.append(name)
            if name == "evr0029_installer":
                return self.h
            if name in ("evr0029_davis_plan", "evr0029_davis_admission"):
                return acquired
            if name == "evr0022_helpers":
                return SimpleNamespace(install_guard=guard)
            if name == "evr0029_frontend_evidence":
                return SimpleNamespace(run=worker)
            raise AssertionError("unexpected module import " + name)
        native = (self.installation, {"installed_inventory": self.h.inventory(self.installation)},
                  {"file_sha256": {}})
        with mock.patch.object(r, "module", side_effect=modules), \
                mock.patch.object(r, "native_contract", return_value=native), \
                mock.patch.object(r, "sys", fake_sys), mock.patch.object(site, "main"), \
                mock.patch.dict(r.os.environ, {key: "4" for key in self.h.THREAD_KEYS}), \
                contextlib.redirect_stdout(io.StringIO()):
            result = r.run(args)
        return result, loads, guard

    def test_worker_gets_only_scheduled_rgb_and_initial_annotations(self):
        def worker(inputs, output, record):
            self.assertEqual(set(inputs), {"qualify_helper", "diagnostic_core", "lightglue_source",
                "runtime", "superpoint_checkpoint", "lightglue_checkpoint", "source_sha256", "cohort"})
            self.assertEqual(len(inputs["cohort"]), 6)
            for cohort, expected in zip(inputs["cohort"], self.plan["cohorts"]):
                self.assertEqual([row["frame"] for row in cohort["images"]],
                                 [0, *expected["query_frame_indices"]])
                self.assertEqual(cohort["initial_mask"]["frame"], 0)
                self.assertIn("initial_supervision", str(cohort["initial_mask"]["path"]))
            return {"status": "completed", "artificial": True}
        result, loads, guard = self.run_fake(worker)
        self.assertTrue(result["eligible"])
        self.assertFalse(result["noninitial_annotation_access"])
        self.assertEqual(result["sam_sequence_invocations"], 0)
        self.assertEqual(loads.count("evr0029_frontend_evidence"), 1)
        guard.assert_called_once()

    def test_postuse_nonquery_mutation_preserved_as_failure(self):
        def worker(inputs, output, record):
            path = self.media["predictive_rgb"][0] / "members/predictive_rgb/dogs-scale/00001.jpg"
            path.write_bytes(b"evil")
            return {"status": "completed"}
        with self.assertRaisesRegex(ValueError, "input payload changed"):
            self.run_fake(worker)
        saved = self.h.read_json(self.base / "output/receipt.json")
        self.assertEqual(saved["status"], "failed")
        self.assertFalse(saved["eligible"])
        self.assertEqual(saved["evidence"]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
