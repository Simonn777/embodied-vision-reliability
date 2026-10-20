"""Manufactured admission/lifecycle checks; no native model or task payloads.

The opaque .npz files in RunFixture are deliberately NOT scientific archives.
Only the lifecycle is real there. Native parity uses separate invented tensors;
packed archive decoding is exercised by the evaluator's manufactured tests.
"""
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch


_spec = importlib.util.spec_from_file_location("d38_complete_test_subject",
    Path(__file__).with_name("run_complete.py"))
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)


def save(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False, sort_keys=True))


class RunFixture:
    """Replace computation, never the source/preseal/postseal lifecycle."""
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.repo = self.root / "repo"
        self.here = self.repo / "experiments/0038-late-memory-readout"
        self.old = self.repo / "experiments/0035-transformation-support"
        self.here.mkdir(parents=True)
        self.old.mkdir(parents=True)
        self.common_path = self.root / "common.py"
        self.protocol = self.here / "FULL_COMPARISON.md"
        paths = [self.here / name for name in ("run_complete.py", "runtime_adapter.py",
            "operators.py", "evaluate_complete.py", "ADMISSION.md")]
        paths += [self.old / name for name in ("admit.py", "native_bank.py", "evaluate.py", "metrics.py")]
        paths += [self.common_path, self.protocol]
        for path in paths:
            path.write_text("# invented source: " + path.name + "\n")
        self.output = self.root / "result"
        self.spec_path = self.root / "spec.json"
        self.spec = dict(repo=str(self.repo), output=str(self.output), common_path=str(self.common_path),
            protocol_path=str(self.protocol), source_sha256={str(p): worker.sha(p) for p in paths},
            max_output_bytes=16 * 1024**2, operation_wall_seconds=1800,
            launch_admitted=True, preparation_only=False)
        save(self.spec_path, self.spec)
        self.events = []
        self.failure = None
        self.incomplete = False
        self.mutate_seal = False
        self.context = {"prediction_receipt": dict(model_initial_hash="model", model_final_hash="model",
            rows=[dict(sequence=name) for name, _ in worker.COHORT])}
        self.sequences = [dict(sequence=name, count=count) for name, count in worker.COHORT]
        self.common = SimpleNamespace(save=save, load_runtime=self.load_runtime, finish=self.finish)
        self.native = SimpleNamespace(validate_sequence=lambda row: None, descriptor=self.descriptor)
        self.admission35 = SimpleNamespace(postcheck=lambda context: self.events.append("postcheck"))
        self.evaluate35 = SimpleNamespace(frame_path=self.frame_path)
        self.evaluator = SimpleNamespace(evaluate_complete=self.evaluate)

    def descriptor(self, common, root, path, kind, **metadata):
        path = Path(path)
        return dict(path=str(path.relative_to(root)), bytes=path.stat().st_size,
                    sha256=worker.sha(path), kind=kind, **metadata)

    def frame_path(self, root, desc):
        path = Path(root) / desc["path"]
        worker.require(path.stat().st_size == desc["bytes"] and worker.sha(path) == desc["sha256"],
                       "manufactured output changed")
        return path

    def admit(self, spec, record, common, admission35, here):
        self.events.append("admission")
        if self.failure == "admission":
            raise ValueError("manufactured failed outer qualification")
        record.update(bindings={"manufactured": True}, data_index={"manufactured": True})
        return self.context, self.sequences

    def load_runtime(self, spec, record):
        self.events.append("runtime")
        record["model_initial_hash"] = "wrong" if self.failure == "model_hash" else "model"
        return dict(np=None, torch=None, Image=None)

    def finish(self, runtime, record):
        self.events.append("finish")
        record["model_final_hash"] = "model"

    def execute(self, runtime, common, sequence, oldrow, context, out, record, modules):
        self.events.append("prediction:" + sequence["sequence"])
        name, count = sequence["sequence"], sequence["count"]
        directory = Path(out) / name
        directory.mkdir()
        predictions = []
        for frame in range(count):
            if self.incomplete and name == worker.COHORT[-1][0] and frame == count - 1:
                break
            path = directory / ("%05d.npz" % frame)
            path.write_bytes(b"manufactured opaque prediction; never decoded")
            desc = self.descriptor(common, out, path, "d38-readout-frame", sequence=name, frame=frame)
            predictions.append(desc)
            record["outputs"].append(desc)
            record["rows"].append(dict(sequence=name, frame=frame, output=desc))
            record["bytes_written"] += desc["bytes"]
            record["completed_frames"] += 1
            record["track_step_intent"] += 1
            record["track_step_completed"] += 1
            if frame >= 2:
                record["eligible_side_frames"] += 1
            if self.failure == "prediction" and frame == 0:
                raise RuntimeError("manufactured native failure after first saved prediction")
        record["initial_label_decodes"] += 1
        record["sequences"].append(dict(sequence=name, status="completed", predictions=predictions,
            completed_frames=len(predictions), final_state_exact=True))
        record["completed_sequences"] += 1

    def evaluate(self, spec, out, record, context, **runtime):
        # This is the sole simulated noninitial-label access point.
        receipt_path, seal_path = Path(out) / "prediction-receipt.json", Path(out) / "prediction-seal.json"
        receipt, seal = worker.load_json(receipt_path), worker.load_json(seal_path)
        worker.require(record["all_predictions_sealed_before_gt"] is True and seal["complete"] is True,
                       "evaluation reached without seal")
        worker.require(worker.sha(receipt_path) == seal["receipt_sha256"], "receipt not sealed")
        worker.require(len(receipt["rows"]) == len(seal["outputs"]) == 446, "all rows precede GT")
        worker.require([(r["sequence"], r["frame"]) for r in receipt["rows"]] ==
            [(name, frame) for name, count in worker.COHORT for frame in range(count)], "ordered complete rows")
        self.events.append("GT")
        save(Path(out) / "evaluation.json", dict(manufactured=True, rows=446))
        if self.mutate_seal:
            seal_path.write_text("{}")
        return dict(manufactured=True)

    def module(self, name, path):
        modules = {"d38_complete_common": self.common, "d38_complete_admission35": self.admission35,
            "d38_complete_native35": self.native, "d38_complete_evaluate35": self.evaluate35,
            "d38_complete_evaluator": self.evaluator}
        return modules.get(name, SimpleNamespace())

    def patched(self):
        stack = ExitStack()
        stack.enter_context(patch.object(worker, "__file__", str(self.here / "run_complete.py")))
        for constant, path in (("COMMON_SHA", self.common_path), ("ADAPTER_SHA", self.here / "runtime_adapter.py"),
                               ("OPERATORS_SHA", self.here / "operators.py")):
            stack.enter_context(patch.object(worker, constant, worker.sha(path)))
        stack.enter_context(patch.object(worker, "module", side_effect=self.module))
        stack.enter_context(patch.object(worker, "admission", side_effect=self.admit))
        stack.enter_context(patch.object(worker, "execute_sequence", side_effect=self.execute))
        stack.enter_context(patch.object(worker.shutil, "disk_usage", return_value=SimpleNamespace(free=2**32)))
        return stack


class CompleteLifecycleTests(unittest.TestCase):
    def test_unadmitted_preparation_cannot_create_output_or_load_runtime(self):
        with TemporaryDirectory() as directory:
            f = RunFixture(directory)
            f.spec.update(launch_admitted=False, preparation_only=True)
            save(f.spec_path, f.spec)
            with f.patched(), self.assertRaisesRegex(ValueError, "not admitted"):
                worker.run(f.spec_path)
            self.assertFalse(f.output.exists())
            self.assertEqual(f.events, [])

    def test_old_receipt_link_and_qualification_raw_command_are_required(self):
        # The legacy verifier is stubbed; the NEW qualification linkage is real.
        for change in (None, "failed", "command", "outer_result"):
            with self.subTest(change=change), TemporaryDirectory() as directory:
                base = Path(directory).resolve()
                here, installed = base / "d38", base / "installation"
                here.mkdir()
                sources = {str(base / ("old-%02d.py" % i)): "old-source" for i in range(70)}
                q_sources = {str(here / name): digest for name, digest in
                    (("runtime_adapter.py", worker.ADAPTER_SHA), ("operators.py", worker.OPERATORS_SHA),
                     ("qualify_native.py", worker.QUALIFIER_SHA))}
                identity = dict(repo=str(base / "repo"), native_root=str(base / "native"),
                                native_commit="native", common_path=str(base / "common.py"))
                spec = dict(identity, source_sha256={**sources, **q_sources})
                def bound(key, value, location=None):
                    path = location or base / (key + ".json")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    save(path, value)
                    spec[key] = dict(path=str(path), sha256=worker.sha(path))
                    return spec[key]
                for key in worker.BINDINGS:
                    bound(key, {})
                bound("d35_spec", dict(identity, source_sha256=sources))
                result_path = base / "d35-evaluation/evaluation.json"
                bound("d35_result", {"manufactured_numerical_result": True}, result_path)
                bound("d35_receipt", dict(schema="evr0035.evaluation.v1", status="completed",
                    spec_sha256=spec["d35_spec"]["sha256"], source_sha256=sources,
                    completed_frames=446, parity_frames=446, label_decodes=446,
                    label_decode_attempts=446, completed_sequences=4, offline_attempts=[],
                    outputs=[dict(path="evaluation.json", sha256=spec["d35_result"]["sha256"],
                                  bytes=result_path.stat().st_size)]), result_path.parent / "receipt.json")
                # The old outer seals its receipt, not the numerical evaluation.
                target = "d35_result" if change == "outer_result" else "d35_receipt"
                bound("d35_outer", dict(output_sha256=spec[target]["sha256"]))
                qpath = base / "qualification_spec.json"
                qreceipt = base / "qualified/receipt.json"
                qouter = base / "qualified-outer/receipt.json"
                bound("qualification_spec", dict(identity, output=str(qreceipt.parent), source_sha256=q_sources))
                qualified = dict(schema="evr0038.native-qualification.v1", status="completed", eligible=True,
                    manufactured_only=True, scientific_payload_reads=0,
                    spec_sha256=spec["qualification_spec"]["sha256"], source_sha256=q_sources,
                    full_state_parity=True, side_state_unchanged=True, native_passes=2,
                    observer_final_state_sha256="same-state", plain_final_state_sha256="same-state",
                    native_output_parities=[dict(frame=i, exact=True) for i in range(10)],
                    side_frames=[dict(frame=8, readouts=dict(selected_context_exact=True,
                        selected_stack_exact=True, selected_sam_exact=True,
                        calls=dict(context_materializations=4, attention_stacks=5, sam_heads=6)))])
                if change == "failed":
                    qualified["status"] = "failed"
                bound("qualification_receipt", qualified, qreceipt)
                command = [str(installed / "venv/bin/python"), "-B", "-S",
                           str(here / "qualify_native.py"), "--spec", str(qpath)]
                if change == "command":
                    command[3] = str(here / "qualify_optimized.py")
                bound("qualification_outer_spec", dict(command=command, repo=identity["repo"],
                    commit="qualified-commit", timeout=60, operation_output=str(qouter.parent),
                    result_file=str(qreceipt)))
                bound("qualification_outer", dict(spec_sha256=spec["qualification_outer_spec"]["sha256"],
                    expected_commit="qualified-commit", hard_timeout_seconds=60,
                    output_sha256=spec["qualification_receipt"]["sha256"]), qouter)
                index = bound("invented_data_index", dict(sequences=[dict(sequence=name, rgb=list(range(n)))
                    for name, n in worker.COHORT]))
                context = dict(fingerprints={}, installation=installed,
                    prediction_receipt=dict(source_sha256=sources, rows=[dict(object_ids=[1]) for _ in range(4)]),
                    prediction_spec=dict(data_index=index), proof=dict(predict_receipt={"manufactured": True}))
                decoded = []
                def read(desc, fingerprints):
                    decoded.append(desc["path"])
                    fingerprints[desc["path"]] = desc["sha256"]
                    return worker.load_json(desc["path"])
                outer_targets = []
                def clean_outer(desc, digest, fingerprints):
                    outer_targets.append(digest)
                    value = worker.load_json(desc["path"])
                    worker.require(value["output_sha256"] == digest, "manufactured outer result binding")
                    return value
                old = SimpleNamespace(bound_json=read, evaluation_context=lambda oldspec: context,
                                      clean_outer=clean_outer)
                record = {}
                with patch.object(worker, "D35_SPEC_SHA", spec["d35_spec"]["sha256"]), \
                        patch.object(worker, "D35_RESULT_SHA", spec["d35_result"]["sha256"]):
                    if change is None:
                        returned, rows = worker.admission(spec, record, None, old, here)
                        self.assertIs(returned, context)
                        self.assertEqual(sum(len(r["rgb"]) for r in rows), 446)
                    else:
                        reason = {"failed": "qualification required", "command": "raw argv/spec/output",
                                  "outer_result": "outer result binding"}[change]
                        with self.assertRaisesRegex(ValueError, reason):
                            worker.admission(spec, record, None, old, here)
                        self.assertNotIn(index["path"], decoded)
                self.assertEqual(outer_targets[0], spec["d35_receipt"]["sha256"])
                # Prior metric/result JSONs are only byte/hash bindings preseal.
                self.assertNotIn(spec["d35_result"]["path"], decoded)
                self.assertNotIn(spec["d37_receipt"]["path"], decoded)

    def test_source_tamper_rejected_before_runtime_or_output(self):
        with TemporaryDirectory() as directory:
            f = RunFixture(directory)
            (f.here / "operators.py").write_text("# modified after specification\n")
            with f.patched(), self.assertRaisesRegex(ValueError, "source closure"):
                worker.run(f.spec_path)
            self.assertEqual(f.events, [])
            self.assertFalse(f.output.exists())

    def test_existing_output_is_not_overwritten(self):
        with TemporaryDirectory() as directory:
            f = RunFixture(directory)
            f.output.mkdir()
            marker = f.output / "original.txt"
            marker.write_text("preserve")
            with f.patched(), self.assertRaisesRegex(ValueError, "fresh single-operation"):
                worker.run(f.spec_path)
            self.assertEqual(marker.read_text(), "preserve")
            self.assertEqual(f.events, [])

    def test_failed_admission_or_model_binding_cannot_reach_predictions_or_gt(self):
        for failure in ("admission", "model_hash"):
            with self.subTest(failure=failure), TemporaryDirectory() as directory:
                f = RunFixture(directory)
                f.failure = failure
                with f.patched(), self.assertRaises(ValueError):
                    worker.run(f.spec_path)
                self.assertFalse(any(e.startswith("prediction:") for e in f.events))
                self.assertNotIn("GT", f.events)
                if failure == "admission":
                    self.assertNotIn("runtime", f.events)
                self.assertEqual(worker.load_json(f.output / "receipt.json")["status"], "failed")

    def test_partial_prediction_is_preserved_but_never_sealed_or_scored(self):
        with TemporaryDirectory() as directory:
            f = RunFixture(directory)
            f.failure = "prediction"
            with f.patched(), self.assertRaisesRegex(RuntimeError, "after first saved"):
                worker.run(f.spec_path)
            self.assertEqual(len(list(f.output.rglob("*.npz"))), 1)
            self.assertFalse((f.output / "prediction-seal.json").exists())
            self.assertNotIn("GT", f.events)
            receipt = worker.load_json(f.output / "receipt.json")
            self.assertEqual(receipt["completed_frames"], 1)
            self.assertFalse(receipt["eligible"])

    def test_one_missing_final_frame_prevents_gt_even_when_all_sequences_return(self):
        with TemporaryDirectory() as directory:
            f = RunFixture(directory)
            f.incomplete = True
            with f.patched(), self.assertRaisesRegex(ValueError, "prediction/initial-only"):
                worker.run(f.spec_path)
            self.assertEqual(worker.load_json(f.output / "receipt.json")["completed_frames"], 445)
            self.assertFalse((f.output / "prediction-seal.json").exists())
            self.assertNotIn("GT", f.events)

    def test_complete_prediction_is_immutable_before_and_after_gt_boundary(self):
        for mutate in (False, True):
            with self.subTest(mutate=mutate), TemporaryDirectory() as directory:
                f = RunFixture(directory)
                f.mutate_seal = mutate
                with f.patched():
                    if mutate:
                        with self.assertRaisesRegex(ValueError, "immutable prediction barrier"):
                            worker.run(f.spec_path)
                    else:
                        answer = worker.run(f.spec_path)
                        self.assertTrue(answer["eligible"])
                        self.assertFalse(answer["scientific_eligible"])
                self.assertEqual(f.events.count("GT"), 1)
                receipt = worker.load_json(f.output / "receipt.json")
                self.assertEqual(receipt["status"], "failed" if mutate else "completed")

    def test_budget_checkpoint_preserves_failure_diagnostics_before_raising(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            record = dict(monotonic_start=time.perf_counter() - 10, operation_wall_seconds=1,
                max_output_bytes=1000, bytes_written=0, partial_output_bytes=0, status="started")
            with self.assertRaisesRegex(ValueError, "wall budget"):
                worker.checkpoint(SimpleNamespace(save=save), out, record)
            self.assertGreater(worker.load_json(out / "receipt.json")["wall_seconds"], 1)

    def test_exact_calls_reject_extra_encoder_and_incomplete_native_call(self):
        # Four invented frames: 4 native writes; two eligible side positions.
        native = [4, 4, 4, 3, 4]
        writes = [0, 0, 4, 0, 4]
        readouts = [0, 10, 0, 16, 10]
        calls = {phase: {kind + suffix: count for kind, count in zip(worker.CALL_KINDS, values)
                        for suffix in ("_intent", "_completed")}
                 for phase, values in zip(worker.PHASES, (native, writes, readouts))}
        worker.verify_counts(calls, 4)
        calls["readouts"]["image_encoder_intent"] = 1
        calls["readouts"]["image_encoder_completed"] = 1
        with self.assertRaisesRegex(ValueError, "readouts/image_encoder"):
            worker.verify_counts(calls, 4)
        calls["readouts"]["image_encoder_intent"] = calls["readouts"]["image_encoder_completed"] = 0
        calls["native"]["mask_decoder_completed"] -= 1
        with self.assertRaisesRegex(ValueError, "native/mask_decoder"):
            worker.verify_counts(calls, 4)

    def test_native_parity_checks_all_slots_not_only_selected_output(self):
        import numpy as np
        import torch
        with TemporaryDirectory() as directory:
            path = Path(directory) / "manufactured-bank.npz"
            low = torch.stack([torch.full((256, 256), x) for x in (-1., 2., 1.)])[None]
            quality, presence = torch.tensor([[.2, .9, .4]]), torch.tensor([[1.]])
            selected, pointer = low[:, 1:2], torch.zeros(1, 256)
            resize = lambda value: torch.nn.functional.interpolate(value, (2, 3), mode="bilinear", align_corners=False)
            original = resize(selected)
            bank = np.stack([original[:, 0].numpy(), resize(low[:, 0:1])[:, 0].numpy(),
                             resize(low[:, 2:3])[:, 0].numpy()], axis=1)
            names = ("B", "B+M1", "B+M2", "B+M1+M2")
            def compose(values, ids):
                return {name: (np.max(values[:, indices], axis=1)[0] > 0).astype(np.uint8)
                        for name, indices in zip(names, ([0], [0, 1], [0, 2], [0, 1, 2]))}
            controls = compose(bank, [1])
            np.savez(path, low_bank=low.numpy(), native_original_logits=original[:, 0].numpy(),
                     predicted_quality=quality.numpy(), object_score_logits=presence.numpy().reshape(1))
            sam = (low, None, quality, selected, None, pointer, presence)
            compact = dict(maskmem_features=torch.zeros(1), maskmem_pos_enc=[torch.zeros(1)],
                           pred_masks=selected, obj_ptr=pointer, object_score_logits=presence)
            state = {"output_dict_per_obj": {0: {"non_cond_frame_outputs": {0: compact}}}}
            # Frame 1 selects the noninitial bucket; row arrays remain ordinal-indexed.
            state["output_dict_per_obj"][0]["non_cond_frame_outputs"][1] = compact
            row = dict(frame_records=[None, dict(selected_slots=[1], alternative_slots=[[0, 2]])],
                       predictions=[None, dict(manufactured=True)])
            capture = dict(sam_outputs=sam, selected_index=1)
            args = (SimpleNamespace(_get_orig_video_res_output=lambda state, value: (None, resize(value))),
                dict(np=np, torch=torch), state, capture, original, row, 1,
                dict(prediction_root=Path(directory)), SimpleNamespace(COMPACT=set(compact)),
                SimpleNamespace(frame_path=lambda root, desc: path,
                    reconstruct_frame=lambda *args, **kwargs: (bank.copy(), controls)),
                SimpleNamespace(CONTROLS=names, compose_controls=compose))
            result = worker.compare_native(*args)
            self.assertTrue(result["bank_exact"] and result["controls_exact"])
            # The selected mask is unchanged; only a losing alternative is corrupted.
            low[0, 0, 0, 0] = -2.
            with self.assertRaisesRegex(ValueError, "bank/quality/presence parity"):
                worker.compare_native(*args)


if __name__ == "__main__":
    unittest.main()
