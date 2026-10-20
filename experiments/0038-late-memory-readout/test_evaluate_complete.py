"""Manufactured records/tensors only. No runtime, model, media, or task inputs."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

definition = importlib.util.spec_from_file_location("d38_evaluation_tested", Path(__file__).with_name("evaluate_complete.py"))
e = importlib.util.module_from_spec(definition)
definition.loader.exec_module(e)


def population():
    rows, sequences, old, inputs = [], [], [], []
    for name, count in e.COHORT:
        seq = dict(sequence=name, frames=count, status="completed", completed_frames=count,
            object_ids=[1], shape=[3, 5], final_state_exact=True,
            final_native_state_sha256="a"*64, expected_final_native_state_sha256="a"*64)
        sequences.append(seq)
        old.append(dict(seq))
        inputs.append(dict(sequence=name, rgb=[dict(path=f"/invented/{name}/{i:05d}.jpg") for i in range(count)]))
        for frame in range(count):
            rows.append(dict(sequence=name, frame=frame, filename=f"{frame:05d}.jpg",
                shape=[3, 5], object_ids=[1], output=dict(path=f"{name}/{frame}.npz", sha256="b"*64, bytes=1),
                native_parity={k: True for k in ("bank_exact", "output_exact", "input_exact",
                                               "controls_exact", "stored_write_exact")}))
    prediction = dict(schema="evr0038.complete-prediction.v1", status="completed", frames=446,
        completed_frames=446, completed_sequences=4, arms=list(e.READOUTS), rows=rows, sequences=sequences,
        outputs=[r["output"] for r in rows], source_sha256={}, spec_sha256="c"*64, bindings={})
    seal = {k: prediction[k] for k in ("outputs", "arms", "source_sha256", "spec_sha256", "bindings")}
    seal.update(schema="evr0038.prediction-seal.v1", complete=True, frames=446, sequences=4)
    return prediction, seal, dict(rows=old), dict(sequences=inputs)


def aggregate(candidate=.03):
    methods = {a: dict(J=0., Jtr=0.) for a in e.ARMS}
    methods["late_mean"] = dict(J=candidate, Jtr=candidate)
    return dict(methods=methods, sequence_count=4, object_count=4,
        per_sequence=[dict(sequence=n, frame_count=c, methods=copy.deepcopy(methods)) for n,c in e.COHORT])


def references():
    old, prior = {}, {}
    for policy, key in zip(e.POLICIES, e.REPORT_KEYS):
        sequences = []
        for name, count in e.COHORT:
            interior = list(range(1, count-1))
            raw = [.2 if policy == "author_background" else .4]*count
            methods = {a: dict(per_object=[dict(object_id=1, frame_J=raw[:])])
                       for a in e.FIXED + ("fixed_negative",)}
            sequences.append(dict(sequence=name, frame_count=count, object_ids=[1], methods=methods,
                frame_positions=dict(J=interior, Jtr=interior[3*len(interior)//4:])))
        value = dict(void_policy=policy, sequence_count=4, object_count=4, per_sequence=sequences)
        old[key], prior[key] = copy.deepcopy(value), copy.deepcopy(value)
    return old, prior


class EvaluationTests(unittest.TestCase):
    def test_historical_outer_receipt_and_its_output_are_distinct_bindings(self):
        spec = dict(d35_receipt=dict(path="/invented/receipt.json", sha256="r"*64),
            d35_result=dict(path="/invented/evaluation.json", sha256="e"*64),
            d35_spec=dict(sha256="s"*64))
        oldspec = dict(source_sha256={"/invented/source.py": "a"*64})
        receipt = dict(schema="evr0035.evaluation.v1", status="completed", spec_sha256="s"*64,
            source_sha256=oldspec["source_sha256"], completed_frames=446, completed_sequences=4,
            outputs=[dict(path="evaluation.json", sha256="e"*64, bytes=10)])
        e.validate_d35_receipt(spec, oldspec, receipt)
        for change in (lambda r: r.update(completed_frames=445),
                       lambda r: r["outputs"][0].update(sha256="r"*64),
                       lambda r: r.update(source_sha256={})):
            altered = copy.deepcopy(receipt)
            change(altered)
            with self.assertRaises(ValueError):
                e.validate_d35_receipt(spec, oldspec, altered)
        spec["d35_receipt"]["path"] = "/invented/evaluation.json"
        with self.assertRaises(ValueError):
            e.validate_d35_receipt(spec, oldspec, receipt)

    def test_complete_population_and_missing_final_or_false_parity_reject(self):
        args = population()
        e.validate_population(*args)
        self.assertEqual(len(args[0]["rows"]), 446)
        for change in (lambda p: p["rows"].pop(),
                       lambda p: p["rows"][20]["native_parity"].update(input_exact=False),
                       lambda p: p.update(status="started")):
            mutated = copy.deepcopy(args)
            change(mutated[0])
            with self.assertRaises(ValueError):
                e.validate_population(*mutated)

    def test_final_state_is_anchored_to_old_native(self):
        args = population()
        args[0]["sequences"][0].update(final_native_state_sha256="d"*64,
                                     expected_final_native_state_sha256="d"*64)
        with self.assertRaisesRegex(ValueError, "final-state"):
            e.validate_population(*args)

    def test_references_keep_void_policies_endpoints_and_reject_mismatch(self):
        old, prior = references()
        values = e.reference_scores(old, prior)
        first = e.COHORT[0][0]
        self.assertEqual(values["author_background"][first]["fixed_negative"], [.2]*86)
        self.assertEqual(values["void_ignore"][first]["fixed_negative"], [.4]*86)
        self.assertEqual(sum(len(s["frame_positions"]["J"]) for s in old["primary"]["per_sequence"]), 438)
        self.assertEqual(sum(len(s["frame_positions"]["Jtr"]) for s in old["primary"]["per_sequence"]), 110)
        prior["primary"]["per_sequence"][0]["methods"]["B"]["per_object"][0]["frame_J"][-1] = .1
        with self.assertRaisesRegex(ValueError, "fixed parity"):
            e.reference_scores(old, prior)
        old, prior = references()
        prior["primary"]["per_sequence"][0]["methods"]["fixed_negative"]["per_object"][0]["frame_J"][0] = float("nan")
        with self.assertRaises(ValueError):
            e.reference_scores(old, prior)

    def test_every_control_and_exact_threshold_no_epsilon(self):
        self.assertTrue(e.continuation(aggregate())["numerical_passed"])
        self.assertEqual(e.continuation(aggregate())["strongest_control_by_metric"], dict(J="B", Jtr="B"))
        for arm in e.CONTROLS:
            value = aggregate()
            value["methods"][arm]["Jtr"] = .0001
            result = e.continuation(value)
            self.assertEqual(result["strongest_control_by_metric"]["Jtr"], arm)
            self.assertFalse(result["clauses"]["over_every_control_3pp_Jtr"])
        self.assertFalse(e.continuation(aggregate(.03-1e-10))["numerical_passed"])

    def test_three_positive_clips_and_native_loss_floor(self):
        value = aggregate()
        for seq in value["per_sequence"][:2]:
            seq["methods"]["B+M1+M2"]["Jtr"] = .03
        self.assertFalse(e.continuation(value)["clauses"]["three_clips_positive_Jtr_over_union"])
        value = aggregate()
        seq = value["per_sequence"][0]
        seq["methods"]["late_mean"]["J"] = 0.
        seq["methods"]["B"]["J"] = .005
        self.assertTrue(e.continuation(value)["clauses"]["no_clip_native_loss_over_0_5pp"])
        seq["methods"]["B"]["J"] = .00500001
        self.assertFalse(e.continuation(value)["clauses"]["no_clip_native_loss_over_0_5pp"])

    def test_original_layout_masks_and_padding_reject(self):
        import numpy as np
        import torch
        low = np.empty((4, 1, 256, 256), dtype=np.float32)
        for i, value in enumerate((1., -1., 0., 2.)):
            low[i].fill(value)
        bits = np.stack([np.full((1, 15), v > 0, dtype=np.uint8) for v in (1., -1., 0., 2.)])
        packed = np.packbits(bits, axis=-1, bitorder="big")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"invented.npz"
            def write():
                np.savez_compressed(path, low_logits=low, packed_masks=packed,
                    object_ids=np.asarray([1], dtype=np.int64), shape=np.asarray([3, 5], dtype=np.int64))
            write()
            result = e.reconstruct_readouts(path, dict(shape=[3, 5], object_ids=[1]), np=np, torch=torch)
            self.assertEqual([int(result[a].sum()) for a in e.READOUTS], [15, 0, 0, 15])
            packed[0, 0, -1] |= 1
            write()
            with self.assertRaisesRegex(ValueError, "padding"):
                e.reconstruct_readouts(path, dict(shape=[3, 5], object_ids=[1]), np=np, torch=torch)
            packed[0, 0, -1] &= 254
            packed[0, 0, 0] ^= 128
            write()
            with self.assertRaisesRegex(ValueError, "mask differs"):
                e.reconstruct_readouts(path, dict(shape=[3, 5], object_ids=[1]), np=np, torch=torch)


if __name__ == "__main__":
    unittest.main()
