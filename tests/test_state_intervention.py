"""Synthetic controller tests only: no torch, dataset, model or GT import."""
import math
import unittest

from evr.state_intervention import StateIntervention, clone_context, zero_context


class Tensor:
    def __init__(self, value):
        self.value = value
        self.shape = (1,)
        self.dtype = "float32"
        self.device = "fake:cpu"

    def clone(self):
        result = Tensor(self.value)
        result.shape, result.dtype, result.device = self.shape, self.dtype, self.device
        return result

    def zero_(self):
        self.value = 0.0
        return self


def context(value=0):
    return [{"history": Tensor(value)}]


def validate(prediction, state):
    if not math.isfinite(prediction["position"]):
        raise ValueError("nonfinite prediction")
    if any(not math.isfinite(t.value) for layer in state for t in layer.values()):
        raise ValueError("nonfinite state")


def backend(frame_value):
    # Intentionally mutates input, so aliasing bugs corrupt later branches.
    def infer(state):
        history = state[0]["history"]
        history.value = 2 * history.value + frame_value
        return {"position": history.value, "coarse": frame_value}, state
    return infer


def controller(**kwargs):
    options = dict(total_frames=8, shadow_starts=(2,), shadow_horizon=3,
                   control_frames=(1, 4), validate_result=validate)
    options.update(kwargs)
    return StateIntervention(context(), **options)


class StateInterventionTests(unittest.TestCase):
    def test_keep_alone_advances_baseline_and_shadows_carry_independently(self):
        stream = controller()
        rows = [stream.step(t, backend(t + 1)) for t in range(8)]
        self.assertEqual([r.keep["position"] for r in rows],
                         [1, 4, 11, 26, 57, 120, 247, 502])
        self.assertEqual([r.zero["position"] for r in rows], list(range(1, 9)))
        shadows = [s for row in rows for s in row.shadows]
        self.assertEqual([(s.start, s.offset, s.prediction["position"]) for s in shadows],
                         [(2, 0, 3), (2, 1, 10), (2, 2, 25)])
        self.assertEqual(rows[1].clone_control, rows[1].keep)
        self.assertEqual(rows[4].clone_control, rows[4].keep)
        self.assertIsNone(rows[0].clone_control)
        self.assertEqual(stream.finish()["calls"],
                         {"keep": 8, "zero": 8, "clone_control": 2, "shadow": 3})

    def test_clone_breaks_upstream_list_and_dictionary_tensor_aliases(self):
        value = Tensor(7)
        layer = {"a": value, "b": value}
        original = [layer, layer]
        cloned = clone_context(original)
        zeroed = zero_context(original)
        leaves = [v for item in cloned + zeroed for v in item.values()]
        self.assertEqual(len({id(v) for v in leaves}), 8)
        cloned[0]["a"].value = 123
        self.assertEqual(original[0]["a"].value, 7)
        self.assertEqual(cloned[1]["a"].value, 7)
        self.assertTrue(all(t.value == 0 for layer in zeroed for t in layer.values()))
        self.assertEqual(zeroed[0]["a"].device, "fake:cpu")

    def test_caller_initial_state_and_saved_outputs_are_owned(self):
        initial = context(5)
        stream = StateIntervention(initial, total_frames=2, shadow_starts=(),
                                   shadow_horizon=1, control_frames=(),
                                   validate_result=validate)
        initial[0]["history"].value = 999
        first = stream.step(0, backend(1))
        first.keep["position"] = -1000
        self.assertEqual(stream.step(1, backend(2)).keep["position"], 24)
        stream.finish()

    def test_prefix_invariance_under_future_input_change(self):
        def run(values):
            stream = controller()
            return [stream.step(i, backend(value)) for i, value in enumerate(values)]
        before = run(list(range(1, 9)))
        changed = run([1, 2, 3, 4, 900, 800, 700, 600])
        self.assertEqual(before[:4], changed[:4])
        self.assertNotEqual(before[4], changed[4])

    def test_state_signature_change_fails_closed_without_committing_frame(self):
        stream = controller()
        def wrong(state):
            state[0]["history"].shape = (2,)
            return {"position": 1}, state
        with self.assertRaisesRegex(ValueError, "changed"):
            stream.step(0, wrong)
        self.assertEqual(stream.counts["completed_frames"], 0)
        self.assertEqual(stream.counts["calls"]["keep"], 1)
        with self.assertRaisesRegex(RuntimeError, "no further"):
            stream.step(0, backend(1))

    def test_failure_in_zero_branch_keeps_attempt_counts_and_forbids_retry(self):
        stream = controller()
        calls = []
        def fails_second(state):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("synthetic zero failure")
            return backend(1)(state)
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            stream.step(0, fails_second)
        self.assertTrue(stream.counts["failed"])
        self.assertEqual(stream.counts["completed_frames"], 0)
        self.assertEqual(stream.counts["total_calls"], 2)
        with self.assertRaises(RuntimeError):
            stream.finish()
        with self.assertRaises(RuntimeError):
            stream.step(0, backend(1))
        self.assertEqual(len(calls), 2)

    def test_numeric_validator_applies_to_nonbaseline_branches(self):
        stream = controller()
        count = 0
        def bad_zero(state):
            nonlocal count
            count += 1
            return {"position": 1 if count == 1 else float("nan")}, state
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            stream.step(0, bad_zero)
        self.assertEqual(stream.counts["completed_frames"], 0)

    def test_invalid_config_rejected_before_any_callback(self):
        cases = [dict(total_frames=True), dict(total_frames=0),
                 dict(shadow_horizon=0), dict(shadow_starts=(2, 2)),
                 dict(shadow_starts=(4, 2)), dict(shadow_starts=(2, 4)),
                 dict(shadow_starts=(6,)), dict(shadow_starts=(True,)),
                 dict(control_frames=(8,)), dict(control_frames=(1, 1)),
                 dict(validate_result=None)]
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                controller(**options)

    def test_frame_order_and_finish_are_strict(self):
        stream = controller()
        for invalid in (True, -1, 1, 0.0):
            with self.subTest(frame=invalid), self.assertRaises(ValueError):
                stream.step(invalid, backend(1))
        with self.assertRaises(RuntimeError):
            stream.finish()
        stream.step(0, backend(1))
        with self.assertRaises(ValueError):
            stream.step(0, backend(1))
        for i in range(1, 8):
            stream.step(i, backend(i + 1))
        self.assertEqual(stream.finish()["total_calls"], 21)
        with self.assertRaises(ValueError):
            stream.step(8, backend(9))

    def test_multiple_windows_reset_once_and_do_not_share_shadow_state(self):
        stream = controller(shadow_starts=(1, 5), shadow_horizon=2)
        rows = [stream.step(i, backend(i + 1)) for i in range(8)]
        self.assertEqual([(s.start, s.offset, s.prediction["position"])
                          for row in rows for s in row.shadows],
                         [(1, 0, 2), (1, 1, 7), (5, 0, 6), (5, 1, 19)])
        self.assertEqual(stream.finish()["calls"]["shadow"], 4)


if __name__ == "__main__":
    unittest.main()
