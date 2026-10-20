"""Manufactured tests only: tiny integers/logits; no files, model, or dataset."""

import itertools
import unittest

import numpy as np

import metrics as m


class MetricsTests(unittest.TestCase):
    def test_competition_positive_ties_and_zero_background(self):
        bank = np.full((2, 3, 1, 4), -2.0)
        bank[0, 0, 0] = [1, 2, 0, -1]
        bank[1, 0, 0] = [1, 1, 0, -2]
        bank[1, 1, 0] = [3, 2, 0, 1]
        original = bank.copy()
        result = m.compose_controls(bank, [1, 2])
        np.testing.assert_array_equal(result["B"], [[1, 1, 0, 0]])
        np.testing.assert_array_equal(result["B+M1"], [[2, 1, 0, 2]])
        np.testing.assert_array_equal(result["B+M2"], result["B"])
        np.testing.assert_array_equal(bank, original)
        # Adding object-2 support can REMOVE object-1 categorical assignment.
        self.assertNotEqual(result["B"][0, 0], result["B+M1"][0, 0])

    def test_exhaustive_support_oracle_is_simultaneous_j_maximum(self):
        # All 64 two-object support patterns on three pixels; every allowed
        # categorical assignment; every 0/1/2 GT assignment (primary policy).
        for bits in itertools.product((False, True), repeat=6):
            support = np.asarray(bits).reshape(2, 1, 3)
            bank = np.where(support[:, None], 1.0, -1.0)
            bank = np.repeat(bank, 3, axis=1)
            choices = [[0] + [k + 1 for k in range(2) if support[k, 0, x]]
                       for x in range(3)]
            for truth_values in itertools.product((0, 1, 2), repeat=3):
                truth = np.asarray([truth_values], dtype=np.uint8)
                oracle = m.support_oracles(bank, truth, (1, 2))["O"]
                ceiling = m.score_frame(truth, {"O": oracle}, (1, 2))["O"]
                for prediction_values in itertools.product(*choices):
                    pred = np.asarray([prediction_values], dtype=np.uint8)
                    score = m.score_frame(truth, {"P": pred}, (1, 2))["P"]
                    for actual, maximum in zip(score, ceiling):
                        self.assertLessEqual(actual, maximum)
                np.testing.assert_array_equal(
                    oracle, np.where(
                        ((truth == 1) & support[0]) |
                        ((truth == 2) & support[1]), truth, 0))

    def test_decomposition_separates_native_cleanup_from_new_support(self):
        truth = np.array([[1, 1, 0]], dtype=np.uint8)
        bank = np.full((1, 3, 1, 3), -1.0)
        bank[0, 0, 0] = [1, -1, 1]  # TP1 / union3
        bank[0, 1, 0] = [-1, 1, -1]
        predictions = {**m.compose_controls(bank, [1]),
                       **m.support_oracles(bank, truth, [1])}
        row = m.score_frame(truth, predictions, [1])
        report = m.summarize_sequence("tiny", [row] * 5, [1])
        delta = report["oracle_decomposition"]["J"]
        self.assertAlmostEqual(delta["total_O_minus_B"], 2 / 3)
        self.assertAlmostEqual(delta["native_cleanup_and_ownership_O_B_minus_B"], 1 / 6)
        self.assertAlmostEqual(delta["additional_support_O_minus_O_B"], 1 / 2)

    def test_void_background_and_ignore_diverge(self):
        truth = np.array([[1, 255, 0]], dtype=np.uint8)
        pred = np.array([[1, 1, 0]], dtype=np.uint8)
        primary = m.score_frame(truth, {"B": pred}, [1])["B"]
        sensitivity = m.score_frame(
            truth, {"B": pred}, [1], void_policy="void_ignore")["B"]
        self.assertEqual(primary, [0.5])
        self.assertEqual(sensitivity, [1.0])
        bank = np.ones((1, 3, 1, 3))
        for policy in m.POLICIES:
            oracle = m.support_oracles(bank, truth, [1], void_policy=policy)["O"]
            np.testing.assert_array_equal(oracle, [[1, 0, 0]])
            self.assertEqual(m.score_frame(
                truth, {"O": oracle}, [1], void_policy=policy)["O"], [1.0])

    def test_empty_object_and_empty_valid_domain(self):
        truth = np.zeros((1, 2), dtype=np.uint8)
        self.assertEqual(m.score_frame(truth, {"B": truth}, [1])["B"], [1.0])
        self.assertEqual(m.score_frame(truth, {"B": np.ones_like(truth)}, [1])["B"], [0.0])
        void = np.full_like(truth, 255)
        self.assertEqual(m.score_frame(
            void, {"B": np.ones_like(truth)}, [1], void_policy="void_ignore")["B"], [1.0])

    def test_endpoint_exclusion_and_tail_after_exclusion(self):
        self.assertEqual(m.frame_intervals(7), {"J": [1, 2, 3, 4, 5], "Jtr": [4, 5]})
        rows = [{"B": [value]} for value in [1, 0, .2, .4, .6, .8, 1]]
        report = m.summarize_sequence("interval", rows, [1])
        self.assertAlmostEqual(report["methods"]["B"]["J"], .4)
        self.assertAlmostEqual(report["methods"]["B"]["Jtr"], .7)
        self.assertEqual(len(report["methods"]["B"]["per_object"][0]["frame_J"]), 7)
        for n, count, first, last in [(86, 21, 64, 84), (90, 22, 67, 88),
                                      (201, 50, 150, 199), (69, 17, 51, 67)]:
            tail = m.frame_intervals(n)["Jtr"]
            self.assertEqual((len(tail), tail[0], tail[-1]), (count, first, last))

    def test_equal_object_weight_not_equal_video_or_frame_pooling(self):
        first = m.summarize_sequence("one_object", [{"B": [1.0]}] * 3, [1])
        second = m.summarize_sequence("two_objects", [{"B": [0.0, 0.0]}] * 9, [1, 2])
        total = m.aggregate_sequences([first, second])
        self.assertEqual(total["object_count"], 3)
        self.assertAlmostEqual(total["methods"]["B"]["J"], 1 / 3)
        self.assertAlmostEqual(total["methods"]["B"]["Jtr"], 1 / 3)
        self.assertNotEqual(total["methods"]["B"]["J"], .5)

    def test_stacked_and_streaming_agree_all_ids_and_raw_endpoints(self):
        truth = np.array([[[1, 2, 0]], [[1, 0, 2]], [[0, 2, 1]],
                          [[1, 2, 0]], [[1, 2, 0]]], dtype=np.uint8)
        predictions = {"B": truth.copy()}
        predictions["B"][2] = [[0, 2, 0]]
        stacked = m.score_sequence("stream", truth, predictions)
        rows = [m.score_frame(t, {"B": p}, [1, 2])
                for t, p in zip(truth, predictions["B"])]
        self.assertEqual(stacked, m.summarize_sequence("stream", rows, [1, 2]))
        self.assertEqual(stacked["object_ids"], [1, 2])

    def test_contract_failures_do_not_drop_invalid_rows(self):
        with self.assertRaises(ValueError):
            m.initial_object_ids(np.array([[1, 3]], dtype=np.uint8))
        with self.assertRaises(ValueError):
            m.initial_object_ids(np.array([[0, 255]], dtype=np.uint8))
        with self.assertRaises(ValueError):
            m.compose_controls(np.full((1, 3, 1, 1), np.nan), [1])
        with self.assertRaises(ValueError):
            m.compose_controls(np.ones((2, 3, 1, 1)), [2, 1])
        with self.assertRaises(ValueError):
            m.score_frame(np.array([[2]]), {"B": np.array([[0]])}, [1])
        with self.assertRaises(ValueError):
            m.score_frame(np.array([[1]]), {"B": np.array([[255]])}, [1])
        with self.assertRaises(ValueError):
            m.summarize_sequence("bad", [{"B": [1]}, {"B": [float("nan")]},
                                         {"B": [1]}], [1])
        truth = np.ones((3, 1, 1), dtype=np.uint8)
        with self.assertRaises(ValueError):
            m.score_sequence("incomplete", truth, {"B": truth[:2]})
        with self.assertRaises(ValueError):
            m.score_sequence("ids", truth, {"B": truth}, [1, 2])
        primary = m.summarize_sequence("p", [{"B": [1]}] * 3, [1])
        other = m.summarize_sequence("q", [{"B": [1]}] * 3, [1], void_policy="void_ignore")
        with self.assertRaises(ValueError):
            m.aggregate_sequences([primary, other])


if __name__ == "__main__":
    unittest.main()
