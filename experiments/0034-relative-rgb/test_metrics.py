"""Manufactured saved-scalar fixtures only; no scientific files or model imports."""
from copy import deepcopy
import math
import unittest

import metrics as m


def score(j, f=None):
    f = j if f is None else f
    return dict(J=j, F=f, JF=(j + f) / 2)


def fixture():
    rows = []
    for sequence, frames in m.PARENTS:
        for frame in range(frames):
            arms = {arm: score(.51) for arm in m.ARMS}
            arms.update(native=score(.5), signed_gram=score(.53), displaced_gram=score(.52),
                        rgb256=score(.528))
            rows.append(dict(sequence=sequence, frame=frame, frames=frames, target_id=2,
                             arms=arms, diagnostic=dict(native_noop=False, reason=None)))
    return rows


def change(rows, arm, j, f=None, parents=None, interiors_only=False):
    names = set(parents) if parents is not None else {name for name, _ in m.PARENTS}
    for row in rows:
        if row["sequence"] in names and (not interiors_only or 0 < row["frame"] < row["frames"] - 1):
            row["arms"][arm] = score(j, f)


class MetricsTests(unittest.TestCase):
    def test_complete_population_and_secondary_duplicates(self):
        rows = fixture()
        original = deepcopy(rows)
        report = m.build_report(list(reversed(rows)), [])
        self.assertTrue(report["decision"]["passed"])
        self.assertEqual(report["coverage"]["raw_frames"], 232)
        self.assertEqual(report["coverage"]["raw_arm_rows"], 2552)
        self.assertEqual([p["included_count"] for p in report["primary"]["parents"]], [23, 36, 81, 84])
        secondary = report["secondary"]
        self.assertEqual((secondary["event_count"], secondary["unique_frames"], secondary["final_frame_events"]),
                         (96, 87, 4))
        self.assertEqual([p["included_count"] for p in secondary["parents"]], [24] * 4)
        self.assertLess(len({(r["sequence"], r["frame"]) for r in secondary["events"]}), 96)
        self.assertEqual(report["rows"], original)
        report["rows"][0]["diagnostic"]["reason"] = "copy only"
        self.assertEqual(rows, original)

    def test_endpoint_exclusion_but_secondary_keeps_final(self):
        rows = fixture()
        for row in rows:
            row["arms"]["native"] = score(1 if row["frame"] in (0, row["frames"] - 1) else 0)
        report = m.build_report(rows, [])
        self.assertEqual(report["primary"]["mean"]["native"]["JF"], 0)
        self.assertEqual(report["secondary"]["mean"]["native"]["JF"], 1 / 24)
        self.assertEqual(sum(r["arms"]["native"]["JF"] for r in report["secondary"]["events"]), 4)

    def test_parent_equal_not_pooled_frame_weight(self):
        rows = fixture()
        levels = dict(zip((name for name, _ in m.PARENTS), (.125, .25, .375, .5)))
        for row in rows:
            row["arms"]["native"] = score(levels[row["sequence"]])
        report = m.build_report(rows, [])
        self.assertEqual(report["primary"]["mean"]["native"]["JF"], .3125)
        pooled = math.fsum(levels[name] * (n - 2) for name, n in m.PARENTS) / 224
        self.assertNotEqual(pooled, .3125)

    def test_strongest_control_fixed_tie_order(self):
        rows = fixture()
        report = m.build_report(rows, [])
        self.assertEqual(report["decision"]["strongest_control"]["arm"], "logit")
        change(rows, "bilateral", .52)
        self.assertEqual(m.build_report(rows, [])["decision"]["strongest_control"]["arm"], "bilateral")

    def test_each_numerical_clause_can_reject(self):
        cases = [
            ("primary_JF_gain_at_least_1pp", lambda r: change(r, "signed_gram", .505)),
            ("margin_over_strongest_control_at_least_quarter_pp", lambda r: change(r, "absolute", .529)),
            ("beats_ordinary_relations_on_at_least_three_parents",
             lambda r: change(r, "ordinary_relations", .54, parents=[x[0] for x in m.PARENTS[:2]])),
            ("beats_signed_no_gram_on_at_least_three_parents",
             lambda r: change(r, "signed_no_gram", .54, parents=[x[0] for x in m.PARENTS[:2]])),
            ("no_parent_native_JF_loss_exceeds_half_pp",
             lambda r: change(r, "signed_gram", .49, parents=[m.PARENTS[0][0]])),
            ("mean_native_J_loss_at_most_half_pp", lambda r: change(r, "signed_gram", .49, .57)),
            ("mean_native_F_loss_at_most_half_pp", lambda r: change(r, "signed_gram", .57, .49)),
            ("mean_JF_above_displaced_gram", lambda r: change(r, "displaced_gram", .53)),
            ("mean_JF_above_rgb256_by_at_least_tenth_pp", lambda r: change(r, "rgb256", .5295)),
        ]
        for clause, mutate in cases:
            with self.subTest(clause=clause):
                rows = fixture()
                mutate(rows)
                report = m.build_report(rows, [])
                self.assertFalse(report["decision"]["clauses"][clause])
                self.assertFalse(report["decision"]["passed"])

    def test_no_epsilon_at_native_gain_boundary(self):
        rows = fixture()
        change(rows, "native", 0)
        change(rows, "signed_gram", .01)
        self.assertTrue(m.build_report(rows, [])["decision"]["clauses"]["primary_JF_gain_at_least_1pp"])
        change(rows, "signed_gram", .01 - 1e-12)
        self.assertFalse(m.build_report(rows, [])["decision"]["clauses"]["primary_JF_gain_at_least_1pp"])

    def test_failures_preserved_even_with_finite_fallback_metrics(self):
        failures = [dict(fold="dog-agility", arm="nn", status="numerical_failure", reason="manufactured")]
        report = m.build_report(fixture(), failures)
        self.assertEqual(report["fit_failures"], failures)
        self.assertTrue(report["decision"]["clauses"]["full_coverage"])
        self.assertFalse(report["decision"]["clauses"]["no_fit_failures"])
        self.assertFalse(report["decision"]["passed"])
        self.assertFalse(report["goal_achieved"])
        self.assertFalse(report["official_benchmark_result"])

    def test_incomplete_duplicate_or_wrong_identity_rejected(self):
        cases = []
        rows = fixture(); rows.pop(); cases.append(rows)
        rows = fixture(); rows[-1] = deepcopy(rows[0]); cases.append(rows)
        rows = fixture(); rows[0]["sequence"] = "DEV"; cases.append(rows)
        rows = fixture(); rows[0]["frames"] += 1; cases.append(rows)
        rows = fixture(); rows[0]["frame"] = True; cases.append(rows)
        rows = fixture(); rows[0]["target_id"] = 3; cases.append(rows)
        rows = fixture(); del rows[0]["arms"]["nn"]; cases.append(rows)
        for rows in cases:
            with self.subTest(row=rows[0]):
                with self.assertRaises(ValueError):
                    m.build_report(rows, [])

    def test_nonfinite_range_or_inconsistent_metrics_rejected(self):
        for value in (float("nan"), float("inf"), -.01, 1.01, True, "0.5"):
            rows = fixture()
            rows[0]["arms"]["native"]["J"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                m.build_report(rows, [])
        rows = fixture()
        rows[0]["arms"]["native"]["JF"] = .6
        with self.assertRaises(ValueError):
            m.build_report(rows, [])


if __name__ == "__main__":
    unittest.main()
