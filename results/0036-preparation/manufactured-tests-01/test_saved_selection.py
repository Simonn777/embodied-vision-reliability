"""Manufactured scalar cohorts only; never opens the real D35 result."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("saved_selection_tested", Path(__file__).with_name("saved_selection.py"))
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)


def cohort(mode):
    rows = []
    for k, (name, n) in enumerate(selection.COHORT):
        positions = {"J": list(range(1, n - 1)), "Jtr": list(range(1 + 3 * (n - 2) // 4, n - 1))}
        values = {arm: [.25] * n for arm in selection.ARMS}
        if mode == "alternating":
            for frame in range(n):
                values[selection.ARMS[frame % 4]][frame] = .75
        elif mode == "clip_constant":
            values[selection.ARMS[k]] = [.75] * n
        elif mode != "ties":
            raise ValueError(mode)
        values.update(O_B=[.875] * n, O=[1.] * n)
        rows.append(dict(sequence=name, frame_count=n, object_ids=[1], frame_positions=positions,
                         methods={arm: {"per_object": [{"object_id": 1, "frame_J": raw}]}
                                  for arm, raw in values.items()}))
    report = dict(aggregation="equal_sequence_object_means", sequence_count=4, object_count=4, per_sequence=rows)
    return dict(schema="evr0035.metrics.v1",
                primary=dict(copy.deepcopy(report), void_policy="author_background"),
                void_ignore_sensitivity=dict(copy.deepcopy(report), void_policy="void_ignore"))


class Tests(unittest.TestCase):
    def test_ties_keep_native_and_endpoints_do_not_change_primary(self):
        data = cohort("ties")
        for policy in ("primary", "void_ignore_sensitivity"):
            for row in data[policy]["per_sequence"]:
                for arm in selection.ARMS:
                    row["methods"][arm]["per_object"][0]["frame_J"][0] = 0
                    row["methods"][arm]["per_object"][0]["frame_J"][-1] = 0
        result = selection.analyze(data)["primary"]
        self.assertFalse(result["whole_arm_support_sufficient"])
        self.assertEqual(result["counts"], dict(frames=446, interior=438, tail=110, objects=4))
        self.assertEqual(result["per_frame_oracle"]["values"], dict(J=.25, Jtr=.25))
        self.assertTrue(all(set(r["best_arm_by_frame"]) == {"B"} for r in result["raw_sequences"]))
        self.assertEqual(sum(len(r["best_arm_by_frame"]) for r in result["raw_sequences"]), 446)
        self.assertEqual(set(result["raw_sequences"][0]["frame_J"]), set(selection.ALL_ARMS))

    def test_alternating_frame_oracle_beats_constant_and_uses_fixed_order(self):
        result = selection.analyze(cohort("alternating"))["primary"]
        self.assertTrue(result["whole_arm_support_sufficient"])
        self.assertEqual(result["per_frame_oracle"]["values"], dict(J=.75, Jtr=.75))
        self.assertEqual(result["raw_sequences"][0]["best_arm_by_frame"][:4], list(selection.ARMS))
        self.assertLess(result["per_clip_constant_oracle"]["J"]["value"], .6)
        self.assertAlmostEqual(result["per_frame_oracle"]["gap_to_pixel_O"]["J"], .25)
        for row in result["raw_sequences"]:
            self.assertEqual(row["per_frame_oracle"], dict(J=.75, Jtr=.75))
            self.assertEqual(set(row["fixed_scores"]), set(selection.ALL_ARMS))
            self.assertAlmostEqual(row["per_frame_oracle_gain_over_strongest_global_fixed"]["J"],
                                   .75 - result["strongest_global_fixed"]["J"]["value"])

    def test_clip_constant_is_equal_clip_not_length_weighted_and_rejects_invalid(self):
        data = cohort("clip_constant")
        result = selection.analyze(data)["primary"]
        self.assertEqual(result["strongest_global_fixed"]["J"]["arm"], "B")
        self.assertEqual(result["strongest_global_fixed"]["Jtr"]["arm"], "B")
        for metric in ("J", "Jtr"):
            self.assertAlmostEqual(result["strongest_global_fixed"][metric]["value"], .375)
            self.assertEqual(result["per_clip_constant_oracle"][metric]["value"], .75)
            self.assertAlmostEqual(result["per_frame_oracle"]["gain_over_strongest_global_fixed"][metric], .375)
        for change in ("nan", "short", "exceeds_O", "missing_clip"):
            bad = copy.deepcopy(data)
            row = bad["primary"]["per_sequence"][0]
            if change == "nan":
                row["methods"]["O_B"]["per_object"][0]["frame_J"][3] = float("nan")
            elif change == "short":
                row["methods"]["B"]["per_object"][0]["frame_J"].pop()
            elif change == "exceeds_O":
                row["methods"]["O"]["per_object"][0]["frame_J"][1] = .1
            else:
                bad["primary"]["per_sequence"].pop()
            with self.subTest(change=change), self.assertRaises(ValueError):
                selection.analyze(bad)

    def test_run_rejects_source_or_input_before_analysis_and_preserves_failure(self):
        for kind in ("source", "input"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source_sha = selection.digest(Path(selection.__file__).read_bytes())
                (root / "invented.json").write_text("{}")
                spec = dict(schema="evr0036.saved-selection-spec.v1",
                            input=dict(path=str(root / "invented.json"), sha256=selection.INPUT_SHA),
                            source_sha256="0" * 64 if kind == "source" else source_sha,
                            output=str(root / "output"))
                (root / "spec.json").write_text(json.dumps(spec))
                with patch.object(selection, "analyze") as analyze, self.assertRaisesRegex(
                        ValueError, "executing source" if kind == "source" else "saved input"):
                    selection.run(root / "spec.json")
                analyze.assert_not_called()
                receipt = json.loads((root / "output/receipt.json").read_text())
                self.assertEqual(receipt["status"], "failed")
                self.assertEqual(receipt["error"]["type"], "ValueError")
                self.assertFalse((root / "output/report.json").exists())


if __name__ == "__main__":
    unittest.main()
