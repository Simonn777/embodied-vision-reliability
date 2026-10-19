"""Independent cohort barrier probes using artificial files and fake loaders.

No HDF5 parser, scientific asset, image or native estimator is accessed. The
shared fixture supplies opaque text files and manufactured two-point geometry.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import test_diagnose_trace as fixture


class IndependentBarrierTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fake = fixture.cohort_fixture(self.root)
        self.output = self.root / "diagnosis"

    def assert_blocks_before_labels(self, expected_completed):
        f, d = self.fake, fixture.d
        with mock.patch.object(d, "_load_reference",
                side_effect=AssertionError("No reference access is admitted")) as labels:
            code = d.run(f.data, f.manifest, f.baseline, f.seal, self.output,
                         _modules=(f.c, f.runner), _pair_loader=f.loader)
        self.assertEqual(code, 1)
        labels.assert_not_called()
        receipt = json.loads((self.output / "operation.json").read_text())
        self.assertEqual(receipt["status"], "ineligible")
        self.assertEqual(receipt["failed_phase"], "reconstruction")
        self.assertEqual(receipt["completed_reconstructions"], expected_completed)
        self.assertFalse((self.output / "reconstruction-seal.json").exists())
        self.assertFalse((self.output / "evaluation.json").exists())
        self.assertEqual(list(self.output.glob("*.labels.json")), [])
        return receipt

    def test_second_whole_seal_failure_after_160_blocks_labels(self):
        f = self.fake
        f.runner.verify_seal.side_effect = [f.runner.verify_seal.return_value,
                                           ValueError("Changed second barrier")]
        receipt = self.assert_blocks_before_labels(160)
        self.assertIn("Changed second barrier", receipt["error"])
        self.assertEqual(f.runner.verify_seal.call_count, 2)
        self.assertEqual(f.loader.call_count, 80)

    def test_source_mutation_after_160_blocks_labels(self):
        self.fake.c.source_hashes = mock.Mock(side_effect=[
            {"artificial": "fixed"}, {"artificial": "changed"}])
        receipt = self.assert_blocks_before_labels(160)
        self.assertIn("Bindings changed during reconstruction", receipt["error"])
        self.assertEqual(self.fake.loader.call_count, 80)

    def test_serialized_output_cap_preserves_ineligible_receipt(self):
        # The small start marker fits; the first reconstructed case does not.
        with mock.patch.object(fixture.d, "OUTPUT_BYTES_LIMIT", 1000):
            receipt = self.assert_blocks_before_labels(0)
        self.assertIn("Serialized diagnostic output exceeds cohort cap", receipt["error"])
        self.assertTrue((self.output / "started.json").exists())
        self.assertEqual(list(self.output.glob("*.reconstruction.json")), [])

    def test_unsafe_last_trace_path_preserves_159_without_labels(self):
        self.fake.records[-1]["trace_path"] = "../outside.trace.json"
        receipt = self.assert_blocks_before_labels(159)
        self.assertIn("unsafe fixture path", receipt["error"])
        self.assertEqual(len(list(self.output.glob("*.reconstruction.json"))), 159)
        self.assertEqual(self.fake.loader.call_count, 80)


if __name__ == "__main__":
    unittest.main()
