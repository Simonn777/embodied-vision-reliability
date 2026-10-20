"""Small manufactured contract checks; no Torch/model/decoder/media imports."""
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np

spec = importlib.util.spec_from_file_location("evidence_subject", Path(__file__).with_name("frontend_evidence.py"))
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)
COHORT = (("dogs-scale",83), ("upside-down",65), ("dog-agility",25),
          ("cat-girl",89), ("planes-water",38), ("dog-gooses",86))


def schedule(n):
    return [4 + (2*j*(n-5)+7)//14 for j in range(8)]


class FrontendContracts(unittest.TestCase):
    def test_path_records_and_early_interruption_keep_all_pending_rows(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            cohorts = []
            for sequence, n in COHORT:
                def media(frame, suffix):
                    return {"path": root/sequence/("%05d.%s" % (frame,suffix)),
                            "bytes": 7, "sha256": "a"*64}
                cohorts.append({"sequence":sequence, "frame_count":n,
                    "initial_mask":media(0,"png"),
                    "images":[{"frame":frame, **media(frame,"jpg")} for frame in [0,*schedule(n)]]})
            inputs = {"source_sha256":{"fake":"b"*64}, "qualify_helper":root/"unused_q.py",
                      "diagnostic_core":root/"unused_core.py", "cohort":cohorts}
            fake = {name:ModuleType(name) for name in
                    ("numpy","torch","cv2","PIL","kornia","kornia.config","lightglue")}
            fake["PIL"].Image = object()
            fake["kornia"].__path__ = []
            fake["kornia.config"].InstallationMode = SimpleNamespace(RAISE="raise")
            fake["kornia.config"].kornia_config = SimpleNamespace(lazyloader=SimpleNamespace())
            fake["lightglue"].SuperPoint = mock.Mock(side_effect=AssertionError("model must not be created"))
            fake["lightglue"].LightGlue = mock.Mock(side_effect=AssertionError("model must not be created"))
            fake["torch"].set_num_threads = mock.Mock(side_effect=KeyboardInterrupt("synthetic early stop"))
            q = SimpleNamespace(capture_warnings=lambda _:contextlib.nullcontext())
            core = SimpleNamespace(COHORT=COHORT, query_schedule=schedule)
            record = {}
            with mock.patch.dict(sys.modules, fake), mock.patch.object(f, "module",
                    side_effect=lambda module_name,_: q if module_name.endswith("_q") else core), \
                    mock.patch.object(f, "module_origins", return_value={}):
                with self.assertRaisesRegex(KeyboardInterrupt, "synthetic early stop"):
                    f.run(inputs, root/"output", record)
            saved = json.loads((root/"output/evidence-record.json").read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["error_type"], "KeyboardInterrupt")
            self.assertEqual(saved["failed_stage"], "execution_configuration")
            self.assertEqual((len(saved["images"]),len(saved["queries"]),len(saved["initial_masks"])), (54,48,6))
            self.assertTrue(all(row["status"] == "not_attempted" for row in saved["images"]+saved["queries"]))
            self.assertTrue(all(isinstance(row["input"]["path"],str) for row in saved["images"]+saved["initial_masks"]))
            self.assertIsInstance(cohorts[0]["images"][0]["path"], Path)
            self.assertEqual(sum(saved["calls"].values()), 0)
            self.assertIsNone(saved["coverage"])
            self.assertFalse(saved["complete"])
            self.assertEqual(list((root/"output").iterdir()), [root/"output/evidence-record.json"])
            fake["lightglue"].SuperPoint.assert_not_called()
            fake["lightglue"].LightGlue.assert_not_called()

    def test_empty_native_tensor_format_keeps_raw_and_adds_compact_shape(self):
        arrays = {"matches":np.empty((1,0,2),dtype=np.int64),
                  "scores":np.empty((1,0),dtype=np.float32),
                  "matches0":np.empty((1,0),dtype=np.int64),
                  "matches1":np.full((1,4),-1,dtype=np.int64),
                  "stop":np.asarray(1,dtype=np.int64)}
        raw = arrays["matches"]
        pairs, scores = f.canonical_matches(arrays,[0,4],np)
        self.assertEqual(pairs.shape,(0,2)); self.assertEqual(scores.shape,(0,))
        self.assertIs(arrays["matches"],raw)
        self.assertEqual(arrays["matches"].shape,(1,0,2))

    def test_dense_compact_disagreement_is_technical_failure(self):
        arrays = {"matches":np.array([[0,1]],dtype=np.int64), "scores":np.array([.5]),
                  "matches0":np.array([[0]],dtype=np.int64),
                  "matches1":np.array([[-1,0]],dtype=np.int64)}
        with self.assertRaisesRegex(ValueError,"dense/compact"):
            f.canonical_matches(arrays,[1,2],np)


if __name__ == "__main__":
    unittest.main(verbosity=2)
