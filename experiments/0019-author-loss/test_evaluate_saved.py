import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('evr0019_eval_test',HERE/'evaluate_saved.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)


def fixture():
    identities=[];records=[];rows=[]
    for i in range(80):
        identity={'pair_index':i,'scene':e.old.SCENES[i//16],'key':str(i)}
        identities.append(identity)
        outcomes={a:{'status':'completed','error':{'max_degrees':3.},'penalized_max_degrees':3.,'penalized_max_signfolded_degrees':3.} for a in e.old.ARMS}
        rows.append({**identity,'outcomes':outcomes,'ground_truth':{'R_original':[[1,0,0],[0,1,0],[0,0,1]],'t_original':[1,0,0]}})
        for arm in e.ARMS:
            records.append({**identity,'arm':arm,'status':'completed','estimate':{'R':[[1,0,0],[0,1,0],[0,0,1]],'t':[1,0,0]},'timings':{'charged_total_seconds':.1}})
    return records,identities,{'rows':rows}


class EvaluationTests(unittest.TestCase):
    def test_failure_retained_in_full_denominator(self):
        records,ids,ref=fixture();records[0].update(status='timeout',estimate=None)
        result=e.compute(records,ids,ref)
        s=result['summaries']['depth_scale_truncated_cauchy']
        self.assertEqual(s['pair_count'],80)
        self.assertEqual(s['status_counts'],{'timeout':1,'completed':79})
        self.assertEqual(s['count_strictly_below']['5'],79)
        self.assertEqual(result['comparisons']['depth_scale_truncated_cauchy_vs_e_norm']['damaged_5deg'],1)
        self.assertEqual(sum(result['scene_summaries'][s]['depth_scale_truncated_cauchy']['pair_count'] for s in e.old.SCENES),80)

    def test_duplicate_rejected(self):
        records,ids,ref=fixture();records[-1]=records[0]
        with self.assertRaisesRegex(ValueError,'duplicate'):
            e.compute(records,ids,ref)

    def test_integrity_failure_never_reads_labels(self):
        fake_runner=types.SimpleNamespace(verify_seal=lambda *args:(_ for _ in ()).throw(ValueError('bad seal')))
        with tempfile.TemporaryDirectory() as tmp,patch.dict(sys.modules,{'common':types.SimpleNamespace(),'run_diagnostic':fake_runner}),patch.object(Path,'read_text',side_effect=AssertionError('label read')):
            with self.assertRaisesRegex(ValueError,'bad seal'):
                e.evaluate(Path('manifest'),Path('baseline'),Path('seal'),Path(tmp)/'output.json')

    def test_material_boundaries(self):
        self.assertTrue(e.harm(4.9,5.))
        self.assertTrue(e.harm(1.,2.))
        self.assertFalse(e.harm(1.,1.999))
        self.assertFalse(e.harm(5.,4.9))


if __name__=='__main__':unittest.main()
