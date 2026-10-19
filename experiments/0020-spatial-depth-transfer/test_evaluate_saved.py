import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('evr0020_eval_test',Path(__file__).with_name('evaluate_saved.py'))
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)


def fixture(author_error=1.):
    records=[];identities=[];refs=[];authors=[]
    def outcome(value):return {'status':'completed','error':{'max_degrees':value},'penalized_max_degrees':value,'penalized_max_signfolded_degrees':value}
    for i in range(80):
        identity={'pair_index':i,'scene':e.old.SCENES[i//16],'key':str(i)}
        identities.append(identity)
        outcomes={a:outcome(0. if j==i%2 else 10.) for j,a in enumerate(e.ARMS)}
        refs.append({**identity,'outcomes':outcomes})
        authors.append({**identity,'outcomes':{**copy.deepcopy(outcomes),**{a:outcome(author_error) for a in e.AUTHOR_ARMS}}})
        selectors={}
        for name in e.SELECTORS:
            selected=i%2 if name=='spatial_transfer' else 2
            scores=[1.]*4;scores[selected]=0.
            selectors[name]={'scores':scores,'selected_index':selected,'selected_arm':e.ARMS[selected]}
        records.append({**identity,'status':'completed','scores':{'selectors':selectors},'timings':{'charged_total_seconds':.1},'candidate_generation_seconds':4.})
    return records,identities,{'rows':refs},{'rows':authors}


class EvaluationTests(unittest.TestCase):
    def test_ten_comparators_and_fixed_scene_comparator(self):
        result=e.compute(*fixture())
        self.assertEqual(len(result['comparisons']),10)
        self.assertEqual(result['development_gate']['best_auc5_comparator'],e.AUTHOR_ARMS[0])
        self.assertTrue(result['development_gate']['passes'])
        self.assertEqual(len(result['development_gate']['positive_scenes_against_same_comparator']),5)
        self.assertEqual(result['historical_all_four_candidates_seconds'],320.)

    def test_stronger_author_control_prevents_false_admission(self):
        gate=e.compute(*fixture(.01))['development_gate']
        self.assertFalse(gate['passes'])
        self.assertFalse(gate['conditions']['auc5_at_least_2pp'])

    def test_choice_must_match_prespecified_tie_order(self):
        with self.assertRaises(ValueError):
            e.choice({'scores':[0.,0.,1.,1.],'selected_index':1,'selected_arm':'e_ts'})

    def test_incomplete_cohort_rejected(self):
        records,ids,ref,author=fixture()
        with self.assertRaises(ValueError):e.compute(records[:-1],ids,ref,author)

    def test_failed_seal_cannot_open_labels(self):
        runner=types.SimpleNamespace(verify_seal=lambda *a:(_ for _ in ()).throw(ValueError('bad seal')))
        with tempfile.TemporaryDirectory() as tmp,patch.dict(sys.modules,{'common':types.SimpleNamespace(),'run_diagnostic':runner}),patch.object(Path,'read_text',side_effect=AssertionError('label read')):
            with self.assertRaisesRegex(ValueError,'bad seal'):
                e.evaluate(Path('manifest'),Path('baseline'),Path('seal'),Path(tmp)/'out.json')

    def test_outer_failure_or_resource_overrun_blocks_interpretation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'prediction.log').write_bytes(b'log')
            sha=lambda p:'helper' if Path(p).name=='run_operation.py' else hashlib.sha256(Path(p).read_bytes()).hexdigest()
            c=types.SimpleNamespace(sha256=sha)
            record={'schema':'evr0020.outer-operation.v1','mode':'prediction','status':'completed','exit_code':0,'termination':None,
                    'remaining_process_groups_after_cleanup':[],'wall_seconds':1.,'peak_group_rss_kib':1,
                    'hard_timeout_seconds':600,'group_rss_limit_kib':8*1024*1024,'helper_sha256':'helper',
                    'log_sha256':sha(folder/'prediction.log'),'source_commit':'a'*40}
            receipt=folder/'prediction-process.json';receipt.write_text(json.dumps(record))
            self.assertEqual(e.verify_outer(folder/'predictions/seal.json',c,'a'*40)['source_commit'],'a'*40)
            for key,value in (('status','failed'),('wall_seconds',600.001),('peak_group_rss_kib',8*1024*1024+1),('source_commit','b'*40)):
                changed={**record,key:value};receipt.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):e.verify_outer(folder/'predictions/seal.json',c,'a'*40)


if __name__=='__main__':unittest.main()
