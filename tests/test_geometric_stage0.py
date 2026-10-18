"""Only explicit artificial 2–5-point fixtures; no protocol seed or study case runs."""
import copy
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
import py_compile
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from evr import geometric_stage0 as core
from evr.geometric_stage0_evaluation import decision_gate, evaluate_cases
from evr.exact_assignment_tail import UnsupportedBoard
spec = importlib.util.spec_from_file_location('stage0_cli_fixture', ROOT/'scripts/run_geometric_stage0.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def fixture(order=(0,1)):
    return {'raw_x':np.array([[0.,0.],[10.,0.]]),
            'raw_y':np.array([[0.,0.],[10.,0.]]),
            'priorities':np.array([.2,.1]),
            'candidates':np.array([np.eye(3),[[1,0,100],[0,1,0],[0,0,1]]],dtype=np.float64),
            'candidate_order':np.array(order,dtype=np.int64)}


def empty_result():
    return {'timings':dict.fromkeys(core.TIMINGS,0.),'hypotheses':[],
            'controls_complete':False,'tail_attempted':0,
            'selections':{'tail':{'selected_id':None,'tied_ids':[]}}}


class FakeRNG:
    """No PRNG: deterministic tiny generator-shape/label fixture."""
    def uniform(self,low,high,size):
        return np.full(size,(low+high)/2,dtype=np.float64)
    def random(self,size):
        return np.arange(size,dtype=np.float64)/(size+1)
    def normal(self,loc,scale,size):
        return np.zeros(size,dtype=np.float64)
    def choice(self,n,size,replace):
        assert replace is False
        return np.arange(size)
    def permutation(self,n):
        return np.arange(n-1,-1,-1)


class ArithmeticFixtures(unittest.TestCase):
    def test_dedup_source_target_both_and_priority_order(self):
        x=np.array([[0,0],[.5,0],[10,0],[0,0],[20,0]],float)
        y=np.array([[0,0],[10,0],[.5,0],[0,0],[20,0]],float)
        kept,why=core.deduplicate(x,y,np.arange(5))
        self.assertEqual(kept,[0,4])
        self.assertEqual([v['reason'] for v in why],[None,'source','target','both',None])
        kept2,_=core.deduplicate(x[:2],y[:2],[0,0])
        self.assertEqual(kept2,[0])

    def test_dedup_inclusive_one_pixel_and_no_replenishment(self):
        kept,_=core.deduplicate([[0,0],[1,0]],[[0,0],[10,0]],[.1,.2])
        self.assertEqual(kept,[0])

    def test_tiny_stub_generator_separates_labels_and_clones(self):
        inp,labels=core.construct_case('uniform',0,{s:FakeRNG() for s in core.STREAMS},
                                      base_count=3,true_count=1,clone_count=1)
        self.assertEqual(inp['raw_x'].shape,(5,2))
        self.assertEqual(labels['base_ids'],[0,1,2,0,0])
        self.assertEqual(labels['true_flags'],[True,False,False,True,True])
        self.assertEqual(labels['clone_flags'],[False,False,False,True,True])
        self.assertNotIn('H_star',inp)
        self.assertNotIn('true_flags',inp)
        self.assertEqual(inp['candidate_order'].tolist(),list(reversed(range(32))))

    def test_generator_noise_attempt_limit_is_per_pair(self):
        class BadNoise:
            calls=0
            def normal(self,*args,**kwargs):
                self.calls+=1
                return np.array([1e9,1e9])
        bad=BadNoise()
        with self.assertRaisesRegex(ValueError,'64'):
            core._noisy_target(np.array([[0.,0.],[1.,1.]]),np.zeros(2),bad)
        self.assertEqual(bad.calls,64)

    def test_artificial_complete_tail_and_control_arithmetic(self):
        r,labels=core.execute_case(lambda:(fixture(),{'fixture':'not used by scorer'}),seconds=1)
        self.assertEqual(r['status'],'completed')
        self.assertTrue(r['controls_complete'])
        self.assertEqual(r['tail_attempted'],2)
        self.assertTrue(all(v['selected_id']==0 for v in r['selections'].values()))
        identity=r['hypotheses'][0]
        self.assertEqual((identity['N'],identity['S'],identity['n']),(2,2,2))
        self.assertEqual(identity['tail']['numerator'],'1')
        self.assertEqual(identity['tail']['denominator'],'2')
        self.assertEqual(identity['excess_key'],2)
        self.assertEqual(identity['ratio_key'],[4,2])
        self.assertEqual(labels,{'fixture':'not used by scorer'})

    def test_all_controls_complete_before_tail_and_no_subset_selection(self):
        r=empty_result()
        calls=[]
        def refuse(board,threshold,**kwargs):
            self.assertTrue(r['controls_complete'])
            self.assertEqual(len(r['hypotheses']),2)
            calls.append(threshold)
            raise UnsupportedBoard('artificial refusal',17)
        core.score_inputs(fixture(),r,tail_function=refuse)
        self.assertEqual(r['status'],'tail_failed')
        self.assertEqual(len(calls),1)
        self.assertEqual(r['selections']['tail']['selected_id'],None)
        self.assertEqual(r['selections']['raw']['selected_id'],0)
        self.assertEqual(r['hypotheses'][1]['tail']['status'],'not_attempted')

    def test_exact_fraction_ranking_does_not_round_to_float(self):
        count=[0]
        def near_one(board,threshold,**kwargs):
            count[0]+=1
            return SimpleNamespace(numerator=2**60-count[0],denominator=2**60,
                                   operations=1,components=())
        r,_=core.execute_case(lambda:(fixture(),None),seconds=1,tail_function=near_one)
        self.assertEqual(r['selections']['tail']['selected_id'],1)
        self.assertEqual(float(Fraction(2**60-1,2**60)),float(Fraction(2**60-2,2**60)))

    def test_ties_use_provided_candidate_order_without_secondary_support(self):
        inp=fixture((1,0)); inp['candidates'][1]=inp['candidates'][0]
        r,_=core.execute_case(lambda:(inp,None),seconds=1)
        for v in r['selections'].values():
            self.assertEqual(v,{'selected_id':1,'tied_ids':[1,0]})

    def test_zero_observed_tail_is_study_invalid_not_unsupported(self):
        def invalid(board,threshold,**kwargs):
            return SimpleNamespace(numerator=0,denominator=2,operations=1,components=())
        r,_=core.execute_case(lambda:(fixture(),None),seconds=1,tail_function=invalid)
        self.assertEqual(r['status'],'invalid')
        self.assertIsNone(r['selections']['tail']['selected_id'])

    def test_tail_timeout_keeps_completed_controls(self):
        old=signal.getsignal(signal.SIGALRM)
        def slow(*args,**kwargs):
            time.sleep(.1)
        r,_=core.execute_case(lambda:(fixture(),None),seconds=.02,tail_function=slow)
        self.assertEqual(r['status'],'tail_failed')
        self.assertTrue(r['controls_complete'])
        self.assertEqual(r['hypotheses'][0]['tail']['status'],'timeout')
        self.assertIsNone(r['selections']['tail']['selected_id'])
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL),(0.,0.))
        self.assertEqual(signal.getsignal(signal.SIGALRM),old)

    def test_generation_timeout_is_incomplete(self):
        def slow():
            time.sleep(.1)
            return fixture(),None
        r,_=core.execute_case(slow,seconds=.02)
        self.assertEqual(r['status'],'incomplete')
        self.assertFalse(r['controls_complete'])

    def test_refuse_preexisting_timer_and_restore_it(self):
        signal.setitimer(signal.ITIMER_REAL,10)
        try:
            with self.assertRaisesRegex(RuntimeError,'existing alarm'):
                with core.deadline(.1):
                    pass
            self.assertGreater(signal.getitimer(signal.ITIMER_REAL)[0],9)
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)

    def test_array_hash_binds_shape_dtype_values_and_is_endian_portable(self):
        a=np.array([[1,2],[3,4]],dtype='<f8')
        self.assertEqual(core.array_sha256(a),core.array_sha256(a.astype('>f8')))
        self.assertNotEqual(core.array_sha256(a),core.array_sha256(a.ravel()))
        self.assertNotEqual(core.array_sha256(a),core.array_sha256(a+1))

    def test_pool_order_and_allocation_metadata_only(self):
        h=core.candidate_pool()
        self.assertEqual(h.shape,(32,3,3))
        np.testing.assert_array_equal(h[4],np.eye(3))
        np.testing.assert_array_equal(h[13,:2,2],[128,128])
        order=core.case_order()
        self.assertEqual(len(set(order)),256)
        self.assertEqual(order[:4],[(f,0) for f in core.FAMILIES])
        self.assertEqual(order[4:8],[(f,1) for f in core.FAMILIES[1:]+core.FAMILIES[:1]])

    def test_gate_uses_paired_harms_not_net_successes(self):
        counts={f:{m:40 for m in core.METHODS} for f in core.FAMILIES}
        counts['clustered_outlier']['tail']=48
        paired={f:{m:{'baseline_success_tail_failure':0} for m in core.CONTROLS}
                for f in core.FAMILIES}
        self.assertEqual(decision_gate(counts,paired,complete=True)['decision'],
                         'ELIGIBLE_PENDING_EXTERNAL_REVIEW')
        paired['uniform']['raw']['baseline_success_tail_failure']=3
        # Net successes are still equal; three losses must fail even if offset by wins.
        self.assertFalse(decision_gate(counts,paired,complete=True)['paired_harm_pass'])
        self.assertEqual(decision_gate(counts,paired,complete=True)['decision'],'NO_GO')

    def test_no_final_go_from_incomplete_or_missing_external_review(self):
        counts={f:{m:0 for m in core.METHODS} for f in core.FAMILIES}
        counts['clustered_outlier']['tail']=8
        paired={f:{m:{'baseline_success_tail_failure':0} for m in core.CONTROLS}
                for f in core.FAMILIES}
        self.assertEqual(decision_gate(counts,paired,complete=False)['decision'],'NO_GO')
        self.assertNotEqual(decision_gate(counts,paired,complete=True)['decision'],'GO')


class EvaluatorAndResources(unittest.TestCase):
    def artificial_record(self):
        pool=core.candidate_pool()
        return {'family':'uniform','seed_index':0,'status':'tail_failed',
                'controls_complete':True,'hypotheses':[{'canonical_id':i,'H_sha256':core.array_sha256(h),
                    'tail':{'status':'not_attempted'}} for i,h in enumerate(pool)],'tail_attempted':0,
                'inputs':{'candidates':pool.tolist(),'candidate_order':list(range(32))},
                'selections':{m:{'selected_id':4,'tied_ids':[4]} for m in core.CONTROLS}}

    def test_failed_tail_preserves_denominator_and_completed_controls(self):
        rec=self.artificial_record()
        out=evaluate_cases({'uniform-00':rec},{'uniform-00':{'H_star':np.eye(3).tolist(),'true_candidate_id':4}},
                           expected_ids=['uniform-00'])
        self.assertEqual(out['family_totals']['uniform'],1)
        self.assertEqual(out['success_counts']['uniform']['raw'],1)
        self.assertEqual(out['success_counts']['uniform']['tail'],0)
        self.assertIsNone(out['cases'][0]['methods']['tail']['transfer_error'])
        self.assertEqual(out['gates']['decision'],'NO_GO')

    def test_evaluator_rejects_wrong_identity_pool_and_unfinished_control(self):
        base=self.artificial_record()
        for field in ('seed','family','pool','control','hash','missing_selection','missing_hypothesis','ties'):
            rec=copy.deepcopy(base)
            if field=='seed': rec['seed_index']=1
            if field=='family': rec['family']='compact_true'
            if field=='pool': rec['inputs']['candidates'][4][0][2]=1
            if field=='control': rec['controls_complete']=False
            if field=='hash': rec['hypotheses'][0]['H_sha256']='0'*64
            if field=='missing_selection': del rec['selections']['cell']
            if field=='missing_hypothesis': rec['hypotheses'].pop()
            if field=='ties': rec['selections']['cell']['tied_ids']=[4,0]
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluate_cases({'uniform-00':rec},{'uniform-00':{'H_star':np.eye(3).tolist(),'true_candidate_id':4}},
                               expected_ids=['uniform-00'])

    def test_evaluator_rejects_wrong_fixed_ground_truth(self):
        truth={'H_star':np.eye(3).tolist(),'true_candidate_id':4}
        truth['H_star'][0][2]=1
        with self.assertRaisesRegex(ValueError,'fixed identity'):
            evaluate_cases({'uniform-00':self.artificial_record()},{'uniform-00':truth},
                           expected_ids=['uniform-00'])

    def test_explicit_loader_uses_hashed_source_not_matching_timestamp_cache(self):
        saved={k:v for k,v in sys.modules.items() if k=='evr' or k.startswith('evr.')}
        old_path=list(sys.path); old_dont_write=sys.dont_write_bytecode
        old_env=os.environ.get('PYTHONDONTWRITEBYTECODE')
        for key in saved: del sys.modules[key]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve(); package=root/'src/evr';package.mkdir(parents=True)
                names=('__init__','geometric_verification','geometry_evaluation',
                       'exact_assignment_tail','geometric_stage0','geometric_stage0_evaluation')
                files={}
                for name in names:
                    p=package/(name+'.py')
                    p.write_text('VALUE = 999\n')
                    st=p.stat();py_compile.compile(str(p),doraise=True)
                    p.write_text('VALUE = 111\n')
                    os.utime(p,ns=(st.st_atime_ns,st.st_mtime_ns))
                    files[p.relative_to(root).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
                with mock.patch.object(cli,'ROOT',root):
                    loaded=cli.load_verified_modules({'files':files})
                    self.assertEqual(loaded['geometric_stage0'].VALUE,111)
                    self.assertEqual(loaded['geometric_stage0_evaluation'].VALUE,111)
        finally:
            for key in list(sys.modules):
                if key=='evr' or key.startswith('evr.'): del sys.modules[key]
            sys.modules.update(saved)
            sys.path[:]=old_path; sys.dont_write_bytecode=old_dont_write
            if old_env is None: os.environ.pop('PYTHONDONTWRITEBYTECODE',None)
            else: os.environ['PYTHONDONTWRITEBYTECODE']=old_env

    def test_resource_gate_mocked_linux_good_and_rejects_unbounded(self):
        names=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')
        with mock.patch.object(cli.sys,'platform','linux'), \
             mock.patch.object(cli.os,'sched_getaffinity',return_value={0,1,2,3},create=True), \
             mock.patch.object(cli.resource,'getrlimit',return_value=(8*1024**3,8*1024**3)), \
             mock.patch.dict(cli.os.environ,dict.fromkeys(names,'4')):
            self.assertEqual(cli.require_collect_resources()['affinity'],[0,1,2,3])
            with mock.patch.object(cli.resource,'getrlimit',return_value=(-1,-1)):
                with self.assertRaises(RuntimeError): cli.require_collect_resources()
            with mock.patch.object(cli.os,'sched_getaffinity',return_value=set(range(5))):
                with self.assertRaises(RuntimeError): cli.require_collect_resources()
            with mock.patch.dict(cli.os.environ,{'OMP_NUM_THREADS':'0'}):
                with self.assertRaises(RuntimeError): cli.require_collect_resources()


class FreezeFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.run_git('init','-q')
        self.run_git('config','user.name','Artificial fixture')
        self.run_git('config','user.email','fixture@example.invalid')
        files={}
        for rel in cli.FROZEN_PATHS:
            path=self.root/rel; path.parent.mkdir(parents=True,exist_ok=True)
            data=('artificial frozen placeholder: '+rel+'\n').encode()
            path.write_bytes(data); files[rel]=hashlib.sha256(data).hexdigest()
        self.run_git('add','.'); self.run_git('commit','-qm','artificial source')
        source=self.run_git('rev-parse','HEAD').strip()
        self.receipt={'schema':'evr0009.freeze.v1','source_commit':source,'files':files}
        path=self.root/cli.RECEIPT
        path.write_text(json.dumps(self.receipt))
        self.run_git('add','.'); self.run_git('commit','-qm','artificial receipt only')
        self.freeze=self.run_git('rev-parse','HEAD').strip()

    def tearDown(self):
        self.tmp.cleanup()

    def run_git(self,*args):
        return subprocess.check_output(['git','-C',str(self.root),*args],stderr=subprocess.PIPE).decode()

    def test_legal_chain_and_later_non_scientific_publication(self):
        binding=cli.verify_freeze(self.freeze,self.root)
        self.assertEqual(len(binding['files']),len(cli.FROZEN_PATHS))
        (self.root/'later-review.md').write_text('artificial publication only')
        self.run_git('add','.'); self.run_git('commit','-qm','artificial later publication')
        self.assertEqual(cli.verify_freeze(self.freeze,self.root)['source_commit'],self.receipt['source_commit'])

    def test_current_source_mutation_rejected(self):
        (self.root/cli.FROZEN_PATHS[0]).write_text('changed')
        with self.assertRaisesRegex(ValueError,'frozen input differs'):
            cli.verify_freeze(self.freeze,self.root)

    def test_index_source_mutation_rejected(self):
        p=self.root/cli.FROZEN_PATHS[0]; original=p.read_bytes(); p.write_text('changed')
        self.run_git('add',cli.FROZEN_PATHS[0]); p.write_bytes(original)
        with self.assertRaisesRegex(ValueError,'frozen input differs'):
            cli.verify_freeze(self.freeze,self.root)

    def test_current_shrunken_manifest_rejected(self):
        self.receipt['files'].pop(cli.FROZEN_PATHS[0])
        (self.root/cli.RECEIPT).write_text(json.dumps(self.receipt))
        with self.assertRaisesRegex(ValueError,'receipt differs'):
            cli.verify_freeze(self.freeze,self.root)

    def test_symlinked_source_rejected_even_with_matching_bytes(self):
        p=self.root/cli.FROZEN_PATHS[0]; copy_path=self.root/'duplicate.py'
        copy_path.write_bytes(p.read_bytes()); p.unlink(); p.symlink_to(copy_path)
        with self.assertRaisesRegex(ValueError,'regular file'):
            cli.verify_freeze(self.freeze,self.root)

    def test_committed_shrunken_manifest_is_rejected(self):
        self.receipt['files'].pop(cli.FROZEN_PATHS[0])
        (self.root/cli.RECEIPT).write_text(json.dumps(self.receipt))
        self.run_git('add',cli.RECEIPT)
        self.run_git('commit','--amend','-qm','artificial malformed canonical receipt')
        updated=self.run_git('rev-parse','HEAD').strip()
        with self.assertRaisesRegex(ValueError,'exact mandatory'):
            cli.verify_freeze(updated,self.root)

    def test_existing_output_refused_without_mutation(self):
        out=self.root/'output'; out.mkdir(); (out/'preserved').write_text('original')
        with self.assertRaises(FileExistsError):
            cli.new_output(out)
        self.assertEqual((out/'preserved').read_text(),'original')


if __name__=='__main__':
    unittest.main(verbosity=2)
