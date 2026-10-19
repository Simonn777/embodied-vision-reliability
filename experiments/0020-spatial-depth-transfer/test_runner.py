"""Fake-only full-cohort, binding and technical-failure tests; no scientific inputs."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent

def load(name,filename):
    spec=importlib.util.spec_from_file_location(name,HERE/filename)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

c=load('evr0020_test_common','common.py')
with c.aliases({'common':c}):worker=load('evr0020_test_worker','predict_pair.py')
with c.aliases({'common':c,'predict_pair':worker}):runner=load('evr0020_test_runner','run_diagnostic.py')


class ScoringTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.data=self.root/'data';self.data.mkdir();self.mp=self.root/'manifest.json';self.bs=self.root/'baseline.json';self.bs.write_text('{}')
        self.library={'version':'3.0.0','binary_sha256':'a'*64};self.sources={'artificial':'b'*64};rows=[]
        for i in range(80):
            path=self.data/f'scene-{i//16}.h5'
            if not path.exists():path.write_bytes(b'ARTIFICIAL NOT HDF5')
            rows.append({'pair_index':i,'scene':f'scene-{i//16}','key':f'pair-{i}','filename':path.name,
                         'dataset_bytes':path.stat().st_size,'dataset_sha256':c.sha256(path)})
        self.manifest={'selected_pairs':rows};self.mp.write_text(json.dumps(self.manifest));self.mh=c.sha256(self.mp)
        self.binding={'seal_sha256':c.sha256(self.bs),'predictions':{f'{i}:{a}':'d'*64 for i in range(80) for a in c.ARMS}}
        self.bank={i:[self.candidate(i,a) for a in c.ARMS] for i in range(80)}
        self.stack=contextlib.ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(c,'baseline_bundle',return_value=(self.manifest,self.bank,self.binding)))
        self.stack.enter_context(mock.patch.object(c,'source_hashes',return_value=self.sources))
        self.stack.enter_context(mock.patch.object(c,'library_config',return_value={'current':self.library}))

    def candidate(self,i,arm):
        return {'pair_index':i,'scene':f'scene-{i//16}','key':f'pair-{i}','arm':arm,'status':'completed','library':self.library,
            'input_arrays':{k:{'shape':[8,2] if k.startswith('x') else [8],'dtype':'float64','sha256':'c'*64}
                            for k in ('x1','x2','depth1','depth2')},
            'calibration':{'artificial':True},'input_diagnostics':{'count':8},
            'estimate':{'R':c.np.eye(3).tolist(),'t':[1.,0.,0.]},'final_scores':{'pixel_msac':2.,'count':8},
            'timings':{'charged_total_seconds':.5}}

    def scores(self,i):
        selectors={name:{'selected_index':0,'selected_arm':c.ARMS[0],'score':.25 if name=='pixel_msac' else 1.25,
                         'scores':[.25 if name=='pixel_msac' else 1.25]*4} for name in c.SELECTORS}
        def fits():
            return {name:[{'fold':fold,'validation_weight_mass':1. if fold is None else .25,
                           'fit':{'status':'unavailable','reason':'depth_normalization_unavailable','svd_attempted':False},
                           'validation':{'status':'fit_unavailable'}}
                          for fold in ([None] if name in ('global_affine_ls','spatial_weight_only') else range(4))]
                    for name in c.SELECTORS[1:]}
        return {'schema':'evr0020.scores.v1','key':f'pair-{i}','count':8,'arms':list(c.ARMS),
                'normalization':{'status':'unavailable','reason':'artificial zero spread'},
                'spatial_partition':{'status':'completed','counts':[2,2,2,2]},'random_partition':{},
                'candidates':[{'arm':arm,'geometric_mean':.25,'geometry':{'pixel_msac':2.,'count':8},
                               'scores':{name:selectors[name]['scores'][j] for name in c.SELECTORS},
                               'fits':fits(),'depth_scores':{name:1. for name in c.SELECTORS if name!='pixel_msac'}} for j,arm in enumerate(c.ARMS)],
                'selectors':selectors,'counters':{'candidate_count':4,'fit_calls':40,'svd_calls':0,
                    'fit_completed_count':0,'fit_unavailable_count':40},'seconds':0.}

    def raw(self,i=0):
        raw=worker.identity(self.manifest['selected_pairs'][i],self.mh,self.sources,self.library,self.binding)
        raw.update({k:copy.deepcopy(self.bank[i][0][k]) for k in ('input_arrays','calibration','input_diagnostics')})
        raw.update(status='completed',phase='complete',scores=self.scores(i),candidate_generation_seconds=2.,
            timings={k:.1 for k in ('binding_seconds','input_seconds','score_seconds','output_seconds')})
        raw['timings']['worker_total_seconds']=.6;return raw

    @staticmethod
    def process():return {'exit_code':0,'termination':None,'wall_seconds':1.,'sampled_peak_rss_kib':128,'watchdog_error':None}

    def fake_child(self,command,log,remaining,env):
        get=lambda k:command[command.index(k)+1]
        i=int(get('--pair-index'));self.calls.append(i)
        self.assertTrue(all(env[k]=='1' for k in runner.THREAD_VARIABLES))
        self.assertNotIn('--runtime',command);self.assertEqual(get('--baseline-seal'),str(self.bs.resolve()))
        raw,process=self.raw(i),self.process()
        if self.mutate:self.mutate(raw,process)
        Path(log).write_text('artificial');c.write_new(Path(get('--output')),raw);return process

    def run_fake(self,mutate=None,name='predictions'):
        self.calls=[];self.mutate=mutate;out=self.root/name
        with mock.patch.object(runner,'run_child',side_effect=self.fake_child),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            code=runner.run(self.data,self.mp,self.bs,out,self.root/'deps')
        return code,out

    def test_complete_80_all_controls_and_all_failed_calibration_still_seals(self):
        with mock.patch.dict(sys.modules,{'h5py':types.SimpleNamespace(File=mock.Mock(side_effect=AssertionError('No H5'))),
                                         'poselib':types.SimpleNamespace(__getattr__=mock.Mock(side_effect=AssertionError('No native')))}):
            code,out=self.run_fake();self.assertEqual(code,0)
            _,seal,records=runner.verify_seal(self.mp,self.bs,out/'seal.json')
        self.assertEqual(self.calls,list(range(80)));self.assertEqual(len(records),80);self.assertEqual(seal['fit_attempts'],3200)
        self.assertEqual(records[0]['candidate_generation_seconds'],2.)
        self.assertEqual(records[0]['timings']['charged_total_seconds'],1.)
        self.assertEqual(set(records[0]['scores']['selectors']),set(c.SELECTORS))

    def test_any_technical_timeout_exception_or_input_error_stops_and_preserves(self):
        for name,mutation in (
            ('timeout',lambda r,p:(r.update(status='started',phase='scoring'),p.update(exit_code=-9,termination='pair_timeout'))),
            ('exception',lambda r,p:(r.update(status='error',phase='error',failure_kind='scoring'),p.update(exit_code=1))),
            ('input',lambda r,p:r.update(input_arrays={}))):
            code,out=self.run_fake(mutation,name);self.assertEqual(code,1);self.assertEqual(self.calls,[0])
            for suffix in ('.worker.json','.process.json','.log'):self.assertTrue((out/('pair-00'+suffix)).exists())
            with self.assertRaises(ValueError):runner.verify_seal(self.mp,self.bs,out/'seal.json')

    def test_complete_seal_tamper_barrier_and_every_pair_required(self):
        code,out=self.run_fake();self.assertEqual(code,0);seal=json.loads((out/'seal.json').read_text());row=seal['rows'][0]
        for name in (row['path'],row['raw_path'],row['process_path'],'pair-00.log','integrity.json'):
            path=out/name;old=path.read_bytes();path.write_bytes(old+b' ')
            with self.assertRaises(ValueError):runner.verify_seal(self.mp,self.bs,out/'seal.json')
            path.write_bytes(old)
        with mock.patch.object(c,'source_hashes',return_value={'changed':'e'*64}),self.assertRaises(ValueError):
            runner.verify_seal(self.mp,self.bs,out/'seal.json')
        for change in ('missing','unsafe','reordered'):
            value=copy.deepcopy(seal)
            if change=='missing':value['rows'].pop()
            if change=='unsafe':value['rows'][0]['raw_path']='../outside'
            if change=='reordered':value['rows'][0],value['rows'][1]=value['rows'][1],value['rows'][0]
            (out/'seal.json').write_text(json.dumps(value))
            with self.assertRaises(ValueError):runner.verify_seal(self.mp,self.bs,out/'seal.json')

    def test_bad_selector_tie_coverage_fit_cap_or_geometric_parity_blocks(self):
        for mutation in (
            lambda s:s['selectors']['spatial_transfer'].update(selected_index=1,selected_arm=c.ARMS[1]),
            lambda s:s['selectors'].pop('random_point_transfer'),
            lambda s:s['counters'].update(fit_calls=41),
            lambda s:s['candidates'][0].update(geometric_mean=.3),
            lambda s:s['selectors']['global_affine_ls']['scores'].__setitem__(0,float('nan'))):
            scores=self.scores(0);mutation(scores)
            with self.assertRaises(ValueError):c.validate_scores(scores,self.bank[0],'pair-0')

    def test_duplicate_hybrid_fields_cannot_override_saved_g_plus_d(self):
        scores=self.scores(0)
        # A mutually consistent row/selector rewrite formerly passed; depth remains one.
        scores['candidates'][0]['scores']['spatial_transfer']=.5
        scores['selectors']['spatial_transfer'].update(scores=[.5,1.25,1.25,1.25],score=.5)
        with self.assertRaisesRegex(ValueError,'geometry plus depth'):
            c.validate_scores(scores,self.bank[0],'pair-0')
        for mutation in (lambda s:s['candidates'][0]['depth_scores'].update(global_affine_ls=-.1),
                         lambda s:s['candidates'][0]['depth_scores'].update(global_affine_ls=1.1),
                         lambda s:s['candidates'][0]['geometry'].update(pixel_msac=2.5),
                         lambda s:s['candidates'][0]['geometry'].update(count=7)):
            scores=self.scores(0);mutation(scores)
            with self.assertRaises(ValueError):c.validate_scores(scores,self.bank[0],'pair-0')

    def test_actual_fit_records_determine_work_counters_and_complete_coverage(self):
        mutations=(lambda s:s['candidates'][0]['fits']['spatial_transfer'].pop(),
                   lambda s:s['candidates'][0]['fits']['global_affine_ls'][0].pop('fold'),
                   lambda s:s['candidates'][0]['fits'].pop('global_affine_ls'),
                   lambda s:s['candidates'][0]['fits']['random_point_transfer'][0].update(fold=1),
                   lambda s:s['candidates'][0]['fits']['global_affine_ls'][0]['fit'].update(svd_attempted=True),
                   lambda s:s['counters'].update(svd_calls=1),
                   lambda s:s['counters'].update(fit_completed_count=1,fit_unavailable_count=39))
        for mutation in mutations:
            scores=self.scores(0);mutation(scores)
            with self.assertRaises(ValueError):c.validate_scores(scores,self.bank[0],'pair-0')
        scores=self.scores(0)
        fit=scores['candidates'][0]['fits']['global_affine_ls'][0]['fit']
        fit.update(status='completed',reason=None,svd_attempted=True)
        scores['counters'].update(svd_calls=1,fit_completed_count=1,fit_unavailable_count=39)
        self.assertEqual(c.fit_count(scores),40)

    def test_unavailable_partition_requires_exact_placeholders_and_four_actual_fits(self):
        scores=self.scores(0);scores['spatial_partition']={'status':'unavailable','counts':[8,0,0,0]}
        for row in scores['candidates']:
            for name in c.SELECTORS[2:]:
                row['fits'][name]=[{'status':'unavailable','reason':'spatial_partition_unavailable'}]
        scores['counters'].update(fit_calls=4,fit_unavailable_count=4)
        self.assertTrue(c.validate_scores(scores,self.bank[0],'pair-0'))
        self.assertEqual(c.fit_count(scores),4)
        for mutation in (lambda s:s['candidates'][0]['fits']['spatial_transfer'].clear(),
                         lambda s:s['candidates'][0]['fits']['spatial_transfer'][0].update(reason='different'),
                         lambda s:s['counters'].update(fit_calls=40,fit_unavailable_count=40)):
            changed=copy.deepcopy(scores);mutation(changed)
            with self.assertRaises(ValueError):c.validate_scores(changed,self.bank[0],'pair-0')

    def test_existing_output_and_global_budget_refuse_before_scoring(self):
        out=self.root/'exists';out.mkdir()
        with mock.patch.object(c,'baseline_bundle',side_effect=AssertionError('No baseline read')),self.assertRaises(FileExistsError):
            runner.run(self.data,self.mp,self.bs,out,self.root)
        with mock.patch.object(runner,'TOTAL_TIMEOUT',0):
            code,out=self.run_fake(name='budget');self.assertEqual(code,1);self.assertEqual(self.calls,[])
        with self.assertRaises(ValueError):c.bound_file(self.root,'../outside')

    def test_scoring_api_fixed_full_candidate_bank_no_candidate_filter(self):
        pair={k:object() for k in ('x1','x2','depth1','depth2','calibration')}
        api=mock.Mock(return_value={'fake':True})
        with mock.patch.dict(sys.modules,{'score':types.SimpleNamespace(score_pair=api)}):
            self.assertEqual(worker.invoke(pair,self.bank[0],'pair-0'),{'fake':True})
        args=api.call_args.args
        self.assertEqual(args[0],'pair-0');self.assertEqual(args[1:6],tuple(pair[k] for k in ('x1','x2','depth1','depth2','calibration')))
        self.assertIs(args[6],self.bank[0]);self.assertEqual([r['arm'] for r in args[6]],list(c.ARMS))

    def test_worker_started_marker_and_scoring_failure_preserved_without_native(self):
        output=self.root/'worker.json';pair={k:object() for k in ('x1','x2','depth1','depth2')}
        pair.update(calibration=self.bank[0][0]['calibration'],input_diagnostics=self.bank[0][0]['input_diagnostics'])
        sentinel=object()
        def read(handle,key):self.assertIs(handle,sentinel);self.assertEqual(key,'pair-0');return pair
        def call(actual,bank,key):
            saved=json.loads(output.read_text());self.assertEqual(saved['phase'],'scoring');self.assertEqual(len(bank),4)
            raise RuntimeError('artificial technical scoring failure')
        fake_h5=types.SimpleNamespace(File=mock.Mock(return_value=contextlib.nullcontext(sentinel)))
        with mock.patch.dict(sys.modules,{'h5py':fake_h5}),mock.patch.object(c,'read_pair',side_effect=read),\
             mock.patch.object(c,'input_hashes',return_value=self.bank[0][0]['input_arrays']),mock.patch.object(worker,'invoke',side_effect=call):
            code=worker.run(self.data/'scene-0.h5',self.mp,self.bs,output,0)
        self.assertEqual(code,1);record=json.loads(output.read_text());self.assertEqual(record['failure_kind'],'scoring')
        self.assertEqual(record['failed_phase'],'scoring');self.assertGreaterEqual(record['timings']['score_seconds'],0.)
        with self.assertRaises(FileExistsError):worker.run(self.root,self.root,self.root,output,0)

    def test_manufactured_core_integrates_with_worker_and_saved_schema(self):
        core=load('evr0020_integration_score','score.py')
        X=c.np.array([[-1.,-1.,4.],[-1.,1.,5.],[1.,-1.,6.],[1.,1.,7.],
                      [-2.,-.5,8.],[-.5,2.,5.],[2.,-.5,7.],[.5,2.,9.]])
        t=c.np.array([.4,.02,.01]);Y=X+t
        params=[500.,510.,320.,240.];K=[[500.,0.,320.],[0.,510.,240.],[0.,0.,1.]]
        cal={f'camera{j}':{'model':'PINHOLE','width':640,'height':480,'params':params} for j in (1,2)}
        cal.update(K1=K,K2=K)
        pair={'x1':X[:,:2]/X[:,2,None]*params[:2]+params[2:],
              'x2':Y[:,:2]/Y[:,2,None]*params[:2]+params[2:],
              'depth1':X[:,2],'depth2':Y[:,2],'calibration':cal,'input_diagnostics':{'count':8}}
        pose={'R':c.np.eye(3).tolist(),'t':t.tolist()}
        geometry=c.base.prepare_geometry(pair['x1'],pair['x2'],cal)
        for candidate in self.bank[0]:
            candidate.update(estimate=pose,final_scores=c.base.final_scores(pose,geometry),
                             calibration=cal,input_diagnostics=pair['input_diagnostics'],input_arrays=c.input_hashes(pair))
        output=self.root/'manufactured.json'
        fake_h5=types.SimpleNamespace(File=mock.Mock(return_value=contextlib.nullcontext(object())))
        with mock.patch.dict(sys.modules,{'h5py':fake_h5,'score':core}),mock.patch.object(c,'read_pair',return_value=pair):
            code=worker.run(self.data/'scene-0.h5',self.mp,self.bs,output,0)
        record=json.loads(output.read_text());self.assertEqual(code,0,record.get('error'))
        self.assertEqual(record['status'],'completed');self.assertEqual(c.fit_count(record['scores']),40)
        self.assertTrue(c.validate_scores(record['scores'],self.bank[0],'pair-0'))

    def test_watchdog_launch_failure_timeout_cleanup_and_kill_race(self):
        with mock.patch.object(runner.subprocess,'Popen',side_effect=OSError('artificial')):
            receipt=runner.run_child([],self.root/'launch.log',600,{})
        self.assertEqual(receipt['termination'],'launch_error');self.assertIsNone(receipt['exit_code'])
        process=mock.Mock();process.poll.side_effect=[None,None];process.kill.side_effect=ProcessLookupError();process.wait.return_value=-9
        with mock.patch.object(runner.subprocess,'Popen',return_value=process),mock.patch.object(runner.time,'perf_counter',side_effect=[0.,31.,32.]):
            receipt=runner.run_child([],self.root/'timeout.log',600,{})
        self.assertEqual(receipt['termination'],'pair_timeout');process.kill.assert_called_once();process.wait.assert_called_once_with(timeout=5)

    def test_resource_limits_remain_mandatory_at_verification(self):
        expected=worker.identity(self.manifest['selected_pairs'][0],self.mh,self.sources,self.library,self.binding)
        for change in ({'wall_seconds':30.001},{'sampled_peak_rss_kib':runner.RSS_LIMIT_KIB+1},{'watchdog_error':'artificial'}):
            process={**self.process(),**change}
            with self.assertRaises(ValueError):runner.wrap_record(self.raw(),expected,process,self.bank[0])
        code,out=self.run_fake();self.assertEqual(code,0)
        seal=json.loads((out/'seal.json').read_text());seal['wall_seconds']=600.001
        (out/'seal.json').write_text(json.dumps(seal))
        with self.assertRaises(ValueError):runner.verify_seal(self.mp,self.bs,out/'seal.json')

    def test_bank_hash_cost_and_timing_accounting_cannot_be_rewritten(self):
        expected=worker.identity(self.manifest['selected_pairs'][0],self.mh,self.sources,self.library,self.binding)
        for mutation in (lambda r:r['candidate_prediction_sha256'].update(e_norm='bad'),
                         lambda r:r.update(candidate_generation_seconds=.5),
                         lambda r:r['timings'].update(score_seconds=10.)):
            raw=self.raw();mutation(raw)
            with self.assertRaises(ValueError):runner.wrap_record(raw,expected,self.process(),self.bank[0])
        changed=copy.deepcopy(self.bank[0]);changed[3]['input_arrays']={}
        with self.assertRaises(ValueError):runner.wrap_record(self.raw(),expected,self.process(),changed)

if __name__=='__main__':unittest.main()
