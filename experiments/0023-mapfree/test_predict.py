"""Artificial receipt/source bytes and fake solvers only; no scientific input."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location("_evr0023_prediction_test", Path(__file__).with_name("predict.py"))
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve(); self.root = self.base / 'repo'; self.root.mkdir()
        self.here = self.root / 'experiments/0023-mapfree'; self.here.mkdir(parents=True)
        self.private = self.base / 'private'; self.private.mkdir()
        for key, value in [('ROOT', self.root), ('HERE', self.here)]:
            patch = mock.patch.object(p, key, value); patch.start(); self.addCleanup(patch.stop)
        self.args = types.SimpleNamespace(protocol=self.root/'protocols/study.md',
            plan=self.private/'plan.json', frontend_cache=self.private/'frontend',
            frontend_supervision=self.private/'frontend-supervision/receipt.json',
            native_runtime=self.private/'native', output_dir=self.private/'prediction', expected_commit='a'*40)
        dump(self.args.plan, {'scenes': ['one','two','three']}); self.args.plan_sha256 = sha(self.args.plan)
        for path in p.source_paths(self.args.protocol):
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# artificial source '+path.name)
        self.args.protocol.write_text('Admitted artificial plan '+self.args.plan_sha256)
        package = self.args.native_runtime/'poselib'; package.mkdir(parents=True)
        (package/'__init__.py').write_text('# fake wrapper, never executed')
        (package/'fake.so').write_bytes(b'FAKE BINARY NEVER LOADED')
        native = dict(version='3.0.0',python_files={'__init__.py':sha(package/'__init__.py')},
                      binary_name='fake.so',binary_sha256=sha(package/'fake.so'),binary_bytes=(package/'fake.so').stat().st_size)
        dump(self.root/'configs/0017-libraries.json', {'current':native})
        self.golden = {str(x.relative_to(self.root)):sha(x) for x in p.source_paths(self.args.protocol)}
        def verify_frozen(commit, paths):
            if commit != self.args.expected_commit: raise ValueError('wrong fake freeze')
            result = {str(x.resolve().relative_to(self.root)):sha(x) for x in paths}
            if any(self.golden[k] != v for k,v in result.items()): raise ValueError('changed fake source')
            return result
        self.sup = types.SimpleNamespace(THREAD_KEYS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'),
                    HELPERS={'frontend':('range_inputs.py','propose_plan.py','acquire_operation.py')},
                    verify_frozen=mock.Mock(side_effect=verify_frozen))
        self.q = types.SimpleNamespace(read_json=lambda path:json.loads(Path(path).read_text()),sha=sha,
            safe_child=lambda root,name:Path(root)/name,frozen_bindings=mock.Mock(return_value={}),
            verify_files=mock.Mock(return_value=({},None,{},dict(installation='f'*64))),install_guard=mock.Mock())
        self.chain = dict(input_operation_sha256='1'*64,input_supervision_sha256='2'*64,
                          input_supervision_log_sha256=dict(stdout='3'*64,stderr='4'*64))
        self.front = types.SimpleNamespace(q=self.q,qualification_args=lambda args:args,
            verify_qualification=mock.Mock(return_value=dict(qualification='e'*64)),
            acquisition=types.SimpleNamespace(supervisor=self.sup,verify_completed_chain=mock.Mock(return_value=self.chain)),
            verify_inputs=mock.Mock(return_value=(None,{'image.jpg':'d'*64})))
        self.front_record = dict(schema='evr0023.frontend.v1',status='completed',eligible=True,
            expected_commit=self.args.expected_commit,frontend_freeze='b'*40,plan_sha256=self.args.plan_sha256,
            input_receipt_sha256='c'*64,wall_seconds=12.,frontend_bindings=dict(installation='f'*64,qualification='e'*64),
            input_acquisition_bindings=self.chain,input_sha256={'image.jpg':'d'*64})
        own = [self.here/name for name in ('frontend.py','range_inputs.py','propose_plan.py','acquire_operation.py','supervise.py')]+[self.args.protocol]
        self.front_record['source_sha256'] = verify_frozen(self.args.expected_commit, own)
        self.outer = dict(schema='evr0023.supervision.v1',operation='frontend',status='completed',exit_code=0,
            expected_commit=self.args.expected_commit,termination=None,remaining_descendant_pids=[],cleanup_errors=[],
            hard_timeout_seconds=1800,group_rss_limit_kib=8*1024**2,wall_seconds=13.,peak_group_rss_kib=100,
            thread_environment={k:'4' for k in self.sup.THREAD_KEYS},
            source_sha256=verify_frozen(self.args.expected_commit,[self.here/name for name in
                ('supervise.py','frontend.py','range_inputs.py','propose_plan.py','acquire_operation.py')]))
        paths={key:self.private/key for key in ('input-receipt','input-operation','input-supervision','inputs','installation',
                'installation-supervision','acquisition-receipt','assets-dir','qualification-receipt','qualification-supervision')}
        paths.update(plan=self.args.plan,protocol=self.args.protocol,**{'output-dir':self.args.frontend_cache})
        flags={**{k:str(v) for k,v in paths.items()},'plan-sha256':self.args.plan_sha256,
               'input-receipt-sha256':'c'*64,'expected-commit':self.args.expected_commit,'frontend-freeze':'b'*40}
        self.outer['command']=['$PYTHON','$REPOSITORY/experiments/0023-mapfree/frontend.py']
        for k,v in flags.items(): self.outer['command'] += ['--'+k,v]
        self.refresh()
        self.module_patch=mock.patch.object(p,'module',return_value=self.front);self.module_patch.start();self.addCleanup(self.module_patch.stop)

    def refresh(self):
        dump(self.args.frontend_cache/'receipt.json',self.front_record)
        folder=self.args.frontend_supervision.parent;folder.mkdir(parents=True,exist_ok=True)
        (folder/'stdout.log').write_text(json.dumps({k:self.front_record[k] for k in ('status','eligible','wall_seconds')})+'\n')
        (folder/'stderr.log').write_text('')
        for k in ('stdout','stderr'):self.outer[k+'_sha256']=sha(folder/(k+'.log'))
        dump(self.args.frontend_supervision,self.outer)

    def fake_runtime(self, fault=None):
        fields=('source_sha256','frontend_bindings','input_acquisition_bindings','input_sha256','expected_commit',
                'frontend_freeze','plan_sha256','input_receipt_sha256')
        self.cache=types.ModuleType('cache_inputs');self.cache.BINDING_FIELDS=fields
        self.cache.load_inputs=mock.Mock(return_value=dict(selection_receipt={'fixed':True},
                scenes=[dict(scene=name) for name in ('one','two','three')]))
        self.cache.verify_cache=mock.Mock(return_value={})
        self.study=types.ModuleType('study');self.study.ARMS=tuple(x for x in p.METHOD_NAMES if x!='native_E')
        self.study.serial=lambda value:value
        def run_scene(scene, backend, emit):
            if fault=='interrupt' and scene['scene']=='two':raise KeyboardInterrupt('artificial interrupt')
            if fault=='source' and scene['scene']=='three':(self.here/'study.py').write_text('changed')
            if fault=='outer' and scene['scene']=='three':self.args.frontend_supervision.write_text('{}')
            count=7 if fault=='short' and scene['scene']=='three' else 8
            return dict(scene=scene['scene'],queries=[dict(query='B'+str(i+1),arms={a:{'pose':None} for a in self.study.ARMS}) for i in range(count)])
        self.study.run_scene=mock.Mock(side_effect=run_scene)
        native=types.ModuleType('poselib');native.__version__='3.0.0';native.__file__=str(self.args.native_runtime/'poselib/__init__.py')
        numpy=types.ModuleType('numpy');numpy.__version__='2.0.2'
        scipy=types.ModuleType('scipy');scipy.__version__='1.13.1'
        return {'numpy':numpy,'scipy':scipy,'poselib':native,'cache_inputs':self.cache,'study':self.study}

    def execute(self, fault=None):
        with mock.patch.dict(sys.modules,self.fake_runtime(fault)),mock.patch.object(p.sys,'path',list(sys.path)), \
             mock.patch.dict(p.os.environ),mock.patch.object(p.sys,'dont_write_bytecode',sys.dont_write_bytecode):
            return p.run(self.args)

    def test_admission_binds_complete_outer_sources_logs_and_both_chains(self):
        _,fa,record,bindings,_=p.frontend_admission(self.args)
        self.assertEqual(record,self.front_record);self.assertEqual(fa.output_dir,self.args.frontend_cache)
        self.assertEqual(bindings['frontend_supervision_sha256'],sha(self.args.frontend_supervision))
        self.front.verify_qualification.assert_called_once();self.front.acquisition.verify_completed_chain.assert_called_once()
        self.assertIn('configs/0017-libraries.json',self.golden)

    def test_failed_outer_and_wrong_sources_block_before_model_chain(self):
        self.outer['exit_code']=1;self.refresh()
        with self.assertRaisesRegex(ValueError,'outer failure'):p.frontend_admission(self.args)
        self.q.verify_files.assert_not_called()
        self.outer['exit_code']=0;self.outer['source_sha256']['experiments/0023-mapfree/supervise.py']='0'*64;self.refresh()
        with self.assertRaisesRegex(ValueError,'supervisor source'):p.frontend_admission(self.args)
        self.q.verify_files.assert_not_called()

    def test_log_tamper_and_command_misdirection_rejected(self):
        (self.args.frontend_supervision.parent/'stderr.log').write_text('changed')
        with self.assertRaisesRegex(ValueError,'log changed'):p.frontend_admission(self.args)
        self.refresh();at=self.outer['command'].index('--output-dir');self.outer['command'][at+1]=str(self.private/'different')
        self.refresh()
        with self.assertRaisesRegex(ValueError,'command inputs'):p.frontend_admission(self.args)

    def test_native_binary_change_blocks_before_guard_or_arrays(self):
        (self.args.native_runtime/'poselib/fake.so').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'native source/binary'):self.execute()
        self.q.install_guard.assert_not_called();self.cache.load_inputs.assert_not_called()
        self.assertFalse((self.args.output_dir/'seal.json').exists())

    def test_full24_seal_hashes_and_no_subprocess_verification_after_guard(self):
        def guard(*args):
            self.sup.verify_frozen.side_effect=AssertionError('Git proof after offline guard')
        self.q.install_guard.side_effect=guard
        record=self.execute();self.assertTrue(record['eligible']);self.assertEqual(record['query_count'],24)
        seal=json.loads((self.args.output_dir/'seal.json').read_text())
        self.assertTrue(seal['complete']);self.assertEqual(seal['receipt_sha256'],sha(self.args.output_dir/'receipt.json'))
        self.assertEqual(len(seal['outputs']),4)
        for name,row in seal['outputs'].items():self.assertEqual(row['sha256'],sha(self.args.output_dir/name))
        self.assertEqual(self.study.run_scene.call_count,3);self.assertFalse(record['ground_truth_access'])

    def test_partial_scene_and_interruption_preserve_failure_without_seal(self):
        for fault,error in [('short',ValueError),('interrupt',KeyboardInterrupt)]:
            with self.subTest(fault=fault):
                self.args.output_dir=self.private/fault
                with self.assertRaises(error):self.execute(fault)
                record=json.loads((self.args.output_dir/'receipt.json').read_text())
                self.assertFalse(record['eligible']);self.assertEqual(record['status'],'failed')
                self.assertFalse((self.args.output_dir/'seal.json').exists())
                self.assertIn('scene-one.json',record['outputs'])

    def test_source_mutation_blocks_seal(self):
        with self.assertRaisesRegex(ValueError,'source changed'):self.execute('source')
        self.assertFalse((self.args.output_dir/'seal.json').exists())
        self.assertFalse(json.loads((self.args.output_dir/'receipt.json').read_text())['eligible'])

    def test_frontend_supervisor_mutation_blocks_seal(self):
        with self.assertRaisesRegex(ValueError,'frontend supervision'):self.execute('outer')
        self.assertFalse((self.args.output_dir/'seal.json').exists())


if __name__=='__main__':unittest.main(verbosity=2)
