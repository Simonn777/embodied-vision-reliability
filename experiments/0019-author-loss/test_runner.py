"""Artificial-only control, failure-classification and full-seal tests; no native/H5 reads."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE/filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

c = load('evr0019_test_common', 'common.py')
with c.aliases({'common': c}):
    worker = load('evr0019_test_worker', 'predict_pair.py')
with c.aliases({'common': c, 'predict_pair': worker}):
    runner = load('evr0019_test_runner', 'run_diagnostic.py')


class ControlTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name); self.data = self.root/'data'; self.data.mkdir()
        self.manifest_path = self.root/'manifest.json'
        self.seal = self.root/'baseline.json'; self.seal.write_text('{}')
        self.library = {'version': '3.0.0', 'binary_sha256': 'a'*64}
        self.sources = {'fake_source': 'b'*64}
        rows = []
        for i in range(80):
            path = self.data/f'scene-{i//16}.h5'
            if not path.exists(): path.write_bytes(b'ARTIFICIAL, NOT HDF5')
            rows.append({'pair_index': i, 'scene': f'scene-{i//16}', 'key': f'key-{i}',
                'filename': path.name, 'dataset_sha256': c.sha256(path), 'dataset_bytes': path.stat().st_size})
        self.manifest = {'selected_pairs': rows}; self.manifest_path.write_text(json.dumps(self.manifest))
        self.mh = c.sha256(self.manifest_path)
        self.binding = {'seal_sha256': c.sha256(self.seal),
            'predictions': {f'{i}:{a}': 'd'*64 for i in range(80) for a in c.ARMS}}
        self.baseline = {(i, a): self.base_record(a) for i in range(80) for a in c.ARMS}
        self.stack = contextlib.ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(c, 'baseline_bundle', return_value=(self.manifest, self.baseline, self.binding)))
        self.stack.enter_context(mock.patch.object(c, 'source_hashes', return_value=self.sources))
        self.stack.enter_context(mock.patch.object(c, 'library_config', return_value={'current': self.library}))

    def base_record(self, arm):
        estimate = {'R': c.np.eye(3).tolist(), 't': [2., 0., 0.]}
        return {'status': 'completed', 'options': c.base.options(arm), 'estimate': estimate,
            'geometry': {**estimate, 'scale': 1., 'shift1': 0., 'shift2': 0.},
            'input_arrays': {k: {'shape': [8, 2] if k.startswith('x') else [8], 'dtype': 'float64', 'sha256': 'c'*64}
                             for k in ('x1', 'x2', 'depth1', 'depth2')},
            'calibration': {'artificial': True}, 'input_diagnostics': {'count': 8},
            'mask_length': 8, 'kept_count': 8, 'native_inliers_packbits_big_hex': 'ff',
            'mask_sha256': hashlib.sha256(c.np.ones(8, dtype=bool).tobytes()).hexdigest(),
            'returned_ransac_stats': {'iterations': 100, 'refinements': 4, 'model_score': 10.125},
            'final_scores': {'pixel_msac': .5},
            'depth_diagnostics': {'all': {'denominator': 8}, 'returned_mask': {'denominator': 8}}}

    def raw(self, i=0, arm='depth_scale'):
        record = copy.deepcopy(self.baseline[i, arm])
        record.update(worker.identity(self.manifest['selected_pairs'][i], arm, self.mh, self.sources, self.library, self.binding))
        record.update(phase='complete', timings={k: .1 for k in ('binding_seconds','input_seconds','call_seconds','score_seconds')})
        record['timings']['worker_total_seconds'] = .6
        record['prefinal_parity'] = c.check_parity(record, self.baseline[i, arm])
        return record

    @staticmethod
    def process():
        return {'exit_code': 0, 'termination': None, 'wall_seconds': 1.,
                'sampled_peak_rss_kib': 1024, 'watchdog_error': None}

    def failed(self, kind='timeout'):
        raw, process = self.raw(), self.process()
        for key in ('estimate','geometry','final_scores','depth_diagnostics','returned_ransac_stats',
                    'native_inliers_packbits_big_hex','mask_sha256','mask_length','kept_count','prefinal_parity'):
            raw.pop(key)
        raw.update(status='started', phase='estimator')
        raw['timings'].update(call_seconds=None, score_seconds=None, worker_total_seconds=None)
        process.update(exit_code=-9, termination='pair_timeout' if kind == 'timeout' else None)
        if kind == 'exception':
            raw.update(status='error', phase='error', failed_phase='estimator', failure_kind='estimator', error='Artificial')
            raw['timings'].update(call_seconds=.2, worker_total_seconds=.6)
            process['exit_code'] = 1
        return raw, process

    def fake_child(self, command, log, remaining, environment):
        get = lambda flag: command[command.index(flag)+1]
        i, arm = int(get('--pair-index')), get('--arm'); self.calls.append((i, arm))
        self.assertTrue(all(environment[k] == '1' for k in runner.THREAD_VARIABLES))
        self.assertEqual(get('--baseline-seal'), str(self.seal.resolve()))
        raw, process = self.raw(i, arm), self.process()
        if self.mutate: self.mutate(raw, process)
        Path(log).write_text('artificial log')
        c.write_new(Path(get('--output')), raw)
        return process

    def run_fake(self, mutate=None, dirname='predictions'):
        self.calls, self.mutate = [], mutate
        out = self.root/dirname
        with mock.patch.object(runner, 'run_child', side_effect=self.fake_child), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = runner.run(self.data, self.manifest_path, self.seal, out, self.root/'native', self.root/'deps')
        return code, out

    def test_only_author_loss_changes_and_native_signature(self):
        pair = {k: object() for k in ('x1','x2','depth1','depth2')}
        pair['calibration'] = {'camera1': object(), 'camera2': object()}
        backend = types.SimpleNamespace(estimate_monodepth_relative_pose=mock.Mock(return_value=('geometry','info')))
        for arm in c.ARMS:
            before = c.base.options(arm); after = c.options(arm)
            self.assertEqual(after.pop('bundle'), {'loss_type': 'TRUNCATED_CAUCHY'})
            self.assertEqual(before.pop('bundle'), {}); self.assertEqual(after, before)
            self.assertEqual(worker.invoke(backend, pair, arm), ('geometry','info'))
            args, kw = backend.estimate_monodepth_relative_pose.call_args
            self.assertEqual(args[:4], tuple(pair[k] for k in ('x1','x2','depth1','depth2')))
            self.assertEqual(args[4:6], tuple(pair['calibration'][k] for k in ('camera1','camera2')))
            self.assertEqual(args[6], c.options(arm)); self.assertEqual(kw, {'initial_pose': None})
        with self.assertRaises(ValueError): c.options('e_norm')

    def test_parity_constrains_prefinal_not_final_geometry(self):
        raw = self.raw(); raw['geometry']['scale'] = 19.; raw['estimate']['t'] = [0., 3., 0.]
        self.assertTrue(c.check_parity(raw, self.baseline[0,'depth_scale'])['passed'])
        c.parity(1.+.5e-12, 1.)
        for a, b in ((1.+2e-12,1.),(101,100),(100.,100),(float('nan'),1.)):
            with self.assertRaises(ValueError): c.parity(a,b)
        for key, value in (('kept_count',7), ('mask_sha256','0'*64)):
            modified = self.raw(); modified[key] = value
            with self.assertRaises(ValueError): c.check_parity(modified, self.baseline[0,'depth_scale'])
        raw = self.raw(); raw['returned_ransac_stats']['iterations'] += 1
        with self.assertRaises(ValueError): c.check_parity(raw, self.baseline[0,'depth_scale'])

    def test_native_failures_eligible_with_unavailable_parity_and_input_failures_block(self):
        expected = worker.identity(self.manifest['selected_pairs'][0], 'depth_scale', self.mh, self.sources, self.library, self.binding)
        for kind, status in (('timeout','timeout'),('crash','estimator_error'),('exception','estimator_error')):
            raw, process = self.failed(kind)
            final = runner.wrap_record(raw, expected, process, self.baseline[0,'depth_scale'])
            self.assertEqual(final['status'], status); self.assertFalse(final['prefinal_parity']['available'])
            self.assertIsNone(final['prefinal_parity']['passed']); self.assertEqual(final['timings']['charged_total_seconds'],1.)
        for field, value in (('phase','inputs'),('input_arrays',{})):
            raw, process = self.failed(); raw[field] = value
            with self.assertRaises(ValueError): runner.wrap_record(raw, expected, process, self.baseline[0,'depth_scale'])
        raw, process = self.failed('exception'); raw['failed_phase'] = 'inputs'
        with self.assertRaises(ValueError): runner.wrap_record(raw, expected, process, self.baseline[0,'depth_scale'])
        for termination in ('total_timeout','rss_limit','watchdog_error'):
            raw, process = self.failed(); process['termination'] = termination
            with self.assertRaises(ValueError): runner.wrap_record(raw, expected, process, self.baseline[0,'depth_scale'])

    def test_complete_160_rotation_and_failure_denominator_without_h5(self):
        def one_failure(raw, process):
            if (raw['pair_index'],raw['arm']) == (0,'depth_scale'):
                f,p = self.failed('exception'); raw.clear(); raw.update(f); process.update(p)
        with mock.patch.dict(sys.modules, {'h5py': types.SimpleNamespace(File=mock.Mock(side_effect=AssertionError('No H5')))}):
            code, out = self.run_fake(one_failure); self.assertEqual(code,0)
            _,seal,records = runner.verify_seal(self.manifest_path,self.seal,out/'seal.json')
        self.assertEqual(len(records),160); self.assertEqual(seal['completed_calls'],159)
        self.assertEqual(seal['native_failure_calls'],1); self.assertEqual(records[0]['status'],'estimator_error')
        self.assertEqual(self.calls[:4],[(0,'depth_scale'),(0,'depth_affine'),(1,'depth_affine'),(1,'depth_scale')])

    def test_completed_parity_failure_stops_and_preserves_raw_process_log(self):
        def mutate(raw, process): raw['returned_ransac_stats']['refinements'] += 1
        code, out = self.run_fake(mutate); self.assertEqual(code,1); self.assertEqual(len(self.calls),1)
        for suffix in ('.worker.json','.process.json','.log'):
            self.assertTrue((out/('pair-00-depth_scale'+suffix)).exists())
        self.assertEqual(json.loads((out/'seal.json').read_text())['status'],'ineligible')
        with self.assertRaises(ValueError): runner.verify_seal(self.manifest_path,self.seal,out/'seal.json')

    def test_changed_options_or_inputs_stop_even_on_estimator_failure(self):
        for key, value in (('options',{'bundle':{}}),('input_arrays',{})):
            def mutate(raw, process):
                raw.update(status='error',failure_kind='estimator',phase='error'); process['exit_code']=1
                raw[key] = value
            code,out = self.run_fake(mutate,dirname='changed-'+key)
            self.assertEqual(code,1); self.assertEqual(len(self.calls),1)

    def test_tampered_log_source_output_missing_case_and_unsafe_paths_block(self):
        code,out = self.run_fake(); self.assertEqual(code,0)
        seal = json.loads((out/'seal.json').read_text()); row = seal['rows'][0]
        for filename in (row['path'],row['raw_path'],row['process_path'],'pair-00-depth_scale.log'):
            path=out/filename; original=path.read_bytes(); path.write_bytes(original+b' ')
            with self.assertRaises(ValueError): runner.verify_seal(self.manifest_path,self.seal,out/'seal.json')
            path.write_bytes(original)
        with mock.patch.object(c,'source_hashes',return_value={'changed':'f'*64}), self.assertRaises(ValueError):
            runner.verify_seal(self.manifest_path,self.seal,out/'seal.json')
        changed=copy.deepcopy(seal);changed['rows'][0]['raw_path']='../outside'
        (out/'seal.json').write_text(json.dumps(changed))
        with self.assertRaises(ValueError): runner.verify_seal(self.manifest_path,self.seal,out/'seal.json')
        seal['rows'].pop(); (out/'seal.json').write_text(json.dumps(seal))
        with self.assertRaises(ValueError): runner.verify_seal(self.manifest_path,self.seal,out/'seal.json')
        with self.assertRaises(ValueError): c.bound_file(out,'../outside')

    def test_existing_directory_refuses_before_binding_and_zero_budget_stops_before_child(self):
        out=self.root/'existing';out.mkdir()
        with mock.patch.object(c,'baseline_bundle',side_effect=AssertionError('No read')), self.assertRaises(FileExistsError):
            runner.run(self.data,self.manifest_path,self.seal,out,self.root,self.root)
        with mock.patch.object(runner,'TOTAL_TIMEOUT',0):
            code,out=self.run_fake(dirname='budget'); self.assertEqual(code,1); self.assertEqual(self.calls,[])

    def test_worker_persists_estimator_phase_and_error_timing_with_fake_backend(self):
        output=self.root/'worker.json'; arm='depth_scale'; baseline=self.baseline[0,arm]
        pair={k: c.np.zeros((8,2) if k.startswith('x') else 8) for k in ('x1','x2','depth1','depth2')}
        pair.update(calibration=baseline['calibration'],input_diagnostics=baseline['input_diagnostics'])
        sentinel=object()
        def fake_read(handle,key):
            self.assertIs(handle,sentinel);self.assertEqual(key,'key-0');return pair
        def fake_call(backend,actual,actual_arm):
            saved=json.loads(output.read_text());self.assertEqual(saved['phase'],'estimator')
            self.assertEqual(saved['input_arrays'],baseline['input_arrays'])
            raise RuntimeError('Artificial estimator failure')
        fake_h5=types.SimpleNamespace(File=mock.Mock(return_value=contextlib.nullcontext(sentinel)))
        with mock.patch.dict(sys.modules,{'h5py':fake_h5,'poselib':object()}), \
             mock.patch.object(c,'read_pair',side_effect=fake_read), \
             mock.patch.object(c,'input_hashes',return_value=baseline['input_arrays']), \
             mock.patch.object(c,'verify_library',return_value=self.library), \
             mock.patch.object(worker,'invoke',side_effect=fake_call):
            code=worker.run(self.data/'scene-0.h5',self.manifest_path,self.seal,output,arm,0)
        self.assertEqual(code,1);record=json.loads(output.read_text())
        self.assertEqual(record['failure_kind'],'estimator');self.assertEqual(record['failed_phase'],'estimator')
        self.assertGreaterEqual(record['timings']['call_seconds'],0.)
        self.assertIsNone(record['timings']['score_seconds'])
        with self.assertRaises(FileExistsError): worker.run(self.root,self.root,self.root,output,arm,0)

    def test_import_aliases_restore_and_timing_components_are_checked(self):
        before=sys.modules.get('common'); token=object()
        with self.assertRaises(RuntimeError):
            with c.aliases({'common':token}):
                self.assertIs(sys.modules['common'],token);raise RuntimeError('artificial')
        self.assertIs(sys.modules.get('common'),before)
        raw=self.raw();raw['timings']['call_seconds']=10.
        expected=worker.identity(self.manifest['selected_pairs'][0],'depth_scale',self.mh,self.sources,self.library,self.binding)
        with self.assertRaises(ValueError): runner.wrap_record(raw,expected,self.process(),self.baseline[0,'depth_scale'])

if __name__=='__main__': unittest.main()
