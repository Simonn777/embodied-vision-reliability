"""Artificial-only replay, provenance, trace and parity tests; no cache/native calls."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import common as c
import predict_pair as worker
import run_diagnostic as runner


class ReplayTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root/'data'; self.data.mkdir()
        self.manifest_path = self.root/'manifest.json'
        self.seal = self.root/'baseline.json'; self.seal.write_text('{}')
        self.library = {'version': '3.0.0', 'binary_sha256': 'a'*64}
        self.sources = {'fake_source': 'b'*64}
        rows = []
        for i in range(80):
            filename = f'scene-{i//16}.h5'
            p = self.data/filename
            if not p.exists(): p.write_bytes(b'ARTIFICIAL, NOT HDF5')
            rows.append({'pair_index': i, 'scene': f'scene-{i//16}', 'key': f'key-{i}',
                         'filename': filename, 'dataset_sha256': c.sha256(p), 'dataset_bytes': p.stat().st_size})
        self.manifest = {'selected_pairs': rows}
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.mh = c.sha256(self.manifest_path)
        self.binding = {'seal_sha256': c.sha256(self.seal), 'predictions': {f'{i}:{a}': 'd'*64 for i in range(80) for a in c.ARMS}}
        self.baseline = {(i, a): self.base_record(i, a) for i in range(80) for a in c.ARMS}
        self.stack = contextlib.ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(c, 'baseline_bundle', return_value=(self.manifest, self.baseline, self.binding)))
        self.stack.enter_context(mock.patch.object(c, 'source_hashes', return_value=self.sources))
        self.stack.enter_context(mock.patch.object(c, 'library_config', return_value={'instrumented': self.library}))

    def base_record(self, i, arm):
        estimate = {'R': c.np.eye(3).tolist(), 't': [2., 0., 0.]}
        return {'status': 'completed', 'options': c.options(arm), 'estimate': estimate,
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
        r = worker.identity(self.manifest['selected_pairs'][i], arm, self.mh, self.sources, self.library, self.binding)
        r.update(copy.deepcopy(self.baseline[i, arm]))
        r.update(phase='complete', timings={k: .1 for k in ('binding_seconds', 'input_seconds', 'call_seconds',
                     'score_seconds', 'trace_save_seconds', 'worker_total_seconds')})
        r['timings']['worker_total_seconds'] = .6
        r['parity'] = c.check_parity(r, self.baseline[i, arm])
        return r

    def trace(self, record):
        binding = {k: record[k] for k in ('pair_index', 'scene', 'key', 'arm', 'source_sha256', 'manifest_sha256',
            'library', 'baseline_seal_sha256', 'baseline_prediction_sha256', 'input_arrays', 'calibration', 'dataset_sha256_verified')}
        return {'schema': 'evr0018.trace.v1', 'bindings': binding,
                'native': self.native_trace()}

    @staticmethod
    def native_trace():
        return {'schema': 'evr0018.native-trace.v1', 'complete': True, 'capacity': c.TRACE_CAPACITY,
                'event_count': 2, 'dropped_events': 0, 'proposal_trace_hash_hex': '0'*16,
                'sampled_draws': 0, 'scored_proposals': 0,
                'events': [{'kind': 'pipeline', 'phase': 'begin', 'lo_id': -1, 'finite': True},
                           {'kind': 'pipeline', 'phase': 'returned', 'lo_id': -1, 'finite': True}]}

    @staticmethod
    def process():
        return {'exit_code': 0, 'termination': None, 'wall_seconds': 1., 'sampled_peak_rss_kib': 1024, 'watchdog_error': None}

    def fake_child(self, command, log_path, remaining, environment):
        get = lambda flag: command[command.index(flag)+1]
        i, arm = int(get('--pair-index')), get('--arm')
        self.calls.append((i, arm))
        self.assertTrue(all(environment[k] == '1' for k in runner.THREAD_VARIABLES))
        self.assertEqual(get('--baseline-seal'), str(self.seal.resolve()))
        Path(log_path).write_text('artificial log')
        raw = self.raw(i, arm)
        raw['trace_byte_budget'] = int(get('--trace-byte-budget'))
        output = Path(get('--output')); tp = output.with_suffix('.trace.json')
        trace = self.trace(raw)
        if self.mutate_trace: self.mutate_trace(trace)
        c.write_new(tp, trace)
        raw.update(trace_path=tp.name, trace_sha256=c.sha256(tp), trace_bytes=tp.stat().st_size,
                   trace_required_bytes=tp.stat().st_size, trace_event_count=2)
        if self.mutate: self.mutate(raw)
        c.write_new(output, raw)
        return self.process()

    def run_fake(self, mutate=None, mutate_trace=None, dirname='predictions'):
        self.calls, self.mutate, self.mutate_trace = [], mutate, mutate_trace
        out = self.root/dirname
        with mock.patch.object(runner, 'run_child', side_effect=self.fake_child), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = runner.run(self.data, self.manifest_path, self.seal, out, self.root/'native', self.root/'deps')
        return code, out

    def test_complete_160_cohort_rotation_and_separate_trace_bindings(self):
        with mock.patch.dict(sys.modules, {'h5py': types.SimpleNamespace(File=mock.Mock(side_effect=AssertionError('No H5')))}):
            code, out = self.run_fake()
            self.assertEqual(code, 0)
            _, seal, records = runner.verify_seal(self.manifest_path, self.seal, out/'seal.json')
        self.assertEqual(len(records), 160)
        self.assertTrue(seal['all_160_parity_passed'])
        self.assertEqual(self.calls[:4], [(0, 'depth_scale'), (0, 'depth_affine'), (1, 'depth_affine'), (1, 'depth_scale')])
        self.assertEqual(records[0]['timings']['charged_total_seconds'], 1.)
        self.assertEqual(sum(r['trace_bytes'] for r in records), seal['trace_bytes'])

    def test_componentwise_float_bound_exact_discrete_and_raw_gauge(self):
        c.parity(1.+.5e-12, 1.)
        for actual, expected in ((1.+2e-12, 1.), (101, 100), (100., 100), (float('nan'), 1.)):
            with self.assertRaises(ValueError): c.parity(actual, expected)
        r = self.raw(); r['geometry']['t'] = [1., 0., 0.]
        with self.assertRaises(ValueError): c.check_parity(r, self.baseline[0, 'depth_scale'])

    def test_any_parity_failure_blocks_cohort_and_preserves_worker_log(self):
        def mutate(raw): raw['returned_ransac_stats']['refinements'] += 1
        code, out = self.run_fake(mutate=mutate)
        self.assertEqual(code, 1); self.assertEqual(len(self.calls), 1)
        self.assertTrue((out/'pair-00-depth_scale.worker.json').exists())
        self.assertTrue((out/'pair-00-depth_scale.process.json').exists())
        self.assertEqual(json.loads((out/'seal.json').read_text())['status'], 'ineligible')
        with self.assertRaises(ValueError): runner.verify_seal(self.manifest_path, self.seal, out/'seal.json')

    def test_trace_overflow_and_global_byte_limit_fail_closed(self):
        code, out = self.run_fake(mutate_trace=lambda t: t['native'].update(complete=False))
        self.assertEqual(code, 1)
        self.assertTrue((out/'pair-00-depth_scale.worker.trace.json').exists())
        r = self.raw(); trace = self.trace(r)['native']
        trace['capacity'] -= 1
        with self.assertRaises(ValueError): c.validate_trace(trace)
        with mock.patch.object(c, 'COHORT_TRACE_BYTES_LIMIT', 1):
            code, capped = self.run_fake(dirname='capped')
            self.assertEqual(code, 1)
            self.assertIn('byte bound', json.loads((capped/'seal.json').read_text())['error'])
        self.assertEqual(json.loads((out/'seal.json').read_text())['all_160_parity_passed'], False)

    def test_timeout_crash_incomplete_and_unavailable_trace_never_eligible(self):
        expected = worker.identity(self.manifest['selected_pairs'][0], 'depth_scale', self.mh, self.sources, self.library, self.binding)
        for status, code, termination in [('started', -9, 'pair_timeout'), ('error', 1, None), ('completed', -9, 'rss_limit')]:
            raw = self.raw(); raw['status'] = status
            p = {**self.process(), 'exit_code': code, 'termination': termination}
            with self.subTest(status=status), self.assertRaises(ValueError):
                runner.wrap_record(raw, expected, p, self.baseline[0, 'depth_scale'])

    def test_tampered_trace_log_source_and_missing_case_block_verification(self):
        code, out = self.run_fake(); self.assertEqual(code, 0)
        seal = json.loads((out/'seal.json').read_text()); row = seal['rows'][0]
        for filename in (row['trace_path'], 'pair-00-depth_scale.log'):
            p = out/filename; original = p.read_bytes(); p.write_bytes(original+b' ')
            with self.assertRaises(ValueError): runner.verify_seal(self.manifest_path, self.seal, out/'seal.json')
            p.write_bytes(original)
        with mock.patch.object(c, 'source_hashes', return_value={'changed': 'f'*64}), self.assertRaises(ValueError):
            runner.verify_seal(self.manifest_path, self.seal, out/'seal.json')
        seal['rows'].pop(); (out/'seal.json').write_text(json.dumps(seal))
        with self.assertRaises(ValueError): runner.verify_seal(self.manifest_path, self.seal, out/'seal.json')

    def test_unsafe_paths_and_one_shot_directory(self):
        with self.assertRaises(ValueError): c.bound_file(self.root, '../outside')
        out = self.root/'exists'; out.mkdir()
        with mock.patch.object(c, 'baseline_bundle', side_effect=AssertionError('No binding read')), self.assertRaises(FileExistsError):
            runner.run(self.data, self.manifest_path, self.seal, out, self.root, self.root)

    def test_isolated_import_aliases_restore_even_on_exception(self):
        prior = sys.modules['common']; absent = 'evr0018_artificial_alias'
        with self.assertRaises(RuntimeError):
            with c.aliases({'common': object(), absent: object()}): raise RuntimeError('artificial')
        self.assertIs(sys.modules['common'], prior); self.assertNotIn(absent, sys.modules)
        self.assertIs(c.base_runner.c, c.base)
        self.assertIs(c.base_runner.worker, c.base_worker)

    def test_dispatch_has_only_frozen_inputs_and_options(self):
        pair = {k: object() for k in ('x1', 'x2', 'depth1', 'depth2')}
        pair['calibration'] = {'camera1': {'a': 1}, 'camera2': {'b': 2}}
        backend = types.SimpleNamespace(estimate_monodepth_relative_pose_trace=mock.Mock(return_value=('geometry', {}, {})))
        for arm in c.ARMS: worker.invoke(backend, pair, arm)
        for call, arm in zip(backend.estimate_monodepth_relative_pose_trace.call_args_list, c.ARMS):
            self.assertEqual(call.args, (pair['x1'], pair['x2'], pair['depth1'], pair['depth2'],
                pair['calibration']['camera1'], pair['calibration']['camera2'], c.options(arm)))
            self.assertEqual(call.kwargs, {'initial_pose': None, 'trace_capacity': c.TRACE_CAPACITY})

    def test_watchdog_failure_kills_and_waits(self):
        child = mock.Mock(pid=123, poll=mock.Mock(return_value=None), wait=mock.Mock(return_value=-9))
        with mock.patch.object(c.base_runner.subprocess, 'Popen', return_value=child), mock.patch.object(c.base_runner.subprocess,
            'run', side_effect=subprocess.TimeoutExpired('ps', 2)):
            result = runner.run_child(['fake'], self.root/'watchdog.log', 1800, {})
        child.kill.assert_called_once(); child.wait.assert_called_once_with(timeout=5)
        self.assertEqual(result['termination'], 'watchdog_error')

    def worker_context(self, native_error=None, bad_inputs=False):
        pair = {k: c.np.ones((8, 2) if k.startswith('x') else 8) for k in ('x1', 'x2', 'depth1', 'depth2')}
        pair.update(calibration={'camera1': {'artificial': 1}, 'camera2': {'artificial': 2}}, input_diagnostics={'count': 8})
        original = self.baseline[0, 'depth_scale']
        original.update(input_arrays=c.input_hashes(pair), calibration=pair['calibration'])
        if bad_inputs:
            original['input_arrays']['x1']['sha256'] = 'f'*64
        backend = types.SimpleNamespace(estimate_monodepth_relative_pose_trace=mock.Mock(
            side_effect=native_error, return_value=(object(), {}, self.native_trace())))
        handle = object()
        file_factory = mock.Mock(return_value=contextlib.nullcontext(handle))
        self.stack.enter_context(mock.patch.dict(sys.modules, {'poselib': backend, 'h5py': types.SimpleNamespace(File=file_factory)}))
        self.stack.enter_context(mock.patch.object(c, 'verify_library', return_value=self.library))
        reader = self.stack.enter_context(mock.patch.object(c, 'read_pair', return_value=pair))
        fields = ('estimate', 'geometry', 'returned_ransac_stats', 'final_scores', 'depth_diagnostics',
                  'mask_length', 'kept_count', 'mask_sha256', 'native_inliers_packbits_big_hex')
        self.stack.enter_context(mock.patch.object(c.base_worker, 'unpack_result', return_value={k: original[k] for k in fields}))
        return backend, reader, handle, file_factory

    def test_worker_fake_backend_writes_bound_trace_without_labels(self):
        backend, reader, handle, file_factory = self.worker_context()
        out = self.root/'worker.json'
        code = worker.run(self.data/self.manifest['selected_pairs'][0]['filename'], self.manifest_path,
                          self.seal, out, 'depth_scale', 0)
        self.assertEqual(code, 0)
        record = json.loads(out.read_text())
        self.assertEqual(record['status'], 'completed')
        self.assertTrue(record['parity']['passed'])
        self.assertGreater(record['timings']['call_seconds'], 0)
        reader.assert_called_once_with(handle, 'key-0')
        file_factory.assert_called_once()
        backend.estimate_monodepth_relative_pose_trace.assert_called_once()
        self.assertGreater(runner.check_trace(self.root, record), 0)

    def test_worker_input_mismatch_prevents_call_and_native_error_retains_time(self):
        backend, _, _, _ = self.worker_context(bad_inputs=True)
        out = self.root/'worker-error.json'
        code = worker.run(self.data/self.manifest['selected_pairs'][0]['filename'], self.manifest_path,
                          self.seal, out, 'depth_scale', 0)
        self.assertEqual(code, 1)
        backend.estimate_monodepth_relative_pose_trace.assert_not_called()
        self.assertEqual(json.loads(out.read_text())['failure_kind'], 'input')
        # Restore the fixture inputs and test an independent native exception receipt.
        backend, _, _, _ = self.worker_context(native_error=RuntimeError('artificial native error'))
        out = self.root/'worker-native-error.json'
        code = worker.run(self.data/self.manifest['selected_pairs'][0]['filename'], self.manifest_path,
                          self.seal, out, 'depth_scale', 0)
        self.assertEqual(code, 1)
        record = json.loads(out.read_text())
        self.assertEqual(record['failure_kind'], 'estimator')
        self.assertGreater(record['timings']['call_seconds'], 0)
        backend.estimate_monodepth_relative_pose_trace.assert_called_once()

    def test_worker_serialized_budget_prevents_oversized_file_write(self):
        backend, _, _, _ = self.worker_context()
        out = self.root/'bounded-worker.json'
        code = worker.run(self.data/self.manifest['selected_pairs'][0]['filename'], self.manifest_path,
                          self.seal, out, 'depth_scale', 0, trace_byte_budget=1)
        self.assertEqual(code, 1)
        record = json.loads(out.read_text())
        self.assertEqual(record['failure_kind'], 'trace')
        self.assertGreater(record['trace_required_bytes'], 1)
        self.assertFalse(out.with_suffix('.trace.json').exists())
        backend.estimate_monodepth_relative_pose_trace.assert_called_once()

    def test_trace_requires_pipeline_boundaries_and_closed_lo_sequence(self):
        t = self.native_trace()
        stages = [{'kind': kind, 'lo_id': 0, 'finite': True, 'stage': 'terminal_lo'}
                  for kind in ('lo_begin', 'initial', 'attempt', 'exit', 'lo_end')]
        t['events'][1:1] = stages; t['event_count'] = len(t['events'])
        self.assertEqual(c.validate_trace(t), 7)
        for index in (0, 2, 4, 5, 6):
            bad = copy.deepcopy(t); bad['events'].pop(index); bad['event_count'] -= 1
            with self.subTest(index=index), self.assertRaises(ValueError): c.validate_trace(bad)
        bad = copy.deepcopy(t); bad['events'][3]['stage'] = 'proposal_lo'
        with self.assertRaises(ValueError): c.validate_trace(bad)


if __name__ == '__main__': unittest.main()
