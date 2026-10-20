"""Artificial file/process barrier tests; no physical cohort or native solver."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import pipeline


class PipelineTests(unittest.TestCase):
    def test_json_rejects_duplicates_and_nonfinite(self):
        for data in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'):
            with self.assertRaises(ValueError):
                pipeline.decode(data)

    def test_source_requires_frozen_head_and_exact_current_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            names = ('pipeline.py', 'fixtures.py', 'study_core.py', 'tangent_solver.py',
                     'controls.py', 'evaluate.py', 'protocol.md')
            for name in names:
                (root/name).write_bytes(b'invented source')
            commit = 'a'*40
            def git(args, **kwargs):
                return commit+'\n' if args[-2:] == ['rev-parse', 'HEAD'] else b'invented source'
            with mock.patch.object(pipeline.subprocess, 'check_output', side_effect=git):
                self.assertEqual(set(pipeline.verify_source(root, commit, root)), set(names))
                (root/'controls.py').write_bytes(b'modified source')
                with self.assertRaisesRegex(ValueError, 'differs from frozen'):
                    pipeline.verify_source(root, commit, root)
                with self.assertRaisesRegex(ValueError, 'HEAD differs'):
                    pipeline.verify_source(root, 'b'*40, root)

    def test_input_complete_order_and_allowlist(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'inputs.jsonl'
            rows = [dict(row_id=i, observation={}) for i in range(138)]
            path.write_bytes(b''.join(pipeline.canonical(row)+b'\n' for row in rows))
            self.assertEqual(len(pipeline.load_inputs(path)[1]), 138)
            rows[37]['truth'] = {'R': 'forbidden'}
            path.write_bytes(b''.join(pipeline.canonical(row)+b'\n' for row in rows))
            with self.assertRaises(ValueError):
                pipeline.load_inputs(path)
            path.write_bytes(b'{}\n')
            with self.assertRaisesRegex(ValueError, '138-row'):
                pipeline.load_inputs(path)

    def test_native_hash_guard_precedes_import(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'poselib').mkdir()
            (root/'poselib/_core.cpython-39-darwin.so').write_bytes(b'not native code')
            with mock.patch.object(pipeline.importlib, 'import_module') as imported:
                with self.assertRaisesRegex(ValueError, 'before import'):
                    pipeline.load_native(root)
                imported.assert_not_called()

    def _prediction_setup(self, root):
        generation = root/'generated'; generation.mkdir()
        inputs = b''.join(pipeline.canonical(dict(row_id=i, observation={'invented':i}))+b'\n'
                          for i in range(138))
        (generation/'inputs.jsonl').write_bytes(inputs)
        pipeline.save_new(generation/'generation-job.json', {'invented': True})
        manifest = dict(schema='evr0027.generation.v1', status='completed', row_count=138,
            expected_commit='a'*40, source_sha256={'invented.py':'b'*64},
            generation_job_sha256=pipeline.sha((generation/'generation-job.json').read_bytes()),
            inputs_sha256=pipeline.sha(inputs), truth_sha256='c'*64)
        pipeline.save_new(generation/'generation.json', manifest)
        # No truth file is created: the prediction call must never open it.
        binary = root/'fake-native'; binary.write_bytes(b'fake binary')
        return generation, binary

    def test_prediction_never_opens_truth_and_seals_complete_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); generation, binary = self._prediction_setup(root)
            fake = types.ModuleType('study_core')
            seen = []
            fake.validate_observation = lambda observation: None
            def process(observation, native, progress):
                seen.append(observation['invented'])
                progress({'status':'started'})
                result = {'status':'completed', 'invented':observation['invented']}
                progress(result)
                return result
            fake.process = process
            with mock.patch.dict(sys.modules, study_core=fake):
                sys.modules.pop('fixtures', None)
                with mock.patch.object(pipeline, 'verify_source', return_value={'invented.py':'b'*64}), \
                     mock.patch.object(pipeline, 'verify_current'), \
                     mock.patch.object(pipeline, 'load_native', return_value=(object(),binary)), \
                     mock.patch.object(pipeline, 'NATIVE_SHA256', pipeline.sha(binary.read_bytes())):
                    seal = pipeline.predict(root, 'a'*40, generation, root, root/'prediction')
            self.assertEqual(seen, list(range(138)))
            self.assertEqual(seal['row_count'], 138)
            self.assertEqual(seal['truth_sha256'], 'c'*64)
            self.assertEqual(len((root/'prediction/predictions.jsonl').read_bytes().splitlines()),138)

    def test_incomplete_prediction_has_no_seal_and_retains_started_row(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); generation, binary = self._prediction_setup(root)
            fake = types.ModuleType('study_core')
            fake.validate_observation = lambda observation: None
            def process(observation, native, progress):
                progress({'status':'started', 'native_called':False})
                raise RuntimeError('invented native abort')
            fake.process = process
            with mock.patch.dict(sys.modules, study_core=fake):
                sys.modules.pop('fixtures', None)
                with mock.patch.object(pipeline, 'verify_source', return_value={'invented.py':'b'*64}), \
                     mock.patch.object(pipeline, 'load_native', return_value=(object(),binary)):
                    with self.assertRaisesRegex(RuntimeError, 'invented native abort'):
                        pipeline.predict(root, 'a'*40, generation, root, root/'prediction')
            self.assertFalse((root/'prediction/seal.json').exists())
            self.assertEqual(json.loads((root/'prediction/active-row.json').read_text())['row_id'],0)
            self.assertEqual((root/'prediction/predictions.jsonl').read_bytes(),b'')


if __name__ == '__main__':
    unittest.main()
