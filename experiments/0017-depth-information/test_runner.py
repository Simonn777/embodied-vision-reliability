"""Artificial supervision/worker tests; no scientific file or native calls."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest import mock

import common as c
import predict_pair as worker
import run_diagnostic as runner
import qualify_native as qualification


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.data=self.root/'data'; self.data.mkdir()
        self.manifest_path=self.root/'manifest.json'
        self.library={'version':'3.0.0','binary_sha256':'a'*64}
        self.libraries=self.root/'libraries.json'; self.libraries.write_text(json.dumps({'current':self.library}))
        self.sources={'artificial-source':'b'*64}; pairs=[]
        for scene in c.SCENES:
            filename=scene+'_roma_moge.h5'; path=self.data/filename
            path.write_bytes(b'ARTIFICIAL BYTES; NOT HDF5')
            for j in range(16):
                pairs.append({'pair_index':len(pairs),'scene':scene,'key':'fake-%02d'%j,'filename':filename,
                    'dataset_sha256':c.sha256(path),'dataset_bytes':path.stat().st_size})
        self.manifest={'selected_pairs':pairs}; self.manifest_path.write_text(json.dumps(self.manifest))
        self.mh=c.sha256(self.manifest_path)
        self.stack=contextlib.ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(c,'load_manifest',return_value=self.manifest))
        self.stack.enter_context(mock.patch.object(c,'source_hashes',return_value=self.sources))
        self.stack.enter_context(mock.patch.object(c,'LIBRARIES',self.libraries))

    def raw(self,index=0,arm='e_norm'):
        row=self.manifest['selected_pairs'][index]
        record=runner.identity(row,arm,self.mh,self.sources,self.library)
        mask=c.np.ones(8,dtype=bool)
        estimate={'R':c.np.eye(3).tolist(),'t':[1.,0.,0.]}
        geometry={**estimate,'scale':1.,'shift1':0.,'shift2':0.} if arm.startswith('depth_') else None
        record.update(status='completed',phase='complete',estimate=estimate,geometry=geometry,
            input_arrays={name:{'shape':[8,2] if name.startswith('x') else [8], 'dtype':'float64','sha256':'c'*64}
                          for name in ('x1','x2','depth1','depth2')},
            calibration={'artificial':True},input_diagnostics={'count':8},
            mask_length=8,kept_count=8,native_inliers_packbits_big_hex='ff',
            mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
            returned_ransac_stats={'iterations':100},final_scores={'pixel_msac':.5},
            depth_diagnostics={'all':{'denominator':8},'returned_mask':{'denominator':8}} if geometry else None,
            timings={'input_seconds':.1,'call_seconds':.1,'score_seconds':.1,'charged_total_seconds':.5})
        return record

    @staticmethod
    def process(code=0,termination=None):
        return {'exit_code':code,'termination':termination,'wall_seconds':1.,'sampled_peak_rss_kib':1024,'watchdog_error':None}

    def fake_child(self,command,log_path,remaining,environment):
        get=lambda flag:command[command.index(flag)+1]
        arm,index=get('--arm'),int(get('--pair-index'))
        self.calls.append((index,arm))
        self.assertTrue(all(environment[k]=='1' for k in runner.THREAD_VARIABLES))
        Path(log_path).write_bytes(b'artificial child')
        record=self.raw(index,arm)
        if self.mutate: self.mutate(record)
        c.write_new(get('--output'),record)
        return self.process(1 if record['status']=='error' else 0)

    def run_fake(self,mutate=None):
        self.calls=[]; self.mutate=mutate; out=self.root/'predictions'
        with mock.patch.object(runner,'run_child',side_effect=self.fake_child), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code=runner.run(self.data,self.manifest_path,out,self.root/'runtime',self.root/'dependencies')
        return code,out

    def test_full_320_seal_rotation_and_no_dataset_decoder(self):
        with mock.patch.dict('sys.modules',{'h5py':types.SimpleNamespace(File=mock.Mock(side_effect=AssertionError('No HDF5 allowed'))) }):
            code,out=self.run_fake()
            self.assertEqual(code,0)
            _,seal,records=runner.verify_seal(self.manifest_path,out/'seal.json')
        self.assertEqual(len(records),320); self.assertEqual(seal['status'],'completed')
        self.assertEqual(self.calls[:4],[(0,a) for a in c.ARMS])
        self.assertEqual(self.calls[4:8],[(1,a) for a in c.ARMS[1:]+c.ARMS[:1]])
        self.assertEqual(records[0]['timings']['charged_total_seconds'],1.)

    def test_estimator_exception_is_preserved_as_coverage_failure(self):
        def mutate(record):
            if record['pair_index']==0 and record['arm']=='e_norm':
                record.update(status='error',phase='error',failed_phase='estimator',failure_kind='estimator',error='Artificial native error')
        code,out=self.run_fake(mutate)
        self.assertEqual(code,0)
        _,_,records=runner.verify_seal(self.manifest_path,out/'seal.json')
        self.assertEqual(records[0]['status'],'estimator_error')

    def test_binding_input_and_output_failures_block_all_gt_eligibility(self):
        for failure in ('binding','input','output'):
            raw=self.raw(); raw.update(status='error',phase='error',failure_kind=failure)
            with self.subTest(failure=failure),self.assertRaises(ValueError):
                runner.wrap_record(raw,runner.identity(self.manifest['selected_pairs'][0],'e_norm',self.mh,self.sources,self.library),self.process(1))
        def mutate(record): record.update(status='error',phase='error',failure_kind='input')
        code,out=self.run_fake(mutate)
        self.assertEqual(code,1); self.assertEqual(len(self.calls),1)
        self.assertEqual(json.loads((out/'seal.json').read_text())['status'],'ineligible')
        with self.assertRaises(ValueError): runner.verify_seal(self.manifest_path,out/'seal.json')

    def test_timeout_and_crash_only_eligible_in_estimator_phase(self):
        expected=runner.identity(self.manifest['selected_pairs'][0],'e_norm',self.mh,self.sources,self.library)
        raw=self.raw(); raw.update(status='started',phase='estimator',timings={'input_seconds':.1})
        for process,status in ((self.process(-9,'pair_timeout'),'timeout'),(self.process(-11),'estimator_error')):
            wrapped=runner.wrap_record(raw,expected,process)
            self.assertEqual(wrapped['status'],status)
            self.assertIsNone(wrapped['timings']['call_seconds'])
            self.assertIsNone(wrapped['timings']['score_seconds'])
        for phase in ('binding','inputs','output'):
            with self.subTest(phase=phase),self.assertRaises(ValueError):
                runner.wrap_record({**raw,'phase':phase},expected,self.process(-9,'pair_timeout'))
        for reason in ('total_timeout','rss_limit','watchdog_error'):
            with self.subTest(reason=reason),self.assertRaises(ValueError):
                runner.wrap_record(raw,expected,self.process(-9,reason))

    def test_settings_library_and_source_mismatch_rejected(self):
        expected=runner.identity(self.manifest['selected_pairs'][0],'e_norm',self.mh,self.sources,self.library)
        for field in ('options','library','source_sha256','dataset_sha256_verified'):
            raw=self.raw(); raw[field]='changed'
            with self.subTest(field=field),self.assertRaises(ValueError):
                runner.wrap_record(raw,expected,self.process())

    def test_existing_output_directory_refused_before_any_binding(self):
        out=self.root/'already'; out.mkdir()
        with mock.patch.object(c,'load_manifest',side_effect=AssertionError('Must not read inputs')),self.assertRaises(FileExistsError):
            runner.run(self.data,self.manifest_path,out,self.root,self.root)

    def test_input_mismatch_between_arms_prevents_completed_seal(self):
        def mutate(record):
            if record['arm']=='e_ts': record['input_arrays']['x1']['sha256']='d'*64
        code,out=self.run_fake(mutate)
        self.assertEqual(code,1)
        self.assertEqual(json.loads((out/'seal.json').read_text())['status'],'ineligible')

    def test_modified_file_and_unsafe_path_fail_verification(self):
        code,out=self.run_fake(); self.assertEqual(code,0)
        seal_path=out/'seal.json'; seal=json.loads(seal_path.read_text())
        row=seal['rows'][0]; path=out/row['path']; original=path.read_text()
        path.write_text(original+' ')
        with self.assertRaises(ValueError): runner.verify_seal(self.manifest_path,seal_path)
        path.write_text(original)
        process=json.loads((out/row['process_path']).read_text())
        log_path=out/process['log_path']; log_original=log_path.read_bytes(); log_path.write_bytes(b'changed log')
        with self.assertRaises(ValueError): runner.verify_seal(self.manifest_path,seal_path)
        log_path.write_bytes(log_original)
        seal['rows'][0]['path']='../outside.json'; seal_path.write_text(json.dumps(seal))
        with self.assertRaises(ValueError): runner.verify_seal(self.manifest_path,seal_path)

    def test_watchdog_exception_kills_and_waits(self):
        child=mock.Mock(pid=123,poll=mock.Mock(return_value=None),wait=mock.Mock(return_value=-9))
        with mock.patch.object(runner.subprocess,'Popen',return_value=child),mock.patch.object(runner.subprocess,'run',side_effect=subprocess.TimeoutExpired('ps',2)):
            result=runner.run_child(['fake'],self.root/'watchdog.log',1800,{})
        child.kill.assert_called_once(); child.wait.assert_called_once_with(timeout=5)
        self.assertEqual(result['termination'],'watchdog_error')

    def test_native_dispatch_never_receives_reference_pose(self):
        pair,_,_=qualification.fixture()
        fake=types.SimpleNamespace(estimate_relative_pose=mock.Mock(return_value=('pose',{})),
            estimate_monodepth_relative_pose=mock.Mock(return_value=('geometry',{})))
        for arm in c.ARMS: worker.invoke(fake,pair,arm)
        self.assertEqual(fake.estimate_relative_pose.call_count,2)
        self.assertEqual(fake.estimate_monodepth_relative_pose.call_count,2)
        for call in fake.estimate_relative_pose.call_args_list:
            self.assertEqual(len(call.args),5)
            self.assertEqual(call.kwargs,{'initial_pose':None})
        for call in fake.estimate_monodepth_relative_pose.call_args_list:
            self.assertEqual(len(call.args),7)
            self.assertEqual(call.kwargs,{'initial_pose':None})


if __name__=='__main__': unittest.main()
