"""Stub HTTP and tiny synthetic bytes only. Never invokes main/pip/cloud/model."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import sys
from unittest.mock import patch
import unittest
import os
import signal
import threading

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prepare_assets_draft',ROOT/'scripts/prepare_tapnext_assets.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Response(io.BytesIO):
    def __init__(self,payload,length=None):
        super().__init__(payload);self.status=200;self.headers={} if length is None else {'Content-Length':str(length)}
    def geturl(self):return 'https://example.invalid/test'

class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.calls=0;self.payload=b'own fixture bytes only'
        self.asset={'id':'fixture','kind':'source','relative_path':'source/example.py','url':'https://example.invalid/test','bytes':len(self.payload),'sha256':hashlib.sha256(self.payload).hexdigest()}
        self.ledger={'base_ceiling_bytes':100,'limit_bytes':1000,'reservations':{},'payload_bytes_observed':0,'config_sha256':'c'*64,'source_sha256':'s'*64}
        self.lp=self.root/'ledger.json';m.save(self.lp,self.ledger,exclusive=True)
        self.transfer={'read_timeout_seconds':60,'per_asset_timeout_seconds':840,'chunk_bytes':4}
    def tearDown(self):self.tmp.cleanup()
    def fetch(self,payload=None,length=None):
        def opener(url,timeout):
            self.calls+=1;self.assertEqual(url,self.asset['url']);self.assertEqual(timeout,60)
            return Response(self.payload if payload is None else payload,length)
        return opener
    def run_asset(self,opener):return m.acquire(self.asset,self.root,self.ledger,self.lp,self.transfer,opener)
    def partial(self):return self.root/'source/example.py.partial'
    def record(self):return json.loads((self.root/'asset-records/fixture.json').read_text())
    def test_success_and_exact_budget_counter(self):
        r=self.run_asset(self.fetch(length=len(self.payload)))
        self.assertEqual(self.calls,1);self.assertEqual((self.root/'source/example.py').read_bytes(),self.payload)
        self.assertFalse(self.partial().exists());self.assertEqual(r['status'],'downloaded_verified')
        self.assertEqual(r['payload_bytes_observed'],len(self.payload));self.assertEqual(self.ledger['payload_bytes_observed'],len(self.payload))
        self.assertEqual(self.ledger['conservative_cumulative_ceiling_bytes'],100+len(self.payload)+1)
    def test_new_destination_race_never_overwrites_existing_file(self):
        def racing(url,timeout):
            self.calls+=1;(self.root/'source/example.py').write_bytes(b'concurrent preserved file')
            return Response(self.payload)
        with self.assertRaises(FileExistsError):self.run_asset(racing)
        self.assertEqual((self.root/'source/example.py').read_bytes(),b'concurrent preserved file')
        self.assertEqual(self.partial().read_bytes(),self.payload);self.assertEqual(self.calls,1)
    def test_hash_failure_preserves_all_bytes_and_forbids_second_get(self):
        payload=b'x'*len(self.payload)
        with self.assertRaises(m.PrepFailure):self.run_asset(self.fetch(payload))
        self.assertEqual(self.partial().read_bytes(),payload);self.assertEqual(self.record()['status'],'failed')
        with self.assertRaises(m.PrepFailure):self.run_asset(self.fetch())
        self.assertEqual(self.calls,1)
    def test_truncation_preserves_observed_bytes(self):
        with self.assertRaises(m.PrepFailure):self.run_asset(self.fetch(self.payload[:-3]))
        self.assertEqual(self.partial().read_bytes(),self.payload[:-3]);self.assertEqual(self.record()['payload_bytes_observed'],len(self.payload)-3)
    def test_oversize_reads_only_one_sentinel_byte(self):
        with self.assertRaises(m.PrepFailure):self.run_asset(self.fetch(self.payload+b'extra bytes'))
        self.assertEqual(self.partial().read_bytes(),self.payload+b'e');self.assertEqual(self.record()['payload_bytes_observed'],len(self.payload)+1)
    def test_existing_matching_final_reuses_without_get(self):
        p=self.root/'source/example.py';p.parent.mkdir();p.write_bytes(self.payload)
        r=self.run_asset(self.fetch());self.assertEqual(r['status'],'reused_verified');self.assertEqual(self.calls,0)
    def test_existing_bad_final_is_not_overwritten(self):
        p=self.root/'source/example.py';p.parent.mkdir();p.write_bytes(b'keep me')
        with self.assertRaises(m.PrepFailure):self.run_asset(self.fetch())
        self.assertEqual(p.read_bytes(),b'keep me');self.assertEqual(self.calls,0)
    def test_existing_partial_not_overwritten_or_retried(self):
        p=self.partial();p.parent.mkdir();p.write_bytes(b'partial keep')
        with self.assertRaises(m.PrepFailure):self.run_asset(self.fetch())
        self.assertEqual(p.read_bytes(),b'partial keep');self.assertEqual(self.calls,0)
    def test_content_length_mismatch_stops_before_payload(self):
        with self.assertRaises(m.PrepFailure):self.run_asset(self.fetch(length=123))
        self.assertEqual(self.record()['payload_bytes_observed'],0);self.assertEqual(self.partial().stat().st_size,0);self.assertEqual(self.calls,1)
    def test_budget_rejection_before_get(self):
        self.ledger['limit_bytes']=101
        with self.assertRaises(m.PrepFailure):self.run_asset(self.fetch())
        self.assertEqual(self.calls,0)
    def test_network_exception_is_one_failed_attempt(self):
        def broken(url,timeout):self.calls+=1;raise TimeoutError('own fixture timeout')
        with self.assertRaises(TimeoutError):self.run_asset(broken)
        self.assertEqual(self.record()['error']['type'],'TimeoutError');self.assertEqual(self.partial().stat().st_size,0)
        with self.assertRaises(m.PrepFailure):self.run_asset(broken)
        self.assertEqual(self.calls,1)
    def test_symlink_target_is_rejected(self):
        p=self.root/'source';p.mkdir();(p/'example.py').symlink_to(self.lp)
        with self.assertRaises(m.PrepFailure):self.run_asset(self.fetch())
        self.assertEqual(self.calls,0)
    def test_overlay_manifest_detects_changes_and_symlinks(self):
        overlay=self.root/'overlay';dist=overlay/'fixture.dist-info';dist.mkdir(parents=True)
        (dist/'RECORD').write_text('fixture');(overlay/'module.so').write_bytes(b'own fake binary')
        before=m.file_manifest(overlay);(overlay/'module.so').write_bytes(b'changed')
        self.assertNotEqual(before,m.file_manifest(overlay));(overlay/'escape').symlink_to(self.lp)
        with self.assertRaises(m.PrepFailure):m.file_manifest(overlay)
    def namespace_fixture(self):
        source=self.root/'frozen-source';files={}
        cfg=json.loads((ROOT/'configs/0005-native-capability.json').read_text())['preparation']
        for a in cfg['assets']:
            if a['kind']!='source':continue
            rel=a['relative_path'].removeprefix('source/');p=source/rel;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(b'# own synthetic source, never imported\n');files[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
        return source,files
    def test_exact_namespace_tree_passes_without_import(self):
        source,files=self.namespace_fixture()
        with patch.object(sys,'path',[str(source)]):m.verify_source_tree(source,files)
        self.assertNotIn('tapnet',sys.modules)
    def test_every_unbound_parent_initializer_is_rejected_without_execution(self):
        source,files=self.namespace_fixture()
        for rel in ['tapnet/__init__.py','tapnet/tapnext/__init__.py','tapnet/tapnextpp/__init__.py','tapnet/tapnextpp/votsp2026/__init__.py']:
            p=source/rel;p.write_text('raise RuntimeError("must never execute")')
            with patch.object(sys,'path',[str(source)]),self.assertRaises(m.PrepFailure):m.verify_source_tree(source,files)
            self.assertNotIn('tapnet',sys.modules);p.unlink()
    def test_later_installed_regular_package_cannot_override_namespace(self):
        source,files=self.namespace_fixture();other=self.root/'installed';(other/'tapnet').mkdir(parents=True)
        (other/'tapnet/__init__.py').write_text('raise RuntimeError("must never execute")')
        with patch.object(sys,'path',[str(source),str(other)]),self.assertRaises(m.PrepFailure):m.verify_source_tree(source,files)
        self.assertNotIn('tapnet',sys.modules)
    def test_extra_namespace_location_is_rejected(self):
        source,files=self.namespace_fixture();other=self.root/'extra';(other/'tapnet').mkdir(parents=True)
        with patch.object(sys,'path',[str(source),str(other)]),self.assertRaises(m.PrepFailure):m.verify_source_tree(source,files)
    def test_modified_source_is_rejected(self):
        source,files=self.namespace_fixture();(source/next(iter(files))).write_text('# changed')
        with patch.object(sys,'path',[str(source)]),self.assertRaises(m.PrepFailure):m.verify_source_tree(source,files)
    def test_source_symlink_is_rejected(self):
        source,files=self.namespace_fixture();p=source/next(iter(files));p.unlink();p.symlink_to(self.lp)
        with patch.object(sys,'path',[str(source)]),self.assertRaises(m.PrepFailure):m.verify_source_tree(source,files)
    def test_preimported_tapnet_is_rejected(self):
        source,files=self.namespace_fixture()
        with patch.object(sys,'path',[str(source)]),patch.dict(sys.modules,{'tapnet':object()}),self.assertRaises(m.PrepFailure):m.verify_source_tree(source,files)
    def test_child_success_records_actual_exit_and_logs(self):
        prefix=self.root/'child'
        record=m.child([sys.executable,'-c','print("own child fixture")'],prefix,dict(os.environ),3)
        self.assertEqual(record['exit_code'],0)
        self.assertEqual(record['raw_wait_status'],0)
        self.assertEqual(record['stdout_sha256'],m.sha(prefix.with_suffix('.stdout.log')))
    def test_child_nonzero_exit_rejected_and_retained(self):
        prefix=self.root/'bad-child'
        with self.assertRaises(m.PrepFailure):
            m.child([sys.executable,'-c','raise SystemExit(7)'],prefix,dict(os.environ),3)
        self.assertEqual(json.loads(prefix.with_suffix('.json').read_text())['exit_code'],7)
    def test_external_term_kills_child_group_and_retains_failure(self):
        prefix=self.root/'term-child'; prior=signal.getsignal(signal.SIGTERM)
        timer=threading.Timer(0.1,lambda:os.kill(os.getpid(),signal.SIGTERM))
        signal.signal(signal.SIGTERM,m.external_timeout)
        try:
            timer.start()
            with self.assertRaises(m.PrepFailure):
                m.child([sys.executable,'-c','import time; time.sleep(60)'],prefix,dict(os.environ),3)
        finally:
            timer.cancel();timer.join();signal.signal(signal.SIGTERM,prior)
        record=json.loads(prefix.with_suffix('.json').read_text())
        self.assertTrue(record['terminated_by_parent'])
        self.assertEqual(record['exit_code'],-signal.SIGKILL)
    def test_real_config_budget_and_oversize_proposal(self):
        cfg=json.loads((ROOT/'configs/0005-native-capability.json').read_text())['preparation']
        self.assertEqual(m.validate_config(cfg),7499066782)
        changed=copy.deepcopy(cfg);changed['assets'][-1]['bytes']=10*1024**3
        with self.assertRaises(m.PrepFailure):m.validate_config(changed)

if __name__=='__main__':
    unittest.main(verbosity=2)
