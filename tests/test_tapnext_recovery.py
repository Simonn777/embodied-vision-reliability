"""Recovery guards on own tiny fixtures only; no ML imports or real HTTP."""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('recovery_under_test',ROOT/'scripts/prepare_tapnext_recovery.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
spec=importlib.util.spec_from_file_location('original_preparer_fixture_helper',ROOT/'scripts/prepare_tapnext_assets.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


class Response(io.BytesIO):
    def __init__(self,payload,length=None):
        super().__init__(payload);self.status=200;self.headers={} if length is None else {'Content-Length':str(length)}
    def geturl(self):return 'https://example.invalid/fixture'


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
        self.failed=self.base/'failed';self.failed.mkdir()
        self.root=self.base/'recovered';self.root.mkdir()
        self.payload=b'own synthetic checkpoint bytes'
        self.asset={'id':'checkpoint-256','kind':'checkpoint','relative_path':'weights/tapnextpp_ckpt.pt',
                    'url':'https://example.invalid/fixture','bytes':len(self.payload),'sha256':hashlib.sha256(self.payload).hexdigest()}
        self.transfer={'per_asset_timeout_seconds':840,'read_timeout_seconds':60,'chunk_bytes':4}
        self.ledger={'config_sha256':'c'*64,'source_sha256':'s'*64,'reservations':{'earlier':11},
                     'conservative_cumulative_ceiling_bytes':100,'planned_ceiling_bytes':101+len(self.payload),
                     'payload_bytes_observed':10,'new_payload_bytes_observed':0,'limit_bytes':1000}
        self.calls=0
    def tearDown(self):self.tmp.cleanup()
    def opener(self,payload=None,length=None):
        def call(url,timeout):
            self.calls+=1;self.assertEqual(url,self.asset['url']);self.assertEqual(timeout,60)
            return Response(self.payload if payload is None else payload,length)
        return call
    def acquire(self,opener):return m.checkpoint_once(self.asset,self.root,self.ledger,h,self.transfer,opener)
    def partial(self):return self.root/'weights/tapnextpp_ckpt.pt.partial'
    def record(self):return json.loads((self.root/'asset-records/checkpoint-256.json').read_text())
    def test_checkpoint_success_inherits_budget_once(self):
        record=self.acquire(self.opener(length=len(self.payload)))
        self.assertEqual(self.calls,1);self.assertEqual(record['status'],'downloaded_verified')
        self.assertFalse(self.partial().exists());self.assertFalse(record['partial_exists'])
        self.assertEqual(self.ledger['payload_bytes_observed'],10+len(self.payload))
        self.assertEqual(self.ledger['new_payload_bytes_observed'],len(self.payload))
        self.assertEqual(self.ledger['reservations']['earlier'],11)
        self.assertEqual(self.ledger['conservative_cumulative_ceiling_bytes'],101+len(self.payload))
        with self.assertRaises(ValueError):self.acquire(self.opener())
        self.assertEqual(self.calls,1)
    def test_payload_is_closed_before_link(self):
        real_open=Path.open;real_link=os.link;opened=[]
        def opening(path,*args,**kwargs):
            stream=real_open(path,*args,**kwargs)
            if path==self.partial() and args and args[0]=='xb':opened.append(stream)
            return stream
        def linking(src,dst):
            self.assertEqual(len(opened),1);self.assertTrue(opened[0].closed)
            return real_link(src,dst)
        with patch.object(Path,'open',opening),patch.object(m.os,'link',linking):
            self.acquire(self.opener())
    def test_hash_failure_keeps_partial_and_forbids_retry(self):
        with self.assertRaises(ValueError):self.acquire(self.opener(b'x'*len(self.payload)))
        self.assertEqual(self.partial().read_bytes(),b'x'*len(self.payload))
        self.assertEqual(self.record()['status'],'failed')
        self.assertEqual(self.record()['error']['reason'],'Closed checkpoint partial failed length/hash verification')
        with self.assertRaises(ValueError):self.acquire(self.opener())
        self.assertEqual(self.calls,1)
    def test_truncated_bytes_preserved(self):
        with self.assertRaises(ValueError):self.acquire(self.opener(self.payload[:-2]))
        self.assertEqual(self.partial().read_bytes(),self.payload[:-2])
        self.assertEqual(self.ledger['new_payload_bytes_observed'],len(self.payload)-2)
    def test_only_one_oversize_sentinel_consumed(self):
        with self.assertRaises(ValueError):self.acquire(self.opener(self.payload+b'extra'))
        self.assertEqual(self.partial().read_bytes(),self.payload+b'e')
    def test_length_mismatch_reads_no_body(self):
        with self.assertRaises(ValueError):self.acquire(self.opener(length=1000))
        self.assertEqual(self.record()['payload_bytes_observed'],0)
    def test_budget_rejected_before_get(self):
        self.ledger['limit_bytes']=100
        with self.assertRaises(ValueError):self.acquire(self.opener())
        self.assertEqual(self.calls,0)
    def test_prior_reservation_rejected_before_get(self):
        self.ledger['reservations']['checkpoint-256']=1
        with self.assertRaises(ValueError):self.acquire(self.opener())
        self.assertEqual(self.calls,0)
    def test_existing_final_is_never_replaced(self):
        final=self.root/self.asset['relative_path'];final.parent.mkdir();final.write_bytes(b'preserved')
        with self.assertRaises(ValueError):self.acquire(self.opener())
        self.assertEqual(final.read_bytes(),b'preserved');self.assertEqual(self.calls,0)
    def test_destination_race_preserves_both_files(self):
        def race(url,timeout):
            (self.root/self.asset['relative_path']).write_bytes(b'concurrent file')
            return Response(self.payload)
        with self.assertRaises(FileExistsError):self.acquire(race)
        self.assertEqual((self.root/self.asset['relative_path']).read_bytes(),b'concurrent file')
        self.assertEqual(self.partial().read_bytes(),self.payload)
    def test_empty_partial_reappearing_after_unlink_fails_closed(self):
        real_unlink=Path.unlink
        def unlinking(path,*args,**kwargs):
            result=real_unlink(path,*args,**kwargs)
            if path==self.partial():path.write_bytes(b'')
            return result
        with patch.object(Path,'unlink',unlinking),self.assertRaises(ValueError):self.acquire(self.opener())
        self.assertTrue(self.record()['partial_exists']);self.assertEqual(self.record()['status'],'failed')
    def source_assets(self):
        names=['tapnet/tapnext/a.py','tapnet/tapnext/b.py','tapnet/tapnext/c.py',
               'tapnet/tapnextpp/votsp2026/d.py','tapnet/tapnextpp/votsp2026/e.py']
        assets=[]
        for index,name in enumerate(names):
            path=self.failed/'source'/name;path.parent.mkdir(parents=True,exist_ok=True)
            payload=f'# own source fixture {index}\n'.encode();path.write_bytes(payload)
            assets.append({'id':f'source-{index}','kind':'source','relative_path':'source/'+name,'bytes':len(payload),'sha256':m.sha(path)})
        return assets
    def test_clean_copy_preserves_original_empty_partials_and_exact_bytes(self):
        assets=self.source_assets()
        for asset in assets[:3]:(self.failed/(asset['relative_path']+'.partial')).write_bytes(b'')
        before=m.inventory(self.failed);files=m.copy_sources(self.failed,self.root/'source',assets)
        self.assertEqual(before,m.inventory(self.failed));self.assertEqual(len(files),5)
        self.assertEqual(m.inventory(self.root/'source'),files)
        with patch.object(sys,'path',[str(self.root/'source')]):
            h.verify_source_tree(self.root/'source',{k:v['sha256'] for k,v in files.items()})
        self.assertNotIn('tapnet',sys.modules)
    def test_copy_refuses_existing_source_tree_without_overwrite(self):
        assets=self.source_assets();(self.root/'source').mkdir();(self.root/'source/keep').write_text('keep')
        with self.assertRaises(FileExistsError):m.copy_sources(self.failed,self.root/'source',assets)
        self.assertEqual((self.root/'source/keep').read_text(),'keep')
    def test_copy_rejects_changed_source(self):
        assets=self.source_assets();(self.failed/assets[0]['relative_path']).write_text('changed')
        with self.assertRaises(ValueError):m.copy_sources(self.failed,self.root/'source',assets)
    def test_inventory_rejects_symlink_even_when_bytes_match(self):
        p=self.failed/'target';p.write_bytes(b'own');(self.failed/'link').symlink_to(p)
        with self.assertRaises(ValueError):m.inventory(self.failed)
    def test_manifest_detects_changed_and_extra_files(self):
        target=self.root/'overlay';target.mkdir();(target/'own.py').write_text('# own')
        manifest=self.root/'manifest.json';h.save(manifest,{'config_sha256':'c','source_sha256':'s','files':m.inventory(target)},exclusive=True)
        reference={'relative_path':'manifest.json','sha256':m.sha(manifest)}
        m.verify_manifest(self.root,reference,target,'c','s')
        (target/'extra').write_bytes(b'')
        with self.assertRaises(ValueError):m.verify_manifest(self.root,reference,target,'c','s')
    def test_bound_reference_rejects_wrong_hash(self):
        path=self.root/'fixture.json';path.write_text('{}')
        with self.assertRaises(ValueError):m.bound_json(self.root,{'path':'fixture.json','sha256':'0'*64})
    def test_unsafe_or_symlink_relative_paths_rejected(self):
        for name in ('../escape','/absolute'):
            with self.assertRaises(ValueError):m.local(self.root,name)
        (self.root/'link').symlink_to(self.failed,target_is_directory=True)
        with self.assertRaises(ValueError):m.local(self.root,'link/file')


class RecoveryStageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
        self.repo=self.base/'repo';self.repo.mkdir()
        self.old=self.base/'old';self.old.mkdir()
        self.failed=self.base/'failed';self.failed.mkdir()
        self.root=self.base/'recovery'
        self.original=copy.deepcopy(json.loads((ROOT/'configs/0005-native-capability.json').read_text())['preparation'])
        self.original['runtime']['old_init_sha256']={}
        for name in ('runtime-fixed/torch/__init__.py','runtime-fixed/numpy/__init__.py'):
            p=self.old/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# own runtime fixture\n')
            self.original['runtime']['old_init_sha256'][name]=m.sha(p)
        self.old_markers={}
        for i in range(3):
            p=self.old/f'old-marker-{i}.json';p.write_text('{}');self.old_markers[p.name]=m.sha(p)
        self.payload=b'own miniature checkpoint'
        for asset in self.original['assets']:
            payload=self.payload if asset['kind']=='checkpoint' else ('# own '+asset['id']+'\n').encode()
            asset['bytes']=len(payload);asset['sha256']=hashlib.sha256(payload).hexdigest()
            asset['url']='https://example.invalid/'+asset['id']
            if asset['kind']!='checkpoint':
                p=self.failed/asset['relative_path'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(payload)
        for asset in self.original['assets'][:3]:(self.failed/(asset['relative_path']+'.partial')).write_bytes(b'')
        overlay=self.failed/'overlay';(overlay/'own.dist-info').mkdir(parents=True)
        (overlay/'own.dist-info/RECORD').write_text('own fixture');(overlay/'own.py').write_text('# own')
        (self.failed/'preparation.lock').write_bytes(b'')
        helper_path=self.repo/'frozen-helper.py';helper_path.write_bytes((ROOT/'scripts/prepare_tapnext_assets.py').read_bytes())
        self.original_path=self.repo/'original.json';h.save(self.original_path,{'preparation':self.original},exclusive=True)
        self.original_sha=m.sha(self.original_path);self.helper_sha=m.sha(helper_path)
        child={'command':['synthetic','not executed'],'exit_code':0,'raw_wait_status':0,'elapsed_seconds':0.01,
               'started_utc':'synthetic','finished_utc':'synthetic','peak_rss_kib':0}
        for stream in ('stdout','stderr'):
            p=self.failed/f'install.{stream}.log';p.write_bytes(b'');child[stream+'_sha256']=m.sha(p)
        h.save(self.failed/'install.json',child,exclusive=True)
        records=[]
        for asset in self.original['assets'][:-1]:
            record={'asset':asset,'status':'downloaded_verified','payload_bytes_observed':asset['bytes']}
            records.append(record);path=self.failed/'asset-records'/(asset['id']+'.json');path.parent.mkdir(exist_ok=True)
            h.save(path,record,exclusive=True)
        base=self.original['budget']['historical_ceiling_bytes']+self.original['budget']['metadata_allowance_bytes']
        ledger={'config_sha256':self.original_sha,'source_sha256':self.helper_sha,'base_ceiling_bytes':base,
                'planned_ceiling_bytes':h.validate_config(self.original),'limit_bytes':10*1024**3,
                'reservations':{a['id']:a['bytes']+1 for a in self.original['assets'][:-1]},
                'payload_bytes_observed':sum(a['bytes'] for a in self.original['assets'][:-1]),
                'conservative_cumulative_ceiling_bytes':base+sum(a['bytes']+1 for a in self.original['assets'][:-1])}
        for name in ('acquisition-ledger.json','acquisition-ledger-dependencies.json'):h.save(self.failed/name,ledger,exclusive=True)
        prior={'status':'failed','stage':'dependencies','config_sha256':self.original_sha,'source_sha256':self.helper_sha,
               'assets':records,'installation':child,'ledger_snapshot':{'relative_path':'acquisition-ledger-dependencies.json','sha256':m.sha(self.failed/'acquisition-ledger.json')}}
        h.save(self.failed/'preparation-dependencies.json',prior,exclusive=True)
        self.review={'import_check':'not_started','checkpoint':'not_started','native':'not_started','overlay_files':2,
                     'old_markers_preserved':self.old_markers,'actual_new_asset_payload_bytes':ledger['payload_bytes_observed'],
                     'conservative_cumulative_ceiling_bytes':ledger['conservative_cumulative_ceiling_bytes']}
        evidence={'failure_receipt':prior,'cache_inventory':m.inventory(self.failed),
                  'overlay_inventory':{'status':'fixture inventory only','files':m.inventory(overlay)},'failure_review':self.review}
        refs={}
        for name,value in evidence.items():
            p=self.repo/(name+'.json');h.save(p,value,exclusive=True);refs[name]={'path':p.name,'sha256':m.sha(p)}
        self.config={'recovery':{'schema_version':1,'original_config':{'path':'original.json','sha256':self.original_sha},
                                'frozen_preparer':{'path':'frozen-helper.py','sha256':self.helper_sha},'evidence':refs}}
        self.cp=self.repo/'recovery.json';h.save(self.cp,self.config,exclusive=True);self.config_sha=m.sha(self.cp)
        self.context=m.load_context(self.repo,self.config);self.calls=0;self.import_calls=0
    def tearDown(self):self.tmp.cleanup()
    def fake_import(self,command,prefix,env,timeout):
        self.import_calls+=1
        self.assertEqual(timeout,120);self.assertNotIn(str(self.failed/'source'),env['PYTHONPATH'].split(':'))
        self.assertEqual(env['PYTHONDONTWRITEBYTECODE'],'1')
        prefix.with_suffix('.stdout.log').write_text(json.dumps({'status':'passed','no_model_constructed_or_forward_called':True}))
        prefix.with_suffix('.stderr.log').write_bytes(b'')
        result={'command':command,'exit_code':0,'raw_wait_status':0,'elapsed_seconds':0.01,'started_utc':'fixture','finished_utc':'fixture'}
        for stream in ('stdout','stderr'):result[stream+'_sha256']=m.sha(prefix.with_suffix('.'+stream+'.log'))
        h.save(prefix.with_suffix('.json'),result,exclusive=True)
        return result
    def fake_http(self,url,timeout):
        self.calls+=1
        return Response(self.payload,len(self.payload))
    def run_stage(self,stage,importer=None):
        helper=self.context[2];before=list(sys.path)
        try:
            with patch.object(m,'load_context',return_value=self.context),patch.object(m.sys,'platform','linux'),\
                 patch.object(m.platform,'python_version',return_value=self.original['runtime']['python']),\
                 patch.object(m.os,'sched_getaffinity',create=True,return_value=set(range(9))),\
                 patch.object(m.os,'sched_setaffinity',create=True),patch.object(m.resource,'setrlimit'),\
                 patch.object(m.subprocess,'check_output',return_value='a'*40+'\n'),\
                 patch.object(helper,'child',side_effect=importer or self.fake_import),\
                 patch.object(helper,'default_open',side_effect=self.fake_http):
                with patch('sys.stdout',new_callable=io.StringIO):
                    return m.run_stage(self.repo,self.old,self.failed,self.root,self.cp,self.config_sha,stage)
        finally:sys.path[:]=before
    def test_both_stages_reuse_all_prior_bytes_without_install_or_repeat_import(self):
        before=m.inventory(self.failed);old_before=m.inventory(self.old)
        self.assertEqual(self.run_stage('dependencies'),0)
        self.assertEqual(self.import_calls,1);self.assertEqual(self.calls,0)
        self.assertEqual(self.run_stage('checkpoint'),0)
        self.assertEqual(self.import_calls,1);self.assertEqual(self.calls,1)
        self.assertEqual(m.inventory(self.failed),before);self.assertEqual(m.inventory(self.old),old_before)
        deps=json.loads((self.root/'recovery-dependencies.json').read_text())
        checkpoint=json.loads((self.root/'recovery-checkpoint.json').read_text())
        self.assertTrue(deps['preserved_failed_cache']);self.assertEqual(deps['new_network_payload_bytes'],0)
        self.assertEqual(checkpoint['dependency_receipt_sha256'],m.sha(self.root/'recovery-dependencies.json'))
        self.assertEqual(checkpoint['new_network_payload_bytes'],len(self.payload))
        self.assertEqual(m.sha(self.root/'acquisition-ledger.json'),checkpoint['ledger_snapshot']['sha256'])
        with self.assertRaises(ValueError):self.run_stage('checkpoint')
        self.assertEqual(self.calls,1)
    def test_changed_failed_cache_stops_before_recovery_marker(self):
        (self.failed/'unbound').write_bytes(b'')
        with self.assertRaises(ValueError):self.run_stage('dependencies')
        self.assertFalse(self.root.exists());self.assertEqual(self.calls,0);self.assertEqual(self.import_calls,0)
    def test_import_failure_is_preserved_and_cannot_unlock_checkpoint(self):
        def bad(*args):raise m.RecoveryFailure('Synthetic import failure')
        self.assertEqual(self.run_stage('dependencies',bad),1)
        receipt=json.loads((self.root/'recovery-dependencies.json').read_text())
        self.assertEqual(receipt['error']['reason'],'Synthetic import failure')
        self.assertTrue((self.root/'markers/dependencies.started.json').exists())
        self.assertEqual(self.run_stage('checkpoint'),1);self.assertEqual(self.calls,0)
    def test_coherently_tampered_ledger_rejected_before_get(self):
        self.assertEqual(self.run_stage('dependencies'),0)
        ledger=json.loads((self.root/'acquisition-ledger.json').read_text());ledger['payload_bytes_observed']=0
        for name in ('acquisition-ledger.json','acquisition-ledger-dependencies.json'):h.save(self.root/name,ledger)
        receipt=json.loads((self.root/'recovery-dependencies.json').read_text())
        receipt['ledger_snapshot']['sha256']=m.sha(self.root/'acquisition-ledger.json');h.save(self.root/'recovery-dependencies.json',receipt)
        self.assertEqual(self.run_stage('checkpoint'),1);self.assertEqual(self.calls,0)
        failure=json.loads((self.root/'recovery-checkpoint.json').read_text())
        self.assertEqual(failure['error']['reason'],'Recovery inherited ledger fields differ')
    def test_changed_dependency_marker_rejected_before_get(self):
        self.assertEqual(self.run_stage('dependencies'),0)
        (self.root/'markers/dependencies.started.json').write_text('{}')
        self.assertEqual(self.run_stage('checkpoint'),1);self.assertEqual(self.calls,0)
    def test_changed_import_log_rejected_before_get(self):
        self.assertEqual(self.run_stage('dependencies'),0)
        (self.root/'import-check.stdout.log').write_text('{}')
        self.assertEqual(self.run_stage('checkpoint'),1);self.assertEqual(self.calls,0)


if __name__=='__main__':unittest.main(verbosity=2)
