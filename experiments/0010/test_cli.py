"""Offline source-portability/layout fixtures; no runtime/model/network calls."""
import hashlib, json, shutil, sys, tempfile, types, unittest
from pathlib import Path
from unittest.mock import patch
import cli


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class IntegrationTests(unittest.TestCase):
    def test_checked_helpers_are_portable_and_exact(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); here=root/'experiments/0010'; here.mkdir(parents=True); (root/'scripts').mkdir()
            for name in ('acquire.py','run_capability.py'): shutil.copyfile(cli.HERE/name,here/name)
            shutil.copyfile(cli.ROOT/'scripts/supervise_0009.py',root/'scripts/supervise_0009.py')
            with patch.object(cli,'HERE',here),patch.object(cli,'ROOT',root):
                for name in cli.PINS:
                    mod=cli.load(name)
                    self.assertEqual(sha(mod.__file__),cli.PINS[name])
            self.assertNotIn('torch',sys.modules)

    def test_changed_helper_rejected_before_execute(self):
        with tempfile.TemporaryDirectory() as t:
            here=Path(t); (here/'acquire.py').write_text('raise AssertionError("must not execute")')
            with patch.object(cli,'HERE',here),self.assertRaisesRegex(ValueError,'hash mismatch'):
                cli.load('acquire.py')

    def test_fixed_manifests_match_layout_and_conservative_caps(self):
        deps=cli.fixed_json('dependencies.json'); assets=cli.fixed_json('downloads.json')
        self.assertEqual(deps['body_cap_bytes']+assets['body_cap_bytes'],531635154)
        self.assertEqual(sum(a['bytes'] for a in deps['assets']),1234470)
        self.assertEqual(sum(a['bytes'] for a in assets['assets']),449615692)
        self.assertEqual({a['id'] for a in deps['assets']},{'ftfy','regex','tqdm','wcwidth'})
        cfg=json.loads((cli.HERE/'config.json').read_text())
        cap=cli.load('run_capability.py'); cap.validate_config(cfg)
        layout=cli.fixed_json('layout.json'); self.assertEqual(set(layout),set(cap.ASSETS))
        cli.bundle_inputs(cfg,layout,cap)
        self.assertEqual(len([x for x in layout.values() if x['kind']=='bundle']),8)
        self.assertEqual(sha(cli.HERE/'selection.json'),cfg['selection_sha256'])
        cli.load('acquire.py').validate(deps); cli.load('acquire.py').validate(assets)

    def tiny_layout(self, root):
        here=root/'package'; bundle=here/'bundle'; bundle.mkdir(parents=True)
        cache=root/'cache'; cache.mkdir(); (bundle/'literal.txt').write_bytes(b'artificial source')
        (cache/'payload').write_bytes(b'fake payload; not a model')
        cap=cli.load('run_capability.py')
        layout={'metadata/literal.txt':{'kind':'bundle','path':'literal.txt'},
                'weights/fake.bin':{'kind':'cache','path':'payload'}}
        cfg={'assets':{k:cap.identity((bundle if v['kind']=='bundle' else cache)/v['path']) for k,v in layout.items()}}
        return here,cache,layout,cfg,cap

    def test_materialize_is_portable_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); here,cache,layout,cfg,cap=self.tiny_layout(root); dest=root/'assets'
            with patch.object(cli,'HERE',here):
                got=cli.materialize(cache,dest,cfg,layout,cap)
                self.assertEqual(got,cfg['assets']); before=(dest/'layout-receipt.json').read_bytes()
                with self.assertRaises(FileExistsError): cli.materialize(cache,dest,cfg,layout,cap)
                self.assertEqual((dest/'layout-receipt.json').read_bytes(),before)
                self.assertEqual((dest/'weights/fake.bin').read_bytes(),(cache/'payload').read_bytes())

    def test_materialize_rejects_hash_symlink_and_overlap(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); here,cache,layout,cfg,cap=self.tiny_layout(root); dest=root/'assets'
            with patch.object(cli,'HERE',here):
                with self.assertRaisesRegex(ValueError,'overlapping'): cli.materialize(cache,cache/'nested',cfg,layout,cap)
                (cache/'payload').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'identity mismatch'): cli.materialize(cache,dest,cfg,layout,cap)
                self.assertFalse(dest.exists())
                (cache/'payload').unlink(); (cache/'payload').symlink_to(here/'bundle/literal.txt')
                with self.assertRaises(cap.Invalid): cli.materialize(cache,dest,cfg,layout,cap)

    def test_existing_asset_directory_refused_before_download(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); dest=root/'assets'; dest.mkdir(); cap=cli.load('run_capability.py')
            cfg=json.loads((cli.HERE/'config.json').read_text())
            def safe_load(name):
                if name=='run_capability.py': return cap
                raise AssertionError('download helper must not be reached')
            with patch.object(cli,'load',side_effect=safe_load),patch.object(cli.sys,'platform','linux'), \
                 patch.object(cli.os,'sched_getaffinity',return_value={0,1,2,3},create=True), \
                 patch.object(cli.resource,'getrlimit',return_value=(8*1024**3,8*1024**3)), \
                 patch.object(cap,'cgroup_memory_limit',return_value={}),self.assertRaises(FileExistsError):
                cli.child_acquire('assets',root/'cache',dest,cfg)

    def test_known_config_hash_is_required_before_runtime(self):
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            cli.checked_config(cli.HERE/'config.json','0'*64)

    def test_public_stages_use_only_frozen_external_supervisor(self):
        original_load=cli.load; calls=[]
        fake=types.SimpleNamespace(supervise=lambda command,output,**kw:calls.append((command,kw)) or {'supervisor_exit_code':0})
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); cfg=cli.HERE/'config.json'
            with patch.object(cli,'load',side_effect=lambda name:fake if name=='supervise_0009.py' else original_load(name)):
                common=['--config',str(cfg),'--config-sha256',sha(cfg),'--python','existing-python']
                self.assertEqual(cli.main(['assets',*common,'--cache',str(root/'cache'),'--assetdir',str(root/'assets'),'--supervision',str(root/'supervision')]),0)
                self.assertEqual(cli.main(['probe',*common,'--assetdir',str(root/'assets'),'--output',str(root/'output'),'--supervision',str(root/'probe-supervision')]),0)
            self.assertEqual([x[1]['seconds'] for x in calls],[900,300])
            self.assertEqual(calls[0][0][2],'_assets')
            self.assertEqual(Path(calls[1][0][1]).name,'run_capability.py')
            self.assertTrue(all(x[1]['cpu_ids']==[0,1,2,3] for x in calls))


if __name__=='__main__': unittest.main(verbosity=2)
