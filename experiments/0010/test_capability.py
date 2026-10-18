"""Artificial byte/path/metadata fixtures only; imports no ML/image runtime."""
import hashlib, io, json, math, tempfile, types, unittest, warnings, zipfile
from pathlib import Path
from unittest.mock import patch
import run_capability as cap


def sha(b): return hashlib.sha256(b).hexdigest()

def spec(b): return dict(bytes=len(b),sha256=sha(b),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest())

def config():
    return dict(schema=cap.SCHEMA,model='ViT-B/32',source_commit=cap.CLIP,stata_commit=cap.STATA,dataset_commit=cap.DATASET,
                assets={n:dict(zip(('bytes','sha256','git_blob_sha1'),s)) for n,s in cap.ASSETS.items()},
                selection_sha256='0'*64,runtime_versions={n:'UNSET' for n in cap.MODULES},class_order=list(cap.CLASSES),template=cap.TEMPLATE,cpu_ids=[0,1,2,3])

class GuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name).resolve()
    def tearDown(self): self.tmp.cleanup()
    def put(self,name,data):
        p=self.root/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(data); return p

    def test_identity_and_strict_config(self):
        p=self.put('binary',b'\x00synthetic\xff'); self.assertEqual(cap.identity(p),spec(p.read_bytes()))
        cfg=config(); cap.validate_config(cfg)
        for key,value in [('model','RN50'),('template','a photo of {}'),('cpu_ids',[0,1]),('class_order',cap.CLASSES[::-1])]:
            changed=config(); changed[key]=value
            with self.subTest(key=key), self.assertRaises(cap.Invalid): cap.validate_config(changed)
        cfg['assets']['source/clip/bpe_simple_vocab_16e6.txt.gz']['git_blob_sha1']='0'*40
        with self.assertRaises(cap.Invalid): cap.validate_config(cfg)

    def test_distribution_version_is_distinct_from_module_attribute(self):
        mods={'regex':types.SimpleNamespace(__version__='2.5.example',__file__='fake-regex.py'),
              'PIL':types.SimpleNamespace(__version__='13.0.0',__file__='fake-PIL.py')}
        versions={'regex':'2026.9.29','Pillow':'13.0.0'}
        with patch.object(cap.importlib.metadata,'version',side_effect=lambda name:versions[name]) as queried:
            result=cap.runtime_evidence(mods)
            cap.check_runtime_versions(result,{'regex':'2026.9.29','PIL':'13.0.0'})
        self.assertEqual(result['regex']['module_version'],'2.5.example')
        self.assertEqual(result['regex']['distribution_version'],'2026.9.29')
        self.assertEqual(result['PIL']['distribution'],'Pillow')
        self.assertEqual([x.args[0] for x in queried.call_args_list],['regex','Pillow'])

    def test_distribution_version_mismatch_rejects_matching_module_version(self):
        mods={'regex':types.SimpleNamespace(__version__='2026.9.29',__file__='fake.py')}
        with patch.object(cap.importlib.metadata,'version',return_value='wrong'):
            result=cap.runtime_evidence(mods)
        self.assertEqual(result['regex']['distribution_version'],'wrong')
        with self.assertRaises(cap.Invalid): cap.check_runtime_versions(result,{'regex':'2026.9.29'})
        with patch.object(cap.importlib.metadata,'version',side_effect=cap.importlib.metadata.PackageNotFoundError),self.assertRaises(cap.importlib.metadata.PackageNotFoundError):
            cap.runtime_evidence(mods)

    def test_cgroup_limit_records_exact_finite_bound(self):
        with patch.object(Path,'read_text',return_value='8589934592\n'):
            self.assertEqual(cap.cgroup_memory_limit(),
                             {'path':'/sys/fs/cgroup/memory.max','raw':'8589934592','bytes':8589934592})
        for raw in ('max','0','4294967296','17179869184','nan'):
            with self.subTest(raw=raw),patch.object(Path,'read_text',return_value=raw),self.assertRaises(cap.Invalid):
                cap.cgroup_memory_limit()
        with patch.object(Path,'read_text',side_effect=FileNotFoundError),self.assertRaises(FileNotFoundError):
            cap.cgroup_memory_limit()

    def test_logit_scale_scalar_finite_without_forward(self):
        class Scalar:
            shape=()
            def __init__(self,raw,exponential): self.raw,self.exponential=raw,exponential
            def detach(self): return self
            def cpu(self): return self
            def item(self): return self.raw
            def exp(self): return Scalar(self.exponential,self.exponential)
        model=types.SimpleNamespace(logit_scale=Scalar(2.,math.exp(2.)))
        self.assertEqual(cap.learned_logit_scale(model),{'raw':2.,'exp':math.exp(2.)})
        for raw,exp in ((float('nan'),1.),(2.,float('inf')),(2.,0.)):
            model.logit_scale=Scalar(raw,exp)
            with self.subTest(raw=raw,exp=exp),self.assertRaises(cap.Invalid): cap.learned_logit_scale(model)
        model.logit_scale=Scalar(2.,math.exp(2.)); model.logit_scale.shape=(1,)
        with self.assertRaises(cap.Invalid): cap.learned_logit_scale(model)

    def test_paths_and_symlinks(self):
        self.put('plain/ok',b'1')
        for name in ('../x','/tmp/x','plain/../plain/ok','plain\\ok','plain//ok'):
            with self.subTest(name=name),self.assertRaises(cap.Invalid): cap.safe_file(self.root,name)
        (self.root/'alias').symlink_to(self.root/'plain',target_is_directory=True)
        with self.assertRaises(cap.Invalid): cap.safe_file(self.root,'alias/ok')

    def test_every_asset_hash_and_exact_source_tree(self):
        fixture={'source/clip/__init__.py':b'pass\n','source/clip/bpe.txt.gz':b'FAKE vocabulary'}
        for name,b in fixture.items(): self.put(name,b)
        expectations={n:(len(b),sha(b),spec(b)['git_blob_sha1']) for n,b in fixture.items()}
        cfg={'assets':{n:spec(b) for n,b in fixture.items()}}
        with patch.object(cap,'ASSETS',expectations):
            got=cap.verify_assets(self.root,cfg); self.assertEqual(len(got),2)
            self.put('source/clip/bpe.txt.gz',b'FAKE vocabularY')
            with self.assertRaises(cap.Invalid): cap.verify_assets(self.root,cfg)
            self.put('source/clip/bpe.txt.gz',fixture['source/clip/bpe.txt.gz'])
            self.put('source/clip/unbound.py',b'pass')
            with self.assertRaises(cap.Invalid): cap.verify_assets(self.root,cfg)

    def test_selection_independent_of_list_order_and_heldout(self):
        names=[f'AnnualCrop_{i}.jpg' for i in range(12)]
        self.put('metadata/eurosat-spatial-train.txt',('\n'.join(names)+'\n').encode())
        self.put('metadata/eurosat-spatial-val.txt',b'Forest_90.jpg\n'); self.put('metadata/eurosat-spatial-test.txt',b'Forest_91.jpg\n')
        selected=sorted(names,key=lambda n:(hashlib.sha256(('0010-clip-capability-v1|'+n).encode()).digest(),n))[:8]
        self.assertEqual(cap.selected_names(self.root,{'names':selected}),selected)
        self.put('metadata/eurosat-spatial-train.txt',('\n'.join(reversed(names))+'\n').encode())
        self.assertEqual(cap.selected_names(self.root,{'names':selected}),selected)
        with self.assertRaises(cap.Invalid): cap.selected_names(self.root,{'names':selected[::-1]})
        self.put('metadata/eurosat-spatial-val.txt',b'AnnualCrop_0.jpg\n')
        with self.assertRaises(cap.Invalid): cap.selected_names(self.root,{'names':selected})

    def archive(self,files):
        raw=io.BytesIO()
        with warnings.catch_warnings(),zipfile.ZipFile(raw,'w') as z:
            warnings.simplefilter('ignore',UserWarning)
            for name,body in files: z.writestr(name,body)
        raw.seek(0); return zipfile.ZipFile(raw)

    def test_archive_metadata_no_decode(self):
        names=[f'Forest_{i}.jpg' for i in range(8)]
        entries=[('2750/Forest/'+n,b'not a real JPEG') for n in names]
        with self.archive(entries) as z:
            self.assertEqual([Path(x.filename).name for x in cap.zip_members(z,names)],names)
        for bad in ([('../escape',b'0')], [('other/'+names[0],b'0')], [entries[0]], [('oversize',b'0'*(2**20+1))]):
            with self.subTest(extra=bad[0][0]),self.archive(entries+bad) as z,self.assertRaises(cap.Invalid): cap.zip_members(z,names)
        with self.archive(entries[:-1]) as z,self.assertRaises(cap.Invalid): cap.zip_members(z,names)

    def test_prompt_ast_never_executes(self):
        body='template = '+repr([cap.TEMPLATE])+'\nNEW_CNAMES = '+repr({n:n+' literal' for n in cap.CLASSES})+'\nraise RuntimeError("must never execute")\n'
        self.put('metadata/stata-eurosat.py',body.encode())
        self.assertEqual(cap.prompts(self.root),[cap.TEMPLATE.format(n+' literal') for n in cap.CLASSES])
        self.put('metadata/stata-eurosat.py',(body+'template = []\n').encode())
        with self.assertRaises(cap.Invalid): cap.prompts(self.root)

    def test_bad_vocabulary_stops_before_any_runtime_import(self):
        assets=self.root/'assets'; b=b'fake compressed vocabulary'; path='source/clip/bpe_simple_vocab_16e6.txt.gz'
        self.put('assets/'+path,b[:-1]+b'X'); vocab=spec(b)
        expected={path:(vocab['bytes'],vocab['sha256'],vocab['git_blob_sha1'])}
        selection=self.put('names.json',b'{"names": []}')
        with patch.object(cap,'ASSETS',expected):
            cfg=config(); cfg['selection_sha256']=sha(selection.read_bytes())
            cfgfile=self.put('cfg.json',json.dumps(cfg).encode())
            with patch.object(cap.importlib,'import_module',side_effect=AssertionError('must not import')) as imported:
                with self.assertRaises(cap.Invalid): cap.run(assets,cfgfile,sha(cfgfile.read_bytes()),selection,self.root/'bad-bpe-out')
                imported.assert_not_called()
        receipt=json.loads((self.root/'bad-bpe-out/receipt.json').read_text())
        self.assertEqual(receipt['failure']['reason'],'asset content identity mismatch')
        self.assertEqual(receipt['image_attempted'],0)

    def test_failure_marker_counts_no_import_or_retry(self):
        assets=self.root/'assets'; assets.mkdir(); cfg=self.put('cfg.json',b'{}'); selection=self.put('names.json',b'{}')
        out=self.root/'out'
        with patch.object(cap.importlib,'import_module',side_effect=AssertionError('no runtime imports')) as imported:
            with self.assertRaises(cap.Invalid): cap.run(assets,cfg,'0'*64,selection,out)
            imported.assert_not_called()
        saved=json.loads((out/'receipt.json').read_text()); self.assertEqual(saved['status'],'failed')
        self.assertEqual([saved[k] for k in ('image_attempted','image_completed','text_attempted','text_completed')],[0,0,0,0])
        marker=(assets/'0010-probe-started.json').read_bytes()
        with self.assertRaises(FileExistsError): cap.run(assets,cfg,'0'*64,selection,self.root/'out2')
        self.assertEqual((assets/'0010-probe-started.json').read_bytes(),marker)
        with self.assertRaises(FileExistsError): cap.run(assets,cfg,'0'*64,selection,out)

if __name__=='__main__': unittest.main()
