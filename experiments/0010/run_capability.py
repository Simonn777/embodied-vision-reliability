#!/usr/bin/env python3
"""Unfrozen offline capability prototype. No acquisition; CLI is future-use only."""
import argparse, ast, hashlib, importlib, importlib.metadata, json, math, os, re, resource, stat, sys, time, zipfile
from pathlib import Path, PurePosixPath

SCHEMA = '0010-clip-capability-v1'
CLIP = 'd05afc436d78f1c48dc0dbf8e5980a9d471f35f6'
STATA = '2635f9355f9389756064becb5d9ed54c2756809c'
DATASET = 'f9fa29ffb3980d9959c3d177dd2bc7c71ffeddcc'
CLASSES = ['AnnualCrop','Forest','HerbaceousVegetation','Highway','Industrial','Pasture','PermanentCrop','Residential','River','SeaLake']
TEMPLATE = 'a centered satellite photo of {}.'
MODULES = ['torch','numpy','torchvision','PIL','ftfy','regex','packaging','tqdm']
# (bytes, SHA256 when previously verified, pinned Git blob SHA1 when available)
ASSETS = {
 'source/clip/__init__.py': (20,'d7930eb290ebed432116795c91473200d098451a94c0aedd4ae0f934f97955e8','dcc5619538c0f7c782508bdbd9587259d805e0d9'),
 'source/clip/clip.py': (9644,'9540f200fbf8145479fa655382a56dab048d238cc698b9cbd8df3b6d86d3f1b6','398a6282cc243e449430d91c2562491fc799929d'),
 'source/clip/model.py': (17374,'9902cbe5ee90a1da2aa3e6f043e8a23dc1f8831193b963785c9af03d5c7bef2c','232b7792eb97440642547bd462cf128df9243933'),
 'source/clip/simple_tokenizer.py': (4632,'d1b09f10ed3d1e2c343619cb98cbfacb92363dd8607526a43aa7fe2d60d15bad','0a66286b7d5019c6e221932a813768038f839c91'),
 'source/LICENSE': (1064,'987e63b32f6c89ff5160e429458a872ff048e6860b590a3912e938f9da8f14db','4e97f0b45803b7c04ae89548934af4f257a97501'),
 'source/clip/bpe_simple_vocab_16e6.txt.gz': (1356917,None,'7b5088a527f720063f044eb928eee315f63b2fc0'),
 'weights/ViT-B-32.pt': (353976522,'40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af',None),
 'data/EuroSAT.zip': (94280567,'8ebea626349354c5328b142b96d0430e647051f26efc2dc974c843f25ecf70bd',None),
 'metadata/eurosat-spatial-train.txt': (316710,None,'4c08f4eeac54040a0fcd3788296f6e0dae782641'),
 'metadata/eurosat-spatial-val.txt': (99786,None,'c3179b28d8fc0e2b5fea3fc7191f7eefb367f6ca'),
 'metadata/eurosat-spatial-test.txt': (105934,None,'6fc8dfdf0c8ee837993d8b395c5cbc9896ac1ed7'),
 'metadata/stata-eurosat.py': (1686,'b6b8d878d54cefdae77e9185767a3f5ea9a0235cdc4cdd47b5507a78ab04ee23','e4165bc45b875c573c6234c75b280c1c550ecbb7'),
}

class Invalid(ValueError): pass

def require(ok, message):
    if not ok: raise Invalid(message)

def safe_file(root, relative):
    p = PurePosixPath(relative)
    require(isinstance(relative,str) and not p.is_absolute() and '\\' not in relative and relative == p.as_posix() and all(x not in ('.','..') for x in p.parts), 'unsafe relative path')
    path = root
    for part in p.parts:
        path = path / part
        require(not path.is_symlink(), 'symlink in asset path')
    require(path.is_file() and stat.S_ISREG(path.stat().st_mode), 'missing regular file')
    return path

def identity(path):
    size = path.stat().st_size; sha = hashlib.sha256(); git = hashlib.sha1(b'blob '+str(size).encode()+b'\0')
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20), b''): sha.update(block); git.update(block)
    return {'bytes':size,'sha256':sha.hexdigest(),'git_blob_sha1':git.hexdigest()}

def write_json(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); os.replace(temporary,path)

def validate_config(cfg):
    require(set(cfg)=={'schema','model','source_commit','stata_commit','dataset_commit','assets','selection_sha256','runtime_versions','class_order','template','cpu_ids'}, 'config fields')
    require((cfg['schema'],cfg['model'],cfg['source_commit'],cfg['stata_commit'],cfg['dataset_commit'])==(SCHEMA,'ViT-B/32',CLIP,STATA,DATASET), 'fixed identity mismatch')
    require(cfg['class_order']==CLASSES and cfg['template']==TEMPLATE, 'fixed prompts mismatch')
    require(re.fullmatch('[0-9a-f]{64}',cfg['selection_sha256']) is not None, 'selection hash')
    require(set(cfg['runtime_versions'])==set(MODULES) and all(isinstance(x,str) and x for x in cfg['runtime_versions'].values()), 'runtime versions missing')
    ids=cfg['cpu_ids']; require(isinstance(ids,list) and len(ids)==4 and all(type(x)is int and x>=0 for x in ids) and ids==sorted(set(ids)), 'four CPU ids required')
    require(set(cfg['assets'])==set(ASSETS), 'asset set mismatch')
    for name,(size,sha,git) in ASSETS.items():
        spec=cfg['assets'][name]
        require(set(spec)=={'bytes','sha256','git_blob_sha1'} and type(spec['bytes'])is int and spec['bytes']==size and spec['git_blob_sha1']==git, 'asset metadata mismatch')
        require(spec['sha256']==sha if sha else (spec['sha256'] is None or isinstance(spec['sha256'],str) and re.fullmatch('[0-9a-f]{64}',spec['sha256'])), 'asset SHA mismatch')

def verify_assets(root,cfg):
    result={}
    for name,spec in cfg['assets'].items():
        path=safe_file(root,name); require(path.stat().st_size==spec['bytes'],'asset size mismatch'); got=identity(path)
        require(all(value is None or got[key]==value for key,value in spec.items()), 'asset content identity mismatch')
        result[name]=got
    allowed={name.removeprefix('source/') for name in ASSETS if name.startswith('source/')}
    for item in (root/'source').rglob('*'):
        require(not item.is_symlink(), 'source tree symlink')
        if item.is_file(): require(item.relative_to(root/'source').as_posix() in allowed, 'extra source file')
        elif item.is_dir(): require(item.relative_to(root/'source').as_posix()=='clip','extra source directory')
        else: raise Invalid('nonregular source entry')
    return result

def selected_names(root, selection):
    require(set(selection)=={'names'}, 'selection fields')
    groups=[]
    for split in ('train','val','test'):
        names=(root/f'metadata/eurosat-spatial-{split}.txt').read_text().splitlines()
        require(len(names)==len(set(names)) and all(re.fullmatch(r'[A-Za-z]+_[0-9]+\.jpg',x) for x in names), 'invalid split names')
        groups.append(set(names))
    require(all(not groups[i]&groups[j] for i in range(3) for j in range(i)), 'split overlap')
    expected=sorted(groups[0],key=lambda n:(hashlib.sha256(('0010-clip-capability-v1|'+n).encode()).digest(),n))[:8]
    require(len(expected)==8 and selection['names']==expected,'fixed train-name selection mismatch')
    return expected

def prompts(root):
    values={}
    for node in ast.parse((root/'metadata/stata-eurosat.py').read_text()).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('template','NEW_CNAMES'):
            require(node.targets[0].id not in values,'duplicate prompt constant'); values[node.targets[0].id]=ast.literal_eval(node.value)
    require(values.get('template')==[TEMPLATE] and list(values.get('NEW_CNAMES',{}))==CLASSES, 'StatA prompt constants mismatch')
    return [TEMPLATE.format(values['NEW_CNAMES'][name]) for name in CLASSES]

def zip_members(archive,names):
    infos=archive.infolist(); require(len(infos)<=30020,'archive entry count'); found={}; seen=set(); total=0
    for info in infos:
        p=PurePosixPath(info.filename); mode=info.external_attr>>16
        require(info.filename not in seen and not p.is_absolute() and '\\' not in info.filename and '..' not in p.parts and '\0' not in info.filename,'unsafe/duplicate ZIP path'); seen.add(info.filename)
        require(not info.flag_bits&1 and info.compress_type in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED),'encrypted/unsupported ZIP')
        require(stat.S_IFMT(mode) in (0,stat.S_IFREG,stat.S_IFDIR) and not stat.S_ISLNK(mode),'special ZIP member')
        if info.is_dir(): continue
        total+=info.file_size; require(info.file_size<=1<<20 and total<=256<<20,'ZIP expanded-size bound')
        if p.name in names:
            require(p.name not in found,'ambiguous selected basename'); found[p.name]=info
    require(set(found)==set(names),'missing selected member')
    return [found[n] for n in names]

def runtime_evidence(modules):
    result={}
    for name,mod in modules.items():
        distribution='Pillow' if name=='PIL' else name
        result[name]={'module_version':str(getattr(mod,'__version__','')),
                      'distribution':distribution,
                      'distribution_version':importlib.metadata.version(distribution),
                      'file':mod.__file__}
    return result


def check_runtime_versions(result,expected):
    require(set(result)==set(expected) and all(result[n]['distribution_version']==expected[n] for n in expected),'runtime distribution version mismatch')


def cgroup_memory_limit():
    path=Path('/sys/fs/cgroup/memory.max'); raw=path.read_text().strip()
    require(re.fullmatch(r'[0-9]+',raw) is not None,'finite cgroup memory.max required')
    value=int(raw); require(value==8*1024**3,'cgroup memory.max must be 8 GiB')
    return {'path':str(path),'raw':raw,'bytes':value}


def learned_logit_scale(model):
    value=model.logit_scale.detach().cpu()
    require(tuple(value.shape)==(),'logit_scale must be scalar')
    raw=float(value.item()); exponential=float(value.exp().item())
    require(math.isfinite(raw) and math.isfinite(exponential) and exponential>0,'invalid learned logit_scale')
    return {'raw':raw,'exp':exponential}


def run(assetdir, config, config_sha256, selection, output):
    root=Path(assetdir).resolve(); out=Path(output).resolve()
    require(root.is_dir() and out!=root and root not in out.parents and out not in root.parents,'output/cache overlap')
    out.mkdir(parents=True,exist_ok=False)
    receipt={'schema':SCHEMA,'status':'started','image_attempted':0,'image_completed':0,'text_attempted':0,'text_completed':0,'timings':{},'failure':None}
    start=time.perf_counter(); stage='marker'; np=None; images=[]; texts=None
    def save():
        receipt['elapsed_seconds']=time.perf_counter()-start; write_json(out/'receipt.json',receipt)
    def elapsed(label,begin): receipt['timings'][label]=time.perf_counter()-begin
    def save_partial():
        t=time.perf_counter(); path=out/'partial-features.npz'; temporary=out/'partial-features.npz.tmp'
        with temporary.open('wb') as f:
            np.savez(f,image_raw=np.concatenate(images),text_raw=texts if texts is not None else np.empty((0,512),dtype=np.float32))
        os.replace(temporary,path); receipt['partial_features']=identity(path)
        receipt['timings']['partial_saves_seconds']=receipt['timings'].get('partial_saves_seconds',0)+time.perf_counter()-t
    try:
        with (root/'0010-probe-started.json').open('x') as f: json.dump({'config_sha256':config_sha256,'output':str(out)},f)
        save(); stage='identities'; t=time.perf_counter()
        require(not Path(config).is_symlink() and not Path(selection).is_symlink(),'config/selection symlink')
        receipt['config']=identity(Path(config)); require(receipt['config']['sha256']==config_sha256,'config SHA mismatch')
        cfg=json.loads(Path(config).read_text()); validate_config(cfg)
        receipt['selection']=identity(Path(selection)); require(receipt['selection']['sha256']==cfg['selection_sha256'],'selection SHA mismatch')
        receipt['assets']=verify_assets(root,cfg); names=selected_names(root,json.loads(Path(selection).read_text())); text=prompts(root)
        receipt.update(names=names,prompts=text,source=identity(Path(__file__))); elapsed(stage,t); save()
        stage='runtime_import'; t=time.perf_counter()
        require(sys.platform.startswith('linux') and sorted(os.sched_getaffinity(0))==cfg['cpu_ids'],'supervised CPU affinity required')
        require(resource.getrlimit(resource.RLIMIT_AS)==(8*1024**3,8*1024**3),'supervised 8 GiB limit required')
        receipt['cgroup_memory_max']=cgroup_memory_limit(); save()
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='4'
        os.environ['CUDA_VISIBLE_DEVICES']=''; sys.dont_write_bytecode=True
        modules={name:importlib.import_module(name) for name in MODULES}
        receipt['runtime']=runtime_evidence(modules)
        check_runtime_versions(receipt['runtime'],cfg['runtime_versions'])
        torch=modules['torch']; np=modules['numpy']; require(torch.version.cuda is None,'CPU-only PyTorch required')
        torch.set_default_device('cpu'); torch.set_default_dtype(torch.float32); torch.set_num_threads(4); torch.set_num_interop_threads(4)
        require(torch.get_num_threads()==4 and torch.get_num_interop_threads()==4,'torch thread mismatch')
        require(not any(k=='clip' or k.startswith('clip.') for k in sys.modules),'CLIP preimported')
        sys.path.insert(0,str(root/'source')); clip=importlib.import_module('clip'); api=importlib.import_module('clip.clip')
        for name,mod in tuple(sys.modules.items()):
            if name=='clip' or name.startswith('clip.'):
                require(Path(mod.__file__).resolve().is_relative_to(root/'source/clip'),'CLIP import escaped verified source')
        elapsed(stage,t); save(); stage='load'; t=time.perf_counter()
        jit=torch.jit.load(str(root/'weights/ViT-B-32.pt'),map_location='cpu')
        model=importlib.import_module('clip.model').build_model(dict(jit.state_dict())).float().eval(); del jit
        require(model.visual.input_resolution==224 and all(p.device.type=='cpu' and p.dtype==torch.float32 for p in model.parameters()),'native model device/dtype/resolution')
        receipt['learned_logit_scale']=learned_logit_scale(model)
        transform=api._transform(model.visual.input_resolution); elapsed(stage,t); save()
        with zipfile.ZipFile(root/'data/EuroSAT.zip') as archive, torch.inference_mode():
            members=zip_members(archive,names); receipt['members']=[]
            from PIL import Image
            for index,member in enumerate(members):
                stage=f'image_{index}_preprocess'; t=time.perf_counter()
                raw=archive.read(member); receipt['members'].append({'name':member.filename,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'crc32':member.CRC})
                import io
                with Image.open(io.BytesIO(raw)) as image:
                    require(image.format=='JPEG' and image.size==(64,64) and image.mode=='RGB','selected image must be 64x64 RGB JPEG')
                    tensor=transform(image).unsqueeze(0)
                require(tuple(tensor.shape)==(1,3,224,224) and tensor.device.type=='cpu','preprocess shape/device'); elapsed(stage,t)
                stage=f'image_{index}_encode'; receipt['image_attempted']+=1; save(); t=time.perf_counter()
                feature=model.encode_image(tensor); elapsed(stage,t); receipt['image_completed']+=1
                value=feature.detach().cpu().numpy(); require(value.dtype==np.float32 and value.shape==(1,512),'image feature shape/dtype invalid')
                images.append(value.copy()); save_partial(); require(np.isfinite(value).all(),'image feature nonfinite')
                norm=np.linalg.norm(value,axis=1); require(np.isfinite(norm).all() and (norm>0).all(),'image feature invalid norm'); save()
            stage='tokenize'; t=time.perf_counter(); tokens=clip.tokenize(text,context_length=77,truncate=False); elapsed(stage,t)
            require(tuple(tokens.shape)==(10,77) and tokens.device.type=='cpu','token shape/device')
            receipt['token_ids']=tokens.tolist(); stage='text_encode'; receipt['text_attempted']+=1; save(); t=time.perf_counter()
            feature=model.encode_text(tokens); elapsed(stage,t); receipt['text_completed']+=1
            texts=feature.detach().cpu().numpy().copy(); require(texts.dtype==np.float32 and texts.shape==(10,512),'text feature shape/dtype invalid')
            save_partial(); require(np.isfinite(texts).all(),'text feature nonfinite')
            norm=np.linalg.norm(texts,axis=1); require(np.isfinite(norm).all() and (norm>0).all(),'text feature invalid norm')
        stage='normalize_save'; t=time.perf_counter(); arrays={'image_raw':np.concatenate(images),'text_raw':texts}
        for key in ('image','text'):
            norm=np.linalg.norm(arrays[key+'_raw'],axis=1,keepdims=True); require(np.isfinite(norm).all() and (norm>0).all(),'invalid feature norm')
            arrays[key+'_normalized']=arrays[key+'_raw']/norm; require(np.isfinite(arrays[key+'_normalized']).all(),'normalized feature invalid')
        receipt['array_hashes']={key:{'shape':list(a.shape),'dtype':str(a.dtype),'sha256_f32le':hashlib.sha256(a.astype('<f4',copy=False).tobytes(order='C')).hexdigest()} for key,a in arrays.items()}
        with (out/'features.npz').open('xb') as f: np.savez(f,**arrays)
        receipt['features']=identity(out/'features.npz'); elapsed(stage,t)
        require((receipt['image_attempted'],receipt['image_completed'],receipt['text_attempted'],receipt['text_completed'])==(8,8,1,1),'call counts')
        receipt['status']='completed'
    except BaseException as exc:
        receipt['status']='failed'; receipt['failure']={'stage':stage,'type':type(exc).__name__,'reason':str(exc) if isinstance(exc,Invalid) else None}
        raise
    finally:
        receipt['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if sys.platform.startswith('linux') else None; save()
    return receipt

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('assetdir','config','config-sha256','selection','output'): parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    run(args.assetdir,args.config,args.config_sha256,args.selection,args.output)
