"""Offline, prospective bindings. No asset/model access occurs on import."""
import gc
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import resource
import sys
from types import ModuleType
import zipfile
from driver import Invalid, require, digest, selected_names

CAP_SHA = '532ef917a87e976dfd1bc0fd1858c16ea8a4b5bebaa9569add017e394f6a7ba6'
HEADROOM_SHA = '40dd054b74217dc43390cb3354c4064ead5215114a6db575ca06ff251cda7b47'
STATA_SHA = '15545b0fbe67655255d64a4343daeb43cac3bc7d649df984d2605a8795aee5be'
UPSTREAM_SHA = '9fcd2eb8a22cabc2c7eb43823625cb04557476c637a24c8a12eb3115ede2b962'
SUPERVISOR_SHA = '2d306a8813d36c207eb276a187a298864965bffbe79c1dd463d9482f3bf7d77f'


def verified_path(ref, base):
    require(set(ref) == {'path','sha256'}, 'input reference fields')
    path = Path(ref['path']); path = path if path.is_absolute() else base/path
    require(not any(p.is_symlink() for p in (path,*path.parents)) and path.is_file(), 'input missing or symlink')
    require(digest(path) == ref['sha256'], 'input hash mismatch')
    return path.resolve()


def module_at(path, expected):
    raw = Path(path).read_bytes(); require(hashlib.sha256(raw).hexdigest() == expected, 'module SHA mismatch')
    m = ModuleType('_0010_' + Path(path).stem); m.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), m.__dict__)
    return m


def positive(value):
    require(type(value) in (int,float) and math.isfinite(value) and value > 0, 'positive measurement required')
    return value


def timing_admission(probe, timing, execution, cfg):
    t = probe['timings']
    setup = sum(positive(t[k]) for k in ('identities','runtime_import','load'))
    # Include preprocessing, conservatively beyond the draft's forward-only term.
    images = sum(positive(t['image_'+str(i)+'_'+part]) for i in range(8) for part in ('preprocess','encode'))
    encoder = 1.5*(setup + 16*images)
    require(encoder <= 450, 'encoder projection exceeds allowance')
    require(timing['mode'] == 'timing' and timing['status'] == 'completed' and timing['resource_admitted'] is True, 'timing not admitted')
    require(timing['actual_torch_threads'] == [4,4] and timing['inputs_unchanged'] is True, 'timing thread/input contract')
    require(timing['inputs']['shape_features'] == [64,512] and timing['inputs']['shape_prototypes'] == [1,512,10], 'timing fixture shape')
    require(timing['runtime']['torch'] == cfg['runtime_versions']['torch'] and timing['runtime']['numpy'] == cfg['runtime_versions']['numpy'], 'timing runtime mismatch')
    attempts = timing['attempts']; require(len(attempts) == 3, 'timing attempt count')
    for a, phase, alpha in zip(attempts, ('T63','T641','T644'), (1,1,4)):
        require(a['phase'] == phase and a['alpha'] == alpha and a['status'] == 'completed' and
                a['native_solver_executions'] == 2 and a['native_solver_execution_count_unknown'] is False, 'timing attempt invalid')
    solver = 1.5*(128*positive(attempts[0]['seconds']) + 2*positive(attempts[1]['seconds']) + 2*positive(attempts[2]['seconds']))
    require(solver <= 360 and math.isclose(solver, timing['projected_seconds'], rel_tol=1e-12, abs_tol=1e-12), 'solver projection mismatch/excess')
    require(execution['supervisor_exit_code'] == execution['child_exit_code'] == 0 and execution['timed_out'] is False and
            execution['address_space_bytes'] == 8*1024**3 and execution['cpu_affinity'] == cfg['cpu_ids'] and
            execution['numeric_threads'] == 4 and positive(execution['wall_seconds']) <= 120, 'timing supervision failed')
    return dict(encoder_projection_seconds=encoder, solver_projection_seconds=solver,
                encoder_projection_includes_preprocess=True, projected_not_guaranteed=True)


class Native:
    def __init__(self, plan, base, assetdir, headroom):
        self.plan, self.base, self.root, self.h = plan, Path(base), Path(assetdir).resolve(), headroom
        self.archive = self.model = None; self.runtime_record = {}
    def admit(self):
        import numpy as np
        refs = self.plan['inputs']
        paths = {key: verified_path(value, self.base) for key,value in refs.items()}
        self.paths = paths
        for key, sha in [('capability_runner',CAP_SHA),('headroom',HEADROOM_SHA),('stata_cpu',STATA_SHA),('stata_upstream',UPSTREAM_SHA),('supervisor',SUPERVISOR_SHA)]:
            require(refs[key]['sha256'] == sha, 'reviewed source differs')
        self.cap = cap = module_at(paths['capability_runner'], CAP_SHA)
        self.cfg = cfg = json.loads(paths['capability_config'].read_text()); cap.validate_config(cfg)
        freeze = json.loads(paths['capability_freeze'].read_text())
        for name,key in [('experiments/0010/run_capability.py','capability_runner'),('experiments/0010/config.json','capability_config'),('experiments/0010/selection.json','capability_selection')]:
            require(freeze['inputs'][name]['sha256'] == refs[key]['sha256'], 'capability freeze mismatch')
        self.probe = probe = json.loads(paths['capability_receipt'].read_text())
        require(probe['status'] == 'completed' and [probe[k] for k in ('image_attempted','image_completed','text_attempted','text_completed')] == [8,8,1,1], 'capability incomplete')
        require(probe['source']['sha256'] == CAP_SHA and probe['config']['sha256'] == refs['capability_config']['sha256'] and probe['selection']['sha256'] == refs['capability_selection']['sha256'], 'probe source/config identity')
        require(probe['features']['sha256'] == refs['capability_features']['sha256'], 'probe feature-file binding')
        assets = cap.verify_assets(self.root, cfg); require(assets == probe['assets'], 'probe/current assets mismatch')
        excluded = cap.selected_names(self.root, json.loads(paths['capability_selection'].read_text()))
        require(probe['names'] == excluded and probe['prompts'] == cap.prompts(self.root), 'probe names/prompts mismatch')
        cap.check_runtime_versions(probe['runtime'], cfg['runtime_versions'])
        names = selected_names((self.root/'metadata/eurosat-spatial-train.txt').read_text().splitlines(), excluded)
        selection = json.loads(paths['selection'].read_text()); require(selection == {'names':names}, 'headroom selection mismatch')
        timing = json.loads(paths['timing_receipt'].read_text()); execution = json.loads(paths['timing_execution'].read_text())
        qualified = timing_admission(probe, timing, execution, cfg)
        solver_manifest = json.loads(paths['solver_manifest'].read_text())
        require(timing['manifest_sha256'] == refs['solver_manifest']['sha256'] and timing['source_files'] == solver_manifest['files'], 'timing source manifest')
        require(timing['source_files']['stata_cpu.py'] == STATA_SHA and timing['source_files']['upstream/StatA.py'] == UPSTREAM_SHA, 'timing solver identity')
        require(timing['adapter_execution']['original_sha256'] == STATA_SHA, 'timing adapter source')
        # Separately supplied actual functional receipt, not this driver's fake tests.
        functional = json.loads(paths['functional_receipt'].read_text())
        require(functional['mode'] == 'functional' and functional['status'] == 'completed' and functional['tests'] ==
                dict(run=20, failures=0, errors=0, skipped=0) and functional['actual_torch_threads'] == [4,4], 'functional admission incomplete')
        require(functional['manifest_sha256'] == timing['manifest_sha256'] and functional['source_files'] == timing['source_files'] and
                functional['adapter_execution'] == timing['adapter_execution'] and functional['runtime'] == timing['runtime'], 'functional/timing provenance differs')
        for prefix in ('timing','functional'):
            bundle = json.loads(paths[prefix+'_artifacts'].read_text())
            require(bundle['source_manifest_sha256'] == refs['solver_manifest']['sha256'], 'admission artifact source')
            for rel,key in [('child/receipt.json',prefix+'_receipt'),('supervision/execution.json',prefix+'_execution')]:
                require(bundle['files'][rel] == dict(bytes=paths[key].stat().st_size, sha256=refs[key]['sha256']), 'admission artifact binding')
            ex = json.loads(paths[prefix+'_execution'].read_text())
            require(ex['supervisor_exit_code'] == ex['child_exit_code'] == 0 and ex['timed_out'] is False and ex['wall_limit_seconds'] == 120 and
                    ex['address_space_bytes'] == 8*1024**3 and ex['cpu_affinity'] == cfg['cpu_ids'] and ex['numeric_threads'] == 4, 'admission external execution')
        with np.load(paths['capability_features'], allow_pickle=False) as z:
            require(set(z.files) == {'image_raw','image_normalized','text_raw','text_normalized'}, 'probe array keys')
            texts = {key:z[key] for key in ('text_raw','text_normalized')}  # No capability image vectors used.
        for key,a in texts.items():
            expected = probe['array_hashes'][key]
            require(a.dtype == np.float32 and a.shape == (10,512) and np.isfinite(a).all(), 'text array shape/dtype')
            require(expected == {'shape':[10,512], 'dtype':'float32', 'sha256_f32le':hashlib.sha256(a.astype('<f4',copy=False).tobytes()).hexdigest()}, 'text array binding')
        norm = np.linalg.norm(texts['text_raw'],axis=1,keepdims=True)
        require(np.isfinite(norm).all() and (norm>0).all() and np.array_equal(texts['text_raw']/norm,texts['text_normalized']), 'text normalization differs')
        pm = probe['members']; require(len(pm) == 8 and [Path(m['name']).name for m in pm] == excluded, 'capability member identity')
        self.archive = zipfile.ZipFile(self.root/'data/EuroSAT.zip'); self.members = cap.zip_members(self.archive,names)
        evidence = dict(inputs={k:v['sha256'] for k,v in refs.items()}, assets=assets, qualification=qualified,
                        prompts=probe['prompts'], token_ids=probe['token_ids'], class_order=cfg['class_order'])
        return dict(names=names, text_normalized=texts['text_normalized'], evidence=evidence,
                    members=[dict(name=m.filename,bytes=m.file_size,crc32=m.CRC) for m in self.members],
                    capability_member_sha256=[m['sha256'] for m in pm])
    def load_encoder(self):
        cap, cfg = self.cap, self.cfg
        require(sys.platform.startswith('linux') and sorted(os.sched_getaffinity(0)) == cfg['cpu_ids'] and
                resource.getrlimit(resource.RLIMIT_AS) == (8*1024**3,8*1024**3), 'external CPU/memory supervision required')
        cap.cgroup_memory_limit()
        require(not any(k == 'clip' or k.startswith('clip.') for k in sys.modules), 'CLIP preimported')
        modules = {name:importlib.import_module(name) for name in cap.MODULES}
        self.runtime_record = cap.runtime_evidence(modules); cap.check_runtime_versions(self.runtime_record,cfg['runtime_versions'])
        torch = self.torch = modules['torch']; require(torch.version.cuda is None, 'CPU-only Torch required')
        torch.set_default_device('cpu'); torch.set_default_dtype(torch.float32); torch.set_num_threads(4); torch.set_num_interop_threads(4)
        require([torch.get_num_threads(),torch.get_num_interop_threads()] == [4,4], 'Torch threads')
        sys.dont_write_bytecode = True; sys.path.insert(0,str(self.root/'source'))
        clip = importlib.import_module('clip'); api = importlib.import_module('clip.clip')
        for name,m in tuple(sys.modules.items()):
            if name == 'clip' or name.startswith('clip.'):
                require(Path(m.__file__).resolve().is_relative_to(self.root/'source/clip'), 'CLIP origin')
        tokens = clip.tokenize(self.probe['prompts'], context_length=77, truncate=False)
        require(tokens.tolist() == self.probe['token_ids'], 'token IDs changed')  # No text forward.
        jit = torch.jit.load(str(self.root/'weights/ViT-B-32.pt'),map_location='cpu')
        self.model = importlib.import_module('clip.model').build_model(dict(jit.state_dict())).float().eval(); del jit
        require(self.model.visual.input_resolution == 224 and all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in self.model.parameters()), 'model device/dtype')
        require(cap.learned_logit_scale(self.model) == self.probe['learned_logit_scale'], 'learned scale changed')
        self.transform = api._transform(224)
    def read_member(self, index): return self.archive.read(self.members[index])
    def preprocess(self, payloads):
        from PIL import Image
        tensors = []
        for raw in payloads:
            with Image.open(io.BytesIO(raw)) as image:
                require(image.format == 'JPEG' and image.mode == 'RGB' and image.size == (64,64), 'JPEG contract')
                tensors.append(self.transform(image))
        batch = self.torch.stack(tensors)
        require(tuple(batch.shape) == (4,3,224,224) and batch.dtype == self.torch.float32 and batch.device.type == 'cpu', 'preprocess contract')
        return batch
    def encode(self, batch):
        with self.torch.inference_mode(): return self.model.encode_image(batch).detach().cpu().numpy()
    def release_encoder(self):
        self.model = None; self.transform = None; gc.collect()
    def backend(self):
        require(self.model is None, 'encoder still held')
        return self.h.TorchStatA(module_at(self.paths['stata_cpu'],STATA_SHA),self.paths['stata_upstream'])
    def labels(self, names):
        return [self.cfg['class_order'].index(name.rsplit('_',1)[0]) for name in names]
    def close(self):
        self.release_encoder()
        if self.archive is not None: self.archive.close(); self.archive = None
