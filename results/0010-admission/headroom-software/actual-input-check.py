"""Saved JSON/arrays only. No model, encoder, StatA, asset archive or network call."""
from pathlib import Path
import ast
import hashlib
import json
import math
import sys
from types import ModuleType
import numpy as np

REPO=Path(__file__).resolve().parents[3]  # Public path-portable copy; see source-redaction.json.
HERE=REPO/'experiments/0010-headroom'
PUBLIC=Path('/tmp/evr-0010-execution-public')
PRIVATE=Path('/tmp/evr-0010-execution-audit-private')
CLOUD='/workspace/scratch/a349d03a0e0a/'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
checks=[]
def check(ok,label):
    checks.append({'check':label,'passed':bool(ok)})
    if not ok: raise AssertionError(label)

def load(name,path,expected):
    check(sha(path)==expected,'source '+name)
    m=ModuleType(name);m.__file__=str(path);sys.modules[name]=m
    exec(compile(Path(path).read_bytes(),str(path),'exec'),m.__dict__)
    return m

plan=read(HERE/'plan.template.json');check(plan['admitted'] is False,'plan initially unadmitted')
actual={
'capability_receipt':('public','scratch/0010-capability-output/receipt.json'),
'capability_features':('private','scratch/0010-capability-output/features.npz'),
'functional_receipt':('public','scratch/0010-solver-functional/child/receipt.json'),
'functional_execution':('public','scratch/0010-solver-functional/supervision/execution.json'),
'functional_artifacts':('public','scratch/0010-solver-functional/artifact-manifest.json'),
' timing_receipt':('public','scratch/0010-solver-timing/child/receipt.json'),
' timing_execution':('public','scratch/0010-solver-timing/supervision/execution.json'),
' timing_artifacts':('public','scratch/0010-solver-timing/artifact-manifest.json')}
actual={key.strip():value for key,value in actual.items()}
local={}
for key,ref in plan['inputs'].items():
    if key in actual:
        kind,relative=actual[key];p=(PUBLIC if kind=='public' else PRIVATE)/relative
        ref.update(path=CLOUD+relative,sha256=sha(p));local[key]=p
    else:
        p=(HERE/ref['path']).resolve();local[key]=p
        check(sha(p)==ref['sha256'],'relative source/reference '+key)
for key,digest in plan['driver_files'].items():check(sha(HERE/key)==digest,'driver file '+key)
parent_rel='scratch/0010-capability-control/probe/execution.json'
parent_path=PUBLIC/parent_rel
plan['external_bindings']['capability_execution']={'path':CLOUD+parent_rel,'sha256':sha(parent_path)}
check(len(plan['inputs'])==18 and all(len(v['sha256'])==64 for v in plan['inputs'].values()),'all18 input SHA populated')
check(plan['admitted'] is False,'plan remains unadmitted')
# Write only the candidate; relative source paths are anchored at the intended repo directory, not /tmp.
candidate=Path('/tmp/evr-0010-headroom-plan-candidate.json')
candidate.write_text(json.dumps(plan,indent=2)+'\n')

driver=load('driver',HERE/'driver.py',plan['driver_files']['driver.py'])
native=load('_plan_native',HERE/'native.py',plan['driver_files']['native.py'])
cap=load('_plan_capability',local['capability_runner'],native.CAP_SHA)
cfg=read(local['capability_config']);cap.validate_config(cfg);checks.append({'check':'original capability validate_config','passed':True})
probe=read(local['capability_receipt']);timing=read(local['timing_receipt']);functional=read(local['functional_receipt'])
execution=read(local['timing_execution']);freeze=read(local['capability_freeze'])
for name,key in [('experiments/0010/run_capability.py','capability_runner'),('experiments/0010/config.json','capability_config'),('experiments/0010/selection.json','capability_selection')]:
    check(freeze['inputs'][name]['sha256']==plan['inputs'][key]['sha256'],'capability historical freeze '+key)
check(probe['status']=='completed' and [probe[k] for k in ('image_attempted','image_completed','text_attempted','text_completed')]==[8,8,1,1],'capability completion/call counts')
for field,key in [('source','capability_runner'),('config','capability_config'),('selection','capability_selection'),('features','capability_features')]:
    check(probe[field]['sha256']==plan['inputs'][key]['sha256'],'capability '+field+' binding')
check(set(probe['assets'])==set(cfg['assets']),'recorded asset set')
for path,spec in cfg['assets'].items():
    got=probe['assets'][path]
    check(set(got)=={'bytes','sha256','git_blob_sha1'} and all(value is None or got[key]==value for key,value in spec.items()),'recorded asset config identity '+path)
cap.check_runtime_versions(probe['runtime'],cfg['runtime_versions']);checks.append({'check':'capability actual saved distribution versions','passed':True})
qualified=native.timing_admission(probe,timing,execution,cfg);checks.append({'check':'actual native.timing_admission saved-record arithmetic','passed':True})
solver_manifest=read(local['solver_manifest'])
check(timing['manifest_sha256']==plan['inputs']['solver_manifest']['sha256'] and timing['source_files']==solver_manifest['files'],'timing source manifest')
check(timing['source_files']['stata_cpu.py']==native.STATA_SHA and timing['source_files']['upstream/StatA.py']==native.UPSTREAM_SHA,'timing solver exact sources')
check(timing['adapter_execution']['original_sha256']==native.STATA_SHA,'timing adapter original')
check(functional['mode']=='functional' and functional['status']=='completed' and functional['tests']==dict(run=20,failures=0,errors=0,skipped=0) and functional['actual_torch_threads']==[4,4],'actual20 functional tests')
check(all(functional[k]==timing[k] for k in ('manifest_sha256','source_files','adapter_execution','runtime')),'functional/timing common provenance')
for prefix in ('timing','functional'):
    bundle=read(local[prefix+'_artifacts']);ex=read(local[prefix+'_execution'])
    check(bundle['source_manifest_sha256']==plan['inputs']['solver_manifest']['sha256'],prefix+' manifest source')
    for rel,key in [('child/receipt.json',prefix+'_receipt'),('supervision/execution.json',prefix+'_execution')]:
        check(bundle['files'][rel]==dict(bytes=local[key].stat().st_size,sha256=plan['inputs'][key]['sha256']),prefix+' artifact binding '+rel)
    check(ex['supervisor_exit_code']==ex['child_exit_code']==0 and ex['timed_out'] is False and ex['wall_limit_seconds']==120 and ex['address_space_bytes']==8*1024**3 and ex['cpu_affinity']==cfg['cpu_ids'] and ex['numeric_threads']==4,prefix+' actual parent resource/exit')
    for name,identity in ex['logs'].items():
        log=local[prefix+'_execution'].parent/name
        check(identity=={'bytes':log.stat().st_size,'sha256':sha(log)},prefix+' log '+name)
    for rel,identity in bundle['files'].items():
        path=local[prefix+'_artifacts'].parent/rel
        if not path.exists():path=PRIVATE/('scratch/0010-solver-'+prefix)/rel
        check(identity=={'bytes':path.stat().st_size,'sha256':sha(path)},prefix+' closed artifact '+rel)
parent=read(parent_path)
check(parent['raw_wait_status']==parent['child_exit_code']==parent['supervisor_exit_code']==0 and parent['timed_out'] is False,'external capability actual parent zero exit')
check(parent['wall_limit_seconds']==300 and parent['address_space_bytes']==8*1024**3 and parent['cpu_affinity']==[0,1,2,3] and parent['numeric_threads']==4,'external capability parent resources')
for name,identity in parent['logs'].items():
    log=parent_path.parent/name;check(identity=={'bytes':log.stat().st_size,'sha256':sha(log)},'external capability parent log '+name)
# Only saved text vectors are read; image arrays remain unopened.
with np.load(local['capability_features'],allow_pickle=False) as z:
    check(set(z.files)=={'image_raw','image_normalized','text_raw','text_normalized'},'capability NPZ keys')
    texts={key:z[key] for key in ('text_raw','text_normalized')}
for key,a in texts.items():
    check(a.dtype==np.float32 and a.shape==(10,512) and np.isfinite(a).all(),'saved '+key+' shape/finite')
    check(probe['array_hashes'][key]=={'shape':[10,512],'dtype':'float32','sha256_f32le':hashlib.sha256(a.astype('<f4',copy=False).tobytes()).hexdigest()},'saved '+key+' exact array identity')
norm=np.linalg.norm(texts['text_raw'],axis=1,keepdims=True)
check(np.isfinite(norm).all() and (norm>0).all() and np.array_equal(texts['text_raw']/norm,texts['text_normalized']),'consumer exact saved text normalization')
metadata=REPO/'experiments/0010/bundle/metadata'
train=(metadata/'eurosat-spatial-train.txt').read_text().splitlines()
excluded=read(local['capability_selection'])['names']
check(probe['names']==excluded and len(probe['members'])==8 and [Path(m['name']).name for m in probe['members']]==excluded,'capability8 member/name order')
check(len({m['sha256'] for m in probe['members']})==8,'capability8 distinct byte identities')
check(read(local['selection'])=={'names':driver.selected_names(train,excluded)},'fixed128 metadata selection')
# The cached author prompt literals are metadata, not imported code.
metadata_source=Path('/tmp/evr-stata-source-admission/source/datasets/eurosat.py')
check(sha(metadata_source)==probe['assets']['metadata/stata-eurosat.py']['sha256'],'cached author prompt metadata SHA')
values={}
for node in ast.parse(metadata_source.read_text()).body:
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('template','NEW_CNAMES'):
        values[node.targets[0].id]=ast.literal_eval(node.value)
check(values['template']==[cfg['template']] and list(values['NEW_CNAMES'])==cfg['class_order'],'author template/class order')
check(probe['prompts']==[cfg['template'].format(values['NEW_CNAMES'][name]) for name in cfg['class_order']],'actual prompts')
check(len(probe['token_ids'])==10 and all(len(row)==77 and all(type(v)is int for v in row) for row in probe['token_ids']),'saved token-ID schema only; no tokenizer invoked')
check(all(type(probe['learned_logit_scale'][k]) in (int,float) and math.isfinite(probe['learned_logit_scale'][k]) for k in ('raw','exp')) and probe['learned_logit_scale']['exp']>0,'saved learned-scale schema only; no model loaded')
check(not any(k=='torch' or k.startswith('torch.') or k=='clip' or k.startswith('clip.') for k in sys.modules),'no Torch/CLIP import')
summary={'scope':'Saved JSON, exact source/hash checks and saved text-vector validation only; no model, forward, StatA fit, pixel decode or network','status':'PASS_AVAILABLE_SAVED_CONSUMER_INPUTS','plan_admitted':False,'plan_sha256':sha(candidate),'intended_plan_directory':'experiments/0010-headroom (repo-relative source paths; do not launch candidate from its /tmp staging location)','checks':checks,'qualification':qualified,'actual_solver_times':{a['phase']:a['seconds'] for a in timing['attempts']},'functional_tests':functional['tests'],'capability_parent_wall_seconds':parent['wall_seconds'],'local_validation_numpy_version':np.__version__,'boundaries_not_verified_locally':['Current cloud asset filesystem/full model and archive bytes (saved identities checked; no re-download/local access)','Selected128 archive member existence/CRC/pixel contracts (no archive opened)','Fresh cloud resource/runtime/module state and one-shot marker absence','Retokenization/learned logit_scale equality on loaded model','Canonical headroom source freeze, actual launch authorization and subsequent128-image outcome'],'plan_inputs':plan['inputs'],'external_bindings':plan['external_bindings']}
Path('/tmp/evr-0010-headroom-plan-check.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'plan_sha256':summary['plan_sha256'],'checks_passed':len(checks),'qualification':qualified,'functional_tests':functional['tests'],'candidate_admitted':False,'numpy':np.__version__}))
