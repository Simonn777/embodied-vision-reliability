"""Artificial 64-file contract test; never read science cache or call any model.
Production scalar metrics generate targets; the standalone checker independently replays.
"""
import copy, hashlib, importlib.util, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import numpy as np

REPO=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
CHECKER=REPO/'scripts/check_antithetic_training.py'
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--report',type=Path,required=True)
report_path=parser.parse_args().report

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

start=time.monotonic();t=load('synthetic_supervision_fixture',REPO/'tests/test_antithetic_training_supervision.py');m=t.m
sys.path.insert(0,str(REPO/'scripts'))
metrics=load('synthetic_target_metrics',REPO/'scripts/antithetic_training_metrics.py')
with tempfile.TemporaryDirectory(prefix='evr0008-contract-') as td:
    root=Path(td)/'fixture';captured=[];original=m.validate_evidence
    def capture(args,cfg,frozen,library):
        result=original(args,cfg,frozen,library)
        if not captured:
            shutil.copytree(m.ROOT,root);captured.append(copy.deepcopy(cfg))
        return result
    with mock.patch.object(m,'validate_evidence',capture):
        t.Supervision('test_complete_saved_grid_and_tamper_gates_before_reference').test_complete_saved_grid_and_tamper_gates_before_reference()
    cfg=captured[0];cfg.update(methods=list(m.METHODS),endpoints=list(m.ENDPOINTS));cache=root/'cache';out=root/'out'
    (root/'configs').mkdir();refs={}
    for group in ('d1','d2','d3','d4'):
        clean=np.random.default_rng(100+int(group[1])).uniform(.2,.8,(128,128)).astype(np.float32)
        path=cache/(group+'-reference.npz');np.savez(path,clean=clean);refs[group]=clean[3:-3,3:-3]
    for c in cfg['cases']:
        c.update(reference_relative=c['group']+'-reference.npz',reference_sha256=sha(cache/(c['group']+'-reference.npz')))
    write(root/'configs/0008-training.json',cfg)
    for relative in m.REQUIRED:
        if relative=='configs/0008-training.json':continue
        destination=root/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(CHECKER if relative=='scripts/check_antithetic_training.py' else REPO/relative,destination)
    frozen={'source_commit':'a'*40,'files':{p:sha(root/p) for p in m.REQUIRED}};write(root/m.FREEZE,frozen);fhash=sha(root/m.FREEZE)
    started=read(out/'study.started.json');started['freeze_sha256']=fhash;write(out/'study.started.json',started)
    complete=read(out/'study.completed.json');rows=[];bindings=[];executions=[]
    args=SimpleNamespace(cache=cache,output=out,runtime_root=root/'runtime',freeze_commit='c'*40)
    for c,method in m.operation_order(cfg):
        name=m.case_name(c,method);folder=out/'cases'/name;prefix=out/'control'/name;r=read(folder/'receipt.json')
        r.update(identity=c,freeze_sha256=fhash,source_hashes={k:v for k,v in frozen['files'].items() if k.startswith('scripts/')})
        for endpoint,row in r['endpoints'].items():
            # Spatially varying values exercise both directional differences and nontrivial blocks.
            pattern=np.random.default_rng(700+c['index']).standard_normal((122,122)).astype(np.float32)
            pred=(refs[c['group']]+pattern*np.float32(.02 if method=='mean_antithetic' else .04)).astype(np.float32)
            path=folder/(endpoint+'-prediction.npy');np.save(path,pred);r['files'][path.name]=sha(path)
            inf=row['inference'];identity='0008-antithetic-training-inference-v1'
            inf.update(schema=identity,source_identifier=identity,helper_identifier='0008-training-core-v1',patch_count=14884,parameter_count=5313,
                expected_forward_calls=750,expected_standard_normal_draws=36465800,dtype='float32',device='cpu',deterministic_algorithms=True,prediction_sha256=m.array_hash(pred))
            row['inference_source_sha256']=frozen['files']['scripts/antithetic_training_inference.py']
        write(folder/'receipt.json',r)
        with mock.patch.object(m,'ROOT',root):write(prefix.with_suffix('.command.private.json'),{'argv':m.expected_command(args,c,method),'cwd':str(root)})
        ex=read(prefix.with_suffix('.execution.json'));ex.update(receipt_sha256=sha(folder/'receipt.json'),command_file_sha256=sha(prefix.with_suffix('.command.private.json')))
        write(prefix.with_suffix('.execution.json'),ex);executions.append(dict(case=name,**ex))
        for endpoint,row in r['endpoints'].items():
            path=folder/(endpoint+'-prediction.npy');pred=np.load(path,allow_pickle=False)
            rows.append(dict(group=c['group'],seed=c['seed'],method=method,endpoint=endpoint,status='completed',inference_mc_samples=50,
                updates=row['training']['step'],training_seconds=row['training']['training_elapsed_seconds'],pipeline_seconds=m.allocated_outer_cost(ex,r,endpoint),score=metrics.score_pair(refs[c['group']],pred)))
            bindings.append(dict(case=name,endpoint=endpoint,receipt_sha256=sha(folder/'receipt.json'),reference_sha256=c['reference_sha256'],prediction_file_sha256=sha(path)))
    complete['cases']=executions;write(out/'study.completed.json',complete)
    scores=metrics.aggregate_cases(rows);scores['bindings']=dict(source_commit='a'*40,freeze_commit='c'*40,freeze_sha256=fhash,endpoints=bindings);write(out/'scores.json',scores)
    # Also pass the production archive validator; it opens no reference/model.
    with mock.patch.object(m,'ROOT',root):assert len(original(args,cfg,frozen,np))==64
    sequence=[]
    def run(label,expected):
        report=Path(td)/(label+'.json')
        result=subprocess.run([sys.executable,str(CHECKER),'--repo',str(root),'--cache',str(cache),'--output-root',str(out),'--report',str(report)],capture_output=True,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'},timeout=30)
        record=read(report);assert result.returncode==expected,(label,result.returncode,result.stdout,result.stderr)
        sequence.append(dict(test=label,actual_exit=result.returncode,status=record['status'],checks=record.get('checks'),numeric_checks=record.get('numeric_checks'),max_absolute_difference=record.get('max_absolute_difference'),failure=record.get('message')))
    run('complete64',0)
    bad=copy.deepcopy(scores);bad['cases'][0]['score']['blocks'][0]['sse']+=.1;write(out/'scores.json',bad);run('block-tamper',1)
    bad=copy.deepcopy(scores);bad['gates']['all_passed']=False;write(out/'scores.json',bad);run('gate-tamper',1)
    bad=copy.deepcopy(scores);bad['cases'][0]['pipeline_seconds']+=.1;write(out/'scores.json',bad);run('outer-cost-tamper',1)
    write(out/'scores.json',scores)
    name=executions[0]['case'];rp=out/'cases'/name/'receipt.json';r=read(rp);r['endpoints']['wall5s']['inference']['forward_calls']=749;write(rp,r)
    ep=out/'control'/(name+'.execution.json');ex=read(ep);ex['receipt_sha256']=sha(rp);write(ep,ex);complete['cases'][0]=dict(case=name,**ex);write(out/'study.completed.json',complete)
    bad=copy.deepcopy(scores)
    for b in bad['bindings']['endpoints']:
        if b['case']==name:b['receipt_sha256']=sha(rp)
    write(out/'scores.json',bad);run('rebound-inference-call-tamper',1)
    result=dict(status='PASS',scope='Synthetic file contract only; production metrics generate targets; no scientific arrays, Torch imports or model calls.',
        checker_sha256=sha(CHECKER),test_source_sha256=sha(Path(__file__)),production_wrapper_sha256=sha(REPO/'scripts/run_antithetic_training.py'),
        production_metrics_sha256=sha(REPO/'scripts/antithetic_training_metrics.py'),tests=sequence,elapsed_seconds=time.monotonic()-start,python_version=sys.version.split()[0],numpy_version=np.__version__)
with report_path.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(result,sort_keys=True))
