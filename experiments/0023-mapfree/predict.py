"""One frozen, offline three-place prediction operation; ground truth is absent."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
METHOD_NAMES=['native_E','geometry_multiview','joint_field','hessian','scalar_deletion','residual_scalar','directional']


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value)
    return value


def source_paths(protocol):
    return [HERE/f for f in ('predict.py','supervise.py','cache_inputs.py','study.py','geometry_ba.py',
            'field_ba.py','conditional.py','covariance.py','frontend.py','acquire_operation.py',
            'range_inputs.py','propose_plan.py')]+[Path(protocol),ROOT/'configs/0017-libraries.json']


def expand(value):
    return value.replace('$REPOSITORY',str(ROOT)).replace('$PRIVATE',str(ROOT.parent)).replace('$PYTHON',sys.executable)


def require(value,message):
    if not value:raise ValueError(message)


def frontend_admission(args):
    """All source/chain subprocess verification occurs before offline import guard."""
    frontend=module('_evr0023_prediction_frontend',HERE/'frontend.py')
    q=frontend.q;supervisor=frontend.acquisition.supervisor
    record=q.read_json(args.frontend_cache/'receipt.json');outer=q.read_json(args.frontend_supervision)
    require(record.get('schema')=='evr0023.frontend.v1' and record.get('status')=='completed'
            and record.get('eligible') is True,'complete frontend required')
    require(record.get('expected_commit')==outer.get('expected_commit')==args.expected_commit,'frontend commit')
    require(outer.get('schema')=='evr0023.supervision.v1' and outer.get('operation')=='frontend'
            and outer.get('status')=='completed' and outer.get('exit_code')==0
            and outer.get('termination') is None and outer.get('remaining_descendant_pids')==[]
            and outer.get('cleanup_errors')==[],'frontend outer failure')
    require(outer.get('hard_timeout_seconds')==1800 and outer.get('group_rss_limit_kib')==8*1024**2
            and 0<=outer['wall_seconds']<1800 and 0<=outer['peak_group_rss_kib']<=8*1024**2
            and outer.get('thread_environment')=={k:'4' for k in supervisor.THREAD_KEYS},'frontend outer limits')
    for kind in ('stdout','stderr'):
        require(q.sha(args.frontend_supervision.parent/(kind+'.log'))==outer[kind+'_sha256'],'frontend outer log changed')
    stdout=(args.frontend_supervision.parent/'stdout.log').read_text().splitlines()
    require(stdout and json.loads(stdout[-1])=={k:record[k] for k in ('status','eligible','wall_seconds')},'frontend stdout/receipt mismatch')
    command=outer['command'];require(command[:2]==['$PYTHON','$REPOSITORY/experiments/0023-mapfree/frontend.py']
            and len(command[2:])%2==0,'frontend command')
    flags={};
    for key,value in zip(command[2::2],command[3::2]):
        require(key.startswith('--') and key not in flags,'duplicate/invalid frontend flag');flags[key]=value
    paths=('plan','input-receipt','input-operation','input-supervision','inputs','output-dir','protocol','installation',
           'installation-supervision','acquisition-receipt','assets-dir','qualification-receipt','qualification-supervision')
    scalars=('plan-sha256','input-receipt-sha256','expected-commit','frontend-freeze')
    defaults={'frontend-protocol':ROOT/'protocols/0022-traceable-frontend-assets.md',
              'asset-manifest':ROOT/'configs/0022-source-model-assets.json',
              'wheel-manifest':ROOT/'results/0022-preparation/dependency-wheels.json',
              'wheel-assets-manifest':ROOT/'configs/0022-wheel-assets.json'}
    require(set(flags)<={ '--'+k for k in (*paths,*scalars,*defaults)} and
            {'--'+k for k in (*paths,*scalars)}<=set(flags),'frontend flag set')
    kwargs={k.replace('-','_'):Path(expand(flags['--'+k])).resolve() for k in paths}
    kwargs.update({k.replace('-','_'):flags['--'+k] for k in scalars})
    kwargs.update({k.replace('-','_'):Path(expand(flags.get('--'+k,str(v)))).resolve() for k,v in defaults.items()})
    fa=argparse.Namespace(**kwargs)
    require(fa.expected_commit==args.expected_commit and fa.plan_sha256==args.plan_sha256
            and fa.plan==args.plan.resolve() and fa.protocol==args.protocol.resolve()
            and fa.output_dir==args.frontend_cache.resolve(),'frontend command inputs')
    own=[HERE/f for f in ('frontend.py','range_inputs.py','propose_plan.py','acquire_operation.py','supervise.py')]+[args.protocol]
    frozen=supervisor.verify_frozen(args.expected_commit,own)
    require(record['source_sha256']==frozen,'frontend source mismatch')
    outer_paths=[HERE/'supervise.py',HERE/'frontend.py']+[HERE/f for f in supervisor.HELPERS['frontend']]
    require(outer['source_sha256']==supervisor.verify_frozen(args.expected_commit,outer_paths),'frontend supervisor source mismatch')
    qa=frontend.qualification_args(fa);qfrozen=q.frozen_bindings(qa)
    _,_,_,bindings=q.verify_files(qa,qfrozen)
    qualification=frontend.verify_qualification(fa,bindings,qfrozen)
    require(record['frontend_bindings']==dict(bindings,**qualification),'model qualification chain')
    require(frontend.acquisition.verify_completed_chain(fa)==record['input_acquisition_bindings'],'input acquisition chain')
    _,inputs=frontend.verify_inputs(fa);require(inputs==record['input_sha256'],'frontend input bytes')
    return frontend,fa,record,dict(frontend_receipt_sha256=q.sha(args.frontend_cache/'receipt.json'),
            frontend_supervision_sha256=q.sha(args.frontend_supervision),
            frontend_supervision_log_sha256={k:outer[k+'_sha256'] for k in ('stdout','stderr')}),qfrozen


def native_binding(runtime,q):
    spec=q.read_json(ROOT/'configs/0017-libraries.json')['current'];package=runtime/'poselib'
    names={**spec['python_files'],spec['binary_name']:spec['binary_sha256']}
    for name,digest in names.items():
        p=q.safe_child(package,name);require(q.sha(p)==digest,'native source/binary changed')
    require((package/spec['binary_name']).stat().st_size==spec['binary_bytes'],'native binary size')
    return dict(version=spec['version'],files=names)


def run(args):
    began=time.perf_counter();require(not args.output_dir.exists() and not args.output_dir.is_symlink(),'fresh prediction output required')
    require(not args.output_dir.resolve().is_relative_to(ROOT),'private outputs outside repository required')
    args.output_dir.mkdir(parents=True)
    record=dict(schema='evr0023.prediction.v1',status='started',eligible=False,stage='verification',
                expected_commit=args.expected_commit,plan_sha256=args.plan_sha256,query_count=0,
                method_names=METHOD_NAMES,outputs={},offline_attempts=[],compatibility_probes=[],ground_truth_access=False)
    def save():
        temp=args.output_dir/'receipt.pending';temp.write_text(json.dumps(record,indent=2,sort_keys=True,allow_nan=False)+'\n')
        temp.replace(args.output_dir/'receipt.json')
    save()
    try:
        front,fa,fr,frontbindings,qfrozen=frontend_admission(args);q=front.q;supervisor=front.acquisition.supervisor
        frozen=supervisor.verify_frozen(args.expected_commit,source_paths(args.protocol));record['source_sha256']=frozen
        require(q.sha(args.plan)==args.plan_sha256 and args.plan_sha256 in args.protocol.read_text(),'admitted plan')
        record.update(frontbindings);record['native_binding']=native_binding(args.native_runtime,q)
        record['frontend_complete_wall_seconds']=fr['wall_seconds'];record['installation_binding']=fr['frontend_bindings']
        record['frontend_source_sha256']=fr['source_sha256'];record['frontend_freeze']=fr['frontend_freeze']
        for key in supervisor.THREAD_KEYS:os.environ[key]='4'
        os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1');sys.dont_write_bytecode=True
        sys.path[:0]=[str(args.native_runtime),str(fa.installation/'runtime'),str(HERE)]
        q.install_guard(record['offline_attempts'],record['compatibility_probes'])
        record['offline_guard_scope']='Python audit events, not an OS sandbox for arbitrary native syscalls'
        import numpy as np
        import scipy
        import poselib
        import cache_inputs
        import study
        require(np.__version__=='2.0.2' and scipy.__version__=='1.13.1' and poselib.__version__=='3.0.0','scientific runtime versions')
        require(Path(poselib.__file__).resolve().parent==(args.native_runtime/'poselib').resolve(),'native import origin')
        bindings={k:fr[k] for k in cache_inputs.BINDING_FIELDS};bindings['frontend_receipt_sha256']=frontbindings['frontend_receipt_sha256']
        record['stage']='canonical_cache';save();inputs=cache_inputs.load_inputs(args.plan,args.frontend_cache,bindings)
        def write_output(name,value):
            p=args.output_dir/name
            with p.open('x') as stream:json.dump(study.serial(value),stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
            record['outputs'][name]=dict(sha256=q.sha(p),bytes=p.stat().st_size);save()
        write_output('selection.json',inputs['selection_receipt'])
        for scene in inputs['scenes']:
            def emit(kind,value):
                record['progress']=dict(scene=scene['scene'],kind=kind,
                    scope=value.get('scope'),query=value.get('query'),status=value.get('status'))
                save()
            record['stage']='scene:'+scene['scene'];save()
            result=study.run_scene(scene,poselib,emit=emit)
            require(len(result['queries'])==8 and all(set(row['arms'])==set(study.ARMS) for row in result['queries']),'complete scene output')
            write_output('scene-'+scene['scene']+'.json',result);record['query_count']+=8;save()
        require(record['query_count']==24 and not record['offline_attempts'],'incomplete or nonoffline prediction')
        record['stage']='postcheck';save()
        require(q.sha(args.frontend_cache/'receipt.json')==frontbindings['frontend_receipt_sha256'],'frontend receipt changed')
        require(q.sha(args.frontend_supervision)==frontbindings['frontend_supervision_sha256'],
                'frontend supervision receipt changed')
        require(all(q.sha(args.frontend_supervision.parent/(kind+'.log'))==digest
                    for kind,digest in frontbindings['frontend_supervision_log_sha256'].items()),
                'frontend supervision logs changed')
        cache_inputs.verify_cache(args.plan,args.frontend_cache,bindings)
        require(all(q.sha(ROOT/name)==digest for name,digest in {**frozen,**qfrozen,**fr['source_sha256']}.items()),'source changed during prediction')
        require(native_binding(args.native_runtime,q)==record['native_binding'],'native binary changed during prediction')
        record.update(status='completed',eligible=True,stage='completed',wall_seconds=time.perf_counter()-began);save()
        seal=dict(schema='evr0023.prediction-seal.v1',complete=True,query_count=24,
                  receipt_sha256=q.sha(args.output_dir/'receipt.json'),outputs=record['outputs'])
        with (args.output_dir/'seal.json').open('x') as stream:json.dump(seal,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
        return record
    except BaseException as exc:
        record.update(status='failed',eligible=False,error=dict(type=type(exc).__name__,message=str(exc)),
                      failed_stage=record['stage'],wall_seconds=time.perf_counter()-began);save();raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','plan','frontend-cache','frontend-supervision','native-runtime','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('expected-commit','plan-sha256'):parser.add_argument('--'+name,required=True)
    result=run(parser.parse_args());print(json.dumps({k:result[k] for k in ('status','eligible','wall_seconds')}))


if __name__=='__main__':main()
