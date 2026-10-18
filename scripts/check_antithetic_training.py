#!/usr/bin/env python3
"""Independent saved-array replay: NumPy/stdlib only; no production/Torch imports.
Fixed tolerance rtol=1e-10, atol=1e-12. --self-test accesses synthetic fixtures only.
Scientific mode reads only pinned references and saved outputs/evidence; no model.
"""
import argparse, hashlib, json, math, sys, tempfile, time
from pathlib import Path
import numpy as np

GROUPS=('d1','d2','d3','d4'); SEEDS=(0,1)
METHODS=('independent','random_antithetic','mean_antithetic','same_patch')
CONTROLS=(METHODS[0],METHODS[1],METHODS[3]); ENDPOINTS=('updates500','wall5s')
RTOL=1e-10; ATOL=1e-12

def need(ok,message):
    if not ok: raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))

def checked(root,relative,digest=None):
    p=Path(relative); need(not p.is_absolute() and '..' not in p.parts,'unsafe relative path')
    path=Path(root).resolve()
    for part in p.parts:
        path=path/part; need(not path.is_symlink(),'symlink in evidence path')
    need(path.is_file(),'missing evidence file')
    if digest is not None: need(sha(path)==digest,'file SHA mismatch: '+str(relative))
    return path

def positive(x):
    need(type(x) in (int,float) and math.isfinite(x) and x>0,'nonpositive/nonfinite timing')
    return x

class Compare:
    def __init__(self): self.checks=0; self.numeric=0; self.max_absolute=0.
    def equal(self,actual,expected,label):
        self.checks+=1
        if isinstance(expected,dict):
            need(isinstance(actual,dict) and set(expected)<=set(actual),'missing mapping: '+label)
            for k,v in expected.items(): self.equal(actual[k],v,label+'.'+k)
        elif isinstance(expected,list):
            need(isinstance(actual,list) and len(actual)==len(expected),'list length: '+label)
            for i,v in enumerate(expected): self.equal(actual[i],v,label+'.'+str(i))
        elif isinstance(expected,float):
            need(type(actual) in (int,float) and math.isfinite(actual) and math.isfinite(expected),'nonfinite number: '+label)
            self.numeric+=1; self.max_absolute=max(self.max_absolute,abs(actual-expected))
            need(math.isclose(actual,expected,rel_tol=RTOL,abs_tol=ATOL),'numeric mismatch: '+label)
        else: need(type(actual) is type(expected) and actual==expected,'identity mismatch: '+label)

def moments(error):
    # Coordinate-wise compensated scalar reduction; not the production np.sum.
    sse=math.fsum(float(v)*float(v) for v in error.flat)
    need(math.isfinite(sse),'nonfinite squared error')
    return dict(sse=sse,count=error.size,mse=sse/error.size)

def metric(clean,prediction):
    need(clean.ndim==2 and clean.shape==prediction.shape and min(clean.shape)>=8,'metric geometry')
    need(clean.dtype in (np.float32,np.float64) and prediction.dtype==np.float32,'metric dtype')
    need(np.isfinite(clean).all() and np.isfinite(prediction).all(),'nonfinite array')
    c=clean.astype(np.float64);p=prediction.astype(np.float64);err=p-c;pixel=moments(err)
    horizontal=moments((p[:,1:]-p[:,:-1])-(c[:,1:]-c[:,:-1]))
    vertical=moments((p[1:]-p[:-1])-(c[1:]-c[:-1])); n=horizontal['count']+vertical['count']
    total=math.fsum((horizontal['sse'],vertical['sse'])); need(math.isfinite(total),'nonfinite FD sum')
    edges=[]
    for length in c.shape:
        q,r=divmod(length,8); edges.append([i*q+min(i,r) for i in range(9)])
    blocks=[]
    for i in range(8):
        for j in range(8):
            a,b=edges[0][i:i+2];u,v=edges[1][j:j+2];sub=moments(err[a:b,u:v])
            blocks.append(dict(row=i,col=j,row_start=a,row_stop=b,col_start=u,col_stop=v,sse=sub['sse'],count=sub['count']))
    return dict(shape=list(c.shape),**pixel,psnr_db=None if pixel['sse']==0 else 10*(math.log10(pixel['count'])-math.log10(pixel['sse'])),
        data_range=1,clipped=False,first_difference=dict(horizontal=horizontal,vertical=vertical,combined=dict(sse=total,count=n,mse=total/n)),
        blocks=blocks,clean_sha256_f64le=hashlib.sha256(c.astype('<f8').tobytes(order='C')).hexdigest(),
        prediction_sha256_f32le=hashlib.sha256(prediction.astype('<f4').tobytes(order='C')).hexdigest())

def reduce_rows(rows):
    keys={(g,s,m,e) for g in GROUPS for s in SEEDS for m in METHODS for e in ENDPOINTS}; indexed={}
    for r in rows:
        k=(r['group'],r['seed'],r['method'],r['endpoint']); need(k in keys and k not in indexed,'duplicate/unexpected endpoint');indexed[k]=r
    need(set(indexed)==keys,'incomplete64 grid')
    endpoints={}
    for endpoint in ENDPOINTS:
        pairs=[]
        for group in GROUPS:
            for seed in SEEDS:
                candidates=[indexed[group,seed,m,endpoint]['score'] for m in CONTROLS]
                best=min(range(3),key=lambda i:candidates[i]['mse']);base=candidates[best];cand=indexed[group,seed,METHODS[2],endpoint]['score']
                pairs.append(dict(group=group,seed=seed,oracle_best_control=CONTROLS[best],candidate_psnr_db=cand['psnr_db'],control_psnr_db=base['psnr_db'],
                    gain_db=None if min(cand['mse'],base['mse'])==0 else cand['psnr_db']-base['psnr_db'],
                    candidate_first_difference_mse=cand['first_difference']['combined']['mse'],control_first_difference_mse=base['first_difference']['combined']['mse']))
        finite=all(p['gain_db'] is not None for p in pairs)
        groups={g:(math.fsum(p['gain_db'] for p in pairs if p['group']==g)/2 if all(p['gain_db'] is not None for p in pairs if p['group']==g) else None) for g in GROUPS}
        mean=math.fsum(p['gain_db'] for p in pairs)/8 if finite else None
        fc=math.fsum(p['candidate_first_difference_mse'] for p in pairs)/8;fb=math.fsum(p['control_first_difference_mse'] for p in pairs)/8
        gates=dict(finite_paired_db_gains=finite,mean_gain_ge_0_5_db=mean>=.5 if finite else None,
            at_least3_group_means_gt0=sum(v>0 for v in groups.values())>=3 if finite else None,
            each_group_mean_ge_minus0_2_db=all(v>=-.2 for v in groups.values()) if finite else None,mean_pooled_first_difference_le_1_05=fc<=1.05*fb)
        endpoints[endpoint]=dict(pairs=pairs,mean_gain_db=mean,group_mean_gain_db=groups,mean_candidate_first_difference_mse=fc,
            mean_corresponding_control_first_difference_mse=fb,gates=gates,pipeline_seconds_by_method={m:math.fsum(positive(indexed[g,s,m,endpoint]['pipeline_seconds']) for g in GROUPS for s in SEEDS) for m in METHODS})
    cost=endpoints['wall5s']['pipeline_seconds_by_method'];ratio=cost[METHODS[2]]/cost[METHODS[0]]
    need(math.isfinite(ratio),'nonfinite cost ratio');quality=all(v is True for e in endpoints.values() for v in e['gates'].values());passed=quality and ratio<=1.10
    return dict(schema='0008-training-quality-metrics-v1',status='not_evaluable' if any(not e['gates']['finite_paired_db_gains'] for e in endpoints.values()) else 'continue_to_novelty_review_only' if passed else 'stop_this_candidate',
        eligible_for_novelty_review=passed,paper_success=False,development_only=True,case_count=64,trajectory_count=32,groups=list(GROUPS),seeds=list(SEEDS),methods=list(METHODS),oracle_control_order=list(CONTROLS),endpoints=endpoints,
        wall5s_candidate_vs_independent_pipeline_ratio=ratio,gates=dict(complete64_grid=True,both_endpoint_quality_gates_pass=quality,wall5s_pipeline_ratio_le_1_10=ratio<=1.10,all_passed=passed))

def array_identity(a):
    header=json.dumps(dict(shape=list(a.shape),dtype=a.dtype.str),sort_keys=True,separators=(',',':')).encode('ascii')
    return hashlib.sha256(header+b'\0'+a.tobytes(order='C')).hexdigest()

def inference_binding(compare,row,fit,index,pred,source_sha):
    identity='0008-antithetic-training-inference-v1';weight=row['training']['state_dict_sha256']
    seed=int.from_bytes(hashlib.sha256(('evr0008|%d|inference|0'%index).encode('ascii')).digest()[:8],'little')&((1<<63)-1)
    expected=dict(schema=identity,source_identifier=identity,helper_identifier='0008-training-core-v1',case_index=index,inference_seed=seed,
        status='completed',failure=None,finite_outputs=True,weight_unchanged=True,weight_content_sha256=weight,weight_before_sha256=weight,weight_after_sha256=weight,
        prediction_sha256=array_identity(pred),mc_samples=50,chunk_size=1024,patch_count=14884,output_shape=[122,122],parameter_count=5313,architecture=[49,64,32,1],
        forward_calls=750,forward_calls_attempted=750,expected_forward_calls=750,standard_normal_draws=36465800,expected_standard_normal_draws=36465800,
        direct_forward_calls=0,optimizer_steps=0,alpha=.5,dtype='float32',accumulation_dtype='float64',device='cpu',interpolation='none',clipping='none',
        clean_reference_accessed=False,numeric_threads=4,interop_threads=1,deterministic_algorithms=True,torch_version='2.10.0+cpu',numpy_version='2.0.2',
        noisy_sha256=fit['extra_setup']['noisy_sha256'],**{k:fit[k] for k in ('patches_sha256','covariance_sha256','factor_sha256')})
    compare.equal(row['inference_source_sha256'],source_sha,'inference source');compare.equal(row['inference'],expected,'inference receipt')

def allocated(ex,r,e):
    row=r['endpoints'][e];t=row['training']['training_elapsed_seconds'];positive(t)
    extra=positive(r['fit']['training_wall_seconds'])-t;need(extra>=0,'negative later training')
    for v in r['endpoints'].values():need(0<positive(v['training']['update_completion_elapsed_seconds'])<=positive(v['training']['training_elapsed_seconds'])<=r['fit']['training_wall_seconds'],'nested endpoint clocks')
    other=math.fsum(positive(v['weights_write_seconds'])+positive(v['restore_infer_save_seconds']) for k,v in r['endpoints'].items() if k!=e)
    cost=positive(ex['wall_seconds'])-extra-other;positive(cost)
    algorithm=math.fsum((positive(r['input_seconds']),positive(r['fit']['shared_setup_seconds']),t,positive(row['weights_write_seconds']),positive(row['restore_infer_save_seconds'])))
    need(cost+ATOL>=algorithm,'allocated cost below measured component sum')
    return cost,algorithm

def replay(repo,cache,out):
    compare=Compare();cfg=read(checked(repo,'configs/0008-training.json'));freeze=read(checked(repo,'protocols/0008-training-freeze.json'))
    compare.equal(sha(checked(repo,'configs/0008-training.json')),freeze['files']['configs/0008-training.json'],'config SHA')
    compare.equal(sha(Path(__file__)),freeze['files']['scripts/check_antithetic_training.py'],'checker frozen SHA')
    for relative,digest in freeze['files'].items():checked(repo,relative,digest)
    scores_path=checked(out,'scores.json');scores=read(scores_path);fhash=sha(checked(repo,'protocols/0008-training-freeze.json'))
    compare.equal(scores['bindings'],dict(source_commit=freeze['source_commit'],freeze_sha256=fhash),'score freeze')
    compare.equal(cfg['methods'],list(METHODS),'fixed methods');compare.equal(cfg['endpoints'],list(ENDPOINTS),'fixed endpoints')
    need(len(cfg['cases'])==8 and len(scores['cases'])==64 and len(scores['bindings']['endpoints'])==64,'fixed grid lengths')
    published={};bindings={};rows=[];inputs=[];receipts=[];refs={};initial={};streams={}
    for row in scores['cases']:
        k=(row['group'],row['seed'],row['method'],row['endpoint']);need(k not in published,'duplicate scored row');published[k]=row
    for b in scores['bindings']['endpoints']:
        k=(b['case'],b['endpoint']);need(k not in bindings,'duplicate binding');bindings[k]=b
    started=read(checked(out,'study.started.json'));completed=read(checked(out,'study.completed.json'))
    compare.equal(started,dict(source_commit=freeze['source_commit'],freeze_commit=scores['bindings']['freeze_commit'],freeze_sha256=fhash),'study start')
    compare.equal(completed,dict(source_commit=freeze['source_commit'],freeze_commit=started['freeze_commit'],inputs_after=started['inputs']),'study completion')
    order=[c['name']+'-'+METHODS[(c['index']+j)%4] for c in cfg['cases'] for j in range(4)]
    compare.equal(started['order'],order,'frozen operation order');compare.equal([r['case'] for r in completed['cases']],order,'completed order')
    compare.equal(started['inputs'],[dict(relative=c['input_relative'],sha256=c['input_sha256']) for c in cfg['cases']],'declared inputs')
    need(len(completed['cases'])==32,'32 completed trajectories');execution={x['case']:x for x in completed['cases']};need(len(execution)==32,'unique completed trajectories')
    for index,c in enumerate(cfg['cases']):
        g,s=GROUPS[index//2],index%2;compare.equal(c,dict(index=index,name=g+'-colored-'+str(s),group=g,seed=s),'config case')
        if g not in refs:
            path=checked(cache,c['reference_relative'],c['reference_sha256'])
            with np.load(path,allow_pickle=False) as z:
                need(z.files==['clean'],'reference member');a=z['clean'];need(a.shape==(128,128) and a.dtype==np.float32 and np.isfinite(a).all(),'reference array');refs[g]=(a[3:-3,3:-3],c['reference_relative'],c['reference_sha256'])
            inputs.append(dict(reference=c['reference_relative'],sha256=c['reference_sha256']))
        clean,refrel,refhash=refs[g];need((refrel,refhash)==(c['reference_relative'],c['reference_sha256']),'group reference differs')
        for method in METHODS:
            name=c['name']+'-'+method;folder='cases/'+name+'/';prefix='control/'+name
            rp=checked(out,folder+'receipt.json');r=read(rp);ex=read(checked(out,prefix+'.execution.json'))
            compare.equal(ex,dict(actual_exit=0,raw_wait_status=0,hard_timeout=False,receipt_sha256=sha(rp)),'successful execution')
            compare.equal(execution[name],dict(case=name,**ex),'study execution linkage')
            for suffix,key in [('.stdout.log','stdout_sha256'),('.stderr.log','stderr_sha256'),('.command.private.json','command_file_sha256')]:checked(out,prefix+suffix,ex[key])
            compare.equal(r,dict(status='completed',identity=c,method=method,source_commit=freeze['source_commit'],freeze_commit=started['freeze_commit'],freeze_sha256=fhash),'receipt identity')
            compare.equal(r['source_hashes'],{k:v for k,v in freeze['files'].items() if k.startswith('scripts/')},'receipt source SHA')
            fit=r['fit'];compare.equal(fit,dict(status='completed',failure=None,parameter_count=5313,clean_reference_accessed=False,inference_calls=0,initialization_seed=620000+100*(index//2)+10*(index%2)+1,numeric_threads=4,interop_threads=1),'fit integrity')
            initial.setdefault(index,fit['initial_weight_sha256']);need(initial[index]==fit['initial_weight_sha256'],'unequal initialization')
            need(set(r['endpoints'])==set(ENDPOINTS) and set(r['files'])=={e+s for e in ENDPOINTS for s in ('-prediction.npy','-weights.npz')},'endpoint files')
            for f,h in r['files'].items():checked(out,folder+f,h)
            receipts.append(dict(case=name,receipt_sha256=sha(rp)))
            for e in ENDPOINTS:
                row=r['endpoints'][e];t=row['training'];compare.equal(t,fit['endpoint_receipts'][e],'endpoint fit linkage')
                need(t['valid'] is True and type(t['step']) is int and t['step']>0,'invalid endpoint');elapsed=positive(t['training_elapsed_seconds'])
                need(t['step']==500 if e=='updates500' else 5<=t['update_completion_elapsed_seconds']<=elapsed<=5.05,'endpoint trigger')
                if e=='updates500':
                    h={k:t['stream_sha256'][k] for k in ('base_indices','gaussian_prefix','step_seed_tuple')};streams.setdefault(index,h);need(h==streams[index],'500-step streams differ')
                predfile=checked(out,folder+e+'-prediction.npy',r['files'][e+'-prediction.npy']);pred=np.load(predfile,allow_pickle=False)
                need(pred.shape==(122,122) and pred.dtype==np.float32 and np.isfinite(pred).all(),'saved prediction')
                inference_binding(compare,row,fit,index,pred,freeze['files']['scripts/antithetic_training_inference.py'])
                calc=metric(clean,pred);cost,algorithm=allocated(ex,r,e);compare.equal(row['algorithm_pipeline_seconds'],algorithm,'algorithm timing')
                compare.equal(bindings[name,e],dict(case=name,endpoint=e,receipt_sha256=sha(rp),reference_sha256=refhash,prediction_file_sha256=sha(predfile)),'score binding')
                rebuilt=dict(group=g,seed=s,method=method,endpoint=e,status='completed',inference_mc_samples=50,updates=t['step'],training_seconds=elapsed,pipeline_seconds=cost,score=calc)
                compare.equal(published[g,s,method,e],rebuilt,'recomputed endpoint');rows.append(rebuilt)
    expected=reduce_rows(rows);compare.equal(scores,expected,'aggregate');need(len(published)==len(rows)==64,'complete recomputation')
    return dict(status='PASS',scope='Independent saved-array arithmetic and internal evidence links; external canonical-commit authenticity/tool observation remains separate.',
        checker_sha256=sha(Path(__file__)),scores_sha256=sha(scores_path),checks=compare.checks,numeric_checks=compare.numeric,max_absolute_difference=compare.max_absolute,
        rtol=RTOL,atol=ATOL,decision_recomputed=expected['status'],reference_files=inputs,trajectory_receipts=receipts,prediction_arrays=64,reference_arrays=4,model_calls=0,production_imports=False,raw_arrays_exported=False)

def self_test():
    checks=0;c=np.zeros((8,8),np.float32);p=np.tile(np.array([-2,2],np.float32),(8,4));score=metric(c,p)
    need(score['sse']==256 and score['first_difference']['combined']['mse']==8 and len(score['blocks'])==64,'analytic metric');checks+=1
    large=metric(np.zeros((122,122),np.float32),np.ones((122,122),np.float32));need(sum(b['count'] for b in large['blocks'])==14884 and large['blocks'][0]['count']==256 and large['blocks'][-1]['count']==225,'uneven blocks');checks+=1
    rows=[dict(group=g,seed=s,method=m,endpoint=e,pipeline_seconds=10.,score=metric(c,p*(.5 if m==METHODS[2] else 1.))) for e in ENDPOINTS for g in GROUPS for s in SEEDS for m in METHODS]
    need(reduce_rows(rows)['gates']['all_passed'],'complete good screen');checks+=1
    for r in rows:
        if r['method']==METHODS[2] and r['endpoint']=='wall5s':r['pipeline_seconds']=11.001
    need(not reduce_rows(rows)['gates']['wall5s_pipeline_ratio_le_1_10'],'cost gate');checks+=1
    for r in rows:
        r['pipeline_seconds']=10.
        if r['method']==METHODS[2] and r['endpoint']=='wall5s':r['score']=metric(c,p*2)
    need(reduce_rows(rows)['status']=='stop_this_candidate','two endpoints separate');checks+=1
    for r in rows:
        if r['method']==METHODS[2]:r['score']=metric(c,c)
    need(reduce_rows(rows)['status']=='not_evaluable','undefined gain');checks+=1
    for bad in (rows[:-1],rows+[rows[0]]):
        try:reduce_rows(bad)
        except ValueError:checks+=1
        else:raise AssertionError('bad grid accepted')
    compare=Compare();compare.equal(1.,1.+1e-13,'tolerance')
    try:compare.equal('a'*64,'b'*64,'identity')
    except ValueError:checks+=1
    else:raise AssertionError('identity mismatch accepted')
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/'evidence').write_bytes(b'fixture');digest=sha(root/'evidence');checked(root,'evidence',digest);(root/'evidence').write_bytes(b'changed')
        try:checked(root,'evidence',digest)
        except ValueError:checks+=1
        else:raise AssertionError('changed bytes accepted')
    # Equal-PSNR control ties use the declared order; FD uses pooled means, not mean ratios.
    for r in rows:
        r['score']=metric(c,p*(.5 if r['method']==METHODS[2] else 1.))
        fd=(20. if r['method']==METHODS[2] else 1.) if r['group']=='d1' else (100. if r['method']==METHODS[2] else 1000.)
        r['score']['first_difference']['combined']['mse']=fd
    summary=reduce_rows(rows)
    need(all(v['oracle_best_control']=='independent' for v in summary['endpoints']['updates500']['pairs']) and summary['gates']['all_passed'],'oracle ties and pooled FD');checks+=1
    # Eight costs are aggregated before the ratio; a single expensive case may pass overall.
    for r in rows:
        if r['method']==METHODS[2] and r['endpoint']=='wall5s':r['pipeline_seconds']=17. if (r['group'],r['seed'])==('d1',0) else 10.
    need(reduce_rows(rows)['gates']['wall5s_pipeline_ratio_le_1_10'],'aggregate cost ratio');checks+=1
    mock=dict(input_seconds=.5,fit=dict(training_wall_seconds=8.,shared_setup_seconds=.25),endpoints={
        'updates500':dict(training=dict(training_elapsed_seconds=4.,update_completion_elapsed_seconds=3.99),weights_write_seconds=.1,restore_infer_save_seconds=2.),
        'wall5s':dict(training=dict(training_elapsed_seconds=5.01,update_completion_elapsed_seconds=5.),weights_write_seconds=.2,restore_infer_save_seconds=3.)})
    cost,algorithm=allocated(dict(wall_seconds=20.),mock,'updates500');compare.equal(cost,12.8,'allocated cost');compare.equal(algorithm,6.85,'algorithm cost');checks+=1
    mock['fit']['training_wall_seconds']=3.
    try:allocated(dict(wall_seconds=20.),mock,'updates500')
    except ValueError:checks+=1
    else:raise AssertionError('negative later training accepted')
    bad=p.copy();bad[0,0]=np.nan
    try:metric(c,bad)
    except ValueError:checks+=1
    else:raise AssertionError('nonfinite pixels accepted')
    need(array_identity(p)!=hashlib.sha256(p.tobytes()).hexdigest() and array_identity(p)!=array_identity(p.reshape(4,16)),'array identity includes geometry');checks+=1
    return dict(status='PASS',self_test_groups=checks,scope='Artificial arrays, summaries and temporary identity fixture only; no scientific file read.',checker_sha256=sha(Path(__file__)))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--self-test',action='store_true')
    for name in ('repo','output-root','cache','report'):ap.add_argument('--'+name,type=Path)
    args=ap.parse_args();start=time.monotonic()
    if not args.self_test and not all((args.repo,args.output_root,args.cache,args.report)):ap.error('scientific replay requires --repo --output-root --cache --report')
    try:result=self_test() if args.self_test else replay(args.repo,args.cache,args.output_root)
    except Exception as exc:result=dict(status='FAIL',error_type=type(exc).__name__,message=str(exc),checker_sha256=sha(Path(__file__)))
    result.update(elapsed_seconds=time.monotonic()-start,numpy_version=np.__version__,python_version=sys.version.split()[0])
    if args.report:
        with args.report.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,sort_keys=True,allow_nan=False));return 0 if result['status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
