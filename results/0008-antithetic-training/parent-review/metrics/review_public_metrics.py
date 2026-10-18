#!/usr/bin/env python3
"""Read-only 0008 public-summary replay. Stdlib only; no private arrays or imports.
This checks algebra and evidence links, not pixel-level correctness or observed cloud tools.
"""
import argparse, hashlib, json, math, re, subprocess, sys, time
from pathlib import Path

PUBLICATION='70bc57fd83634a8df66aecf072ea9b526e3af043'
PREFIX='results/0008-antithetic-training/'
GROUPS=('d1','d2','d3','d4');SEEDS=(0,1)
METHODS=('independent','random_antithetic','mean_antithetic','same_patch')
CONTROLS=('independent','random_antithetic','same_patch');ENDS=('updates500','wall5s')
RTOL=1e-10;ATOL=1e-12

def digest(data):return hashlib.sha256(data).hexdigest()
def pairs(items):
    result={}
    for k,v in items:
        if k in result:raise ValueError('duplicate JSON key')
        result[k]=v
    return result

def decode(data):return json.loads(data,object_pairs_hook=pairs,parse_constant=lambda v:(_ for _ in ()).throw(ValueError('nonfinite JSON')))

class Audit:
    def __init__(self,repo,commit):self.repo=repo;self.commit=commit;self.checks=0;self.numeric=0;self.max_error=0.;self.file_hashes={}
    def git(self,*args):return subprocess.check_output(['git','-C',str(self.repo),*args])
    def blob(self,path,revision=None):return self.git('show',(revision or self.commit)+':'+path)
    def read(self,path):
        data=self.blob(path);self.file_hashes[path]=digest(data);return decode(data)
    def public(self,path):return self.read(PREFIX+path)
    def require(self,ok,label):
        self.checks+=1
        if not ok:raise ValueError(label)
    def equal(self,actual,expected,label):
        self.checks+=1
        if isinstance(expected,dict):
            self.require(isinstance(actual,dict) and set(expected)<=set(actual),'mapping '+label)
            for k,v in expected.items():self.equal(actual[k],v,label+'.'+k)
        elif isinstance(expected,list):
            self.require(isinstance(actual,list) and len(actual)==len(expected),'list '+label)
            for k,v in enumerate(expected):self.equal(actual[k],v,label+'.'+str(k))
        elif isinstance(expected,float):
            self.require(type(actual) in (int,float) and math.isfinite(actual) and math.isfinite(expected),'finite '+label)
            self.numeric+=1;self.max_error=max(self.max_error,abs(actual-expected))
            self.require(math.isclose(actual,expected,rel_tol=RTOL,abs_tol=ATOL),'numeric '+label)
        else:self.require(type(actual) is type(expected) and actual==expected,'identity '+label)
    def positive(self,value,label):self.require(type(value) in (int,float) and math.isfinite(value) and value>0,'positive '+label);return value


def measure(a,s,label):
    a.equal({k:s[k] for k in ('shape','data_range','clipped','count')},dict(shape=[122,122],data_range=1,clipped=False,count=14884),label+' geometry')
    a.require(len(s['blocks'])==64,'64 blocks '+label);ss=[];counts=[]
    # Construct borders by integer quotient/remainder, independently of array_split.
    edges=[15*i+min(i,2) for i in range(9)]
    for k,b in enumerate(s['blocks']):
        i,j=divmod(k,8);lo,hi=edges[i:i+2];left,right=edges[j:j+2]
        a.equal(b,dict(row=i,col=j,row_start=lo,row_stop=hi,col_start=left,col_stop=right,count=(hi-lo)*(right-left)),label+' block')
        a.require(type(b['sse']) in (int,float) and math.isfinite(b['sse']) and b['sse']>=0,'block SSE '+label)
        ss.append(b['sse']);counts.append(b['count'])
    a.equal(sum(counts),14884,label+' block pixel count');sse=math.fsum(ss);a.positive(sse,label+' SSE')
    mse=sse/14884;psnr=10*(math.log10(14884)-math.log10(sse))
    a.equal(s['sse'],sse,label+' SSE');a.equal(s['mse'],mse,label+' MSE');a.equal(s['psnr_db'],psnr,label+' PSNR')
    fd=s['first_difference'];direction_sse=[]
    for direction in ('horizontal','vertical'):
        item=fd[direction];a.equal(item['count'],14762,label+' FD count');a.positive(item['sse'],label+' FD SSE')
        a.equal(item['mse'],item['sse']/14762,label+' FD MSE');direction_sse.append(item['sse'])
    total=math.fsum(direction_sse);fd_mse=total/29524
    a.equal(fd['combined'],dict(sse=total,count=29524,mse=fd_mse),label+' pooled FD')
    for field in ('clean_sha256_f64le','prediction_sha256_f32le'):a.require(re.fullmatch('[0-9a-f]{64}',s[field]) is not None,'array identity syntax')
    return dict(mse=mse,psnr_db=psnr,fd_mse=fd_mse)


def run(repo,commit):
    a=Audit(repo,commit);a.require(re.fullmatch('[0-9a-f]{40}',commit) is not None,'full publication commit')
    freeze_path='protocols/0008-training-freeze.json';freeze=a.read(freeze_path);fh=digest(a.blob(freeze_path))
    sources=a.public('source-bindings.json');fc=sources['canonical_freeze_commit'];sc=freeze['source_commit']
    a.equal(sources,dict(scientific_source_commit=sc,canonical_sha256=fh,required_inputs=freeze['files']),'source archive')
    a.equal(digest(a.blob(freeze_path,fc)),fh,'canonical freeze bytes');a.equal(digest((repo/freeze_path).read_bytes()),fh,'working freeze unchanged')
    a.require(len(freeze['files'])==13,'13 frozen inputs')
    for path,h in freeze['files'].items():
        for ref in (sc,fc,commit,sources['execution_main']):a.equal(digest(a.blob(path,ref)),h,'frozen source '+path)
        a.equal(digest((repo/path).read_bytes()),h,'working source unchanged '+path)
    cfg=a.read('configs/0008-training.json');scores=a.public('study/scores.json');started=a.public('study/study.started.json');completed=a.public('study/study.completed.json')
    a.equal(cfg['methods'],list(METHODS),'methods');a.equal(cfg['endpoints'],list(ENDS),'endpoints');a.equal(len(cfg['cases']),8,'case count')
    a.equal(scores['bindings'],dict(source_commit=sc,freeze_commit=fc,freeze_sha256=fh),'score freeze')
    a.equal(started,dict(source_commit=sc,freeze_commit=fc,freeze_sha256=fh),'start freeze')
    a.equal(completed,dict(source_commit=sc,freeze_commit=fc,inputs_after=started['inputs']),'complete freeze')
    order=[c['name']+'-'+METHODS[(c['index']+j)%4] for c in cfg['cases'] for j in range(4)]
    a.equal(started['order'],order,'balanced order');a.equal([r['case'] for r in completed['cases']],order,'execution order')
    a.equal(started['inputs'],[dict(relative=c['input_relative'],sha256=c['input_sha256']) for c in cfg['cases']],'pinned declared input identities')
    indexed={};metrics={};costs={};clean_hashes={};bindings={};executions={r['case']:r for r in completed['cases']}
    expected={(g,s,m,e) for g in GROUPS for s in SEEDS for m in METHODS for e in ENDS}
    for row in scores['cases']:
        k=(row['group'],row['seed'],row['method'],row['endpoint']);a.require(k in expected and k not in indexed,'unique64 grid');indexed[k]=row
        metrics[k]=measure(a,row['score'],'/'.join(map(str,k)));a.equal(row,dict(status='completed',inference_mc_samples=50),'row status')
        clean_hashes.setdefault(k[0],row['score']['clean_sha256_f64le']);a.equal(row['score']['clean_sha256_f64le'],clean_hashes[k[0]],'same group clean identity')
    a.require(set(indexed)==expected,'complete64 grid')
    for b in scores['bindings']['endpoints']:
        k=(b['case'],b['endpoint']);a.require(k not in bindings,'unique binding');bindings[k]=b
    a.equal(len(bindings),64,'64 bindings');table=a.public('trajectory-costs.json');table_rows={r['trajectory']:r for r in table['trajectories']}
    a.equal(list(table_rows),order,'cost table order');a.equal(table['source_sha256'],digest(a.blob(PREFIX+'study/study.completed.json')),'cost source')
    updates=0;forwards=0;draws=0;walls=[];captures=[];initial={};streams={};steps_by_method={m:[] for m in METHODS}
    output_hashes=a.public('outer-control/postcheck.json')['saved_output_hashes'];a.equal(len(output_hashes),128,'128 private-output identity claims')
    for c in cfg['cases']:
        g,s=c['group'],c['seed'];a.equal(c,dict(index=(int(g[1])-1)*2+s,name=g+'-colored-'+str(s)),'case geometry')
        for method in METHODS:
            name=c['name']+'-'+method;rp=PREFIX+'trajectories/'+name+'/receipt.json';r=a.read(rp);rh=digest(a.blob(rp))
            ex=a.public('trajectory-control/'+name+'.execution.json');a.equal(ex,dict(actual_exit=0,raw_wait_status=0,hard_timeout=False,receipt_sha256=rh),'execution success')
            a.equal(executions[name],dict(case=name,**ex),'execution link');walls.append(a.positive(ex['wall_seconds'],'outer wall'))
            for suffix,field in (('stdout.log','stdout_sha256'),('stderr.log','stderr_sha256')):a.equal(digest(a.blob(PREFIX+'trajectory-control/'+name+'.'+suffix)),ex[field],'closed log hash')
            a.equal(r,dict(status='completed',identity=c,method=method,source_commit=sc,freeze_commit=fc,freeze_sha256=fh),'trajectory identity')
            a.equal(r['source_hashes'],{p:h for p,h in freeze['files'].items() if p.startswith('scripts/')},'receipt source identities')
            fit=r['fit'];a.equal(fit,dict(status='completed',failure=None,parameter_count=5313,clean_reference_accessed=False,inference_calls=0),'fit')
            count=fit['completed_updates'];a.require(type(count) is int and 500<=count<=100000,'actual update count')
            a.equal(fit['optimizer_steps_executed'],count,'optimizer accounting');a.equal(fit['drawn_batches'],count,'batch accounting');updates+=count
            initial.setdefault(c['index'],fit['initial_weight_sha256']);a.equal(fit['initial_weight_sha256'],initial[c['index']],'matched init')
            a.require(set(r['endpoints'])==set(ENDS),'two correlated endpoints')
            summary=table_rows[name];a.equal(summary,dict(actual_outer_wall_seconds=ex['wall_seconds'],peak_rss_kib=ex['peak_rss_kib'],actual_exit=0,input_seconds=r['input_seconds'],shared_setup_seconds=fit['shared_setup_seconds'],training_wall_seconds=fit['training_wall_seconds'],completed_updates=count),'cost table trajectory')
            for e in ENDS:
                er=r['endpoints'][e];tr=er['training'];inf=er['inference'];key=g,s,method,e;row=indexed[key]
                a.equal(tr,fit['endpoint_receipts'][e],'snapshot linkage');a.equal(tr['valid'],True,'valid snapshot');elapsed=a.positive(tr['training_elapsed_seconds'],'capture time')
                a.require(0<tr['update_completion_elapsed_seconds']<=elapsed<=fit['training_wall_seconds'],'nested clocks')
                if e=='updates500':
                    a.equal(tr['step'],500,'500 updates');st={k:tr['stream_sha256'][k] for k in ('base_indices','gaussian_prefix','step_seed_tuple')};streams.setdefault(c['index'],st);a.equal(st,streams[c['index']],'shared500 streams')
                else:
                    a.require(5<=tr['update_completion_elapsed_seconds']<=elapsed<=5.05,'fixed5s capture');captures.append(elapsed);steps_by_method[method].append(tr['step'])
                a.equal(row,dict(updates=tr['step'],training_seconds=elapsed),'score endpoint clock')
                extra=a.positive(fit['training_wall_seconds'],'training clock')-elapsed;a.require(extra>=0,'nonnegative later training')
                other=math.fsum(a.positive(v['weights_write_seconds'],'other write')+a.positive(v['restore_infer_save_seconds'],'other inference') for k,v in r['endpoints'].items() if k!=e)
                allocated=ex['wall_seconds']-extra-other;algorithm=math.fsum((a.positive(r['input_seconds'],'input'),a.positive(fit['shared_setup_seconds'],'setup'),elapsed,a.positive(er['weights_write_seconds'],'write'),a.positive(er['restore_infer_save_seconds'],'inference')))
                a.require(0<algorithm<=allocated<=ex['wall_seconds'],'allocated outer cost bound');a.equal(er['algorithm_pipeline_seconds'],algorithm,'algorithm timing');a.equal(row['pipeline_seconds'],allocated,'allocated timing');costs[key]=allocated
                a.equal(summary['endpoints'][e],dict(training=tr,weights_write_seconds=er['weights_write_seconds'],restore_infer_save_seconds=er['restore_infer_save_seconds'],algorithm_pipeline_seconds=algorithm,allocated_outer_pipeline_seconds=allocated),'cost table endpoint')
                a.equal(inf,dict(status='completed',failure=None,finite_outputs=True,weight_unchanged=True,mc_samples=50,forward_calls=750,forward_calls_attempted=750,expected_forward_calls=750,standard_normal_draws=36465800,expected_standard_normal_draws=36465800,direct_forward_calls=0,optimizer_steps=0,clean_reference_accessed=False),'inference counters')
                for field in ('weight_content_sha256','weight_before_sha256','weight_after_sha256'):a.equal(inf[field],tr['state_dict_sha256'],'frozen snapshot inference identity')
                forwards+=inf['forward_calls'];draws+=inf['standard_normal_draws']
                a.equal(bindings[name,e],dict(case=name,endpoint=e,receipt_sha256=rh,reference_sha256=c['reference_sha256'],prediction_file_sha256=r['files'][e+'-prediction.npy']),'saved prediction identity claim')
            for filename,h in r['files'].items():a.equal(output_hashes['cases/'+name+'/'+filename],h,'original private-output identity claim')
            a.equal(max(v['training']['step'] for v in r['endpoints'].values()),count,'one trajectory actual steps')
    results={}
    for e in ENDS:
        paired=[]
        for g in GROUPS:
            for s in SEEDS:
                best=min(CONTROLS,key=lambda m:metrics[g,s,m,e]['mse']);b=metrics[g,s,best,e];x=metrics[g,s,'mean_antithetic',e]
                paired.append(dict(group=g,seed=s,oracle_best_control=best,candidate_psnr_db=x['psnr_db'],control_psnr_db=b['psnr_db'],gain_db=x['psnr_db']-b['psnr_db'],candidate_first_difference_mse=x['fd_mse'],control_first_difference_mse=b['fd_mse']))
        means={g:math.fsum(p['gain_db'] for p in paired if p['group']==g)/2 for g in GROUPS};gain=math.fsum(p['gain_db'] for p in paired)/8
        fd_c=math.fsum(p['candidate_first_difference_mse'] for p in paired)/8;fd_b=math.fsum(p['control_first_difference_mse'] for p in paired)/8
        gates=dict(finite_paired_db_gains=True,mean_gain_ge_0_5_db=gain>=.5,at_least3_group_means_gt0=sum(x>0 for x in means.values())>=3,each_group_mean_ge_minus0_2_db=min(means.values())>=-.2,mean_pooled_first_difference_le_1_05=fd_c<=1.05*fd_b)
        result=dict(pairs=paired,mean_gain_db=gain,group_mean_gain_db=means,mean_candidate_first_difference_mse=fd_c,mean_corresponding_control_first_difference_mse=fd_b,gates=gates,pipeline_seconds_by_method={m:math.fsum(costs[g,s,m,e] for g in GROUPS for s in SEEDS) for m in METHODS})
        a.equal(scores['endpoints'][e],result,'recomputed '+e);results[e]=result
    ratio=results['wall5s']['pipeline_seconds_by_method']['mean_antithetic']/results['wall5s']['pipeline_seconds_by_method']['independent'];quality=all(all(e['gates'].values()) for e in results.values());passed=quality and ratio<=1.10;decision='continue_to_novelty_review_only' if passed else 'stop_this_candidate'
    a.equal(scores,dict(status=decision,eligible_for_novelty_review=passed,paper_success=False,development_only=True,case_count=64,trajectory_count=32,oracle_control_order=list(CONTROLS),wall5s_candidate_vs_independent_pipeline_ratio=ratio,gates=dict(complete64_grid=True,both_endpoint_quality_gates_pass=quality,wall5s_pipeline_ratio_le_1_10=ratio<=1.10,all_passed=passed)),'decision')
    outer=a.public('outer-control/scientific-run.execution.json');a.equal(outer,dict(actual_exit=0,raw_wait_status=0),'study outer saved exit')
    a.equal(table['actual_case_outer_wall_sum_seconds'],math.fsum(walls),'actual32 wall sum');a.equal(table['study_outer_wall_seconds'],outer['wall_seconds'],'study actual outer wall')
    a.require(math.fsum(walls)<=completed['wall_seconds']<=outer['wall_seconds'],'nested study clocks')
    cloud=a.public('outer-control/independent-replay.json');a.equal(cloud,dict(status='PASS',checker_sha256=freeze['files']['scripts/check_antithetic_training.py'],scores_sha256=digest(a.blob(PREFIX+'study/scores.json')),decision_recomputed=decision,prediction_arrays=64,reference_arrays=4,model_calls=0,production_imports=False),'separate cloud raw-array replay declaration')
    return dict(status='PASS',scientific_decision=decision,publication_commit=commit,source_commit=sc,freeze_commit=fc,freeze_sha256=fh,frozen_inputs_verified=13,
        checks=a.checks,numeric_checks=a.numeric,max_absolute_difference=a.max_error,rtol=RTOL,atol=ATOL,endpoints=results,wall5s_cost_ratio=ratio,
        counts=dict(trajectories=32,endpoints=64,blocks=4096,actual_optimizer_updates=updates,inference_forward_calls=forwards,inference_normal_scalars=draws),
        timing=dict(actual_case_outer_wall_sum_seconds=math.fsum(walls),actual_study_outer_wall_seconds=outer['wall_seconds'],study_inner_wall_seconds=completed['wall_seconds'],wall_capture_range_seconds=[min(captures),max(captures)],wall_endpoint_update_ranges={m:[min(v),max(v)] for m,v in steps_by_method.items()},cost_interpretation='Two correlated allocations per shared trajectory; neither standalone reruns nor additive consumed compute.'),
        cloud_raw_array_replay_report=dict(checks=cloud['checks'],numeric_checks=cloud['numeric_checks'],max_absolute_difference=cloud['max_absolute_difference'],scope='Archived cloud report only; not a locally repeated raw-array replay.'),
        scope='Independent public block/SSE/count algebra and saved receipt links only. No pixel arrays, model calls, production imports, cloud commands or external tool-status observations.',raw_array_reads=0,model_calls=0,production_imports=False,file_sha256=a.file_hashes)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',required=True,type=Path);p.add_argument('--commit',default=PUBLICATION);p.add_argument('--report',required=True,type=Path);args=p.parse_args();began=time.monotonic()
    try:result=run(args.repo,args.commit)
    except Exception as exc:result=dict(status='FAIL',error_type=type(exc).__name__,message=str(exc))
    result.update(reviewer_sha256=digest(Path(__file__).read_bytes()),python_version=sys.version.split()[0],elapsed_seconds=time.monotonic()-began)
    with args.report.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('endpoints','file_sha256')},sort_keys=True));raise SystemExit(0 if result['status']=='PASS' else 1)
