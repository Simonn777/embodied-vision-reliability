"""Prospective one-shot measurement parent; real execution requires reviewed receipt."""
import argparse,datetime,hashlib,json,os,resource,subprocess,time
from pathlib import Path


def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.cache.is_absolute()or a.output.is_absolute()or not a.output.resolve().is_relative_to(Path.cwd()):raise ValueError('relative cache and contained relative output required')
 a.output.mkdir(parents=True,exist_ok=False)
 command=['timeout','900','python','scripts/audit_0003_annotation_consistency_v1.py','--cache',str(a.cache),'--output',str(a.output/'audit')]
 env={'PATH':os.environ['PATH'],'PYTHONPATH':'src','GIT_NO_LAZY_FETCH':'1'}
 for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[name]='2'
 affinity=sorted(os.sched_getaffinity(0))[:2]
 if len(affinity)!=2:raise ValueError('two slots required')
 def limit():resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3));os.sched_setaffinity(0,affinity)
 h=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
 receipt=Path('protocols/0003-annotation-consistency-freeze-v1.json')
 bound=json.loads(receipt.read_text())['inputs_sha256']if receipt.is_file()else{}
 source={name:h(name)for name in bound};source[str(Path(__file__).resolve().relative_to(Path.cwd()))]=h(__file__)
 r={'state':'STARTING','command':command,'source_sha256':source,'receipt_sha256':h(receipt)if receipt.is_file()else None,'execution_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'environment':{**env,'PATH':'inherited runtime PATH value omitted publicly'},'affinity':affinity,'AS_bytes':6*1024**3,'timeout_seconds':900,'child_gate':'unmodified complete published gate runs before numeric access','observed_exit_code':None,'raw_wait_status':None,'boundary':'one actual parent wait4; parent has no further observed supervisor exit receipt'}
 with(a.output/'pre-execution.json').open('x')as f:json.dump(r,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 start=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.perf_counter();code=1
 try:
  with(a.output/'stdout.log').open('xb')as out,(a.output/'stderr.log').open('xb')as err:
   child=subprocess.Popen(command,env=env,stdout=out,stderr=err,preexec_fn=limit)
   _,status,u=os.wait4(child.pid,0);code=os.waitstatus_to_exitcode(status);child.returncode=code
  r.update(observed_exit_code=code,raw_wait_status=status,peak_rss_kib=u.ru_maxrss,CPU_user_seconds=u.ru_utime,CPU_system_seconds=u.ru_stime)
 except Exception as exc:r['measurement_error_type']=type(exc).__name__
 finally:
  r.update(state='COMPLETE',started_utc=start,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.perf_counter()-t,logs_sha256={s:h(a.output/(s+'.log'))for s in ['stdout','stderr']if(a.output/(s+'.log')).exists()})
  with(a.output/'execution.json').open('x')as f:f.write(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r,indent=2));raise SystemExit(code)
if __name__=='__main__':main()
