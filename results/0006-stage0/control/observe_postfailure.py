"""Actual wait4 record for one saved-output review per phase."""
import datetime,hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
C=Path(__file__).parent;B=C.parents[1];PY='/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 mode=sys.argv[1];assert mode=='postfailure';prefix='postfailure-review';assert not(C/(prefix+'-execution.json')).exists()
 os.sched_setaffinity(0,[0,1,2,3]);resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
 env={'PATH':os.environ['PATH'],'PYTHONPATH':str(B/'bootstapir-cache/runtime-fixed'),'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1'}
 for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='4'
 src=C/'check_postfailure_0006.py';r={'command':['timeout','--kill-after=5s','900',PY,str(src),mode],'source_sha256':sha(src),'observer_sha256':sha(__file__),'environment':{k:v for k,v in env.items()if k!='PATH'},'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'boundary':'Parent observes checker once; no further durable observer of the parent itself.'};t=time.perf_counter()
 with(C/(prefix+'.stdout.log')).open('xb')as out,(C/(prefix+'.stderr.log')).open('xb')as err:
  p=subprocess.Popen(r['command'],env=env,stdout=out,stderr=err);_,s,u=os.wait4(p.pid,0);p.returncode=os.waitstatus_to_exitcode(s)
 r.update(exit_code=p.returncode,raw_wait_status=s,wall_seconds=time.perf_counter()-t,CPU_user_seconds=u.ru_utime,CPU_system_seconds=u.ru_stime,peak_rss_kib=u.ru_maxrss,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stdout_sha256=sha(C/(prefix+'.stdout.log')),stderr_sha256=sha(C/(prefix+'.stderr.log')))
 with(C/(prefix+'-execution.json')).open('x')as f:json.dump(r,f,indent=2);f.write('\n')
 print(json.dumps(r,indent=2));print((C/(prefix+'.stderr.log')).read_text());print((C/(prefix+'.stdout.log')).read_text()[:2400]);return p.returncode
if __name__=='__main__':raise SystemExit(main())
