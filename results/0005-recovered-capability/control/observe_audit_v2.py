"""One parent wait4 observation of each saved-evidence checker, serial only."""
import datetime,hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
C=Path(__file__).parent;PY='/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 mode=sys.argv[1];prefix=mode+'-review-v2';assert not(C/(prefix+'-execution.json')).exists()
 src=C/('audit_preparation_v2.py'if mode in ['dependencies','checkpoint']else 'review_native.py')
 os.sched_setaffinity(0,[0,1,2,3]);resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
 env={'PATH':os.environ['PATH'],'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1'}
 for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='4'
 if mode=='native':env['PYTHONPATH']=str(C.parents[1]/'bootstapir-cache/runtime-fixed')
 rec={'command':['timeout','--kill-after=5s','1800',PY,str(src),mode],'environment':{k:v for k,v in env.items()if k!='PATH'},'source_sha256':sha(src),'observer_source_sha256':sha(__file__),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'boundary':'Parent observes review child once; parent itself has no further durable outer wait receipt.'};t=time.perf_counter()
 with(C/(prefix+'.stdout.log')).open('xb')as out,(C/(prefix+'.stderr.log')).open('xb')as err:
  p=subprocess.Popen(rec['command'],env=env,stdout=out,stderr=err);_,s,u=os.wait4(p.pid,0);p.returncode=os.waitstatus_to_exitcode(s)
 rec.update(exit_code=p.returncode,raw_wait_status=s,wall_seconds=time.perf_counter()-t,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),CPU_user_seconds=u.ru_utime,CPU_system_seconds=u.ru_stime,peak_rss_kib=u.ru_maxrss,stdout_sha256=sha(C/(prefix+'.stdout.log')),stderr_sha256=sha(C/(prefix+'.stderr.log')))
 with(C/(prefix+'-execution.json')).open('x')as f:json.dump(rec,f,indent=2);f.write('\n')
 print(json.dumps(rec,indent=2));print((C/(prefix+'.stderr.log')).read_text());print((C/(prefix+'.stdout.log')).read_text()[:1800]);return p.returncode
if __name__=='__main__':raise SystemExit(main())

