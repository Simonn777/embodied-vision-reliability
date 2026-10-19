"""Execution observer only; no changes to frozen scientific or preparation code."""
import datetime,hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];REPO=ROOT/'research-0010-capability'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 label,cap=sys.argv[1:3];original=sys.argv[3:];assert original
 assert {0,1,2,3}<=os.sched_getaffinity(0);os.sched_setaffinity(0,{0,1,2,3});resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
 env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONNOUSERSITE']='1';env['PYTHONPATH']=':'.join(str(ROOT/n) for n in ('clip-0010-overlay','tapnextpp-cache/overlay','bootstapir-cache/runtime-fixed','bootstapir-cache/runtime','tapnextpp-recovery/source'))
 for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='4'
 command=original if cap=='frozen' else ['timeout','--kill-after=5s',cap,*original]
 save(HERE/(label+'.command.json'),{'argv':command,'cwd':str(REPO),'PYTHONPATH':env['PYTHONPATH'],'numeric_threads':4,'affinity':[0,1,2,3],'AS_bytes':8*1024**3,'deadline':'Unmodified frozen CLI supervisor: wheels/assets900s, probe300s' if cap=='frozen' else int(cap),'normal_runtime_TLS_proxy_settings_inherited_without_values_recorded':True})
 start=datetime.datetime.now(datetime.timezone.utc).isoformat();clock=time.monotonic()
 with (HERE/(label+'.stdout.log')).open('xb') as out,(HERE/(label+'.stderr.log')).open('xb') as err:
  child=subprocess.Popen(command,cwd=REPO,env=env,stdout=out,stderr=err,start_new_session=True);_,status,u=os.wait4(child.pid,0);child.returncode=os.waitstatus_to_exitcode(status);out.flush();err.flush();os.fsync(out.fileno());os.fsync(err.fileno())
 d={'label':label,'actual_exit':child.returncode,'raw_wait_status':status,'utc_started':start,'utc_completed':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_seconds':time.monotonic()-clock,'user_cpu_seconds':u.ru_utime,'system_cpu_seconds':u.ru_stime,'peak_rss_kib':u.ru_maxrss,'capture_source_sha256':sha(__file__),'command_sha256':sha(HERE/(label+'.command.json')),'stdout_sha256':sha(HERE/(label+'.stdout.log')),'stderr_sha256':sha(HERE/(label+'.stderr.log')),'source_main':'e9e7c7609e53e1bc1b3228096f9b36008cb78d6b','parent_boundary':'Actual wait4 of CLI/timeout child; outer parent itself has no recursive observer.'}
 save(HERE/(label+'.execution.json'),d);print(json.dumps(d,sort_keys=True));return child.returncode
if __name__=='__main__':raise SystemExit(main())
