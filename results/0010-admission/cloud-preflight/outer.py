"""Single bounded parent observer for the preflight, not a scientific runner."""
import datetime,hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 cpus=sorted(os.sched_getaffinity(0))[:4];assert len(cpus)==4;os.sched_setaffinity(0,cpus);resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
 env={'PATH':os.environ['PATH'],'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','PYTHONPATH':':'.join(str(ROOT/x) for x in ('tapnextpp-cache/overlay','bootstapir-cache/runtime-fixed','bootstapir-cache/runtime','tapnextpp-recovery/source'))}
 for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='4'
 # Only safe presence booleans of inherited networking settings are passed, not values.
 presence={k:bool(os.environ.get(k)) for k in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY','http_proxy','https_proxy','all_proxy','no_proxy','SSL_CERT_FILE','REQUESTS_CA_BUNDLE')}
 cmd=['timeout','--kill-after=5s','120',sys.executable,str(HERE/'preflight.py')]
 with (HERE/'command.json').open('x') as f:json.dump({'argv':cmd,'cwd':str(ROOT),'environment':env,'inherited_network_settings_presence_only':presence,'network_setting_values_passed':False},f,indent=2);f.write('\n')
 start=datetime.datetime.now(datetime.timezone.utc).isoformat();clock=time.monotonic()
 with (HERE/'stdout.log').open('xb') as out,(HERE/'stderr.log').open('xb') as err:
  p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=True)
  _,status,u=os.wait4(p.pid,0);p.returncode=os.waitstatus_to_exitcode(status)
  out.flush();err.flush();os.fsync(out.fileno());os.fsync(err.fileno())
 result={'actual_exit':p.returncode,'raw_wait_status':status,'utc_started':start,'utc_completed':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_seconds':time.monotonic()-clock,'peak_rss_kib':u.ru_maxrss,'user_cpu_seconds':u.ru_utime,'system_cpu_seconds':u.ru_stime,'affinity':cpus,'address_space_bytes':8*1024**3,'command_sha256':sha(HERE/'command.json'),'stdout_sha256':sha(HERE/'stdout.log'),'stderr_sha256':sha(HERE/'stderr.log'),'outer_source_sha256':sha(__file__),'parent_boundary':'wait4 observes timeout/preflight process; outer itself has no recursive observer'}
 with (HERE/'execution.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result,sort_keys=True));return p.returncode
if __name__=='__main__':raise SystemExit(main())
