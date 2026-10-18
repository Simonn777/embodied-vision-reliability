"""Actual outer wait4 record; no network operation in this parent."""
import datetime,hashlib,json,os,platform,resource,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
    cpus=sorted(os.sched_getaffinity(0))[:4];assert len(cpus)==4
    os.sched_setaffinity(0,cpus);resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3));env=os.environ.copy()
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='4'
    argv=['timeout','--kill-after=5s','120',sys.executable,str(HERE/'hpatches_directory.py')]
    write(HERE/'command.json',{'argv':argv,'cwd':str(HERE),'affinity':cpus,'numeric_threads':4,'address_space_bytes':8*1024**3,'hard_timeout_seconds':120,'network':'Unchanged approved runtime default proxy/CA configuration; no credential values recorded.'})
    start=datetime.datetime.now(datetime.timezone.utc).isoformat();clock=time.monotonic()
    with (HERE/'stdout.log').open('xb') as out,(HERE/'stderr.log').open('xb') as err:
        child=subprocess.Popen(argv,cwd=HERE,env=env,stdout=out,stderr=err,start_new_session=True)
        _,status,usage=os.wait4(child.pid,0);child.returncode=os.waitstatus_to_exitcode(status)
        out.flush();err.flush();os.fsync(out.fileno());os.fsync(err.fileno())
    record={'actual_exit':child.returncode,'raw_wait_status':status,'utc_started':start,'utc_completed':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_seconds':time.monotonic()-clock,'peak_rss_kib':usage.ru_maxrss,'user_cpu_seconds':usage.ru_utime,'system_cpu_seconds':usage.ru_stime,'python':sys.version,'python_executable':sys.executable,'platform':platform.platform(),'cpu_affinity':cpus,'numeric_threads':4,'address_space_bytes':8*1024**3,'command_sha256':sha(HERE/'command.json'),'stdout_sha256':sha(HERE/'stdout.log'),'stderr_sha256':sha(HERE/'stderr.log'),'operation_source_sha256':sha(HERE/'hpatches_directory.py'),'capture_source_sha256':sha(__file__),'parent_scope':'Actual wait4 of timeout child; no additional recursive outer observer.'}
    write(HERE/'execution.json',record);print(json.dumps(record,sort_keys=True),flush=True)
    return child.returncode
if __name__=='__main__':raise SystemExit(main())
