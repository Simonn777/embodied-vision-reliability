"""One parent wait4 record per command; no scientific implementation."""
import datetime, hashlib, json, os, platform, resource, subprocess, sys, time
from pathlib import Path

CONTROL = Path(__file__).resolve().parent
REPO = CONTROL.parents[1] / 'research-0007-gradient'
RUNTIME = CONTROL.parents[1] / 'bootstapir-cache/runtime-fixed'
PYTHON = '<PYTHON>'
THREADS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def write(p,v):
    with Path(p).open('x') as f:
        json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
    label=sys.argv[1]; seconds=int(sys.argv[2]); command=sys.argv[3:]
    affinity=sorted(os.sched_getaffinity(0))[:4]
    assert len(affinity)==4
    os.sched_setaffinity(0,affinity)
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    env=os.environ.copy()
    env['PYTHONPATH']=str(RUNTIME)
    for k in THREADS:env[k]='4'
    argv=['timeout','--kill-after=5s',str(seconds),*command]
    cmd=CONTROL/(label+'.command.private.json')
    write(cmd,{'argv':argv,'cwd':str(REPO),'PYTHONPATH':env['PYTHONPATH'],'threads':{k:env[k] for k in THREADS},'affinity':affinity,'address_space_bytes':8*1024**3})
    started=datetime.datetime.now(datetime.timezone.utc).isoformat();before=time.monotonic()
    with (CONTROL/(label+'.stdout.log')).open('xb') as out,(CONTROL/(label+'.stderr.log')).open('xb') as err:
        child=subprocess.Popen(argv,cwd=REPO,env=env,stdout=out,stderr=err,start_new_session=True)
        _,status,usage=os.wait4(child.pid,0)
        child.returncode=os.waitstatus_to_exitcode(status)
        out.flush();err.flush();os.fsync(out.fileno());os.fsync(err.fileno())
    result={'label':label,'utc_started':started,'utc_completed':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_exit':child.returncode,'raw_wait_status':status,'wall_seconds':time.monotonic()-before,'peak_rss_kib':usage.ru_maxrss,'user_cpu_seconds':usage.ru_utime,'system_cpu_seconds':usage.ru_stime,'command_file_sha256':sha(cmd),'stdout_sha256':sha(CONTROL/(label+'.stdout.log')),'stderr_sha256':sha(CONTROL/(label+'.stderr.log')),'outer_source_sha256':sha(__file__),'execution_main':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'python':sys.version,'platform':platform.platform(),'cpu_affinity':affinity,'address_space_bytes':8*1024**3,'hard_timeout_seconds':seconds,'outer_parent_exit_observation':'No additional supervisor observes this parent; record is actual wait4 of the timeout child.'}
    write(CONTROL/(label+'.execution.json'),result)
    print(json.dumps(result,sort_keys=True),flush=True)
    return child.returncode if child.returncode>=0 else 128-child.returncode
if __name__=='__main__':raise SystemExit(main())
