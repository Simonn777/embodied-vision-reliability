"""Actual parent wait4 records; science deadline remains in frozen supervisor."""
import datetime,hashlib,json,os,platform,resource,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];REPO=ROOT/'research-0009-stage0';RUNTIME=ROOT/'bootstapir-cache/runtime-fixed'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
    label,cap=sys.argv[1:3];original=sys.argv[3:];assert original
    cpus=sorted(os.sched_getaffinity(0))[:4];assert len(cpus)==4
    os.sched_setaffinity(0,cpus);resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    env=os.environ.copy();env['PYTHONPATH']=str(RUNTIME)+':'+str(REPO/'src');env['PYTHONDONTWRITEBYTECODE']='1';env['TMPDIR']=str(HERE/'tmp')
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='4'
    # Do not impose a competing scientific deadline before frozen wrapper cleanup.
    command=original if cap=='frozen900' else ['timeout','--kill-after=5s',cap,*original]
    write(HERE/(label+'.command.private.json'),{'argv':command,'cwd':str(REPO),'PYTHONPATH':env['PYTHONPATH'],'PYTHONDONTWRITEBYTECODE':'1','TMPDIR':env['TMPDIR'],'cpu_affinity':cpus,'numeric_threads':4,'address_space_bytes':8*1024**3,'deadline':'900s enforced by unmodified frozen supervisor' if cap=='frozen900' else int(cap)})
    started=datetime.datetime.now(datetime.timezone.utc).isoformat();clock=time.monotonic()
    with (HERE/(label+'.stdout.log')).open('xb') as out,(HERE/(label+'.stderr.log')).open('xb') as err:
        child=subprocess.Popen(command,cwd=REPO,env=env,stdout=out,stderr=err,start_new_session=True)
        _,status,usage=os.wait4(child.pid,0);child.returncode=os.waitstatus_to_exitcode(status)
        out.flush();err.flush();os.fsync(out.fileno());os.fsync(err.fileno())
    r={'label':label,'actual_exit':child.returncode,'raw_wait_status':status,'utc_started':started,'utc_completed':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_seconds':time.monotonic()-clock,'user_cpu_seconds':usage.ru_utime,'system_cpu_seconds':usage.ru_stime,'peak_rss_kib':usage.ru_maxrss,'cpu_affinity':cpus,'numeric_threads':4,'address_space_bytes':8*1024**3,'python':sys.version,'platform':platform.platform(),'command_sha256':sha(HERE/(label+'.command.private.json')),'stdout_sha256':sha(HERE/(label+'.stdout.log')),'stderr_sha256':sha(HERE/(label+'.stderr.log')),'capture_source_sha256':sha(__file__),'parent_scope':'Observed wait4 of the wrapper/CLI child. Parent itself has no additional recursive exit observer; frozen science supervisor retains its own child evidence.'}
    write(HERE/(label+'.execution.json'),r);print(json.dumps(r,sort_keys=True),flush=True)
    return child.returncode if child.returncode>=0 else 128-child.returncode
if __name__=='__main__':raise SystemExit(main())
