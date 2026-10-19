import os,json,time,hashlib,sys
from pathlib import Path
from datetime import datetime,timezone
c=Path(__file__).parent;r=c.parents[1]/'research-0010-headroom';command=[sys.executable,str(c/'review_saved.py')];started=datetime.now(timezone.utc).isoformat();t=time.monotonic()
env=os.environ.copy();old=json.loads((c.parents[1]/'scratch/0010-capability-control/imports.command.json').read_text());env['PYTHONPATH']=old['PYTHONPATH']
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='4'
env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONNOUSERSITE']='1'
(c/'review.command.json').write_text(json.dumps(dict(argv=command,cwd=str(r),PYTHONPATH=env['PYTHONPATH'],numeric_threads=4,source_sha256=hashlib.sha256((c/'review_saved.py').read_bytes()).hexdigest()),indent=2)+'\n')
pid=os.fork()
if not pid:
 os.chdir(r);os.sched_setaffinity(0,[0,1,2,3]);import resource;resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
 out=os.open(c/'review.stdout.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);err=os.open(c/'review.stderr.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.dup2(out,1);os.dup2(err,2);os.execve(sys.executable,command,env)
_,status,usage=os.wait4(pid,0);exitcode=os.waitstatus_to_exitcode(status)
d=dict(actual_exit=exitcode,raw_wait_status=status,started_utc=started,ended_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-t,user_cpu_seconds=usage.ru_utime,system_cpu_seconds=usage.ru_stime,max_rss_kib=usage.ru_maxrss,logs={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [c/'review.stdout.log',c/'review.stderr.log']},parent_boundary='Parent has no extra wait4 observer; actual tool return is recorded separately.')
(c/'review.execution.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d));print((c/'review.stdout.log').read_text());print((c/'review.stderr.log').read_text());sys.exit(exitcode)
