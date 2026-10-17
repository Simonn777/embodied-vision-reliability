"""Actual single parent observation for the one default-route diagnosis."""
import datetime,hashlib,json,os,platform,resource,subprocess,time
from pathlib import Path
B=Path('results/0003-network-followup-20261017');B.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
(B/'diagnosis-source.py').write_bytes(Path('scratch/diagnose_0003_route.py').read_bytes());(B/'parent-source.py').write_bytes(Path(__file__).read_bytes())
def limit():
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3));os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
r={'command':['timeout','15','python','scratch/diagnose_0003_route.py'],'environment':'unchanged runtime defaults inherited; no values/credentials logged','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_source_sha256':sha(__file__),'child_source_sha256':sha('scratch/diagnose_0003_route.py'),'AS_bytes':6*1024**3,'affinity':sorted(os.sched_getaffinity(0))[:2],'python':platform.python_version(),'platform':platform.platform(),'source_commit':'b14e759b03ae992682c95b53466a9d220c4e86a1','boundary':'parent observes one child; no further supervisor receipt for this parent'}
start=time.perf_counter()
with(B/'stdout.log').open('x')as out,(B/'stderr.log').open('x')as err:
 p=subprocess.Popen(r['command'],preexec_fn=limit,stdout=out,stderr=err);_,status,u=os.wait4(p.pid,0);p.returncode=os.waitstatus_to_exitcode(status)
r.update(ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start,child_exit_code=p.returncode,raw_wait_status=status,peak_rss_kib=u.ru_maxrss,cpu_user_seconds=u.ru_utime,cpu_system_seconds=u.ru_stime,log_sha256={s:sha(B/(s+'.log'))for s in ['stdout','stderr']})
(B/'parent.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
