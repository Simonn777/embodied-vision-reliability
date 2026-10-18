"""Observe wrapper exit once; child timeout is enforced by the wrapper's timeout."""
import datetime,hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
OUT=Path('results/0004-stage0')
mode=sys.argv[1];target=OUT/(mode+'-outer.json')
if target.exists():raise ValueError('exclusive supervisor receipt')
command=[sys.executable,str(OUT/'launch_wrapper.py'),mode]
record={'state':'STARTING','mode':mode,'command':command,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'observed_wrapper_exit':None,'raw_wait_status':None,'boundary':'one parent observation of wrapper; this supervisor has no further durable parent wait4 receipt'}
target.write_text(json.dumps(record,indent=2)+'\n');t=time.perf_counter()
with(OUT/(mode+'-wrapper.stdout.log')).open('xb')as out,(OUT/(mode+'-wrapper.stderr.log')).open('xb')as err:
 child=subprocess.Popen(command,stdout=out,stderr=err)
 _,status,u=os.wait4(child.pid,0);child.returncode=os.waitstatus_to_exitcode(status)
record.update(state='FINISHED',ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.perf_counter()-t,observed_wrapper_exit=child.returncode,raw_wait_status=status,peak_rss_kib=u.ru_maxrss,CPU_user_seconds=u.ru_utime,CPU_system_seconds=u.ru_stime,logs_sha256={s:hashlib.sha256((OUT/(mode+'-wrapper.'+s+'.log')).read_bytes()).hexdigest()for s in ['stdout','stderr']})
temp=target.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(target)
print(json.dumps(record,indent=2));raise SystemExit(child.returncode)
