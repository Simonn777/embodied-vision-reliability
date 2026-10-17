"""Observe the g-028 wrapper exit independently; never imports model or labels."""
import datetime,hashlib,json,os,resource,shutil,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'results/0002-pilot-execution/g-028-20261017-supervision'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def limits():
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3))
 slots=sorted(os.sched_getaffinity(0));assert len(slots)>=2
 os.sched_setaffinity(0,slots[:2])
def main():
 os.chdir(ROOT);AUDIT.mkdir(parents=True,exist_ok=False)
 (AUDIT/'supervisor-source.py').write_bytes(Path(__file__).read_bytes())
 env={'PATH':os.environ['PATH']}
 command=['env','-i','PATH='+env['PATH'],'python','scratch/run_g028.py']
 rec={'status':'starting','started_at_utc':utc(),'command':'env -i PATH="$PATH" python scratch/run_g028.py','wrapper_source_sha256':sha('scratch/run_g028.py'),'supervisor_source_sha256':sha(__file__),'observation_scope':'direct parent wait4 observation of wrapper; distinct from wrapper wait4 observations of timeout-wrapped preflight/inference children','phases_external_timeout_seconds':900}
 def save():(AUDIT/'supervisor.json').write_text(json.dumps(rec,indent=2)+'\n')
 save();before=time.perf_counter()
 try:
  with (AUDIT/'wrapper.stdout.log').open('x') as out,(AUDIT/'wrapper.stderr.log').open('x') as err:
   child=subprocess.Popen(command,stdout=out,stderr=err,env=env,preexec_fn=limits)
   pid,status,usage=os.wait4(child.pid,0);child.returncode=os.waitstatus_to_exitcode(status)
  rec.update(wrapper_exit_code_observed_by_parent=child.returncode,wrapper_raw_wait_status=status,wrapper_peak_rss_kib=usage.ru_maxrss,wrapper_cpu_user_seconds=usage.ru_utime,wrapper_cpu_system_seconds=usage.ru_stime,status='completed' if child.returncode==0 else 'failed')
 except Exception as e:
  rec.update(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc().replace(str(ROOT),'<repo>'))
 finally:
  rec.update(ended_at_utc=utc(),elapsed_seconds=time.perf_counter()-before)
  rec['log_sha256']={p.name:sha(p) for p in AUDIT.glob('*.log')};save()
 print(json.dumps(rec,indent=2))
 return 0 if rec['status']=='completed' else 1
if __name__=='__main__':raise SystemExit(main())
