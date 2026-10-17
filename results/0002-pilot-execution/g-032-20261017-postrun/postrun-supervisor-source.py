"""Observe each non-model post-run check's exit and raw stdout/stderr."""
import datetime,hashlib,json,os,resource,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'results/0002-pilot-execution/g-032-20261017-postrun'
COMMANDS={'verify':'scratch/verify_g032.py','publication_audit':'scratch/publication_audit_g032.py'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def limits():
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3))
 slots=sorted(os.sched_getaffinity(0));assert len(slots)>=2
 os.sched_setaffinity(0,slots[:2])
def main():
 os.chdir(ROOT);name=sys.argv[1];script=COMMANDS[name]
 DIRECTORY.mkdir(parents=True,exist_ok=True)
 archive=DIRECTORY/'postrun-supervisor-source.py'
 if archive.exists():assert archive.read_bytes()==Path(__file__).read_bytes()
 else:archive.write_bytes(Path(__file__).read_bytes())
 receipt=DIRECTORY/(name+'.json')
 # A phase's receipts/logs are exclusive and never silently reused.
 with receipt.open('x') as f:f.write('{}\n')
 command=['timeout','900','python',script]
 rec={'status':'starting','phase':name,'started_at_utc':utc(),'command':command,'environment_policy':'env containing inherited PATH only; no other inherited variables','parent_source_sha256':sha(__file__),'child_source_sha256':sha(script),'observation_scope':'parent wait4 of timeout-wrapped software/stdlib check; no model replay','timeout_seconds':900,'address_space_limit_bytes':6*1024**3,'cpu_affinity':sorted(os.sched_getaffinity(0))[:2]}
 def save():receipt.write_text(json.dumps(rec,indent=2)+'\n')
 save();before=time.perf_counter()
 try:
  with (DIRECTORY/(name+'.stdout.log')).open('x')as out,(DIRECTORY/(name+'.stderr.log')).open('x')as err:
   child=subprocess.Popen(command,cwd=ROOT,env={'PATH':os.environ['PATH']},preexec_fn=limits,stdout=out,stderr=err)
   _,status,usage=os.wait4(child.pid,0);child.returncode=os.waitstatus_to_exitcode(status)
  rec.update(child_exit_code_observed_by_parent=child.returncode,raw_wait_status=status,peak_rss_kib=usage.ru_maxrss,cpu_user_seconds=usage.ru_utime,cpu_system_seconds=usage.ru_stime,status='completed'if child.returncode==0 else'failed')
 except Exception as e:
  rec.update(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc().replace(str(ROOT),'<repo>'))
 finally:
  rec.update(ended_at_utc=utc(),elapsed_seconds=time.perf_counter()-before)
  rec['log_sha256']={stream:sha(DIRECTORY/(name+'.'+stream+'.log'))for stream in ['stdout','stderr']if(DIRECTORY/(name+'.'+stream+'.log')).exists()};save()
  # Close all completed phase bindings, including this one, without hashing a
  # receipt into itself or creating another verification child/receipt cycle.
  try:
   closed={}
   archives={'software':ROOT/'results/0002-pilot-execution/g-032-20261017/software-checks-source.py','verify':ROOT/'results/0002-pilot-execution/g-032-20261017/verification-source.py','publication_audit':DIRECTORY/'publication-audit-source.py'}
   for phase in COMMANDS:
    record_path=DIRECTORY/(phase+'.json')
    if not record_path.exists():continue
    record=json.loads(record_path.read_text())
    if record.get('status')=='starting':continue
    assert record['parent_source_sha256']==sha(archive)
    assert record['child_source_sha256']==sha(archives[phase])
    assert os.waitstatus_to_exitcode(record['raw_wait_status'])==record['child_exit_code_observed_by_parent']
    for stream,digest in record['log_sha256'].items():assert sha(DIRECTORY/(phase+'.'+stream+'.log'))==digest
    closed[phase]=True
   rec['closed_phase_source_log_bindings_verified']=closed;save()
   assert json.loads(receipt.read_text())==rec
  except Exception as exc:
   rec.update(status='failed',binding_error_type=type(exc).__name__,binding_error=str(exc));save()
 print(json.dumps(rec,indent=2))
 return 0 if rec['status']=='completed'else 1
if __name__=='__main__':raise SystemExit(main())
