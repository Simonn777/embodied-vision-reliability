"""Private process-only supervisor; reuses the previously frozen process-tree monitor."""
import hashlib, importlib.util, json, os, pathlib, signal, subprocess, sys, time
specification=json.loads(pathlib.Path(sys.argv[1]).read_text())
repo=pathlib.Path(specification['repo']); output=pathlib.Path(specification['operation_output']);output.mkdir(parents=True,exist_ok=False)
path=repo/'experiments/0025c-rs-noise/supervise.py';spec=importlib.util.spec_from_file_location('prior_resource_monitor',path);monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
command=specification['command'];env=dict(os.environ);env.update(specification['environment']);env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONNOUSERSITE']='1'
limit=specification['timeout'];assert 0<limit<=1800
record={'schema':'evr0026.operation.v1','status':'started','command':[monitor.display(v) for v in command],'spec_sha256':monitor.sha256(sys.argv[1]),'supervisor_sha256':monitor.sha256(__file__),'monitor_sha256':monitor.sha256(path),'expected_commit':specification['commit'],'hard_timeout_seconds':limit,'group_rss_limit_kib':8388608,'thread_environment':{k:env[k] for k in monitor.THREAD_KEYS}}
receipt=output/'receipt.json';monitor.save(receipt,record);start=time.perf_counter();groups=set();child=None;peak=0;termination=None;code=None;remaining=[];cleanup=[]
with (output/'stdout.log').open('xb') as stdout,(output/'stderr.log').open('xb') as stderr:
 try:
  child=subprocess.Popen(command,cwd=repo,env=env,stdout=stdout,stderr=stderr,start_new_session=True);groups.add(child.pid)
  while True:
   rows,new=monitor.snapshot(child.pid,groups);groups.update(new);peak=max(peak,sum(r[3] for r in rows))
   if peak>8388608:termination='group_rss_limit';break
   if time.perf_counter()-start>=limit:termination='timeout';break
   if child.poll() is not None:
    if any(r[0]!=child.pid for r in rows):termination='descendants_remain'
    break
   time.sleep(.1)
 except BaseException as error:termination=type(error).__name__+': '+monitor.display(error)
 finally:
  if child is not None:
   if child.poll() is None or termination is not None or child.returncode!=0:
    try:os.killpg(child.pid,signal.SIGSTOP)
    except ProcessLookupError:pass
    try:
     rows,new=monitor.snapshot(child.pid,groups);groups.update(new)
    except Exception as error:cleanup.append(type(error).__name__)
    for group in groups:
     try:os.killpg(group,signal.SIGKILL)
     except ProcessLookupError:pass
     except OSError as error:cleanup.append(type(error).__name__)
   code=child.wait(timeout=5)
   rows,new=monitor.snapshot(child.pid,groups);remaining=[r[0] for r in rows if r[0]!=child.pid]
record.update(status='completed' if code==0 and termination is None and not remaining and not cleanup else 'failed',exit_code=code,termination=termination,wall_seconds=time.perf_counter()-start,peak_group_rss_kib=peak,remaining_descendant_pids=remaining,cleanup_errors=cleanup,log_sha256={n:monitor.sha256(output/n) for n in ('stdout.log','stderr.log')})
if 'result_file' in specification and pathlib.Path(specification['result_file']).exists():record['output_sha256']=monitor.sha256(specification['result_file'])
monitor.save(receipt,record)
print(json.dumps(record,indent=2));sys.exit(0 if record['status']=='completed' else 1)
