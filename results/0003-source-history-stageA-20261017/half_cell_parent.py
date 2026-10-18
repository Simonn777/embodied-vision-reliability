"""One saved-R invocation, binding prepublished method and actual published HEAD first."""
import datetime,hashlib,json,os,resource,subprocess,time
from pathlib import Path
B=Path('results/0003-half-cell-analysis-20261017');original='c1bcc597db980cd3aa14c0747af0183da266e6af'
paths=['protocols/0003-half-cell-analysis-v1.md','configs/0003-half-cell-analysis-v1.json','scripts/half_cell_certificate_v1.py','tests/test_half_cell_certificate_v1.py']
h=lambda raw:hashlib.sha256(raw).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],env={**os.environ,'GIT_NO_LAZY_FETCH':'1'})
head=git('rev-parse','HEAD').decode().strip();assert head==git('rev-parse','origin/research/0003-source-history-20261017').decode().strip()
assert original==git('rev-parse','origin/research/0003-half-cell-method-20261017').decode().strip()
source={}
for p in paths:
 raw=Path(p).read_bytes()
 for spec in [original+':'+p,'HEAD:'+p,':'+p,'origin/research/0003-source-history-20261017:'+p]:assert git('show',spec)==raw
 source[p]=h(raw)
assert source['scripts/half_cell_certificate_v1.py']=='c1a8f2915043af2fda54b6629aadca9b46eb6a25532e38fa1660f4b227f69c6a'
c=json.loads(Path(paths[1]).read_text());excerpt_raw=Path(c['source_excerpt']).read_bytes();diag_raw=Path(c['old_diagnostic']).read_bytes()
assert h(excerpt_raw)==c['source_excerpt_sha256'] and h(diag_raw)==c['old_diagnostic_sha256']
e=json.loads(excerpt_raw);entry=e['camera_access']['extrinsics'];assert entry['header']['dtype']=='<f4' and entry['header']['order']=='C'
raw=b''.join(bytes.fromhex(entry['values'][i][j]['stored_bytes_hex'])for i in range(3)for j in range(3));assert len(raw)==36
B.mkdir(parents=True,exist_ok=False)
with(B/'input.json').open('x')as f:f.write(json.dumps({'R_hex':raw.hex(),'byte_order':'little'},indent=2)+'\n')
terms=e['data_terms'].copy();terms['analysis_output']='hypothetical exact proper-rotation witness and cell statistics are derived data, never a replacement camera or proof of generation; original R bytes unchanged';(B/'data-terms.json').write_text(json.dumps(terms,indent=2)+'\n')
command=['timeout','60','python','scripts/half_cell_certificate_v1.py','--input',str(B/'input.json')]
env={'PATH':os.environ['PATH'],'PYTHONPATH':'src','GIT_NO_LAZY_FETCH':'1'}
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[n]='2'
affinity=sorted(os.sched_getaffinity(0))[:2]
def limit():resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3));os.sched_setaffinity(0,affinity)
pre={'status':'ANALYSIS_STARTING_NO_RETRY','verified_before_invocation_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'prepublished_method_commit':original,'actual_execution_commit':head,'method_source_sha256':source,'parent_source_sha256':h(Path(__file__).read_bytes()),'input_json_sha256':h((B/'input.json').read_bytes()),'original_R36_bytes_sha256':h(raw),'saved_excerpt_sha256':h(excerpt_raw),'old_diagnostic_sha256':h(diag_raw),'command':command,'scope':'saved original upper-left R bytes only; zero archive/cache access, no geometry','counts_if_interrupted':'incomplete/unknown; retain output and do not retry'}
with(B/'pre-execution.json').open('x')as f:json.dump(pre,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
start=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.perf_counter()
with(B/'output.json').open('xb')as out,(B/'stderr.log').open('xb')as err:
 child=subprocess.Popen(command,env=env,stdout=out,stderr=err,preexec_fn=limit)
 _,status,u=os.wait4(child.pid,0);code=os.waitstatus_to_exitcode(status);child.returncode=code
r={'command':command,'prepublished_method_commit':original,'execution_commit':head,'started_utc':start,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_seconds':time.perf_counter()-t,'peak_rss_kib':u.ru_maxrss,'CPU_user_seconds':u.ru_utime,'CPU_system_seconds':u.ru_stime,'observed_exit_code':code,'raw_wait_status':status,'source_sha256':{**source,str(Path(__file__).resolve().relative_to(Path.cwd())):h(Path(__file__).read_bytes())},'input_sha256':pre['input_json_sha256'],'original_R36_bytes_sha256':h(raw),'stdout_sha256':h((B/'output.json').read_bytes()),'stderr_sha256':h((B/'stderr.log').read_bytes()),'pre_execution_receipt_sha256':h((B/'pre-execution.json').read_bytes()),'AS_bytes':6*1024**3,'affinity':affinity,'timeout_seconds':60,'environment':{**env,'PATH':'inherited runtime PATH value omitted publicly'},'boundary':'actual parent wait4 of timeout/checker; parent itself has no additional observed supervisor exit receipt'}
with(B/'execution.json').open('x')as f:f.write(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2));raise SystemExit(code)
