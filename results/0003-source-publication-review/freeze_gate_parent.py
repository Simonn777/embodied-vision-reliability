"""Record the actual local publication-only gate child, not a data experiment."""
from pathlib import Path
import datetime, hashlib, json, os, platform, subprocess, sys, time
out=Path('/tmp/0003-source-publication-review-staging')
child=out/'freeze_gate_check.py'
command=[sys.executable,str(child)]
env={'PATH':os.environ['PATH'],'PYTHONPATH':'src','GIT_NO_LAZY_FETCH':'1'}
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
started=datetime.datetime.now(datetime.timezone.utc).isoformat(); t=time.perf_counter()
r=subprocess.run(command,env=env,capture_output=True,timeout=60)
ended=datetime.datetime.now(datetime.timezone.utc).isoformat();elapsed=time.perf_counter()-t
(out/'freeze-gate.stdout.json').write_bytes(r.stdout);(out/'freeze-gate.stderr.log').write_bytes(r.stderr)
record={'execution_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
'command':command,'started_utc':started,'ended_utc':ended,'elapsed_seconds':elapsed,'observed_child_returncode':r.returncode,
'platform':platform.platform(),'python':sys.version,'timeout_seconds':60,
'child_source_sha256':sha(child),'parent_source_sha256':sha(__file__),
'stdout_sha256':sha(out/'freeze-gate.stdout.json'),'stderr_sha256':sha(out/'freeze-gate.stderr.log'),
'scope':'Local pure source/publication gate. No cache/data/geometry/main call. No outer supervisor exit is claimed.'}
(out/'freeze-gate-execution.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2));print(r.stdout.decode());raise SystemExit(r.returncode)
