"""Published receipt and prior chains only; no numeric cache access."""
import datetime,hashlib,importlib.util,json,os,platform,subprocess,sys
from pathlib import Path
s=importlib.util.spec_from_file_location('r','scripts/run_rotation_diagnostic_v1.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
r=m.verify_published();cache=Path('../bootstapir-cache/0003-pointodyssey')
assert not(cache/'rotation-diagnostic-v1-started.json').exists()
assert not Path('results/0003-rotation-stageB-20261017/audit').exists()
assert m.sha(cache/'fortran-v2-started.json')==json.loads(Path('configs/0003-rotation-diagnostic-v1.json').read_text())['old_marker_sha256']
cpu=next((x.split(':',1)[1].strip()for x in Path('/proc/cpuinfo').read_text().splitlines()if x.startswith('model name')),'unknown')
print(json.dumps({'status':'verified_before_numeric_access','verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'input_commit':r['input_commit'],'receipt_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'receipt_sha256':m.sha(m.RECEIPT),'mandatory_inputs':r['inputs_sha256'],'prior_frozen_distinct_inputs':len(m.verify_prior(Path.cwd())),'python':sys.version,'platform':platform.platform(),'CPU':cpu,'affinity':sorted(os.sched_getaffinity(0)),'new_marker_and_audit':'absent','old_marker':'unchanged','numeric_access':False},indent=2))
