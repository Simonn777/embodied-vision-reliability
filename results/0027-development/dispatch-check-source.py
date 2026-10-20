"""Private read-only dispatcher receipt check; never decodes numeric truth."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

base = Path(__file__).resolve().parent
repo = base/'embodied-vision-reliability'
data = base/'0027-development'
commit = '7c6144492f6f07247fb603db4bb79e9d1344a853'
phase = sys.argv[1]
phases = ['generation', 'prediction', 'evaluation', 'audit']
assert phase in phases
threads = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')
src = repo/'experiments/0027-affine-tangent'
gen, pred = data/'generation-01', data/'predictions-01'
prefix = [sys.executable, '-W', 'error']
expected_commands = {
 'generation': prefix+[str(src/'pipeline.py'),'generate','--repo',str(repo),
   '--expected-commit',commit,'--output',str(gen)],
 'prediction': prefix+[str(src/'pipeline.py'),'predict','--repo',str(repo),
   '--expected-commit',commit,'--generation',str(gen),'--native-runtime',
   str(base/'0016-preparation/runtime'),'--output',str(pred)],
 'evaluation': prefix+[str(src/'evaluate.py'),'--predictions',str(pred/'predictions.jsonl'),
   '--inputs',str(gen/'inputs.jsonl'),'--truth',str(gen/'truth.jsonl'),
   '--seal',str(pred/'seal.json'),'--output',str(data/'evaluation-01.json')],
 'audit': prefix+[str(src/'audit_saved.py'),'--predictions',str(pred/'predictions.jsonl'),
   '--inputs',str(gen/'inputs.jsonl'),'--truth',str(gen/'truth.jsonl'),
   '--seal',str(pred/'seal.json'),'--evaluation',str(data/'evaluation-01.json'),
   '--prediction-receipt',str(data/'operations/prediction-01/receipt.json'),
   '--output',str(data/'audit-01.json')]}
expected_results = {'generation':gen/'generation.json','prediction':pred/'seal.json',
                    'evaluation':data/'evaluation-01.json','audit':data/'audit-01.json'}
def check_spec(name, spec):
    assert spec['repo']==str(repo) and spec['commit']==commit and spec['timeout']==30
    assert spec['command']==expected_commands[name]
    assert spec['result_file']==str(expected_results[name])
    assert spec['operation_output']==str(data/('operations/'+name+'-01'))
    assert spec['environment']==dict({k:'4' for k in threads},
        PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
def display(value):
    return str(value).replace(sys.executable,'$PYTHON').replace(str(repo),'$REPOSITORY').replace(str(base),'$PRIVATE')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip() == commit
ci = json.loads((data/'source-ci.json').read_text())
assert ci['headSha']==commit and ci['status']=='completed' and ci['conclusion']=='success'
supervisor = repo/'experiments/0027-affine-tangent/supervise.py'
monitor = repo/'experiments/0025c-rs-noise/supervise.py'
assert sha(supervisor)=='7c7798fe76756ace692fe9d9a83755d758690c62f0a6cc7f11af655dd5c60efe'
assert sha(monitor)=='4b8b0efa4fd713a032967c9b34539eac128184ef6db4e8535e83d785c6bbebcd'
checked = []
for previous in phases[:phases.index(phase)]:
    specpath=data/(previous+'-spec-01.json')
    spec=json.loads(specpath.read_text())
    check_spec(previous,spec)
    outer=Path(spec['operation_output'])
    receipt=json.loads((outer/'receipt.json').read_text())
    assert receipt['schema']=='evr0026.operation.v1' and receipt['status']=='completed'
    assert receipt['exit_code']==0 and receipt['termination'] is None
    assert not receipt['remaining_descendant_pids'] and not receipt['cleanup_errors']
    assert receipt['expected_commit']==commit and receipt['hard_timeout_seconds']==30
    assert receipt['group_rss_limit_kib']==8388608
    assert math.isfinite(receipt['peak_group_rss_kib']) and 0<=receipt['peak_group_rss_kib']<=8388608
    assert math.isfinite(receipt['wall_seconds']) and 0<=receipt['wall_seconds']<=30
    assert receipt['thread_environment']=={k:'4' for k in threads}
    assert receipt['command']==[display(v) for v in spec['command']]
    assert receipt['spec_sha256']==sha(specpath)
    assert receipt['supervisor_sha256']==sha(supervisor) and receipt['monitor_sha256']==sha(monitor)
    assert set(receipt['log_sha256'])=={'stdout.log','stderr.log'}
    for name,digest in receipt['log_sha256'].items():assert sha(outer/name)==digest
    assert receipt['output_sha256']==sha(spec['result_file'])
    result=json.loads(Path(spec['result_file']).read_text())
    assert result['status']=='completed'
    if previous in ('generation','prediction'):assert result['row_count']==138
    checked.append({'phase':previous,'receipt_sha256':sha(outer/'receipt.json'),
                    'spec_sha256':sha(specpath),'result_sha256':sha(spec['result_file'])})
specpath=data/(phase+'-spec-01.json')
spec=json.loads(specpath.read_text())
check_spec(phase,spec)
assert not Path(spec['operation_output']).exists() and not Path(spec['result_file']).exists()
record={'schema':'evr0027.dispatch-check.v1','phase':phase,'status':'ready',
        'expected_commit':commit,'source_ci_sha256':sha(data/'source-ci.json'),
        'spec_sha256':sha(specpath),'prior_clean_operations':checked,
        'supervisor_sha256':sha(supervisor),'monitor_sha256':sha(monitor),
        'dispatcher_sha256':sha(__file__),'numeric_truth_decoded':False}
with (data/(phase+'-dispatch-check.json')).open('x') as stream:
    json.dump(record,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps(record,sort_keys=True))
