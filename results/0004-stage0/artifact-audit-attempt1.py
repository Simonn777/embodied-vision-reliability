"""Saved-file and provenance assertions only; no NPZ decode or inference."""
import csv,hashlib,json,math,re,subprocess
from pathlib import Path
OUT=Path('results/0004-stage0')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  while b:=f.read(1024**2):h.update(b)
 return h.hexdigest()
baseline=json.loads((OUT/'tracked-baseline.json').read_text())
for p,want in baseline.items():assert sha(p)==want,p
assert len(baseline)==1691
freeze=json.loads(Path('protocols/0004-stage0-freeze.json').read_text())
assert sha('protocols/0004-stage0-freeze.json')=='e14bb4f3f4b4dd91c96fe63b1d28bc9a95636d560854de190a701c25dc34667d'
for p,want in freeze['files'].items():
 assert sha(p)==want
 for spec in [freeze['source_commit']+':'+p,'HEAD:'+p,':'+p,'origin/main:'+p]:assert hashlib.sha256(subprocess.check_output(['git','show',spec])).hexdigest()==want
for mode in ['software','scientific']:
 child=json.loads((OUT/(mode+'-child.json')).read_text());outer=json.loads((OUT/(mode+'-outer.json')).read_text())
 assert child['exit_code']==child['raw_wait_status']==outer['observed_wrapper_exit']==outer['raw_wait_status']==0
 for p,want in child['outer_source_sha256'].items():assert sha(p)==want
 assert outer['source_sha256']==sha(OUT/'launch_supervisor.py')
 for s,want in child['logs_sha256'].items():assert sha(OUT/(mode+'.'+s+'.log'))==want
 for s,want in outer['logs_sha256'].items():assert sha(OUT/(mode+'-wrapper.'+s+'.log'))==want
 assert json.loads((OUT/(mode+'-wrapper.stdout.log')).read_text())==child
checker=json.loads((OUT/'independent-check-execution.json').read_text())
assert checker['observed_exit_code']==checker['raw_wait_status']==0
assert checker['checker_sha256']==sha(OUT/'check_saved.py')and checker['parent_sha256']==sha(OUT/'check_parent.py')
for s,want in checker['logs_sha256'].items():assert sha(OUT/('independent-check.'+s+'.log'))==want
receipt=json.loads((OUT/'run/receipt.json').read_text());summary=json.loads((OUT/'run/summary.json').read_text())
for p,want in receipt['artifacts'].items():assert sha(OUT/'run'/p)==want
for module,v in receipt['upstream_modules'].items():assert sha(Path('../bootstapir-cache/source')/v['relative_path'])==v['sha256']
for name,v in receipt['cache_objects'].items():assert sha(Path('../bootstapir-cache')/name)==v['sha256']
marker=Path('../bootstapir-cache/0004-stage0.started.json')
assert sha(marker)==receipt['marker_sha256']==json.loads((OUT/'scientific-child.json').read_text())['persistent_marker_sha256']
assert json.loads(marker.read_text())['source_commit']==freeze['source_commit']
assert summary['validated_rows']==4488 and len(summary['shadow_episodes'])==56
assert summary['continuation']['status']=='stop_this_candidate'
assert 'Ran 21 tests'in(OUT/'software.stderr.log').read_text()
files=sorted(p for p in OUT.rglob('*')if p.is_file()and p.name!='artifact-audit.json'and '__pycache__'not in p.parts)
for p in files:
 assert not p.is_symlink()and p.stat().st_size<1024**2
 assert p.suffix in {'.json','.py','.md','.csv','.log'}
 text=p.read_text();assert '\0'not in text
 assert not re.search(r'sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN .*PRIVATE KEY',text)
 if p.suffix=='.json':json.loads(text,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
 if p.suffix=='.csv':
  for row in csv.DictReader(text.splitlines()):
   for k,v in row.items():
    if k not in ['branch']:assert math.isfinite(float(v)),(p,k)
audit={'status':'PASS','execution_main':'ddadc5e17a6ceff3c358d3e3f440d80bf3fd0306','frozen_inputs_unchanged':len(freeze['files']),'all_original_tracked_paths_byte_preserved':len(baseline),'old_results_byte_preserved':sum(p.startswith('results/')for p in baseline),'new_files_indexed':len(files),'all_small_utf8_files':True,'recorded_child_and_wrapper_exits_bound_to_sources_and_logs':True,'new_marker_present_unchanged_sha256':sha(marker),'raw_predictions_sha256':sha(OUT/'run/predictions.csv'),'no_new_data_or_weights_published':True,'checker_source_sha256':sha(__file__),'boundary':'in-process saved-file assertions, not another scientific run or separately supervised audit exit','artifacts_sha256':{str(p):sha(p)for p in files}}
(OUT/'artifact-audit.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in audit.items()if k!='artifacts_sha256'},indent=2))
