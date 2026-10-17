"""Pre-freeze review refinements: helpers/hand fixtures/byte checks only."""
import datetime,hashlib,json,subprocess,unittest
from pathlib import Path
from evr.motion_design import MANDATORY
B=Path('results/0003-data-design-20261017');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs={p:sha(p)for p in MANDATORY};tests=unittest.defaultTestLoader.discover('tests',pattern='test_motion_design.py');r=unittest.TextTestRunner(verbosity=2).run(tests);assert r.wasSuccessful()
b=json.loads((B/'baseline.json').read_text())
for p,h in b['protected_sha256'].items():assert sha(p)==h
for p,h in inputs.items():assert sha(p)==h
for p,h in json.loads((B/'evidence-closure.json').read_text())['closed_sha256'].items():assert sha(p)==h
exclude={B/'review-validation.json',*[B/'execution'/('review.'+s)for s in ['json','stdout.log','stderr.log']]}
closed={str(p):sha(p)for p in B.rglob('*')if p.is_file()and p not in exclude}
out={'status':'passed','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'pre-freeze parent-review helper validation; no model/data/old test analysis','tests_run':r.testsRun,'mandatory_input_sha256_at_test_time':inputs,'preserved_old_paths':len(b['protected_sha256']),'changes':'strict risk/count domains, exactly three nonempty positions, NUL/empty archive paths, finite overflow failures; before input commit/freeze','previous_versions':'all prior tested source snapshots/receipts retained; original validation-v2 in-progress two-hash closure issue disclosed in evidence-closure.json and closed here','closed_evidence_sha256':closed,'exclusions':[str(p)for p in sorted(exclude)],'boundary':'current review parent receipt/logs close after child; published Git tree binds them and this output without self hash','source_sha256':sha(__file__)}
with(B/'review-validation.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in out.items()if k not in ['mandatory_input_sha256_at_test_time','closed_evidence_sha256']},indent=2))
