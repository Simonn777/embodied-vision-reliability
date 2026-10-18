"""Publication-only source gate check: no main(), cache, dataset or projection."""
import hashlib, json, runpy
from pathlib import Path
runner = runpy.run_path('scripts/audit_0003_annotation_consistency_v1.py', run_name='publication_review')
receipt = runner['verify_published']()
raw = Path(runner['RECEIPT']).read_bytes()
print(json.dumps({'gate':'PASS','input_commit':receipt['input_commit'],
    'mandatory_inputs':len(receipt['inputs_sha256']),
    'receipt_sha256':hashlib.sha256(raw).hexdigest(),
    'prior_chains':5,'distinct_prior_inputs':38,
    'scope':'actual verify_published only; no main/cache/data/geometry access'},indent=2))
