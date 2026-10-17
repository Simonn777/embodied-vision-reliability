import unittest,hashlib,json,datetime
from pathlib import Path
B=Path('results/0003-resource-audit-20261017');B.mkdir(exist_ok=False)
files=['src/evr/bounded_point_audit.py','tests/test_bounded_point_audit.py','protocols/0003-resource-addendum-v1.md','configs/0003-resource-addendum-v1.json'];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();inputs={p:sha(p)for p in files}
r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests',pattern='test_bounded_point_audit.py'));assert r.wasSuccessful()
(B/'fixture-validation.json').write_text(json.dumps({'status':'passed','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tests_run':r.testsRun,'source_sha256':inputs,'numeric_GT_access':False,'scope':'hand-crafted NPY byte/header/cap fixtures only'},indent=2)+'\n')
