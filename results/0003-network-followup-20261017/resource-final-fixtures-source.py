import datetime,hashlib,json,unittest,importlib.util
from pathlib import Path
B=Path('results/0003-resource-audit-20261017');spec=importlib.util.spec_from_file_location('resource_audit',Path('scripts/audit_point_sample.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();inputs={p:sha(p)for p in module.INPUTS}
r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests',pattern='test_bounded_point_audit.py'));assert r.wasSuccessful()
(B/'final-fixture-validation.json').write_text(json.dumps({'status':'passed','tests_run':r.testsRun,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':inputs,'GT_values_accessed':False,'scope':'final software fixtures and all ten resource input hashes before resource publication'},indent=2)+'\n')
