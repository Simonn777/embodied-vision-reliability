import unittest,hashlib,json,datetime
from pathlib import Path
B=Path('results/0003-network-followup-20261017');sources=['protocols/0003-timing-addendum-v1.md','configs/0003-timing-addendum-v1.json','src/evr/motion_phases.py','tests/test_motion_phases.py'];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();bindings={p:sha(p)for p in sources}
r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests',pattern='test_motion_phases.py'));assert r.wasSuccessful()
(B/'timing-tests.json').write_text(json.dumps({'status':'passed','tests_run':r.testsRun,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':bindings,'scope':'prospective typed phase fixtures only, no data/model/GT access'},indent=2)+'\n')
