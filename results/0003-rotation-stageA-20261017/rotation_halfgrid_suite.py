"""Targeted synthetic Stage A revision only: no real data or source execution."""
import importlib.util, json, platform, sys, unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('runner','scripts/run_rotation_diagnostic_v1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.verify_prior(Path.cwd())
assert not Path(m.RECEIPT).exists()
assert not Path('../bootstapir-cache/0003-pointodyssey/rotation-diagnostic-v1-started.json').exists()
print(json.dumps({'python':sys.version,'platform':platform.platform(),'prior_frozen_inputs':30,'additional_real_camera_decode':0,'scope':'targeted artificial fixtures only'}),flush=True)
suite=unittest.defaultTestLoader.discover('tests',pattern='test_rotation_diagnostic_v1.py')
r=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if r.wasSuccessful()else 1)
