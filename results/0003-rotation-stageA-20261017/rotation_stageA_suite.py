"""Artificial fixtures + published prior-chain verification only; no cache arrays."""
import importlib.util,json,os,platform,unittest
from pathlib import Path
print(json.dumps({'python':platform.python_version(),'platform':platform.platform(),'CPU':next((s.split(':',1)[1].strip()for s in Path('/proc/cpuinfo').read_text().splitlines()if s.startswith('model name')),'unknown'),'affinity':sorted(os.sched_getaffinity(0)),'pytest_available':importlib.util.find_spec('pytest')is not None,'local_pytest_tests':0,'real_camera_numeric_access':0},indent=2),flush=True)
suite=unittest.TestSuite()
for pattern in ['test_motion_design.py','test_motion_phases.py','test_bounded_point_audit.py','test_point_geometry_v2.py','test_rotation_diagnostic_v1.py']:
 suite.addTests(unittest.defaultTestLoader.discover('tests',pattern=pattern))
r=unittest.TextTestRunner(verbosity=2).run(suite)
if not r.wasSuccessful():raise SystemExit(1)
spec=importlib.util.spec_from_file_location('new_runner','scripts/run_rotation_diagnostic_v1.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
prior=m.verify_prior(Path.cwd());assert len(prior)==30
assert not Path(m.RECEIPT).exists()
print(json.dumps({'stdlib_tests':r.testsRun,'prior_frozen_inputs':len(prior),'production_receipt':'absent','real_camera_numeric_access':0}))
