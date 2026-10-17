import importlib.util,json,os,platform,subprocess,sys,unittest
from pathlib import Path
print(json.dumps({'python':platform.python_version(),'OS':platform.platform(),'cpu':next((x.split(':',1)[1].strip()for x in Path('/proc/cpuinfo').read_text().splitlines()if x.startswith('model name')),'unknown'),'affinity':sorted(os.sched_getaffinity(0)),'pytest_available':importlib.util.find_spec('pytest')is not None,'local_pytest_tests':0,'scope':'stdlib fixtures only, no cached data decode; no package install'},indent=2),flush=True)
suite=unittest.TestSuite()
for pattern in ['test_motion_design.py','test_motion_phases.py','test_bounded_point_audit.py','test_point_geometry_v2.py']:
 suite.addTests(unittest.defaultTestLoader.discover('tests',pattern=pattern))
result=unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful():sys.exit(1)
print(json.dumps({'stdlib_tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures)}))
