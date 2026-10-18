"""Only the four authorized artificial test modules; verify seven Linux tests."""
import hashlib,json,sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
def flatten(suite):
    for item in suite:
        if isinstance(item,unittest.TestSuite):yield from flatten(item)
        else:yield item
def main():
    suite=unittest.TestSuite();loader=unittest.TestLoader()
    for name in ('test_exact_assignment_tail.py','test_geometric_verification.py','test_geometric_stage0.py','test_0009_supervisor.py'):
        suite.addTests(loader.discover('tests',pattern=name,top_level_dir='tests'))
    linux=[x.id() for x in flatten(suite) if '.LinuxSupervisorIntegration.' in x.id()]
    assert len(linux)==7,linux
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    record={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'skips':len(result.skipped),'failures':len(result.failures),'errors':len(result.errors),'linux_process_resource_termination_test_ids':linux,'linux_test_count':len(linux),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Artificial test fixtures only; not a scientific collection or warmup.'}
    with (HERE/'tests.json').open('x') as f:json.dump(record,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(record,sort_keys=True));return 0 if result.wasSuccessful() and not result.skipped else 1
if __name__=='__main__':raise SystemExit(main())
