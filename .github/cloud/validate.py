"""Fixed container entrypoint; import the real test framework before candidate code."""

import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
if len(sys.argv) == 2:
    # Separate interpreter sessions prevent candidate conftest hooks from affecting
    # the immutable baseline suite. Import pytest before adding the candidate path.
    import pytest
    suite = sys.argv[1]
    if suite not in {"baseline", "candidate"}:
        raise SystemExit("Unknown suite")
    sys.path.insert(0, "/work/src")
    raise SystemExit(pytest.main([
        "-q", "-p", "no:cacheprovider", "--import-mode=importlib", "--confcutdir=/baseline/tests" if suite == "baseline" else "--confcutdir=/work/tests",
        "--junitxml=/work/scratch/cloud-validation/" + suite + ".xml",
        "/baseline/tests" if suite == "baseline" else "/work/tests",
    ]))

for suite in ("baseline", "candidate"):
    subprocess.run([sys.executable, "-I", __file__, suite], check=True, timeout=180)
    report = Path("/work/scratch/cloud-validation") / (suite + ".xml")
    suites = list(ET.parse(report).getroot().iter("testsuite"))
    if sum(int(s.attrib.get("tests", 0)) for s in suites) < 17:
        raise SystemExit("The diagnostic tests did not execute")
    if any(any(int(s.attrib.get(key, 0)) for key in ("failures", "errors", "skipped")) for s in suites):
        raise SystemExit("A test failed, errored, or was skipped")
subprocess.run([
    sys.executable, "-I", "scripts/run_experiment.py", "--smoke",
    "--output", "scratch/cloud-validation/smoke", "--protocol",
    "protocols/0001-synthetic-localization.md",
], check=True, timeout=180)
