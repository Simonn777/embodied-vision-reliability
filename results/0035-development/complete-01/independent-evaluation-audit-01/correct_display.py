"""Correct one auditor-only command-display assumption; do not repeat metrics."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
D35 = HERE.parent
REPO = D35.parent / "embodied-vision-reliability"
EXPECTED_AUDIT = "75e6b973945a3440719afc6f0aebe4a804a585767c547e016d6c61e2ebb0460c"
EXPECTED_REPORT = "70e034161cb7a507aa925550e3fda52582c33c8a967cb7e8c88669d59bc61470"
EXPECTED_MONITOR = "4b8b0efa4fd713a032967c9b34539eac128184ef6db4e8535e83d785c6bbebcd"
inputs, errors = {}, []


def read(path):
    raw = path.read_bytes()
    raw.decode("utf-8")
    inputs[str(path)] = hashlib.sha256(raw).hexdigest()
    return raw


def load(path):
    return json.loads(read(path))


def check(value, label):
    if not value:
        errors.append(label)


report = load(HERE / "report.json")
check(inputs[str(HERE / "report.json")] == EXPECTED_REPORT, "first report changed")
read(HERE / "audit.py")
check(inputs[str(HERE / "audit.py")] == EXPECTED_AUDIT, "first audit source changed")
check(report["passed"] is False and report["errors"] == ["evaluation: exact command"]
      and report["checks"] == 8313, "only the documented failed assertion may be corrected")
first_outer = load(HERE / "supervision/receipt.json")
check(first_outer["status"] == "failed" and first_outer["exit_code"] == 1
      and first_outer["termination"] is None and not first_outer["remaining_descendant_pids"]
      and not first_outer["cleanup_errors"] and first_outer["output_sha256"] == EXPECTED_REPORT,
      "preserved first audit outer identity")
for name, digest in first_outer["log_sha256"].items():
    read(HERE / "supervision" / name)
    check(inputs[str(HERE / "supervision" / name)] == digest, "first audit log changed")
monitor = REPO / "experiments/0025c-rs-noise/supervise.py"
read(monitor)
check(inputs[str(monitor)] == EXPECTED_MONITOR, "pinned display source changed")
spec = load(D35 / "evaluate-outer-spec-01.json")
outer = load(D35 / "evaluation-supervision-01/receipt.json")
# Pinned monitor.display substitutes its own (supervisor) interpreter, not the
# distinct qualified-venv interpreter in the child command.
correct = [str(v).replace(str(sys.executable), "$PYTHON").replace(spec["repo"], "$REPOSITORY")
           .replace(str(Path(spec["repo"]).parent), "$PRIVATE") for v in spec["command"]]
check(correct == outer["command"], "exact pinned display still mismatches")
check(correct[0] == "$PRIVATE/0029-runtime-02/installation/venv/bin/python",
      "qualified child interpreter must remain its explicit private-relative path")
check(report["max_aggregate_absolute_difference"] <= report["aggregate_tolerance"]
      and report["recomputed"]["primary"]["decision_recomputed"]["numerical_passed"] is True,
      "inherited arithmetic checks must remain successful")
result = dict(schema="evr0035.saved-text-audit-display-correction.v1", passed=not errors,
              errors=errors, first_audit_status="failed_preserved",
              original_successful_checks=8312, corrected_assertion="evaluation: exact command",
              exact_command=correct, original_inputs_sha256=inputs,
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              scope="Only correct the pinned supervisor display comparison; no metric, model or payload rerun.",
              independent_saved_text_review_passed=not errors)
with (HERE / "correction-review.json").open("x") as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
    stream.write("\n")
print(json.dumps(dict(passed=not errors, errors=errors, original_audit_status="failed_preserved")))
raise SystemExit(0 if not errors else 1)
