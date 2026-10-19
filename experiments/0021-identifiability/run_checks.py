"""One-shot artificial check, retaining failures and measured test receipt."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time
import traceback
import math_checks as m


HERE = pathlib.Path(__file__).resolve().parent


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(output):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (HERE / "math_checks.py", HERE / "test_checks.py", HERE / "run_checks.py", HERE / "README.md")}
    receipt = {"schema": "evr0021.artificial-receipt.v1", "status": "started",
               "source_sha256": hashes, "python_version": sys.version,
               "numpy_version": m.np.__version__, "fixture_spec": m.FIXTURE_SPEC,
               "rank_rtol": m.RANK_RTOL, "rank_sensitivity": m.RANK_SENSITIVITY,
               "finite_difference_step": m.FD_STEP, "finite_difference_atol": m.FD_ATOL,
               "finite_difference_rtol": m.FD_RTOL, "scientific_data_read": False,
               "query_pose_fit": False, "thread_limit": 1}
    write(output / "receipt.json", receipt)
    try:
        write(output / "mathematics.json", m.evaluate())
        before = time.monotonic()
        test = subprocess.run([sys.executable, str(HERE / "test_checks.py")],
                              text=True, capture_output=True, timeout=30, env=os.environ.copy())
        (output / "tests.log").write_text(test.stdout + test.stderr)
        receipt["test_process"] = {"returncode": test.returncode,
                                   "wall_seconds": time.monotonic() - before,
                                   "log_sha256": hashlib.sha256((output / "tests.log").read_bytes()).hexdigest()}
        receipt["mathematics_sha256"] = hashlib.sha256((output / "mathematics.json").read_bytes()).hexdigest()
        receipt["status"] = "passed" if test.returncode == 0 else "failed"
        receipt["interpretation"] = "Manufactured algebra checks only; no pose-accuracy or novel-method result."
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        (output / "error.log").write_text(traceback.format_exc())
    finally:
        receipt["wall_seconds"] = time.monotonic() - started
        receipt["source_unchanged"] = all(hashlib.sha256((HERE / name).read_bytes()).hexdigest() == sha
                                          for name, sha in hashes.items())
        if not receipt["source_unchanged"]:
            receipt["status"] = "failed"
        write(output / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "wall_seconds": receipt["wall_seconds"],
                      "test_returncode": receipt.get("test_process", {}).get("returncode")}))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    sys.exit(run(args.output_dir))
