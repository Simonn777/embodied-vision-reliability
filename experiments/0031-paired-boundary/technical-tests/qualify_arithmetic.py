"""One manufactured-only strided reduction qualification; never reads scientific inputs."""
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import unittest
import warnings


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    pending.replace(path)


def main(spec_path):
    began = time.perf_counter()
    spec = json.loads(Path(spec_path).read_text())
    out = Path(spec["output"])
    out.mkdir(exist_ok=False)
    record = dict(schema="evr0031.manufactured-strided-arithmetic.v1", status="started",
        scientific_data_access=False, model_calls=0, cases=[], warnings=[],
        spec_sha256=sha(spec_path), source_sha256=spec["source_sha256"])
    receipt = out/"receipt.json"
    save(receipt, record)
    try:
        for filename, digest in spec["source_sha256"].items():
            if sha(filename) != digest:
                raise ValueError("changed qualification source")
        if not all(os.environ.get(k) == "4" for k in
                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")):
            raise ValueError("four-thread envelope")
        # Required ordering: Torch initializes first and explicitly configures four threads.
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        import numpy as np
        record["runtime"] = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                                 torch_threads=torch.get_num_threads(), torch_interop_threads=1)
        sys.path.insert(0, spec["source_root"])
        adapter = importlib.import_module("screen_sum")
        screen = importlib.import_module("screen")
        test_module = importlib.import_module("test_screen")
        rng = np.random.default_rng(3100101)
        eps = np.finfo(np.float64).eps
        record.update(seed=3100101, cases_requested=100, unit_arrays_shape=[128, 2, 32],
            dot_tolerance="64*float64eps*max(1,sum(abs(products)))",
            score_invariance_tolerance=256*eps, max_explicit_fsum_error=0.0,
            max_successful_matmul_fsum_error=0.0, max_adapter_fsum_score_error=0.0,
            max_swap_or_permutation_error=0.0, matmul_attempts=0, matmul_successes=0)
        for case in range(100):
            bank = rng.uniform(-1, 1, (128, 2, 32))
            bank /= np.sqrt(np.sum(bank*bank, axis=-1, keepdims=True))
            current = rng.uniform(-1, 1, (2, 32))
            current /= np.sqrt(np.sum(current*current, axis=-1, keepdims=True))
            u, v = bank[:, 0], bank[:, 1]
            if u.flags.c_contiguous or v.flags.c_contiguous:
                raise AssertionError("manufactured bank no longer exercises strided rows")
            dots, case_warnings = [], []
            max_sum_error, max_matmul_error = 0.0, 0.0
            for name, matrix, vector in (("xu", u, current[0]), ("xv", v, current[0]),
                                          ("yu", u, current[1]), ("yv", v, current[1])):
                products = matrix*vector[None, :]
                explicit = np.sum(products, axis=1, dtype=np.float64)
                reference = np.array([math.fsum(float(z) for z in row) for row in products])
                tolerance = 64*eps*np.maximum(1, np.sum(np.abs(products), axis=1))
                error = np.abs(explicit-reference)
                if not np.isfinite(explicit).all() or not np.all(error <= tolerance):
                    raise AssertionError("explicit reduction differs from fsum bound")
                max_sum_error = max(max_sum_error, float(error.max()))
                dots.append(reference)
                record["matmul_attempts"] += 1
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        observed = matrix @ vector
                    if not np.isfinite(observed).all():
                        raise FloatingPointError("matmul returned a nonfinite vector without warning")
                    error = np.abs(observed-reference)
                    if not np.all(error <= tolerance):
                        raise AssertionError("finite matmul exceeds declared fsum bound")
                    max_matmul_error = max(max_matmul_error, float(error.max()))
                    record["matmul_successes"] += 1
                except (Warning, FloatingPointError) as exc:
                    detail = dict(case=case, operation=name, type=type(exc).__name__, message=str(exc))
                    case_warnings.append(detail)
                    record["warnings"].append(detail)
                    print(json.dumps(detail), file=sys.stderr, flush=True)
            permutation = np.roll(np.arange(128), -1)
            result, _ = adapter.scores(current, bank, permutation)
            xu, xv, yu, yv = dots
            expected = dict(paired=float(max(xu+yv)-max(xv+yu)),
                unpaired_max=float(max(xu)+max(yv)-max(xv)-max(yu)),
                deranged=float(max(xu+yv[permutation])-max(xv[permutation]+yu)))
            score_error = max(abs(result[k]-expected[k]) for k in result)
            swapped, _ = adapter.scores(current[::-1], bank, permutation)
            order = rng.permutation(128)
            inverse = np.argsort(order)
            reordered_permutation = inverse[permutation[order]]
            reordered, _ = adapter.scores(current, bank[order], reordered_permutation)
            invariance = max(max(abs(result[k]+swapped[k]) for k in result),
                             max(abs(result[k]-reordered[k]) for k in result))
            if not all(math.isfinite(x) for x in result.values()) or \
                    max(score_error, invariance) > 256*eps:
                raise AssertionError("adapter score or swap/permutation identity failed")
            row = dict(index=case, bank_strides=list(bank.strides), side_strides=list(u.strides),
                max_explicit_fsum_error=max_sum_error, max_successful_matmul_fsum_error=max_matmul_error,
                adapter_fsum_score_error=score_error, swap_permutation_error=invariance,
                matmul_warning_count=len(case_warnings), status="completed")
            record["cases"].append(row)
            for key, value in (("max_explicit_fsum_error", max_sum_error),
                               ("max_successful_matmul_fsum_error", max_matmul_error),
                               ("max_adapter_fsum_score_error", score_error),
                               ("max_swap_or_permutation_error", invariance)):
                record[key] = max(record[key], value)
            if case % 10 == 9:
                save(receipt, record)
        original = screen.scores
        screen.scores = adapter.scores
        log = io.StringIO()
        try:
            suite = unittest.defaultTestLoader.loadTestsFromModule(test_module)
            outcome = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        finally:
            screen.scores = original
        (out/"adapter-geometry-tests.txt").write_text(log.getvalue())
        print(log.getvalue(), file=sys.stderr, end="", flush=True)
        record["geometry_tests"] = dict(count=outcome.testsRun, failures=len(outcome.failures),
            errors=len(outcome.errors), skipped=len(outcome.skipped),
            log_sha256=sha(out/"adapter-geometry-tests.txt"))
        if outcome.testsRun != 7 or not outcome.wasSuccessful():
            raise AssertionError("adapter geometry qualification failed")
        record["source_post_sha256"] = {name: sha(name) for name in spec["source_sha256"]}
        if record["source_post_sha256"] != spec["source_sha256"]:
            raise ValueError("source changed during manufactured qualification")
        record.update(status="completed", stable_adapter_passed=True,
            matmul_warning_reproduced=bool(record["warnings"]), completed_cases=100,
            interpretation="Manufactured arithmetic only; no inference about exact scientific input failure trigger")
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record["wall_seconds"] = time.perf_counter()-began
        save(receipt, record)
    print(json.dumps({k: record[k] for k in ("status", "completed_cases", "matmul_attempts",
        "matmul_successes", "matmul_warning_reproduced", "stable_adapter_passed", "wall_seconds")}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
