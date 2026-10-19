"""ctypes bridge and reproducible local build for the dependency-free scorer."""
from __future__ import annotations

import argparse
import ctypes as ct
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np


class NSResult(ct.Structure):
    _fields_ = [("winner", ct.c_int64), ("score", ct.c_double),
                ("residuals", ct.c_uint64), ("nodes", ct.c_uint64),
                ("refinements", ct.c_uint64), ("pruned", ct.c_uint64),
                ("setup_seconds", ct.c_double), ("scoring_seconds", ct.c_double)]


class NSValidation(ct.Structure):
    _fields_ = [("cell_checks", ct.c_uint64), ("aggregate_checks", ct.c_uint64),
                ("violations", ct.c_uint64), ("first_model", ct.c_int64),
                ("first_cell", ct.c_int64), ("max_lower_violation", ct.c_double),
                ("max_upper_violation", ct.c_double)]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_native():
    source = Path(__file__).with_name("native_score.cpp").resolve()
    output = source.parents[3] / "native"
    output.mkdir(parents=True, exist_ok=True)
    library = output / ("libnative_score.dylib" if platform.system() == "Darwin" else "libnative_score.so")
    flags = ["-std=c++17", "-O3", "-fPIC", "-shared", "-fno-fast-math", "-ffp-contract=off"]
    command = ["/usr/bin/clang++", *flags, str(source), "-o", str(library)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("Native build failed:\n" + result.stderr)
    receipt = {"utc": datetime.now(timezone.utc).isoformat(), "source": str(source),
        "source_sha256": sha256(source), "library": str(library),
        "library_sha256": sha256(library), "command": command,
        "compiler": subprocess.check_output(["/usr/bin/clang++", "--version"], text=True),
        "stdout": result.stdout, "stderr": result.stderr,
        "platform": platform.platform(), "architecture": platform.machine()}
    path = output / "build.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    return path


def record(structure):
    return {name: getattr(structure, name) for name, _ in structure._fields_}


class NativeScorer:
    def __init__(self, library):
        self.lib = ct.CDLL(str(Path(library).resolve()))
        doubles, integers = ct.POINTER(ct.c_double), ct.POINTER(ct.c_uint64)
        base = [doubles, doubles, ct.c_uint64, doubles, ct.c_uint64, ct.c_double]
        error = [ct.POINTER(ct.c_char), ct.c_uint64]
        self.lib.ns_select.argtypes = base + [integers, ct.c_int, ct.POINTER(NSResult)] + error
        self.lib.ns_scores.argtypes = base + [doubles] + error
        self.lib.ns_validate.argtypes = base + [ct.POINTER(NSValidation)] + error
        for name in ("ns_select", "ns_scores", "ns_validate"):
            getattr(self.lib, name).restype = ct.c_int

    @staticmethod
    def _inputs(x1, x2, matrices, tau2):
        x1, x2, matrices = [np.ascontiguousarray(a, dtype=np.float64) for a in (x1, x2, matrices)]
        if x1.ndim != 2 or x1.shape[1] != 2 or x1.shape != x2.shape or len(x1) == 0:
            raise ValueError("nonempty equal Nx2 coordinate arrays required")
        if matrices.ndim != 3 or matrices.shape[1:] != (3, 3) or len(matrices) == 0:
            raise ValueError("nonempty Hx3x3 matrices required")
        if not all(np.isfinite(a).all() for a in (x1, x2, matrices)) or not np.isfinite(tau2) or tau2 <= 0:
            raise ValueError("finite inputs and positive threshold required")
        pointer = ct.POINTER(ct.c_double)
        arguments = [x1.ctypes.data_as(pointer), x2.ctypes.data_as(pointer), len(x1),
                     matrices.ctypes.data_as(pointer), len(matrices), float(tau2)]
        return (x1, x2, matrices), arguments  # retain arrays until the C call returns

    @staticmethod
    def _call(function, arguments):
        error = ct.create_string_buffer(1024)
        status = function(*arguments, error, len(error))
        if status:
            raise RuntimeError("native status %d: %s" % (status, error.value.decode("utf-8", "replace")))

    def select(self, x1, x2, matrices, tau2, permutation, mode):
        arrays, arguments = self._inputs(x1, x2, matrices, tau2)
        permutation_pointer = ct.POINTER(ct.c_uint64)()
        if mode in (1, 3, 4):
            permutation = np.ascontiguousarray(permutation, dtype=np.uint64)
            if permutation.shape != (len(arrays[0]),):
                raise ValueError("permutation must have one index per point")
            # Native linear seen-array validation checks range and uniqueness.
            permutation_pointer = permutation.ctypes.data_as(ct.POINTER(ct.c_uint64))
        result = NSResult()
        self._call(self.lib.ns_select, arguments + [permutation_pointer,
                                                   int(mode), ct.byref(result)])
        output = record(result)
        if not all(np.isfinite(output[k]) for k in ("score", "setup_seconds", "scoring_seconds")):
            raise RuntimeError("native result contains nonfinite values")
        if not 0 <= output["winner"] < len(arrays[2]) or min(output["setup_seconds"], output["scoring_seconds"]) < 0:
            raise RuntimeError("native result has invalid winner/timing")
        return output

    def scores(self, x1, x2, matrices, tau2):
        arrays, arguments = self._inputs(x1, x2, matrices, tau2)
        scores = np.empty(len(arrays[2]), dtype=np.float64)
        self._call(self.lib.ns_scores, arguments + [scores.ctypes.data_as(ct.POINTER(ct.c_double))])
        if not np.isfinite(scores).all():
            raise RuntimeError("native exact scores are nonfinite")
        return scores

    def validate(self, x1, x2, matrices, tau2):
        arrays, arguments = self._inputs(x1, x2, matrices, tau2)
        result = NSValidation()
        self._call(self.lib.ns_validate, arguments + [ct.byref(result)])
        output = record(result)
        if not np.isfinite([output["max_lower_violation"], output["max_upper_violation"]]).all():
            raise RuntimeError("native validation returned nonfinite diagnostics")
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", required=True)
    parser.parse_args()
    print(build_native())
