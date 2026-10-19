"""0012d bridge; the frozen 0012c residual and partial controls are included."""
from __future__ import annotations
import argparse
import ctypes as ct
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import numpy as np
from native_backend import NativeScorer, NSResult, NSValidation, record, sha256


class NGResult(ct.Structure):
    _fields_ = [("base", NSResult)] + [(name, ct.c_uint64) for name in
        ("moment_cells", "moment_points", "grid_cells", "eligible_cells",
         "geometry_calls", "cantelli_calls")]


def build_grid():
    source = Path(__file__).with_name("native_grid.cpp").resolve()
    output = source.parents[3] / "native-grid"
    output.mkdir(parents=True, exist_ok=True)
    library = output / ("libnative_grid.dylib" if platform.system() == "Darwin" else "libnative_grid.so")
    command = ["/usr/bin/clang++", "-std=c++17", "-O3", "-fPIC", "-shared",
               "-fno-fast-math", "-ffp-contract=off", str(source), "-o", str(library)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("Grid build failed:\n" + result.stderr)
    receipt = {"utc": datetime.now(timezone.utc).isoformat(), "source": str(source),
        "source_sha256": sha256(source),
        "included_source_sha256": {"native_score.cpp": sha256(source.with_name("native_score.cpp"))},
        "library": str(library), "library_sha256": sha256(library), "command": command,
        "compiler": subprocess.check_output(["/usr/bin/clang++", "--version"], text=True),
        "stdout": result.stdout, "stderr": result.stderr,
        "platform": platform.platform(), "architecture": platform.machine()}
    path = output / "build.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    return path


class GridScorer(NativeScorer):
    def __init__(self, library):
        super().__init__(library)
        doubles, integers = ct.POINTER(ct.c_double), ct.POINTER(ct.c_uint64)
        base = [doubles, doubles, ct.c_uint64, doubles, ct.c_uint64, ct.c_double]
        error = [ct.POINTER(ct.c_char), ct.c_uint64]
        self.lib.ng_select.argtypes = base + [integers, ct.c_int, ct.POINTER(NGResult)] + error
        self.lib.ng_validate.argtypes = base + [ct.POINTER(NSValidation)] + error
        self.lib.ng_select.restype = self.lib.ng_validate.restype = ct.c_int

    def select(self, x1, x2, matrices, tau2, permutation, mode):
        arrays, arguments = self._inputs(x1, x2, matrices, tau2)
        pointer = ct.POINTER(ct.c_uint64)()
        if mode == 1:
            permutation = np.ascontiguousarray(permutation, dtype=np.uint64)
            if permutation.shape != (len(arrays[0]),):
                raise ValueError("permutation must have one index per point")
            pointer = permutation.ctypes.data_as(ct.POINTER(ct.c_uint64))
        result = NGResult()
        self._call(self.lib.ng_select, arguments + [pointer, int(mode), ct.byref(result)])
        output = record(result.base)
        output.update({name: getattr(result, name) for name, _ in NGResult._fields_[1:]})
        if not all(np.isfinite(output[k]) for k in ("score", "setup_seconds", "scoring_seconds")):
            raise RuntimeError("grid returned nonfinite result")
        if not 0 <= output["winner"] < len(arrays[2]) or min(output["setup_seconds"], output["scoring_seconds"]) < 0:
            raise RuntimeError("grid returned invalid winner/timing")
        return output

    def validate(self, x1, x2, matrices, tau2):
        arrays, arguments = self._inputs(x1, x2, matrices, tau2)
        result = NSValidation()
        self._call(self.lib.ng_validate, arguments + [ct.byref(result)])
        output = record(result)
        if not np.isfinite([output["max_lower_violation"], output["max_upper_violation"]]).all():
            raise RuntimeError("grid validation returned nonfinite diagnostics")
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", required=True)
    parser.parse_args()
    print(build_grid())
