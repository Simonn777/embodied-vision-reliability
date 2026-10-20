# Direct TC refinement bridge — source preparation

No build or tests have been executed at source preparation time. This wrapper
does not call RANSAC, reselect support, normalize translation, or load models.
It links the existing pinned PoseLib archive. Preserve the upstream BSD notice
in NOTICE.md with any distributed binary.

API: `NativeRefiner(library_path,build_receipt_path).refine(x1,x2,d1,d2,R,t,a,b,c,
focal1,focal2,max_iterations=100)`. xy is **already calibrated Nx2**; R is
row-major and maps camera1 into camera2. Source lifted points are `(d1+b)r1`;
target points are `(a*d2+c)r2`. Mapping to native fields is `(a,b,c/a)`.
Both corrected input forms are checked for finite positivity, including the
native rounding order `a*(d2+c/a)`. Raw negative depth is permitted. Support
count must be 4..1,000,000; max_iterations is integer 0..100. Zero iterations is
reserved for manufactured mapping/cost checks; the real continuation default is
100. No caller-supplied weights or mask are hidden inside the bridge.

The fixed options exactly match the 0019 final stage: TC, reprojection weight
1/64, Sampson weight1, refine_shift=true, empty/native uniform weights, normalized
loss scale `.25*2*(1/focal1+1/focal2)`, native remaining defaults. This differs
from the old estimate API, which resets the supplied pose and runs additional
LO logic. Native negative transformed-Z residual omission is preserved.

Return includes raw R/q/t/scale/shift1/shift2, equivalent a/b/c, all eight native
BundleStats fields, settings, timings and positive-output-depth counts. Finite
but physically invalid output is retained with `physical_admissible=false`;
it is never repaired. Status1 is input rejection before the native call;
status2 is a native/C++ exception; status3 is nonfinite/invalid native output.
`RefinementError.as_record()` retains raw evidence with explicit null plus named
NaN/Inf locations for strict JSON. A status1 physical-seed rejection is an
explicit unavailable candidate after the caller has validated shapes/count/ABI.
Shape, ABI/load/source-binding faults remain technical failures; status2 is an
unexpected exception; status3 is a retained numerical failure.

`build.py` creates a fresh output directory, checks a metadata-only lock of
the existing archive/compiler/header trees, compiles only refiner.cc, records
the exact command and source/dependency/library hashes, and checks for mutation
afterwards. No upstream edits or dependency downloads. Its timeout is180s;
execute under the parent's bounded resource supervisor. The artifact path and
receipt are explicit inputs to the Python loader. Build receipt is local and
contains absolute compiler/include paths; sanitize only a separate public copy.

After source review and explicit parent admission, intended commands are:

```
$PYTHON native_bridge/build.py --project-root <asset-workspace-root> \
  --output-dir <fresh-private-build-dir>
PYTHONPATH=<native_bridge-dir>:<existing-runtime> $PYTHON -W error \
  native_bridge/test_bridge.py --library <build-dir>/libevr0026_refiner.dylib \
  --build-receipt <build-dir>/receipt.json
```

Tests are fixed manufactured data only: exact nonidentity pose/affine mapping,
negative raw but positive corrected depths, independent scalar TC cost including
inactive-Z/count behavior, only supplied support, native input guards, ABI/source
hash checks, finite statistics, no input mutation and error retention. The test
runner requires a supplied native library instead of silently skipping native
checks. Keep the first build/test logs and receipts, including failures. Passing
these tests qualifies an API bridge only, not task accuracy or a new method.
