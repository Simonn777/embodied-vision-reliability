# 0028 normal-information opportunity test

The [prospective protocol](protocol.md) tests whether a restricted, angular
normal-guided depth correction improves relative pose after camera calibration.
This is a comparison of known constructions, not an established new method.
No scientific result or manuscript is admitted at source preparation.

The fixed population is 27 existing Map-free development images and 24 query
pairs. Eleven arms include unmodified essential-matrix pose, three camera
controls, scalar LS/IRLS/angular correction, full normal integration, two
normal-guided pose controls and the existing free-affine native refiner.
Unavailable predictions remain in the denominator with explicit fallback.

Execution is separated into four programs: `frontend.py` saves one raw forward
per image; `predict.py` saves all corrections and poses; `evaluate.py` opens
already-development truth only after complete clean prediction seals;
`audit_saved.py` independently checks the saved metrics and decision. Every
program requires `--job`, `--expected-commit` and a fresh private `--output`.
An outer supervisor limits each operation to four CPU threads, 8 GiB and
at most 1,800 seconds. Exact-source CI is required before scientific execution.

Run artificial checks with `python -W error -m pytest -q
experiments/0028-normal-information`. Tests do not fetch data, train models or
run the scientific cohort. The optional real MoGe helper compatibility check
requires the previously qualified local source/runtime; ordinary CI skips it.

Private job paths identify existing admitted caches, model/runtime and native
artifacts; none are embedded here. Images, feature arrays, weights and upstream
truth are not rehosted. Qualification receipts and source review are recorded
in [the admission note](../../docs/0028-admission.md).

A passing development gate permits further contribution assessment and a
separately frozen validation only. The final two-source, matched-cost held-out
improvement gate remains required before writing a manuscript.
