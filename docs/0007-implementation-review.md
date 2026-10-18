# 0007 prospective implementation verification

October 18, 2026. AI-assisted software verification only; no scientific gradient
collection, saved-weight load, new asset acquisition or paid service occurred.

The implementation has independent core, NumPy scorer and cache-only supervisor.
Four estimators share the same sampled occurrences and augmentation seed schedule;
the same-patch control explicitly duplicates half the occurrences. Tiny artificial
fixtures check actual Gaussian-prefix coupling, target signs, manual loss/bias
gradient, sampling/ordering, model immutability, RNG isolation and partial failure.
The scoring fixtures check trace covariance against coordinate-wise variance,
Gram/count bootstrap against explicit resampled gradients, paired projection
standard errors, degenerate inputs, timing gates and all-controls intersection.

Integrated `python -W error -m unittest discover -s tests -p
 'test_antithetic_gradient*.py'` passed **43 tests, zero skips**, in **1.024s**
using existing local Python3.9.6 / NumPy2.0.2 / Torch2.8.0. This does not verify
cloud Torch2.10.0+cpu: the same fixtures must pass there before scientific work.
The scorer's first isolated artificial test incorrectly demanded exact variance
invariance after a 1e10 floating-point shift; its roughly 1.2e-12 discrepancy led
to an explicit numerical tolerance in that test. NumPy small-matrix BLAS warnings
were avoided by transparent einsum contractions, checked against direct formulas;
no warning suppression or scientific outcome selection occurred.

The first full local suite had **549 passed / 7 failed** in **49.47s**. All seven
failures were existing 0003 path-binding tests whose temporary roots passed through
macOS's `/var` symlink; their strict guards rejected those paths. No old frozen
source or test was changed. A second full run uses the resolved physical temporary
root through `TMPDIR=/private/tmp`: **556 passed in 48.38s**, actual exit 0.
This is a physical-path correction to the test environment, not relaxed guards.

Independent supervision review required fixes before freeze: reject a complete
array grid if its receipt failed or weights changed; bind subprocess exit/logs,
actual private command and receipt to the study record; require the single fixed
sibling output and matching started-study marker; require exactly four affinity
slots and verify completed-mask schema. Synthetic tamper checks cover those
saved-evidence paths. No experiment was run before these repairs.

The 900-second alarm kills an active case before exiting. A hard interruption may
leave an incomplete execution receipt; preserve it as failure, never retry or
infer success from directory existence. Actual command argv stay private with
SHA bindings and a reproducible public placeholder recipe; raw image/weight/
gradient artifacts remain in the private cache.
