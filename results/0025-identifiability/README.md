# 0025 manufactured rolling-shutter identifiability

This archive establishes mathematical feasibility and specific degeneracies.
It contains **no real-image accuracy, trained model, novel-method performance,
held-out result or manuscript**. [Scope and next admission](../../docs/0025-admission.md).

## Original fixed numerical check

Six manufactured full-unknown cases and eight known-R/omega cases were declared
in source before the single execution. No random seed, fitting or fixture search
was used. Exposure times solve the physical implicit row equation; rays, row
times and predicted depths remain fixed during parameter differentiation.

| Full unknown case | Points | Rank / 21 |
|---|---:|---:|
| General motion | 7 | 21 |
| General motion | 8 | 21 |
| Zero intra-readout twists, nonzero inter-frame pose | 8 | 21 |
| Zero readout time | 8 | 9 |
| Vertical flow | 8 | 19 |
| Equal row times | 8 | 15 |

| Known R and angular rates | Four-point rank / 12 | Five-point rank / 12 |
|---|---:|---:|
| General motion | 12 | 12 |
| Zero intra-readout twists | 12 | 12 |
| Vertical flow | 11 | 11 |
| Identity reference rotation | 11 | 12 |

Source SHA-256: `0b058405fbc151b8765dafdb4e4a81f8c7e48f515da3d217b2df33b9a9565683`.
Original result SHA-256: `befcf2f368ad964324bee7f72c2179c5487212c9899c1cd39d276fceccf2f7c5`.
Outer wall time 0.337930375 s; inner 0.245473667 s; exit 0; empty stderr.
Four CPU thread limits, 120 s timeout and 8 GiB sampled RSS limit applied.
The 25,056 KiB sampled peak is not a continuous memory maximum.

The [independent audit](audit-01/README.md) reconstructed every known-R matrix
from the saved observations, checked spectra/thresholds and bindings, and examined
translation-direction ambiguity. Its first invocation passed in 0.083567084 s.
The last saved Jacobian can be recomputed from its array; earlier Richardson
rank counts can only be checked against their saved spectra. Finite-difference
cross-step agreement is not a certified error bound.

## Exact four-point existence extension

This extension was declared **after** the first numerical result. It uses the
same first four points and stated constants, with a fixed rational Cayley
rotation. All geometry, row times, matrix entries and elimination use exact
fractions. The complete 12-by-12 determinant is nonzero and the unique solved
vector exactly matches the physical generating parameters. It completed once,
exit 0, 0.107683834 s outer / 0.00802825 s inner, empty stderr. A root-run, algorithmically independent
integer Bareiss computation agrees with the exact determinant after 506 exact
divisions and verifies the saved solution residual is exactly zero. This is an
algorithmic cross-check, not an independent-team replication.

Source SHA-256: `f91dc813e7deee89e44992ba0b3522eb68567d7ab9e8b1a440ca9271b2bb9a17`.
Result SHA-256: `57ec1642d82ab8afbc7ad900caa4702fa64119dc6f266b248bb722fff47ca0ef`.
The large numerator/denominator are preserved in [the exact record](exact-01/mathematics.json).
This is an existence result for all 12 parameters under given orientations,
not a proof of the minimum sample size for translation direction, a generic
global uniqueness theorem, or a noise-robustness result.

## Reproduce

Only Python and NumPy are required for the numerical check; the exact check uses
only Python's standard library. From the repository root, choose fresh paths:

```sh
python experiments/0025-rs-identifiability/check_manufactured.py --output-dir scratch/0025-numerical
python experiments/0025-rs-identifiability/check_exact_four_point.py --output-dir scratch/0025-exact
python experiments/0025-rs-identifiability/audit_exact.py scratch/0025-exact/mathematics.json
```

To independently audit the original saved numerical arrays without regenerating
observations, use a fresh temporary directory. The auditor creates audit-01 and
intentionally refuses to overwrite an existing result:

```sh
check_dir=$(mktemp -d)
cp experiments/0025-rs-identifiability/check_manufactured.py experiments/0025-rs-identifiability/audit_saved.py "$check_dir/"
cp -R results/0025-identifiability/manufactured-01 results/0025-identifiability/execution-01 "$check_dir/"
python "$check_dir/audit_saved.py"
```

[Publication bindings](publication-bindings.json) retain original and public
SHA-256 hashes. Known absolute paths are replaced only in public execution
receipts; original receipts remain private. The original audit's execution hash
therefore names the original receipt, while a fresh audit of the public copy
will bind its redacted receipt. Scientific arrays and source scripts are
byte-identical. Source-only notes preserve their pre-numerical wording; the
[current admission note](../../docs/0025-admission.md) gives the later status.
The original independent Bareiss check ran as a short inline command; the
published audit_exact.py packages the same integer-elimination calculation for
reproduction and CI. It does not regenerate observations or import the generator.
