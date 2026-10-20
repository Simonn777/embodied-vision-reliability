# Independent review of the 0024 prospective falsifier

2026-10-19. Protocol/source and public receipt metadata only; no scientific RGB,
calibration values, depth, scored errors, labels, inference, or estimator calls.
This reviews the protocol before implementation admission, not a successful run.

**Conclusion:** the protocol supports a bounded test of cross-axis transfer of
an approximate predictor-error model. It does not test physical depth accuracy,
pose improvement, novel geometry, or paper readiness. Preparation may proceed;
the completed implementation and exact source freeze still require review.

The principal mathematical ambiguity found during review was corrected before
implementation: with nuisance design `[d_j/h_j, 1/h_j]`, the nonnegative-scale
boundary is `s_j=0`, `t_j=dot(1/h_j,d0+t0)/dot(1/h_j,1/h_j)`. It is **not** the
mean of the target in base coordinates. The protocol now gives this formula and
the appropriate constant-in-virtual-axial-depth interpretation. Before any run,
the two-column nuisance criterion was fixed to column-L2-normalized SVD with
`sigma_min > max(n_rows,2)*eps_float64*sigma_max`. This permits the stronger
two-parameter baseline except at numerical rank loss; it does not apply the
five-parameter scientific excitation threshold to weaken that control.

The joint five-parameter model, scale gauge, homography direction, and axial
factor are consistent. Full column rank is conditional identifiability within
the affine-error model, not truth or calibrated uncertainty. Yaw and pitch
transfer fits have the same two nuisance parameters as their zero-offset
controls, share support, and exclude evaluation cells from the fit objectives.
Full-image medians and positivity guards nevertheless use the declared full
image; this is a held-out-grid evaluation with fixed transductive preprocessing,
not independent data or an IID validation sample. A coherent wrong shape can
pass both axes. OLS attenuation and image resampling remain real confounders.

The camera construction is a useful substantive control: identical centered,
square-pixel canonical cameras make the installed MoGe `fov_x` interface
applicable, while direct original-to-view remapping avoids compounded warps.
Implementation must verify the full coordinate map and expected normalized
intrinsics. The algorithm must decode only the 27 selected RGB frames and parse
only their numeric intrinsics. The earlier acquisition's complete member hashes
can be verified opaquely; its reference images need not be decoded.

Metadata checks performed here confirm the existing plan SHA256
`cc9b02cac8946107f1c945643b316695e5ebbf6339b34baf737541f8cec886d5`
and original acquisition SHA256
`bf6cd9baafa4b8c72fc3a33021becc3cf37f3bf3a6ecec406df77c4b25114905`.
The plan has three scenes and eight queries per scene, hence 27 unique selected
frames including anchors. The preserved qualification-05 receipt is completed
at `fa454585e67a8f0395ad5f67a03d5a28cc33a66a`. These checks do not replace the
runtime's complete original acquisition/model/source-chain verification.

All 54 image-directions remain in the denominator. Unavailable offsets tie the
baseline; unavailable support has a declared common fixed error mass. These
rules prevent dropping difficult cases, but the aggregate is an operational
model-test score, not a depth metric. Requiring 19/27 usable images, coverage in
each place, a 10% aggregate reduction, and scene safeguards is a prospective
development decision rule. Passing would only justify consideration of a later
pose experiment with the stronger controls already listed in the protocol.

Exactly 135 depth forwards, no matcher/pose calls, no new payload acquisition,
and charged loading/warping/serialization give a small, coherent resource scope.
The 4-thread, 8-GiB, 1800-second limits remain hard execution limits rather than
promised runtime. The saved-output seal must precede gate computation; technical
failure cannot be reclassified as a model-test success. No old failed variant or
scientific result is changed by this admission.

An initial source-only pass of the new wrapper found the planned 27-image
metadata selection, selected-only intrinsics parsing, old command/input/model
binding, pre-call attempt accounting, partial array preservation, post-guard
byte checks, and output-seal-before-summary order consistent with these rules.
The new supervisor's resource/environment/cleanup body is unchanged from 0023;
its differences are operation/helper paths, schema and the added fixed protocol
binding. This is not final numerical-kernel admission or a substitute for the
pending focused artificial checks.

## Final implementation review before freeze

No mathematical or orchestration blocker was found at these independently read
source hashes:

| File | SHA256 |
|---|---|
| `math_core.py` | `9e6f18fc1781c39e087ee8f40edfac79963fac5068ff4f99ec9b96b0897ae737` |
| `run.py` | `56db02536178762d540bef75feceef2eae8ea6a5fabac1f8d3bde2c1bbe53f1f` |
| `supervise.py` | `3e7124bfccf8642c40d212b62c1bb638fdd094e060e291fa0a8813f2f97f3bcd` |
| Protocol | `4ae8f943112afa705a072e194e94a424252c282dd6beff54606584310007da3b` |

The kernel's offset sign, five-column scale-normalized solve, distinct nuisance
rank criterion, positive axial factor, forward correspondence versus inverse
image remap, four-neighbor validity, full-image median gauge, checkerboard fit
exclusion, 99% positivity guard, two-direction fallback, and all-54 aggregation
match the protocol. The wrapper attempts five depth calls per selected image,
requires 135 completed calls, preserves raw/partial arrays, and seals all 27
JSON/NPZ pairs before the gate is computed. Source/input/model postchecks remain
mandatory. The final remap guard compares float32-rounded coordinates after
promotion back to float64, avoiding NumPy scalar-promotion ambiguity; OpenCV
still consumes the prescribed float32 coordinates.

Four additional independent manufactured probes were actually executed with
the existing Python interpreter and `-W error`, with exit code 0 (tool output
chunk `c64178`). They used source imports and manufactured NumPy arrays only:

- Shared-offset coefficients differed from separately specified analytic truth
  by at most `3.6359804056473877e-14`.
- A two-column design with singular ratio `2.2826012860917937e-08` correctly
  remained usable under the machine-rank rule, returning scale
  `2.000000011896782` for analytic scale 2. This checks that the strong nuisance
  baseline is not rejected by the five-column `1e-5` excitation threshold.
- The constrained `t/h` boundary agreed with an independent scalar sum formula
  within `4.884981308350689e-15`.
- Direct homogeneous projection and the remap agreed within
  `1.4210854715202004e-14` pixels on a manufactured off-center camera.

The full artificial test matrix was not rerun by this reviewer. These checks
establish limited implementation consistency, not a formal floating-point proof
or an empirical model-transfer result. The exact-commit CI and prospective
one-shot execution conditions still apply.
