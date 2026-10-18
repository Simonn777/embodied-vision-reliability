# 0003 saved-R half-rounding compatibility analysis v1

October 17, 2026 (America/New_York). AI-assisted exploratory analysis of an already
observed development failure. This method is published before its first actual
saved-R execution; exposure to the prior R and failure is acknowledged. It is not
an independent sample, confirmatory test or a new geometry audit.

## One question and one proposal

Can one fixed exact proper-rotation proposal round to all nine original stored
frame-zero R coefficients under independent nearest-even binary16 rounding and
exact f32 widening? A successful exact certificate proves existence under this
model, not this sample's actual generation history. If the fixed proposal fails,
the answer remains unresolved. No infeasibility test or causal probability is
claimed. All old gates, thresholds, results and markers remain untouched.

Use only the published 25-value camera excerpt bound by the new config. Extract
its nine original upper-left RT f32 byte strings, preserving order, endian and
signed zeros. Bind the excerpt/old diagnostic/config/code hashes in the real-run
receipt before invoking the checker. No NPZ, TAR, cached trajectories, metadata,
RGB, old evaluation set, model or additional source downloads may be opened.

Let a=(1+R00+R11+R22, 1+R00-R11-R22, 1-R00+R11-R22,
1-R00-R11+R22). Select its largest entry with ties w,x,y,z. Sum(a)=4, so the
selected entry is at least1. Let wx=R21-R12, wy=R02-R20, wz=R10-R01,
xy=R01+R10, xz=R02+R20, yz=R12+R21. The corresponding homogeneous quaternion
is (aw,wx,wy,wz), (wx,ax,xy,xz), (wy,xy,ay,yz), or (wz,xz,yz,az).
Build the standard rational quaternion rotation using division by sum(q_i²),
without a square root. Exactly verify both Gram identities and determinant+1.

Construct each stored coefficient's exact half rounding cell using neighboring
representable values and rational midpoints. Powers-of-two spacing is asymmetric;
nearest-even endpoint ownership is determined by the significand low bit. Handle
subnormal and finite-overflow boundaries explicitly. An exact rational zero maps
to +0 by convention; a -0 target requires a negative witness entry, so failure is
conservatively inconclusive. Values off the half grid cannot receive a positive
certificate. Compare the proposal's exact nine coefficients to all nine cells.

Exactly one proposal is allowed, with no optimization, alternative branch trial,
perturbation, fit, repeated search or changed rounding rule. There is no preliminary
interval-infeasibility procedure in this implementation. Save all nine cell checks,
exact candidate entries/quaternion, both Grams, determinant and outcome. The
hypothetical matrix is NEVER used for projection, data qualification, threshold
replacement or as an asserted pre-quantization camera. It is a mathematical
existence witness only, not a repair of the actual observation.

## Execution and review

The standalone stdlib checker was written and independently reviewed using only
artificial test inputs before this actual invocation. Artificial tests cover all
four branches/ties, half grid/rounding endpoints, signs/zeros/subnormals/overflow,
reflection/scale inconclusive cases and byte-order/input rejection. A repository
import-path adaptation of these tests does not change the checker algorithm.

Publish this method/config/checker/tests input commit first. The cloud must fetch
and verify its published bytes, record the executed commit and hashes, then run
once on the saved excerpt under two CPU slots, 6 GiB AS and timeout60. Use a
brand-new output folder; retain errors without trying alternate proposals. Record
actual UTC/elapsed/CPU/RSS/child wait/command/environment and stdout/stderr. Source
keys should be repo-relative from the start. The receipt cannot claim an outer
supervisor exit that was not observed. A saved-result-only independent verifier
must check a claimed positive certificate using exact arithmetic; software-suite
success does not imply a positive actual result.

This analysis consumes no new dataset bytes and does not reset the existing
22,910,156-byte access charge. No installation, paid resource or schedule/permission
change. All numerical outputs derived from PointOdyssey retain CC BY-NC-SA 4.0,
with the same authors, dataset revision and source attribution as the excerpt.
Original software/design prose remains under repository terms. Publish a finding
of mathematical compatibility separately from verified source-generation links.
Even a positive certificate cannot fill a missing archive-conversion history.

Next geometry work remains prospective: review and freeze a distinct descriptive
stored-transform projection consistency plan. Do not execute that plan this turn.
