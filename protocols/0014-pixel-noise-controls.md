# 0014: separate continued optimization from a known pixel-noise correction

## Question and prior-work boundary

0013 measured AUC@5=81.169622 percent with all16 development errors below5
degrees, mostly translation error. These outcomes are already known. This
prospective diagnostic asks whether ordinary continued optimization or an
existing change of noise units explains some of that error. It is not a new
method, untouched validation, or paper admission.

Covariance-aware Sampson error is existing literature, including
[Tangent Sampson Error, ICCV2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Terekhov_Tangent_Sampson_Error_Fast_Approximate_Two-view_Reprojection_Error_for_Central_ICCV_2023_paper.pdf)
and its covariance derivation. For PINHOLE cameras the proposed P control below
is ordinary Sampson error of the raw-pixel fundamental matrix. Current PoseLib
main also contains a tangent-Sampson option, unlike the pinned2.0.5 wrapper.
Neither a good result nor this version difference may be called novel or SOTA.

## Frozen cohort and four fixed outputs

Keep precisely the same16 manifest keys, original saved poses and saved
pre-final RANSAC inlier masks from0013's successful sealed operation. Do not
select difficult/positive pairs, resample points, refit masks, rerun RANSAC,
change calibration, read confidence or access any held-out numeric array.
The predictor reads only the already admitted x1/x2,K1/K2,camera fields for
these keys. It must not load0013 evaluation.json or any R/t label.

For every pair retain four outputs:

1. Original frozen PoseLib baseline.
2. Native continuation: PoseLib2.0.5 refine_relative_pose, raw pixels and
   authored cameras, starting from the original saved pose on its saved mask.
3. N: independent NumPy optimizer with the same normalized-coordinate Sampson
   objective, expressed in common pseudo-pixel units.
4. P: the same NumPy optimizer with raw-pixel Sampson error.

All three continuations start independently from output1; they are not chained.
Build the NumPy effective PINHOLE K matrices from the exact same authored
camera parameters passed to native. Retain saved K only for the unchanged
0013 consistency check (which allowed1e-6 differences); record both. This
prevents calibration discrepancies from being mistaken for a noise-model effect.
Native options explicitly set CAUCHY,loss_scale0.5,max_iterations100,
gradient_tol1e-10,step_tol1e-8,initial_lambda1e-3,min_lambda1e-10,
max_lambda1e10,verbosefalse. The standalone wrapper defaults to scale1, so an
empty dictionary is not equivalent to the prior baseline's final step.

## Objectives and implementation controls

Let x=K1^-1[u,1],y=K2^-1[v,1],E=[t]xR,e=y^TEx,
g1=(E^Ty)[:2],g2=(Ex)[:2],fj=(fxj+fyj)/2 and a=(1/f1+1/f2)/2.

- rN=e/(a*sqrt(||g1||^2+||g2||^2)).
- rP=e/sqrt(g1x^2/fx1^2+g1y^2/fy1^2+g2x^2/fx2^2+g2y^2/fy2^2).
- The dimensionless common objective is C=sum(log1p((r/0.5)^2)).

Recompute C_N and C_P for every output, rather than equating native statistics
in different units. Native normalized Cauchy cost equals0.25*a^2*C_N.
N/P use identical left-SO3 and unit-S2 local retraction, analytic residual
Jacobians and LM with initial lambda1e-3; diagonal Hessian scaling floor1e-12,
lambda/10 on accepted cost decrease and*10 on rejection, bounded1e-10..1e10.
At most100 iterations including rejected steps. Stop at actual cost gradient
infinity norm<=1e-8, step norm<=1e-8 or relative cost decrease<=1e-10.
Reject nonfinite values, collapsed residual denominators and invalid poses.
No scale/grid/threshold/iteration sweep or result-dependent parameter change.

This is an independent optimizer, not a native trajectory reproduction.
Artificial tests must check both analytic Jacobians by central differences,
N=P with all four focals equal, raw-pixel F equivalence of P, finite proper
poses, and monotonic accepted objective values. Native's right-SO3 and
translation-scale treatment are separately disclosed.

Before attributing a difference to N-versus-P noise modeling, require on every
pair: both return valid poses without objective increase; C_N(N)<=C_N(native)
+1e-6*max(1,C_N(native)); neither NumPy branch exhausts its iteration budget;
and its recomputed local Gauss-Newton stationarity proxy satisfies D<=1e-8,
where W=diag(1/(0.25+r^2)),g=2J^TWr,H=2J^TWJ and
D=g^T H^-1 g/(2max(1,C)). Compute via an undamped Cholesky solve; nonfinite
or non-positive-definite H fails this interpretation gate, without a
pseudoinverse or damping rescue. D is a local IRLS/Gauss-Newton model diagnostic,
not a bound on true remaining improvement or proof of global convergence.
Record all final costs, raw gradients, D values, termination
reasons, accepted/rejected counts and N/native rotation/translation disagreement.
Failure of this interpretation gate does not delete outputs or permit reruns;
report convergence confounding and withhold a causal noise-model conclusion.

Before any0014 real call, artificial tests showed a raw angular-gradient/C
threshold was not scale appropriate: optimized noisy fixtures had raw ratios
about0.005/0.003 but tiny GN steps. The prospective D check separates these
states (about2.91e-12/1.20e-12) from their poor initial poses (about0.19).
This correction changes only interpretation, not either optimizer's fixed
stopping rules. No real outcome was available when choosing it.

## Resource, sealing and scoring rules

Use one thread, roughly8GiB memory,90seconds per pair for all three continuations
and30minutes for the complete operation. Preserve actual subprocess exits,
watchdog failures, API/kernel times, loading/setup and parent wall times.
The existing0013 watchdog may be reused as unchanged infrastructure, with its
source hash recorded. No source from the rejected0012 scorer is integrated.
Exclusive output paths and a permanent start marker prevent silent repeats.

Preserve an incomplete/ineligible seal on any computational/input failure;
do not replace that pair or use a missing branch's original pose as a silent
fallback. Only after all16 outputs are sealed may the separate scorer load
the already-public0013 evaluation artifact and compare every output with its
saved GT. The scorer need not reopen HDF5. Bind all source/protocol, original
baseline seal/prediction, manifest and output hashes. Hash the actual input
HDF5 opaque bytes before predictor access as in0013.
The saved evaluation SHA256 is
f8fe46392810b78d485a8dd90ae1849eae6491eaba620d2c9454b6c0f15cb5a4.

Report all16 paired rotation/directed-translation/max errors and AUC@5/10/20
under the unchanged DMS convention. Compare P with original, native and N;
do not select the best branch per pair using GT. Report complete timing and
avoid equating100iterations with matched compute. One scene gives descriptive
paired differences only, no population confidence interval or generalization.

A gain from P is evidence for an existing control, not a contribution. A loss
or null result is preserved. No new correlated-bias method is admitted here:
estimating non-diagonal correspondence covariance from residuals could confuse
pose error with matching bias, and needs independent information or a specific
identifiable model before a later experiment. No manuscript follows0014.

## Disclosed arithmetic-only repair before any N/P outcome

The first frozen32f3e40 operation stopped at pair0's initial N evaluation with
`nonfinite or collapsed residual arithmetic`. Its native continuation had
completed, but neither N nor P returned a pose and no0014 scoring ran. Preserve
`results/0014-pixel-noise/` and its ineligible seal without rewriting anything.

Separate bounded artificial diagnostics reproduced false floating-point flags
in the current NumPy2.0.2/Accelerate backend for large Nx3 and normal-equation
matrix products. An independent ones(10000,3) times identity(3) also returns
exact ones while flagging divide-by-zero, overflow and invalid values. This
matches the class of [upstream reported issue](https://github.com/numpy/numpy/issues/28687),
but artificial validity does not establish that the original real pair's full
arithmetic was valid. Its abbreviated original exception is preserved.

Replace only observation-count-dependent matrix products with mathematically
equivalent `einsum(..., optimize=False)` contractions in the N/P kernel and
common cost/gradient/GN diagnostics. Preserve strict exception scopes, explicit
finite and denominator checks, precision, original poses/masks, both objectives,
optimizer parameters/stops, interpretation gates and native library. No guard
is suppressed, no output clipped, and no dependency upgraded. Summation order
and runtime may change; make no bitwise or performance-equivalence claim.
Large artificial N/P optimizations and contraction equivalence tests must pass
before separately freezing one technical retry in
`results/0014-pixel-noise-arithmetic-repair/predictions`.

The new cohort executes all three continuations once per pair; explicitly count
the extra native pair0 call from the first attempt (17 native calls across both
attempts). Do not conceal its cost or mix its partial record into the new seal.
The retry is a technical repair decided without any N/P pose outcome, not a
choice among quality results. Any further failure remains visible.
