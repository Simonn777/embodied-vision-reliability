# 0017: prior art for state-dependent depth-residual omission

2026-10-19; bounded primary-source review. No 0017 outcomes, cache arrays,
images, reference poses, or estimator calls were accessed. This note does not
claim the source property causes real pose error.

**Verdict:** admit only the already specified baseline diagnostic and, if its
prospective conditions pass, a causal refinement audit. Do not admit “positive
scale + fixed residual support” as a new method: MADPose already implements
both in the same affine-depth relative-pose problem. Combining established
ingredients could still be useful, but a contribution must demonstrate an
additional mechanism and substantial improvement beyond that close control.

## Closest sources and what was actually read

**Yu et al., CVPR 2025, MADPose.** The
[paper, arXiv v3, §§3–4.4, Eqs. 4 and 7–10](https://arxiv.org/html/2501.05446v3#S4)
models corrected depths `d1+beta1` and `alpha(d2+beta2)`, filters solver solutions
for positive corrected depths, and combines bidirectional depth reprojection
with Sampson error. Its hybrid pipeline includes geometric solvers, and its LO
uses three specified inlier sets. Read full relevant sections, not only abstract.

The official implementation was pinned during review to
`c17e179dd4f2794ea7d6942f402f12e15504ead7`:

- [`HybridPoseOptimizer::SetUp`, optimizer.h](https://github.com/MarkYu98/madpose/blob/c17e179dd4f2794ea7d6942f402f12e15504ead7/src/optimizer.h):
  adds Ceres residual blocks once from directional inlier indices; sets
  `scale >= 1e-2`; with the minimum-depth option, bounds each offset below by
  `-min_depth + 1e-2`. These are actual source operations, not proposed fixes.
- [`optimizer_config.h`](https://github.com/MarkYu98/madpose/blob/c17e179dd4f2794ea7d6942f402f12e15504ead7/src/optimizer_config.h)
  and [`estimator_config.h`](https://github.com/MarkYu98/madpose/blob/c17e179dd4f2794ea7d6942f402f12e15504ead7/src/estimator_config.h)
  default `min_depth_constraint` to true.
- [`LiftProjectionFunctor0/1`, cost_functions.h](https://github.com/MarkYu98/madpose/blob/c17e179dd4f2794ea7d6942f402f12e15504ead7/src/cost_functions.h):
  directly projects every constructed residual; there is no conditional
  transformed-z omission in these calibrated functors. This retains support
  but does **not** guarantee physical transformed-depth feasibility: perspective
  projection also produces finite coordinates behind a camera and is singular
  at zero depth. Read the complete two calibrated functors and setup routine.

**Ding et al., RePoseD, ICCV 2025.** The
[paper, arXiv v3, calibrated solver and analysis discussion](https://arxiv.org/html/2501.07742v3)
chooses positive relative scale from the squared-scale solution. It explicitly
discusses physically impossible solutions obtaining better scores and suggests
additional feasibility filtering. This already anticipates a broad
“unphysical optimum” concern; it does not establish the particular accepted-step
omission mechanism in our later pinned implementation. Read those full-text
passages and the stated experimental refinement description. Distinguish paper
version from subsequent code.

For the actual inspected PoseLib revision
`6a636b1d84273d5e3ba97676463476569611572a`,
[`monodepth_relpose.h`, residual and step routines](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/PoseLib/robust/optim/monodepth_relpose.h#L133)
conditionally adds each depth term only when that transformed point has positive
z, while scale/shifts receive additive updates. The
[wrapper](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/PoseLib/robust.cc#L383)
selects final-refinement support using a prior epipolar mask. Therefore
pre-final support, current positive-depth support, and final pose accuracy are
three different quantities. Read the relevant functions directly.

**Established robust optimization and cheirality handling.**
[Ceres' official modeling specification](https://ceres-solver.readthedocs.io/latest/nnls_modeling.html)
already defines bounded robust least squares over constructed residual blocks,
`0.5 sum rho_i(||f_i(x)||^2)` subject to parameter bounds. Fixed support,
robustification, and scalar positivity bounds are ordinary modeling tools.
In the official
[GTSAM 4.2 ProjectionFactor source](https://github.com/borglab/gtsam/blob/4.2/gtsam/slam/ProjectionFactor.h),
a cheirality exception yields a nonzero constant residual when exceptions are
not rethrown, with zero Jacobians. Thus assigning a penalty rather than a zero
contribution behind the camera also has direct precedent; a zero-gradient
penalty need not guide recovery. Read the full relevant `evaluateError` path.

For broader published context,
[pOSE, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Hong_pOSE_Pseudo_Object_CVPR_2018_paper.html)
already changes the reprojection objective to improve initialization/convergence
using a hybrid affine/projective model. This review read its official abstract
only. The indexed primary PDF of
[expOSE, CVPR 2023, §2.1](https://openaccess.thecvf.com/content/CVPR2023/papers/Iglesias_expOSE_Accurate_Initialization-Free_Projective_Factorization_Using_Exponential_Regularization_CVPR_2023_paper.pdf)
explicitly discusses the zero-depth reprojection barrier and cheirality; direct
full-PDF retrieval failed here. Neither limited-depth source is used to claim
an exact solution to the present affine-depth omission case.

## What the source property means mathematically

Let theta contain the raw native `(R,t,s,beta1,beta2)` gauge, and let the fixed
input support be S. For one directional term define

\[
Z_i(\theta)=R[(d_i+\beta_1)h_i]+t,\quad
a_i(\theta)=\mathbf 1[Z_{i,z}(\theta)>0],\quad
q_i(\theta)=\|\pi(Z_i(\theta))-p'_i\|^2.
\]

Ignoring floating-point invalidity for this identity, the implemented gated
cost has the form `C_drop(theta)=sum_S a_i(theta) rho(q_i(theta))`, plus the
other direction and geometric terms. For a step from theta0 to theta1, the
depth-cost change decomposes into residual change on common active terms,
minus the old cost of lost terms, plus the new cost of gained terms. Thus a
decrease can partly reflect lost evidence rather than better alignment.
This algebra does not prove that an optimizer crosses the boundary or that
the resulting pose is worse. The Cauchy singularity near z=0 and the finite
step acceptance rule must be inspected in an actual trace.

A fixed-support capped diagnostic is

\[
\ell_i(\theta)=
\begin{cases}
\min(q_i/\tau^2,1),&Z_{i,z}>0\text{ and projection finite},\\
1,&\text{otherwise},
\end{cases}\qquad
C_{\rm fixed}=\frac{1}{2|S|}\sum_{i\in S}(\ell_{12,i}+\ell_{21,i}).
\]

Replacing an invalid zero contribution by the cap prevents that single term
from improving merely by becoming invalid. **A fixed denominator alone does
not do this** if invalid terms still contribute zero. Also, a constant invalid
penalty does not remove the incentive for an *unbounded* Cauchy term to disappear
when its valid cost exceeds that penalty. The bounded diagnostic must not be
presented as the same objective as native Cauchy refinement. The gap between
capped omission and capped fixed-support scores equals the omitted fraction;
this identity is not an empirical contribution or causal evidence.

Gauge consistency likewise supplies a correctness requirement, not novelty.
Simultaneously changing `d1,d2,beta1,beta2,t` by a positive common unit factor
leaves all projections and depth signs unchanged (s remains a ratio).
Normalizing t alone changes the model. A numerical positive-depth margin must
transform with the depth gauge; a fixed 0.01 in depth units is not unit-invariant.
Reparameterizing `s=exp(gamma)` ensures positive scale but cannot ensure
`[R((d+beta)h)+t]_z>0`. Positive corrected source depths alone also cannot do so.
These facts separate scalar constraints from transformed cheirality.

## Conditional admission and decisive controls

The potentially identifiable question is narrow: **does accepted joint
refinement exploit changes in directional support, beyond proposal error,
and does this explain reproducible loss of pose quality?** Final inactive-term
counts are only a trigger. If 0017 meets its declared trigger, a separate
prospective audit must retain the same initial state and log every accepted
step's raw parameters, native objective, active set, capped fixed-support cost,
and pose error. Include all allocated cases and controls with no support change.
Do not select a favorable initialization or tune visibility rules using labels.

The minimum mechanistic controls are: unchanged native refinement; the same
implementation with MADPose-style scalar bounds only; fixed constructed support
without bounds; their combination; and an explicitly declared transformed-depth
constraint/penalty variant. Keep initialization, observations, loss, iteration
budget and gauge treatment matched so solver replacement is not the explanation.
An end-to-end MADPose comparison remains necessary for any accuracy claim;
its different proposals, support selection, and optimization backend cannot
substitute for the matched causal ablation.

Reject a new-method claim if the gain is exhausted by those known controls,
if no accepted step loses relevant support, if harm originates before
refinement, or if any improvement is numerical overflow cleanup alone. A
distinct contribution would need a demonstrated failure class that survives
the strongest existing treatment and an identifiable remedy preserving useful
depth evidence under outliers, with substantial independent accuracy gains.
No such mechanism, remedy, or gain is established by this review. An ordinary
bug fix remains valuable and should be reported honestly as such.
