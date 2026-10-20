# 0026: conditional nonminimal-calibration diagnostic

This is preparation for the [prospective protocol](../experiments/0026-calibration-moments/protocol.md),
not scientific-run or manuscript admission. The [qualification evidence](../results/0026-preparation/README.md)
contains 44 arithmetic/lift/control/optimizer/pipeline tests plus 8 native-bridge tests,
all passed on manufactured/fake inputs. The first native build failure and its
compiler-only repair are retained. No new real-pair accuracy result follows from
these tests; exact-source freeze and passing CI remain required before execution.

The narrow question is whether a nonminimal, pose-independent affine-depth
calibration initialization can improve subsequent native relative-pose refinement
on an already fixed robust support. Complete product-weight pair-distance loss
has an exact constant-size moment representation. A small convex lift supplies
numerical lower-bound/recovery diagnostics; this implementation does not supply
a rigorous floating-point global certificate. Proper Procrustes profiling also
has constant-size evaluation after moment construction and is a mandatory strong
control, alongside the existing author-recommended truncated-Cauchy estimator
and its continued-refinement control.

The ingredients have substantial prior overlap. [RePoseD: Efficient Relative Pose
Estimation With Known Depth Information](https://openaccess.thecvf.com/content/ICCV2025/papers/Ding_RePoseD_Efficient_Relative_Pose_Estimation_With_Known_Depth_Information_ICCV_2025_paper.pdf)
(ICCV 2025; [precursor §3.1](https://arxiv.org/html/2501.07742v1)) and
[Relative Pose Estimation through Affine Corrections of Monocular Depth Priors](https://arxiv.org/html/2501.05446v3)
(MADPose, CVPR 2025, §4.1) already eliminate pose through interpoint distances and
lift affine-depth monomials. The checked official nonminimal refinement paths use
pose-initialized joint optimization; that bounded implementation difference is
not an exhaustive novelty finding.

[Optimal Pose and Shape Estimation for Category-level 3D Object Perception](https://arxiv.org/html/2104.08383v2)
(PACE*, RSS 2021, §IV, Eqs. 3–4 and 21) already eliminates nuisance parameters
from a weighted nonminimal objective and constructs a small SDP.
[Optimal and Robust Category-level Perception](https://arxiv.org/html/2206.12498)
(PACE, §§VI–VII) is an adjacent precedent for eliminating nuisance parameters,
certifiable relaxations and robust perception, with different CAD-library inputs.
[Linear-Time Gromov Wasserstein Distances using Low Rank Couplings and Costs](https://arxiv.org/html/2106.01128)
(§2 Eq.3, §3) already exploits low-rank squared-distance structure. Moment
compression, variable projection, lifting and generic robust outer loops are
therefore not claimed as new principles.

Remaining risks include collapse near constant depth, reflected congruences,
correlated pair noise and a distance objective that can improve while pose worsens.
The protocol retains all 80 previously exposed development pairs and every failed
proposal, seals predictions before saved-label scoring, and compares the fixed
method against the strongest prescribed controls. A failure closes this variant
without retuning the same outcomes. A pass only permits separately frozen,
matched-cost validation on untouched sources; it does not satisfy the final
two-source improvement requirement or justify writing a paper.
