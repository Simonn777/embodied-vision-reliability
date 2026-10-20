# Rolling-shutter relative pose with two affine depth priors — source-only screen

2026-10-19. No dataset payload, numerical scientific arrays, model execution, solver, or implementation was used. **Immediate experimental admission: NO_GO.** This is not a proof that the narrow problem is already solved or generically unidentifiable. A bounded primary-source search did not find an exact joint solver; it also did not establish two independent, licensed, small datasets satisfying the required pose/readout contract. Novelty remains unresolved.

## Nearest prior and the actual gap

* **Barath, Rolling Shutter Relative Pose Estimation Made Practical, ECCV 2026.** [Paper](https://arxiv.org/html/2606.26863v1), [author code](https://github.com/danini/rolling_shutter_made_practical). Section 3, Eqs. 1–3 model row-dependent rotation/translation and an essential matrix with 17 degrees of freedom: five reference-pose and twelve velocity parameters. Seven affine correspondences provide its robust-estimation solver. Section 4.2 uses linearization and a pose-dependent elimination approximation; it is not an exact constant-twist solution. No learned-depth correction appears in its formulation. This is a mandatory RS baseline, not GS PoseLib alone. Its C++ estimator is CPU code under MIT; the supplied real feature pipeline uses RoMa/CUDA. An existing small affine-correspondence cache suitable for our inputs has not been verified.
* **MADPose (CVPR 2025)** [§3, Eqs. 1–4](https://arxiv.org/html/2501.05446v2) and **RePoseD** [§3, Eqs. 1–4](https://arxiv.org/html/2501.07742v3) already estimate independent affine depth corrections with calibrated relative pose. Both use one rigid transform for the entire image pair. MADPose's pairwise-distance elimination relies on points being expressed in one camera frame; it cannot be reused unchanged for row-camera points. RePoseD correctly retains three translation degrees of freedom after fixing the depth-scale gauge.
* [Order-One Rolling Shutter Cameras](https://arxiv.org/html/2403.11295v2), [Single-Scanline Relative Pose Estimation, ICCV 2025](https://arxiv.org/html/2506.22069v1), and [Single-View Rolling-Shutter SfM, 2026](https://arxiv.org/html/2603.11888v1) already cover substantial RS geometry/minimal-problem theory. Their point/line formulations do not supply the proposed two affine learned-depth priors. Their rational/polynomial or scanline models must not be equated with an exact SE(3) constant-twist trajectory.
* The broad combination is not itself a new research idea: an [official CTU/Sattler project description](https://cyber.felk.cvut.cz/study/student-projects/detail/?spid=1280&xml=1) explicitly proposes predicted single-view depth for RS pose estimation. It is an advertised project, not evidence of a completed algorithm.

The defensible unresolved target would therefore be a **specific identifiable joint RS/depth-affine problem with a useful solver**, not “combine RS and monocular depth,” or simply add twelve motion variables to nonlinear least squares.

## Model and identifiability limits

Let calibrated rays be \(r_{ij}=(x,y,1)^T\), known row times \(s_{ij}\), corrected axial depths \(z_{ij}=a_i d_{ij}+b_i>0\), and row world-to-camera transforms \(C_i(s)=\exp(s\hat\xi_i)C_i(0)\). Each correspondence requires

\[
 z_{2j}r_{2j}=C_2(s_{2j})C_1(s_{1j})^{-1}[z_{1j}r_{1j};1],
\]

with the affine action understood on the right. Fix the first reference camera. Two independent six-dimensional twists, a six-dimensional relative pose, and four affine coefficients yield 22 parameters, minus one common spatial-scale gauge: **21**. Seven 3D consistency equations are only a counting lower bound; they do not prove finite solutions, local rank, or a practical minimal solver. Fixing both \(a_1=1\) and \(\|t\|=1\) would remove a physical degree of freedom. A common trajectory across frames is a different, more constrained problem; unknown line time and physical velocity cannot independently be recovered when only their products are observed.

An exact special ambiguity exists even without angular motion. Write row time as normalized ordinate and

\[
z_2r_2=z_1r_1+t+y_2v_2-y_1v_1.
\]

If \(x_2=x_1\), then simultaneously adding \(\delta\) to both depth offsets and \(\delta e_y\) to both velocities leaves every equation unchanged: \(\delta(r_2-r_1)=\delta(y_2-y_1)e_y\). Positive depths can be preserved for small \(\delta\). Thus vertical-flow data cannot identify this offset/velocity mode. Equal-row correspondences likewise fail to separate common translational motion in this setting. These are explicit degeneracies, not a claim of generic failure. A prospective generic-rank test would need diverse row pairs, depths and motions, and must check functional reference-pose observability separately from nuisance identifiability.

The more immediate modeling risk is that \(d_{ij}\) must represent depth in **that pixel's exposure-time camera**, not a reference-row GS reconstruction. The [2026 Anchor Points for Monocular Depth Estimation Under Severe Rolling Shutter Ego-motion author page](https://focus.ece.ufl.edu/research/2026/rolling-shutter-anchor-point/) makes this temporal/geometric reference central and adapts a foundation model. Its page did not expose a verifiable paper/dataset download contract in this screen. It does not justify assuming frozen MoGe on distorted RGB has a single per-image affine error relative to row depth. More flexible motion can otherwise absorb network shape errors.

## Two independent small licensed RS datasets: not established

| Source | Verified useful contract | Why it is not yet an admitted small, licensed independent test |
|---|---|---|
| [TUM RS-VIO](https://cvg.cit.tum.de/data/datasets/rolling-shutter-dataset) | Ten synchronized RS/GS/IMU sequences; GT is the IMU frame with camera extrinsics; approximate manufacturer-derived 29.4737 μs line delay. | Official page has no dataset license statement. Linked archive directory failed in this browser; no exact small payload bound established. Author 2026 baseline describes sequence downloads as several GB. GT requires timing/extrinsic interpretation and row interpolation. |
| [PennCOSYVIO](https://daniilidis-group.github.io/penncosyvio/) | Independent UPenn capture; GoPro/Tango RGB are RS; AprilTag-derived trajectory, approximately 10 cm position accuracy. | [Official download page](https://daniilidis-group.github.io/penncosyvio/download/) lists GoPro packages around 29.4 GB and Tango RGB 9.2 GB; no explicit dataset license or row-delay contract verified. The smaller 2.7 GB Tango Top is GS. |
| Fastec / [ICCV 2025 experiment §5.2](https://arxiv.org/html/2506.22069v1#S5.SS2) | Paired first-/middle-row GS images support a controlled reconstruction comparison. | Paper explicitly says GT poses are unavailable; its poses are COLMAP pseudo-GT plus row interpolation. It cannot be represented as independent measured pose GT. |
| [WHU-RSVI author paper](https://mee-jieling.github.io/journals/2020-IEEE-ACCESS.pdf) | Independently authored synthetic sequences with known trajectories and synchronized RS/GS. | Author dataset page unavailable here; archive sizes and explicit data license unverified. The paper's CC BY license is not automatically the downloaded dataset's license. |
| [UCR author repository](https://github.com/DelinQu/Urban-Crossroads-Rolling-Shutter-Dataset) | Seven Kubric synthetic sequences, camera/readout description and pose-file links; repository MIT. | Hosted payload is Google Drive; exact sizes, pose timing semantics and applicability of the repository license to hosted data unverified. Dynamic objects require a fixed static-scene contract. |

ShutTUM offers start/end GT coverage rather than a fully observed trajectory, and BS-RSC's paired GS-image “ground truth” is not 6DoF pose truth. Neither resolves the stated two-source requirement. No payload was fetched to try to rescue these gaps.

## Decision and smallest meaningful next decision

Do not start another real-data pipeline on present evidence. First resolve the exact motion/depth convention and a source-backed pair of bounded data subsets. If those prerequisites become available, a fixed, small synthetic **falsification** should precede real inference: exact RS rendering with exact affine depths, known-shift/known-velocity ablations, diverse-row versus vertical-flow cases, and Jacobian null directions after one scale gauge. Failure to identify reference pose with exact inputs rejects the solver formulation immediately; success establishes algebra only. A real study must then compare the 2026 RS baseline, GS MADPose/RePoseD, and the same RS estimator with zero shifts under equal correspondences and charged computation. An affine-depth RS estimator beating only a misspecified GS model would not isolate the proposed contribution.

A four-CPU/8-GiB small solver diagnostic may be feasible; a robust 21-parameter solver, its strong affine-correspondence baseline, and two proper real RS evaluations have not been demonstrated feasible within one 30-minute job. This screen supplies neither a new theorem nor an accuracy result.
