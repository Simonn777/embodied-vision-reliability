# RS affine-depth: source contract, smallest physical falsifier, and novelty boundary

Source-only assessment, 2026-10-19. No images, depth payloads, model calls, fits, or new numerical experiments were used for this note. The separately reviewed manufactured rank check establishes algebraic witnesses only; it does not establish the following learned-depth assumption.

## 1. What the existing MoGe-2 output actually promises

**Neither “guaranteed instantaneous row-camera depth” nor “necessarily middle-frame depth” is supported.** At pinned commit `925b8ed835a7a9cdb7578ba15c658a0afc969030`, [`moge/model/v2.py`, infer](https://github.com/microsoft/MoGe/blob/925b8ed835a7a9cdb7578ba15c658a0afc969030/moge/model/v2.py#L193) takes RGB and optional horizontal FOV, but no row times, shutter trajectory, or temporal anchor. It recovers a global z shift/focal, takes the point-map z channel as depth, optionally projects it onto input pixel rays, and applies a global metric scale. This defines a numerical camera-ray representation. Applying the same K to all pixels does not prove the points share one physical camera time: each row can also use the same local camera-axis convention.

The pinned [training loader](https://github.com/microsoft/MoGe/blob/925b8ed835a7a9cdb7578ba15c658a0afc969030/moge/train/dataloader.py#L127) loads depth plus intrinsics; [training](https://github.com/microsoft/MoGe/blob/925b8ed835a7a9cdb7578ba15c658a0afc969030/moge/scripts/train.py#L275) lifts those labels by K; [losses](https://github.com/microsoft/MoGe/blob/925b8ed835a7a9cdb7578ba15c658a0afc969030/moge/train/losses.py#L30) perform scale/z-shift alignment. None specifies an RS reference time. The [MoGe-2 paper](https://arxiv.org/html/2507.02546v1#S3), §§3.1–3.3, combines affine point-map learning, separate metric scale, and refinement of imperfect real depth labels. Its mixed real/synthetic training is not evidence that every training image was physically global shutter, or that arbitrary RS errors are globally affine.

Thus the useful, falsifiable hypothesis is only

`z_row(u) ≈ a d_MoGe(I_RS,u) + b`, with one positive a and one b per image,

over the admitted static-scene pixels and motion range. The model could approximate this through local appearance cues; it could instead interpret distorted shapes as a fictitious GS scene; errors can vary spatially. Neither possibility is settled by `force_projection=True`. Known FOV removes a source of calibration ambiguity but does not validate the affine error model.

## 2. Temporal semantics are physically different

Let row-to-middle motion be represented by `p_row = Q(s) p_mid + u(s)`, with `p_row=z_row r`, `r_z=1`. Then

`z_mid = h(u,s) z_row − e3ᵀ Q(s)ᵀ u(s)`, where `h=e3ᵀ Q(s)ᵀr`.

Consequently, a prediction affine in the middle-frame z of the **same physical point** requires a row-dependent division and translation correction to become row axial depth. It is not generally another global affine transform. A depth map on a rectified GS image additionally requires a pixel correspondence; the same array index is not that correspondence.

Two explicit precedents clarify the distinction. [Anchor Points for Monocular Depth Estimation Under Severe Rolling Shutter Ego-motion](https://focus.ece.ufl.edu/research/2026/rolling-shutter-anchor-point/paper.pdf), §III and §IV-B (PDF pp.3–4), selects a temporal target for learned RS-to-GS correction. Its depth reference is a frozen DepthAnythingV2 teacher on GS images; §IV-B explicitly reports no metric depth ground truth. It therefore does not establish the temporal interpretation of frozen MoGe-2. Conversely, [IMU-Assisted Learning of Single-View Rolling Shutter Correction](https://arxiv.org/html/2011.03106v2#S3), §3.1 and Eqs.1–3, explicitly constructs per-row axial depths using RS/GS correspondences and row poses, then trains depth/row-pose networks. This shows row depth is a meaningful learnable target, not that an unadapted general-purpose network already predicts it. Its reconstructed depth uses flow and interpolated poses, rather than independent perfect sensor truth.

## 3. Minimum physical falsifier, before any pose estimator

The proposed ICL-NUIM clean RGB-D route is scientifically useful. Fix a tiny set of frames from both physical layouts before inference; use their independent geometric labels, not another depth network. This is a development qualification, with no claim of foundation-model-unseen data. Native ray range must first become axial depth under the documented camera convention. A negative fy or image flip must be handled consistently in RGB, depth, rays, and row-time direction.

For one predetermined nonzero angular trajectory and its sign reversal, inverse-warp a common narrower-FOV output from each GS RGB image. Entire output RGB must remain in observed source support, avoiding black borders/inpainting as an extra network perturbation. For output row ray r and known Q(s), compute `r0=Q(s)ᵀr`, `h=r0_z`, source pixel `π(K0 r0)`, and

`z_RS(r)=z_GS(π(K0 r0))/h`.

With a static scene and pure camera rotation, this is the physical row-depth relation. Interpolation near depth discontinuities is not exact ray casting: use a fixed GT-only surface-valid rule, preserve its support fraction, and state the approximation. Inverse-depth interpolation preserves planar surfaces; the pinned MoGe loader itself uses it for that reason. Do not reject pixels based on model error or predicted mask.

Use the existing frozen model/settings and known output FOV, with exactly GS and the two fixed RS inputs per frame. A 4-frame / 2-layout / 12-forward pilot would be enough to reject gross mismatch without creating a new pose pipeline. This count is a proposal, not an admitted allocation.

Controls and scoring, fixed before inference:

1. Fit only a positive scale plus offset on predetermined spatial training tiles; score all eligible held-out tiles. A nonpositive fitted scale or invalid predicted depth remains an explicit failed/penalized outcome. Report absolute row-axial relative-depth error and a tail statistic, per image and per layout, with fixed denominators.
2. Retain the paired GS inference. Transport its calibrated depth by known geometry as an oracle diagnostic floor: `(a*d_GS(warp)+b)/h`. Do **not** warp d first and then pretend its offset stays constant; that changes b into b/h.
3. Separately pre-register the same-point middle-z interpretation and convert it correctly. It diagnoses semantics; whichever interpretation happens to score best cannot retrospectively redefine the candidate.
4. Report a scale-only calibration control to expose whether fitting b is merely hiding a large distortion. Preserve every image, including poor GS cases; no replacement frames.

The admission gate needs **both** an absolute RS-depth error ceiling and a maximum paired excess over the GS floor, plus finite/support coverage. For a small falsifier, candidate prospective tolerances are capped mean AbsRel ≤0.10 and RS-minus-GS ≤0.02 on each layout, with individual-image values retained; these are engineering admission tolerances, not a theorem that 10% depth error suffices for pose. Root must fix the exact loss, invalid penalty, tail/support requirements, and thresholds before execution. Poor GS performance makes this test uninformative for RS-specific causation and is not permission to change the sample. A near-zero physical difference between row-z and same-point middle-z makes semantic discrimination weak; report that difference rather than interpreting similar errors as proof.

**Pure-rotation image reprojection is not a depth-validation metric:** depth cancels from the mapping. This pilot directly tests depth against independent GT. Passing only permits a later nonzero readout-translation and pose test; it says nothing yet about velocity accuracy, noisy gyro, time synchronization, motion blur, moving objects, or real RS generalization. Failure closes use of this unadapted global-affine row-depth approximation in the tested regime. Training a specialized RS depth model would be a different proposal and would collide with existing RS-correction work unless it adds something more specific.

## 4. Closest estimator priors and what, if anything, remains

| Verified primary source | Overlap and remaining distinction |
| --- | --- |
| [Barath, Rolling Shutter Relative Pose Estimation Made Practical (2026)](https://arxiv.org/html/2606.26863v1), §§3–4, Eqs.1–3,17 | Seven affine correspondences, first-order row motion, 17-parameter RS refinement, and GS fallback. RGB-only strong opponent. Our given-orientation branch has stronger inputs and cannot advertise 4 versus 7 as an equal-information comparison. |
| [Ding et al., Fixing the Scale and Shift in Monocular Depth For Camera Pose Estimation](https://arxiv.org/html/2501.07742v1), §§2–3 | Independent affine depth calibration plus GS pose is already established. Adding the same variables to an RS optimizer is not, by itself, a substantive new estimator. |
| [Yu et al., Relative Pose Estimation through Affine Corrections of Monocular Depth Priors (CVPR2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Yu_Relative_Pose_Estimation_through_Affine_Corrections_of_Monocular_Depth_Priors_CVPR_2025_paper.pdf) | Affine-depth minimal solvers and hybrid geometric estimation already provide the obvious depth baseline. |
| [Lee et al., Gyroscope-aided Relative Pose Estimation for Rolling Shutter Cameras (2019)](https://arxiv.org/html/1904.06770v1), §§3.1–3.2, Eqs.3–7 | Known angular rates, unknown middle R, five-point solver; explicitly assumes zero within-readout linear velocity. It is not the proposed unknown-v1/v2 affine-depth system. |
| [Lee and Yoon, Inertial-aided Rolling Shutter Relative Pose Estimation (2017)](https://arxiv.org/html/1712.00184v1), §4.2, Eqs.7–11 and Table1 | Already separates the linear t/v block conditional on R; provides uniform-motion 11/9-point branches with angular rates / additional gravity. “Known rotation makes translation/velocity linear” is old. |

I did not locate an explicit prior for the **exact constant-SE(3), known middle-R and known angular-rate, independent affine-depth, unknown two linear velocities** conditional 12-variable system in these inspected full texts. This is a bounded search result, not a novelty guarantee. Middle R is extra information beyond two instantaneous gyro samples; an uncertain visual R estimate cannot silently be treated as known. With measured orientation, call this conditional translation/velocity/depth calibration, not four-point recovery of the complete relative pose.

The potentially substantive theoretical question is the critical-configuration structure and translation-functional identifiability of that conditional system: why a four-point affine-depth design can be full rank in one regime but necessarily loses rank for zero angular rates with R e_x parallel to e_x, how five points can remove that particular obstruction, and which null directions actually affect the desired translation direction. The observed full-rank fixtures are witnesses, not a generic proof, a certified minimal solver, or accuracy evidence. A normal least-squares stack or “RS + depth + IMU” combination alone is too weak a contribution claim.

If the depth semantics pass, the smallest defensible estimator comparison must use the same measured K/readout/gyro/orientation and fixed features for all relevant arms: known-gyro angular RS baseline; the stronger uniform RS formulation; GS affine-depth with its official robust settings; and joint exact RS-plus-affine-depth nonlinear refinement using the same inputs. Barath's full RGB-only solver remains a practical baseline with its differing input contract disclosed. Include the direct conditional linear solution as an initialization/control and charge all depth/feature computation. Any gain must survive comparable refinement and computational budgets. If joint same-information refinement gives the same solution, the contribution can only be a demonstrated robust-initialization/conditioning advantage with a precise regime—not new information and not a new geometric objective.

**Recommendation:** proceed only with the small physical depth-semantics falsifier and the already isolated mathematical proof work. Do not build a full candidate pose pipeline from rank success alone. There is a narrow conditional problem worth checking, but no source-backed CVPR contribution or measured practical gain yet.
