# 0025b — Physical row-depth qualification on fixed ICL-NUIM frames

**Prospective source protocol, 2026-10-19. Execution requires this exact source/protocol commit and successful CI; this is not an acquisition or inference receipt.** No scientific image/depth/pose payload, network prediction, or scientific diagnostic was inspected to choose this design. Manufactured-array checks qualify the implementation only. Root integration, source review, a separate protocol/source freeze and exact-commit CI remain required before acquisition or inference. The completed 0025 algebra checks and the failed 0024 network-consistency experiment remain separate, unchanged records.

## Question and limited decision

Does the existing frozen MoGe-2 prediction on a manufactured rolling-shutter (RS) image admit one positive global scale and one offset that approximates **instantaneous row-camera axial depth** on fixed spatially held-out pixels? This is a necessary engineering qualification for considering that approximation in a later estimator, **not a mathematical necessary/sufficient condition for accurate pose**, an estimator, a new trained model, or a CVPR result.

The two ICL-NUIM layouts are simulated physical scenes and become mechanism-development data. They are not two independent real-world validation sources, and no claim of absence from foundation-model training is made. A pass only permits proposing a separately frozen stage with within-readout translation and task-level pose evaluation. A fail closes this unadapted global-affine row-depth approximation in the tested regime; it does not permit changing the selected frames, motion, field of view, fit, support, or gate on these outcomes. The final requirement of substantial pose improvement on two independent sources, with strong baselines, matched costs, and uncertainty reporting, remains unchanged.

## 1. Assets, selection, and barriers

Use only the official **clean TUM-compatible PNG** trajectory-2 archives, not native ray-range `.depth`, noisy `traj2n`, or a mirror:

| Layout | Official HTTPS archive | Exact response bytes |
| --- | --- | ---: |
| Office | https://www.doc.ic.ac.uk/~ahanda/traj2_frei_png.tar.gz | 301,828,475 |
| Living room | https://www.doc.ic.ac.uk/~ahanda/living_room_traj2_frei_png.tar.gz | 429,825,445 |
| Total | | **731,653,920** |

The preparation acquisition contract and HEAD receipts are in `work/0025b-physical-depth/`; no archive has been acquired at this protocol revision. Full gzip archives necessarily transport and decompress past unselected/pose bytes to enumerate headers. Do not describe this as never transporting those bytes. No source-pose text is extracted, parsed, or used. Archive download/index bounds, first-observed SHA256, exact sizes/ETags, interruption preservation, and a completed outer receipt remain required. Source terms are CC BY 3.0 with Handa, Whelan, McDonald, and Davison / ICL-NUIM attribution.

From the structural inventory alone, identify the unambiguous numeric filename-stem intersection of RGB and depth PNGs in each sequence. Reject duplicate numeric stems or ambiguous path associations. Sort by integer stem. For N paired entries, select exactly eight ranks

`r_k = floor(k*(N-1)/7 + 1/2), k=0,...,7`,

implemented by exact integer arithmetic, not language-dependent floating-point rounding. Require N >= 8 and eight distinct ranks; otherwise fail without a substitute sequence. Save the complete sorted paired-stem inventory with RGB/depth member names and sizes, as well as the selected ranks/names, before any image pixel decode. Extract only the 16 selected RGB members at this stage and persist their encoded-byte SHA256 hashes, byte counts, member identities and relative output filenames in a separate input record before the first RGB decode or model construction. This includes the first and last paired entry. No image appearance, predicted mask, depth quality, pose, or eventual score may select or replace a frame.

There are **16 original frames, 16 canonical GS inputs, 64 RS inputs, and at most 80 complete MoGe forwards**. Each source frame has fixed order `GS, +x, -x, +y, -y`. The GS result is reused by the four controls, not inferred four times. There are no feature matching or native pose calls.

Before prediction, depth is bound by the complete archive SHA256 and structural member metadata only; depth members are neither extracted nor individually hashed. This does not promise a per-depth-PNG hash before inference. The inference stage extracts and decodes selected RGB only. **All 80 raw predictions, input hashes, camera/motion metadata, warnings, call counts, and completed inference/outer receipts must be sealed and verified before selected depth PNG extraction or numeric GT decode.** After that barrier, extract the 16 selected depth members, persist their encoded SHA256 hashes/member identities/byte counts, and validate their PNG headers before decoding any GT samples for support or fits. An incomplete inference cohort is not silently completed or analyzed as a successful subset. Model errors preserve partial artifacts; a technical retry requires a separate recorded decision and never overwrites the failed operation. GT remains diagnostic-only and is never an estimator input.

## 2. Source depth and coordinate contract

The official [ICL-NUIM page](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html) declares the PNGs TUM-compatible and links the [TUM format specification](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats#color_images_and_depth_maps): 16-bit monochrome PNG, zero invalid, **axial z in metres = stored value / 5000**. Do not convert those values from ray range a second time. In contrast, native `.depth` is Euclidean range, as documented by the [author codes page](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html) and `compute3Dpositions.m`. The exact original PNG-export program has not been located; the official linked format is the documented basis. Preserve that limitation rather than claiming a converter audit.

Require selected source RGB and depth dimensions 640×480 and matching filename identity. Before RGB pixel decoding, validate the PNG signature and IHDR length/type/CRC, dimensions, bit depth 8, and color type 2 (RGB) or 6 (RGBA). Decode RGB unchanged. For RGBA, require every alpha sample to equal 255, then discard only that opaque alpha channel; any transparency fails the operation, with no compositing or substitute input. Reject palette, grayscale, other bit depths and nontrivial EXIF orientation; perform no orientation transform, vertical flip, colour-profile conversion or fallback mode conversion. After the post-prediction depth barrier, require the depth PNG IHDR to specify 640×480, bit depth 16 and grayscale color type 0, decode unchanged integer samples, and divide by 5000 exactly once; zero samples remain invalid GT. The [ICRA2014 paper](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/icra2014.pdf), §II Eq.1, supplies zero-based source intrinsics

`K_minus = [[481.2,0,319.5],[0,-480,239.5],[0,0,1]]`.

Define a new pixel-down 3D coordinate convention with `S=diag(1,-1,1)`, `p_plus=S p_minus`, and

`K_source = K_minus S = [[481.2,0,319.5],[0,480,239.5],[0,0,1]]`.

Stored arrays and axial z remain unchanged. **S is a reflection, not an SO(3) rotation.** No original world/camera pose is converted or read. All newly manufactured rotations below are defined directly in the new coordinate convention. This avoids confusing negative fy with an instruction to vertically flip the images.

## 3. Fixed image generation and known FOV

All five inputs for a frame have W=448, H=336 and square-pixel canonical intrinsics

`Kc = [[481.2,0,223.5],[0,481.2,167.5],[0,0,1]]`.

This is a calibrated inverse warp from the anisotropic source K, not a silent replacement of source fy or an untracked resize. Integer `(u,v)` are array pixel centers; no extra half-pixel is added to unprojection. MoGe's normalized principal point 0.5 converts back by `c_pixel=c_normalized*size - 0.5`.

The full readout is 0.040 seconds; fixed total first-to-last-row angular displacement is 12 degrees. For each signed axis vector `a ∈ {+12° e_x,-12° e_x,+12° e_y,-12° e_y}` in radians,

`s(v)=v/(H-1)-1/2`, `Q(v)=Exp(s(v) [a]_x)`.

Q maps a same-point middle-camera vector into the instantaneous row-camera coordinates. Its angular rate is `a/0.040` radians/second. GS uses Q=I. For output homogeneous ray `r=Kc^-1[u,v,1]`, compute

`r0 = Q(v)^T r`, `h = (r0)_z`, `w_source = π(K_source r0)`.

Require finite positive h. Generate **every** output RGB pixel by ordinary bilinear interpolation in the original observed RGB. All four source neighbors must be in bounds; no border replication, black fill, inpainting, hole filling, or per-image crop is allowed. Geometry-only prechecks must establish this for the fixed camera/motions before any inference. A violation is a configuration failure, not permission to decrease the angle or change the focal length. The fixed sampling rule is to decode uint8 RGB, divide by 255, interpolate its stored RGB channel values with float64 bilinear weights, and cast the resulting image once to float32 for inference. Do not linearize/re-encode colour or quantize the generated image back to uint8. Use the identical rule for GS and RS, record generated float32 input hashes, and preserve original RGB hashes.

Pass the same exact known horizontal FOV to all calls:

`fov_x_degrees = 2*atan(W/(2*481.2))*180/pi`.

The W convention refers to the image extent between outside pixel edges and matches normalized intrinsics; substituting W−1 would change the supplied focal. The returned normalized K must be a finite 3×3 matrix with expected value `[[481.2/W,0,0.5],[0,481.2/H,0.5],[0,0,1]]`. Apply exactly the earlier 0024 elementwise check `abs(K_returned-K_expected) <= 1e-5*abs(K_expected)`, with no absolute tolerance: expected zeros must be exact zeros. This check precedes accepting each raw prediction. Retain the qualified MoGe-2 small checkpoint, source/runtime/weight bindings and CPU float32 settings: `num_tokens=1200`, `use_fp16=False`, `apply_mask=False`, `force_projection=True`, author native SDPA, serial OpenCV and bounded four-thread CPU pools. No training, weight change, learned correction, model-mask filtering, or extra high-token/TTA forward is admitted.

## 4. Fixed lattice and common GT-only support

Use a **64×64 lattice**, not 64-pixel tiles. For j,i=0,...,63,

`u_j = floor(j*(W-1)/63 + 1/2)`, `v_i = floor(i*(H-1)/63 + 1/2)`.

Use exact half-up integer evaluation, e.g. `(2*j*(W-1)+63)//126`. Training is `(i+j)%2 == 0`; held-out evaluation is odd parity. There are 4096 possible samples and 2048 per split before support filtering. Spatial dependence is retained; neither lattice samples nor trajectory frames are treated as independent statistical replicates.

For each RS lattice point, after the complete prediction seal, form common support using only fixed geometry and GT:

1. h is finite and positive; the original source projection and corresponding canonical GS projection `w_GS=π(Kc r0)` both have their required four-neighbor interpolation support in bounds.
2. The four source GT z values are finite and strictly positive.
3. Their fixed discontinuity condition is `(max(z_neighbors)-min(z_neighbors))/min(z_neighbors) <= 0.02`.

On qualifying source neighbors, interpolate **inverse GT depth** with the ordinary bilinear coefficients `lambda_k`:

`z_GS_same_point = 1 / sum_k(lambda_k/z_k)`, `z_row = z_GS_same_point / h`.

Inverse-depth interpolation is exact on a planar visible surface under the pinhole model. It is an approximation on curved geometry; the four-neighbor spread rule does not prove visibility or eliminate every occlusion boundary. Preserve neighbor/support summaries and explicitly limit conclusions to this sampled smooth-surface support.

All four fitting arms below use precisely this common support. **No model mask, predicted positivity, residual, fit condition number, or eventual error may remove a point.** The GS prediction is sampled at `w_GS` using **ordinary bilinear depth interpolation**, not inverse-depth interpolation: positive affine calibration commutes with ordinary interpolation, whereas inverse-depth interpolation of a network prediction changes the control nonlinearly.

For every RS case require common support / 4096 >= 0.50, at least 256 training points, and at least 256 held-out points. These are simultaneous checks. The denominator is always the original 4096 lattice points, not an already filtered set. Keep support counts and reasons for all 64 cases even if the gate fails. GT support cannot expand to compensate for unavailable predicted values.

## 5. Four oracle calibration diagnostics on the same objective

These are **oracle GT-assisted diagnostic fits, not deployment algorithms**. For each RS case, use exactly the same supported training points and target `z_row`. Let `d_R` be the unmodified RS prediction at that lattice point, and `d_G(w_GS)` the ordinary bilinear canonical-GS prediction at the same physical ray. Fit the following unweighted row-Z least-squares designs:

| Fixed arm | Design X | Predicted row depth | Purpose |
| --- | --- | --- | --- |
| `RS_row_affine` | `[d_R, 1]` | `a*d_R+b` | Primary instantaneous-row affine hypothesis |
| `GS_calibrate_then_transport` | `[d_G(w_GS)/h, 1/h]` | `(a*d_G(w_GS)+b)/h` | Paired GS oracle transport control |
| `RS_same_point_mid_affine` | `[d_R/h, 1/h]` | `(a*d_R+b)/h` | Explicit same-point middle-Z interpretation |
| `RS_row_scale_only` | `[d_R]` | `a*d_R` | Diagnostic need for an offset |

All affine arms have two degrees of freedom. The two transported arms must **not** first fit in middle-Z coordinates: that would change the row-dependent weighting of this common LS objective. Neither may substitute a constant post-warp offset for b/h. The GS control is calibrated separately for each RS case on its same supported training points, with the one stored GS prediction; it is not claimed to have one deployable calibration shared across views.

Use float64 OLS/SVD. For X with n rows and p columns, divide each column by its finite positive L2 norm for numerical equilibration, apply SVD, and undo this scaling in the returned coefficients. This reparameterizes the same original unweighted row-Z LS objective. Require `sigma_min > max(n,p)*eps64*sigma_max` for this equilibrated matrix, finite singular values and finite coefficients. A zero/nonfinite column norm is unavailable. This is the machine-precision numerical-rank criterion, not a tuned weak-observability cutoff. Record column norms, singular values and their ratio, but do not reject additional cases using a hand-selected condition threshold. Require a>0. An unconstrained optimum with a<=0 lies outside the allowed positive-scale model and is recorded unavailable, not moved to an arbitrary epsilon, clipped, or refit. This does not assert that no positive-scale vector with a worse objective exists. No robust loss, model-error trimming, weighting, regularization, nonlinear calibration, or parameter search is used.

Evaluate the fitted prediction on **every** common supported held-out point. Finite raw model values may be negative for the affine arms: an offset can validly correct them. Nonfinite values required by a fixed design, rank failure, nonpositive scale, or nonpositive/nonfinite fitted physical depths on the fixed support make the arm unavailable; they never create a positive-only subset. The scale-only arm naturally fails when a required nonpositive raw value cannot become positive with a>0. Retain any finite diagnostics already computed and the exact reason. Model masks are saved solely for description and never supply support or validity exceptions; MoGe's own frozen internal mask-dependent recovery is unchanged. Out-of-support raw-output anomalies must still be reported; they are not extra samples in the fixed diagnostic.

## 6. Metrics, fixed denominator and prospective admission gate

For a fully available case and arm, define its primary error as the **uncapped mean** held-out AbsRel:

`E = mean(|z_pred - z_row| / z_row)`.

Save every case/arm value, fitted coefficients, support counts, finite/rank flags, and positivity failures. Also report held-out pixel error quantiles and all case values for context, without selecting a favorable tail or case subset. An optional GS-only display statistic is descriptive and does not replace this same-point matched objective.

The fixed gate is a conjunction:

1. All 80 inference outputs and provenance/resource seals are complete and valid; all 64 RS cases meet the common-support checks and have **both primary arms `RS_row_affine` and `GS_calibrate_then_transport` available**. Both explanatory arms are attempted and reported for every one of the 64 cases, including unavailable outcomes, but their fit failure is not an additional primary gate. In particular, scale-only failure must not by itself reject a valid affine model merely because an offset is necessary.
2. In **each** scene, the median of its fixed 32 `RS_row_affine` case mean-AbsRel values is <=0.10.
3. In **each** scene, the median of its fixed 32 paired differences `E_RS_row_affine - E_GS_calibrate_then_transport` is <=0.02.

These scalar thresholds are prospective engineering tolerances, not a pose-error guarantee or evidence of a new method. A median can hide a poor minority of fully available cases; the complete per-case table and tail statistics remain mandatory. The paired statistic is a median of paired differences, not a difference of unrelated medians.

**An unavailable required primary case means gate failure**, not a numeric error of 1 mixed with an unbounded loss. If any of the 64 primary cases is unavailable, the complete-cohort primary aggregate is unavailable and the gate is false. Successful-case diagnostics may still be displayed with an explicit partial-coverage label, but never as the primary result or as an alternative pass. Explanatory-arm failures retain all-case coverage/counts and do not alter the predefined primary interpretation. Do not replace failed scenes/frames/directions or report only a success subset. A technical fault is distinguished from a mathematically unavailable fit, but neither qualifies an incomplete primary cohort.

## 7. Interpretation controls and hard limitations

- Record the same-support physical separation `|z_GS_same_point-z_row|/z_row` per case. At only ±6 degrees from the middle row, the two temporal interpretations may be close. Passing the row test cannot by itself identify the network's unique physical time anchor; a small difference between the two diagnostic arms is inconclusive.
- GS transport and RS inference see different image sampling/distortions. A failure can reflect warp-induced appearance/resampling sensitivity as well as a wrong temporal interpretation. This test rejects the practical unadapted approximation; it does not attribute that failure to a single learned mechanism.
- Pure rotation has no translational parallax or disocclusion, and image reprojection is independent of depth. Do not use that reprojection as validation of the learned depth. This stage does not test unknown row velocities, readout translation, middle-R estimation, gyro noise, synchronization, blur, dynamic objects, or real sensor effects.
- The truth interpolation and 2% local-spread support restrict the evaluated geometry. Poor GS performance makes RS-specific interpretation weaker and is retained, not used to discard a frame. Passing an absolute and excess threshold is still only this synthetic qualification.
- The generic/full-rank and exact four-point algebra checks concern a model with given rotation/rates and suitable depth observations. They do not establish that this frozen monocular network satisfies its assumptions. Conversely, this depth-only stage cannot establish solver minimality, robust relative-pose improvement, or novelty over RS and affine-depth estimators.

## 8. Resource, records and next decision

No fees, new weights, model training, package installation, extra matches, or extra dataset are part of this stage. Use the already qualified pinned runtime/model. Each admitted acquisition, prediction and diagnostic operation has an exclusive fresh directory, exact source/protocol/input hashes, captured logs, a 1800-second outer wall limit, <=8 GiB aggregate process-group RSS, and <=4 CPU threads; the model is CPU float32. Count all 80 attempted forward calls, including any failed call, and preserve timings/warnings. No success claim survives an outer resource failure. The exact future code/protocol/runtime and expected commit must be filled and verified at freeze; this pending protocol invents none.

Save original file hashes, generation parameters, all generated-input hashes, raw depth/mask/intrinsics, per-call timings, the pre-GT prediction seal, support reasons and fixed indices, all four diagnostic fit/evaluation records, and complete aggregate/decision receipts. Published material may derive tables/receipts without republishing licensed full archives. A separate saved-output audit must verify formulas, all-case coverage, and bindings before any pass is reported.

Only a complete pass supports writing a **new prospective translation/pose-stage proposal**. It does not admit a paper or a claim of useful task improvement. Failure or inability to evaluate remains recorded and does not reopen the closed 0024 calibration proposal. Subsequent same-image tuning would be new development work and cannot be sold as independent confirmation.

## Source scope

This protocol reuses the source-only input audit `work/0025-icl-metadata/format-contract.md` (SHA256 `f13c9a9b3e4053e3961aef0cd822fb2599a81a055e9fd593012b70b3659bfaba`) and semantics/prior review `work/0025-rs-depth-semantics-review.md` (SHA256 `3be57ce8637e9ae5b41f818e435c9997d35edb959d31bb4f57d13b1f3c2f535a`). Their earlier pilot counts/loss suggestions are historical proposals; the explicit cohort/objective/gate here supersedes them prospectively when frozen. MoGe-2 source is pinned at `925b8ed835a7a9cdb7578ba15c658a0afc969030`, [`moge/model/v2.py`](https://github.com/microsoft/MoGe/blob/925b8ed835a7a9cdb7578ba15c658a0afc969030/moge/model/v2.py), including the known-FOV path. No RS temporal-depth guarantee is inferred from that API.
