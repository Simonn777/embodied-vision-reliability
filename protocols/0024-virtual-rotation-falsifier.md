# 0024: virtual-rotation offset transfer falsifier

Status: prospective source preparation. **No real run is admitted until this
protocol, implementation, independent review and exact-commit CI are frozen.**
This is a necessary model-assumption test, not a pose/depth-accuracy experiment.
The closed 0023 hypothesis and all its results remain unchanged.

## Question and prior boundary

Can additional frozen monocular predictions from known virtual rotations of the
same image produce an axial-depth offset which transfers to another rotation
axis, beyond a zero-offset alignment with the same nuisance-parameter budget?

The affine-depth equation is established: Ding et al., *Fixing the Scale and
Shift in Monocular Depth For Camera Pose Estimation*, Eq. 4 specializes to known
rotation and zero translation. 360MonoDepth (CVPR 2022), panoramic calibration
(ICCV 2023), RPG360 (2025), DepthMaster (2026), and MoGe panorama inference already
cover related virtual-view consistency/fusion. We do not claim a new solver,
geometry theorem, physical depth measurement or proven novelty.

The remaining empirical hypothesis is an actively probed, correspondence-
independent offset prior. Only if useful could it later constrain a **fixed-shift**
pose estimator. Applying an affine correction before a free-affine pose solver
is merely a reparameterization. The later method would have to beat the strong
free-affine author solver, known-FOV control, same-forward-count fusion, and
higher-token alternatives at measured complete-pipeline cost. No such pose run
is admitted by this protocol, and the existing two-source manuscript gate is
unchanged.

## Fixed population and information

Use all 27 already acquired 0023 frames: each of the three anchors and its eight
query frames, in existing plan order. The same three places remain development
forever. No new scenes, frames, GT, model weights or dependencies are acquired.
Read RGB and the authored intrinsics only; do not read pose files, old scored
errors, keypoints/matches, reference views or native-estimator inputs. Opaque
hash verification of the original acquisition payloads is permitted.

The original plan is `configs/0023-prospective-plan.json`, SHA256
`cc9b02cac8946107f1c945643b316695e5ebbf6339b34baf737541f8cec886d5`.
Its acquisition/input freeze is `acf502b831ad9e3485ab56549217c8a603d56bf7`.
Original acquisition receipt SHA256 is
`bf6cd9baafa4b8c72fc3a33021becc3cf37f3bf3a6ecec406df77c4b25114905`.
Retain and reverify the complete original acquisition/source/model qualification
chain. Use qualified 0022 assets at `fa454585e67a8f0395ad5f67a03d5a28cc33a66a`.
Existing source files and cached scientific results must not be modified.

## Camera and frozen model calls

Zero-based pixel centers; an image of width W and height H has centered principal
point `((W-1)/2,(H-1)/2)`. Use output dimensions from the original 0023 max-edge
512 rounding rule, preserving aspect ratio. All five views have identical
centered square-pixel Kc. This makes the author's `fov_x` interface an exact fit
for the canonical camera; it is not an arbitrary full-K interface.

Let the available half-angle be the minimum of
`atan((cx-2)/fx)`, `atan((W-3-cx)/fx)`,
`atan((cy-2)/fy)`, `atan((H-3-cy)/fy)` using original K and dimensions.
All four distances must be positive. Rotation angle is the smaller of 8 degrees
and one quarter of that half-angle. The five W2C virtual rotations are identity,
+/- yaw about camera y, and +/- pitch about camera x. There is no translation.

Choose the smallest common focal f that keeps all four corners of every virtual
image, back-projected by `K_original R_j^T Kc^-1`, inside the original rectangle
`[2,W-3] x [2,H-3]` with positive homogeneous depth. Starting lower bound is
`max(output_width,output_height)/100`; double an upper bound until feasible
(at most 64 doublings), then do exactly 60 bisections. Use the feasible upper
bound multiplied by `1+1e-12`. Verify full remap bounds, not just corners.
Decode each original once; remap directly from original RGB using float64
homography coordinates cast to float32 for OpenCV INTER_LINEAR. No iterative
warping, inpainting, reflection or unobserved border content is allowed. Any
out-of-bounds input coordinate is a technical failure of this implementation.

Use the same strict frozen MoGe2-S-normal loader as 0022/0023: CPU float32,
1200 tokens, no fp16, apply_mask=False, force_projection=True, with explicit
`fov_x = degrees(2*atan(output_width/(2*f)))`. The normalized returned intrinsics
must agree with Kc under the pixel-center convention (float32 tolerance 1e-5
relative). Execute exactly five depth forwards per image, 135 total, in fixed
identity,+yaw,-yaw,+pitch,-pitch order. No learned-feature or matcher calls.
Loading unchanged unused SP/LightGlue modules through the qualified loader is
allowed and charged; their inference is not. Preserve every raw depth, model
mask, intrinsics, array hash, per-call timing and execution warning privately.

## Estimator and held-out spatial evaluation

Write d0 for base depth divided by its full positive-valid-image median, and dj
for each virtual depth divided by its own full positive-valid-image median.
These are positive scale gauges, not calibrations to GT. Known correspondence
is `u_j ~ Kc R_j Kc^-1 u`; `h_j=(R_j Kc^-1[u,v,1])_z`.

The restricted model gives
`s_j*dj/h_j + t_j/h_j - t0 = d0`,
where corrected base depth is `d0+t0`. Each yaw/pitch pair has five unknowns:
shared t0 and independent `(s_j,t_j)` for its two views. Solve ordinary linear
least squares after column-L2 normalization. Require all five singular values
positive and `sigma_min/sigma_max >= 1e-5`; no regularization or parameter search.
Require both fitted s_j > 0 and finite coefficients. This is an errors-in-
variables approximation, not an unbiased or likelihood-optimal estimator.

Take a 64 x 64 grid at subcell centers over the base image:
`x=(i+.5)*output_width/64-.5`, similarly y. Bilinearly sample raw depths using
four distinct in-bounds positive model-valid neighbors. The common grid support
must be valid in base and all four virtual views. Keep that same support for all
fits and controls. Training cells have `(i+j)%2==0`; evaluation cells have odd
parity. Require at least 256 of each. Neither fitted offset nor nuisance
parameter objectives may use the evaluation cells. Full-image median gauges
and the predeclared positivity guard do access those locations; the split is
not a claim of fully inductive preprocessing or statistical independence.

Estimate yaw t0 using its two views' training cells and pitch t0 separately.
Each offset must leave at least 99% of ALL base positive-valid pixels positive.
Both estimates must meet rank/scale/positivity conditions for an image to have
usable offset coverage. Report offset disagreement and singular spectra; these
are not GT-error estimates or evidence of physical correctness.

Test both directions: yaw offset -> the two pitch views, and pitch offset ->
the two yaw views. For each held-axis view, fix the transferred t0 and fit only
its two nuisance parameters `(s_j,t_j)` on training cells. Compare to t0=0,
which gets the same two nuisance parameters and identical training/eval cells.
For the two-column nuisance design, use column-L2-normalized SVD and the
standard machine-rank criterion sigma_min > max(number_of_rows,2) * float64_eps
* sigma_max. This controls numerical singularity without imposing the five-
parameter offset-excitation threshold on the two-parameter baseline. For either fit enforce
s_j>=0: solve OLS; if its scale is negative, take s_j=0
and use t_j=dot(v,y)/dot(v,v), where v=1/h_j and y=d0+t0. The resulting
base-coordinate prediction is t_j/h_j-t0, not a spatial constant. A rank-
deficient nuisance design uses this same boundary (constant only in the virtual
axial-depth coordinates). Evaluate mean absolute residual on evaluation cells, in
units of the base positive-valid median. Average the two views per direction.
No clipping of residuals. No spatial/axis IID uncertainty claims.

If an offset image is unavailable, both candidate directional errors are exactly
the respective zero-offset baseline errors. If common sample support or a base/
virtual positive-valid median is unavailable, assign error 1 to both methods
in both directions and record the unavailable reason. This fixed failure mass
is part of this diagnostic, not a physical depth metric. All 27 images and 54
image-directions remain in aggregate denominators. Technical errors stop the
whole run and never count as an eligible scientific result.

## One-shot decision and limitations

All conditions must hold to proceed to a separately frozen real-pose study:

1. Usable offset coverage >=19/27 images and >=5/9 in each place.
2. `1-mean(candidate_error)/mean(zero_offset_error) >= .10` over all 54
   image-directions. If both means are zero, gain is zero; if only baseline is
   zero, the condition fails.
3. At least two places have strictly positive mean error reduction; no place
   has more than 2% relative error increase. A zero-baseline place ties only
   when its candidate error is also zero; otherwise it fails the scene floor.

A pass establishes only transfer of this approximate predictor-error model.
A wrong rotation-equivariant scene can pass perfectly. Small rotation,
interpolation, scene degeneracy, shared base-map bias, non-affine model errors
and OLS attenuation can all confound the estimated offset. Passing neither
proves improved depth nor meets any manuscript/result gate.

Failure closes this fixed virtual-offset variant; no angle, mask, threshold,
loss, solver, crop or subset tuning on these outputs to rescue it. Another
mechanism would need its own justification. Preserve all failures and source
changes. No paper is authorized by either outcome.

## Execution

One exclusive offline operation, <=4 CPU threads, <=8 GiB sampled process-group
RSS, <=1800 seconds, no paid services, and no download. Record the enclosing
operation and all constituent costs; processing shared anchors once is explicit.
A fresh output directory is mandatory. Bind the exact Git source revision,
protocol/config, original inputs and model chain before and after the run;
seal every output before computing the diagnostic gate from those saved outputs.
Use artificial geometry tests only before source freeze. Save small original
aggregate/receipt evidence to GitHub after independent audit; do not rehost RGB,
depth payloads or GT. The broad research goal remains active.
