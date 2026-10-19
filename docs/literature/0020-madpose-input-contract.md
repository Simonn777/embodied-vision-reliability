# MADPose input-contract admission — source only

Date: 2026-10-19. Official source pin: `MarkYu98/madpose@c17e179dd4f2794ea7d6942f402f12e15504ead7`. No scientific arrays, labels, images, native calls, compilation, or dependency installation were used in this review. This note does not admit an experiment or change frozen experiment 0020.

## Decision

**The current matched-depth caches do not establish the inputs needed for the official default calibrated demo.** That demo supplies the minimum of each complete predicted depth map and enables `min_depth_constraint`. An authentic pair of minimum scalars from the same original maps would suffice; obtaining entire maps is not inherently necessary. These scalars and their provenance have not been identified in the existing cache contract.

**A documented, nondefault affine-depth comparison is possible at the input-contract level:** the official Python API exposes `min_depth_constraint=False`, and the README explicitly explains that the supplied minimum depths are used only when that option is enabled. This retains the affine solver and hybrid estimation; it does not require a source modification. It is not the default demo, an exact paper reproduction, or an already qualified native implementation. There is no verified official *default* no-minimum route for the full affine method. [Pinned README](https://github.com/MarkYu98/madpose/blob/c17e179dd4f2794ea7d6942f402f12e15504ead7/README.md), [configuration](https://github.com/MarkYu98/madpose/blob/c17e179dd4f2794ea7d6942f402f12e15504ead7/src/estimator_config.h), [bindings](https://github.com/MarkYu98/madpose/blob/c17e179dd4f2794ea7d6942f402f12e15504ead7/src/bindings.cpp).

## Inputs and conventions

The calibrated affine entry point is `HybridEstimatePoseScaleOffset(x0, x1, depth0, depth1, min_depth, K0, K1, options, est_config)`. It accepts two matched pixel-coordinate arrays, two corresponding scalar-depth arrays, a two-element minimum-depth vector, two intrinsic matrices, and configuration objects. `min_depth` remains a required argument even when its constraint is disabled. Source inspection finds all effective uses conditional on that flag; finite, explicitly declared placeholders can therefore be qualified artificially for the disabled variant. Do not use NaN or infinity sentinels. The return includes pose, relative scale, two offsets, and RANSAC statistics. [Bindings](https://github.com/MarkYu98/madpose/blob/c17e179dd4f2794ea7d6942f402f12e15504ead7/src/bindings.cpp).

| Required information | Existing 0017 contract | Consequence |
|---|---|---|
| Matched raw pixels and corresponding depth samples | Available in the previously admitted cache schema | Pass unchanged; do not normalize pixels twice. |
| Calibrated intrinsics | Authored PINHOLE camera parameters and stored-K cross-check | Use the same effective K as the other fixed-input controls. |
| Whole-map minima in the same depth gauge | Not established | Default constrained affine demo remains blocked. A matched-only minimum is not a substitute. |
| Images and complete depth maps | Not required by the native estimator beyond the two minima | The demo uses them to obtain samples and minima. |
| Exact frontend provenance | Exporter/checkpoint, resizing and sampling provenance remain incomplete | A downstream same-cache comparison is possible; exact end-to-end reproduction is not established. |

The estimator lifts a pixel using its homogeneous ray and axial depth. The model is

\[
R\big[(d_0+o_0)K_0^{-1}p_0\big]+t
=s(d_1+o_1)K_1^{-1}p_1.
\]

Preserve raw translation for depth-model diagnostics; unit-normalized translation is only appropriate for directional pose evaluation. The demo helper samples maps with rounded, clipped coordinates after image-to-map scaling. That frontend sampling has not been proved identical to the opaque RoMa/MoGe cache exporter. [Estimator header](https://github.com/MarkYu98/madpose/blob/c17e179dd4f2794ea7d6942f402f12e15504ead7/src/hybrid_pose_estimator.h), [sampling helper](https://github.com/MarkYu98/madpose/blob/c17e179dd4f2794ea7d6942f402f12e15504ead7/madpose/utils.py). Existing cache provenance is documented in `0016-monodepth-consumer-contract.md`.

## What the minimum actually changes

The minimum is **not merely an optional optimizer bound**. With the default flag enabled, the implementation rejects affine-depth minimal-solver candidates whose offsets violate minimum-depth conditions. It also checks offsets obtained when fitting affine depths to five-point geometric proposals. The optimizer then imposes offset lower bounds involving each minimum and a 0.01 margin. Disabling the flag removes those proposal checks and offset bounds. The scale lower bound of 0.01 remains. Reprojection scoring still rejects transformed points behind its Z threshold, but that is not an equivalent guarantee of positive corrected depth everywhere in both original maps. [Proposal filtering and fitting](https://github.com/MarkYu98/madpose/blob/c17e179dd4f2794ea7d6942f402f12e15504ead7/src/hybrid_pose_estimator.cpp), [optimizer bounds](https://github.com/MarkYu98/madpose/blob/c17e179dd4f2794ea7d6942f402f12e15504ead7/src/optimizer.h).

A minimum over matched samples is generally no smaller than the whole-map minimum; substituting it can weaken the intended offset bound and change candidate acceptance. Conversely, disabling the constraint must be reported, rather than silently presented as default behavior. A separate `HybridEstimatePoseAndScale` API has no minimum argument, but it is a different scale-only estimator and is not a substitute for the full affine baseline.

## Exact source settings to retain or disclose

The calibrated demo uses reprojection **8 px**, epipolar **2 px**, epipolar weight 1, minimum/maximum RANSAC iterations 100/1000, success probability 0.9999, seed 0, four LO steps, threshold multiplier 5, and final least squares enabled. It enables shifts and the minimum constraint. These reprojection settings differ from the existing RePoseD 16 px setting; do not silently equalize them and call that the original demo. The demo's separate PoseLib comparison also uses a different iteration budget. [Calibrated demo](https://github.com/MarkYu98/madpose/blob/c17e179dd4f2794ea7d6942f402f12e15504ead7/examples/calibrated.py).

The hybrid wrapper expands the two error types to forward reprojection, reverse reprojection, and epipolar residuals. For the demo inputs, squared thresholds become `[64,64,4]` and internal weights `[1,1,32]`. The configuration defaults to HYBRID for proposal generation, scoring, and LO; `EstimatorConfig(0,0,0)` expresses this explicitly. Geometric hypotheses are already fitted with affine depths, so that operation is not a new contribution. [Estimator implementation](https://github.com/MarkYu98/madpose/blob/c17e179dd4f2794ea7d6942f402f12e15504ead7/src/hybrid_pose_estimator.cpp), [CVPR 2025 paper, verified full text](https://arxiv.org/html/2501.05446v3).

Ceres settings copied from `EstimatorConfig` are function tolerance 1e-6, gradient tolerance 1e-8, parameter tolerance 1e-6, maximum 25 iterations, and nonmonotonic steps enabled. The class thread default is 1; the demo sets 8. A future one-thread resource-limited control must disclose that difference. The estimator's copied 25-iteration setting overrides the optimizer configuration's standalone 100-iteration default. Joint inlier optimization uses the source's trivial losses and DENSE_QR. Remaining inherited RansacLib defaults must be pinned and recorded before execution, rather than invented here. [Configuration](https://github.com/MarkYu98/madpose/blob/c17e179dd4f2794ea7d6942f402f12e15504ead7/src/estimator_config.h), [header propagation](https://github.com/MarkYu98/madpose/blob/c17e179dd4f2794ea7d6942f402f12e15504ead7/src/hybrid_pose_estimator.h), [optimizer](https://github.com/MarkYu98/madpose/blob/c17e179dd4f2794ea7d6942f402f12e15504ead7/src/optimizer.h).

## Smallest defensible prospective comparison

1. Separately admit and qualify a pinned build. The source requests CMake >=3.17, C++17, Eigen >=3.4, Ceres >=2.0, OpenCV, pybind11/RansacLib submodules, and PoseLib v2.0.4 by default. The existing PoseLib 3.0 binary is not an established drop-in reproduction. No build feasibility or dependency availability is asserted by this source review. [Build configuration](https://github.com/MarkYu98/madpose/blob/c17e179dd4f2794ea7d6942f402f12e15504ead7/CMakeLists.txt).
2. Before scientific input access, use manufactured scenes to check the binding, output conventions, failure handling, and invariance to changing finite minimum placeholders with the flag disabled. Pin all inherited options and build flags. Do not add corrective clipping or a replacement positivity filter to the official code.
3. Freeze one explicitly named **MADPose affine HYBRID, minimum constraint disabled** arm on the same fixed 80 development pairs, seed 0, one thread, official 8/2 px thresholds and demo budgets. Preserve full coverage and failures; retain raw geometry, parameters, all reported inlier types, statistics, inputs, timings, and hashes. Seal predictions before label evaluation and compare against the already fixed strongest controls. No data-dependent option selection.
4. Report the missing constraint as a limitation. A future exact default arm needs authentic whole-map minima with source binding; do not infer them from the selected correspondences. Neither a nondefault baseline gain nor the already published geometric-to-affine fitting mechanism establishes a novel method.

## Review provenance

Critical pinned source SHA256 values obtained from source bytes:

- `src/estimator_config.h`: `4a76e18f739fc6c659c1de5a5df51846aeade2e9e993ecced38b63951a34bee1`
- `src/hybrid_pose_estimator.cpp`: `af7bd85d240e8786d167922c7771ab2790d226bd2f14d1c916580b8b090f9eea`
- `src/optimizer.h`: `203c8a1e2e7c239cc2230e99653b5f1520a1e6adf3e7f5377d5157f6f8b46376`
- `src/bindings.cpp`: `a873f390bc70f4c3bdb665f28a110ed9b4334d9e45fa61c3981226687227df42`

Explicitly metered source/metadata responses totaled **164,077 bytes**, including repeated source reads: 17,882 bytes for repository-tree metadata and 70,317 + 75,878 bytes for source batches. Browser-tool primary-source response bodies were not byte-metered; 164,077 is therefore not a claim of total network traffic. No weights, image/depth payload, dependencies, or scientific outputs were acquired or read for this review.
