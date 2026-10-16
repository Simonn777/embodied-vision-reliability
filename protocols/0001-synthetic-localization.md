# Protocol 0001: synthetic localization and abstention diagnostic

**Version:** 1, 2026-10-16. **Status:** specified before the first complete run; the freezing Git commit is the authoritative version. **Experiment ID:** `0001-synthetic-localization`.

This is a pipeline diagnostic on deliberately constructed RGB scenes. It is not a confirmatory study, an evaluation of a learned vision model, or a claim that a new error detector is better. Planning and implementation are AI-assisted; see [contributions](../docs/CONTRIBUTIONS.md). The broader [research charter](../docs/RESEARCH_CHARTER.md) describes the twelve-week question and the evidence needed to advance beyond synthetic data.

## Question and deliberately limited interpretation

For exactly the same image-derived position predictions, how do a current-image score and that score augmented with a causal motion residual change accepted-prediction error and coverage? Does an appearance change create a confidently wrong estimate that either score fails to reject? Does an abrupt motion change cause temporal rejection of a correct estimate?

The two scores are fixed heuristics, not calibrated probabilities. The experiment is designed to exercise the measurement pipeline and reveal counterexamples. Its renderer intentionally creates failures of a simple color segmenter. Success on these scenes would not establish novelty, general reliability under shift, physical-state observability, or performance on natural video. The state is an **image-plane 2D target center**, in pixels; no depth, 3D pose, physical units, robot actions, viewpoint geometry, or real camera is measured.

## Independent unit, partitions, and randomization

There are **160 base sequences**, each **32 frames of 96 × 96 RGB pixels**, rendered in each of four paired conditions. A base sequence, including all its condition variants and all its frames, belongs to exactly one partition:

| Partition | Base sequence IDs | Count | Use |
| --- | --- | ---: | --- |
| Fit | `000`–`079` | 80 | Reserved for future fitting; no parameters are fitted in this diagnostic |
| Development | `080`–`099` | 20 | Implementation and diagnostic checks; no score tuning in version 1 |
| Calibration | `100`–`119` | 20 | Clean-condition threshold selection only |
| Final test | `120`–`159` | 40 | One frozen diagnostic evaluation, all four conditions |

Frame 0 provides the initial query and observed reference features. It is excluded from every accuracy, coverage, threshold, and curve calculation. Each condition therefore has 620 calibration frames and 1,240 test frames, while the independent test sample remains **40 sequences**, not 1,240 independent observations. The fit and development labels do not imply that a trained model exists.

The master seed is `20261016`. Per-scene and per-frame generators use the first eight SHA-256 digest bytes, interpreted as an unsigned little-endian integer, of UTF-8 `master|sequence_id|stream|frame`. Sequence IDs have three digits. The scene stream is `scene` with frame `-1`; frame noise uses `background` with the integer frame index. NumPy `default_rng` consumes these seeds; record the NumPy version. All condition variants reuse the scene parameters and corresponding frame noise, so comparisons are paired. The method-independent tie key is the hexadecimal SHA-256 digest of UTF-8 `sequence_id|frame`, with the same three-digit sequence IDs. No labels enter a tie key.

## Renderer and conditions

Pixel coordinates are integer `(x, y)` locations from 0 through 95. Scene parameters are sampled once per base sequence: target radius uniformly from integers 5 through 8 inclusive; initial center uniformly from `[28, 68]` in each coordinate; target color `(0.20, 0.75, 0.25)` plus independent uniform `[-0.03, 0.03]` channel offsets; velocity angle uniform on `[0, 2π)`; speed uniform on `[0.6, 1.2]` pixels per frame. Motion reflects at center-coordinate bounds `[16, 80]`, preserving any overshoot rather than clipping and losing it. A disk contains pixels whose squared distance to its center is at most its squared radius.

The background is RGB `(0.12, 0.12, 0.12)` plus independent per-pixel/channel Gaussian noise of standard deviation 0.015, clipped to `[0, 1]`. The disk is painted in its constant target color without added target noise. Rendered RGB is quantized using `round_ties_to_even(255 × clip(image, 0, 1))` to unsigned 8-bit pixels; the estimator converts these pixels back to `[0, 1]` floats. Frame 0 is clear in every condition. Perturbations are fixed before results are seen:

| Condition | Intervention | Purpose and limitation |
| --- | --- | --- |
| `clean` | Unmodified target and motion throughout | Nominal reference; not a realistic scene distribution |
| `occlusion` | Frames 8–23 inclusive paint an opaque RGB `(0.20, 0.22, 0.25)` rectangle over `center_x - radius - 2 <= x <= center_x + 0.35 × radius` and `center_y - radius - 2 <= y <= center_y + radius + 2`; in frames 14–17 extend the right edge to `center_x + radius + 2`, fully covering the target | Partial-mask centroid bias and missing observations; renderer visibility is evaluation metadata only |
| `appearance` | Frames 8–31 use target color `0.45 × original + 0.55 × (0.12, 0.12, 0.12)` and add a same-radius, original-color distractor at the farthest of `(18,18)`, `(78,78)`, `(18,78)`, `(78,18)` from the initial target center; ties use that listed order | A deliberately adversarial identity switch with plausible area/shape; does not model the full diversity of appearance shifts |
| `abrupt_motion` | Immediately before the transition into frame 16, rotate velocity +90 degrees and multiply its speed by two, once; continue with the same reflection rule | Control for rejection caused by changed motion despite a clear image; not a camera-viewpoint shift |

The occlusion mask and distractor position depend on renderer state. This is allowed for constructing a controlled challenge, but those quantities are not inputs to the estimator or scoring code. Initial conditions, noise, and motion up to the intervention are shared across variants. RGB frames must actually be rendered and consumed; the estimator must not receive renderer masks or later true centers.

## Fixed position estimator

The estimator receives the RGB sequence and the **provided true center in frame 0 as an initialization query**. This privileged initialization is explicit and resembles a point-tracking query; it is not autonomous target discovery. A 5 × 5 patch centered at the query, rounded to the nearest integer using ties-to-even, supplies the mean reference RGB color. The radius ensures this patch lies inside the initial disk in these scenes.

For each frame, threshold Euclidean RGB distance to the reference color at `<= 0.18`. Select the largest 4-connected component, breaking equal-area ties by its smallest row-major pixel index. Its mean pixel coordinate is the predicted target center. An empty mask yields a missing prediction. The same centroid predictions are used by every deployable selection method; temporal information never changes them. No ground-truth tracking, identity matching, renderer visibility, or future frame may repair a prediction.

Let `a_t` be the detected component's pixel count and `C_t` its population pixel-coordinate covariance (divide by `a_t`, not `a_t - 1`). Let `λ_max,t` and `λ_min,t` be its eigenvalues, each floored at 0.25 pixel², and `h_t = λ_max,t / λ_min,t`. Frame 0 establishes observed `a_0` and `h_0`. If the initial component is missing, fail the sequence initialization explicitly; do not replace it with ground truth. This is a renderer/estimator validity failure requiring investigation, not a zero-error sample.

The detected area is an observable proxy, **not the true visible fraction**. The pixel covariance describes a component's shape, **not a calibrated uncertainty covariance for its centroid**.

## Fixed selection scores and controls

Lower scores indicate acceptance priority. No weights, cutoffs, or score features are fitted or selected from this experiment's outcomes.

1. **Static image score:** `s_static(t) = abs(log(a_t / a_0)) + 0.25 × max(0, log(h_t / h_0))`.
2. **Temporal score:** `s_temporal(t) = s_static(t) + min(d_t / sqrt(a_0 / π), 10)`. The motion residual `d_t` is the Euclidean distance from the current predicted centroid to a constant-velocity extrapolation from the last two earlier valid **raw predictions**, using their actual frame timestamps. With fewer than two earlier valid predictions, the residual is zero. This warm-up limitation must remain visible. Missing frames do not fabricate a centroid; extrapolation spans elapsed time. Rejected but valid raw predictions still update history, so selection has no feedback into future scores or predictions.
3. **Random ranking:** the deterministic label-independent hash key supplies a uniform-like ordering. It is an empirical negative control, not a guarantee of independent temporal errors.
4. **Accept all available:** accept every valid centroid; missing predictions still count against coverage. This is the full-prediction baseline, with no calibrated threshold.
5. **Oracle ranking:** order valid predictions by their actual localization error, with the same tie key. This unavailable-at-deployment control appears only in descriptive ranking curves; it must never be presented as an implementable score or threshold policy.

An empty component forces abstention for every policy. Ground truth may appear only in metrics, labeled oracle ranking, diagnostic visualizations, and evaluation strata. A causal implementation must pass a prefix test: changing future frames cannot change earlier predictions or deployable scores.

## Threshold selection and ties

Select a separate threshold for static, temporal, and random policies using **only clean calibration frames 1–31**. The target is 80% of **all** calibration frames, including missing ones. Set `m = ceil(0.8 × N_calibration)`. Sort valid frames by `(score, tie_key)`; for random ranking use score zero and the tie key as its ordering. If at least `m` valid predictions exist, use the `m`th pair as the threshold and accept future valid predictions whose pair is less than or equal to it. If fewer than `m` but at least one valid prediction exist, use the largest observed calibration pair, accepting all valid calibration predictions, and report that the requested calibration coverage is unreachable. The same finite threshold still applies on test: a larger test pair remains rejected. With no valid calibration predictions, record an undefined threshold and reject all test predictions for that policy. Missing predictions always remain rejected.

Hash tie breaking is part of the frozen policy; do not silently accept all equal-score frames or retune per condition. Report realized calibration coverage. Apply each frozen threshold unchanged to every test condition. Neither 80% test coverage nor a test risk bound is guaranteed. This is empirical coverage calibration, **not conformal prediction or conformal risk control**. In particular, correlated frames and distribution shifts do not justify an exchangeable-frame guarantee.

## Endpoints and complete failure accounting

Normalize localization error by the fixed image diagonal `D = sqrt(96² + 96²)` pixels. For valid predictions, `e_t = ||prediction_t - truth_t||₂ / D`. A valid prediction is a localization failure when `e_t > 0.025` (strict inequality). The tolerance is fixed as a diagnostic choice and has no physical safety interpretation.

For each test condition and policy, export underlying frame rows and sequence-level totals, then report:

- Mean and 90th percentile normalized error **among valid predictions**, together with missing-prediction rate. Missing prediction error is `null`, not zero; do not silently remove missing frames from denominators that should include them.
- All-frame estimator failure rate: `(valid errors above tolerance + missing predictions) / all evaluated frames`. It describes the common estimator, independent of the gate.
- Coverage: `accepted predictions / all evaluated frames`.
- Selected mean normalized error and selected failure rate among accepted predictions. With zero accepted predictions, both are `null`/undefined, not zero.
- Coverage and selected-risk differences for temporal minus static, paired by base sequence. Less selected error with much less coverage is not sufficient evidence of a better selector.

Produce descriptive risk–coverage curves for static, temporal, random, and oracle orderings within each condition. Sort all valid test predictions by their frozen score/key ordering solely to describe ranking, not to choose a deployable test threshold. At prefix length `j`, coverage is `j / N_all` and risk is the fraction of that prefix above the fixed error tolerance. The discrete AURC is `sum(prefix_risk_j) / N_all` over all valid ranks. Its domain stops at attainable coverage `N_valid / N_all`; with no valid predictions it is undefined. Always report attainable coverage alongside AURC. These are ranking diagnostics, not a new operating threshold; comparing AURC across conditions with different attainable coverage can be misleading. Oracle uses the same base predictions and missingness.

The main diagnostic comparison is temporal versus static **with their calibration-derived thresholds**, reporting risk and coverage together. There is no prespecified superiority margin or significance-based success declaration in protocol 0001. A later study must fix a practical effect size, matched-coverage or utility comparison, and sample-size justification before its own final evaluation.

## Uncertainty and paired resampling

Use a paired cluster bootstrap with **1,000 replicates**, seed `20261017`, and percentile 95% intervals. Sample 40 test base-sequence IDs with replacement per replicate. Carry each selected sequence's complete frames, all conditions, and all policies together. Recompute pooled numerator/denominator metrics and temporal-minus-static differences within each replicate; do not average ratios with a different estimand or independently resample the two methods. Keep the original calibration thresholds fixed.

These intervals describe variation across the 40 generated trajectories conditional on this renderer, frozen calibration sample, and protocol. They do not quantify calibration-set uncertainty, robustness to another renderer, or performance on natural video. Bootstrap samples with no accepted predictions yield undefined conditional risk; report their count and the number of valid replicates used, rather than replacing undefined values with zero. Report an interval only if at least 95% of bootstrap replicates are defined (950 of the complete run's 1,000 replicates); otherwise its endpoints are undefined. This fixed reporting rule does not validate a conditional interval when abstention is extreme. Do not use per-frame binomial confidence intervals or count condition variants as independent extra sequences.

## Execution, artifact, and regression checks

Freeze this protocol in a Git commit **before the complete test run**. Unit tests and bounded fit/development smoke checks may run first. Record the protocol/source revision, resolved configuration, library versions, platform, seeds, exact command, duration, and raw results. Store source/configuration hashes with generated artifacts; omit private absolute paths. Generate a small set of representative RGB frames and overlays from a prespecified sequence (test ID `120`, frames 0, 8, 14, 16, 23, and 31) with visible condition, prediction, and ground-truth labels. These are diagnostic examples, not cherry-picked success illustrations.

Required tests cover:

- Repeat rendering is deterministic; split IDs and paired variants do not overlap across partitions.
- The estimator reads RGB pixels, and a fixed image/prediction stream yields the same deployable score when hidden renderer truth is changed.
- Future-image changes cannot affect earlier results; rejected valid predictions still update motion history identically.
- Component ties, empty masks, one-pixel masks, covariance regularization, and missing/short history have explicit behavior.
- Frame 0 is excluded; calibration uses clean/calibration IDs only; hash ties yield the specified order; impossible coverage targets are reported.
- Missing predictions remain in coverage/all-frame failure denominators; all-rejected selected risk is undefined; error tolerance uses strict `>`.
- Bootstrap resamples full paired sequences and handles undefined ratios; the saved thresholds are reused unchanged on test conditions.
- A second execution from the same source/configuration reproduces numerical artifacts; volatile run timestamps/durations are compared separately.

The full diagnostic uses CPU only, at most four threads and approximately 8 GiB memory. Aim for completion within 15 minutes; the repository's hard experiment timeout is 30 minutes. Generate frames sequentially or in bounded batches rather than retaining all 20,480 RGB arrays. If the run fails, preserve the error and report it. Do not silently reduce the held-out sample or change conditions to obtain a favorable result. A necessary protocol change requires a new version and a dated reason. After results are visible, methodological changes and reruns are exploratory; they cannot reuse this test set as untouched confirmatory evidence. An exact reproduction is an integrity check, not a second independent experiment.

## Progression and claim boundaries

This experiment can establish that the code, split discipline, coverage accounting, causal feature flow, and reproducibility machinery work on a transparent toy problem. A result may identify a failure mode worth testing. It cannot establish a novel algorithm, a competitive computer-vision benchmark, real-robot reliability, or NVIDIA hardware performance.

The next protocol should use a licensed public point-tracking/video source, a verified fixed visual estimator, and sequence/scene-disjoint partitions. Check the exact data/checkpoint terms, annotation semantics, and online versus finite-lookahead behavior before acquisition or evaluation. Do not tune on an official evaluation set and then report it as untouched. Natural-video validation, stronger confidence controls, and an explicit practical effect threshold are required before an efficacy claim.

Selective prediction and uncertainty under shift are established research topics. Relevant primary foundations are [Geifman and El-Yaniv, 2017](https://arxiv.org/abs/1705.08500), [Ovadia et al., 2019](https://arxiv.org/abs/1906.02530), and [Angelopoulos et al., Conformal Risk Control](https://arxiv.org/abs/2208.02814). They motivate the evaluation and its assumptions; they are not evidence that this heuristic meets their theoretical guarantees. [TAP-Vid](https://proceedings.neurips.cc/paper_files/paper/2022/hash/58168e8a92994655d6da3939e7cc0918-Abstract.html) is one candidate source for the later real-video stage, subject to the repository's data-rights audit.
