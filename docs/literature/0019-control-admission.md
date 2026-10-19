# 0019: smallest useful accuracy control, before another scorer

Source-only review, 2026-10-19. No scientific array, saved outcome, native call, model, build or installation was accessed for this note. The 0018 replay and tracked sources remain unchanged. **Recommendation: first measure the two author-recommended truncated-Cauchy depth arms on all 80 fixed development pairs.** This is 160 ordinary calls with the existing pinned binary and inputs, and answers an accuracy question directly. It is an existing-method control, not a contribution.

## Exact current API and intervention

The [RePoseD author README, 7955c0761a2fa7de30d18aa03dd888321bf2a0dc](https://github.com/kocurvik/mdrp/blob/7955c0761a2fa7de30d18aa03dd888321bf2a0dc/README.md#use-in-your-own-project), recommends `TRUNCATED_CAUCHY` final refinement. Its example uses an older argument layout. For our already pinned PoseLib `6a636b1d84273d5e3ba97676463476569611572a`, use:

```python
opt = {
    "ransac": {"min_iterations": 100, "max_iterations": 1000,
               "dyn_num_trials_mult": 3.0, "success_prob": 0.9999,
               "seed": 0, "progressive_sampling": False,
               "max_prosac_iterations": 100000},
    "bundle": {"loss_type": "TRUNCATED_CAUCHY"},
    "max_errors": [16.0, 2.0],
    "estimate_shift": False,  # second arm True
    "weight_sampson": 1.0,
}
geometry, info = poselib.estimate_monodepth_relative_pose(
    x1, x2, depth1, depth2, camera1, camera2, opt, initial_pose=None)
```

This layout is explicitly handled by [the current option parser](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/pybind/helpers.h#L153) and [binding](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/pybind/bindings/estimators/relative_pose.cc#L54). `max_errors` means `[reprojection, epipolar]`; do not use the README's old flat names. The nested parser deliberately omits `score_initial_model`; `initial_pose=None` is the actual no-initial-model condition. Authored raw pixels/camera dictionaries and unmodified axial-depth samples remain identical to 0017.

Only final loss changes. Internal proposal/terminal LO constructs its own **TRUNCATED, 25-iteration** options, independent of `opt.bundle`. Final refinement retains default 100 iterations, gradient tolerance `1e-12`, step tolerance `1e-8`, relative-cost tolerance `1e-10`, initial/min/max lambda `1e-3/1e-10/1e10`, Nielsen update and Levenberg damping. The [wrapper](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/PoseLib/robust.cc#L383) overrides final `loss_scale`: with `a=.5*(1/f1+1/f2)`, it is `a`, while internal loss scale is `2a`; `scale_reproj=2²/16²=1/64`. Consequently the final truncation corresponds to 1-pixel epipolar and 8-pixel reprojection cutoffs in this average-focal normalization. Do not pretend a supplied `bundle.loss_scale` survives the wrapper.

The [implemented loss](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/PoseLib/robust/robust_loss.h#L138) is Cauchy up to the cutoff, constant beyond it, with zero derivative weight beyond the cutoff. It is not a visibility/negative-depth repair. Final support is still the pre-final epipolar mask; returned RANSAC statistics/masks are not recomputed after final refinement. Because the intervention occurs after RANSAC, require the returned pre-final mask and RANSAC statistics to agree with 0017; record any mismatch as an implementation/qualification failure rather than explaining it as an algorithm gain.

## Minimum decisive comparison

Freeze these two arms once, seed 0, all 80 pairs, one thread and existing resource caps. Retain all failures with the same failure penalty; no replacement cases or loss/threshold search. Seal all predictions before evaluation. Reuse existing 0017 predictions for the two CAUCHY arms and geometric `e_norm`/`e_ts`; no additional geometric call is needed. Report each truncated-Cauchy arm against its own default and against fixed `e_norm`: directed and sign-folded AUC5/10/20, all five scene results, every pair's delta, ≥1-degree regressions and crossings of 5 degrees. Old timing is historical; separately report new complete-call cost without claiming a cross-run speedup.

This comparison can deliver a real accuracy increase from a stronger existing baseline. If it removes most observed damage, the original baseline settings explain that portion of the gap; it is not evidence for a new method. If it leaves damage, 0018's lineage/support evidence can identify which part persists. A small positive aggregate, a ground-truth oracle, or a win only on a chosen scene does not admit paper writing. Even a substantial development gain from this option becomes the stronger control that any proposed method must subsequently beat.

## Proposal coverage: genuinely unresolved, with an API trap

The [current depth estimator](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/PoseLib/robust/estimators/relative_pose.cc#L111) generates **only** depth-based three-point candidates: scale-only uses P3P from first-view depths; affine uses `relpose_monodepth_3pt`. It never samples the point-only five-point solver. Its candidate score is **ordinary normalized Sampson MSAC**, not a joint depth score. Depth enters proposal generation and joint LO. Thus “joint solver” does not mean the same candidate coverage or hybrid scoring as MADPose. The four-output oracle does not reveal whether good unseen proposals were generated and discarded; the 0018 rolling proposal hash is not a stored oracle over all hypotheses.

There is a concrete warm-start bug to avoid: although the Python binding accepts `initial_pose` and enables initial scoring, [calibrated `ransac_monodepth_relpose`](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/PoseLib/robust/ransac.cc#L199) **unconditionally replaces its input rotation/translation with identity/zero**. A proposed E-seeded run through that API would not test the intended seed. Do not execute it as a control without a separately qualified source fix. No such fix is made here.

If the stronger final-loss control still leaves headroom, the smallest next proposal test is a separately frozen **known geometric-seed control**, not another residual transform: retain the saved `e_norm` pose, fit its depth calibration without labels, and compare joint refinement from that seed against native three-point initialization under the same final loss. Charge calibration and the geometric estimate in full-pipeline costs. An improvement would show that the available geometric seed/initialization helps this pipeline; it would not by itself distinguish proposal diversity, calibration fitting, and LO basin effects. Use an unrefined calibrated-seed control to separate these effects. Do not sweep seeds, choose a seed using ground truth, or claim that this reproduces official MADPose.

One additional source-level fragility deserves a manufactured qualification, not a scientific claim: scale-only initialization divides one transformed x coordinate by `depth2[sample[0]]*x2[sample[0]].x()`. Near the second principal-axis coordinate this can be unstable even when P3P pose is finite. A robust multi-coordinate scale fit is a known numerical control; whether this failure occurs on the 80 pairs is unknown.

## MADPose boundary

[MADPose's paper, §§4.3–4.4](https://arxiv.org/html/2501.05446v3#S4.SS3) already lifts geometric proposals by fitting monocular-depth affine parameters and uses hybrid scoring/LO. Therefore affine calibration, mixing five-point/depth proposals, and rescoring them with depth are not novel. Its [official calibrated example](https://github.com/MarkYu98/madpose/blob/main/examples/calibrated.py) uses `HybridEstimatePoseScaleOffset`, 100–1000 iterations, seed 0, four LO steps, threshold multiplier 5, final least squares, and thresholds 8/2 pixels. It supplies **whole-depth-map minima** to the enabled minimum-depth constraint. Our current contract supplies only matched-point depth samples; replacing a whole-map minimum with their minimum would change the constraint. A complete official MADPose comparison must resolve that input difference and pin its implementation/dependencies; it cannot be claimed from a four-candidate approximation. The example's point-only baseline also has a different iteration budget, so copied demo numbers are not a matched-cost comparison.

The actionable first step is therefore the 160-call author-loss control above. A follow-up that merely fixes warm starts, stabilizes a scale division, or adds known hybrid proposals may improve accuracy and be useful engineering, but is not yet a defensible distinct CVPR contribution. A new method needs a further, falsifiable mechanism beyond those controls and substantial unchanged-evaluation improvement on untouched scenes.

Subsequent execution note: 0018 reconstruction stopped on a nonfinite initial state before label access. Its mechanism gate is unassessed. The source-level rationale for this author-loss control does not depend on recovering or relaxing that failed diagnostic.
