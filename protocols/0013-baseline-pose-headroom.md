# 0013: baseline-only calibrated pose headroom

The0012 scorer family failed its speed criterion and remains closed. This is a
separate baseline diagnostic to observe actual pose errors before proposing a
new quality method. It neither integrates that scorer nor changes its result.
No new-method, superiority or paper claim follows from running a baseline.

## Prospective inputs and information barrier

Use precisely the16 previously allocated lexicographic scene0015 pair keys in
`configs/0012-development-manifest.json`, all10000 supplied matches per pair.
Scene0015 is permanently development; scene0022 and all other pairs stay unread
for this operation. The reused16 pairs are not independent scene replicates.
No data selection based on pose errors, confidence or visible results.

Newly admit the selected pairs' authored camera dictionaries and K1/K2 for
calibration, and their R/t for a separately ordered evaluation stage. This is
the first numeric camera/GT access in this continuation, prospectively declared.
The prediction worker reads only x1/x2,K1/K2,camera1/camera2. Require finite
PINHOLE parameters [fx,fy,cx,cy], positive focal lengths, nonnegative integer
dimension metadata (preserving authored zero placeholders), and
agreement with the saved3x3 intrinsics (relative and absolute tolerances both
1e-6); unsupported cameras fail explicitly.
Pass raw pixels plus authored cameras to PoseLib, without double normalization.
Do not read confidence, matching_score, name fields, images or other pair arrays.

Finish or record a failure for every selected pair, then write a seal binding
all16 output records, source/configuration and manifest hashes. Only after that
seal is verified may the separate evaluator read the selected R/t fields.
No reference pose, oracle score, initial pose or GT-derived choice enters any
estimator call. Retain prediction and evaluation artifacts separately.
The supervisor and evaluator each verify the whole HDF5 file's size/SHA256 by
reading opaque bytes before opening it; this integrity check does not decode
held-out arrays and prevents evaluating labels from a different file.

## One fixed official baseline

Use the official PoseLib2.0.5 release (tag commit
7e9f5f53372e43f89655040d4dfc4a00e5ace11c). The verified CPython3.9 macOSarm64
wheel is812666bytes, SHA256
43a2275d8a336e1d6c41557d2c59366b0f8d1716ff14468602143f9af8bb2d59.
It is installed only in local work runtime with no extra dependencies; existing
NumPy/h5py are reused. Record actual installed module hash and version.

Call estimate_relative_pose once per pair, with its five-point, cheirality-aware
LO-RANSAC and final nonlinear refinement. Fix seed0, min_iterations1000,
max_iterations100000, dyn_num_trials_mult3, success_prob0.9999,
max_epipolar_error1.0pixel and progressive_samplingfalse. Retain the official
final-bundle defaults through an empty option dictionary: Cauchy loss,0.5pixel
scale,max100iterations and source gradient tolerance1e-10. No unmodified
BundleOptions instance overriding that scale, and no supplied initial pose.
Do not tune settings after observing errors or shorten difficult cases to a
different iteration cap. Read exact API/calibration distinctions in the
[source admission](../docs/0013-baseline-source-admission.md).

PoseLib scales the pixel threshold using the average inverse camera focal
length; record its resulting normalized value. This differs from the DMS
wrapper's convention, so this baseline is not called a reproduction of DMSDDD.
Returned RANSAC mask/stats precede final refinement; label them accordingly.

## Execution and failures

Serial pair subprocesses, one native/BLAS thread,90-second external timeout per
pair and30-minute hard total timeout; roughly8GiB memory budget. Preserve stdout,
stderr, actual exits, timing and hashes. All output paths are exclusive: do not
overwrite or repeat a pair. A failed/timed-out pair remains a recorded row and
is not replaced. Unsupported calibration, nonfinite inputs/output, broken source
binding or a data-access violation stops evaluation eligibility and is not
silently repaired. Ordinary estimator/runtime failure is reported with coverage.

Record estimator-call duration separately from loading and total subprocess
wall time; preprocessing/calibration supplied by this worker is measured. No
speed comparison against a new method is made. No paid hardware, network model,
image payload or further dependency download is required.

## Evaluation, interpretation and next decision

Follow official DMS pose conventions: SVD-project saved R to U@Vt and normalize
saved t. Record the projection change and original t norm, refusing invalid
or zero GT rather than inventing a pose. Translation angle retains direction:
acos(dot(t_est,t_gt)), never acos(abs(dot)). Rotation is the angle of
R_est@R_gt.T. Primary pair error is max(rotation,translation), in degrees.
Compute standard trapezoidal pose AUC at5,10,20degrees, reported as fractions
and percentages, plus counts below each threshold and each individual error.

Report attempted/successful coverage and all16 rows. For completeness, failed
estimates receive180degrees only in an explicitly labeled coverage-penalized
all-pair AUC; report successful-only AUC secondarily. This penalty is not a
measurement of an unreturned pose. Invalid GT or broken seals stop metrics.
No population confidence interval from16 overlapping pairs in one scene.

This descriptive headroom study has no candidate-success gate. Its next output
is a specific observed failure/remaining-error question and a comparison plan,
or an explicit finding that the current evidence provides no useful target.
It cannot be reported as improvement. Any new method requires a separate
mechanism/protocol freeze and strong official PoseLib/DMS comparisons, followed
by the unchanged independent two-source accuracy/time and novelty gates before
a manuscript. The old held-out set is not available for this diagnosis.

## Disclosed input-compatibility amendment, before any estimation or GT access

The first frozen attempt at2e6501c stopped on pair0 before the estimator: authored
camera1 width/height are both zero. Preserve `results/0013-baseline/` and its
ineligible seal; it is not a completed baseline. The original worker had already
read that pair's x1/x2 and K1/K2, then rejected the first camera's width.
No other pair was attempted and no R/t was read.

Official PoseLib2.0.5 [camera documentation](https://github.com/PoseLib/PoseLib/blob/v2.0.5/README.md#camera-models)
states these dimensions are unused metadata. The
[relative-pose implementation](https://github.com/PoseLib/PoseLib/blob/v2.0.5/PoseLib/robust.cc#L169-L206)
uses only unprojection and focal values; the Python camera loader stores the
dimensions without positive validation. Our positive-dimension requirement was
an unnecessary compatibility restriction, not a geometric validity requirement.

Permit nonnegative integer dimension metadata without substituting image sizes;
retain positive focal lengths, exact camera model, K checks, all estimator
settings and all other failure gates. Freeze this two-line validation repair
and an artificial zero-dimension regression before one separately named retry,
`results/0013-baseline-calibration-compatibility/predictions`. No pose outcome
has been observed or used to choose this change. Any later failure stays visible.
