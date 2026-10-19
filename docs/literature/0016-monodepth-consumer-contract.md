# 0016: monocular-depth cache consumer contract

2026-10-19. Source-only follow-up to `0016-monodepth-resources.md`; that note,
the active 0016 prediction sources, and all result files remain unchanged. No
cache body, numerical dataset array, image, label, or model was read or run.

**Conclusion:** the pinned calibrated PoseLib consumer interprets each cached
depth as a camera-axis **z-depth proxy**, with optional additive shifts in that
same domain. It does not silently convert depth into distance along a unit ray.
This resolves the algorithm input convention, but does not identify the cache
generator/checkpoint. A comparison using identical author-supplied caches is
therefore defensible within this consumer contract; claims about a named model's
inference accuracy, inference cost, or training independence remain unsupported.

## Exact normalization path

The reviewed local PoseLib source is commit
`6a636b1d84273d5e3ba97676463476569611572a`, whose `VERSION.txt` reports `3.0.0`.
This is a commit/version statement, not a verified `v3.0.0` release-tag claim.
All links below are pinned to that commit.

1. [`estimate_monodepth_relative_pose`, `robust.cc`, lines 383–435](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/PoseLib/robust.cc#L383)
   unprojects pixels into `Point2D` calibrated coordinates. It passes `depth_1`
   and `depth_2` unchanged to RANSAC, then copies the same depths for selected
   inliers into final refinement. Only image-error thresholds/loss scale are
   multiplied by focal-dependent factors.
2. The underlying three-dimensional PINHOLE unprojection normalizes its ray,
   but the [`Point2D` overload, `camera_models.h`, lines 98–102](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/PoseLib/misc/camera_models.h#L98)
   immediately applies `hnormalized()`: it divides by the ray's third component.
   Thus downstream `.homogeneous()` reconstructs `(u,v,1)`, not a unit ray.
   This distinction prevents misreading the new three-dimensional camera API.
3. In the [minimal-proposal path, `relative_pose.cc`, lines 112–140](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/PoseLib/robust/estimators/relative_pose.cc#L112),
   scale-only estimation constructs `X = d1 * x1.homogeneous()`. Its target
   bearings are normalized for the P3P solver; this does not alter the source
   3D points or reinterpret `d1`. The scale-plus-shift solver receives
   homogeneous rays and unchanged depths.
4. The [calibrated refiner, `monodepth_relpose.h`, lines 133–165](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/PoseLib/robust/optim/monodepth_relpose.h#L133),
   explicitly forms forward and reverse transformed points from
   `(d1 + shift1) * x1.homogeneous()` and
   `scale * (d2 + shift2) * x2.homogeneous()`, respectively. The corresponding
   Jacobian repeats these constructions. This is direct implementation evidence,
   not an inference from method names.

Writing `h=(u,v,1)` and `b=h/||h||`, physical axial depth `z` and ray distance
`rho` satisfy

\[
X=z h=\rho b,\qquad \rho=z\lVert h\rVert.
\]

The consumer's model is

\[
R[(d_1+\beta_1)h_1]+t=s(d_2+\beta_2)h_2.
\]

Consequently, feeding a range prediction `rho` unchanged into this interface
would construct `rho*h`, adding an off-axis, point-dependent scale distortion.
It generally cannot be repaired by one image-wide scale/shift. Equivalently,
an implementation using unit bearings would need range
`(d+beta)*||h||`; converting only `d` while retaining an unchanged scalar shift
would change the model. These are algebraic consequences of the consumer,
**not evidence that the unread cache contains range or needs conversion**.
The faithful reference treatment is to pass the author fields unchanged.

## Explicit options for a future comparison

The pinned [monodepth option parser, `helpers.h`, lines 153–173](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/pybind/helpers.h#L153)
reads top-level `max_errors=[reprojection, epipolar]`, `estimate_shift`, and
`weight_sampson`, with nested `ransac` and `bundle` dictionaries.
[`MonoDepthRelativePoseOptions`, `types.h`, lines 155–168](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/PoseLib/types.h#L155)
confirms that order and defaults `[12,1]`. The intended PoseBench setting is
therefore explicitly `[16.0, 2.0]` pixels, with RANSAC min/max iterations
100/1000 and success probability 0.9999. The calibrated wrapper converts both
thresholds with `a=0.5*(1/focal1+1/focal2)`; it leaves depths unchanged. Its
reprojection/Sampson weighting ratio is `(2/16)^2=1/64`.

The reviewed current [PoseBench configuration](https://github.com/PoseLib/posebench/blob/7b494256e89a13c47ab37b6917c2e0d3dae0c2e4/src/posebench/monodepth_relative_pose.py#L95)
already uses that nested layout. It should not be confused with the historical
flat options. **There is nevertheless a concrete control-setting hazard:** its
ordinary essential-matrix wrapper forwards the same dictionary containing
plural `max_errors`, while the pinned
[`update_relative_pose_options`, `helpers.h`, lines 108–120](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/pybind/helpers.h#L108)
reads singular `max_error`. A future matched-threshold E baseline must explicitly
set `max_error=2.0`; otherwise the source indicates the default 1-pixel threshold
would remain. This is a static interface finding, not a claim about any past
published result or a measurement. No wrapper was executed here.

## What provenance history does and does not establish

PoseBench commit `7b494256e89a13c47ab37b6917c2e0d3dae0c2e4` is dated
2026-03-18. Its [loader](https://github.com/PoseLib/posebench/blob/7b494256e89a13c47ab37b6917c2e0d3dae0c2e4/src/posebench/monodepth_relative_pose.py#L115)
reads `depth1/depth2` directly; its
[calibrated estimator wrapper](https://github.com/PoseLib/posebench/blob/7b494256e89a13c47ab37b6917c2e0d3dae0c2e4/src/posebench/estimators.py#L224)
passes them directly to PoseLib. This connects the cache fields to the consumer.

Targeted history inspection reached the
[initial monodepth evaluation addition, 4c1556f, 2025-10-17](https://github.com/PoseLib/posebench/commit/4c1556f19787c458db584556162f679e4ce7dc20).
Its complete added `src/posebench/monodepth_pose.py` already listed the same
24 scene/matcher/depth basenames and directly loaded these depth fields. The
[November 2025 API-renaming commit](https://github.com/PoseLib/posebench/commit/b656c8ab6a121768b834e3662a75767de33be15b)
changed the consumer call rather than providing an exporter. The complete
repository trees at the initial addition and March 2026 pin, and these relevant
loader/history files, did not identify an exporter or checkpoint manifest.
This is a bounded source-search result, not proof no such artifact exists
elsewhere. Filename continuity also does not prove the hosted file bytes were
unchanged between dates.

**Still absent from the reviewed evidence:** exact pretrained checkpoint name
and hash, model generation, inference resize/crop, the current cache exporter
revision, and its interpolation/filtering rules. The separate RePoseD
`make_pair.py` demo's MoGe-2 model name and `prepare_pt_test.py`'s sampling of
precomputed arrays, discussed in the earlier resource note, do not bind these
hosted files to that pipeline. The suffixes `moge` and `unidepth` are author
cache labels; they are not checkpoint certificates.

## Defensible experiment scope if separately admitted

Treat each acquired, content-hashed file as an **opaque author-supplied input
bundle**, evaluated under the explicit calibrated z-proxy consumer contract.
Give every compared depth method the same ordered coordinates, camera models,
depths, validity policy, hypothesis seeds, and time accounting. Compare at least
point-only E, scale-only RePoseD, and scale-plus-shift RePoseD with explicitly
translated settings; any proposed transformation must improve beyond the
existing affine-depth control. Preserve the author fields as the reference arm.
Separate valid/invalid coverage from accuracy and retain every preallocated pair.

This permits a claim about **downstream pose estimation on these fixed cached
predictions**, without reproducing the neural front end. It does not permit an
end-to-end runtime/energy claim, an exact-model depth accuracy claim, or an
unseen-training-data claim. Geometric depth reconstructed later from reference
poses would still be a correspondence-conditioned triangulation proxy, not
independent sensor truth. No method or experiment is admitted by this note.

Read depth: complete relevant local camera overload, calibrated robust wrapper,
proposal/scoring/refinement functions and option parsers; complete relevant
PoseBench loader/estimator files; initial-addition/API-renaming history; complete
tree metadata at two revisions. No dataset headers or bodies were accessed.
