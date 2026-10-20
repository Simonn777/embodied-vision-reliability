# 0024: primary-source boundary for the virtual-rotation falsifier

Source review completed before the 0024 probe. No accuracy result or new solver
is claimed. The admitted question is only whether offsets estimated from one
known virtual rotation axis transfer to another axis on the same frozen depth
predictor. The failed 0023 method and its labels are not being retuned.

**Direct algebraic prior.** Ding et al., *Fixing the Scale and Shift in
Monocular Depth For Camera Pose Estimation* (2025), §2 Eq. (4), gives
`s(dj+v) rj = (d0+u) R r0 + t`. With known rotation, `t=0`, and
`rj=Rr0/h`, this is exactly `s*dj/h + s*v/h - u = d0`, the proposed
shared-offset linear model. §5.2/Fig. 1 and Appendix B.1 also discuss pure
rotation. This is an existing equation, not a new minimal solver or geometry
theorem. [Full paper](https://arxiv.org/html/2501.07742v1#S2)

**Virtual-view alignment and adaptation are established.**

* Rey-Area, Yuan, and Richardt, *360MonoDepth: High-Resolution 360° Monocular
  Depth Estimation* (CVPR 2022), §3.2 converts tangent perspective disparity to
  spherical inverse-distance; §3.3 Eq. (1–3) estimates spatial affine fields on
  overlapping predictions before blending. This is a direct predecessor for
  training-free overlap consistency and affine fusion. Shifts applied before
  and after axial/radial conversion are not interchangeable.
  [Full paper](https://arxiv.org/html/2111.15669v2#S3.SS3)
* Kim, Lee, and Kim, *Calibrating Panoramic Depth Estimation for Practical
  Localization and Mapping* (ICCV 2023), §3.1 Eq. (2–4) uses panorama stretching
  and its known depth transformation for adaptation. Appendix B.1.2 Eq. (14)
  compares pseudo-labels averaged from four rotated panoramas. It preempts a
  general claim that virtual transformations newly enable depth calibration.
  [Full paper](https://arxiv.org/html/2308.14005v1)
* Jung et al., *RPG360: Robust 360 Depth Estimation with Perspective Foundation
  Models and Graph Optimization* (2025), §3.1 Eq. (2) converts axial to radial
  depth. In §3.2, Eq. (4) describes prior affine alignment; its actual Eq. (6)
  and (8) optimize local point/normal consistency and per-face scales. Its six
  non-overlapping cube faces do not instantiate this shared-offset regression,
  but foundation-model geometric alignment/fusion is already its subject.
  [Full paper](https://arxiv.org/html/2509.23991v1#S3.SS2)

**Official predictor control.** Pinned MoGe2 already provides
`infer(fov_x=...)`, recovering the point-map shift with fixed focal length.
It constructs centered intrinsics from one focal scalar; it is not a full
arbitrary-K interface. The 0024 common centered virtual camera makes that API
applicable to all five views. Known FOV and increased token count are existing
controls, not proposed contributions.
[Pinned code](https://github.com/microsoft/MoGe/blob/925b8ed835a7a9cdb7578ba15c658a0afc969030/moge/model/v2.py#L201)

The possible empirical increment is **actively obtaining frozen predictor
responses under exactly known rotations independently of real query motion**,
then using the offset as a fixed prior in a later scale-only pose estimator.
The current diagnostic does not establish that increment is novel or useful.
It tests a necessary predictor-error assumption at bounded cost.

For one rotation, the exact model identifies an offset only when
`[h*d0,h,1]` has rank three and the relative scale is nonzero. Constant `h`
(identity/roll), constant scene depth, and `z=A+B/h` are degenerate; a generic
tilted plane need not be. Small rotations can be weakly conditioned even with
full algebraic rank. No exact-rank claim justifies noisy real identifiability.
Ordinary least squares also has noisy regressors and correlated spatial/view
errors; it is not an unbiased likelihood estimator. A wrong but mutually
rotation-equivariant shape can pass the entire consistency test.

Finally, a nonzero affine pre-correction followed by a free-affine depth/pose
solver preserves the same feasible corrected depths. That branch cannot
establish a new shape constraint. Any later accuracy study must separately
freeze fresh evaluation and compare fixed-offset pose against the strongest
free-affine author setting, geometry-only pose, known-FOV/high-token controls,
and same-forward-count fusion at full pipeline cost. Passing this falsifier
does not meet the project's two-source accuracy or manuscript gate.
