# 0028: single-image normal-assisted axial-depth offset — source/math screen

Status: source-only, 2026-10-20. Exactly three literature search queries; primary full text and author code inspected. No scientific arrays, models, fitting, new payloads, or experiments. This is separate from the rejected two-normal-pair rotation / two-point pose construction and does not reopen 0027.

**Conclusion.** A separately predicted normal field can constrain a global additive axial-depth offset before pose fitting, under perspective projection and a correct shared camera convention. The equation and its one-variable least-squares solution are standard normal–depth consistency, not a new solver or integration principle. The strongest immediate overlap is an already-shipped normal-guided refinement helper in the pinned MoGe2 code. The remaining empirical question is whether a very restricted global correction uses that normal output more reliably than existing point-based shift recovery and normal integration. No evidence here establishes that, or establishes novelty.

## Algebra and identifiable information

Let \(r=K^{-1}(u,v,1)^T\), \(E=\partial r/\partial(u,v)\), raw axial depth \(d\), and \(g=\nabla d\). Assume \(z=s(d+b)\), with spatially constant \(s>0,b\), and a camera-frame surface normal \(n\) at the same pixel. Then

\[
\partial(zr)=s[(d+b)E+r g^T],\qquad
(d+b)n^TE+(n^Tr)g^T=0.
\]

Writing \(A=n^TE\) and \(c=dA+(n^Tr)g^T\), fixed-weight least squares gives
\[
\hat b=-\frac{\sum_p w_p A_pc_p^T}{\sum_p w_p\|A_p\|^2}.
\]

This is a one-dimensional restriction of the known perspective relation
\(\nabla\log z=-n^TE/(n^Tr)\). Scale remains unobservable. The division-free tangency form avoids an explicit grazing-angle division, but does not remove errors-in-variables or poor conditioning. Both \(A,c\) change sign under \(n\mapsto-n\); normal orientation sign therefore cancels here, unlike cross-view rotation estimation.

Important counterexamples and assumptions:

- A frontoparallel constant-depth patch has \(A=0,g=0\), so every offset is compatible. Orthographic normal integration also cannot identify an additive depth constant. Weak perspective, small slopes, and almost-frontoparallel support can be numerically weak.
- If normals are computed from exactly the same \(d,g,K\), the residual is already zero at \(b=0\). There is no independent correction signal. A separate learned head is an additional output, **not** evidence of statistically independent errors: encoder features, training targets and biases can be shared.
- Tangency requires axial depth, consistent pixel/crop coordinates and intrinsics, and normals expressed in the same camera frame. A range map, normal-frame reflection, or focal mismatch can masquerade as an offset.
- Discontinuities do not have this differential model. Depth, gradient and normal errors enter both coefficient and residual; ordinary LS is not automatically an unbiased or angular maximum-likelihood estimator. A small training residual can reflect common bias. Corrected-depth positivity must be explicit; clipping invalid values would change the model.
- A global offset preserves all raw depth differences. It cannot repair general spatial shape error.

## Closest verified primary sources

| Source and exact material read | What it already supplies; remaining distinction |
|---|---|
| [MoGe2 pinned `geometry_torch.py`](https://github.com/microsoft/MoGe/blob/925b8ed835a7a9cdb7578ba15c658a0afc969030/moge/utils/geometry_torch.py), lines 206–237, `refine_depth_with_normal`; [model `v2.py`](https://github.com/microsoft/MoGe/blob/925b8ed835a7a9cdb7578ba15c658a0afc969030/moge/model/v2.py), lines 50–55, 166–178, 248–296; [NumPy geometry](https://github.com/microsoft/MoGe/blob/925b8ed835a7a9cdb7578ba15c658a0afc969030/moge/utils/geometry_numpy.py), lines 79–115 | The helper computes the perspective log-depth gradient from normals, then applies damped, discontinuity-weighted updates. Defaults: 10 iterations, damping \(10^{-3}\), kernel 5. A source-wide `.py` search found only its definition, not an `infer` call. Default inference instead fits point-map reprojection for focal/Z-shift (or shift with known focal), optionally projects points, and applies metric scale. Normal and point heads share features. Thus default point recovery and the shipped optional helper are different mandatory controls. The helper mutates its depth input through an in-place clamp; any future comparison must copy inputs and disclose this behavior. No execution was performed. |
| [Metric3D v2](https://arxiv.org/html/2404.15506v2), §§III-B–D, especially Eqs. 2–5 and §III-D normal loss | A learned recurrent module jointly updates depth and normals; a consistency loss compares predicted normals to normals derived from predicted real-camera depth using local least squares. This is direct prior for exploiting joint depth/normal consistency. The checked sections do not describe the proposed explicit test-time global-\(b\) solve. |
| [Bilateral Normal Integration, ECCV2022 author repository](https://github.com/xucao-42/bilateral_normal_integration), perspective/fusion README; [author CPU implementation](https://raw.githubusercontent.com/xucao-42/bilateral_normal_integration/main/bilateral_normal_integration_cpu.py), lines 247–455 | Perspective integration optimizes log depth and can use a sparse or dense depth prior. The README explicitly distinguishes the later depth–normal fusion code from the ECCV paper. This is a stronger same-input shape-correction control than zero correction. Current-main code has explicit OpenGL-normal/OpenCV-camera conversions; its `offset` in log-depth alignment is a scale gauge, **not** the proposed additive axial offset. Current source was read, not pinned or qualified for execution. |
| [NDDepth: Normal-Distance Assisted Monocular Depth Estimation](https://arxiv.org/html/2309.10592v2), §§3.1–3.3, Eqs. 1–3 | Uses the perspective plane relation \(z=D_{\rm plane}/(n^Tr)\), a normal/distance branch and learned depth refinement. It is another direct normal-informed depth prior, but predicts plane distance rather than solving one global offset of an existing scalar field. |

[MoGe-2 paper](https://arxiv.org/html/2507.02546v1), §§3.1–3.2, describes affine point maps, camera recovery and metric scale. Its inspected v1 text does not describe the later source normal helper; do not attribute that code feature to those paper sections. This bounded search is not proof that no paper has already used exactly the scalar restriction.

## Conditional empirical question and controls

The narrow question is: **after the existing known-FOV point-map recovery, does a separately predicted normal head contain reliable residual offset information, and does restricting its use to one global variable prevent the detail distortion of more flexible integration?** This would be a comparison of existing principles and model outputs; a positive answer alone would not establish a CVPR contribution.

Minimum same-information controls are: (1) unchanged default MoGe2 with supplied camera/FOV and existing point-based shift recovery; (2) the plain scalar LS above, plus a prospectively fixed robust scalar fit and a scalar angular-normal-consistency objective; (3) the shipped normal-refinement helper and a properly adapted, pinned normal-integration/depth-prior control. All use the same original depth, normals, camera, support and disclosed compute. Metric3Dv2 is a conceptual prior, not a free additional pretrained-model comparison.

A cheap prospective falsifier could fit one offset on fixed spatial support and test normal consistency on disjoint support. Non-identifiability, unstable offsets, nonpositive corrected depths or no held-out consistency gain would reject this restricted model early. Passing is only internal compatibility: an independent, sealed depth reference is still needed to test whether it corrects physical shape rather than shared network bias. Neither consistency nor depth improvement alone establishes relative-pose gain.

For downstream pose, pre-shifting \(d\) followed by a solver with a free independent depth offset is an exact reparameterization of the same family. A meaningful test must expose a fixed/regularized offset prior and compare against strong free-affine RePoseD/MADPose and the above same-normal controls, at matched total cost. Numerical changes under the unchanged free-affine objective must not be relabeled new geometric information.

## Authored-camera control and cache limitation

Additional source-only check requested before deciding on an opportunity test:

- In the pinned `v2.py`, `self.forward(image, num_tokens=...)` precedes all uses of `fov_x`. Supplying FoV changes postprocessing only: fixed-focal point reprojection estimates the Z-shift, instead of jointly recovering focal and shift. One fresh raw forward could therefore support both postprocessing controls. Its tensors must be copied before the in-place Z-shift and masking operations.
- The existing `0023-mapfree/frontend.py` calls `infer` with `0022-frontend/qualify.py` options `num_tokens=1200`, `use_fp16=False`, `apply_mask=False`, `force_projection=True`, and no supplied FoV. It caches the returned outputs. The returned `points` are recomputed from depth and inferred intrinsics; the original learned lateral point coordinates have been discarded. Neither raw shift nor metric-scale head output is returned. The original raw point map cannot in general be recovered from this cache. Running point-based shift recovery on those already projected cached points is a different surrogate, not the official same-forward-output control.
- Consequently, comparing cached unknown-FoV depth with an authored-\(K\) normal correction would confound the new use of calibration with the use of normals. A supplied-FoV control is materially missing and must precede a claim about the normal restriction.
- Official `fov_x` is **not** arbitrary supplied-\(K\) support: recovery assumes centered optical axis and isometric image-plane coordinates; `infer` constructs normalized principal point \((0.5,0.5)\) and equal pixel focal lengths. Whether an actual authored camera exactly fits those assumptions requires later metadata admission, not assumed here. A general-\(K\) point reprojection recovery would be a separately documented adaptation. No authored calibration values or cached arrays were inspected.

These limitations do not require new pretrained weights, but they do prevent an exact known-FoV control using only the existing projected cache. This memo does not authorize the fresh forward or choose a new cohort.

## Local source identity

Pinned MoGe revision: `925b8ed835a7a9cdb7578ba15c658a0afc969030`.

- `moge/model/v2.py`: `e736ed59fdb8ad89ec0c29ddfeeccd6a4246c387b3f423b922708366b69758fd`
- `moge/utils/geometry_torch.py`: `2f8d5de7d671af16d25fe13c555fd73855d8447bc086bb23aa42e859b303fb05`
- `moge/utils/geometry_numpy.py`: `f540c5b8ef55bfffcc32b1a62b38be605683e29fbf399ca86dfcbfc458ce4c10`

Source read from the already acquired installation-05 tree. No new dependency or experiment is admitted by this memo.
