# 0028: two normal pairs, two point pairs, and affine monocular depths

Source-only screen, 2026-10-20. Exactly three targeted search queries were used; only primary papers and already installed author source were read. No scientific arrays, labels, models, solver calls, implementation, or payload acquisition. The fixed 0027 gradient proposal remains closed.

## Decision

**Do not claim a new two-point relative-pose solver.** The rotation and translation-direction construction is already explicit in Cui et al., ICCV 2019. Moreover, the proposed three-unknown affine-depth solve, when nonsingular and physically accepted, produces the **same translation direction** as that depth-free construction for the same estimated rotation and measured pixels. This equivalence also applies to noisy inputs after fixing their normal-derived rotation; it is not limited to a noiseless truth configuration.

There is a conditional empirical question: whether directly predicted normals provide useful image-derived orientation information, and whether affine-depth compatibility rejects bad normal/point hypotheses without losing good ones. Neither replacing polarimetric normals with learned normals nor appending the calibration solve establishes novelty. A gain would have to survive the same-normal depth-free control, normal-aware refinement, and current strong point/depth pipelines. No experiment is admitted by this note.

## Closest primary sources and reading scope

| Source and status | Sections actually checked; consequence |
|---|---|
| Cui et al., **Polarimetric Relative Pose Estimation**, ICCV 2019; [author PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2019/09/cui2019polarimetric.pdf) | PDF pp. 3–4, §§4.1–4.3, Eqs. 6–10 and 14–17. Two corresponding normal pairs determine a proper-SVD rotation; two point epipolar equations then determine translation direction. The paper also combines normal alignment with Sampson error in local optimization. Its normals come from polarization and have explicit ambiguities. It does not estimate independent affine depth corrections. This is the decisive pose-generation and normal-refinement baseline. |
| Choi et al., **Voting-Based Pose Estimation for Robotic Assembly Using a 3D Sensor**, ICRA 2012; [author technical report](https://merl.com/publications/docs/TR2012-032.pdf) | §§III-A.1 and III-C, Eqs. 1–2 and 9. Oriented point-pair registration and pose voting are established; the paper explicitly attributes the surface-pair construction to Drost et al., CVPR 2010. These are measured 3D/model points, not independently affine-ambiguous monocular depths. Drost's original full text was not independently read in this bounded screen. |
| Eichhardt and Barath, **Relative Pose from Deep Learned Depth and a Single Affine Correspondence**, ECCV 2020; [paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570613.pdf) | §§2–3, especially Eqs. 7–11; §4's surface-normal-based manufactured construction, Eqs. 15–16. It aligns lifted tangents from one AC plus depth/gradients, with relative scale. It does not use two separately predicted normal pairs with independent depth offsets. A normal is only the direction perpendicular to a tangent plane; it is not the complete two-column tangent mapping supplied by an AC. |
| Yu et al., **Relative Pose Estimation through Affine Corrections of Monocular Depth Priors**, CVPR 2025; [author full text](https://arxiv.org/html/2501.05446v3) | §§3–4.4 and Appendix C. Three point/depth pairs solve the unknown rotation plus relative scale and two shifts; Eq. 5 eliminates pose through distances. Hybrid sampling, scoring and local optimization already combine depth and point geometry. The checked formulation contains no observed surface normals. |
| Ding et al., **RePoseD: Efficient Relative Pose Estimation With Known Depth Information**, ICCV 2025; [published paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Ding_RePoseD_Efficient_Relative_Pose_Estimation_With_Known_Depth_Information_ICCV_2025_paper.pdf), [precursor full text](https://arxiv.org/html/2501.07742v1) | Re-read precursor §2, Eqs. 2–4, and §3.1. The precursor title is **Fixing the Scale and Shift in Monocular Depth For Camera Pose Estimation**, not the final paper title. Its calibrated affine-depth equation becomes linear once rotation is supplied. The novel claim cannot be the substitution of a normal-derived rotation into this known equation. |
| MoGe2 pinned author implementation, revision `925b8ed835a7a9cdb7578ba15c658a0afc969030`; [model/v2.py](https://github.com/microsoft/MoGe/blob/925b8ed835a7a9cdb7578ba15c658a0afc969030/moge/model/v2.py) | Read local source lines 50–55, 150–186, 248–296. `normal_head` and `points_head` are separate decoders of shared features. Normal output is interpolated and normalized; `force_projection` recomputes points from depth/intrinsics, not normals from depth derivatives. Thus the normal head can contain information not recoverable from the saved scalar depth samples. It is not a statistically independent sensor, nor a guaranteed geometrically consistent field. |

The bounded current search did not establish a dedicated learned-normal-plus-affine-depth publication covering the complete proposed input tuple. That absence is not a priority claim, especially because the pose part is already directly covered by 2019 and the remaining calibration is a fixed-rotation linear specialization.

## Exact reduction and why depth does not add a two-point pose direction

Let `r_ij = K_i^-1 [u_ij,v_ij,1]^T`, with axial ray coordinate three equal to one, and fix the first raw-depth scale:

```
X_j = (d_1j+b) r_1j
Y_j = (a d_2j+c) r_2j = R X_j+t,       a>0, j=1,2.
```

For oriented unit normals, require `n_2j=R n_1j`. Two nonparallel pairs determine one proper rotation when their pairwise inner products agree. With noise, use the established proper-SVD normal alignment; exact consistency is generally impossible.

Define `Δr_i=r_i2-r_i1` and `Δp_i=d_i2 r_i2-d_i1 r_i1`. Subtracting the two point equations gives

```
[ Δp_2   -R Δr_1   Δr_2 ] [a,b,c]^T = R Δp_1,
t = (a d_21+c)r_21 - R((d_11+b)r_11).
```

This is a 3×3 linear calibration system, not a new nonlinear minimal problem. Nonparallel normals do **not** guarantee that this separate matrix is nonsingular. Finite coefficients, positive `a`, positive corrected depths and nonzero translation must be checked; a singular calibration can have a pose-identifiable family, so a calibration failure must not be relabeled universal pose ambiguity.

For any solution, take `e_j = R r_1j × r_2j`. The point equation immediately implies `e_j^T t=0`. If the two epipolar normals are independent, therefore

```
t / ||t|| = ± (e_1 × e_2) / ||e_1 × e_2||.
```

This is exactly the direction in Cui et al.'s Eq. 10. It does not contain a depth value. Consequently the full-rank affine-depth solve cannot denoise translation direction while holding this rotation and these two measured matches fixed. It supplies a depth-gauge translation magnitude and calibration, and can reject a hypothesis or resolve its physical sign. The depth-free control can also triangulate both sign choices and enforce ordinary cheirality; do not artificially deny it that standard operation.

A nonminimal fit or joint rotation update may use depth to change pose, but that is a different estimator. It must be compared with the same normal-supported point fit and existing joint affine-depth refinement, not attributed to this two-point reduction.

## Normal/depth compatibility and fatal configurations

Normals must refer to the same physical surface points and to each camera's actual 3D frame, with a common orientation convention. Unit normal length is not enough. Camera handedness, display RGB encodings, world/camera conversion, back-facing conventions, and unoriented `±n` outputs must be resolved prospectively without labels. Enumerating sign alternatives is an explicit hypothesis cost. Near-parallel normals amplify rotation error; near-parallel epipolar planes amplify translation-direction error even if normal alignment is excellent.

Depth-derived normals cannot be treated as independent physical normal observations under an unknown axial shift. For a local depth field, its lifted tangent columns are

```
U_x=(d+b)J_x+r ∂_x d,   U_y=(d+b)J_y+r ∂_y d,
n(b) ∝ U_x × U_y.
```

The additive shift changes the normal. Computing normals from uncorrected depth and fixing their rotation silently assumes away this dependence. Positive global depth scale preserves normals, but axial offset generally does not. Range versus axial depth likewise requires the ray-length and derivative conversion.

A direct learned normal head avoids literal finite differencing, which is a meaningful information distinction from closed 0027. It may still share systematic scene, camera and boundary errors with its depth head. The checked MoGe source enforces neither cross-view normal consistency nor equality with the corrected depth surface; separate heads alone prove neither independence nor accuracy.

## Remaining conditional question and strongest controls

The only defensible next question here is **whether the already available direct normal predictions help pose on fixed matches, and whether depth compatibility adds useful rejection beyond normal geometry**. A new solver implementation is unnecessary to discover the algebraic equivalence above.

Before any efficacy claim, compare on identical matches/normals and charged inference:

1. The 2019 proper-normal-alignment plus two-point epipolar direction, both cheirality signs, and its normal-plus-Sampson refinement.
2. The exact same hypotheses with affine-depth calibration/admission, so any difference is identified as acceptance/selection rather than a new pose direction.
3. Nonminimal normal alignment/point geometry and joint normal/point/depth fitting with declared common support and budget; simple extra residuals are not themselves novel.
4. Current strong five-point/PoseLib and author-loss RePoseD, plus a trustworthy MADPose/hybrid control. Normal extraction and all failed branches count toward cost.

A smallest future falsifier would separate normal-angle diversity, epipolar-plane diversity, sign errors and correlated normal bias, with full failure-inclusive pose outcomes. If the normal-guided depth-free control already produces the same useful hypotheses and depth admission has no precision/recall benefit, close this candidate. If direct learned normals help, the result is first an empirical known-control opportunity; a distinct research contribution still needs a specific mechanism and reproducible gains beyond these controls. No thresholds, new data allocation or experiment are created by this screen.

Local author source read: `work/0022-frontend-preparation/installation-05/sources/moge/moge/model/v2.py`, SHA256 `e736ed59fdb8ad89ec0c29ddfeeccd6a4246c387b3f423b922708366b69758fd`. Source text only; no checkpoint, prediction cache or array was opened.
