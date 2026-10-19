# 0015 follow-up prior and identifiability screen

Status: source/math review only; no follow-up method admitted. No scientific images, HDF5 values, saved scientific predictions, or evaluation outputs were read for this note; no experiment ran. The original review preceded knowledge of the 0015 score. The coordinator subsequently reported that the sparse control failed (full AUC@5 gain about 0.00956 percentage points; paired-subset AUC@5 loss about 3.09105 points). Those figures are coordinator-reported context, not independently audited here. Neither propagation nor joint refinement is admitted by that outcome. This note does not change frozen 0015 code, protocol, or results.

## Verified sources and reading depth

| Primary source | Version/status checked | Material actually read and overlap |
|---|---|---|
| Henry Bradler, Matthias Ochs, Nolang Fanani, Rudolf Mester, [Joint Epipolar Tracking (JET): Simultaneous optimization of epipolar geometry and feature correspondences](https://arxiv.org/pdf/1703.05065) | WACV 2017, pp. 445–453; consulted arXiv:1703.05065v1, 15 March 2017. The PDF title page lists four authors; the arXiv abstract metadata omits Fanani. [Proceedings contents](https://www.proceedings.com/content/034/034540webtoc.pdf) confirm the four-author conference entry. | Full-text §§3.1–3.3, Eqs. 1–15. Eliminates patch displacement under epipolar constraints, optimizes five pose parameters using quadratic photometric summaries, and relinearizes while the actual image loss decreases. Includes a motion prior, which is not supplied by our isolated-pair assets. Generic joint patch/pose optimization without fixed arbitrary depths, and local-summary elimination, are already covered. |
| Philipp Lindenberger, Paul-Edouard Sarlin, Viktor Larsson, Marc Pollefeys, [Pixel-Perfect Structure-from-Motion with Featuremetric Refinement](https://arxiv.org/html/2108.08291v1) | ICCV 2021; arXiv:2108.08291v1, 18 August 2021. | Full-text §§4.3–4.4 and preceding drift paragraph. Featuremetric bundle adjustment jointly refines structure and cameras; Schur elimination reduces the camera system. Locally stored, precomputed feature-distance patches reduce repeated feature work. Sparse support, direct cost surfaces, and cost caching alone are therefore not new mechanisms. |
| Jérôme Revaud, Philippe Weinzaepfel, Zaid Harchaoui, Cordelia Schmid, [EpicFlow: Edge-Preserving Interpolation of Correspondences for Optical Flow](https://arxiv.org/pdf/1501.02565) | CVPR 2015; consulted arXiv:1501.02565v2, 19 May 2015. | Full-text §§3.2–3.4. Nadaraya–Watson and locally affine sparse-to-dense interpolation, edge-aware geodesic neighborhoods, and an efficient piecewise approximation. Applying these operators to correction vectors instead of absolute correspondences is a baseline adaptation, not an established new interpolation mechanism. |
| Yuxin Deng, Kaining Zhang, Linfeng Tang, Jiaqi Yang, Jiayi Ma, [ArgMatch: Adaptive Refinement Gathering for Efficient Dense Matching](https://openaccess.thecvf.com/content/ICCV2025/html/Deng_ArgMatch_Adaptive_Refinement_Gathering_for_Efficient_Dense_Matching_ICCV_2025_paper.html) | ICCV 2025, pp. 27369–27379; author/title/venue verified in the indexed CVF entry. | **Indexed primary-text only:** abstract, methods overview and related-work passages from the CVF paper; direct PDF retrieval failed. Separates residual-offset estimation, neighborhood-based match rectification and upsampling, using semantic, spatial and certainty information. This is a close recent overlap with transporting corrections through reliable neighborhoods. Equations/code have not been audited. |
| Andreas L. Teigen, Annette Stahl, Rudolf Mester, [Differential pose optimization in descriptor space — Combining Geometric and Photometric Methods for Motion Estimation](https://arxiv.org/html/2602.14297v1) | arXiv:2602.14297v1, 15 February 2026; **preprint**, no peer-reviewed acceptance verified. | Full-text §3.2 and abstract. Fits local descriptor-distance paraboloids and substitutes them into JET. The abstract reports no improvement over reprojection-error optimization. This is an additional direct precedent and caution, not a universal negative result. The HTML's anomalous journal-template label is not accepted as publication evidence. |

This is a targeted overlap screen, not an exhaustive novelty search. Combining established ingredients is allowed; the burden is to demonstrate a specific useful increment beyond these controls.

## Joint photometric geometry: what information could remain?

The following is a direct algebraic check, not a proposed theorem. For a fixed source query, let a target-coordinate likelihood be

\[
L_i(q)=\tfrac12(q-\mu_i)^T H_i(q-\mu_i),\qquad H_i\succ0,
\]

and let the epipolar line be \(a_i(\theta)^Tq+c_i(\theta)=0\), with \(a_i\ne0\). Constrained minimization gives

\[
\min_q L_i(q)=
\frac{(a_i^T\mu_i+c_i)^2}{2a_i^TH_i^{-1}a_i}.
\]

Thus joint optimization with a frozen quadratic likelihood is covariance-weighted geometric refinement. Local affine/radiometric nuisance parameters can first be eliminated by a Schur complement when the required blocks are nonsingular. Flat or aperture-limited patches must not be assigned fictitious information by an inverse or arbitrary regularization.

Possible additional information is the **nonquadratic shape or separated modes** lost by retaining one optimum and curvature. This is a conditional empirical question; repeated relinearization is itself already part of JET. The relevant control is independent patch fitting followed by geometry using complete local curvature, with identical support and nuisance treatment, not an isotropic-only point estimator.

Omitting explicit depth does not remove structure: position along an epipolar line is a latent structural degree of freedom. Little parallax, pure rotation, flat texture and repetitive patterns still limit identifiability. An epipolar-only cost is invariant under translation-sign reversal. Projecting matches onto the *initial* epipolar line merely enforces that initial geometry; it supplies no independent evidence that the pose is correct.

Sparse support can make direct refinement cheaper than refining every dense match. However, a cached one-dimensional line profile cannot be reused exactly after its epipolar line changes. Cached two-dimensional surfaces and their approximation error, construction time, interpolation and memory must be counted; PixSfM is a close efficiency control.

## Propagated corrections: information versus duplicated influence

Write a dense target prediction as \(q_i^D=q_i^*+b(p_i)+\eta_i\), where \(b\) is a hypothesized spatial bias. A photometric seed correction satisfies

\[
d_s=q_s^P-q_s^D=-b(p_s)+\epsilon_s-\eta_s+m_s,
\]

where \(m_s\) includes photometric-model bias. A single correction cannot separate these terms. Image edges need not be boundaries of *matcher error*, and smooth correction vectors do not establish that their propagation is causal or beneficial.

For a fixed interpolation operator \(W\), the added correction is \(\Delta=Wd\), with covariance \(WC_dW^T\) and rank at most twice the seed count. Propagated matches are not independent new measurements. When geometric Jacobians are nearly constant within a neighborhood, propagating a seed can largely resemble increasing that seed's influence. This is a limiting comparison, not an exact equivalence for arbitrary geometry or robust loss.

A future admissible diagnostic would reserve spatially separated, nonoverlapping patches **before** field fitting and test correction prediction there. Controls: zero, global constant/affine correction, ordinary local-affine or EpicFlow-style residual interpolation, shuffled seed locations, sparse correction only, and influence-matched seed weighting. Do not fit field hyperparameters to these 16 pose labels. A useful residual-field combination would have to outperform these controls and show that its spatial relation predicts new image-supported corrections, not just amplify accepted seeds.

## Minimal ambiguity falsification before a more complex method

No such diagnostic is authorized or implemented here. A separate prospective protocol could ask only whether the proposed patch likelihood contains **recoverable alternative localization modes**:

1. Fix a small query cohort and all failures/coverage before examining cost surfaces. Use the existing query coordinate convention, bounded displacement region and identical pixel support. A cheap first diagnostic may fix the initialized affine warp and profile gain/bias; it must be labeled a translation-profile diagnostic, not a complete affine-profile search. Grid spacing, mode separation and numerical tolerances must be frozen using artificial fixtures rather than the real pose labels.
2. Inspect the bounded two-dimensional profile, retain every qualifying local minimum, and compare it with its best local quadratic approximation. Artificial single-quadratic and flat-patch fixtures must yield no claimed multimodal benefit; repeated-texture fixtures with known correspondences check whether genuine separated modes are detected. Sampling cannot rule out modes narrower than its resolution.
3. Keep image evidence separate from geometry. A diagnostic-only, explicitly oracle check against the already available ground-truth epipolar geometry could ask whether any retained alternative offers a meaningful reduction in epipolar inconsistency. Failure of that necessary check, at adequate coverage and resolution, rejects this bounded ambiguity-rescue rationale. Passing does **not** establish the true correspondence: many wrong positions lie on the correct epipolar line, and oracle selection is not deployable.
4. Only if alternative modes exist, contain demonstrably recoverable information, and survive coverage accounting should a pose-level study be considered. It must compare JET-style refinement, covariance-complete quadratic compression and the proposed nonquadratic mechanism at matched image support and total computation. No result from that future study may be inferred from 0015's failed sparse control.

The immediate recommendation is to preserve the 0015 failure and stop its propagation/joint-rescue branches. The profile diagnostic is a possible separately admitted falsification study, not a new method or an instruction to run it.
