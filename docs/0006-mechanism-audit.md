# Denoising: one defensible falsification screen, two rejected shortcuts

2026-10-18. Mathematical/design review of the supplied source audit and its R2R/ZS-N2N/CFC follow-up. No data, weights or papers downloaded; no scientific computation, fitting, repository edit or real-image inspection. **Conditional residualization is standard Gaussian conditioning, not a new theorem.** The exact target-centered application may merit a cheap falsification screen; absence from the inspected papers is not novelty evidence.

## 1. Reject scalar pair orthogonality as a CNN justification

Let `u_i=(m+d)^T y_patch`, `v_i=(m−d)^T y_patch`, with symmetric noise covariance C. Equal energy `d^T C d=m^T C m` gives `Cov(u_i,v_i)=0`. But the variances are `2m^TCm ± 2m^TCd`, **not both2m^TCm unless m^TCd=0**. If d annihilates degree≤2 polynomials, the views have equal clean signal on such patches. That signal is `m^Tx`, not necessarily the center value: an ordinary low-pass m blurs quadratic curvature.

Even satisfying all three properties does not make a neighborhood CNN valid. Explicit white-noise counterexample: set m=e0 and, at offsets `[−2,−1,0,1,2]`,

`d = [−1,2,0,−2,1]/sqrt(10)`.

Its zeroth/first/second moments vanish, norm is1 and center coefficient is0. Thus the two noises at the same center are independent Gaussian, each with variance2σ². Nevertheless,

`Cov(u_(i+1),v_i) = (0.4+4/sqrt(10))σ² ≈ 1.6649σ²`.

Their correlation is about0.8325. On a zero clean image, a predictor using only that adjacent transformed input learns a large nonzero noise prediction, although the clean-optimal output is0. This is a mathematical counterexample, not a measured experiment.

For a CNN, require the cross-covariance with **every input offset in its receptive field** to vanish. For unrestricted stationary convolutional fields the stronger condition is `A(ω)B*(ω)S_n(ω)=0` at every frequency, not merely its integral being0. With nonzero noise spectrum this forces disjoint spectral support; a d with zero low-order moments cannot match an identity-like m near DC. Finite receptive fields require only finitely many lag constraints, so this does not prove every possible finite-filter design impossible; it does kill the proposed scalar equal-energy justification. Smooth correlated noise also makes high-pass noise energy tiny, requiring large d scaling and amplifying texture mismatch. Do not proceed to a CNN real-data test of that unchanged construction.

## 2. Explicit minimal candidate: a full conditional residual vector

For center i let P contain the48 non-center offsets of a7×7 patch. Assume `y=x+n`, with fixed known C, zero-mean **joint Gaussian** additive noise independent of x. Define

`k_j = C(j,0)/C(0,0)` and `z_ij = y_(i+j) − k_j y_i`, for every j∈P.

Then `Cov(z_noise,j,n_i)=0` simultaneously for all48 coordinates. Joint Gaussianity gives independence of the **entire residual-input vector** from target noise n_i. Residual noise covariance is `C_PP − c c^T/C00`; the channels need not be mutually independent.

Use a per-center pointwise MLP `48 → 64 ReLU → 32 ReLU → 1`, with biases, receiving only z_i. There is **no subsequent spatial convolution over neighboring centers' residual vectors**, coordinate input, unprojected center skip, or other target-dependent side channel. Its population noisy-target identity, for a fixed predictor f, is

`E[(f(z_i)−y_i)^2] = E[(f(z_i)−x_i)^2] + C00`.

The center appears algebraically in z; that alone does not violate this identity because its noise has been jointly decorrelated from all inputs. Conversely, mixing residual vectors constructed for other centers would generally break it. This identity does not guarantee finite-sample generalization after fitting f to the same single noisy image; bounded capacity/steps and independent development scoring remain necessary.

**Choose no variance standardization in the first screen.** If introduced later, the correct diagonal variance is `Cjj−Cj0²/C00`, not Cjj; it is only a scaling, not whitening. Reject a degenerate channel rather than hiding a zero denominator with epsilon.

### Fixed noise family

Use grayscale clean values in[0,1], σ=25/255, and independent standard Gaussian innovations ε. Let

`g=[1,2,1]/sqrt(6)`, `h=g⊗g`, `n=σ(h*ε)`.

Here `||h||²=1`, so C00=σ². With `r_g(0)=1`, `r_g(±1)=2/3`, `r_g(±2)=1/6`, and0 otherwise,

`C(δx,δy)=σ² r_g(δx)r_g(δy)`.

Generate enough padded innovations and retain a valid image interior; do not clip noisy inputs/targets to[0,1], which would invalidate the Gaussian assumptions. Use only valid7×7 centers for every method and metric. Correlation support ends at offset2, so this patch includes genuine noise-independent far neighbors that can preserve DC. The largest non-center k is2/3. A white-noise identity fixture, k=0, must reduce exactly to ordinary blind-spot inputs; no separate learned advantage should be attributed to that case.

### Necessary negative cases

Common-mode covariance `C=σ²11^T` gives k_j=1. Every constant clean level disappears from z: all flat levels have identical residual input. No residual-only predictor can recover the level universally. Near-common-mode noise is ill-conditioned; do not infer universal real-camera validity from the finite-support kernel above. Edges and repeating textures are also essential: residualization preserves signal as `x_(i+j)−k_j x_i`, not an unchanged neighbor image. A correct risk identity cannot restore information the transform discarded.

## 3. Estimated C: precisely what cross-fitting cannot fix

With an independent but inaccurate estimate k_hat, the remaining target-noise covariance is

`Cov(z_hat_noise,j,n_i)=C(j,0)−k_hat_j C00`.

For fixed differentiable f, Gaussian integration by parts gives the loss bias

`L_hat = R_hat + C00 − 2 sum_j [C(j,0)−k_hat_j C00] E[∂f/∂z_j]`.

A covariance estimate from the same target neighborhood introduces additional dependence; this simple formula need not apply. Spatial cross-fitting with sufficient noise/receptive-field guards can remove that reuse under a genuinely finite-range model. It **does not remove signal/texture contamination** from the estimate. High-order differences annihilate certain polynomials, not edges or oscillatory textures; their squared moments contain an additional nonnegative clean-signal term that need not have the same orientation/spectrum as the noise.

For R2R with independently fixed estimated covariance and pair `u=y+Az`, `v=y−Bz`, `AB^T=C_hat`, the analogous exact relation is

`E||f(u)−v||² = R_u + tr(C+BB^T) − 2 E tr[(C−C_hat)J_f(u)]`.

Thus merely overestimating C is not uniformly safe: the Jacobian need not be positive semidefinite. Scalar illustration for a symmetric R2R pair, with clean variance τ², actual noise σ² and assumed added variance s: a linear pair-loss optimizer has gain `(τ²+σ²−s)/(τ²+σ²+s)`, while the supervised gain for u is `τ²/(τ²+σ²+s)`. Underestimating s rewards noise copying; contamination `s=τ²+σ²` collapses the pair-loss gain to0 and erases signal. Cross-fitting leaves this bias intact.

One can bound the directional bias with a low-dimensional covariance uncertainty set, but an honest set must include texture bias, not only sampling uncertainty. This is robust Stein-risk reasoning, not new Gaussian theory. Moreover, observations alone cannot distinguish `x=0,n~N(0,C)` from independent `x~N(0,C/2),n~N(0,C/2)`: both produce exactly the same y distribution but have different optimal denoisers. No universal single-image covariance estimator resolves that ambiguity without signal/noise assumptions. **Do not add a large uncertainty framework before the known-C information test passes.**

## 4. Smallest executable development decision

First run only artificial mathematical/transform fixtures: exact full-vector covariance, white-noise identity, polynomial/constant responses, degenerate common mode and the scalar-pair counterexample. No result from these fixtures is an accuracy result.

Then freeze four64×64 toy canvases: multiple broad flat levels; affine/curved ramps; sharp edges; two fixed repeated-texture frequencies. Two independent noise seeds per canvas, one fixed training-initialization rule. Compare:

1. Ordinary blind-spot MLP on48 raw neighbors, noisy center target.
2. Conditional-residual MLP above, same target and widths.
3. Known-C R2R control, same widths but49 input pixels including center, fresh independent Gaussian field η~N(0,C), input y+0.5η and target y−2η. Cross-covariance is0; input noise variance is1.25C and target noise variance5C. Its extra input weights and noise-generation cost are reported.

No clean targets enter training, checkpoint choice or selection. Use500 fixed Adam updates, learning rate0.001, batch256 valid centers, final checkpoint only; maximum90 seconds per fit,4CPU threads and8GiB. This is a prospective budget, not a measured runtime guarantee. A timeout/NaN is a failed run, not permission to reduce steps after observing scores. Score unclipped predictions on identical valid centers. For the primary R2R control, use50 fresh draws and average f(y+0.5η), charging every draw. Root subsequently verified this against the main paper Eq.16 and its α=0.5,T=50 setting. Direct f(y) is secondary only; using it as the main baseline would weaken the comparison. Matching this construction still does not claim reproduction of the paper's full trained architecture.

This is24 fits, at most36 minutes of fit time. Preserve both seed results. **Kill this fixed candidate** unless it achieves ≥0.25dB mean PSNR gain over the stronger aggregate control, improves at least3/4 canvas families, loses no more than0.2dB on any family, and does not worsen repeated-texture relative-amplitude error by more than0.05 versus that control. These are development continuation heuristics, not confidence intervals or a paper-level result. Do not replace an unfavorable kernel, texture, seed or architecture after the screen.

Only if it passes: use four metadata-selected natural parent images, permanently allocated to development, with the same frozen synthetic noise/seed/model procedure and one fixed128×128 crop each. Require ≥0.2dB mean gain over the stronger control, gains on≥3/4 parents and no>0.2dB parent harm. This may justify **one** predeclared real-camera pilot with an explicitly frozen non-oracle covariance estimator and all estimator failures reported. It does not carry the known-C guarantee into unknown, signal-dependent, non-Gaussian processed camera noise. Failure at either stage stops this candidate. Passing still requires a stronger contemporary correlated-noise baseline and genuinely separate scene/camera-held-out evidence before any claim of a useful new method.

### Parent execution decision after this memo

The parent chose to **skip toy-model training**, retaining only analytical/artificial unit counterexamples, and to screen four metadata-selected PolyU reference images with newly generated known-C Gaussian corruption. The four selected groups become permanent development data. This is synthetic-corruption development on natural reference images, **not native real-camera-noise denoising**. The toy-training schedule and its continuation thresholds above remain an unadopted design suggestion, not an executed result or the current frozen protocol. No training or results are claimed here; the parent will freeze the actual architecture, exact allocation, R2R50-draw baseline and gates separately.
