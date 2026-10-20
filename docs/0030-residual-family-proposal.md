# Proposed restricted memory residual family

2026-10-20. Analytical proposal only; no execution, model/data access, or public edit. Actual assignment follows gradient qualification.

**Assessment:** this is a coherent 128-dimensional teacher/student family, substantially more constrained than a free 262,144-dimensional memory delta. It does not by itself establish useful information, learnability, or novelty. Its channelwise affine modulation is a known construction; the testable question remains whether future-task optimization of its coefficients transfers through a causal student better than matched direct training and denoising.

## Family, null directions and energy

For a fixed causal native memory \(m\in\mathbb R^{C\times N}\), with \(C=64\), \(N=4096\), let spatial averaging be \(\langle\cdot\rangle\). Compute and freeze

\[
\mu_c=\langle m_c\rangle,\quad
x_c=m_c-\mu_c,\quad
\sigma_c=\sqrt{\langle x_c^2\rangle}.
\]

The proposed correction is
\[
\delta_c=\gamma_c x_c+\beta_c\sigma_c .
\]

For \(\sigma_c>0\), the two basis vectors are orthogonal and have identical squared RMS \(\sigma_c^2\). Consequently
\[
\langle\delta_c^2\rangle=\sigma_c^2(\gamma_c^2+\beta_c^2),\qquad
\operatorname{RMS}(\delta)^2=\frac1C\sum_c\sigma_c^2(\gamma_c^2+\beta_c^2).
\]

Thus a common Euclidean bound or penalty on \((\gamma,\beta)\) is **not** a common absolute residual-energy bound. It instead controls correction relative to each channel's causal variation. Either choice is valid if stated; they are different hypotheses.

If \(\sigma_c=0\), then \(x_c=0\) and **both corrections are identically zero**. Neither coefficient is identifiable. Set both coefficients to zero, exclude them from optimization, and retain the channel unchanged. Do not add an epsilon floor to \(\sigma_c\) and silently introduce a constant-shift capability absent from the proposed family. There are at most 128 effective coefficients, not necessarily exactly 128.

## Recommended implementation coordinates

For active channels, define a frozen unit-RMS basis \(u_c=x_c/\sigma_c\) and energy coordinates
\[
a_c=\sigma_c\gamma_c,\qquad h_c=\sigma_c\beta_c,\qquad
\delta_c=a_cu_c+h_c .
\]

For inactive channels set \(u_c=a_c=h_c=0\). Optimize/predict \(q=(a,h)\) directly; reconstructing potentially huge \(\gamma,\beta\) in nearly constant channels is unnecessary. In exact arithmetic this is precisely the original family, and
\[
\operatorname{RMS}(\delta)^2=\|q\|_2^2/C .
\]

Energy coordinates permit a small absolute change in a low-variance channel; a bound on the original coefficients would restrict it more strongly. This is an explicit bound choice, not an algebraic equivalence of the two constrained problems. Pooling causal memory/pointer/confidence features is allowed, but supplies no future observation. Channelwise modulation cannot create an arbitrary spatial pattern: each channel retains its centered spatial pattern up to gain plus a constant.

One clean proposed common bound is \(\operatorname{RMS}(\delta)\le\rho s\), where \(s=\sqrt{C^{-1}\sum_c\sigma_c^2}\) is frozen causal variation and \(\rho\) is fixed before real targets. Equivalently \(\|q\|_2\le B=\sqrt C\,\rho s\). This relative scale is a proposal, not an already selected hyperparameter.

For \(B>0\), use the exact ball projection
\[
\Pi_B(q)=\frac{Bq}{\sqrt{\max(\sum_jq_j^2,B^2)}} .
\]

Mask inactive coordinates first. Put the lower bound **inside** the square root; computing \(\sqrt{\sum q^2}\) first and masking afterward can leave a zero-point backward singularity. At zero this formula has the identity derivative. For \(B=0\), return an explicit zero correction. Projection is nonsmooth at the boundary; use the same declared convention for teacher, direct-control and student. A teacher can alternatively project its coefficients outside autograd after each optimizer step; record which scheme is used.

The statistics and bases are detached from the teacher/head optimization. This removes a square-root derivative through \(\sigma\), not the need for finite numerical checks. Do not choose a variance cutoff after outcomes. Exact-zero handling is sufficient algebraically; any numerical near-zero guard must be declared prospectively and report lost active channels.

## Exact GT-memory denoising target

Let \(\Delta_c\) be the current-GT-memory minus native-memory target using the same encoder and declared native policy. On each active channel the unconstrained spatial least-squares coefficients are
\[
a_c^*=\langle u_c\Delta_c\rangle,\qquad h_c^*=\langle\Delta_c\rangle.
\]

Inactive channels remain zero. Then \(\Pi_B(q^*)\) is the **global constrained least-squares projection** under the proposed global energy ball: orthogonality leaves a constant residual term plus \(\sum_c[(a_c-a_c^*)^2+(h_c-h_c^*)^2]\). Per-channel balls would instead require separate projections. Applying the original coefficient-space Euclidean clip would solve a different constrained problem.

These are exact real-arithmetic identities. Finite precision makes measured basis means/RMS slightly imperfect. Verify the resulting actual tensor energy and report numerical deviations; do not claim bit-exact orthogonality. If exact numerical least squares is needed, use each measured two-basis Gram matrix and its corresponding energy metric consistently.

The bound above applies **before native BF16 storage**. The difference between quantized corrected memory and quantized native memory can exceed that pre-cast bound through rounding. Preserve the same native cast for all arms, verify zero-residual parity, and record post-cast energy separately rather than asserting an unproved quantized bound.

## Matched teacher and student

Optimize the future-loss teacher **within these same coefficient coordinates and bound from the beginning**. In general,
\[
P_{\mathcal F}(\arg\min_\delta L(\delta))
\ne\arg\min_{\delta\in\mathcal F}L(\delta).
\]

A free full-tensor teacher followed by projection would therefore introduce a target/representation mismatch and spend compute on inaccessible directions. The current-GT target's explicit least-squares projection is a defined control, not a claim to minimize future task loss.

Regressing energy-coordinate error with loss \(C^{-1}\|\widehat q-q^*\|^2\) equals spatial tensor MSE in this family; averaging over \(2C\) coefficients differs only by a factor of two. Record normalization so the one-step equivalence/direct-training comparison is not changed by a hidden learning-rate scale.

Restricting the teacher prevents arbitrary pixelwise latent targets but does not eliminate target nonuniqueness, future-label overfitting, or causal ambiguity. Multiple coefficient vectors can still be invisible to the selected future loss. Use the same fixed initialization, regularization and bounded optimizer across examples; selection remains based on complete causal development rollouts. This smaller family is a reasonable prototype, not a reason to claim FiLM itself is new or to demand another prerequisite accuracy study.
