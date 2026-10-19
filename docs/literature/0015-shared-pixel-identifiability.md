# Shared-pixel covariance: an analytical identifiability screen

This is a linear-model derivation and a hand-calculated example, not an experiment, a new theorem, or evidence of a new method. No images, correspondences, new ground truth, or estimators were accessed for this analysis. A photometric experiment is not admitted by this note.

## Fixed local fits are a compression of pixel observations

Let the underlying physical observations satisfy

\[
z=G\theta+\epsilon,\qquad \mathbb E\epsilon=0,\qquad
\operatorname{Cov}(\epsilon)=\Lambda\succ0.
\]

The design, covariance, and local fit linearization are fixed. For local patch fitting, a coordinate perturbation can have the form

\[
\delta u_i=L_i\epsilon,\qquad
L_i=-C_i(A_i^\top W_iA_i)^{-1}A_i^\top W_iB_i.
\]

Here \(A_i\) is the local photometric parameter Jacobian, \(B_i\) samples the same underlying noisy pixels, and \(C_i\) selects the reported coordinate components. Invertibility or an explicitly defined identifiable subspace is required. Shared pixels yield
\(\Sigma_{ij}=L_i\Lambda L_j^\top\); proximity alone does not specify this covariance. Estimated gradients, parameters, visibility, and match selection generally make this only a first-order approximation.

Stacking the fits gives \(y=Lz\), \(J=LG\), and \(\Sigma=L\Lambda L^\top\). Define

\[
A=L\Lambda^{1/2},\quad
P=A^\top(AA^\top)^\dagger A,\quad
q=\Lambda^{-1/2}(z-G\theta).
\]

The full-covariance GLS objective is exactly

\[
(y-J\theta)^\top\Sigma^\dagger(y-J\theta)=q^\top Pq=\|Pq\|^2.
\]

The direct, correctly whitened photometric objective is
\(\|q\|^2=\|Pq\|^2+\|(I-P)q\|^2\). Thus compression preserves all pose information, and the objectives differ only by a pose-independent term, exactly when

\[
(I-P)\Lambda^{-1/2}G=0.
\]

Otherwise the information lost is positive semidefinite:

\[
I_{\rm direct}-I_{\rm GLS}
=G^\top\Lambda^{-1/2}(I-P)\Lambda^{-1/2}G\succeq0.
\]

For Gaussian noise these are Fisher information matrices. More generally they describe linear unbiased least-squares information. This does not establish a nonlinear or misspecified-model guarantee. Singular \(\Sigma\) requires residual support consistency; it holds automatically here because \(y-J\theta=L(z-G\theta)\). A pseudoinverse alone must not silently discard inconsistent zero-variance constraints in a different model.

If brightness nuisance parameters have design \(N\), let \(T=\Lambda^{-1/2}N\) and \(Q=I-TT^\dagger\). When local fitting removes them exactly, \(LN=0\), the profiled direct information uses \(Q\), and equivalence requires \((Q-P)\Lambda^{-1/2}G=0\). Both arms must use the same nuisance model. Unique pixels do not automatically imply independent photometric residuals: interpolation and shared target-image samples still require their physical noise model. For a pose residual with a fixed coordinate Jacobian \(D\), replace \(L\) by \(DL\); relinearizing \(D\) is a further approximation requiring a specified profiling or marginal-likelihood interpretation.

## Which shared modes can improve point estimates?

For \(y=J\theta+b\eta+\varepsilon\), take \(\operatorname{Cov}(\varepsilon)=\sigma^2I\).

- Exact duplicate rows carry no additional information. Full GLS counts them once and can match a fixed deduplicated subset. Treating duplicates as independent can inflate confidence or distort relative weights.
- If the shared covariance is pose-aligned, \(\Sigma=\sigma^2I+JCJ^\top\), full GLS has the same point estimate as OLS for full-column-rank \(J\). Its covariance is \(\sigma^2(J^\top J)^{-1}+C\). The shared disturbance is indistinguishable from pose.
- If \(J^\top b=0\), that shared mode does not affect the OLS pose estimate either.
- Writing \(b=Jc+b_\perp\), an accuracy correction becomes possible when both components exist: the pose-orthogonal residual can reveal the realization of the disturbance that also contaminated the pose direction. Mere patch overlap does not establish this condition.

## Two-observation analytical example

Let \(J=(1,1)^\top\), \(b=(1,2)^\top\), \(\operatorname{Var}(\eta)=9\), and independent noise variance be one. Then

\[
\Sigma=\begin{pmatrix}10&18\\18&37\end{pmatrix}.
\]

| Estimator | Weights applied to the two observations | Actual variance |
|---|---|---:|
| Full GLS | \((19,-8)/11\) | \(46/11\approx4.182\) |
| Marginal-variance GLS | \((37,10)/47\) | \(30710/2209\approx13.902\) |
| Fixed first observation | \((1,0)\) | \(10\) |

The full estimator beats both marginal weighting and either single-observation subset. However, realize the same example with three physical observations by setting

\[
G=(1,1,0)^\top,\quad \Lambda=I,\quad
L=\begin{pmatrix}1&0&3\\0&1&6\end{pmatrix}.
\]

Then \(LG=J\) and \(LL^\top=\Sigma\), while the direct estimator has variance \(1/2\). The covariance-aware coordinate estimator still loses information through compression. This illustrative matrix construction is not evidence that real patch fitting produces a favorable mode.

## Bias and rejection conditions

Real corrected matches can satisfy \(y=J\theta+d+L\epsilon\), where \(d\) includes matching, visibility, warp, and illumination bias. For a linear estimator \(\hat\theta=Hy\) with \(HJ=I\),

\[
\mathbb E\|\hat\theta-\theta\|^2
=\operatorname{tr}(H\Sigma_{\rm true}H^\top)+\|Hd\|^2.
\]

Correct noise covariance does not control the bias term. Negative GLS weights can amplify it. If \(d=Jc\), its pose bias \(c\) cannot be distinguished from the actual pose by these observations. Injecting noise around already-biased matches does not demonstrate correction of their original errors.

A minimal prospective mechanism test can calculate expected errors directly from fixed matrices before simulating images: no overlap, exact duplicates, pose-aligned shared noise, a mixed-mode positive case, and fixed-bias stress. Require \(J=LG\), the same nuisance model, full and marginal covariance on the same coordinates, a predetermined nonoverlapping subset, and a correctly specified direct photometric control. Report variance and bias separately; do not tune the covariance to the test errors.

Reject the proposed accuracy direction if its gain is entirely duplicate counting that fixed thinning matches; shared disturbances are predominantly pose-indistinguishable; improvements require injected noise to dominate fixed matching bias; or the direct control matches or improves accuracy without a demonstrated computational or robustness advantage. Reject an apparent gain produced by inconsistent physical means, unmatched nuisance freedoms, or an incorrectly independent direct baseline. Even a successful analytical or artificial mechanism test is not novel-method evidence, a real-data improvement, or paper admission.
