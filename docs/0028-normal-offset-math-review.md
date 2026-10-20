# 0028: normal-based axial-depth offset — independent mathematical review

2026-10-20. Source and symbolic reasoning only. No model, scientific array, image, ground truth, implementation, fit, or experiment was run. This note establishes a conditional identifiable parameter and its failure modes; it does not establish novelty or useful accuracy.

**Conclusion:** a single global additive depth offset can be identified from reliable perspective surface normals and depth derivatives on sufficiently tilted surfaces. This is a one-parameter restriction of established depth–normal consistency, including the relation already implemented by pinned MoGe. Its plausible benefit is reduced estimation variance and preservation of the original depth shape, not new geometric information or a new tangency equation. Shared errors can make the correction uninformative or harmful.

## 1. Exact constraint, gauge, and estimator

Use axial depth, conventional calibrated pinhole intrinsics with last row `(0,0,1)`, and a smooth local depth field. For pixel coordinates `p=(x,y)`,

```
r = K^-1 (x,y,1)^T,   E = ∂r/∂p,   g = ∇d,
X(p) = s(d(p)+b)r(p),                 s>0.
```

The columns of `∂X/∂p` are `s[(d+b)E+r g^T]`. A camera-frame normal satisfies

```
(d+b) E^T n + (n^T r) g = 0.                         (1)
```

The common scale `s` cancels. Define `c=E^T n`, `α=n^T r`, and `q=d c+α g`. For fixed nonnegative weights and fixed observations, the ordinary tangency least-squares problem is

```
min_b Σ_i w_i ||q_i+b c_i||²,
I = Σ_i w_i ||c_i||²,
b_hat = −Σ_i w_i c_i^T q_i / I,      if I>0.           (2)
```

This is the unique unconstrained scalar least-squares minimizer, not automatically a statistically consistent estimate. Physical admission additionally needs `d+b_hat>0` on the prospectively required support. Raw `d` need not already be positive under an affine-depth hypothesis. If the optimum lies at or beyond zero depth, clipping to an arbitrary positive epsilon is not evidence of an identified physical correction.

The raw-depth gauge is consistent: under `d'=κd+η`, `g'=κg`, with `κ>0`, the same surface uses `b'=κb−η`, `s'=s/κ`. Fixed-weight equation (2) transforms accordingly. A reported numerical `b` has no gauge-independent physical meaning without its raw-depth normalization.

The parameter is conditional on known `K`. Intrinsics, coordinate-resize conventions, gradient units, and normals must describe the same camera frame. Image resizing or anisotropic coordinate scaling changes the two residual components; equal component weights are not coordinate-invariant unless the chosen normalization is transformed consistently.

## 2. Observability and exact counterexamples

- **One tilted smooth sample can constrain `b`.** For normalized image coordinates, `E^T n=(n_x,n_y)^T`. At least one retained nonzero transverse normal component is necessary and sufficient for the scalar coefficient in (2) to have rank one. Multiple samples supply redundancy, not absolute scale.
- **A tilted plane is not a degeneracy.** Let its true depth be `z=L/(n^T r)`, with `n^T r≠0`, and raw depth `d=z/s−b_*`. Then `g=−(L/s)c/(n^T r)²`, and substitution into (1) recovers `b_*` whenever `c≠0`. No metric plane distance is needed.
- **A frontoparallel plane is a degeneracy.** For `n=(0,0,±1)`, `c=0`; compatible constant depth has `g=0`, so every positive offset is equally consistent. If such a normal is paired with nonzero `g`, the residual is inconsistent but still contains no information about `b`.
- **Small normal error can induce collapse.** Let the true surface have constant raw depth `d=D>0`, so `g=0`, but let the predicted normal have any nonzero transverse component. Its least-squares solution is exactly `b_hat=−D`, regardless of how small the normal tilt is. All corrected depths collapse to zero. Thus a positive scalar denominator alone is insufficient for useful observability; near-frontoparallel support is unstable.
- **Grazing geometry is not benign information.** For `α=n^T r=0` and `c≠0`, equation (1) forces `d+b=0`. A smooth positive-depth surface expressed as this radial depth graph cannot satisfy that exact configuration. Near-grazing normals can make the implied log-depth gradient enormous; a coefficient test that sees only large `||c||` misses this inconsistency.
- **The two component equations need not agree.** When `α≠0` and `c≠0`, their component orthogonal to `c` requires `(I−cc^T/||c||²)g=0`. A scalar offset cannot repair a wrong gradient direction. The remaining residual is a model-mismatch diagnostic, not an extra free correction.

Normal sign is immaterial to (1)–(2): replacing any `n_i` by `−n_i` changes both `c_i` and `q_i` signs and leaves its squared objective unchanged. Normal magnitude is different: it changes the effective weight, so finite nonzero normals require a declared normalization. Interpolating oppositely signed normals can cancel before normalization; sign-invariance of the mathematical constraint does not repair such an interpolation failure.

A scalar matrix has condition number one when nonzero, which is misleading here. Sensitivity depends on the magnitude of `I` relative to actual observation error and coordinate units. Under a merely illustrative fixed-design independent residual-noise model with variance `σ²`, the unweighted offset variance is `σ²/I`. Real dense normal/depth errors need not satisfy that model, and spatially correlated pixels do not provide independent sample counts.

## 3. Errors in variables and the shared-error blind spot

Both the regressor `c` and response `q` contain the predicted normal. The gradient is derived from the same raw depth as `d`. Linearizing the tangency residual at a true offset `b_*`, with fixed `K`, gives

```
η ≈ c δd + α δg
     + [(d+b_*) E^T + g r^T] δn.                    (3)
```

The regressor perturbation is `δc=E^T δn`; it is correlated with `η`. The least-squares offset error has the form

```
b_hat−b_* = −Σ w(c+δc)^T η / Σ w||c+δc||²,
```

with higher-order observation terms included in `η` for the exact identity. Zero-mean input errors therefore do not generally imply an unbiased estimate. Ordinary residual weighting is not a likelihood or an uncertainty calibration without an explicit joint error model.

**Exact blind spot:** take any smooth positive but wrong depth surface `d`, and let the normals be its actual lifted-surface normals. Then `b=0` satisfies every tangency equation exactly even if the depth is wrong in a way no global affine correction can fix. A separate normal decoder does not exclude this correlated error. Equally, normals consistent with `d+b_bad` make that incorrect offset perfectly self-consistent. Neither training/held-out pixel consistency nor lower tangency residual alone establishes physical improvement.

Finite differencing adds a practical errors-in-variables issue at occlusions, thin structures, interpolation boundaries, and non-smooth surfaces. Large derivative residuals there cannot be interpreted purely as a global offset. Required support and failure accounting must be fixed before examining results; deleting pixels because their corrected depth or residual is inconvenient changes the test.

## 4. Relation to the existing MoGe helper

The pinned author implementation contains [`refine_depth_with_normal`](https://github.com/microsoft/MoGe/blob/925b8ed835a7a9cdb7578ba15c658a0afc969030/moge/utils/geometry_torch.py#L206), local lines 206–237. It computes the perspective relation

```
∇log z = −E^T n / (n^T r),                          (4)
```

then uses depth-dependent neighborhood weights, damping toward the original log depth, a clamped neighborhood update term, and iterative log-depth updates. Its defaults are ten iterations, damping `1e-3`, and a five-pixel kernel. This is an existing practical depth–normal refinement control, not an exact optimizer of every possible continuous integration objective.

Equation (1) is equation (4) restricted to `z∝d+b`, since `∇log(d+b)=g/(d+b)`. It is therefore a restriction of the same physical consistency model. Algebraic tangency least squares in (2) is not exactly the same noisy objective as log-gradient integration: multiplying a log-gradient residual by `α(d+b)` introduces a parameter-dependent residual scale. A comparison must disclose that loss difference, rather than attributing all differences to the number of parameters.

The pinned [`model/v2.py`](https://github.com/microsoft/MoGe/blob/925b8ed835a7a9cdb7578ba15c658a0afc969030/moge/model/v2.py#L243) returns a separately decoded normal field while adjusting points/depth through focal/shift recovery and metric scaling. Source search found the refinement helper's definition but no call to it from pinned default inference. Thus “already available in MoGe source” does not mean the cached default predictions have already undergone normal integration.

Source bytes read and hashed in this review:

- `installation-05/sources/moge/moge/utils/geometry_torch.py`: `2f8d5de7d671af16d25fe13c555fd73855d8447bc086bb23aa42e859b303fb05`.
- `installation-05/sources/moge/moge/model/v2.py`: `e736ed59fdb8ad89ec0c29ddfeeccd6a4246c387b3f423b922708366b69758fd`.

These paths are under private `work/0022-frontend-preparation`; only source text was read. This review does not independently establish literature priority for any restricted estimator.

## 5. Necessary angular control: remove the magnitude-collapse incentive

The collapse above is partly a defect of the unnormalized tangency objective, not an intrinsic observation of offset information. Let `w=d+b>0`, with `E_x,E_y` the columns of `E`. The surface-normal cross product is

```
T_x × T_y = w[(d+b)k+ℓ],
k = E_x × E_y,
ℓ = g_y(E_x × r) + g_x(r × E_y).
```

Since positive `w` does not change normal direction, define

```
v(b) = (T_x × T_y)/(d+b) = (d+b)k+ℓ,
u(b) = v(b)/||v(b)||.                                 (5)
```

The unnormalized vector is affine in `b`; its unit direction is not. It can be compared with a unit predicted normal using the sign-invariant angle `acos(|n^T u|)`. A smooth local alternative is the projective loss `½[1−(n^T u)²]`; its global penalty is different from half the squared angle and must not be silently substituted. Known orientation conventions could instead define an oriented comparison. The cross-product order must agree with that convention.

Writing `ρ=||v||`, the exact derivatives are

```
u'  = (I−u u^T)k / ρ,
β   = (u^T k)/ρ,
u'' = −2βu' − ||u'||²u.                               (6)
```

For `L=½[1−(n^T u)²]`,

```
L'' = −(n^T u')² − (n^T u)(n^T u'').
```

At an exact sign-equivalent normal match, both this loss and the half-squared angular loss have local Hessian `||u'||²`. Summing fixed-weight contributions gives genuine local angular sensitivity to `b`; away from a match the residual-curvature term matters, and a Gauss–Newton value is not the exact Hessian.

For normalized coordinates, the vector is especially transparent:

```
v(b) = (−g_x, −g_y, d+b+x g_x+y g_y)^T.
```

When `g=0`, `v=(d+b)k` and `u'=0` throughout the positive-depth domain. Its direction is constant even if the observed normal is tilted. The angular objective is flat, correctly reporting zero offset information. In contrast, the algebraic least-squares objective in (2) has positive curvature whenever the noisy target normal is tilted and artificially rewards shrinking tangent magnitudes to zero.

Thus an angular normalized-normal estimator is a necessary strong control against any claim based on the scalar algebraic estimator. It avoids that specific magnitude incentive and is invariant to a consistent nonsingular change of image differential coordinates, apart from orientation. This does not make it new or universally reliable. Nearly flat raw depth has weak angular sensitivity; very large positive offsets approach a frontoparallel normal; a mismatched target field can still have a best fit only at a positivity boundary or at infinity. Such boundary/unbounded behavior must remain visible. Global uniqueness or convexity of a multi-pixel angular objective does not follow from the scalar parameterization.

An angular Hessian and the algebraic denominator `I` quantify different things. In particular, normal noise can make `I>0` when the normal direction generated by the depth field is exactly independent of offset. Neither curvature value by itself is a calibrated uncertainty estimate under correlated normal/depth errors.

## 6. Plausible advantage, decisive failure, and downstream scope

A credible favorable regime is a depth field whose dominant error is one global axial offset, while its detailed shape is useful and its predicted normals contain sufficiently accurate, spatially distributed surface orientations. A one-parameter correction can pool those observations without allowing per-pixel shape drift. Unrestricted integration may propagate biased normal gradients or smear discontinuities. This is a conventional bias–variance and model-restriction argument; it is not a guarantee that the scalar method wins.

A credible unfavorable regime is spatially varying depth bias or coherent normal bias. The scalar correction cannot represent the first and may globally propagate the second. The frontoparallel collapse and self-consistent wrong-surface examples above already falsify universal reliability, without any numerical experiment.

A smallest future empirical falsifier should compare, on the same fixed images, calibration, normal/depth inputs and declared support: unchanged depth, scalar algebraic offset calibration, scalar angular-normal offset calibration, and the existing depth-anchored normal integration control. No new model inference is intrinsically required to expose normal outputs already cached. However, the cached default inference did not supply known field of view; a known-FoV MoGe depth baseline is an important separate control because the author pipeline already uses known focal information to recover a point-map axial shift. Its additional inference, if needed, must be charged and distinguished from cached reuse. The integration control must receive its actual authored coordinate convention and fair fixed settings, and its failures and cost must be retained. There is no need to claim a new normal estimator.

Physical evaluation must be independent of the normal-consistency fit. For a depth metric, use the same declared scale treatment across methods and do not perform a subsequent ground-truth affine alignment that removes the very offset being tested. Held-out tangency residual is useful only as an internal model check. If lower tangency error fails to improve independent physical error, or the simple integration control matches or beats the scalar correction, the proposed low-dimensional advantage is unsupported.

For pose, a global affine depth preprocessing followed by an unconstrained free-affine depth solver is an exact reparameterization of its attainable corrected-depth family. It adds no identifiable pose information by itself. To test normal-derived calibration as information, a downstream system must actually retain the estimated offset constraint or prior and still face a strong free-affine pose baseline. Solver conditioning changes alone must not be described as new geometric evidence.

**Admission conclusion:** the mathematics supports a cheap known-control calibration diagnostic only. Ordinary tangency, the closed-form scalar least-squares calculation, and default-vs-refined consistency are not sufficient novelty claims. Actual independent physical gain beyond the existing normal integration and free-affine pose controls remains entirely unestablished.
