# R2R baseline implementation check

2026-10-18. Source: Pang, Zheng, Quan, Ji, CVPR2021. Read [author-hosted manuscript](https://csyhquan.github.io/manuscript/21-cvpr-Recorrupted-to-Recorrupted%20Unsupervised%20Deep%20Learning%20for%20Image%20Denoising.pdf), §2, §3 Eqs.10–16, §4.1–4.2. [CVF original](https://openaccess.thecvf.com/content/CVPR2021/papers/Pang_Recorrupted-to-Recorrupted_Unsupervised_Deep_Learning_for_Image_Denoising_CVPR_2021_paper.pdf) indexed Tables4–5 also checked. Author copy was not hash-compared with proceedings bytes. No code/data/model execution; no direct large-file download.

**Verified paper settings:**

- General training: `y_plus=y+Aε`, `y_minus=y−Bε`, `ε~N(0,I)`, `ABᵀ=C`; minimize `||f(y_plus)−y_minus||²`.
- Eq16 inference is Monte Carlo averaging of `f(y+Aε)`. Use a normalized arithmetic mean: the text explicitly says averaging; do not implement an unnormalized sum copied from PDF text extraction.
- AWGN (§4.1): `D=αI`, **α=0.5**, **T=50** inference samples. Direct `f(y)` is the weaker Table4 ablation.
- SIDD (§4.2): **D=2I**, with NLF-derived raw-space pairs before rendering; explicitly uses direct `f(y)` because training spans varied noise levels.
- CC/PolyU single-image (§4.2): **A=20σI, B=σI/20**, per-channel estimated σ, 8,000 iterations. Thus α=0.5 is not a universal real-image/single-image default. The inspected paragraph does not specify a separate CC/PolyU T; do not invent one.
- R2R directly learns the clean-target denoiser through its negatively recorrupted target. **No Noisier2Noise output correction should be applied.**

## Implementation interpretation for the proposed controlled pilot

For fixed covariance `C=LLᵀ`, a convenient specialization is:

```text
u = L @ standard_normal()       # newly sampled, independent of original noise
input  = y + alpha*u
label  = y - u/alpha            # SAME u, opposite sign, reciprocal scale
loss   = MSE(f(input), label)
prediction = mean_t f(y + alpha*u_t)
```

This specialization is our implementation prescription for the controlled covariance setting. A Cholesky factor is sufficient; a symmetric square root is unnecessary because `ABᵀ=(αL)(L/α)ᵀ=C`. For normalized pixels, transform covariance in squared pixel units. Avoid clipping either noisy training member, which changes the assumed Gaussian relationship. Keep normalization fixed or analyze its target dependence. If spatial augmentations are used, transform C/noise consistently too.

**Why not correct twice (algebraic check):** define `w=y+αu`. The R2R label equals `(1+α⁻²)y−α⁻²w`. Therefore the optimum R2R predictor already equals `((1+α²)g(w)−w)/α²` when `g(w)=E[y|w]` is the corresponding noisy-target predictor. This is the relationship motivating the Noisier2Noise correction; adding it again to R2R would change the estimator.

For a minimal synthetic development comparison, fixing α=0.5 and reporting T=50 plus direct-input ablation is defensible as an explicitly adapted R2R control, not a reproduction of the PolyU single-image setup. Count all 50 forwards in deployed cost. A smaller network, training cap or changed alpha must be labeled as adaptations. Redrawing u does not turn one observed image into independent clean-scene samples. Do not report superiority over original full R2R from this pilot.
