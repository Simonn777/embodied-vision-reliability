# Prospective question: signed, target-relative RGB detail

Source recommendation, 2026-10-20; **not an admission or execution specification**. No scientific arrays, labels, masks or outcome files were read. The fixed 0033 head remains closed. This recommendation incorporates the parent's later choice of all 228 noninitial TRAIN frames and 224 held-parent interior frames; the earlier 96-event subset is secondary only.

**One question:** can local, original-resolution RGB evidence distinguish target from background at a native boundary, with a learned *nonnegative reliability* controlling how strongly that evidence changes the native logit? The particular increment to test is local color relations in that reliability, beyond ordinary foreground/background appearance matching. This is a known-baseline/inductive-bias question, not a claim that prototype conditioning, Gram matrices or antisymmetry are new.

## Observation choice and the possible advantage

Choose original-resolution RGB, not a second 256-grid native-feature Gram campaign. RGB supplies the proposed readout with actual adjacent-pixel observations without another image encoder. Native 256-grid feature channels might already encode fine detail: their spatial dimensions alone do **not** prove information loss. An otherwise identical RGB-downsample/upsample control tests whether the original pixel lattice matters.

The causal possibility is limited: when a native contour is slightly displaced, the current pixel's color may resemble nearby deep target or background pixels more than its native classification suggests. Local color relations may distinguish a coherent transition from unhelpful texture. This cannot repair a missing object, a wrong target identity or a boundary outside the allowed band. Camouflage, illumination changes, mixed pixels and contaminated native supports can defeat it. Failure does not retrospectively diagnose absolute-channel bias in 0033.

## Fixed geometry, support and quantities

All coordinates are original-image pixel centers. Let native logit be \(\ell(p)\), \(P(p)=\sigma(\ell(p))\), and native binary mask \(M=[\ell>0]\), after the already established native presence handling. Use decoded sRGB values \(x(p)\in[0,1]^3\); **no per-pixel unit normalization or intensity removal**.

1. Define distance to the opposite native class using Euclidean distance between in-image pixel centers. The query band is \(B=\{p:d_{\rm opposite}(p)\le8\}\), on either side. Empty/full native masks have no eligible band and retain native output.
2. Reference pixels must lie outside this band, have the corresponding native class, and lie on the fixed image grid \(x\bmod4=y\bmod4=0\). For each query, consider only references within Chebyshev distance 32. Independently for foreground/background retain at most 32, ordered by squared spatial distance then row-major pixel index. Require at least four per class. Thus reference labels never come from the evaluated band, and the search examines at most a \(17\times17\) grid per query.
3. Query pixels require their center and four immediate axial neighbors to be in bounds. No coordinate clipping. Missing either reference class, invalid inputs, absent native target or unavailable stencil means exact native output for **every learned arm at that pixel**. Report these counts; do not remove their frames or errors.
4. All arms share these native-only supports, band, eligible pixels and sampling. The current GT is never a reference label, support selector, radius selector or confidence filter. Native support can still be wrong.

For \(c\in\{F,B\}\), using the identical capped set \(S_c(p)\), define
\[
\mu_c(p)=|S_c|^{-1}\sum_{q\in S_c}x(q),\quad
D_c^{P}(p)=\|x(p)-\mu_c(p)\|^2/3,\quad
D_c^{N}(p)=\min_{q\in S_c}\|x(p)-x(q)\|^2/3.
\]
Both distances are in \([0,1]\). The primary signed evidence is
\[
e(p)=\frac{D_B^N-D_F^N}{D_B^N+D_F^N+10^{-6}}.
\]
Positive means foreground-favoring; negative means background-favoring. The denominator floor is fixed in these RGB units, not fitted. Prototype distances remain a separate control, rather than assuming one mean captures a multimodal object.

For ordered offsets \((-1,0),(1,0),(0,-1),(0,1)\), let \(v_i=x(p+\delta_i)-x(p)\), and retain the ten upper-triangle entries
\[
G_{ij}=v_i^\top v_j/3,\quad i\le j.
\]
These preserve brightness differences. They are invariant to a *common additive RGB offset and orthogonal color-coordinate change*, not to image rotation, arbitrary illumination or independent changes across pixels. They do not identify target polarity. Moreover, they are algebraically equivalent to squared distances among the center and four neighbors; calling this a new relation operator would be misleading.

## One small correction and its controls

Use symmetric reliability inputs
\[
s_0=[1,\;4P(1-P),\;D_F^N+D_B^N,\;|D_F^N-D_B^N|],\qquad
s=[s_0,\operatorname{vech}(G)].
\]
The proposed primary is
\[
\ell'(p)=\ell(p)+4e(p)\,\sigma(\theta^\top s(p)).
\]
It has 14 coefficients, no spatial network and no temporal feedback. Swapping foreground/background reference sets leaves \(s\) fixed and reverses the correction; zero evidence produces exactly zero correction. The **native base logit is not swapped**. This is an operator check, not a statement that swapping reference labels describes a real video. The gate can attenuate but cannot reverse wrong evidence, which is a substantial restriction and possible reason to reject it.

For the ordinary learned controls below, use \(\ell'=\ell+4\tanh(w^\top z)\). Their shared native input is \(c=[1,\ell/8\text{ clipped to }[-1,1],4P(1-P)]\). These are pointwise affine readouts followed by the stated fixed nonlinearity, not copies or continuations of the closed 1377-parameter recipe.

| Arm | Inputs/operator | What it isolates |
|---|---|---|
| Native | No correction | Actual useful improvement |
| Logit-only | \(z=c\) | Recalibration without appearance |
| Absolute | \(c\), RGB at the five stencil pixels, and the 32 native fine-feature channels sampled at the center | Strong absolute-coordinate control with **both** available observation sources |
| Prototype | \(c,D_F^P,D_B^P\) | Ordinary two-prototype conditioning |
| Capped NN | \(c,D_F^N,D_B^N\) | Same-support FEELVOS-style matching cue |
| Ordinary relations | \(c,D_F^P,D_B^P,D_F^N,D_B^N,\operatorname{vech}(G)\) | Strong same-information unconstrained correction |
| Signed, no Gram | \(4e\,\sigma(\theta^\top s_0)\) | Whether the local-relation increment helps |
| **Signed + Gram** | Primary above | Fixed question |
| Signed + displaced Gram | Primary with the ten Gram entries cyclically shifted by half the row-major eligible-pixel count within the same frame | Same-dimensional local-association/capacity negative control |
| Signed + Gram, RGB256 | Primary recomputed from RGB bilinearly resized to \(256\times256\), then back to original H/W, align-corners false, no antialiasing | Whether original-lattice RGB contributes beyond a specified low-grid surrogate |

The displaced control uses shift \(\lfloor n/2\rfloor\); for \(n<2\), all learned arms keep native output. It preserves the frame's Gram marginal distribution, not every spatial correlation. Do not use held-parent results to choose another permutation. The RGB256 arm retains original native supports and coordinates; only its RGB observations change.

A cheap classical affinity control is also necessary before interpreting a positive result as more than color-guided unary smoothing: at each query, compute one normalized bilateral average of native \(P\) over its in-image \(7\times7\) neighborhood, excluding itself, with kernel
\[
k_{pq}=\exp[-\|p-q\|^2/(2\cdot2^2)-\|x(p)-x(q)\|^2/(2\cdot0.1^2)].
\]
Give \(c\) and \(\bar P_k-P\) to the same ordinary readout. This is explicitly a bounded bilateral-message control, **not** a faithful full DenseCRF reproduction. A larger efficacy claim would require the actual strong image-guided refiners; this screen does not substitute for them.

The absolute control has 50 coefficients and ordinary-relations control 17 versus the primary's 14; neither is weakened by fewer parameters. The displaced-Gram control exactly matches primary capacity. Zero padding would not establish equal effective capacity. Same RGB availability, sampling and labels matter more than an artificial exact parameter-count claim.

## Four-fold fitting and evaluation

Use the four existing TRAIN recordings as four groups. For each fold, fit on **all noninitial frames of the other three parents** and predict all noninitial frames of the fourth. Totals are 228 unique frame identities, not 228 independent observations. The primary metric uses the 224 official interior frames, averaged within each parent and then equally across four parents. Original 96 horizon-event results may be reported secondarily with duplicate identities explicit. No DEV use.

For fitting, take at most 1,024 eligible band pixels per frame by uniformly spaced ranks in row-major order, with round-half-up rank selection including endpoints. Selection precedes labels and is common to all arms; do not balance using GT. Official void pixels contribute no loss, with their count retained. Weight parents equally, then frames equally, then valid sampled pixels equally. A frame with no valid sample has zero data-loss contribution and remains in the frame denominator. Labels outside this shared sample do not enter fitting.

All folds use the same fixed binary cross-entropy on original-coordinate corrected logits, plus \(10^{-3}\) times the same weighted mean squared correction. Ordinary coefficients start at zero. Gate coefficients start at zero except intercept \(-4\), giving a small, explicitly nonzero initial gate; do not call this identical initial predictions. Use one deterministic float64 L-BFGS run per arm/fold, maxiter 80, maxfun 400, ftol \(10^{-12}\), gtol \(10^{-6}\); no restart, parameter search or best-checkpoint selection. Retain termination, gradient norm and last finite parameters. A numerical/optimizer failure is reported as a failed arm/fold with native fallback, not omitted or retried into success. Such failure cannot establish absence of information.

RGB quantities use the fixed natural scales above. Only the absolute 32 native channels need per-channel mean/std, estimated from that fold's three training parents, with population std floor \(10^{-6}\). No held-parent labels or feature statistics enter normalization, optimization, regularization or model selection.

Predict the **entire eligible band**, not just fitted sample locations; leave all other original pixels native. Threshold corrected logits at zero. All outputs are independent current-frame corrections; no corrected output becomes a support mask for another frame. Preserve frame-level J, F, J&F, changed-pixel count, coverage and fallback counts. Also report every parent's boundary precision/recall, so apparent F improvement is not mistaken for universal contour recovery.

## What would warrant more work

The proposed minimum useful screen result is at least **+0.5 absolute J&F points** in the equal-parent mean over the strongest pooled control, positive margins against ordinary-relations and signed-no-Gram in at least three of four parents, and no parent worse than native by more than 0.5 point. These are recommendations for the parent's later admission decision, not an adopted gate.

Require the primary to beat displaced-Gram; otherwise there is no evidence that its local association is useful. Compare RGB256 separately: a tie does not reject all target-relative information, but it defeats the proposed *subgrid-observation* explanation. If ordinary prototype/NN or bilateral correction matches the primary, report the known baseline's utility, not a special Gram/antisymmetry gain. A positive ordinary-relations result with a failed signed primary rejects the sign constraint, without authorizing a post-result rescue.

This uses no encoder forwards, downloads or new labels. Cache shared cue calculations and tile full-band prediction. Budget each fold within 4 CPU threads, 8 GiB and 1,800 seconds, with all four folds required; actual feasibility still needs a small artificial/source qualification. Do not reduce frames/controls after observing runtime or performance. Four exposed parents cannot establish generalization confidence intervals, two-source efficacy, or the cause of 0033's failure.

## Closest primary coverage and read limits

- **GrabCut, SIGGRAPH/TOG 2004**, [author paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2004/08/siggraph04-grabcut.pdf), §§2–3, Eqs. 3–11: foreground/background color likelihoods plus contrast-sensitive neighborhood consistency, with iterative estimation. A signed foreground/background appearance preference is a classical likelihood-ratio idea. Our normalized NN difference is not a calibrated Bayesian likelihood; adding a reliability gate does not by itself establish novelty.
- **DenseCRF, NIPS 2011**, [proceedings paper](https://proceedings.neurips.cc/paper_files/paper/2011/file/beda24c1e1b46055dff2c39c98fd6fc1-Paper.pdf), §§2–3, Eqs. 3–5: position/RGB affinity changes unary segmentation through mean-field messages. Thus local color relations repairing a coarse mask are already known. The proposed test concerns a constrained, target-conditioned transfer comparison, not a new affinity.
- **FEELVOS, CVPR 2019**, [author v2](https://arxiv.org/html/1902.09513v2), §3, Eqs. 1–4: object/background minimum embedding distances and previous probabilities guide a learned segmentation head. Our current-frame RGB, capped pseudo-label supports and pointwise gate are a restricted control, not an implementation of the learned temporal FEELVOS system.

Actual relevant method sections were read through web text, including browser-rendered primary PDF text. PerSAM §3.2–3.3 was also rechecked at its existing [author v2](https://arxiv.org/html/2305.03048v2); it reinforces the already recorded prototype-conditioning precedent but is not an additional proposed mechanism. No local PDF, source archive, model or data payload was acquired. Requests: two existing primary HTML opens; two official publication-page opens; two paper-link opens; one focused GrabCut section read. Web response-body sizes are unavailable. This limited check neither proves novelty nor vetoes a useful combination.
