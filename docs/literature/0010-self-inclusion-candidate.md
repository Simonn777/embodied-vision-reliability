# StatA query self-inclusion: bounded prospective audit

No experiment, assets, model calls, source edits or performance claim. Two search batches, followed by primary-source reading. Decision: **a computable self-reinforcement mechanism exists and a bounded application test is defensible; neither its novelty nor its benefit is established.** This is permitted transduction, not target-label leakage or automatically a software defect.

## Verified implementation and closest prior

* StatA fits soft-assignment means/covariances using the query batch, then predicts assignments for that batch. Its default anchor beta uses hard argmax cardinality; means normalize before and after anchoring. The graph intends to exclude the direct self neighbor but still permits indirect feedback; a later source audit found it discards the first top-k entry instead of masking the ID, so ties can retain self-edges. Read: complete author solver, initially mutable main, subsequently [pinned and qualified](0010-data-and-baseline-admission.md); not executed in this audit. [StatA code](https://raw.githubusercontent.com/MaxZanella/StatA/2635f9355f9389756064becb5d9ed54c2756809c/solvers/StatA.py).
* TransCLIP explicitly fits moments using all queries (Eq. 6–7), then returns their converged assignments. It therefore shares the relevant reuse; the read method does not implement query-specific deleted moments. Read: §§3.1–3.4 and implementation details; author repository identifies NeurIPS 2024 Spotlight. [Paper](https://arxiv.org/html/2406.01837v1), [repository](https://github.com/MaxZanella/transduction-for-vlms).
* TDA's official runner calls `update_cache` on the current query **before** `compute_cache_logits`. If admitted, its normalized feature has self-affinity 1, contributing the maximal positive-cache kernel value to its pseudo-label. Read: complete runner, especially lines 70–82; no execution. [CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Karmanov_Efficient_Test-Time_Adaptation_of_Vision-Language_Models_CVPR_2024_paper.pdf), [code](https://raw.githubusercontent.com/kdiAAA/TDA/main/tda_runner.py).
* **Exact conceptual precedent:** Afonso/Berton's LGC_LVOf sets the diagonal of the complete propagation operator P to zero, removing a label's influence on its own prediction while retaining its graph location. That removes multi-hop self-influence, not merely W's direct diagonal. Read: §4 and algorithm; arXiv v1 September 2020, page lists Pattern Recognition. This already covers the generic noisy-label/self-reinforcement rationale. [Primary paper](https://arxiv.org/html/2009.11811v1).
* **Recent VLM precedent:** Kim/Um's SS-TPT Eq. 7 uses a weighted leave-one-view-out reference, and its reference-selection ablation explicitly compares including versus excluding the view. It tunes prompts over augmentations, not per-query Gaussian moments. Read: §3.4–3.5 and the reference-selection ablation; arXiv v1 June 5, 2026, author metadata reports ICML 2026 acceptance. [Primary PDF](https://arxiv.org/pdf/2606.06943), [metadata](https://arxiv.org/abs/2606.06943).

Thus “leave out the query to avoid self-reinforcement” is already covered. The bounded search did not establish an exact prior for all of StatA's deleted normalized moments, covariance anchor and hard-count adjustments. Absence from this search is not novelty evidence.

## Exact cheap deletion, and what it cannot remove

With frozen responsibilities r_ik, accumulate R_k=sum r_ik, S_k=sum r_ik*x_i and elementwise Q_k=sum r_ik*x_i². For query i:

    R_-i,k = R_k-r_ik; S_-i,k=S_k-r_ik*x_i;
    Q_-i,k=Q_k-r_ik*x_i²;
    m_-i,k=S_-i,k/R_-i,k;
    n_-i,k=n_k-1[argmax(r_i)=k].

Normalize m_-i,k, apply beta_-i,k=n_-i,k/(n_-i,k+alpha), then normalize the anchor mixture, exactly as the chosen StatA mean update. If beta is zero, use the text anchor without dividing by an absent class mass. An empty/invalid resultant needs explicit anchor fallback. For the covariance, recompute residuals around the **deleted, anchored mean**:

    v_-i,k = Q_-i,k/R_-i,k - 2*mu_-i,k*S_-i,k/R_-i,k + mu_-i,k².

Apply the original covariance-anchor formula with deleted beta. A complete frozen-responsibility deletion must also remove i from the shared initial-covariance sum and denominator using its original initialization responsibilities; otherwise it retains another direct path. Entirely empty batches need a declared zero-shot readout, not an invented covariance. All these are sufficient-statistic identities, not new algorithms.

For an unanchored mean, leverage ell_ik=r_ik/R_k gives the exact identity

    x_i-m_k = (1-ell_ik)*(x_i-m_-i,k).

Its own squared distance is reduced by (1-ell_ik)². This is a concrete class-dependent self-support term. Normalization, anchoring and determinants prevent treating it as the full StatA logit correction.

Other responsibilities, fitted covariance and graph neighbors have already responded to i. Frozen-moment deletion leaves those indirect paths. A full refit must specify how a held-out query receives a **one-way readout** from the remaining graph without re-entering fitting. This differs from retaining its geometry but deleting only its pseudo-label influence, as in LGC_LVOf. Cross-fitting removes whole folds and also reduces useful batch support; it is not an exact isolation of self-inclusion.

An implicit fixed-point derivative (I-D_state Phi)^(-1) D_weight Phi can approximate smooth weight deletion only at a stable differentiable solution. Hard counts, kNN changes and singleton boundaries violate that simplification. A fixed-graph influence approximation must not claim to approximate graph deletion automatically.

## A two-dimensional classification win AND loss

Use unit vectors specified by angles. Query q has angle 0°; another point p has angle -30°. Text anchors are a_A=-120°, a_B=90°. Zero-shot cosine labels q as B and p as A. Freeze these hard responsibilities (the limiting soft case has the same strict inequalities), alpha=1.

The implemented normalized mean/anchor mixture gives mu_A=-75° and mu_B=45°. The query cosines are cos75°=(sqrt(6)-sqrt(2))/4≈0.258819 for A, and 1/sqrt(2)≈0.707107 for B. Inclusive prototype classification is **B**. Delete q's moments/cardinality: A stays -75°; B returns to its text anchor at 90°, whose query cosine is zero. Deleted classification is **A**.

This also gives an actual Gaussian-plus-text readout flip, not merely a distance change: disable graph messages, hold both diagonal covariances at 10^-4 I, keep the author's log-likelihood scaling 1/50 and text coefficient 1, and explicitly use text logits **100 times cosine**. The query's zero-shot log(A/B) is -50. Deleted log(A/B) becomes -50+cos75°/(50*10^-4)≈+1.764; inclusive is approximately -139.658. These covariances are a controlled mathematical fixture, **not** asserted to arise from StatA's default covariance initialization or converged solver.

* World 1: q's true class is A. Deletion corrects B→A.
* World 2: q's true class is B and it is the sole B example. Identical observable vectors/anchors/responsibilities; deletion corrupts B→A.

There is no unlabeled rule that can distinguish these two semantic worlds without additional assumptions. High leverage is therefore not evidence that the initial label is wrong. For a genuine singleton with an inaccurate text anchor, deleting its only visual support can simply destroy the opportunity to adapt.

## What a bounded development hypothesis would need

Hypothesis: in a predeclared small-batch setting with independently informative peers, direct self-support causes enough **wrong-to-right minus right-to-wrong** changes that deleted-statistic readout improves classification over unchanged StatA without unacceptable singleton/rare-class loss. Use frozen embeddings; no new encoder pass is required. This is a hypothesis, not authorization to acquire or run anything.

Mandatory controls: zero-shot and unchanged alpha=1 StatA; naive LOO mean-only with other terms fixed; full frozen-statistic deletion; actual full LOO refit on a tiny fixed batch; a fixed cross-fit partition; and one predeclared stronger-anchor control to test whether simply increasing regularization explains any gain. A predict-before-cache-insert TDA ablation is relevant supporting evidence, not a substitute for StatA tests.

Independent mathematical review added a necessary **leave-one-other-out** control: select a donor j distinct from i using a fixed derangement before outcomes; refit on N-1 samples exactly as in full LOO. This matches the support-size reduction and helps distinguish self-removal from deleting any sample. Full refits must recompute shared covariance initialization for every class, all remaining responsibilities and the neighbor graph. A frozen-direct deletion must not be described as doing that. Zero-mass classes bypass division and use the declared anchor fallback. For the first real-data headroom screen, prioritize unchanged StatA, zero-shot, exact LOO, matched other-out and a fixed stronger anchor; implement cheap approximations only if this mechanism first shows useful headroom.

Matching fit sizes is not complete causal isolation: absent-query LOO can read
from N-1 peers, while other-out retains the query among its N-1 fitted rows and
has N-2 other peers. Their native graph extensions therefore differ. An initial
quality screen can compare these algorithms but cannot attribute a difference
purely to moment self-influence. A later common-peer diagnostic would use the
same intersection D\\{i,j}, identical query-to-peer weights and respective fitted
responsibilities for both arms, with a full-fit/common-readout control if needed.
This diagnostic is prospective, not implemented or measured here.

Report top-1 accuracy and paired corrections/harms across all predetermined batches, with true occupancy strata used **only in evaluation**. Also compare cheap-deletion logits against exact-refit logits; agreement with refit alone is not task benefit. Stop this mechanism if exact refit has no quality headroom, if improvements are matched by stronger anchoring, if singleton harms erase gains, or if the approximation misses refit specifically at high leverage. A positive result would establish a useful narrowly scoped application; novelty still requires an explicit distinction from LGC_LVOf, SS-TPT and classical deletion/influence methods.
