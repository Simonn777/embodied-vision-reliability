# One conditional next hypothesis: covariance geometry, not query deletion

2026-10-18. Prospective critique only. I read the frozen 0010 protocol, pinned StatA solver, cached 0010 source/prior-art audits, and the published 0006/0008/0009 stop summaries. I did **not** inspect the ongoing 128-image results, access pixels/embeddings, fit anything, search the network, or change repository files. This memo neither authorizes a study nor changes 0010.

**Hypothesis:** if 0010 fails, one genuinely different mechanism worth a mathematical admission check is that StatA's coordinatewise covariance model mistakes correlated feature variation for class evidence. A strongly anchored, rotation-equivariant covariance likelihood could improve **classification**, even when removing query self-inclusion does not. Its usefulness and originality are UNKNOWN; the construction below is classical shrinkage, not an invented new theorem.

## Specific information gap

The pinned [StatA solver](https://github.com/MaxZanella/StatA/blob/2635f9355f9389756064becb5d9ed54c2756809c/solvers/StatA.py) estimates 512 diagonal variances per class from a 64-image batch. Jointly applying an orthogonal Q to images and text leaves zero-shot cosine logits, norms, and the neighbor graph unchanged, but generally changes its diagonal Gaussian likelihood. This is a property of the implemented model, **not proof that learned CLIP axes are useless or that rotation invariance improves accuracy**.

Analytical illustration: two zero-mean Gaussian classes with covariances [[1,.8],[.8,1]] and [[1,-.8],[-.8,1]] have identical diagonal approximations. At x=(1,1), their full quadratic forms are 10/9 and 10, with equal determinants. A 45-degree change of basis makes both full covariances diagonal and exposes the distinction. This establishes information discarded by diagonalization. It is not a unit-sphere StatA execution, and does not establish that unlabeled responsibilities can identify the correct classes.

## Exactly one proposed intervention

Keep text prompts, image encoder, graph, responsibility schedule, normalized mean update and alpha=1 unchanged. Replace only the covariance family, including its initial prior, by

    Sigma_k = beta_k S_k + (1-beta_k)(s0 I + delta_k delta_k^T),
    S_k = sum_i r_ik (x_i-mu_k)(x_i-mu_k)^T / sum_i r_ik,
    delta_k = text_anchor_k - mu_k.

Here s0 is the trace-per-dimension of the original initial covariance. Use the full quadratic form AND log determinant. This rule is jointly rotation-equivariant, assuming valid masses and positive s0. It preserves correlated directions instead of projecting features away. No adaptive rotation, new prompt, extra encoder pass, learned gate, or parameter sweep is proposed. Zero masses/degenerate priors remain explicit unsupported boundaries, never filtered successes.

An isotropic-plus-low-rank representation permits Woodbury/determinant identities: rank is at most N+2, so 512-dimensional dense inverses need not be computed. Feasibility under four CPUs/8 GiB/900 seconds is plausible algebraically, **unmeasured**. Admit it only after a single prospective artificial timing check; no shrink-to-fit rescue.

## What would make this informative rather than a renamed baseline?

Closest inspected methods are **StatA (CVPR 2025)** and **TransCLIP (NeurIPS 2024)**, which already adapt Gaussian feature distributions and graphs. Cached method/code audits establish this proximity; I did not newly read their full papers today. Full covariance estimation, isotropic shrinkage, and Woodbury computation are established techniques. Therefore “use feature covariance” or “make adaptation rotation-invariant” is not currently a defensible novelty claim. The exact full-covariance anchored construction may itself be an ordinary extension of StatA; absence from its current code proves no priority.

Mandatory cheap controls: unchanged zero-shot/StatA alpha1/alpha4; the same update with classwise **spherical** covariance; and ordinary **pooled full-covariance shrinkage**. If spherical or pooled covariance explains the gain, reject the claim that class-specific correlated structure is necessary. Merely reducing rotation sensitivity, likelihood loss, or numerical variance is not success.

Only a new, prospectively selected spatial-train development cohort could test quality, excluding every used 0010 filename. Cached assets suffice; no new acquisition is needed. Freeze one cohort and all controls before pixels/outcomes. A possible demanding screen retains +8 net correct/128 against the strongest fixed control and positive gains in both fixed batches, complete coverage, and the resource cap; this is a proposed development rule, not power or scene-independence evidence. Do not tune it using the current cohort.

**Kill conditions:** no gain beyond spherical/pooled controls; wrong pseudo-label correlations amplify harms; unsupported/resource cases; or failure of the fixed task-quality gate. Stop the exact candidate rather than tuning rank, alpha, basis or cohort. The 0006 failure shows a valid identity does not ensure useful representation; 0007→0008 shows a variance gain does not imply quality; 0009 shows a sophisticated score can have no headroom beyond cheap controls. A failed 0010 would reject that deletion screen, not prove all adaptation impossible.

**Recommendation:** retain this as one concrete, falsifiable architectural question, not an approved next experiment. No credible new-method/CVPR claim is established under the evidence currently read. If originality beyond classical covariance adaptation is required before proceeding, there is presently no admitted next candidate.
