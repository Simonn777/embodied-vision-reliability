# 0011 covariance proposal: reject as a new-method core

2026-10-18. Bounded primary-source audit of [the prospective candidate](0011-covariance-candidate.md). Four targeted search queries; selective paper reading only. No assets, models, packages, experiments, repository edits or paid API. The parent reported 0010 NO_GO; its conditional common-peer/validation continuation remains stopped.

**Decision: the proposed anchored full-covariance update is already explicitly present in StatA's appendix. Do not advance it as an original 0011 method or run a new real-data study to establish its novelty.** Spherical initialization and a low-rank implementation do not supply an identified new principle.

## Decisive overlap

StatA Appendix C.2, **Eq. 49–51**, writes the full outer-product update, without a Diag operator:

    Sigma_k = beta_k T_k + (1-beta_k)
              [Sigma'_k + (mu'_k-mu_k)(mu'_k-mu_k)^T].

Its Eq. 50 defines T_k using full residual outer products. The candidate sets `Sigma'_k = s0 I` and `mu'_k = text_anchor_k`: that is this existing construction with a spherical anchor. This is **direct prior text**, stronger than deriving a similar formula independently. The paper's main method and practical code use diagonal covariances; that does not erase the published matrix formula. Hard-count beta and normalized means in the implementation differ from the paper's soft-count stationary derivation; therefore this is an overlap in construction, not a claim that the complete proposed program is byte-equivalent to the published solver. [StatA v1, Appendix C.2](https://arxiv.org/html/2501.03729v1#A3.SS2).

## Closest sources and actual reading depth

| Primary source, authors, version/status | Relevant evidence; reading scope |
|---|---|
| **Realistic Test-Time Adaptation of Vision-Language Models** — Maxime Zanella, Clément Fuchs, Christophe De Vleeschouwer, Ismail Ben Ayed. arXiv **2501.03729v1, 2025-01-07**; CVPR 2025, also identified by the already pinned author README. [Paper](https://arxiv.org/html/2501.03729v1) | Statistical Gaussian KL anchor; direct matrix formula above. Newly inspected §§4.1–4.2, Appendix C.2 and selected future-work text; not a fresh complete-paper read. |
| **Boosting Vision-Language Models with Transduction** — Maxime Zanella, Benoît Gérin, Ismail Ben Ayed. arXiv **2406.01837v1, 2024-06-03**; NeurIPS 2024. [Paper](https://arxiv.org/html/2406.01837v1) | §3 uses a **shared diagonal** covariance and explicitly motivates fewer parameters; §4.2 also compares isotropic restriction. It does not supply the candidate's classwise full update. Newly inspected these passages and Appendix D; earlier repository audit covered §§3.1–3.4. |
| **BayesTTA: Continual-Temporal Test-Time Adaptation for Vision-Language Models via Gaussian Discriminant Analysis** — Shuang Cui, Jinglin Xu, Yi Li, Xiongxin Tang, Jiangmeng Li, Jiahuan Zhou, Fanjiang Xu, Fuchun Sun, Hui Xiong. **2507.08607v1, 2025-07-11**; preprint, no accepted venue verified. [Version](https://arxiv.org/abs/2507.08607v1), [method](https://arxiv.org/html/2507.08607v1) | §§IV-C–F.1: CLIP, covariance-structure testing, pooled/full classwise outer-product updates (21–22), scaled-identity ridge shrinkage (23), quadratic form **and log determinant** (25). Thus full correlated class models plus shrinkage in CLIP TTA are already explicit. Temporal accumulation and encoder normalization adaptation differ from the proposed frozen batch procedure. Method sections selectively read; no code or full-paper review. |
| **Multi-modal Test-time Adaptation via Adaptive Probabilistic Gaussian Calibration** — Jinglin Xu, Yi Li, Chuxiong Sun, Xiao Xu, Jiangmeng Li, Fanjiang Xu. **2604.19093v1, 2026-04-21**; CVPR 2026, official indexed CVF entry pp.30268–30277. [v1](https://arxiv.org/html/2604.19093v1), [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Xu_Multi-modal_Test-time_Adaptation_via_Adaptive_Probabilistic_Gaussian_Calibration_CVPR_2026_paper.html) | §§3,4.1–4.2: class-specific d×d covariance via second moments, QDA likelihood, identity initialization and EMA. This is video/audio multimodal adaptation; **do not mislabel it as an identical frozen CLIP method**. Selected methods read in HTML; CVF search metadata verified venue, direct CVF page returned403. |

## Classical ingredients versus any remaining gap

Full Gaussian DA/QDA, covariance shrinkage, whitening/Mahalanobis distances and Woodbury/determinant identities are established tools, not a contribution created by their combination. Independently, the candidate has the form `aI+UU^T`, with `a=(1-beta)s0`; residual vectors and one anchor-offset vector give rank at most N+1. Orthogonal change of basis preserves its quadratic form/log determinant when all vectors and the scalar prior are transformed consistently. These are algebraic properties, **not evidence of classification gain or a new estimator**. Conversely, the learned feature axes need not be statistically arbitrary; rotation equivariance alone is not a task-quality objective.

The untested implementation question is whether this particular spherical-anchor/full-matrix StatA variant helps small frozen batches within the CPU limit. That is a legitimate existing-method ablation, not the requested originality gap. If revisited for a separate benchmarking purpose, spherical and pooled-shrinkage controls would still be mandatory. No such study is admitted here.

One concrete limitation comes from **TransCLIP Appendix D**: dependence on classwise text-embedding quality, including preference for frequent concepts. The authors discuss a labeled few-shot mitigation. **Our inference**, not their newly solved problem: under label-free deployment, separating semantic anchor error from real class-prevalence change remains important; an incorrect anchor can reinforce a wrong partition. This audit establishes neither a novel solution nor that this problem remains unaddressed elsewhere. It is not permission to relabel adaptive alpha, covariance, or LOO as that solution. No next candidate is admitted.

## Search and coverage boundary

Exactly four queries, restricted to primary paper domains:

1. `CLIP "test-time adaptation" "full covariance"`
2. `CLIP "covariance" "shrinkage" Gaussian 2025 2026`
3. `CLIP test time adaptation Gaussian discriminant covariance low rank 2025 2026`
4. `StatA TransCLIP Gaussian covariance 2025 2026`

Search also surfaced OGA, ADAPT and retrieval/GDA work. OGA's direct main-paper fetch failed; its formula was not admitted as verified evidence. No further searches or implementation downloads followed. Four selected papers above suffice for the specific rejection; this is not an exhaustive novelty certificate, benchmark reproduction, code-license approval, or claim that all covariance-based research is exhausted.
