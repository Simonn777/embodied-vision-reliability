# Admission review after 0011 — no new experiment admitted

October18,2026 New York. AI-assisted, bounded literature/method review, not an
exhaustive novelty search. This does not change any earlier protocol or allocate
another reserve group. No manuscript or positive result follows.

## Avoid repeating the same research mistake

0011's independent numerical replay agrees with its negative task-quality result.
A new proposal must identify information a strong comparator cannot already use,
or a concrete source of substantial end-to-end computation savings. A surrogate
quantity or successful implementation does not meet the owner's improvement goal.

Few-shot rotation-robust CLIP classification was considered and not admitted.
[Tip-Adapter](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136950487.pdf)
(ECCV2022, method section3/equations3–4 read by the independent reviewer) already
combines a labeled feature cache with CLIP scores. Rotating a sample supplies
another representation of the same observation. This review found no justified
new mechanism beyond familiar augmentation/cache operations.

Likewise, ordinary token selection is insufficient as a new method. The official
abstracts of [TCA](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Is_Less_More_Exploring_Token_Condensation_as_Training-free_Test-time_Adaptation_ICCV_2025_paper.html)
(ICCV2025) and [Patch Ranking](https://openaccess.thecvf.com/content/WACV2025/html/Wu_Patch_Ranking_Token_Pruning_as_Ranking_Prediction_for_Efficient_CLIP_WACV_2025_paper.html)
(WACV2025) were checked by the independent reviewer; their full methods were not
read in this pass. The parent's direct HTML opens failed, so no stronger reading
claim is made. Cached ViT-B/32 has49 image tokens at224px: shrinking attention
alone need not dominate total CPU latency. That is a cost argument, not a measured
speedup or proof that all efficient adaptation is exhausted.

## A question retained, but not an admitted method

Could delayed true labels repair later pseudo-label decisions affected by an
earlier error, while reproducing complete replay at materially lower cost?
The new information would be arriving labels, simulated from a disclosed
dataset annotation budget. It would not be zero-label adaptation, retroactive
improvement of already-issued predictions, or a new benchmark result.

[EATTA v1](https://arxiv.org/html/2503.14564v1), Wang and Ding, March18,2025,
uses feature perturbations to select annotations and balances supervised and
unsupervised gradients. Sections3.1–3.3 were inspected; its use of sparse feedback
already precedes this question. [Label Delay in Online Continual Learning v2](https://arxiv.org/html/2312.00923v2)
was read selectively by the independent reviewer in sections3–4 and Algorithm2:
delayed labels and feature-guided memory sampling are also established. Merely
combining them is not a contribution. No upstream code, weights or data from
these papers was run or acquired.

The main objections are substantive: a classifier using only arrived true labels
may already outperform pseudo-label repair; replay of small cached feature sets
may already be cheap; general incremental computation and dependency reuse are
known. A credible decision-stability argument would have to preserve complete
replay, including its worst-case full work, and charge certificate/selection
overhead. Constructing repeated artificial long streams to manufacture a speedup
would not establish visual task value.

**Decision: no immediate implementation or real-data allocation.** First resolve
whether there is useful correction beyond strong true-label-only and full-replay
baselines, and whether a specific mechanism differs from the closest prior art.
This is an unresolved research question, not a promised route to acceptance.
