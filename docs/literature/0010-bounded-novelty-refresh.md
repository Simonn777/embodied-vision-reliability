# 0010 bounded novelty refresh — 2026-10-18 (America/New_York)

**Outcome: no newly verified directly overlapping 2025–2026 paper was admitted.**
This is a narrow retrieval outcome, not evidence of novelty or a complete
literature search. The existing novelty assessment is unchanged.

## Exact scope and retrieval record

Read the complete existing `docs/literature/0010-self-inclusion-candidate.md` first.
Did not re-fetch StatA, TransCLIP, LGC_LVOf, TDA or SS-TPT. Exactly two new web
search queries were issued, restricted to arXiv, OpenReview, CVF Open Access,
PMLR and NeurIPS proceedings:

1. `CLIP "leave-one-out" "2025" OR "2026" adaptation`
2. `"test-time" ("cross-fitting" OR "self-influence" OR "leave-one-out" OR "query exclusion") CLIP`

The first retrieval was noisy. Its primary-source snippets mostly concerned
held-out *domains*, policy-gradient baselines or unrelated multimodal evaluation.
The second query returned no results. No third search, dataset/model acquisition,
experiment, API charge or repository edit was made.

A surfaced [OpenReview PDF](https://openreview.net/pdf?id=ACWBlCfcVY) describes
masking a pseudo-target domain's LoRA column during labeled meta-training and
holding out entire DomainNet domains for testing. That snippet does **not**
establish per-query deletion from a transductive unlabeled CLIP batch. It was
excluded rather than promoted to a new closest work. Its exact title, authors,
submission/revision date and final acceptance status were not verified; the
search snippet's “under review” label must not be converted into ICLR acceptance.
Other versions/snippet duplicates of that submission were not counted separately.

**New-paper reading depth:** search-result method snippets only. No new abstract
page, full method section, complete paper or official implementation was read.
Accordingly there are **zero new papers with fully verified title/authors/version/
date/venue metadata** in this update; no unverified citation is used to reject or
approve the candidate. The narrow lexical queries can miss differently named
jackknife, cross-prediction, out-of-fold or sample-decoupling methods and can miss
unindexed recent releases. Search dates are not paper publication dates.

## Falsifiable novelty boundary

The broad claim “remove a sample's influence on its own prediction” remains
unavailable: the already documented LGC_LVOf and SS-TPT precedents cover that
rationale in related graph/view settings. This refresh does not weaken that
prior-art boundary.

A direct novelty kill would be a verified method that fits the remaining
unlabeled batch while excluding the queried feature from **all** fitted moments,
shared initial covariance, responsibilities and graph updates, then applies a
one-way held-out CLIP/StatA-like readout. A full-sized-fold or jackknife formulation
that is operationally equivalent would also count; a new name or application
alone would not rescue novelty. This refresh found no verified new example of
that exact construction, and does not establish that none exists.

Even if a future exact-LOO screen improves classification, the current LOO versus
OTHER-out readouts use different peer pools. That would remain an algorithmic
headroom observation, not proof that correcting moment self-influence caused the
gain. Stronger anchoring may explain the gain without a new mechanism. A result
must first beat the fixed native α1/α4, zero-shot and OTHER-out controls under the
unchanged quality gate. No new method variant, tuning rule, threshold or manuscript
claim follows from this memo.
