# Post-0029 bounded successor review: joint object ownership is not an admitted method

2026-10-20. Source-only review. **Decision: NO_GO for implementation of the one mechanism considered below.** It changes the information contract relative to 0029, but the available formulation reduces to established multi-object identity/assignment methods. No new mechanism with a defensible advantage over those controls survived this screen. This is not a claim that useful successors are impossible.

The parent reported that attempt02 and its independent audit completed, with inadequate D1/D3 oracle headroom and a failed signed selector. I have not opened those prediction arrays, annotations, or evaluation files for this review. The fixed instantaneous-deletion variant stays closed; this note does not change its ages, thresholds, cohort, or interpretation. The innovative +3 J&F on two independent sources objective remains unmet.

## The single hypothesis screened

**Use the independently initialized identities of other visible objects to resolve component ownership jointly**, rather than treating every non-target pixel as undifferentiated background. In embodied navigation, confusing two pedestrians or manipulated objects is consequential even if their union is segmented well.

For object identity \(k\) and candidate visible component \(j\), let \(c_{kj}\) be a causal appearance/correspondence/trajectory cost. Assign components with binary \(x_{kj}\), allowing an explicit absent state, subject to each visible component having at most one owner. A candidate implementation would use object-indexed tracks from the supplied initial masks, then solve a joint assignment instead of independent maximum-score selection.

The additional observation is **the other objects' supplied identities and their separately observed tracks**. Mutual exclusion alone is a constraint, not an observation. There is a potential advantage over 0029's one-object initialization: “belongs to a particular other object” is more specific than “not our target.” There is no information advantage over a baseline receiving those same identities, images, tracks, and candidate components. Giving a proposed method all initial object labels while its baseline gets one binary mask would be an information-unfair comparison.

## Primary-source overlap actually checked

| Primary source and inspected location | Material overlap and limit of the comparison |
|---|---|
| [SENTRY: SAM2-Enhanced Neighbor-Aware and Temporally Reasoned Memory for Visual Tracking](https://arxiv.org/html/2606.24449v2), §3.2, “Neighbor Reasoning and Matching,” Eq. 4 and Algorithm 1 | Builds target/neighbor tracklets, backward-checks candidates, and uses one-to-one Hungarian matching as a temporal consistency filter. Neighbor trajectories supply negative identity context. This directly blocks a novelty claim based on merely adding competitors, correspondence-based candidate verification, or assignment before updating memory. Its neighbor identities need not be externally initialized; that distinction changes supervision, not automatically the algorithmic contribution. |
| [SAM-MT: Real-Time Interactive Multi-Target Video Segmentation](https://arxiv.org/html/2607.08688v1), §§3.2–3.7, Eqs. 2–8; §4.2, Eq. 9 | Explicit user-associated target queries, shared context, identity-specific temporal memories, and an overlap penalty already address joint segmentation without identity interference. This July 2026 author preprint is a closer task-level prior than single-target SAM2. Its trained architecture differs from a frozen-model wrapper, but “training-free” alone does not establish a new information mechanism. No checkpoint/runtime was acquired or qualified here. |
| [LiAM-SAM: Lifecycle-Aware Memory for Robust SAM2-Based MOT](https://arxiv.org/html/2609.28078v1), §§3.4.1–3.4.2 | Distinguishes colliding objects from duplicate hypotheses using motion; reallocates overlap components using depth support and re-encodes corrected memories. Thus appending monocular depth or motion to ownership arbitration would also collide with a direct September 2026 author preprint. It is a detection-based MOT setting, so its results are not interchangeable with mask-initialized VOS. |

These are mechanism comparisons, not independent reproductions or claims that their reported accuracy transfers to this runtime. This bounded search does not establish exhaustive novelty absence. It does establish that the currently specified proposal is too close to existing operations to justify engineering.

## A decisive, inexpensive necessary test — already failed at the formulation level

Write the proposal's complete score and admissible assignments. If it is

\[
\min_x \sum_{k,j}c_{kj}x_{kj}
\]

with ordinary one-to-one/absence constraints, it is conventional assignment over the supplied costs. Replacing mask affinity by LightGlue tracks does not, by itself, change that fact. A more complicated temporal graph also needs a distinct observation or estimator argument, rather than a new name for matching.

A fatal information counterexample is two visually indistinguishable objects that become fully hidden together and then reappear. Suppose the available initial-to-current matches and admissible trajectories are invariant to exchanging their IDs. For the exchange permutation \(P\),

\[
E(X;\mathcal O)=E(PX;\mathcal O),
\]

and both assignments obey exclusivity. A no-overlap solution can therefore be a complete identity swap. A deterministic tie-break chooses one solution; it does not identify the correct one. Conversely, if asymmetric causal evidence breaks this symmetry, the same evidence must be available to the assignment baseline. This argument requires no generated arrays, model call, or GT inspection.

The counterexample is a special unobservable configuration, not a universal failure theorem. It explains precisely why counting fewer duplicate pixels or lower overlap loss cannot demonstrate identity recovery.

## Controls and evidence missing before a successor could be credible

The smallest meaningful control set would include independent SAM2 predictions with ordinary non-overlap competition; joint assignment using the **same** identity-indexed correspondence and motion costs; and a neighbor-aware temporal control implementing SENTRY's relevant mechanism. A trained multi-target method such as SAM-MT is a task-level reference, with training/supervision differences disclosed. If depth is used, LiAM-SAM's overlap-correction mechanism is material. All alternatives must receive the same initial labels and causal frames; absence must be legal; extra encodings, object passes, and tracking work must be charged.

Three concrete pieces are missing:

1. **An estimator or observation that distinguishes the proposal from those controls.** None has been specified. Generic mutual exclusion, negative prompts, depth ordering, backward matching, and memory rewriting do not supply it.
2. **A prospective task contract that actually measures identity information.** The closed 0029 protocol initializes one binary target per video. It does not establish an all-object identity assay. I did not inspect mask IDs to select convenient multi-object cases. Any later assay would need a fixed multi-object population and identity-sensitive error accounting alongside official J&F, without choosing videos or objects after observing failures.
3. **A feasible strong-control comparison under the stated resources.** Only the existing SAM2-tiny frontend is qualified. This review has not established that a faithful SENTRY or trained multi-target reference fits four CPU threads, 8 GiB, the operation limit, and the remaining acquisition budget. Their omission cannot be repaired by calling independent single-target SAM2 the strongest comparator.

Accordingly, **do not build another candidate-bank or memory experiment from this note**. A later concrete mechanism must first survive the assignment/symmetry test and state what measurable identity-bearing evidence remains unused by the strongest same-information control. Passing that source/math check would only justify a newly frozen necessary diagnostic, not an efficacy claim or a manuscript. The six 0029 development videos and their observed outcomes are not a pool for tuning such a successor.

## Inspection and access record

Local source/protocol notes inspected: `protocols/0029-signed-memory-diagnostic.md`; private `0029-screening/{memory-repair-prior,counterfactual-memory-prior,embodied-options,driving-options,signed-memory-pressure-test}.md`; and `0029-preparation/exre-full-source-review.md`. Earlier rejected pose/depth/normal calibration, TAPNext workload compression, delayed replay, and traffic-timing routes were not reopened.

This task made three web-tool calls, comprising the following operations:

1. One search call with three queries: `video object segmentation multi object exclusivity correspondence assignment identity ambiguity occlusion SAM2 CVPR 2026`; `SAM2 multi object segmentation joint inference mutual exclusion negative prompts identity 2025 2026`; `video object segmentation topology object parts split merge material correspondence VOST CVPR 2025 2026`. Search-only hits beyond the three sources above were not used to assert method details.
2. One open call for the three exact arXiv HTML URLs in the table.
3. One find call: SENTRY patterns `3.2`, `Algorithm`, `neighbor`; LiAM-SAM patterns `3.4.1`, `collision`, `depth`. `collision` returned no exact match.

Web-tool response-body bytes are **not independently metered**, so this is not a zero-byte transfer claim. There were no direct HTTP downloads, archive/model/media payload requests, new acquisition reservations, library installations, model/solver runs, new GT reads, or repository changes. The only created artifact is this private note. Access record closed 2026-10-20 at approximately 08:30 UTC.
