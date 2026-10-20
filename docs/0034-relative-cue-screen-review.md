# Prospective relative-cue screen: methodological review

2026-10-20. Source/text only: read `target-conditioning-source-question.md`, `relative-cue-independent-critique.md`, and the fixed 0033 source declarations. The design author also supplied its prospective summary. No arrays, labels, scientific outcome files, new primary sources, specifications, implementation or executions were accessed or created. This note recommends a comparison; it does not admit an experiment.

**Recommendation:** use four TRAIN leave-one-parent-out folds to test a particular transferable cue against strong same-input controls. Prefer all existing noninitial frames over recycling only the old horizon-event subset. Choose one input-resolution hypothesis before execution; do not combine a representation change, new high-resolution input and larger learner into one unexplained candidate. The closed 0033 architecture/training recipe remains closed. A useful known baseline is a legitimate outcome; novelty is not a prerequisite for ordinary development.

## Population and leakage boundary

The four source-declared TRAIN recordings have 25/38/83/86 frames. Each fold fits on all noninitial frames of three recordings and predicts the entire held recording. Across folds, this yields 228 noninitial out-of-parent predictions and **224 interior scoring frames**, excluding each recording's first and last frames from the primary mean. Keep all 232 raw frame records, with initial outputs native. Use equal frame weights within each held parent and four equal parent means. For fitting, parent-balanced loss avoids letting the longest recording dominate; every arm must use that same declared objective.

The old 96 horizon events contain only 87 unique current frames. They can remain an explicitly secondary comparison, but repeated anchor contexts do not create additional current-frame observations or independent replication. If resource constraints force that smaller population, fit each unique frame once and label the result a restricted development screen. It should not be presented as full-parent transfer.

All six parents are already exposed development. LOPO prevents direct use of a held recording's labels in its own fold; it does not undo earlier research exposure or create four independent external tests. Neither original DEV parent should select this screen's settings. Any feature statistics, learned metric, regularization selection, temperature or threshold calibration must use only the three fitting parents. A single fixed setting is simplest; a small predeclared tuning procedure may instead be nested within those three parents and equally charged across arms. Selecting it from the four outer-fold scores invalidates their role as transfer estimates.

## Target support is an input, not a correctness certificate

Current-frame foreground/background supports may be derived from the unchanged native prediction of the held frame. This is causal, label-free test-time conditioning, not forbidden training leakage. Do not pool later frames or future labels into a held-frame prototype. Initial-mask supervision is also legitimate for ordinary VOS, but changing from current native support to an initial-GT reference changes the comparison: fix one policy and give every target-conditioned control the same reference access.

Fix the native-defined correction band, interior/exterior erosion rules, reference cap, deterministic sampling, query-exclusion halo, feature interpolation and missing-support behavior before execution. Reuse exactly those sites across arms, preserve native outside them, and retain absent/empty/support-failed cases as native no-ops in the full denominator. No GT trimap, GT-straddling query selection or removal of contaminated supports is allowed. Erosion does not prove that a predicted interior belongs to the target, and removing the query's own feature from its prototype only avoids self-similarity; it does not make the observations independent.

## Smallest interpretable control ladder

For a **cached-feature representation question**, a compact nested ladder is:

| Arm | Added observation/readout |
|---|---|
| Native | No learned correction |
| Logit | Same correction sites and readout using native logits |
| Absolute | Native logits plus absolute cached features |
| Prototype | Native logits plus foreground/background mean similarities |
| Strong relative | The same conditioning plus fixed-cap foreground/background nearest-neighbour or affinity cues |
| Relational | Exactly the strong-relative inputs plus the proposed fixed local relation maps |

The incremental comparison is the last arm against strong relative, not merely against a weak single-mean prototype. The nearest-neighbour arm must include the prototype inputs if the intended claim is an increment over that nested base. Keep the readout form, terminal-checkpoint policy, fitting labels, loss, correction bound and threshold identical. A fixed regularized linear readout is a reasonable first restriction. Different feature dimensions then represent different hypothesis-class complexity; zero-padding does **not** make effective capacity equal. One fixed spatially permuted relation block of the same dimensions can serve as a capacity/association negative control, preserving the native and conditioning inputs. It must not be selected from a bank after seeing labels. This is useful evidence, not a perfect isolation of every regularization effect.

For an **original-resolution RGB detail question**, use RGB at that resolution in the ordinary same-RGB target-affinity/nearest-neighbour control and a direct RGB/logit readout, not only in the candidate. A native-unary/local RGB-affinity control is particularly relevant to the existing bilateral/CRF precedent. Pair the candidate with its identical RGB-downsample-to-256-and-upsample version to test the resolution hypothesis. The absolute-feature reference can remain a reference, but a victory over it alone cannot identify the relational mechanism. Avoid a large Cartesian product: choose this RGB question instead of automatically adding every feature-Gram variant.

Original RGB is additional information at the readout interface, not a new sensor or proof that the cached feature tensor lacks detail. Normalized dot products are invariant only under a common orthogonal channel transform and an equivariant normalization. Fold-specific per-channel whitening defines an anisotropic metric and breaks the simple invariance claim. RGB differences need an explicit color space and scaling; no physical alpha-matting interpretation follows from arithmetic in sRGB. Local Gram/distance maps alone do not supply target polarity.

## Decision, metrics and cost

Seal every fold/arm prediction before held-parent label scoring. Primary outcomes are whole-mask original-resolution J, F and J&F with the existing target-ID and official-background treatment of 255; boundary-only loss, pixel AUROC and fitted loss are secondary. Retain all parents, frames, failures and native no-ops. Report both pooled differences and all four paired parent differences.

A concrete **proposal for the owner's prospective choice**, not an already fixed gate: require candidate J&F at least +1.0 pp above native and +0.25 pp above the strongest named pooled control; positive margins over that same control on at least three of four parents; no parent worse than native by more than 0.5 pp; and neither pooled J nor F down more than 0.5 pp versus native. Choose the strongest named control once by its complete pooled score for this diagnostic comparison, never use per-parent GT to deploy different controls. Four exposed parents cannot support a strong confidence claim. The owner may select different meaningful margins before execution, but should not substitute “some fold improves” afterward.

If ordinary conditioning/affinity helps and the relational increment does not, record a useful known baseline and close only the claimed increment. If original-resolution RGB helps but its ordinary matched control explains the benefit, do not claim a new relational mechanism. If the fixed readout fails, that rejects the stated implementation/budget; it is not a theorem that no target-relative information exists.

Use the same declared optimization allowance and label weighting for learned controls. Record actual feature construction, support search, fitting, prediction and evaluation costs, plus the inherited native cache. Shared preprocessing can be reused operationally, but must not vanish from a candidate's complete-pipeline accounting. Limit reference banks and stream frames; do not materialize all-pixel pairwise matrices. A 4-thread/8-GiB/1,800-second complete-operation plan still needs practical qualification, not an estimate promoted to a measurement. Do not split a failed operation and describe the fragments as a successful original budget.

This development comparison neither establishes driving/embodied performance nor meets the unchanged two-independent-held-out-source +3 official J&F manuscript gate. It is small enough to make a concrete development decision without requiring a new evidence framework or an unsupported novelty argument.
