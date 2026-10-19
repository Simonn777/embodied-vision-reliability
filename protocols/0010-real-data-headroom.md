# 0010: fixed real-data development headroom screen

**Requires its separate published source/plan receipt and exact CI before execution.** Question: on one fixed small EuroSAT development cohort, does exact query-deleted StatA improve actual top-1 classification enough to justify investigating a cheaper approximation? This is an algorithm comparison and a stop/go screen, not a novelty claim, causal proof, manuscript result or confirmatory benchmark.

## Prerequisites and fixed cohort

Proceed only after separately admitted acquisition, the eight-image capability check, matched image/text feature checks, and cloud CPU artificial parity tests succeed. The eight capability images and their image embeddings are not scientific samples. Verified text prototypes from that probe may be reused as a common fixed artifact. Use the pinned official OpenAI CLIP ViT-B/32, exact ten author-template prompts, verified RGB archive and pinned spatial-train filenames described in [baseline and data admission](../docs/literature/0010-data-and-baseline-admission.md). No third-party cached embeddings, new encoder, prompt variants, augmentations or replacement assets.

Select **128 distinct spatial-train basenames**, excluding the eight predeclared capability names. Sort the remaining names by `(SHA256(UTF8("0010-stata-headroom-v1|" + basename)), basename)` and take the first 128. Save the exact filename/hash list and batch membership before reading pixels or features. Consecutive blocks form **two batches of 64**, preserving hash order. No balancing, outcome-based selection, replacement, alternative salts or batch-size search. Within each batch the donor for query i is the next ID cyclically in this fixed order.

**Why 64, rather than 16 or 32:** the already pinned [author README, lines 112–120](https://github.com/MaxZanella/StatA/blob/2635f9355f9389756064becb5d9ed54c2756809c/README.md#L112-L120) explicitly uses B=64 as its small-batch regime and B=1000 as its larger regime; `main.py:25` defaults to 64. Only this cached source/README was additionally read, not a newly retrieved full paper. With ten candidate classes, B=16 would artificially emphasize singleton deletion; B=32 still has no better author-regime justification. Use B=64 prospectively even though only two computational groups provide little dispersion evidence. Do not shrink it after runtime or outcome failure. The actual class occupancies remain unknown and are not engineered. This matches a documented author batch scale, **not** the author's full benchmark: our ViT-B/32, spatial-train split, fixed unlabeled selection and two groups differ from the documented multi-task/class-sampled evaluation.

All selected files must exist exactly once in the verified archive and satisfy the frozen decode contract. Missing/corrupt files, duplicate-byte anomalies, or unsupported numerical boundaries stop admission rather than trigger substitution. Preserve this cohort as used development data afterward. Spatial-train membership and disjoint filenames do **not** prove independent physical scenes, geographic buffers, independence of batches, or absence from CLIP pretraining. Validation/test filenames remain unused.

## Fixed extraction and methods

Extract each image once using official preprocessing, CPU float32, batch size four, four Torch threads and four inter-op threads, no augmentation. This prospective thread setting matches the frozen capability runtime; the earlier unfrozen draft said one inter-op thread. The artificial solver timing must use the same four/four setting. Reuse the capability probe’s already computed text prototypes, verifying the saved feature/file hashes and identical checkpoint, source, token IDs, prompt bytes/class order, normalization and runtime contract; do not re-encode unchanged prompts. The eight capability image features remain excluded from the scientific cohort. Save the 128 new unit-normalized image vectors and the reused text-feature identities privately; reject nonfinite/zero norms. All methods consume these identical saved arrays. Use StatA's authored **100×normalized-feature** logits, not a substituted learned CLIP scale. Separate predictor inputs (vectors and opaque IDs) from the scorer's filename-to-class mapping. Filenames already encode class information, so do not claim those labels were unknowable to the pipeline; they must not enter fitting, selection changes or parameter choice.

| Fixed arm | Definition |
|---|---|
| Zero-shot | Argmax of fixed 100× cosine logits. |
| Native StatA α=1 | Pinned device-only CPU adaptation, original finite schedule and defaults. |
| Native StatA α=4 | Same algorithm with the prospectively fixed stronger anchor. |
| Exact LOO α=1 | Full refit on D minus query; declared one-way held-out readout from final-block z4. |
| OTHER-out α=1 | Full refit on D minus fixed donor; native one-way readout of retained query. |

Use the reviewed 11×5 responsibility schedule, ten mean/covariance updates, k=3 signed graph and hard-count beta; no convergence extension, flooring, tie repair or zero-mass fallback. A boundary is not a usable prediction. Each fitted state retains its independently verified authored-reference parity execution. For each batch, fit full D at α1/4 and each of its 64 distinct deleted subsets **once**: the same D\{j} fit supplies LOO query j and OTHER-out query immediately preceding j. This gives **66 fit states/batch, 132 states total, 264 authored solver executions** with current parity instrumentation. Reusing the identical subset state avoids duplicate fitting; it does not approximate or skip a method.

Native LOO and OTHER-out match training cardinality but use different readout peer sets: LOO can message N−1 peers on an N-position graph; OTHER-out has N−2 other peers on N−1 positions. Report this confound prominently. Do not attribute a difference purely to moments or self-influence. The prospective common-peer diagnostic is explicitly deferred, not silently mixed into this screen.

## Bounded execution and interpretation

Before freezing, use the actual eight-image timing breakdown only for resource admission. A conservative projection—1.5×(one measured load/setup cost plus 16×the sum of the measured eight-image preprocessing and forward costs; reuse verified text features without another text forward)—must fit a **450-second encoder allowance**; this is not a speed guarantee. Separately qualify real-shaped artificial solver fixtures (63/64 unit rows, 512 feature dimensions, ten anchors; no actual image features or semantic labels). Measure once the complete α1 deleted-state fit plus one held-out readout (T63), and full-state α1/α4 fits (T64,1 and T64,4), including both parity executions and native-readout checks. Require 1.5×(128×T63 + 2×T64,1 + 2×T64,4) ≤ **360 seconds**. The separately frozen artificial timings have now completed once. They establish only resource admission and do not show quality. If timing breakdowns are missing or either projection fails, this design is resource-unadmitted; do not silently shrink it. Leave 90 seconds for checks/records under **one 900-second outer hard limit**, four CPU threads and 8 GiB. Actual measurement can still exceed this estimate: stop and retain the incomplete record, without rerunning selected cases or reporting partial-cohort accuracy as a completed result. Downloads are separately admitted; no additional paid service or new asset download is part of this screen. Record encoder, solver/parity and full pipeline costs separately; exact LOO is an expensive headroom reference, not a deployment-efficiency claim.

Seal all 128×5 predictions, statuses and hashes before the independent scorer accesses class labels. Require complete coverage and all numerical/parity gates for a completed comparison; no paired-survivor filtering. Any unsupported case, timeout or hash mismatch is **INVALID/INCOMPLETE**, not a zero-quality score and not a replacement opportunity.

Report top-1 counts/accuracy for all five fixed arms; LOO wrong→right and right→wrong counts against **each** control; per-batch net changes; and true within-original-batch occupancy strata 1, 2, and ≥3 used only by the scorer. No per-image or per-batch oracle baseline selection. Computational batches are not verified independent scenes; show paired counts and batch dispersion without nominal population-confidence claims.

A provisional, intentionally demanding **continue-to-mechanism** gate is:

- LOO yields at least **8 net additional correct /128 (+6.25 percentage points)** against the strongest aggregate fixed control among zero-shot, native α1, native α4 and OTHER-out (equivalently, at least eight net wins against each). The comparator shortlist is fixed now; no per-image oracle choices.
- LOO is strictly better in **both 64-image batches** against that strongest fixed comparator (against every co-best comparator if tied). Report all other paired batch deltas. Singleton-stratum net corrections versus each native control must be nonnegative when singleton examples exist; an empty stratum is unassessed and supports no singleton-safety claim.
- All 128 cases and all arms are valid. If the quality gate fails and fewer than eight errors or disagreements against the strongest control make the required net gain impossible, label the screen **low-headroom/insensitive**, not proof that deletion can never help. Otherwise report the actual failure as no-go for this cohort. No additional sample-size, occupancy or disagreement threshold is chosen after outcomes.

These thresholds require a visible quality signal rather than variance reduction or a few anecdotal flips. They are prospective development criteria, not a powered hypothesis test. Failing the improvement criteria stops this exact direction for this cohort; no rehashing, prompt/alpha tuning, reinterpretation of excluded cases or enlarged same-cohort rescue. Passing would justify a separately frozen common-peer diagnostic and new-data study; it would not establish novelty, scene generalization or an efficient method.

## Actual resource evidence and launch boundary

The [saved capability and solver receipts](../results/0010-capability/README.md)
precede this freeze. Eight excluded capability images and ten prompts completed
once; their finite float32 vectors and normalization were checked from saved
files without another forward or classification score. Twenty artificial
functional tests passed with zero skips. The three fixed timing fits completed.

The actual encoder projection is **25.022365751745383 s**, below450 s. Timing
T63=0.11852241800079355 s, T641=0.08900357699894812 s and
T644=0.06562033699447056 s give **23.220175998132618 s**, below360 s. These
admission projections use the declared formulas; the one-shot900-second outer
limit still governs the real study. No cohort/quality threshold was changed.

The filled plan binds the actual saved input identities and original sources.
Before execution the caller must verify its separate canonical Git receipt,
source-parent and receipt-only delta, all bound repository files against current/index/Git
objects and each external saved input against its full file hash, and the exact merged-main CI. The plan's external_bindings entry for
the original capability parent execution is checked by this admission step;
the current consumer does not independently validate that external entry.

Use the existing cloud interpreter and original PYTHONPATH from
scratch/0010-capability-control/imports.command.json, CPU0–3, four intra/inter-op
threads, existing8GiB cap and no new download/install. Preserve both historical
worktrees and all markers. Run experiments/0010-headroom/run.py **launch** once
with the separately frozen plan SHA, existing clip-0010-assets, fresh
scratch/0010-headroom-output and scratch/0010-headroom-supervision. The launcher
and child create exclusive asset-root0010-headroom-launch.started.json and
0010-headroom-started.json markers. Existing marker/output stops launch;
changing directories is not permission to retry. Retain the outer real exit,
partial records and failures. No direct _child invocation, warmup, prompt change,
new text encoding, repeat inference or surviving-subset score is allowed.

Fixed selection and metadata precede pixel access. The128 selected JPEGs have
not yet been decoded or scored; archive membership/byte uniqueness and real
numerical coverage are execution gates. The eight capability samples remain
excluded. Scene independence, CLIP pretraining exposure and peer-pool causal
identification remain unverified. These results will be exploratory development,
not an untouched confirmatory test or manuscript contribution.
