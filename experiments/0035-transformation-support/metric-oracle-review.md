# 0035 categorical support oracle and metric review

Source/math review only, 2026-10-20. No predictions, images, annotations, tensors, or scientific runs were accessed. This reviews `DRAFT_SCOPE.md`; it neither admits execution nor establishes a method contribution.

Recommendation: report the four actual categorical arms with pinned-author J and Jtr, and one explicitly privileged **pixel-support ceiling**. The ceiling below simultaneously maximizes every object’s framewise J over a relaxed, precisely stated prediction family. It is not a deployable selector. Also report the same ceiling using native support alone; this inexpensive decomposition prevents false-positive cleanup from being mistaken for new candidate support.

## 1. Fix the actual categorical controls

At each frame, let objects be the original positive first-mask IDs, ordered increasingly. For object \(k\), let \(\ell_{k,0}\) be its native selected original-resolution logit, and \(\ell_{k,1},\ell_{k,2}\) the two alternatives in original decoder-slot order after removing the selected slot. Use the same interpolation, presence handling, and finite-value requirements for every arm. “Support” below means the actual eligible post-presence output, not an ungated raw candidate resurrected when the native object-presence rule excludes it.

For arm \(a\in\{B,B+M_1,B+M_2,B+M_1+M_2\}\), define

\[
L_{a,k}(x)=\max_{j\in I_a}\ell_{k,j}(x),\qquad
I_a\in\{\{0\},\{0,1\},\{0,2\},\{0,1,2\}\}.
\]

Output background if \(\max_k L_{a,k}(x)\leq0\). Otherwise output the object with largest score, breaking exact positive-score ties by lowest original object ID. Thus background wins a zero tie. Do not independently threshold overlapping binary masks and call them a categorical prediction.

Every arm contains native **per-object support before composition**. That does not make its categorical output add-only: an alternative for object \(j\) can take a pixel previously assigned to object \(k\). All real controls, including B, must use this identical competition rule. If “native” elsewhere uses a different multi-object consolidation rule, report this B as the declared native-bank composition; do not claim equality without checking it. The supplied initial mask is preserved separately and is outside the official evaluated interval.

## 2. The smallest simultaneous J ceiling

Fix the evaluation pixel domain and void convention first. Let \(G_k\) be the ground-truth pixels of object \(k\), and

\[
S_k=\{x:\max_{j=0,1,2}\ell_{k,j}(x)>0\}.
\]

Consider the relaxed categorical family

\[
\mathcal F(S)=\{f:\ f(x)=0\ \text{or}\ [f(x)=k\text{ and }x\in S_k]\}.
\]

Background is always permitted. Every declared real arm belongs to this family, as does any mixed per-object arm choice followed by the same positive-logit categorical composition. The family additionally permits arbitrary pixelwise choices and rejection to background.

Define the privileged output

\[
O(x)=
\begin{cases}
g(x),&g(x)=k>0\ \text{and }x\in S_k,\\
0,&\text{otherwise}.
\end{cases}
\]

It is one categorical image, with \(O_k=G_k\cap S_k\); the object masks are disjoint because the ground-truth masks are disjoint. For any \(f\in\mathcal F(S)\), if \(|G_k|>0\),

\[
J_k(f)=\frac{TP_k(f)}{|G_k|+FP_k(f)}
\leq\frac{|G_k\cap S_k|}{|G_k|}
=J_k(O).
\]

The first inequality uses \(TP_k(f)\leq|G_k\cap S_k|\) and \(FP_k(f)\geq0\). If \(G_k\) is empty, O is empty for that object and obtains the conventionally maximal J of 1. Thus **the same categorical O maximizes every object’s framewise J simultaneously**. Any predetermined nonnegative weighted average of those J values—including author J and Jtr—inherits the upper bound. This needs one support pass, not enumeration of \(4^K\) joint arm assignments.

The ceiling includes native when native predictions obey the stated support/zero-threshold rule. The special initial prompt mask need not obey it, which is harmless because initial-frame metrics are excluded. Keep all raw-frame outputs, but do not claim the proof covers a special raw initial-frame report unless its oracle is explicitly defined.

This proof is about J. It gives no F/boundary-score upper bound, no attainable whole-mask selection policy, no ancestry guarantee, and no estimate of generalization.

## 3. Interpret headroom without overclaiming information

Let \(O_B\) use \(S^B_k=\{x:\ell_{k,0}(x)>0\}\), with the same privileged construction. Then

\[
J(O)-J(B)
=[J(O_B)-J(B)]+[J(O)-J(O_B)].
\]

Both terms are nonnegative for the fixed aggregation:

- The first includes perfect native false-positive removal and perfect ownership arbitration wherever multiple native object supports overlap.
- The second measures extra ground-truth support made available by the alternative slots. At a nonempty object-frame it equals \(|G_k\cap(S_k\setminus S^B_k)|/|G_k|\).

Neither term is a deployable gain. In particular, a large total ceiling with negligible second term does not establish that alternative masks recover omitted material. Conversely, zero extra support does not prove that alternatives cannot improve a practical competition rule: their scores might resolve ownership already possible inside native support. Report these as separate mechanisms.

Insufficient total ceiling can rule out a proposed J/Jtr gain within this fixed candidate-supported family. Adequate ceiling only establishes room, and the relaxed ceiling may be very loose. It cannot authorize changing clips, slots, score thresholds, or the evaluation population after results.

Avoid two misleading replacements:

1. Independently choosing the best binary object result from the four **already composed global arms** may give mutually incompatible labels. It is not an upper bound for the larger family obtained by independently choosing original per-object arms before global composition.
2. Keeping native labels on ground-truth background in an “add-only oracle” leaves removable false positives and does not maximize J for the declared globally competing arms. The background-always-allowed construction is clearer.

## 4. Pinned author metric: a material draft correction

Read scope: author repository commit `fe274574cb03c8a3ea83e121dd76e20b703781fd`, evaluation source only. The following are implementation facts, not an assertion about the authors’ intended void semantics.

The [dataset loader, lines 80–93](https://github.com/TRI-ML/VOST/blob/fe274574cb03c8a3ea83e121dd76e20b703781fd/evaluation/source/dataset.py#L80-L93) records void pixels but replaces label 255 with 0. It enumerates object IDs as 1 through the maximum initial label. The [evaluation caller, lines 50–69](https://github.com/TRI-ML/VOST/blob/fe274574cb03c8a3ea83e121dd76e20b703781fd/evaluation/source/evaluation.py#L50-L69) passes **None** for void, so released-script J counts predictions on original-255 pixels as false positives. `DRAFT_SCOPE.md` currently says “255 ignored”; that is incompatible with this pinned execution path.

Recommendation: primary “pinned-author J/Jtr” follows that exact background behavior. A separately named void-ignore sensitivity result is possible, but must not silently replace the author-compatible primary. The ceiling proof works under either fixed convention.

The [J helper](https://github.com/TRI-ML/VOST/blob/fe274574cb03c8a3ea83e121dd76e20b703781fd/evaluation/source/metrics.py) computes intersection over union, with empty/empty equal to 1; it supports a void mask, but the above caller does not supply one. Its boundary helper is not a justification for labeling additional F/JF results official VOST.

The caller evaluates frame positions `1:-1`, then uses the suffix beginning at `floor(0.75 * evaluated_frame_count)` for `J_last`/Jtr. The [statistics helper](https://github.com/TRI-ML/VOST/blob/fe274574cb03c8a3ea83e121dd76e20b703781fd/evaluation/source/utils.py) averages each object’s framewise J. The [top-level reporter, lines 35–52](https://github.com/TRI-ML/VOST/blob/fe274574cb03c8a3ea83e121dd76e20b703781fd/evaluation/evaluation_method.py#L35-L52) equally averages sequence-object means. This is not equal-video weighting when object counts differ, nor frame-pooling across sequences. Object identities are fixed; do not use GT matching to relabel predictions.

For the draft’s four lengths, zero-based positional intervals—not filename numbers—are:

| Clip | All frames | J positions | Jtr positions | J / Jtr frame counts |
|---|---:|---|---|---:|
| 4691_cut_guava | 86 | 1–84 | 64–84 | 84 / 21 |
| 817_cut_onion | 90 | 1–88 | 67–88 | 88 / 22 |
| 9675_fold_clothes | 201 | 1–199 | 150–199 | 199 / 50 |
| 1934_tear_film | 69 | 1–67 | 51–67 | 67 / 17 |

That is 438 full-interval and 110 tail frame positions, each repeated for that clip’s evaluated objects. Counts of objects remain unknown before the permitted initial-mask read. Verify positive initial IDs are contiguous if reproducing the author’s 1..max enumeration; if not, explicitly settle the discrepancy before evaluation rather than silently padding nonexistent objects or dropping IDs. Keep per-frame/per-object values and named equal-video summaries as descriptive, alongside the primary object-weighted aggregate. Nonfinite or incomplete outputs are technical failures, not entries to remove with a successful-subset mean.

## 5. Minimal report and decision scope

Report B and all three unions, each as a complete categorical result: per-object J/Jtr, per-clip values with the weighting stated, and the author-compatible pooled values. Separately report \(O_B\), O, and the two headroom components. Keep oracle results visibly privileged and outside real-control rankings.

No independent-parent uncertainty claim is justified while original recording identities are unresolved. No F-based official VOST claim or completion of the existing two-source +3 official JF objective follows from this screen. A numerical continuation threshold remains the parent protocol’s prospective decision; this review supplies a valid ceiling and interpretation, not an efficacy gate or novelty finding.

Access accounting: local draft/cached source and metadata plus primary source-page text reads only. No separate source payload, dataset member, model, or asset download was allocated. Web text transfer was not independently byte-metered.
