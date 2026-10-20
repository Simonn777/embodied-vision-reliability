# Independent source review of fixed-query evaluation

Read-only review; no test repetition, model import, scientific prediction or query-annotation access. **No blocking discrepancy found in the pure evaluator's declared scope.** Artifact admission and GT acquisition remain the caller's responsibility.

Reviewed sources:

- `evaluate_queries.py`: `272ae306b0f80573d3350d892cefbc13cf9df5a06d89b6188ee9c86584c5237f`.
- Frozen `diagnostic_core.py`: `4c76bdb492bef0507b4322be67a2a157ec34c420bfacb874388b1f7facef0c34`.
- Frozen `0029-signed-memory-diagnostic.md`: `2a8072b610db97833049215b34ffb3144e7a63c2f6b741f2ed605be7091a722c`.

The public protocol and arithmetic core have no diff from `c09e240009869402e2b97a55af42617eea3c6405`.

| Contract | Source finding |
|---|---|
| Complete population | Exact 48 unique predetermined identities are required. A technical failure makes the whole cohort ineligible before annotation access; no successful-subset means are returned. Ordinary nonactionable queries remain valid baseline outcomes. |
| J and void | Logits use strict `>0`; official void 255 is excluded from intersection and union. Empty/empty is one. Eight queries are averaged per video and six video means equally. |
| Deletion oracle | The gate uses only `B,D1,D3`; nonactionable rows have admissible oracle equal to B. Unrestricted deletion headroom is separate. Native `B,M0,M1,M2` oracle is descriptive and never enters the gate or deployed selector. |
| Strongest controls | The pooled comparator is the maximum pooled mean of the 12 fixed named controls, including B, prompt variants and native-bank ranking. Each video's comparator is independently its maximum named-control video mean. This is not a per-query GT oracle. Ungated D1/D3 and both oracle families are excluded from the control maximum. |
| Exact gate | Coverage requires evidence in at least four videos. The other clauses are admissible deletion-oracle gain `>=0.02`, pooled signed gain `>=0.02`, strictly positive per-video margins in at least four videos, and every video margin `>=-0.02`. All are conjoined. |
| Transferred negatives | At evaluation, any nonvoid label other than the fixed target is a correct negative. Void is reported as unknown and excluded from precision's known-label denominator; retained, unknown, correct and incorrect counts remain visible. This precision denominator does not change the 48-query performance population. |
| Sealed choices | Candidate availability, target consistency, witness coordinates, actionability and allowed choice banks are preflighted for all rows before annotations. Evaluation consumes saved choices and does not rerank them using GT. It does not independently reconstruct all saved loss-based decisions; that is a separate audit responsibility. |
| Costs | Each timing/count column is summed separately with a reported-query count; wall and stage columns are never combined into a synthetic total. Query costs exclude setup and intervening baseline frames, so the complete video and frontend outer costs must remain separately reported by the caller. |

The native-worker source uses disjoint sequential timing contexts for the per-query categories. Its call-count categories describe different, sometimes overlapping operations; they likewise must not be added into an unlabeled “total calls.” The complete outer wall time remains the authoritative end-to-end charge.

## Prospective independent saved-output audit arithmetic

This is a plan only, not permission to access annotations or run a second scientific pipeline.

1. Before any query-GT decoding, verify all six complete prediction receipts and clean outer operations, fixed source/input/evidence bindings, all 48 identities, all candidate archives and raw-array hashes, trace/parity checks and selected branches. Bind the existing 392-input/54-feature/48-match witness audit. Reject a partial video population.
2. Using saved logits and the sealed witness coordinates, independently reconstruct pixel-center bilinear samples, balanced logistic losses, unsigned mask change, native predicted-IoU ranking, deterministic SHA choice, exact margin/tie rules, and all 13 deployed selections. Compare, never replace, the saved selections. Do not import production selector or evaluator arithmetic.
3. Only after that barrier, decode the 48 specifically admitted query masks. Independently count integer intersections/unions outside void for every stored candidate, then evaluate the already verified selected branches. Retain all 48 rows and every control; compute both oracle families only as labeled diagnostics.
4. Recompute the six eight-query means, pooled means, fixed named-control maxima, all four numerical continuation clauses and coverage. Also reproduce witness precision, no-op/action counts, loss/J sign tables and strict `J drop>0.05` harm diagnostics. GT cannot change eligibility, evidence, selections or cohort.
5. Check each reported cost column against the saved per-query timing records and all-video counts. Keep the six complete video outer charges and the already completed frontend charge separate from overlapping stage summaries. Retain a failed audit unchanged; any arithmetic tolerance must be declared before execution.

This can establish saved-evidence consistency and correct metric arithmetic. It cannot independently certify that model correspondences are true physical matches, rerun the native memory intervention, or infer closed-loop segmentation gains from baseline-history forks.
