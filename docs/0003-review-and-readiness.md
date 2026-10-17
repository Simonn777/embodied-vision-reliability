# Motion changes versus localization drift: design and readiness

October 17, 2026. AI-assisted research; reviewed cloud candidate
`b605ae13a8ff15ccfe6df2930673ce196f4d0ef7`.

**The study design is frozen. An independent evaluation cohort is not ready.**
This step audits a new data source and fixes the next study's rules. It does not
establish a better tracker or turn the already inspected 0002 test set into new data.

## What the cloud actually verified

| Check | Observed result |
|---|---|
| PointOdyssey v1.2 sample | One 3,324,284,510-byte download; full SHA256 matches pinned upstream LFS metadata |
| Complete cached TAR inventory | 14,415 members; 3,334,075,426 declared member bytes; two named annotation parents |
| Fixed development annotation | `sample/r4_new_f/anno.npz`, 98,527,880 bytes; full member hash recorded |
| Numeric annotation audit | Stopped at the frozen C-order-only gate; `valids` and `visibs` have Fortran order |
| Actual geometry evidence | No numeric GT values interpreted, no selected query points, no projection checks; errors remain undefined |
| Resource use | Main cached audit: 21.083 seconds, peak RSS 29,820 KiB; its failure exit is preserved |
| Independent evaluation families | None established; names, checksums and a different dataset do not establish family or pretraining independence |

The full sample and any unresolved linked family are reserved for audit/development.
The [cached-data report](0003-cached-resource-audit-2026-10-17.md) links the complete
inventory, headers, bounded-read accounting, actual failures and cloud verification.
The [network follow-up](0003-network-and-timing-followup-2026-10-17.md) preserves the
initial DNS failure and the successful use of the existing runtime network settings.
No paid API, GPU, credit purchase or account change was introduced. The acquisition
ledger's conservative cumulative ceiling is 4,823,387,101 bytes, below 10 GiB.
Raw datasets and weights are not redistributed.

## What the frozen comparison can distinguish

For the same three timestamps, the signed prediction innovation decomposes exactly
into the innovation of the true image trajectory plus that of the localization
error. Their lengths do not add: cancellation can make a wrong track look smooth.
Controls cover a correct turn, constant offset, linear drift, curved error,
cancellation, irregular frame intervals and a moving camera viewing a static point.

This distinguishes **image-motion changes from 2D localization-error curvature**.
It does not identify every kind of drift: a constant offset or linear error can
have zero innovation. World-motion diagnostics require independently checked
3D annotations and camera geometry; image curvature alone does not establish
physical acceleration, and no physical-unit or robot result is claimed.

## Improvement target and decision rule

At matched **80% coverage**, the augmented score must reduce conditional failure
risk by **at least 20% relative** to the unchanged native score. Define
`C = R_augmented - 0.8 * R_native`; require `C <= 0` and the upper endpoint of its
paired 95% family-bootstrap interval below zero. Report both risks, their absolute
difference and coverage. This is a prospective pilot continuation target, not an
achieved result, a power calculation or a guarantee of practical robot benefit.

The decision also needs at least 20 verified independent evaluation families,
five families containing native accepted failures, 20 native matched failures,
valid GT support, reachable coverage and at least 1,900 of 2,000 defined bootstrap
replicates. Insufficient support or a zero baseline means **INCONCLUSIVE**. No
adaptive hard-case additions, threshold refits or old-test tuning can repair it.

Whole linked scene/session/story/asset/multiview families stay together. Allocation
and the executable pipeline must be frozen separately before predictions. Held-out
post-query GT quality checks occur only after all allocated predictions and their
hashes are frozen. A failed quality gate retains the allocation and yields an
inconclusive study; it does not permit replacing families.

## Read the frozen documents together

1. [Design v1](../protocols/0003-motion-design-v1.md) fixes estimands, scores,
   controls, grouping, targets and uncertainty. Input `66e5330`, receipt `adb3cf2`.
2. [Timing addendum](../protocols/0003-timing-addendum-v1.md) clarifies held-out GT
   access and supersedes the earlier quality-check timing. Input `d2787e7`, receipt
   `31e2b0f`.
3. [Resource addendum](../protocols/0003-resource-addendum-v1.md) permits one bounded
   cached annotation audit and complete inventory. Input `227972d`, receipt
   `8e418de`. It leaves the scientific design unchanged.

Historical network and readiness statements in the original frozen documents
describe their execution stage. Later reports supply new observations without
rewriting those inputs. None of these receipts is a final cohort/pipeline freeze.

## Validation and remaining work

Public CI on the reviewed candidate passed **251 tests** and the existing
**1,240-frame synthetic smoke partition**; the latter is software validation,
not new-data inference. [Exact candidate CI](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39076339860).
Independent artifact review checks the published inventory, input/receipt chains,
source/log/exit bindings and preservation of earlier science. The parent reviewer
does not independently re-download or rehash the unpublished raw archive.
[Parent review receipt](../results/0003-publication-review/parent-review.json).
The no-additional-fees statement describes this task's operations; it is not an
audit of all account billing or unrelated activity.

Next, prospectively review support for the observed annotation layout and verify
development geometry without changing the fixed model scores. Establish the
complete family/access graph and sufficient untouched independent families before
any cohort allocation. Assess precision using new development data only, then
freeze the final manifest and runnable pipeline if feasible. The current sample
and headers do not satisfy those conditions. No further decoding, model run or
reader relaxation was performed in this step.
