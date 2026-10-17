# Why the frozen motion residual did not improve this pilot

October 17, 2026. **AI-assisted, post-hoc descriptive analysis of already inspected
evaluation labels. Not a new confirmatory experiment, method or tuned score.**

At fixed matched coverage, adding the residual exchanged 93 native selections for
93 other rows. It removed 2 failures but admitted 5, increasing failures from 4
to 7. **All seven failure-changing rows belong to g-037, point 1**. The newly
admitted failing predictions have smaller local curvature/residual than the
correct predictions displaced by the penalty. This explains the ranking exchange
on these fixed outputs; it does not establish the physical cause of tracking error.

## Prior exposure and fixed diagnostic scope

Before this run, the full evaluation and labels had been reviewed: native 4/1,768
versus augmented 7/1,768, risk delta +0.169683 pp, original paired 95% CI
[-0.157781, +0.712631] pp; operational 8/1,855 versus 9/1,826. The practical
-2 pp criterion failed. The [diagnostic plan](0002-motion-residual-posthoc-plan.md)
was written afterwards, before this diagnostic execution, with fixed questions
and deterministic case rules. It is deliberately not called preregistration.

Plan SHA256 `c56169ddfae3582506f2a1b68ab03977fd79cb640bd498a2a40e601821610572`.
Executed analysis source SHA256 `2c90c8f3a9871cbbd07d3a10f6a7aa2e9c1b7c1c72b1fb25fbb88556b3e91c25`.
Execution main `83a8971ef0abbbfe1d929f4f77bf3a65c4f5bd6c` differs from the later publication head.
The [CLI](../../scripts/analyze_0002_residuals.py) reads the 30 published review CSVs
and original prediction CSVs/receipts, saved evaluation, config and manifest only.
No archive decode, model/torch/numerical package import, weights, inference, frozen
evaluate invocation, threshold fit, scale search or new bootstrap occurred. All
1,066 old tracked results/docs/protocols and all prediction bytes are preserved.

## Matched acceptance: exact exchange accounting

The opportunity mask remains visible and frame >0, with the original common
coordinate/logit availability. Original (score,tie) ranks select 1,768 / 2,209
opportunities for each method; no new target or boundary was chosen.

| Membership | Rows | Correct | >4 px failures |
|---|---:|---:|---:|
| both | 1675 | 1673 | 2 |
| native_only | 93 | 91 | 2 |
| augmented_only | 93 | 88 | 5 |
| neither | 348 | 285 | 63 |

Native: 1,675 + 93 = 1,768 selections and 2 + 2 = 4 failures. Augmented:
1,675 + 93 = 1,768 selections and 2 + 5 = 7 failures. Thus +3 failures = 5
admitted minus 2 removed; 91 correct rows were removed and 88 admitted. All **186
changed decisions**, not just examples, are in [matched exchanges](../../results/0002-residual-posthoc-20261017/analysis/decision-changes-matched.csv).
Complete group and original-point cells, including zero cells, are in the
[group/point table](../../results/0002-residual-posthoc-20261017/analysis/group-point-cells.csv).

The native last accepted score is 0.009586143979; the augmented last accepted
score is 0.058942567309. These are different score scales, not a common fixed
confidence probability. A nonnegative penalty never lowers a row's own score,
but it changes relative rank: displacing high-curvature correct predictions
allows some larger-native-score, smoother erroneous predictions into the fixed
number of accepted slots.

All matched failure exchanges (same group g-037, original point 1):

| Frame | Membership | EPE px | Raw residual d px | Penalty d/8 | Native | Augmented |
|---:|---|---:|---:|---:|---:|---:|
| 46 | native_only | 4.467877 | 0.466707 | 0.058338 | 0.004514 | 0.062853 |
| 48 | native_only | 5.059601 | 0.602723 | 0.075340 | 0.008512 | 0.083852 |
| 49 | augmented_only | 5.676046 | 0.358330 | 0.044791 | 0.011510 | 0.056301 |
| 50 | augmented_only | 6.649333 | 0.230909 | 0.028864 | 0.010743 | 0.039606 |
| 51 | augmented_only | 7.388254 | 0.222766 | 0.027846 | 0.011906 | 0.039751 |
| 52 | augmented_only | 7.836560 | 0.246652 | 0.030831 | 0.014540 | 0.045372 |
| 53 | augmented_only | 8.328853 | 0.085905 | 0.010738 | 0.024037 | 0.034775 |

For frames 49–53, saved EPE rises from 5.676 to 8.329 px while d is only
0.086–0.358 px. These are locally smooth **predicted** positions with large saved
errors, so a constant-velocity consistency signal need not reject them. This
does not identify a swapped object, true acceleration, occlusion event or real
physical mechanism. The EPE/visibility values come from the previously published
frozen row builder, not new independent GT labeling.

## Operational transfer is a different comparison

The existing calibration score/tie thresholds were read unchanged: native score
0.01816350515609092, augmented 0.07856091778632783, with their original full
tie boundaries retained in summary.json and the frozen evaluation result. Nothing
was refit. Both had accepted 913 / 1,141 calibration opportunities.

| Membership | Rows | Correct | >4 px failures |
|---|---:|---:|---:|
| both | 1774 | 1767 | 7 |
| native_only | 81 | 80 | 1 |
| augmented_only | 52 | 50 | 2 |
| neither | 302 | 240 | 62 |

Native selects 1,855 / 2,209 (83.974649% coverage), 8 failures, risk 0.431267%.
Augmented selects 1,826 / 2,209 (82.661838%), 9 failures, risk 0.492881%.
The 81 removed rows include 80 correct and 1 failure; the 52 added rows include
50 correct and 2 failures. Coverage changes by -29 selections; failures by +1.
These unequal coverages must not be treated as the matched comparison above.
All **133 operational decision changes** are in the [separate CSV](../../results/0002-residual-posthoc-20261017/analysis/decision-changes-operational.csv).

All operational failure exchanges are again g-037, point 1:

| Frame | Membership | EPE px | d px | Penalty |
|---:|---|---:|---:|---:|
| 48 | native_only | 5.059601 | 0.602723 | 0.075340 |
| 53 | augmented_only | 8.328853 | 0.085905 | 0.010738 |
| 55 | augmented_only | 9.093622 | 0.116652 | 0.014581 |

## Raw-history and component checks

For each original point, d uses only the two previous finite raw coordinates,
actual frame-index gaps and the current prediction, before appending the current
coordinate. Rejected or GT-occluded predictions and invalid-logit rows are not
removed from coordinate history. The history function cannot read labels or gates.
With consecutive frames this is the norm of the second predicted-position
difference. It describes prediction curvature, not measured physical acceleration.

All **3,840** reconstructed min(d/8,10) increments agree with saved augmented-native
within the fixed absolute tolerance 1e-12, rtol 0; maximum absolute difference
is 8.881784197001252e-16. No scientific score was changed.

| Stratum | Rows | Median native | Median d px | Median penalty |
|---|---:|---:|---:|---:|
| All correct visible opportunities | 2137 | 0.000279 | 0.076487 | 0.009561 |
| All visible failures | 72 | 0.190839 | 0.625082 | 0.078135 |
| Correct removed by matched exchange | 91 | 0.001963 | 0.641349 | 0.080169 |
| Failures admitted by matched exchange | 5 | 0.011906 | 0.230909 | 0.028864 |

The 91 displaced correct rows have median penalty 0.080169 versus median native
score 0.001963. The 5 admitted failures have median penalty only 0.028864. This
is a descriptive contrast conditioned on the selected exchange, not a new test
or an estimate that residual is globally useless: all 72 failures actually have
larger median d than all 2,137 correct opportunities (0.625 versus 0.076 px).
The local ranking at the fixed acceptance boundary is what worsened.

The cap was hit **0 times** across all 3,840 rows. Maximum d is 40.464 px in
calibration and 38.025 px in evaluation, both below the 80 px cap onset. Exactly
120 raw rows have insufficient two-position history and zero penalty (40 calibration,
80 evaluation; frames 0 and 1 across 60 trajectories). Among evaluation opportunities,
40 rows have this initialization zero; all are correct. None of the failure exchanges
is an initialization or cap event. Full native/visibility/local/d/penalty distributions,
including group and point/outcome strata, are in [component distributions](../../results/0002-residual-posthoc-20261017/analysis/component-distributions.json).
These frame counts/quantiles are descriptive; no frame-iid inference is used.

## Deterministic correct-but-penalized cases and exhaustive contexts

The plan selects at most three distinct (group,point) trajectories from correct
matched native-only rows, descending penalty then original tie. Among correct rows, EPE magnitude was not used for ordering or choosing cases.
Correctness was explicitly part of the post-hoc eligibility filter; labels were
already known when the plan was written. The complete changed-row CSVs
remain available.

| Group | Point | Frame | EPE px | d px | Penalty | Native | Augmented |
|---|---:|---:|---:|---:|---:|---:|---:|
| g-005 | 0 | 21 | 0.426864 | 2.204361 | 0.275545 | 0.000610 | 0.276155 |
| g-011 | 0 | 28 | 1.261832 | 2.170471 | 0.271309 | 0.003784 | 0.275093 |
| g-031 | 0 | 18 | 1.862008 | 1.586350 | 0.198294 | 0.007994 | 0.206288 |

These positions are correct under the frozen 4 px endpoint despite larger
prediction curvature. Whether that curvature tracks true changes in object motion,
tracker jitter or something else is a hypothesis requiring other evidence.

All **72 evaluation visible failure rows** are published, not just exchanged
failures. Their +/-2-frame same-trajectory neighbors plus the three planned
correct cases form **115 distinct context rows**, with every center ID recorded
in the [context CSV](../../results/0002-residual-posthoc-20261017/analysis/failure-and-case-context.csv). Neighboring future frames
are displayed only for retrospective description; they never enter the residual.

## Concentration and practical headroom

| Group with any error | Visible opportunities | All failures | Native matched failures | Augmented matched failures |
|---|---:|---:|---:|---:|
| g-031 | 93 | 1 | 0 | 0 |
| g-015 | 51 | 4 | 0 | 0 |
| g-004 | 126 | 1 | 0 | 0 |
| g-011 | 95 | 3 | 0 | 0 |
| g-049 | 88 | 31 | 0 | 0 |
| g-037 | 126 | 19 | 4 | 7 |
| g-027 | 79 | 13 | 0 | 0 |

Errors occur in 7 of 20 groups and 8 of 40 point trajectories. The largest
three groups (g-049:31, g-037:19, g-027:13) contain 63/72 = 87.5% of failures.
All 4 native and 7 augmented matched accepted failures, including the entire
+3 net difference, are g-037 point 1. These are not 7 independent error trials.
Unknown scene relationships and repeated neighboring frames limit inference;
no new subgroup confidence intervals or bootstrap samples were generated.

The observed native matched risk is only 4/1,768 = **0.226244%**. Even a perfect
alternative with zero accepted failures could improve by at most **0.226244 pp**,
not the prespecified 2 pp. The unchanged -2 pp point criterion would require at
least 36 fewer failures, but only 4 were available to remove. This is a post-hoc
arithmetic headroom bound on this observed sample, not an irreducible error floor,
a revised target, evidence of success, or an assertion about a new population.
The original paired CI [-0.157781, +0.712631] pp remains unchanged.

## Reproduction, verification and boundaries

The [execution receipt](../../results/0002-residual-posthoc-20261017/execution/analysis.json) records one analysis child:
elapsed 0.259575421 s, peak RSS 29,960 KiB, parent-observed
exit/raw wait 0/0. Python 3.12.14, CPU reported AMD EPYC 9V74 80-Core Processor.
Two CPU affinity slots, five numeric variables=2, 6 GiB AS, external timeout
900 s, env containing inherited PATH plus PYTHONPATH=src only. Standard-library
fixtures passed; no pytest installation. An independent set/integer-conservation
and second-difference audit checks the new statistics without importing this CLI.
The independent audit passed 3,840 algebraic residual checks, 5,400 descriptive
numeric comparisons and all set/integer/file assertions in 2.224942344 s,
peak RSS 32,992 KiB, actual parent-observed exit/raw wait 0/0. Its actual counts
and tolerances are in [verification.json](../../results/0002-residual-posthoc-20261017/verification.json). The first audit attempt
failed on a blank failure field for an occluded raw row; its source, parent failure
receipt and publicly redacted traceback are preserved. The audit parser was fixed
to retain undefined labels. A second audit passed the numeric checks but its
publication scanner flagged harmless literal patterns in its archived source;
that failed receipt is also preserved. The scanner was corrected to identify
actual host paths/full key headers and made the artifact-map exclusions use full
paths so that both failed attempts remain included. The original diagnostic CLI and its outputs were not rerun or changed; only the
independent audit was corrected and rerun. Parent own exit has
no further supervisor receipt; there is one wait4 layer, not recursive supervision.

All artifacts are under [the diagnostic directory](../../results/0002-residual-posthoc-20261017):
`analysis/` contains exhaustive derived tables and summary; `execution/` contains
actual commands/logs/wait receipts and both preserved failed audit attempts;
`baseline.json` and `plan-source-binding.json` bind old files and pre-run source;
`verification-source.py` and `verification.json` expose the independent checker.
The Work environment had no pytest: zero pytest tests ran locally, while all
11 standard-library unittest fixtures passed. Public exact-head CI is a separate
publication gate. The coordinating session attached
[PR #15](https://github.com/Simonn777/embodied-vision-reliability/pull/15) for review.

## Independent parent review

Two AI-assisted local reviewers checked immutable candidate
`ac4b221dc2dffe783e4504d6448ace3102e6fde1` against the approved input commit.
The [parent receipt](../../results/0002-residual-posthoc-20261017/parent-review.json)
binds all 35 original result artifacts and records the full review scope.
All 1,066 historical files, 44 existing scientific/control files, 29 frozen
inputs, 60 prediction files and 77 formal-evaluation evidence files are unchanged.
All 96 diagnostic input hashes match the approved base and candidate.

An independent reconstruction prepared from the saved inputs matched the cloud
artifacts: all 186 matched exchanges, 133 operational exchanges, 72 failure rows,
three selected cases and 115 context rows. The five CSV exports contain 509 rows
and 15,779 checked fields. All 480 group/point cells, 60 concentration rows and
366 links from 75 centers to context rows were verified. All 5,400 distribution
statistics across 135 strata matched exactly; 12,546 numerical comparisons in
total had zero difference. Reconstructed raw-score arithmetic has at most
8.88e-16 floating difference, with exact native/augmented ranks. These are distinct
checks of arithmetic and stored labels, not independent annotation validation.

Five parent-observed process records cover 11 unittest fixtures, the sole analysis
child, two failed verifier attempts and the successful verifier. All source,
command, resource and published-log bindings agree. Original failed stderr bytes
are not public: reviewers verified the recorded raw-digest references and the
redacted public copies, not the unavailable original text. Plan/source binding
predates the recorded child timestamps but is not a pre-exposure preregistration.
The final verifier's 31 closed artifact hashes plus four current-phase exclusions
cover all 35 original result files; this added receipt does not hash itself.

[Candidate Research checks](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39067668933)
passed 195 tests in 8.75 seconds and 1,240 synthetic smoke frames in 3.90 seconds.
Actual head metadata and logs were inspected. Final reviewed-head and merged-main
CI are recorded on [PR #15](https://github.com/Simonn777/embodied-vision-reliability/pull/15).
Local parent review did not run the candidate diagnostic, frozen evaluator or model.

## Reusing the saved diagnostic

The CLI refuses reused output directories and binds inputs to execution main.
For later reproduction from the repository root, choose unused scratch paths:

```sh
git worktree add --detach scratch/posthoc-inputs 83a8971ef0abbbfe1d929f4f77bf3a65c4f5bd6c
python scripts/analyze_0002_residuals.py --root scratch/posthoc-inputs --output scratch/posthoc-reproduction
```

These commands describe a future optional diagnostic reproduction, not an
additional run performed during this review. No model/cache/archive dependency
is needed. Plan, source,
config, protocol, input/file hashes, exact commands, stdout/stderr and resource
records are retained. Frozen source/config/protocol/thresholds, original 3,840 rows,
60 prediction files and previous results remain byte-identical. No raw videos, model weights or private host paths are published. The cloud analysis required no new data/model/package downloads or installations;
ordinary Git and CI dependency traffic is separate. No paid APIs, GPU services or
credits, or account/permission/schedule changes were used.

One follow-up study is warranted, not executed: on genuinely new independently
audited capture groups with GT trajectories and object identity, freeze this same
residual and predeclare stable versus changing true-motion regimes before error
access. Test whether correct motion changes produce high residuals while smooth
localization drift produces low residuals, using fixed coordinates and matched
coverage. This can distinguish a curvature confound from error sensitivity; it
requires adequate new error headroom and a justified prospective design, never
a scale search or retuned success claim on these already-inspected videos.

This remains exploratory, with shared objects/camera, unknown scene dependence
and upstream checkpoint exposure. Native confidence is itself temporal; the strict
single-image question remains unresolved. No SOTA, confirmatory, real-robot or
paper-reproduction claim. All content and verification are AI-assisted.
