# Current state

Updated: 2026-10-18. Program day 3. Phase: 0004 Stage 0 completed and independently reviewed; its frozen continuation rule failed. The broader tracking-improvement objective remains active; no manuscript or demonstrated general improvement.

## Active direction: improve tracking, with evidence before a paper

The owner requested meaningful tracking improvement before any paper. The first
candidate's [frozen 250-frame/eight-query test](protocols/0004-state-intervention-stage0.md)
has completed on cached synthetic development video 0. Zeroing short-term TAPIR
context repaired 46 visible errors but created 120; reset-once shadow return was
negative (−26/448), so the fixed decision is **`stop_this_candidate`**. All 4,488
prediction rows and 13 equality controls are complete. The CPU child took
227.699 s, peak RSS 756,272 KiB, exit 0. [Full result](results/0004-stage0/README.md).

[Independent parent review](results/0004-stage0/parent-review/README.md) verifies
the saved arithmetic and provenance, with explicit limits on what was rechecked.
The original proposal is retained as a failed development hypothesis, not a
positive method result. No selector is being trained to rescue these selected
episodes. Visibility-only gains must not be called coordinate recovery.

Next, establish a stronger native causal baseline's CPU feasibility and a usable
independent real-video development/evaluation path before freezing a different
mechanism. A read-only TAPNext++ environment preflight is pending; no new model
weights or real-video shard have been acquired. The substantive improvement and
paper gates in the [direction](docs/0004-research-direction.md) remain unchanged.
The old 0002 inspected evaluation remains excluded.
0003 is deferred with the missing-cache/one-shot limitations below preserved; this
new direction does not authorize retries or threshold changes to that operation.

## Historical 0003 blocker and original question

Current blocker: [restored-shell cache inventory](docs/0003-restored-shell-cache-inventory-2026-10-18.md).
The original Work entry now runs shell commands, independently inspected by the
parent. The exact PointOdyssey cache directory is missing from that visible view;
a bounded metadata search of two authorized project roots found no required NPZ,
original markers or recovery candidate (1,302 entries / 278 directories).
This does not prove global deletion or reconcile cross-workspace one-shot state.
No geometry was attempted in this follow-up. Recover the existing cache and
markers from a supported view/backup before continuation; preserve the freeze and
old failures without redownload or marker reset. The earlier offline report and
publication review remain historical evidence, unchanged.

For a fixed visual position estimator, does causal temporal consistency identify large localization errors under held-out occlusion/appearance shifts better than confidence from the current image alone, at comparable accepted-prediction coverage?

## Status

- The [research charter](docs/RESEARCH_CHARTER.md) defines the year and twelve-week milestones.
- The first experiment is a controlled synthetic 2D localization diagnostic. It tests the measurement pipeline; it cannot establish real-robot performance or methodological novelty.
- Experiment 0001 completed: 160 base sequences / 19,840 scored rows overall; 40 independent test sequences / 4,960 test rows. Protocol `a5ab4e3`, code `bc9921f`; source unchanged at execution. See the [day-one note](notes/2026-10-16.md) and [artifacts](results/0001-pilot/summary.json).
- The temporal heuristic did not resolve a constructed stable wrong-identity failure. Under occlusion it reduced coverage from 42.10% to 37.02%, with zero observed selected failures for both gates. No general efficacy claim follows.
- A [post-hoc analysis of the saved rows](docs/analysis/0001-failure-mechanism.md) now identifies the mechanism: two rejected wrong positions fill the raw history, then the residual becomes zero; the separately calibrated temporal threshold accepts all 880 wrong positions in frames 10–31. Correct turn/recovery positions are also rejected. This separates score-memory and threshold effects from color-estimator and renderer limitations; it is not a new held-out evaluation.
- The follow-up analysis passed 107 repository tests, independently checked counts, and a six-file byte-for-byte replay. It only reads frozen measurements; no original experiment was repeated. [Analysis verification](results/0001-failure-analysis/verification.json).
- 17 tests and GitHub Actions passed. A separate full rerun matched six deterministic artifacts byte for byte; independent metric recomputation matched all 112 interval endpoints. See [reproduction.json](results/0001-pilot/reproduction.json). This is same-environment verification, not independent-data validation.
- Eleven primary-paper entries and a pinned data-source/size audit are available. No listed paper's published result has been reproduced.
- [Methods and code reading](docs/literature/2026-10-16-confidence-methods.md) distinguishes native position-error confidence from visibility and identifies temporal/future context that must be audited before the external-data protocol.
- [Dataset/model audit](docs/data-model-audit-2026-10-16.md) conditionally selects TAP-Vid RGB-Stacking and Online BootsTAPIR at pinned source revision `730cda1c`. Fresh HEAD metadata identifies 406,178,609 combined compressed bytes. Rights, coordinates and temporal score context were audited before acquisition. [Development CPU feasibility](docs/0002-cpu-feasibility.md) now loads real official weights with patched torch 2.10.0+cpu, validates source/data identity, and runs only video index 0: one query/16 frames took 8.276s, main process peak RSS 746,092 KiB. Prefix, late-arrival, real single/batch and independent-process checks pass at fixed atol 1e-4. Raw outputs, prior setup/provenance issues and serial repairs are retained. No calibration/evaluation video was inferred; the historical 16-frame stage is now followed by the bounded 64-frame stage below.
- [Independent artifact review](results/0002-development/parent-review.json) verifies the published CPU run's 20 artifact hashes, source/config/input bindings, raw comparisons and latency accounting. This verifies saved evidence; it does not add an accuracy result or authorize the unfrozen evaluation draft.
- Hardware checked: Apple M5, 24 GiB memory; CPU and MPS available, CUDA unavailable. Initial runs use CPU.
- The retained [Codex API workflow](docs/CLOUD_EXECUTION.md) remains disabled under the owner's **no-additional-fees** constraint. Its [cloud preflight](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/38855628928) passed without a model call or GitHub publication; this was infrastructure verification, not autonomous research.
- A subscription-based cloud Scheduled trial ran on October 16, but could not clone the repository or create a branch through the GitHub integration. That failed trial remains recorded in the [trial record](docs/CLOUD_EXECUTION.md#subscription-based-cloud-trial).
- Interactive ChatGPT Work execution and publication to main are verified: [capability PR #1](https://github.com/Simonn777/embodied-vision-reliability/pull/1) merged as `d910736e` after [final CI 38859145851](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/38859145851) passed on reviewed head `835cbc9a`. Work shell and GitHub CI measurements are distinguished in the [capability evidence](docs/CLOUD_CAPABILITY.md), including a clarification of the raw shallow-history protocol field.
- [Non-interactive Work execution and publication are verified](docs/SCHEDULED_VERIFICATION.md): the saved automation's Run now trial passed 107 shell tests and a 1,240-frame smoke, then merged [PR #2](https://github.com/Simonn777/embodied-vision-reliability/pull/2) as `9d415308` after exact-head CI/log review. The settings session subsequently paused the local heartbeat; one Work cloud schedule remains enabled, daily 09:00 America/New_York through October 15, 2027. Next regular run: October 17, 2026. No additional paid service or usage-credit purchase is authorized; future clock-triggered runs and uninterrupted availability are not yet established.

- [64-frame development and metadata grouping](docs/0002-64frame-and-freeze.md) completed on real weights: two-query64-frame forward39.337s, fresh-process repeat39.450s, maximum replay difference0, peak794764KiB. Only video0 was inferred; all50 identities/prefix/frame hashes yielded50singleton exact-duplicate groups and partitions1/10/20/19. Scene independence and full upstream non-exposure remain unestablished. Complete executable 0002 metrics and future inference use the v2 freeze after the pre-merge provenance repair; v1 is preserved but superseded before any evaluation. V2 enforces canonical published receipts and mandatory commit/current-file hashes and dependencies; [Independent review](docs/0002-independent-review.md) verified the artifacts and v2 repair; merge and main-publication verification remain mandatory before held-out inference. No real-data score comparison has run.

- [Calibration group g-033 completed after authorized runtime recovery](docs/0002-g033-calibration-inference-2026-10-17.md). The initial NumPy2.3.5 gate failure remains preserved in its own checkpoint. Official NumPy2.0.2 was installed into cache; the newly recorded wheel/RECORD identities establish version-matched reconstruction, not byte-identical reproduction. The real gate passed all29 frozen inputs and16 exact versions/module origins. Video33 first64frames, points[0,1]:128rows/512finite outputs, outer wall49.287s, peak2553156KiB; independent raw/grid/latency/Git/provenance checks passed. **1/10 calibration inference groups complete,0/20 evaluation groups complete.** No GT error, confidence score, threshold fit, other group or evaluation was run. The [independent publication review](docs/0002-g033-publication-review.md) passed. Preserve and use existing g-033 outputs; do not re-infer.


- [Calibration group g-013 completed and independently reviewed](docs/0002-g013-calibration-inference-2026-10-17.md) from main `5561cf5e`, reusing the verified cached environment with no downloads. The real gate passed all 29 frozen files and 16 exact dependency versions. Video 13's first 64 frames and original point indices [0,1] yielded 128 rows / 512 finite outputs. The inference subprocess took 49.411 seconds, with peak RSS 2,553,240 KiB; its exit and the separately parent-observed wrapper exit were both 0. Two independent artifact reviews passed, and all 37 g-033 result files remain unchanged. **2/10 calibration inference groups complete, 0/20 evaluation**; no errors, scores, thresholds or replay. The next allocated group, g-022, was not run.

- [Calibration group g-022 completed and independently reviewed](docs/0002-g022-calibration-inference-2026-10-17.md) from execution main `767aa160`. The real gate passed all 29 frozen files and 16 exact cached dependency versions, with no downloads or frozen changes. Video 22's first 64 frames selected point indices [0, 1]: 128 rows and 512 finite scalars. Inference child wall time was 57.829 seconds and peak RSS was 2,553,116 KiB; child and parent-observed wrapper exits were both 0. Two independent artifact audits passed, including 65 byte-preserved g-033/g-013 result/review files. A post-run verifier's raw-versus-normalized RECORD mismatch was diagnosed and preserved; matching text semantics passed without changing runtime or restarting inference. The report distinguishes the diagnostic's claimed failure exit from a separate raw process observation, which was not retained. **3/10 calibration inference groups complete, 0/20 evaluation**; no GT errors, scores or threshold fitting. The next allocated group, g-040, was not run.

- [Calibration group g-040 completed and independently reviewed](docs/0002-g040-calibration-inference-2026-10-17.md) once from execution main `3c6b26ac`. Real gates passed all 29 frozen files and 16 exact cached dependency versions; no download or frozen change. Video 40's first 64 frames selected original indices [0, 1], saving 128 rows and 512 finite scalars. Inference child wall time was 47.834 seconds and peak RSS 2,553,160 KiB; child and separately parent-observed wrapper exits were 0. Two independent artifact audits passed with 93 prior result/report/review files byte-preserved. Post-run checks now also have verified parent-observed exits and raw logs. **4/10 calibration inference, 0/20 evaluation**; no GT errors, scores, threshold fitting or replay. Next allocation g-039 was not run.

- [Calibration group g-039 completed and independently reviewed](docs/0002-g039-calibration-inference-2026-10-17.md) once from execution main `1f4d14f8`. Real gates passed all 29 frozen files and 16 exact cached dependency versions; no package/data/weight downloads or frozen changes. Video 39's first 64 frames selected indices [0, 1]: 128 rows and 512 finite scalars. Inference child wall time was 48.157 seconds, peak RSS 2,553,996 KiB; child and parent-observed wrapper exits were 0. Two independent artifact audits passed with 130 prior files byte-preserved; post-run source/log bindings and observed child exits matched without circular hashes. **5/10 calibration inference, 0/20 evaluation**; no GT errors, scores, threshold fitting or replay. Next allocation g-020 was not run.

- [Calibration group g-020 completed and independently reviewed](docs/0002-g020-calibration-inference-2026-10-17.md) once from execution main `69737174`. Real gates passed all 29 frozen files and 16 exact cached dependency versions; no downloads or frozen changes. Video 20's first 64 frames selected indices [0, 1]: 128 rows and 512 finite scalars. Inference child wall time was 54.620 seconds, peak RSS 2,553,192 KiB; child and separately parent-observed wrapper exits were 0. Two independent artifact audits passed with 167 prior files byte-preserved and post-run source/log bindings checked without cycles. Post-run child exits are observed; their parent program has no further supervisor exit receipt. **6/10 calibration inference, 0/20 evaluation**; no real-data errors, scores, threshold fitting or replay. Next allocation g-016 was not run.

- [Calibration collection complete](docs/0002-calibration-complete-2026-10-17.md): g-016, g-032, g-035 and g-026 ran serially once from execution main `d6eb1518`, each after real 29-file/16-version/cache gates and before the next group's launch only after artifact verification. Prior six outputs were reused; all 204 historical files remain unchanged. One shared toy smoke/pytest availability probe served the unchanged software. Raw CSVs, per-group independent checks and child/wrapper/post-run source/log evidence are retained. **10/10 calibration, 0/20 evaluation predictions; only 10/30 prescribed prediction sets exist.** No GT errors, scores, threshold fit, final evaluation or method efficacy claim. Two independent candidate audits passed: all 1,280 rows / 5,120 finite scalars and 127 original batch result files matched, with no missing/nonfinite outputs. The [parent review](results/0002-calibration-completion/parent-review.json) records evidence and limits; next stage remains 20 frozen evaluation-group inference steps, not `evaluate`.

- [All frozen predictions collected](docs/0002-prediction-collection-complete-2026-10-17.md): all 20 evaluation groups ran serially once from execution main `6bdce825`, after real 29-file/16-version/cache gates, each independently verified before the next launch. All 10 existing calibration outputs were reused; 422 historical result/doc/protocol files remain unchanged. One shared software probe/toy smoke served the batch. **10/10 calibration plus 20/20 evaluation prediction sets, 30/30 total; no GT errors, scores, threshold fit, method comparison or evaluator invocation.** Two independent artifact audits and the coordinating publication review passed, including all 564 original batch result hashes, complete grids and 42 post-run child records; see the [parent review](results/0002-evaluation-completion/parent-review.json). A later separately authorized stage may run the complete frozen evaluator after review/merge; no method efficacy or robot result follows from collection.

- [Frozen v2 evaluation measured and independently recomputed](docs/0002-frozen-evaluation-v2-2026-10-17.md): one evaluator invocation on all saved predictions from execution main `43b64a39`, no model/inference. Calibration 1,141 visible post-query opportunities; evaluation 2,209. Matched 80% accepts 1,768: native 4 failures (0.226244%), augmented 7 (0.395928%); delta +0.169683 pp, 95% paired group bootstrap CI [-0.157781, +0.712631] pp, 2,000/2,000 defined. Practical signal false. All controls, operational policies and diagnostics are retained. The cloud CSV/math checker matched 26,566 numeric leaves and all 23,040 score-entry comparisons (19,200 finite non-oracle; 3,840 oracle including nulls). Independent local review matched all official numeric results and 6,000 bootstrap effect values exactly; maximum raw-history score difference was 8.88e-16, with all 12 ranking orders identical. All 988 historical files/60 prediction files remain unchanged. Evaluation labels are now inspected, never again described as untouched. Native confidence is temporal, so the strict single-image question remains unresolved. Two independent parent artifact reviews passed; see the [parent receipt](results/0002-frozen-evaluation-v2-audit-20261017/parent-review.json). Next is an explicitly post-hoc diagnosis or properly new frozen design/data, not tuning this test.

- [Post-hoc motion-residual diagnosis](docs/analysis/0002-motion-residual-posthoc.md) explains the fixed-output exchange: matched methods share 1,675 selections and swap 93 each; removing 2 failures but adding 5 yields the +3, entirely g-037 point 1. Large prediction curvature penalizes 91 correct native-only rows, while 5 smoother erroneous predictions enter. Operational thresholds remain separate: 81 out/52 in, failures 1 out/2 in, coverage -29. All 3,840 residual increments match within 1e-12; no cap events; initialization zeros do not explain the failure exchanges. The observed 4/1,768 native baseline permits at most 0.226244 pp improvement, below the unchanged 2 pp target: this is a post-hoc headroom bound, not a success or new criterion. All prior files, frozen inputs and confidence intervals remain unchanged. No model inference, archive decode, frozen evaluator invocation, threshold refit, scale search or new bootstrap. Two independent parent artifact reviews passed; the [parent receipt](results/0002-residual-posthoc-20261017/parent-review.json) records the 35-file evidence map, numerical checks and limits. Next requires separately authorized research on genuinely new groups or a justified new frozen design, not tuning this inspected test.

## Claims boundary

No accepted paper, new state of the art, real-robot validation, or successful research hypothesis is claimed. Bibliographic verification and reading depth are tracked separately. See [contributions](docs/CONTRIBUTIONS.md).

## Continuation

Read [NEXT_ACTIONS.md](NEXT_ACTIONS.md) and the active protocol before running work. Check Git status and GitHub before assuming the local state is current. Do not rerun a final-test comparison to tune a method.

Scheduled research ends on October 15, 2027 (America/New_York), unless the user renews the program.


## 0003 design / acquisition readiness (2026-10-17)

AI-assisted [new-data audit](docs/0003-data-audit-2026-10-17.md) and
[design protocol](protocols/0003-motion-design-v1.md). Input commit
`66e5330ad7404b4e7678d006e2537e4703f4683d` was published before the separate canonical
nine-file design receipt. DESIGN_ONLY; BLOCKED_NO_FINAL_COHORT, not a final test
freeze or inference authorization. All 11 direct source/sample HEAD requests
failed DNS, 0 bytes; no sample GET, archive inventory, annotation/RGB/GT access,
model, old-result analysis, threshold fit or formal effects. Official source
text checked through cloud search; exact archive/source revisions are byte-unverified
leads. All 1,149 protected old paths remain byte-identical.

Design separates signed 2D GT-image and localization-error curvature. Norms and
perspective camera/world components do not add; 2D predictions cannot identify
3D drift. New prospective continuation goal is 20% relative error reduction with
paired cluster 95% CI upper <0, minimum 20 independent test families, 5 native-error-
bearing families, 20 native matched errors, valid GT/coverage/bootstrap support.
These are planning choices, not estimated efficacy. No actual cohort exists.
Thirty-five helper/freeze/adversarial stdlib fixtures passed, including pre-freeze
review domain/overflow/path refinements. Earlier software snapshots and evidence
closure correction remain archived. Old 0002 science/results remain unchanged.

Next: permitted sample/source acquisition; one quarantined family's geometry,
units and time validation; complete family/access graph; new-development-only
precision assessment; then separate final cohort freeze if gates can be met.
Unknown independence/checkpoint exposure stay unknown. No added fee, install,
account/schedule/permission change or confirmatory/robot/SOTA claim.

## 0003 default-route and prospective timing follow-up (2026-10-17)

The [follow-up report](docs/0003-network-and-timing-followup-2026-10-17.md) supersedes
only the current network-readiness interpretation, preserving every v1 byte.
V1 clean-env DNS failures did not prove default runtime unavailability: existing
proxy/CA names were present, and one unchanged-default official HF request returned
HTTP 200 / 2,328 bytes. No values/credentials/private hostnames or security changes.
Exactly one authorized sample GET downloaded 3,324,284,510 bytes and verified full
SHA256 against the pinned official LFS identity. Parent wall 82.836491 s, RSS
18,092 KiB, exit/raw 0/0. Six bounded pinned source files/tree were checked as code
text only; no remote code executed. Conservative cumulative ceiling 4,823,387,101
bytes remains below 10 GiB; no fee/installation/dependency/weight acquisition.

Safe tar inventory STOPPED after 2,756 headers when an annotation exceeded unchanged
512 MiB cap: actual exit/raw 1/256, 2.904198 s, 14,080 KiB. Inventory incomplete;
no NPZ/GT arrays, frame-0 query fields, RGB or predictions accessed. No relaxed cap,
retry, re-download, extraction, geometry computation or model. Entire sample stays
audit/development UNKNOWN family; no independent evaluation cohort established.

The [prospective timing addendum](protocols/0003-timing-addendum-v1.md) input commit
`d2787e7f4fefaca6fbcc6ffb4a1a2d1686e97647` precedes a separate eight-file receipt
including the unchanged canonical v1 receipt. It supersedes GT-quality timing only:
metadata/quarantined development/frame0 query access (truthfully recorded) before
final allocation/pipeline freeze; all prediction hashes before held-out GT quality
adjudication. Pre-inference quality remains PENDING_POST_PREDICTION. Required-family
quality failure/undefined => INCONCLUSIVE, retaining all families/zero-query units
and original primary definition. V1 readiness() is legacy with no execution authority;
a future separately frozen pipeline must bind both versions and phases. Eleven new
typed phase fixtures passed. No old scales/20% goal/samples/results changed.

Parent provenance clarification: earliest 27 fixtures are not fully source-bound;
29/31 records do not preserve exact old protocol text. Final 35 nine-file hashes match
published input 66e5330. Old records remain historical, not retroactively repaired.
Next is independent review and explicit resolution of archive-cap/provenance evidence,
not sample retry, cohort invention, GT decoding or inference under this receipt.

## 0003 final bounded cached-resource audit (2026-10-17)

[Actual resource audit](docs/0003-cached-resource-audit-2026-10-17.md) follows a
published input 227972d3 and separate resource receipt 8e418de4 BEFORE any numeric
GT decode. V1/timing science unchanged. Complete safe inventory: 14,415 members,
3,334,075,426 declared bytes; original ordinal 2,756 annotation is now newly observed
as sample/scene_d78_0318_3rd/anno.npz, 705,904,917 bytes. Whole sample remains unknown-
family development, not an independent held-out cohort. No new archive GET.

One fixed first-scene annotation sample/r4_new_f/anno.npz saved exclusively to cache:
98,527,880 bytes, SHA256 382d1edca0d6d88279ed1a61ee63c9ba27b7c5418f2269f4592c1658fd7d8e77.
Reader stopped at Fortran-order valids; header-only diagnosis also finds Fortran
visibs, while track/camera arrays are C order. No layout relaxation, GT numeric
payload, selected point IDs, RGB or geometry checks. Max/RMS error undefined, not
zero. Combined metadata/buffering charge 590,976 bytes within 64 MiB. Resource child
21.082816 s, 29,820 KiB, real exit/raw 1/256, preserved. Ten bounded-reader software
fixtures passed; a test-builder failure/fix is retained. No model/old-test/score
work, downloads/install/fee/permission/schedule change. Last resource expansion for
this task is complete; actual geometry, units/FPS and independence stay unverified.

Independent science, source and provenance reviews passed for candidate b605ae13;
the [readiness guide](docs/0003-review-and-readiness.md) and
[parent receipt](results/0003-publication-review/parent-review.json) distinguish
publication readiness from the still-blocked geometry and independent cohort.
Candidate CI passed 251 tests and the existing 1,240-frame synthetic smoke.

Next: prospectively review support for the observed annotation layout, no further
numeric access or resource relaxation this task. A later prospectively frozen
pipeline/cohort must bind v1+timing+applicable resource decisions; no final cohort
or inference authority exists. Do not mistake complete filenames for independent
families or metadata/source-convention checks for numeric GT validation.

## 0003 Fortran stage A candidate (2026-10-17)

[Prospective revision](docs/0003-fortran-stageA-2026-10-17.md) adds versioned C/F
streaming and storage-aware geometry helpers without editing old frozen inputs.
Cached archive/one NPZ raw hashes match; no numeric GT, point selection, geometry,
RGB/model or old-test access. Seventy-five stdlib fixtures (19 new) and the existing
1,240-frame synthetic smoke pass under two slots / 6 GiB AS / timeout900. Pytest
unavailable locally (0), no installation/download/fee/schedule/permission change.
Carry prior 590,976 bytes into the same 64 MiB audit budget. Sole saved-RT then
camera-z normalization and K convention is checked from pinned cached source.
Candidate protocol/config/reader/runner/tests and nine mandatory bindings await
review; NO v2 receipt exists. Do not decode real GT before approval and a separately
published canonical receipt. Independent cohort, units/FPS and actual geometry
remain blocked/unknown; whole sample stays development.

## 0003 frozen Fortran stage B: preserved geometry failure

[Actual one-shot audit](docs/0003-fortran-stageB-2026-10-17.md) followed approved
input 93be3c95 and separately published receipt/execution commit 384489d9, canonical
receipt SHA256 ad6250bb93453b0ff8f0e9ef4e25b853cac99eb37f910ea007fa389f285f2da1.
Fresh-fetch real gate passed nine inputs and all three prior maps (23 distinct
inputs), then the unchanged runner reacquired locks/unique marker. No retries.
All six C/F numeric prefixes read: 246,216 retained scalars, 20,604,126 numeric
span bytes including 19,830,240 discarded gap bytes. Cumulative charge 22,778,336
bytes includes 590,976 carried bytes and all metadata/header/directory reserves.
Four fixed metadata members completed; cross-container identity stays UNKNOWN.
Frame 0-visible finite original IDs [0,1] selected; first attempted frame 0 / point 0
failed the proper-rotation gate before projection. Planned 6 / attempted 1 /
unattempted 5; validity 1/visibility 1, projection_consistent null, valid-visible
support false, finite-error count 0, max/RMS null. No tolerance/axis/convention
change or extra GT read. Child wall 13.159663 s, RSS 36,288 KiB, exit/raw 1/256.
Independent saved-record byte/budget review passed and 1,438 pre-execution files
remain unchanged; 76 synthetic stdlib tests pass, local pytest absent / 0. No
download/install/re-extraction/RGB/model/old-test analysis or additional fees.
Whole sample remains development, 0 established independent held-out families;
actual geometry, units/FPS/checkpoint exposure remain unverified. Next: independent
review of the camera-gate failure and missing quantitative diagnostic evidence,
not automatic rerun or post-outcome tolerance fitting. Final cohort remains blocked.


Independent publication review of the frozen Fortran audit is complete; see the
[review](docs/0003-fortran-independent-review.md) and
[parent receipt](results/0003-fortran-publication-review/parent-review.json).
Three separate reviews found no publication blockers. This approves the evidence
of a failed geometry audit, not geometry itself. The offending matrix/statistics
were not saved, so the cause remains unresolved. Candidate CI passed 271 tests
and the 1,240-frame synthetic smoke; final publication/main checks remain required.

## Rotation-failure diagnostic: Stage A only

[Review candidate](docs/0003-rotation-diagnostic-stageA-2026-10-17.md) proposes
one new frame0-only K9+RT16 diagnostic, preserving all previous gates/failures
and marker. No additional real camera values decoded in this Stage A, new receipt
or Stage B command. New
versioned source uses exact Fraction verification, original tau and carried
22,778,336-byte budget; no projection/repair/convention search. Cached-source
float16 save path is a possible mechanism only, not actual-scene causal proof.
The narrow single-final-f32 hypothesis has fixed proved rational exclusion bounds,
not a replacement gate. Proposed 25-value excerpt retains CC BY-NC-SA 4.0 data
attribution/license, separate from original MIT software.
All 93 stdlib tests (17 new) pass and 30 old frozen inputs verify. Baseline 1,468
protected files/old marker preserved; no download/install/model/old-test tuning
or extra fees/schedule/permission changes. Candidate must be independently reviewed
before a separate 12-input receipt and any real camera access.

Pre-freeze revision adds exact binary16 representability for those same future
25 values, separated into K9/R9/RT16/translation3/homogeneous4. This describes
compatibility with a direct half-to-f32 path, never proves generation history or
repairs the matrix. The source-bound 19-case targeted synthetic suite passed;
previous evidence remains unchanged. Still awaiting independent review; no new
receipt or additional real camera decode.


## Rotation-failure diagnostic: Stage B recorded, original gate failed

[Actual report](docs/0003-rotation-diagnostic-stageB-2026-10-17.md): approved input
8a7591b, separate receipt/execution 8ffb8bc, complete published-chain preflight then
one cache-only frame0 K9+RT16 diagnostic. All three unchanged rotation checks fail:
Gram defects 0.0003642439842224121 / 0.0003985762596094844, determinant defect
0.000009985611434615294, versus original tau 9.5367431640625e-7. Diagnostic exit0
means recording complete, not geometry passed. Wall 0.466789 s / RSS 16,128 KiB;
25 scalars, 100 payload bytes; total carried charge 22,910,156 bytes. No projection.

Single-final-f32 rounding of exact SO(3) is excluded by the predeclared rational
bounds. R9 and K9 are all exactly binary16-representable; RT14/16, translation1/3,
homogeneous4/4. This is descriptive compatibility, not proof of generation history;
two translation values exclude only a direct all-values half-to-f32 path without
intermediate arithmetic. Cached source's half cast is a possible mechanism only.
Independent saved-excerpt Gaussian/Fraction and dyadic-lattice checks pass without
new cache access. Exact 25-value excerpt/derived diagnostics retain CC BY-NC-SA 4.0.
Old files/failures/markers preserved; new marker retained. No model, old-test tuning,
threshold change, download/install, new fees or schedule/permission changes.
Next: independent review of the recorded diagnostic; actual generation chain and
independent cohort remain unresolved. No automatic matrix repair or audit rerun.


## Rotation diagnostic independent publication review completed

[Review](docs/0003-rotation-independent-review.md) approves candidate bb5fe92 for
publication of the bounded failure diagnostic. Independent exact arithmetic over
only the published 25 values confirms all 18 Gram entries, determinant, unchanged
failed gate and half-grid counts; no cache or model was rerun. Parent review
closes all 19 Stage B artifacts and confirms 1,512 protected execution-baseline
files unchanged, including 1,390 result files and 37 Stage A artifacts. Candidate
CI passed 290 tests and the existing 1,240-frame synthetic smoke. The local
saved-record check (0.081646 s) is distinct from the cloud diagnostic (0.466789 s).

Numerical trigger established; actual generation root cause remains unknown.
Next is a version-specific provenance review beginning with cached source and
published records. Any new geometry/data-access proposal requires a prospective
freeze; original tau, failures and null projection errors stay unchanged. No
independent cohort, geometry pass, model improvement or additional fee follows.


## Cached conversion history: Stage A complete, next candidate unfrozen

[Source audit and candidate](docs/0003-source-history-stageA-2026-10-17.md) used only
five cached bodies plus source-tree text (39,425 bytes) and published JSON. Five
old freeze chains / 38 inputs verify. Source audit 0.447624 s, RSS 12,416 KiB,
exit/raw 0/0; three synthetic audit fixtures pass. No archive/NPZ/metadata/RGB,
new numeric camera/GT read, projection, model, old-test analysis or downloads.

Tracking already writes intrinsics/extrinsics keys with f16 cameras to
annotations.npz. Reprojection's annot.npz reader widens in memory but does not
save a f32 archive. Selected sample anno.npz/six keys/f32 dtype lack a verified
rename/repack/precision conversion run. Only 5/38 listed bodies are cached and
0/39 upstream Git objects available: no upstream version-history audit possible.
R half-grid membership plus this code is compatibility, not confirmed causation.

Unfrozen candidate reuses frames [0,1,2], allocated IDs [0,1], at most six
annotation self-consistency checks. It explicitly preserves original rigidity
failure/tau and distinguishes algebraic agreement from valid rigid geometry.
Numeric ledger stays 22,910,156 bytes; whole sample remains development/cohort
blocked. Independent candidate review and complete versioned executable freeze
are required before any new data/geometry. No receipt or Stage B; old bytes and
markers preserved. Parent's planned compatibility certificate was not executed.


### Prepublished saved-R analysis: inconclusive; candidate revised, still unfrozen

Parent method c1bcc597 (four byte-identical files) was verified and included before
execution at 565c1aa4. Exactly one fixed z-branch quaternion proposal passed exact
SO(3), but 8/9 half cells passed: [0,1] failed. Outcome INCONCLUSIVE, not proof of
infeasibility or a historical source mechanism. No second proposal/search/repair.
Real saved-R-only child: 0.034895 s, RSS 11,392 KiB, exit/raw 0/0; separate exact
saved-output Gaussian/cell audit passed. No archive/cache/GT/projection/model read.
The previous paragraph's pending-analysis status is historical and superseded.

The new report explicitly corrects earlier camera-key shorthand without changing
old bytes: source already writes intrinsics/extrinsics; actual filename, other
fields, disk dtype and generation/version linkage remain unresolved. Candidate
geometry output scope is narrowed to derived errors/status/hash/count, no new
camera or original point values; prior 25-value exception is not expanded. Same
fixed six potential checks, original tau/failure and budget remain unchanged.
Wait for independent candidate review; no new geometry receipt or execution.


### Complete prospective geometry implementation: synthetic validation passed

[Updated full report](docs/0003-source-history-and-geometry-plan-2026-10-17.md)
retains the original source report/candidate files as immutable history and records
the key-name correction plus one INCONCLUSIVE saved-R proposal. New formal
annotation-consistency-v1 protocol/config/helper/gated runner/bounded parent/tests
are implemented. Fourteen concentrated synthetic tests pass: actual 0.312825 s,
16,752 KiB, exit/raw 0/0, with six tested-source snapshots. Fixed six potential
checks retain original rigid failure and distinguish annotation support from rigid
qualification. Full sparse spans 470,884 bytes, retained 432 bytes / 117 scalars;
carry remains 22,910,156. Only derived errors/status/rigidstats/counts/hashes may be
published, no new camera/point originals, projected uv or depth. Complete exact
mandatory receipt gate includes source/analysis evidence and all prior38 inputs.
No geometry receipt or real data execution. Parent review is next, then a separate
canonical receipt only on approval. Original source/thresholds/results/markers
preserved; no costs, downloads, installs or permission/schedule changes.


Final synthetic revision: 15 tests now pass at attempt2 (0.311270 s, 16,128 KiB,
exit/raw 0/0). A last-resort uncaught-stop label now reports access unknown and
INCOMPLETE rather than implying pre-access status; a subprocess fixture covers it.
Attempt1's 14 passing source/log snapshots remain historical. No real dataset or
saved-R calculation repeated. Review the latest v2 closure/map for final sources.


### Pre-freeze initializer binding correction

[Review correction](docs/0003-initializer-binding-review-fix-2026-10-17.md): formal
new runner now explicitly imports/binds evr.__init__ source path and digest. The
initializer itself is unchanged; mandatory map grows 44 to 45, old chain stays 38.
Only the unfrozen new runner/tests changed; previous code/attempts and actual half
results preserved. Attempt3 passed 17 synthetic tests, 0.394826 s / 16,128 KiB /
exit/raw 0/0. New v3 review map closes these sources; older maps are historical.
No geometry receipt/marker/run, dataset access or saved-R rerun. Parent review and
explicit receipt approval remain required; numerical method/scope unchanged.

## Source history review complete; next descriptive operation frozen

[Final review](docs/0003-source-history-independent-review.md) records the corrected
camera-key interpretation, missing archive-conversion history and independently
verified INCONCLUSIVE saved-R result (one proposal, 8/9 cells, no repeat). Actual
float16 causation remains unestablished. The original tau and failures are intact.

Input 84dd6aa passed independent review after adding the package initializer to
the 45-file binding. Receipt-only commit 48ff18f froze that exact input. The actual
publication gate passed 45 inputs and 5 historical chains/38 inputs without any
cache/data/geometry call. Candidate CI passed 320 tests / 4 subtests and 1,240 synthetic
frames. All historical results/snapshots remain unchanged, notes append-only.

Next: verify merged main and the canonical annotation-consistency receipt before
one bounded descriptive operation on fixed frames [0,1,2] / IDs [0,1]. Preserve original
rigid failure separately from annotation agreement and the unchanged 0.5 px rule.
This turn does not execute it. No new dataset/source downloads, inference, old-test
tuning, paid service or schedule/permission change. Actual provenance, units/FPS
and independent cohort readiness remain unresolved.


## Limited cache recovery follow-up: runtime offline (October 18)

[Recovery observation](docs/0003-cache-recovery-2026-10-18.md): the exact original
workspace/cache path check was rejected before shell launch because no execution
environment is connected. Current mounts, processes, marker/output state and cache
identity are unknown, not proof that historical data was deleted. No new gate,
synthetic/smoke run, data access, geometry, download or marker write occurred.
Earlier PR20 measurements remain historical and unchanged. Restore a supported
connection to the original workspace and reconcile its existing cache/markers
before any one-shot execution; no exposed recovery interface was available here.
