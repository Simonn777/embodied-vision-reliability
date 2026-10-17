# Current state

Updated: 2026-10-17. Program day 2. Phase: 0003 Fortran development-audit stage A candidate; synthetic checks only, awaiting independent review.

## Active question

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
