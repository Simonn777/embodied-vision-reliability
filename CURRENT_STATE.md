# Current state

Updated: 2026-10-16. Program day 1. Phase: development CPU feasibility verified; exploratory external-data protocol frozen.

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

## Claims boundary

No accepted paper, new state of the art, real-robot validation, or successful research hypothesis is claimed. Bibliographic verification and reading depth are tracked separately. See [contributions](docs/CONTRIBUTIONS.md).

## Continuation

Read [NEXT_ACTIONS.md](NEXT_ACTIONS.md) and the active protocol before running work. Check Git status and GitHub before assuming the local state is current. Do not rerun a final-test comparison to tune a method.

Scheduled research ends on October 15, 2027 (America/New_York), unless the user renews the program.
