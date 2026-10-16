# Current state

Updated: 2026-10-16. Program day 1. Phase: first diagnostic completed; preparing external-data validation.

## Active question

For a fixed visual position estimator, does causal temporal consistency identify large localization errors under held-out occlusion/appearance shifts better than confidence from the current image alone, at comparable accepted-prediction coverage?

## Status

- The [research charter](docs/RESEARCH_CHARTER.md) defines the year and twelve-week milestones.
- The first experiment is a controlled synthetic 2D localization diagnostic. It tests the measurement pipeline; it cannot establish real-robot performance or methodological novelty.
- Experiment 0001 completed: 160 base sequences / 19,840 scored rows overall; 40 independent test sequences / 4,960 test rows. Protocol `a5ab4e3`, code `bc9921f`; source unchanged at execution. See the [day-one note](notes/2026-10-16.md) and [artifacts](results/0001-pilot/summary.json).
- The temporal heuristic did not resolve a constructed stable wrong-identity failure. Under occlusion it reduced coverage from 42.10% to 37.02%, with zero observed selected failures for both gates. No general efficacy claim follows.
- 17 tests and GitHub Actions passed. A separate full rerun matched six deterministic artifacts byte for byte; independent metric recomputation matched all 112 interval endpoints. See [reproduction.json](results/0001-pilot/reproduction.json). This is same-environment verification, not independent-data validation.
- Eleven primary-paper entries and a pinned data-source/size audit are available. No listed paper's published result has been reproduced.
- No third-party datasets or pretrained weights have been downloaded for this project.
- Hardware checked: Apple M5, 24 GiB memory; CPU and MPS available, CUDA unavailable. Initial runs use CPU.
- The retained [Codex API workflow](docs/CLOUD_EXECUTION.md) remains disabled under the owner's **no-additional-fees** constraint. Its [cloud preflight](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/38855628928) passed without a model call or GitHub publication; this was infrastructure verification, not autonomous research.
- A subscription-based cloud Scheduled trial ran on October 16, but could not clone the repository or create a branch through the GitHub integration. Its daily schedule is paused while access is configured. Cloud research execution and publication remain **unverified**; see the [trial record](docs/CLOUD_EXECUTION.md#subscription-based-cloud-trial).
- The existing local research schedule remains in place until a replacement has actually executed, passed validation, and published verified work. No additional paid service or usage-credit purchase is authorized.

## Claims boundary

No accepted paper, new state of the art, real-robot validation, or successful research hypothesis is claimed. Bibliographic verification and reading depth are tracked separately. See [contributions](docs/CONTRIBUTIONS.md).

## Continuation

Read [NEXT_ACTIONS.md](NEXT_ACTIONS.md) and the active protocol before running work. Check Git status and GitHub before assuming the local state is current. Do not rerun a final-test comparison to tune a method.

Scheduled research ends on October 15, 2027 (America/New_York), unless the user renews the program.
