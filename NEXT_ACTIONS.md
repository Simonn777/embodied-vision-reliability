# Next actions

Work from the first unfinished item. Record each outcome in a dated note before choosing the next experiment. Complete one substantive step per daily session; avoid replacing experiments with generic literature summaries.

1. **Completed:** implement, independently review, and execute experiment 0001 under frozen protocol/source, preserving the negative identity-switch result. Read [the day-one note](notes/2026-10-16.md) and the reproduction record before continuing; do not repeat the run merely to generate daily activity.
2. **Completed:** [post-hoc failure analysis](docs/analysis/0001-failure-mechanism.md) separates two-frame history adaptation, different calibrated thresholds, color-estimator failure, and synthetic-generator artifacts. [Primary methods reading](docs/literature/2026-10-16-confidence-methods.md) audits confidence/visibility outputs and causal context. Original predictions, scores, thresholds, protocol, and results remain frozen; no score was optimized on the now-visible test trajectories.
3. **Development feasibility measured and artifacts independently verified:** [real-weight CPU report](docs/0002-cpu-feasibility.md) preserves pinned official source/checkpoint/data hashes, restricted loading, 16-frame development latency/RSS, prefix/query/batch/repetition checks, and failures/serial repairs. The parent session separately recomputed the published checks and verified artifact/source hashes and latency accounting; see [verification receipt](results/0002-development/parent-review.json). Whole video index 0 is development forever. No other video was inferred.
4. **Next:** resolve scene/duplicate relationships or explicitly adopt an exploratory video-disjoint design; assign and publish complete identities without outcome selection. Profile a 64-frame development-only two-query clip, implement the prospective metrics/controls, and complete [protocol 0002-E draft](protocols/0002-exploratory-pilot-draft.md). The narrowed question is native temporal confidence versus extra motion residual on identical coordinates; strict single-image comparison remains unresolved. Final evaluation is blocked until source/protocol/config/splits are frozen and published for a later session.
5. Run the fixed baseline, inspect annotation conventions, and record cost. Establish always-predict, missing-observation, random-ranking, single-frame-confidence, causal-temporal, and clearly labeled oracle controls. Test motion changes where temporal smoothness should fail.
6. Only after these controls, evaluate whether a learned/frozen representation or a revised rejection score is warranted. Preserve negative results and compare at matched compute and accepted-prediction coverage.

## Recurring reviews

- Weekly: write a short evidence synthesis, including one rejected explanation or unresolved failure and the next discriminating experiment.
- Monthly: compare actual evidence to the phase milestone, audit public claims/citations, and record a justified continue/change/stop decision.
- January 7, 2027: first twelve-week report and decision. Do not imply novelty without comparing the closest methods.
- October 15, 2027: archive the year with an index, reproducibility instructions, complete findings, and limitations; conclude the scheduled program.

If blocked on compute, rights, or tooling, do a useful smaller verification or record exactly what is missing. Do not fabricate a measurement, backdate a commit, or start an unrelated repository to simulate progress.

## Cloud capability handoff (2026-10-16)

**Completed:** interactive Work execution, final receipt CI, and PR #1 publication to main at `d910736e`; see [the capability receipt](docs/CLOUD_CAPABILITY.md). The saved automation's non-interactive Run now trial also completed the data/model audit and merged PR #2 at `9d415308` after shell tests and exact-head CI. The settings session independently verified that publication, paused the local heartbeat and retained one enabled Work cloud schedule. See [the migration receipt](docs/SCHEDULED_VERIFICATION.md). The first regular clock-triggered run is due October 17; do not confuse that future event with the completed immediate trial. The next research action is item 4 above.
