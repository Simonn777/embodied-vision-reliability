# Next actions

Work from the first unfinished item. Record each outcome in a dated note before choosing the next experiment. Complete one substantive step per daily session; avoid replacing experiments with generic literature summaries.

1. **Completed:** implement, independently review, and execute experiment 0001 under frozen protocol/source, preserving the negative identity-switch result. Read [the day-one note](notes/2026-10-16.md) and the reproduction record before continuing; do not repeat the run merely to generate daily activity.
2. **Completed:** [post-hoc failure analysis](docs/analysis/0001-failure-mechanism.md) separates two-frame history adaptation, different calibrated thresholds, color-estimator failure, and synthetic-generator artifacts. [Primary methods reading](docs/literature/2026-10-16-confidence-methods.md) audits confidence/visibility outputs and causal context. Original predictions, scores, thresholds, protocol, and results remain frozen; no score was optimized on the now-visible test trajectories.
3. **Source audit completed; feasibility gates remain:** [RGB-Stacking / Online BootsTAPIR audit](docs/data-model-audit-2026-10-16.md) pins upstream code, object metadata, terms, coordinates and causal score interfaces. No model/data body or inference was run. Before final protocol freeze, verify bounded acquisition, safe serialization, video/scene grouping, minimum inference dependencies, CPU memory/runtime and prefix/query-independence checks on development inputs. Refined native confidence is temporal; a strict current-image baseline remains unresolved.
4. **Next:** draft protocol 0002, beginning with a bounded acquisition/development feasibility stage and the unresolved baseline controls from item 3. Do not freeze the final evaluation until these gates are resolved. Lock train/development/calibration/final-test scene identities; use a new untouched evaluation partition and specify the primary endpoint, practical effect threshold, and uncertainty unit before results are inspected.
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
