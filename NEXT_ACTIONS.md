# Next actions

Work from the first unfinished item. Record each outcome in a dated note before choosing the next experiment. Complete one substantive step per daily session; avoid replacing experiments with generic literature summaries.

1. **Completed:** implement, independently review, and execute experiment 0001 under frozen protocol/source, preserving the negative identity-switch result. Read [the day-one note](notes/2026-10-16.md) and the reproduction record before continuing; do not repeat the run merely to generate daily activity.
2. Inspect the first failure cases and read the most relevant methods/evaluation sections of the closest papers. Distinguish a confidence problem from an estimator or synthetic-generator artifact. Do not optimize scores on experiment 0001's now-visible test trajectories.
3. Audit one public point-tracking/video dataset and one suitable frozen estimator: source, exact version, image/video and annotation licenses, redistribution conditions, size, CPU/MPS compatibility, and whether annotations support localization error. Candidate selection is a research task, not permission to download an unrestricted corpus. Use a small legally usable subset; keep raw downloads out of Git.
4. Write protocol 0002 for external-data validation. Lock train/development/calibration/final-test scene identities; use a new untouched evaluation partition and specify the primary endpoint, practical effect threshold, and uncertainty unit before results are inspected.
5. Run the fixed baseline, inspect annotation conventions, and record cost. Establish always-predict, missing-observation, random-ranking, single-frame-confidence, causal-temporal, and clearly labeled oracle controls. Test motion changes where temporal smoothness should fail.
6. Only after these controls, evaluate whether a learned/frozen representation or a revised rejection score is warranted. Preserve negative results and compare at matched compute and accepted-prediction coverage.

## Recurring reviews

- Weekly: write a short evidence synthesis, including one rejected explanation or unresolved failure and the next discriminating experiment.
- Monthly: compare actual evidence to the phase milestone, audit public claims/citations, and record a justified continue/change/stop decision.
- January 7, 2027: first twelve-week report and decision. Do not imply novelty without comparing the closest methods.
- October 15, 2027: archive the year with an index, reproducibility instructions, complete findings, and limitations; conclude the scheduled program.

If blocked on compute, rights, or tooling, do a useful smaller verification or record exactly what is missing. Do not fabricate a measurement, backdate a commit, or start an unrelated repository to simulate progress.

## Cloud capability handoff (2026-10-16)

**Completed:** interactive Work execution, final receipt CI, and PR #1 publication to main at `d910736e`; see [the capability receipt](docs/CLOUD_CAPABILITY.md). The remaining migration check is an actual standalone Scheduled invocation with execution, validation, and publication evidence. Scheduled tool parity remains unverified and its schedule remains paused; retain the local schedule until that check succeeds without additional fees. The next research action remains item 2 above.
