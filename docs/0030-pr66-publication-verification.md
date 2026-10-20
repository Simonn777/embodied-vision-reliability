# PR66 publication verification

Saved-record inspection on 2026-10-20; no new GitHub query, model run or metric recomputation.

[PR66](https://github.com/Simonn777/embodied-vision-reliability/pull/66) was confirmed merged by the parent dispatcher at `2674245ec1ecbc88220415e59e5f09b3ef3f4677`. The inspected saved CI records report:

- Result/source head `7896599ee0f99643f482cc72dd7f5f35940bf189`: completed, success; [run 39412894612](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39412894612).
- Merged main head `2674245ec1ecbc88220415e59e5f09b3ef3f4677`: completed, success; [run 39413677312](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39413677312).

The source records are `campaign/source-ci-7896599-completed.json` (SHA256 `7295fcbd892173fb6d014ab0ae15f780b3e562fc6f214b87bd88403247abcaa4`) and `campaign/main-ci-2674245.json` (SHA256 `b90ef6fe1e103de7442bd257f909ebc8ddb2997b5b276cdc5c2646ed5c9193e3`).

PR66 includes the completed first six-arm DEV comparison, fixed TRAIN-fit diagnostic, and privileged TRAIN teacher-task alignment result. The multistep student did not improve DEV J&F; the modest GT-control improvement and small fixed-context TRAIN benefit do not meet the final paper gate. Original negative arms, costs and earlier technical failure references remain preserved.

The later full-GT-write diagnostic and all-active-write development continuation are separate PR67 work, not retroactively attributed to PR66's CI. CI success and merge establish publication state; they do not establish novelty, independently replicated accuracy or success on two protected sources. This note checks the exact saved CI identities and reporting scope only.
