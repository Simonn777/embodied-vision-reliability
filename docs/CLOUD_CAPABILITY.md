# Cloud capability verification

Date: 2026-10-16. Status: **execution and PR publication verified; final receipt CI and merge are required by the publication gate**.

This AI-assisted infrastructure check is not a research finding or proof that a scheduled invocation has equivalent tools. Scheduling remains paused and unchanged.

## Scope and actual tools

Only `Simonn777/embodied-vision-reliability` was accessed for publication. Main was `e05a5877d1985edfdce48a8991420900fd95adc7`; no open PRs or existing capability record were found before this check. The current Work cloud toolset exposes create_branch, create_file, create_blob/tree/commit, create_pull_request, fetch_commit_workflow_runs, fetch_workflow_run_jobs, fetch_workflow_job_logs and merge_pull_request with expected_head_sha. This session actually created the branch, file and [PR #1](https://github.com/Simonn777/embodied-vision-reliability/pull/1) through the connector.

## ChatGPT Work cloud shell measurement

The public repository resolved and cloned successfully using ordinary Git HTTPS; no network-policy bypass or user-computer dependency was used. Declared dependencies were installed with `python -m pip install -e '.[dev]'`.

Source: [e05a5877](https://github.com/Simonn777/embodied-vision-reliability/commit/e05a5877d1985edfdce48a8991420900fd95adc7), clean at execution. Linux x86_64, Python 3.12.14, NumPy 2.0.2, Pillow 11.3.0, Matplotlib 3.9.4.

Both commands ran with `ulimit -v 6291456` (6 GiB address-space limit), `timeout 900`, and OMP/OPENBLAS/MKL/VECLIB/NUMEXPR thread variables set to 2:

- `python -m pytest -q`: **83 passed in 1.35s**.
- `python scripts/run_experiment.py --smoke --output scratch/cloud-capability-smoke --protocol protocols/0001-synthetic-localization.md`: exit 0, **1,240 scored rows in 3.3404s**. Provenance confirms `threads_cap: 2` and `source_dirty: false`.

The smoke uses IDs 10000–10009 and 100 bootstrap draws. It does not rerun the full final-test comparison. Master seed 20261016; bootstrap seed 20261017. [Machine-readable provenance](cloud-capability/2026-10-16-shell-provenance.json) preserves configuration/source/protocol and artifact hashes; raw smoke outputs remain scratch and are not promoted to research results.

## New GitHub-hosted CI evidence

Initial PR head: [c1fdc3a9](https://github.com/Simonn777/embodied-vision-reliability/commit/c1fdc3a9dcfd32fcfcb6a2384f9e7abe4514a370).

[Research checks run 38859005472](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/38859005472), [job 116836792187](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/38859005472/job/116836792187), completed successfully. Its decoded logs were read: Ubuntu 24.04.5 GitHub-hosted runner, Python 3.11.17, all five thread variables 2; **83 passed in 1.67s**, smoke **1,240 rows in 2.88s**. PR CI checked GitHub's merge ref `50a9869d0923bc10973bd729e5af91b843f4e4bd`, combining that head with the stated base.

This receipt update must itself pass a second Research checks run before merging. The PR Checks tab and merge commit provide the final head/run/merge evidence without a self-referential receipt commit. Merge is authorized only with `merge_method="merge"` and `expected_head_sha` equal to the reviewed, passing head. Older run 38858417741 is not counted.

## Boundaries

No schedules were created, resumed, or changed. No paid service, API key, extra credit, larger runner, GPU, credential transfer, or connector-scope expansion was used. The disabled API workflow and activation variable were not changed. The GitHub CI retains its existing 10-minute timeout and two-thread configuration; this check does not claim a separately enforced CI memory cap. Prior failed trials remain recorded. Success here establishes this interactive Work session's capability only; unattended Scheduled tool availability remains to be validated separately.
