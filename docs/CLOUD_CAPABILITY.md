# Cloud capability verification

Date: 2026-10-16. This receipt records **interactive Work execution, final receipt CI, and publication to main**. A later non-interactive trial and completed schedule migration are recorded separately in [SCHEDULED_VERIFICATION.md](SCHEDULED_VERIFICATION.md).

This AI-assisted infrastructure check is not a research finding. At the time of this interactive check, Scheduled execution was unverified and paused, and the local schedule was unchanged. The later receipt supersedes that scheduling status without changing these measurements.

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

The publication gate required a second Research checks run on the receipt update, followed by `merge_method="merge"` with `expected_head_sha` equal to the reviewed, passing head. The completed evidence is recorded below. Older run 38858417741 is not counted.

## Boundaries

No schedules were created, resumed, or changed. No paid service, API key, extra credit, larger runner, GPU, credential transfer, or connector-scope expansion was used. The disabled API workflow and activation variable were not changed. The GitHub CI retains its existing 10-minute timeout and two-thread configuration; this check does not claim a separately enforced CI memory cap. Prior failed trials remain recorded. Success here establishes this interactive Work session's capability only; unattended Scheduled tool availability was still pending at that stage and was tested later in the separate receipt.

## Final publication receipt

[PR #1](https://github.com/Simonn777/embodied-vision-reliability/pull/1) was merged into main as [d910736e](https://github.com/Simonn777/embodied-vision-reliability/commit/d910736ee6e6964ee216a16df264f3e9482697b3), preserving reviewed head [835cbc9a](https://github.com/Simonn777/embodied-vision-reliability/commit/835cbc9ab2a3c936194a6bf0e5b46b4cd06141c9).

The final [Research checks run 38859145851](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/38859145851), [job 116837226499](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/38859145851/job/116837226499), passed before merge: **83 tests in 1.83s** and **1,240 smoke rows in 4.76s**. Those measurements belong to the GitHub-hosted runner, distinct from the Work shell measurements above. CI checked merge ref `bd0a3a8f6d80a0a0176f4d2ca82d2f56168b7436`, combining head `835cbc9a` with base `e05a5877`. The merged commit was then independently confirmed. No frozen protocol or result file changed.

## Protocol provenance clarification

The raw [shell provenance JSON](cloud-capability/2026-10-16-shell-provenance.json) is preserved unchanged. Its `protocol_commit` field reports `e05a5877`, the shallow checkout's history boundary returned by the recorder's `git log` lookup; this is not the original protocol freeze. Full history identifies [a5ab4e3](https://github.com/Simonn777/embodied-vision-reliability/commit/a5ab4e31699d6e55a81153aedd891c577bcb29d8) as that freeze. An independent comparison verified every recorded source hash against `e05a5877` and the protocol blob SHA-256 `259f93c67321f5dfde7ecfc698751ef17586eaa7cb61eb2f0bfb86d4306ea3e8`. This clarification corrects the interpretation of the history field; it does not rewrite the original run record or imply a new protocol.
