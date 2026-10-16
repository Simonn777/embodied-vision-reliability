# Cloud capability verification

Date: 2026-10-16. Status: **verification in progress**.

This is an AI-assisted infrastructure check, not a research finding or proof of scheduled execution.

## Confirmed in this ChatGPT Work cloud session

- Remote main resolved to `e05a5877d1985edfdce48a8991420900fd95adc7`.
- No open pull requests were returned before work began. No capability record existed on main.
- The GitHub connector exposes branch/file/tree/commit creation, pull-request creation, workflow runs/jobs/logs, and merge with `expected_head_sha`.
- An actual connector branch creation succeeded for `cloud/capability-2026-10-16` at that source revision.
- The cloud shell successfully ran `git ls-remote` and cloned the public repository. Dependency installation and bounded tests are being checked.

## Verification plan

Run the declared small dependencies, `python -m pytest -q`, and `python scripts/run_experiment.py --smoke --output scratch/cloud-capability-smoke --protocol protocols/0001-synthetic-localization.md`, using two numerical-library threads and a 6 GiB address-space cap. Create a real PR, inspect its new Research checks run, jobs and logs, then update this receipt with measured results. Require successful CI again on the updated head, review the diff, and merge only that exact head.

No scheduler changes, paid API calls, credential transfers, permission expansion, or activation of the disabled API workflow are authorized or performed by this check. Existing CI run 38858417741 is not evidence from this session. Full final-test results are not rerun.
