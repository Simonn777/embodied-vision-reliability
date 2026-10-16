# Hosted daily research

The owner requested research that continues when their Mac is shut down, with
**no additional fees**. The current migration candidate is ChatGPT Pro cloud
Scheduled within existing account entitlements. Its first trial encountered access
failures. Repository-scoped connector installation is now verified, but the recurring
task remains **paused pending a retry; execution and publication are unverified**.
Do not purchase extra usage credits or use separately billed APIs.

## Current execution status

**The paid Codex API workflow must remain disabled.** Keep
`CLOUD_RESEARCH_ENABLED=false`; do not configure an API secret or activate a paid
model run under the current authorization. Its initial preflights were infrastructure
tests only, with no model call or GitHub publication.

The existing local research schedule remains in place until the proposed cloud
replacement has actually executed the work, passed checks, and published a verified
commit. Creating a scheduled task or passing an infrastructure test is insufficient.
Pause the local schedule only after that verification, and avoid concurrent research
sessions. If the replacement requires an additional charge, stop the migration and
retain the local schedule. Keep account balances and private billing details out of
public documentation. Never copy ChatGPT session credentials into Actions.

## Subscription-based cloud trial

The October 16 trial reported a Linux/Python 3.13.5 shell with a 4 GiB container
limit, but `git clone` exited 128 because `github.com` could not be resolved. It
did not install dependencies or run the tests or smoke experiment. Its actual
GitHub `create_branch` call returned `403: Resource not accessible by integration`;
a separate connector check reproduced that error. No branch, PR, or cloud-authored
commit was published. Existing successful CI was inspected, not triggered by the
trial. At the time, the account had authorized the connector's identity access but
had not installed it for this personal repository.

After that failed trial, the owner explicitly authorized the official ChatGPT Codex
Connector for **only `Simonn777/embodied-vision-reliability`**. The installation and
its exactly-one-repository scope are now verified. This fixes the missing-installation
step; it does not establish that shell networking, test execution, or repository
publication now works. An actual retry and verification of its outputs are still
pending. Do not expand connector access to other repositories.

The proposed schedule is daily at 09:00 America/New_York through October 15, 2027.
It is paused to avoid repeated blocked runs; the original local schedule remains
the active fallback. The separately billed API workflow is also disabled at the
GitHub workflow level, in addition to `CLOUD_RESEARCH_ENABLED=false`.

The trial exposed a separate resource-control issue: the experiment script always
overrode numerical-library thread settings with four. The runner now respects the
smallest positive inherited limit, capped at four, and records the effective limit
in both configuration and provenance. Local verification passed 83 tests and a
two-thread smoke run of 1,240 rows. That verifies the fix, not cloud publication.

## Retained inactive API infrastructure

`.github/workflows/cloud-research.yml` is retained as an inactive implementation
reference: disposable GitHub Ubuntu 24.04 runners, the official OpenAI Codex Action,
isolated validation, and a separate publisher. Its API calls are separately billed;
the ChatGPT subscription does not cover them. The details below describe this
disabled workflow, not the proposed subscription-based replacement or an active
autonomous research service. No activation is authorized by this document.

## Schedule and persistence

- Daily at **09:17 America/New_York**, avoiding the busy start of the hour. DST is
  handled by GitHub's timezone-aware schedule. GitHub can delay or drop scheduled
  runs; this is not an exact-time or uninterrupted-service guarantee.
- Research is allowed only October 16, 2026 through October 15, 2027. The gate
  refuses work outside that window; after the year, disable the workflow itself
  to stop the small scheduling gate as well.
- A concurrency group serializes runs. A committed `.research/cloud-runs/DATE.json`
  prevents a second successful publication for the same New York date. Failed runs
  can be retried deliberately; a failure before publication can still incur API usage.
- State lives in Git: CURRENT_STATE, NEXT_ACTIONS, protocols, dated notes, and result
  provenance. Each ephemeral runner starts from the current main commit.
- No empty calendar-filling commits. If the model produces no material change,
  the workflow declines publication. A changed file is not proof of scientific
  value; the research charter and periodic evidence reviews remain necessary.
- GitHub can disable public schedules after 60 days without repository activity.
  Check the Actions page for failures or missing runs; do not manufacture activity
  to avoid this rule. Failure notifications follow the owner's GitHub settings.

## Execution boundaries

1. **Gate:** checks the date, activation, API-secret presence, and daily ledger.
2. **Research:** read-only GitHub permission, no persisted Git credential. Dependencies
   are installed before the model receives API access. Codex runs with `:workspace`
   permissions and `drop-sudo`. Native web search is requested; shell networking is
   disabled. Only a trusted fixed helper packages a patch after model execution.
3. **Validation:** a fresh runner checks hashes, paths, file modes, size, the base
   revision, and immutable prior results/protocols. Candidate and baseline tests
   run in a CPU/memory-limited container with no network, secrets, Docker socket,
   privileges, or write access to the repository. A small scratch directory is
   writable for the smoke diagnostic. Only the exact original artifact proceeds.
4. **Publication:** a fresh runner with `contents: write` revalidates the artifact
   using infrastructure from the trusted base. It runs no candidate Python or
   dependency installer. If main has moved, it stops; there is no force push or
   automatic conflict resolution. A validated patch and dated execution ledger
   are committed and the remote commit is checked.

The candidate cannot edit workflows, the prompt, its validator, dependency setup,
AGENTS, the charter, contribution policy, this document, or the ledger. Existing
results and protocols are immutable; dated notes can only be appended. Deletions,
symlinks, submodules, executable files, and oversized changes are rejected. These
checks reduce publication risk; they are not a proof of numerical correctness or
novelty and cannot detect every possible secret or misleading claim.

Research has a 32-minute model-step limit and a 40-minute job limit. Individual
experiments should end within 15 minutes, use two CPU threads and about 6 GiB RAM.
Validation ends within 15 minutes, with a ten-minute container timeout. No GPU is
provided. Third-party data and additional libraries need a controlled, licensed
dependency/data update; a blocked shell download must not be bypassed.

New final-test protocols and implementation changes must be published first and
run in a later session. A local model-created commit that disappears during patch
publication is not a durable preregistration. Results must refer to already-published
source/protocol revisions, and the original final partition cannot become a tuning set.

## Preflight and maintenance

**Cloud research → Run workflow → preflight** is an infrastructure-only mode with
no model/API call. It makes a labeled fixture, transfers it as an artifact, applies the same
patch checks, tests inside the isolated container, and rehearses a commit/push to
a temporary local bare Git repository. It does not call Codex or publish to GitHub.
It cannot validate model access, API funding, model tool behavior, branch protection,
or the actual write token. Those production capabilities remain unverified; do not
activate a billed research run to test them under the current no-additional-fees constraint.

Keep `CLOUD_RESEARCH_ENABLED=false` or disable the workflow.
Cancel any already-running job separately: changing a variable does not interrupt
a model step that has already started. Preserve failure logs. The API workflow must
stay disabled under the current constraint; do not retry with another paid service.
The existing local schedule is retained explicitly while the replacement is unverified.

A push made by `GITHUB_TOKEN` does not trigger ordinary push CI. This workflow runs
validation before publication and links that run in the commit. It does not deploy
GitHub Pages or send messages, and it does not claim that a separate CI ran afterward.
Artifact retention is seven days; durable research evidence belongs in the published
repository. Never treat an artifact upload as a completed publication.

## Official references checked on October 16, 2026

- [OpenAI Codex Action and required API credentials](https://github.com/openai/codex-action/tree/bdf19a4a223ec2549a3e2274a0cf61556bc07675)
- [Codex Action security](https://github.com/openai/codex-action/blob/bdf19a4a223ec2549a3e2274a0cf61556bc07675/docs/security.md)
- [OpenAI API spend limits](https://developers.openai.com/api/docs/guides/spend-limits)
- [GitHub schedule semantics](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule)
- [GitHub token behavior](https://docs.github.com/en/actions/concepts/security/github_token)
- [GitHub Actions billing](https://docs.github.com/en/actions/concepts/billing-and-usage)

## Initial verification

On October 16, 2026, 83 local tests and workflow lint passed. The ordinary
[CI run](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/38855613524)
and the four-job
[hosted preflight](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/38855628928)
passed at `3aa6ac1293226dfbe1f14f746fcbeea2cd040bcc`. The model step and real
GitHub publication step were explicitly skipped. The publisher rehearsed against
a temporary local bare repository only. `CLOUD_RESEARCH_ENABLED=false`; no API
secret is configured. This verifies infrastructure, not autonomous research.
The workflow now pins Ubuntu 24.04 instead of the moving `ubuntu-latest` alias.

## Interactive Work retry evidence

The interactive Work cloud retry on October 16 successfully cloned and executed tests/smoke, and created capability PR #1 with new passing GitHub-hosted CI. See [the capability receipt](CLOUD_CAPABILITY.md) for exact source, commands, timings, logs, and publication gates. This updates the historical pending status above for interactive execution only. Scheduled tool parity and scheduling migration are not verified; all schedules remain unchanged in this session. The API workflow remains disabled.
