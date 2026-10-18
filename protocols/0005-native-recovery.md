# 0005 preparation recovery — prospective revision

This revision follows the preserved dependency failure in
[the first attempt](../results/0005-native-capability/README.md). It changes asset
layout and preparation provenance only. The original seven frozen inputs,
canonical receipt, failed receipts, cache contents, and all previous scientific
results remain unchanged. There has been no native TAPNext++ model construction
or forward to repeat. No accuracy, novelty or research improvement is implied.

## Observed failure and unresolved cause

All five source files and three wheels matched their frozen hashes; installation
returned exit 0. The source-tree allowlist then rejected an empty `.partial` file
and prevented the import probe. Inventory records three empty source partials and
two empty wheel partials. Their creation/persistence mechanism is unresolved.
The old downloader unlinks a completed partial while its stream remains open;
that ordering is a possible filesystem interaction, not a demonstrated cause.
`partial_bytes_stored: 0` alone cannot distinguish an absent file from an empty
entry. Preserve both observations. Do not turn the failed receipt into success.

## Cache-only dependency recovery

Use the original cloud CPU workspace, with no overlapping research process.
Keep `bootstapir-cache` and failed `tapnextpp-cache` read-only. Create one new
sibling `tapnextpp-recovery` with exclusive phase markers. Before any operation,
verify the published recovery freeze against current files, index, HEAD and its
source commit, and independently verify the original canonical freeze.
The external stage caller performs this dual `--verify-only` check immediately
before **each** dependency/checkpoint invocation and checks the preparer hash
against the published receipt before launching it. The preparer additionally
checks its config and referenced evidence; it does not independently attest Git
history. The native supervisor performs both freeze checks itself. Keep only one
active cloud command: while a stage is running, poll that same execution handle;
do not launch concurrent cache inspections or other writers. This removes a
possible source of interference without claiming it caused the earlier failure.

Compare the complete failed-cache file set, sizes and hashes to its archived
inventory; reject symlinks and any missing, added or changed files. Check the
failed receipt, eight acquisition records, installation exit/log hashes, inherited
ledger and 378-file overlay inventory. Check old runtime identities and the three
recorded historical markers. An inventory mismatch stops recovery for review;
never repair it implicitly.

Copy exactly the five verified source files to a clean `source/` tree under the
new recovery directory, using exclusive destinations and closed-file verification.
Do not copy the partial entries, create package initializers, delete anything in
the failed cache, reinstall dependencies or make a network request in this phase.
Reuse the verified installed overlay from the failed cache. Run the original
fixed import-only probe once with the clean namespace and frozen package versions.
Verify namespace origins before import, actual package versions/module origins,
CPU-only Torch, and unchanged overlay/failed-cache files after import. No model
construction, data decoding or forward occurs in this phase.

Record the actual child exit, wall/CPU/RSS, source and environment bindings,
stdout/stderr hashes, both manifests and a new `recovery-dependencies.json`.
Only a completed receipt with successful actual import permits checkpoint access.

## First checkpoint acquisition

The separate checkpoint phase verifies successful recovery and all protected
inputs again. It acquires only the original frozen native 256px checkpoint into
the recovery directory. The URL, revision, expected 2,532,282,370 bytes and SHA256
are unchanged. This is the first checkpoint payload attempt, not a redownload.
Use normal HTTPS verification, the original 840-second asset deadline and one
logical GET with standard HTTPS redirects. No fallback URL or automatic retry.

Carry the failed-stage ledger's eight reservations and 9,179,574 acquired bytes;
reserve only the new checkpoint plus one sentinel byte. The historical ceiling,
128 MiB metadata allowance, 10 GiB acquisition limit and full planned ceiling of
7,499,066,782 bytes remain unchanged. Copying cached bytes is not network
acquisition and does not refund any earlier acquisition. Close the temporary
stream before verifying and publishing the final checkpoint; retain any failure,
partial bytes and phase marker. Never start inference from an incomplete phase.

## Unchanged first native capability attempt

Use the unchanged native adapter and scientific configuration: CPU float32,
four threads/cores 0–3, 8 GiB address-space limit, one constructor and official
weights-only checkpoint load, development video0 frames 0–7 and queries 0–7.
Exactly eight native forwards, 64 output rows, persistent native state, no support
queries, warmup, state reset, model modification or post-query GT interpretation.
The source root is the new clean tree; the overlay is the original verified
installation; the checkpoint is the new verified file. Check both the old and
new native markers before launch. Any previous native attempt prevents another.
An exclusive supervisor launch marker also prevents a second launch if a child
fails before the adapter creates its scientific marker.

Each preparation phase and the entire native invocation has an external
`timeout --kill-after=5s 1800`. Native child limits and the original 1,790-second
inner deadline remain. Save actual outer tool exits separately from inner wait4
receipts. Stop following phases on failure; no runtime fix, repeated phase,
extended resource limit, different model or unreviewed fallback is allowed.

PASS still requires every prescribed output and complete finite state with the
correct step progression within the unchanged limits. FAIL/partial remains a
failure of this configuration. Even PASS establishes only short-prefix native
CPU feasibility. It cannot establish throughput for longer clips, AJ/AJ_RD,
method benefit, real-video transfer or a paper result.

## Review, freeze and publication

The new canonical receipt binds the original seven inputs, original canonical
receipt, this protocol, new config/preparer/supervisor and four archived failure
evidence files: 16 inputs in total. Commit reviewed inputs first, then freeze
their hashes in a separate commit and verify GitHub publication before execution.
Use small artificial tests for recovery rejection, cache preservation, exact
copies, accounting and child failures, plus independent review. These tests are
software checks, not real model evidence.

Publish small receipts, original project source, raw capability outputs if any,
failure details and measured costs. Do not publish wheels, checkpoint, video
frames, upstream source bodies, secrets or private account information. Append
the daily note and update current state/next action after the actual outcome.
0004 remains stopped; 0003 remains deferred. No old evaluation tuning, additional
paid service, new real-data acquisition, paper writing or submission is enabled
by this recovery. Work is AI-assisted.
