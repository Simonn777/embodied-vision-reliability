# Independent review: 0005 saved preparation failure

**Verdict: approve publication of the failure evidence; no claim of successful preparation or native capability.** Reviewed candidate `dec0546a33c41eabd731da903d95cd2651bf080e`, execution base `e168ba830f9f0eff37192ee8745207ff7d3484ef`, frozen source `a6cb51ee60bdee2650ce6068cb3a2c0255e84931`.

## Verified evidence

- All **35 copied records/logs/scripts** match the copy manifest's size and SHA256; all **37 candidate result files** have hashes in the accompanying review JSON. Original cloud source files were not reread locally; origin-copy claims remain bounded by the saved copy operation and audit.
- The baseline map has **1,769 entries**, independently matched against the execution-base Git blobs. Candidate retains **1,766 byte-identical prior files**, including **all 1,595 prior results**. Only `CURRENT_STATE.md`, `NEXT_ACTIONS.md`, and an append to `notes/2026-10-18.md` changed; the note prefix is exact.
- All **seven frozen inputs** match current worktree, index, candidate, execution base, and frozen source commit. Canonical freeze bytes are unchanged, SHA256 `3e5aad0b0123affb3991997607c04310bd7d08a38172b8ffbc2c0f29b01f36ef`.
- Saved immediate-child `wait4` exit/raw statuses and closed log/source hashes agree: verify **0/0**, dependencies **1/256**, synthetic software **0/0**, evidence review **0/0**. Installation **0/0** is identical in its separate receipt and dependency receipt. The software log contains **44 tests, OK**. These are saved cloud results, not tests rerun by this reviewer.
- Eight asset records bind exact configured lengths/hashes, one HTTP-200 attempt each, and the saved final-file inventory. Payload **9,179,574 bytes**; reservations and inherited ledger give cumulative ceiling **4,966,784,411 bytes**. All **378 overlay entries** agree between the separate inventory and full cache inventory. That inventory is explicitly not import compatibility approval.

## Why `.partial` is not yet a proven downloader root cause

The frozen implementation creates an empty file at `scripts/prepare_tapnext_assets.py:169`, writes and flushes it, then checks exact size/hash at lines 190–194. Its success path explicitly calls `os.link(partial, final)` and `partial.unlink()` at lines 195–196 **while the stream is still open**, then sets success at line 197. Line 203 records zero both when a partial is absent and when an empty partial exists. This is a real observability ambiguity, not proof that unlink failed.

The saved inventory contains **five** empty partials: three source files (`pscan.py`, `tapnext_lru_modules.py`, `votsp2026/utils.py`) and two wheels (einops and Pillow). The strict source allowlist correctly rejects the first encountered source extra. No later source import child ran; the failure receipt has no import result or completed overlay manifest.

A 16-byte synthetic ordinary-local-filesystem fixture reproduced the exact open/write/flush/fsync/link/unlink-before-close sequence. The partial was absent both before closing and after closing; final bytes stayed correct. This ran on Darwin/Python 3.9.6 without importing candidate code, downloading, installing, or touching cache/model/data. **The cloud persistence/recreation mechanism remains unknown**; neither a deterministic Python unlink bug nor a cloud snapshot explanation is established.

## Minimal offline recovery recommendation

1. Preserve frozen v1, all failed cache files/partials, marker, receipt, logs and ledger. Do not delete extras or rewrite failure as success.
2. Review and freeze a distinct recovery procedure binding v1, this failure, all eight asset records and the complete 378-file overlay map. Use a fresh exclusive recovery marker and outputs.
3. Rehash the existing five final source files and three wheels. Copy only the verified five sources into a fresh exact-allowlist tree; verify copy bytes, namespace parents, no symlinks/initializers/extras. Retain the originals as evidence.
4. Reuse the installed overlay only after complete path/size/hash equality, then perform a separately supervised **import-only** version/origin check under unchanged limits and package pins. No pip, network, checkpoint fetch, model construction, or forward belongs in this recovery stage. Carry the existing cumulative acquisition ledger and record zero new payload.
5. Stop on any mismatch. Successful import recovery would still leave checkpoint acquisition and native execution uncompleted; those require their separately bound stages.

## Evidence boundaries

The saved phase parents observe their immediate children; no further durable outer `wait4` receipt for the stage parent or review parent is published. Three old marker hashes match the saved baseline/audit but were not reread in cloud by this reviewer. No model/GT activity is evidenced by this failed execution path; this is not a global claim about unrelated processes. Parent command observations and recovery files added concurrently are outside this candidate review.

No repository files, real assets, network, model, or dataset were modified/accessed by this audit. All new reviewer files are temporary local audit artifacts.
