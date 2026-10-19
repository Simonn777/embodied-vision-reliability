# 0022 independent installation and qualification review

**Decision:** no remaining source-level blocker to one isolated installation and,
only after its clean completion, one manufactured-input qualification under the
next exact source/protocol freeze and CI. Both stages must use that same reviewed
commit and their own supervision receipts. This is not admission of benchmark
inputs, a scientific allocation, or any accuracy/novelty claim.

## Completed acquisition checked independently

The acquisition at `ce7b02ef130ca6f99a71fdb13f590a41e4293641` has clean outer
completion: exit 0, 34.676637916 seconds, sampled peak 28,048 KiB, no remaining
descendants or cleanup errors, and empty stderr. This review independently read
all **119 files as opaque bytes**, checked their sizes and SHA256s against their
ledgers, and checked all predeclared publisher/Git-blob identities. It did not
deserialize model or wheel contents.

| Batch | Files | File bytes | Observed response-body bytes | Requests |
|---|---:|---:|---:|---:|
| Source/model | 105 | 194,142,010 | 194,143,010 | 108 |
| Wheels | 14 | 76,809,693 | 76,809,693 | 14 |
| Total | 119 | 270,951,703 | 270,952,703 | 122 |

The 1,000-byte difference is accounted redirect-body content. Three normal 302
redirects and all 119 final 200 responses are complete. Request and read-event
body sums agree with both ledgers and the operation record. Private event-log
hashes, public/private receipt copies, stdout/stderr hashes, the final stdout
operation summary, and frozen source identities all agree. The two LightGlue
release hashes remain explicitly first-download identities, not publisher
digest comparisons.

The first read-only review attempt stopped on an overly strict reviewer
assertion that every request status must be `complete`; normal redirects have
status `redirected`. Correcting that assertion yielded a complete passing
review. No acquisition, installation or inference was repeated.

## Narrow corrections verified before execution

The installer now requires completed acquisition operation **and** supervision
records. It verifies prior-commit source identities, current manifest identity,
command/output location, log hashes and final operation summary, both batch
ledger hashes and body accounting. A completed inner ledger cannot bypass a
failed outer operation. It permits the acquisition commit to precede the new
installation commit. Every source/model file is rehashed before use; wheels have
publisher SHA256 pins. Sources are copied intact and pip uses a fresh private
target with `--no-index --no-deps --no-compile`; the shared environment is not
modified.

The qualifier verifies installation source/commit identity, clean outer
supervision, command inputs, log hashes, installed-file inventory, acquired
weights, and dependency metadata before numerical/model imports. It repeats
input checks afterward. It uses local `weights_only=True` loads, checks complete
learned state, preserves official LightGlue legacy renaming, and permits only
the documented deterministic confidence buffer completion. Constructor config
keys are checked rather than silently ignored.

The fixed full MoGe `infer` path, CPU float32, token count, raw depth/mask storage,
SuperPoint and LightGlue settings, manufactured image pair and output checks
match the prospective source contract. The final narrow change calls the
authored `enable_pytorch_native_sdpa()` after strict weight checks and records
Torch intra/inter-op limits 4/1 and OpenCV limit 1. The attention wrapper exists
in the pinned author source; model compatibility is still to be measured.

Captured warnings now enter the receipt in `finally`, including when loading or
inference raises. An independent standard-library-only artificial probe verified
warning preservation and propagation of the original exception, exit 0. No
Torch/NumPy/model import occurred in that probe. The offline guard explicitly
describes its Python audit-event scope; it is not an OS sandbox for arbitrary
native syscalls. Known download paths use local files, offline settings and
blocked socket/DNS/process/input entrypoints.

## Evidence and remaining limits

Reviewed installer SHA256:
`ae94b94817081ed31d41b7191c8c98dd835f5f363837318e195a7267a05e4ca4`.
Reviewed qualifier SHA256:
`cde436f3cb5131b5c10e5b01d1c3c80e3d4beecb6ac0fe609ec7bf5b9f584d99`.
Reviewed qualifier test SHA256:
`c4b9670f20433eead3072c8afdd7f444119d0e21d24dbdcbfdcd5cb445486037`.
Supervisor remains the previously reviewed
`76c7291004d181150a2d1a0ea5d287fffec3cb41f07645ea5bd1409ef241ace5`.
The revised second-stage protocol was reviewed at SHA256
`1d92a3067bf2fff459e9af4cd6d2971f799d16a6b1304ff6ab962b121f1a0ea4`.

The [installer-chain checks](../results/0022-preparation/installer-chain-artificial-checks.json)
are author-recorded mocked checks. The latest
[qualification test receipt](../results/0022-preparation/qualification-artificial-tests-02.json)
and [log](../results/0022-preparation/qualification-artificial-tests-02.log)
record 11 artificial tests passing, exit 0, subprocess wall 0.453951834 seconds.
This review matched their current source/log hashes and the preserved first
verification hashes; it did not rerun the full suite. No real pip installation,
model load or inference had occurred at this review. Native import compatibility,
valid outputs, runtime/memory and warning behavior remain qualification outcomes,
not facts established by the artificial tests. No thresholds or fixture changes
are admitted after those outcomes.


## Attempt-02 compatibility repair review

Attempt01 is retained as an import failure, not promoted to a successful
qualification. Independent saved-file review verified both operation source/log
hashes against b1d3376, the successful installation's 14 distributions and 2,667
inventoried files, and the failed qualifier's installation binding. Its only
output is a receipt; no fixture, checkpoint loads or inference outputs exist.
All four public JSON copies are byte-identical; public path-redacted logs have
separate, verified hashes and the private raw logs are unchanged.

The revised guard permits no subprocess. Only on macOS, with exact lscpu
executable/argv/cwd/env, exact subprocess.run keyword state, and the immediate
NumPy check_support_sve module/frame/path contract, it raises FileNotFoundError.
NumPy's own optional-probe fallback catches that OSError and returns false.
The suppressed probe is recorded separately; all other process and network
violations retain their original rejection. Shared NumPy and model settings are
unchanged. The reviewer independently ran all 12 artificial tests, exit0.
Author test receipt03 separately records exit0 at 0.381808291 seconds.

Reviewed qualifier SHA256:
`fca5c42e420948769ebed16d4287da8adbaebdbb2453d9f2cee61977edce60f3`.
Reviewed tests SHA256:
`fed2cc3cac4cd9a8e7d2b19f75ccdf35ebe238b4b3eeb491064c5540c08567f7`.
No remaining source-level blocker to one fresh installation02 from the same
assets and one qualification02 after the third exact freeze passes CI. The
same-commit installation chain remains enforced. This authorizes no scientific
payload, model-quality claim, changed fixture or acceptance threshold.
