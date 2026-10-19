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


## Attempt03 legacy-checkpoint compatibility review

Independent source review confirms the pinned MatchAssignment owns only
matchability/final_proj and never defines or reads r; its official release loader
uses strict=False. Attempt02's stored exception names exactly nine .r entries,
with no missing active weights. Imports, MoGe and SuperPoint strict checks passed;
no fixture or inference arrays exist. The single optional CPU probe is recorded
separately and ordinary violation count is zero.

The narrow conversion is independently reviewed: exactly log_assignment.{0..8}.r,
nine model layers and no r attribute, float32 shape[1] and exact constant one,
and a canonical little-endian hash per entry. It copies the state mapping,
removes only these unused entries, then retains strict active-key/shape/value
checks. Other extra keys, partial legacy sets, nonfinite/changed constants or
missing active weights remain fatal. Model configuration and inputs are unchanged.
Reviewed qualifier SHA256 `d41711d25107c06862d78750dcd973a6429a5332eaefd64705662ae9bdb97cbb`;
test SHA256 `cbc89ab722af7ffba44b202fcce2c1adbc1c98256fbb05a81029cb209d134d84`.
Receipt04 binds the 13 passing artificial tests (exit0,0.273327417s). The reviewer
checked the source/receipt/log bindings without an unnecessary duplicate rerun.
No source-level blocker remains to fresh installation03/qualification03 after
the next exact freeze passes CI. Earlier failed attempts remain unchanged.


## Attempt04 macOS thread-report compatibility review

The independent saved-file audit verified attempt03's source hashes against
`472deb8a39bd9e4de754241aaa760044e4891787`, installation binding, supervisor logs,
and all public/raw publication hashes. The qualifier failed with
`execution_thread_configuration` after all three strict model loads, before
creating the fixture or running inference. Its output directory contains only
the receipt. The outer operation exited 1 in 20.239189334 seconds, with sampled
peak group RSS 652,288 KiB; it recorded one suppressed optional NumPy probe and
no ordinary offline-guard violation. Original attempt03 settings were not retained
in its error details, so the later API probes are separate diagnostic evidence.

The saved API-only probes report Torch intra/inter-op 4/1, OpenCV GCD reporting
10 after requests 1 or 4, and reporting 1 after request 0. The reviewer separately
checked [OpenCV 4.11.0's implementation](https://github.com/opencv/opencv/blob/4.11.0/modules/core/src/parallel.cpp):
`parallel_for_impl` bypasses parallel dispatch for both 0 and 1; the GCD getter
otherwise returns the CPU count. Thus the observed 10 is a reporting mismatch,
not evidence that ten OpenCV workers executed. These probes did not load models
or run an inference workload, and do not establish overall runtime scalability.

The reviewed change requests OpenCV 0, explicitly records parallelism disabled,
retains Torch 4/1 and OpenCV reported 1, and preserves actual settings in any
future configuration exception. Model weights, architecture, manufactured
fixture, inference options, thresholds and resource caps are unchanged. The
reviewed qualifier SHA256 is
`3f4fb5f38641c137b27a273fdffb284b65b0d024d94ce29986310b6091e07716`;
test SHA256 is
`a8164dd8bf0e66e5daa8366554d1863711aba1714774028d568dd6766f7c954c`.
[Receipt05](../results/0022-preparation/qualification-artificial-tests-05.json)
and its [log](../results/0022-preparation/qualification-artificial-tests-05.log)
match these source hashes and record 13 fake-interface tests passing, exit 0,
wall 0.483554208 seconds. This reviewer checked the source/receipt/log bindings
without repeating model loads, API probes or the test suite.

No source-level blocker remains to one fresh installation04 and the identical
qualification04 after exact freeze and successful CI. All earlier failures stay
unchanged; this is compatibility admission only, with no scientific payload,
accuracy result or manuscript admission.


## Attempt05 module-local annotation compatibility review

The saved attempt04 audit verified all 11 publication entries against their
private byte hashes, ten frozen source bindings against
`fa4a335b9eb1da7056ee20a07c50d181a35c7b74`, installation bindings and supervisor
log hashes. Strict model loading and Torch4/inter1/OpenCVserial checks passed.
The first manufactured MoGe forward reached postprocessing but failed at
`utils3d.torch.helpers` line16: Python3.9 did not supply `ParamSpec` through
`from typing import *`. No complete depth, feature or match output was produced.
The outer operation exited1 in19.422827125 seconds, sampled peak group RSS
2,053,168 KiB. Its CPU-autocast warning, optional NumPy-probe record and original
failure remain preserved; this attempt was not a completed qualification.

The repair preloads only the exact original helper module with one annotation
global supplied by the pinned existing typing_extensions4.16.0. Independent
source/hash checks confirm the helper and provider match the declared digests.
Source files and the global typing namespace are not patched by the repair.
Unexpected helper/provider bytes, origin, version or prior module registration
remain fatal; failed execution removes its partial module registration. A small
Python3.9 annotation-only check confirmed the backport works inside the helper's
nested `typing.Callable` type expressions, without importing Torch, utils3d,
model code or checkpoints.

Reviewed qualifier SHA256:
`174727207576ea4b9958fff6e6891016edcbe1a0729cd4e73e18b6543a6a8e8e`;
test SHA256:
`a23d18986b26458d8c2623d6155c404cbdcdea8ea2833920f64d87482ec854cb`.
[Receipt06](../results/0022-preparation/qualification-artificial-tests-06.json)
and its log match these sources and retain 14 fake-interface tests passing,
exit0, wall0.269343583 seconds. The reviewer did not repeat the model run or the
full test suite. No source-level blocker remains to the declared fresh
installation05/qualification05 after exact freeze and CI, with unchanged
manufactured inputs, weights, inference settings, thresholds and resource caps.
This compatibility review provides no scientific accuracy or novelty evidence.
