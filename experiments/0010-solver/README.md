# Prospective 0010 artificial StatA admission

Software-only preparation. **Neither the 20 Torch functional fixtures nor the three real-shaped StatA fits have been run for this bundle.** Only the separately recorded fake-backend tests ran locally. This is not classification evidence, model-quality evaluation, a CPU throughput result or authorization to execute a study.

## Fixed prospective checks

Two fresh, separately supervised processes are prescribed, once each, in this order. Stop after any failed/unsupported/timeout stage; do not change seed, examples, precision, tolerance, graph, alpha or batch size.

1. `functional`: the existing 20 artificial functionality tests on the current cloud runtime. Retain the exact original test file. Its single `torch.set_num_interop_threads(1)` setter is replaced in the executed text with `(4)`, with original/executed SHA256 and exact replacement count recorded. No test logic changes. Expected result is exactly 20 tests, zero skips, failures or errors.
2. `timing`: exactly three measured calls, in the fixed order T63, T641, T644. T63 is a 63-row alpha-1 fit followed by one held-out readout of excluded row 0; T641/T644 are full 64-row alpha-1/alpha-4 fits. Every `fit` already includes its instrumented and uninstrumented authored solver executions and all fitted-row native-readout parity checks. No extra warmup, repeated timing or best-of selection. Total six authored executions if complete.

Each phase has its own external **120-second hard wall limit, CPUs [0,1,2,3], 8 GiB RLIMIT_AS and cgroup memory.max**, four intra-op and four inter-op Torch threads. The unchanged existing supervisor kills/reaps the process group and records raw wait status, child exit, monotonic wall/CPU times, Linux peak KiB and log hashes. No source/resource mutation of frozen 0010 capability files occurs. Interop=4 follows the latest prospective headroom decision; it is transparently applied to this new functional execution, not relabeled as the previous interop=1 test run.

Timing draws exactly one artificial input: `Generator(PCG64(202610180010))`, first a float64 normal `(64,512)` feature array, then float64 normal `(10,512)` anchors. Normalize rows in float64, cast float32, transpose anchors to `(1,512,10)`. IDs are `artificial-000` through `artificial-063`; query 0 alone is held out for T63. There is no resampling on a numerical refusal. This one random fixture is not representative of real CLIP geometry and cannot establish real-data coverage.

The measured admission estimate is exactly

`1.5 * (128*T63 + 2*T641 + 2*T644) <= 360 seconds`.

All raw times are retained. Each fit interval includes its own adapter/source loading, checks and native parity/readout work; the T63 interval additionally includes the held-out call. Common imports, thread setup, input generation/persistence and record serialization are separately visible through setup/outer time rather than hidden inside an inferred total. This projection is a conservative proposed resource filter, **not a guarantee**: future predictor serialization and additional repeated native readouts are not measured by this formula. The future headroom driver still needs its full 900-second cap and must retain any incomplete run.

## Source identity, portability and license

`upstream/StatA.py` and `upstream/LICENSE` are byte-for-byte cached copies from official StatA commit `2635f9355f9389756064becb5d9ed54c2756809c`. The author's source is **AGPL-3.0**, not MIT. Its exact license and original source are included with their source URLs, Git blob identities and SHA256 in `upstream/identity.json`; preserve these notices when distributing the bundle. This bundle does not relicense upstream. No author package initializer or additional author module is imported.

The existing `stata_cpu.py` remains byte-for-byte unchanged. Before isolated compilation, `load_exact` replaces exactly its one hard-coded `UPSTREAM_PATH = Path(...)` line with the bundle-relative equivalent; both texts are hashed. The adapter then independently verifies the unchanged upstream SHA and applies exactly its original four `.cuda()` to CPU substitutions. Existing test source is loaded explicitly after this adapter is registered as `stata_cpu`, avoiding an ambient same-named helper. The two transparent transformations do not modify any original file. Hash verification is against an externally pinned `manifest.json`; that manifest is not a signature or a substitute for a future published source/receipt freeze.

Runtime admission is prospective Python 3.12, NumPy 2.0.2 and Torch 2.10.0+cpu; actual module versions/origins and resource values are recorded before checking. No installation or dependency download is part of this script.

## Recipe after explicit approval and source freeze

Copy this whole source bundle into the existing cloud workspace; no source GET is needed. Obtain the expected manifest SHA from the independently reviewed/frozen publication, not from an unchecked remote copy. Use the already admitted interpreter/runtime paths. Fresh output directories must not overlap the source bundle or prior scientific outputs. These are future commands, **not executed here**:

```sh
PYTHONDONTWRITEBYTECODE=1 python run_admission.py functional --output /approved-output/0010-functional --manifest-sha256 APPROVED_MANIFEST_SHA
# Only after the first parent receipt has actual exit 0 and child tests are 20/0/0/0:
PYTHONDONTWRITEBYTECODE=1 python run_admission.py timing --output /approved-output/0010-timing --manifest-sha256 APPROVED_MANIFEST_SHA
```

The default CLI always invokes the existing external supervisor. `--child` is its internal command and still independently rejects incorrect Linux resource limits. Neither entry point retries or replaces outputs. A caller must preserve attempt outputs and enforce the once-only recipe across launches; choosing a different directory is not authorization to repeat. If the process is killed, an attempted checkpoint is not completion. The parent execution receipt is authoritative for exit/kill status.

Outputs: `child/receipt.json`; timing also `child/artificial-inputs.npz`; parent `supervision/{command.json,execution.json,stdout.log,stderr.log}`; and `artifact-manifest.json` covering all closed files with explicit self-exclusion. `status=failed` or `resource_not_admitted` gives a nonzero child exit. The functional and timing processes are independent; success in one cannot silently substitute for failure in the other.

## Local verification performed

Only `test_admission_fake.py` was run locally, using NumPy, fake core methods and tiny `(7,5)` artificial arrays. It checks exact fit/readout order and IDs, alpha, first failure termination, no readout after failed parity, attempt counts, projection arithmetic/refusal, deterministic input construction, exact one-line portable/thread transformations, manifest mismatch, and absence of any Torch/StatA import. See the sibling review receipt for actual logs and the preserved initial macOS path-alias fixture failures. Do not run `unittest discover` indiscriminately locally: the copied `test_stata_cpu.py` intentionally contains real artificial Torch fits for the later authorized cloud functional phase.
