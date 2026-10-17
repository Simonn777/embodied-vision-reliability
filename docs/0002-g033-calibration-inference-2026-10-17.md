# Calibration inference g-033: one group completed under frozen v2

2026-10-17. AI-assisted execution, audit and writing. **1/10 calibration inference
groups complete;0/20 evaluation groups complete.** This is prediction collection,
not calibration-threshold fitting, tracking-error evaluation or efficacy evidence.

## Preserved failure and authorized recovery

The [first attempt](0002-g033-gate-failure-2026-10-17.md) stopped before inference:
actual NumPy2.3.5 versus required2.0.2. That complete failure checkpoint was
published as `c585733510ed144be1305ca36eebc22f613dce62` and remains unchanged.
Its original stdout/stderr/status/time/source evidence and failed cache marker
are retained. The first planned prediction directory was never created.

A subsequent explicit user instruction authorized restoration of the exact16
versions, followed only by g-033 in a new directory if the real gate passed.
The original cwd was the isolated main05c2d7cb worktree. PYTHONPATH was and remains:

```text
scratch/boot-cache/runtime-fixed:scratch/boot-cache/runtime:scratch/boot-cache/source:src
```

Both cache layers initially lacked NumPy; the imported module and dist-info both
came from the base interpreter's NumPy2.3.5, not a stale mixed metadata/module
pair. The recorded original paths are in the
[path diagnosis](../results/0002-pilot-execution/g-033-20261017/numpy-path-diagnosis.json).
No alternate existing interpreter supplied2.0.2. The base pip cache was absent.

We selected the exact official PyPI2.0.2 cp312/manylinux x86_64 wheel:

- [Official release metadata](https://pypi.org/pypi/numpy/2.0.2/json),90923bytes;
- [Official wheel](https://files.pythonhosted.org/packages/39/68/e9f1126d757653496dbc096cb429014347a36b228f5a991dae2c6b6cfd40/numpy-2.0.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl),19236173bytes;
- newly recorded SHA256 `0123ffdaa88fa4ab64835dcbde75dcdf89c453c922f18dced6e27c90d1d0ec5a`.

The first download returned HTTP502 with zero wheel bytes; one retry of the same
URL succeeded and matched exact length/hash. Both events remain recorded.
Installed only that verified local wheel into existing runtime-fixed using
`--no-index --no-deps --no-cache-dir --no-compile`; no base package was replaced.
The existing bin-directory warning is preserved; no force/upgrade was used.

This is **version-matched environment reconstruction**, not byte-identical
reproduction. The wheel SHA was NOT in the frozen wheel list. NumPy's new
installed RECORD SHA is
`6d302eecd15ef4c12e6554eda66085772e4ea21390b34533c32a10345aa84f8f`, versus historical
`67a695d72a6cf98dce6db472040169a63498fd7ff0e35ab059755f9e1e640e08`.
The original dependencies.json,29 inputs and v2 parameters/gates are unchanged.
The other15 installed RECORD digests match the historical records. The recovery
[receipts](../results/0002-pilot-execution/g-033-20261017-recovery/) retain URLs,
bytes, hashes, installer command/exit/logs and explicit warning-prefix redaction.

Before the new inference, all16 installed versions and resolved module origins
were checked against their real dist-info locations. NumPy imported2.0.2 from
`<cache>/runtime-fixed/numpy/__init__.py`, with matching2.0.2 dist-info. Torch
imported2.10.0+cpu from cache, CUDA=None. Their native modules also resolve inside
the cache. Other module origins match their selected cached distributions; this
is not a claim of importing every optional API in every package. Newly recorded
entry-module hashes do not retroactively become frozen wheel identities.

## Actual inference and resource measurements

The unchanged runner ran only `infer --group-id g-033` from main
`05c2d7cb4ec1f81b6bf080252deb69bd9a72a319`, bound to v2 input
`bbe7428e126a566a62d3c123d7158f120aa66e2b` and published receipt `fbb6dd92`.
The real gate passed all29 input/current/Git bindings and16 actual versions.
No mocks, bypass, code change or second model run occurred. A group lock and a
new exclusive recovery marker protected the launch while retaining the failure.

```bash
env -i PATH="$PATH" PYTHONPATH=scratch/boot-cache/runtime-fixed:scratch/boot-cache/runtime:scratch/boot-cache/source:src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 timeout 900 python scripts/run_frozen_pilot.py infer --cache scratch/boot-cache --group-id g-033 --output results/0002-pilot-predictions/v2-reconstructed/g-033
```

This command HAS run and its output exists. **Verify/publish it; do not rerun.**
Video33 only: first64 consecutive256x256frames; selected point indices[0,1],
arrival0, using only frame0 visibility. No other group's model outputs were
computed. The frozen restricted loader decodes the existing archive to select
this video; it does not run other videos or compute their errors.

| Measurement | Actual value |
|---|---:|
| Outer inference start UTC | 2026-10-17T06:27:43.050288+00:00 |
| Outer inference end UTC | 2026-10-17T06:28:32.337375+00:00 |
| Outer subprocess exit code | 0 |
| Outer inference wall seconds | 49.286824886 |
| Runner wall seconds, including gate/load | 48.658732169 |
| Sum of recorded per-frame streaming intervals, seconds | 32.783010926 |
| First-frame interval, seconds | 0.542128687 |
| Subsequent-frame median interval, seconds | 0.493310807 |
| Peak RSS, KiB (runner and outer wait4 agree) | 2553156 |
| CSV rows / raw xy+two-logit shape | 128 / [64,2,4] |
| Finite output scalars | 512 |

The interval sum is not an independently measured whole-forward wall time.
Hardware: AMD EPYC9V74, Python3.12.14. CPU affinity[0,1], all five numeric thread
variables2, torch intra/inter-op[2,2], CPU/eval/no gradients/inference_mode, fixed
seed20261016, external timeout900 and6GiB AS cap. Raw stdout/stderr (empty for
inference), wait status, CPU/wall/RSS, environment, exact commands and all frozen
hashes are in the [outer/preflight records](../results/0002-pilot-execution/g-033-20261017-reconstructed-run/).
The source and preflight program are archived unchanged as execution evidence;
archive paths are not future entry points. Availability snapshots are observations,
not dedicated resource reservations. This short run establishes no realtime,
all-point or independent-domain performance guarantee.

## Saved evidence and independent checks

[Raw CSV](../results/0002-pilot-predictions/v2-reconstructed/g-033/video-033.csv),
SHA256 `22bb58db053c0cb011c72cadee1e0d91012301ff6e15bb3264ae23e39cd9819e`,
and [runner receipt](../results/0002-pilot-predictions/v2-reconstructed/g-033/receipt.json)
are unchanged original outputs. Independent standard-library verification checks
all128 unique frame/query rows and point IDs, four raw values per row, finite
counts, CSV digest, per-frame time agreement across queries and the runner list,
first/median/summed intervals, process exits/RSS/timing, source/log hashes, wheel
identity and all29 Git/current hashes. It does not call candidate model/decoder/
metrics, rerun inference, inspect GT coordinates/errors or calculate confidence.
Initial query selection relies on the unchanged frozen runner and verified input
identity; this check does not perform a second annotation decode. Full video33
identity from the frozen manifest is included in the
[verification receipt](../results/0002-pilot-execution/g-033-20261017-reconstructed-run/independent-verification.json).

No score, risk, coverage, threshold, reliability comparison, final evaluation or
repeat-model run was performed. The raw logits remain unscored. Shared objects/
camera, unknown scene dependence and unresolved upstream checkpoint exposure
remain the frozen design's limitations. No real-robot or methodological claim.

## Software checks, acquisition and continuation

Work pytest was unavailable in this reconstituted base interpreter; its real
exit1/zero-tests log is retained. No pytest package was installed in Work. The
existing toy smoke completed1240scored frames in3.77s under restored NumPy; its
synthetic software outputs are separate from g-033. Full tests plus smoke are
verified on the latest-head standard public Research checks CI. Software fixtures
do not evaluate RGB-Stacking predictions.

Research acquisition ceiling: inherited1462956526 +90923metadata +19236173wheel
=1482283622bytes, under10GiB. This includes the prior conservative torch dry-run/
install ledger; no new data/model downloads. Git/standard-CI traffic is separate
from this research package/data ledger. No paid API/GPU/runner, purchase,
connector permission expansion or schedule change occurred.

Keep g-033's completed output and both attempt records. Parent independently
reviews/merges PR#6. Future authorized collection continues from the frozen
allocation after g-033 (next calibration group g-013); this session did not run
it or any other group. No threshold fitting/evaluation until the prescribed
complete prediction collection and a later authorized evaluation step.
