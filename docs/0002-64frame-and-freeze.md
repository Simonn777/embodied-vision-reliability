# 64-frame CPU development and exploratory freeze

2026-10-16. AI-assisted implementation, measurements and writing. No held-out
model inference, calibration fit or real-data reliability evaluation occurred.

## Real development measurements

The published pre-run source/plan was `db5134341831ffa7a01166a1515193ef384c6ba6`.
The metadata audit and original serial 64-frame runs remain unchanged under
`results/0002-preflight`: 42.868s and fresh-process repeat 43.121s, peak795600KiB.
Recording addendum/source `0773f28f4a5e3d990b4fd4ec5834c156d89c855c` was published
before a serial recording-complete replay. It adds environment/source binding;
there was no outcome-based model, query, score or tolerance change. These extra
runs are same-host/data verification, not independent data or an efficacy result.

| Recording-complete subrun | Wall seconds | First-frame seconds | Later median seconds |
|---|---:|---:|---:|
| Two queries, 64 frames | 39.337325 | 0.724249 | 0.585714 |
| One query, 16 frames | 10.349778 | 0.530385 | 0.673953 |
| Fresh-process two-query 64-frame repeat | 39.450014 | 0.650207 | 0.586656 |

Only video0, first64 frames, frame0-visible indices0 and1, arrival0. Every
[raw xy/logit and frame latency](../results/0002-preflight-recording/dev64/batch64-frames.csv)
is preserved. Outputs [64,2,4] have512finite scalars. Prefix versus earlier16-frame
batch maxdiff0; single/batch16 maxdiff0.00002765655517578125; independent64 replay
maxdiff0. Fixed atol0.0001, rtol0 throughout. The first process including loading
and single-query check took52.538606s, peak794612KiB; replay total42.073556s,
peak794764KiB (~776.1MiB). No hidden warmup; these are short measurements, not a
realtime/full-dataset/all-point feasibility claim.

CPU AMD EPYC9V74, affinity[0,1], torch intra/inter-op2, Python3.12.14,
Linux6.18.44/glibc2.39, torch2.10.0+cpu, NumPy2.0.2, dm-tree0.1.9, einshape1.0.
Each process: external timeout900s and6GiB address-space cap. Host MemAvailable
9490412KiB, cgroup memory8GiB, scratch free29212004352bytes at recording start;
these are observed availability, not exclusive reservation. Full timestamps,
commands, package/input/source/protocol/config hashes and limits are in
[dev receipt](../results/0002-preflight-recording/dev64/receipt.json) and
[repeat receipt](../results/0002-preflight-recording/repeat64/receipt.json).

All inputs reused verified cache. No new weights, data or packages downloaded.
The inherited conservative ledger includes old torch2.5.1 dry-run AND install,
then fixed2.10 dry-run/install, not cache-size accounting. This session adds an
8MiB allowance for small primary-source/metadata reads; total conservative ceiling
1462956526bytes <10GiB. No paid service, API, GPU or larger runner was used.

## Metadata grouping and limitations

Restricted decoding audited all50 entries'150arrays and12500frame hashes,
including actual64-frame RGB prefixes. No inter-video exact frame, prefix or full
video duplicate was found:50singleton groups, development1/calibration10/
evaluation20/reserve19. Complete identities/allocations are in the
[manifest](../results/0002-preflight/identity/manifest.json). Hashing other49
videos/annotations is metadata access, not model inference or tracking-error
inspection. Independent graph/allocation/array/frame/latency recomputation
[passed](../results/0002-preflight-recording/independent-verification.json), with
metadata peak2556360KiB (~2.44GiB). This uses an independent graph algorithm and
numeric recomputation on the same host and trusted restricted decoder.

Shared triplet-4 objects/front-left camera are known. Schema has no scene ID;
near duplicates and scene dependence remain unexcluded. Upstream checkpoint
training/model-selection exposure remains uncertain; paper benchmark zero-shot
language does not establish global novelty. See [source audit](0002-exposure-and-scope.md).

## Complete prospective implementation

[Protocol 0002-E v2](../protocols/0002-exploratory-frozen-v2.md) and
[config](../configs/0002-exploratory-v2.json) bind native temporal versus extra
causal motion residual on identical coordinates. This is an exploratory
exact-duplicate-group-disjoint, video-held-out pilot. Strict single-image
confidence remains unresolved. The native head's6px training target differs from
our fixed strict4px endpoint; no6px endpoint was added.

Independent0002 scoring, calibration, controls, matched80% risk, operational
thresholds and paired2000 whole-group bootstrap are executable. Empty groups,
repeated group multiplicity, common availability, missingness and unreachable
coverage are tested. [Synthetic full-pipeline fixture](../results/0002-preflight/fixture-frozen-v1.json)
exercises complete prediction-grid loading through bootstrap, without held-out
outputs/labels. Original0001 protocol/results and old0002 draft remain intact.
The future inference runner releases unrelated arrays before torch loading;
this memory guard is a prospective implementation change, not a claim of having
run a held-out group. No new model experiment is needed for pure metric fixes.

The separate [freeze receipt](../protocols/0002-freeze-v2.json) is created only
after the input commit exists; it binds input commit/file/dependency/upstream/
weight identities without referring to its own hash. Parent independent review
and merge must precede any next calibration/evaluation inference. Revisions must
preserve this version and publish a new freeze before outcome access.

After merge/fetch, next session may run the first allocated calibration group
(g-033, video33), using an unused output directory:

```bash
env -i PATH="$PATH" PYTHONPATH=scratch/boot-cache/runtime-fixed:scratch/boot-cache/runtime:scratch/boot-cache/source:src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 timeout 900 python scripts/run_frozen_pilot.py infer --cache scratch/boot-cache --group-id g-033 --output scratch/pilot-predictions/g-033
```

This command was NOT executed here. Preserve every allocated group, stop on
resource failure and publish real progress; do not drop difficult videos.

Validation: 148 repository tests passed (3.28s), existing protocol0001 toy smoke
completed1240scored frames (3.07s). These are software checks, not additional
held-out research results. Exact-head GitHub Research checks must also pass.

## Pre-merge provenance repair: v1 superseded by v2

Independent review found a P1 in v1: caller-provided maps could omit mandatory
inputs/dependencies, and current hashes were not checked against input-commit
blobs. A dirty/untracked receipt could be substituted. V1 spec/config/receipt and
all original measured artifacts are retained unchanged, but v1 is superseded
before ANY calibration/evaluation. This is an implementation/provenance repair;
no new model inference, metadata audit, acquisition or paid resource was used.

V2 requires the exact canonical receipt, tracked and unchanged in index/HEAD,
identical to origin/main's published regular blob. It checks all 29 mandatory
input SHA256 values against both specified input-commit blobs and current
tracked files (also HEAD), all six prerequisite bindings and 16 exact dependency
versions/provenance. Canonical manifest/config/protocol paths, upstream, archive
and model identities are bound. There is no publication bypass flag. Empty,
missing or extra maps; old commit/new hashes; dirty, staged, untracked or locally
committed-but-unpublished receipts; path substitutions/symlinks; changed package
versions; duplicate JSON keys and self-containing input trees are rejected.
Tests use tiny temporary Git repositories and mocked installed-version metadata;
no torch import, checkpoint, video decode, model or network is needed.

The CLI now defaults only to v2. V2 inputs are published in a first commit; its
receipt is published separately referencing that input SHA, with no self hash.
The actual receipt must remain non-executable until independent review/merge and
an authorized fresh origin/main update. Local Git refs are conventional local
publication evidence, not cryptographic remote attestation. No held-out data has
been inferred or inspected for errors, so no outcome-informed method change is
introduced. See [v2 protocol](../protocols/0002-exploratory-frozen-v2.md).

Repair validation: 184 tests passed in 7.08s; existing 1,240-frame toy smoke
completed in 3.10s. No model or metadata rerun. The v2 software receipt is
results/0002-preflight/freeze-v2-validation.json; final-head CI is checked separately.
