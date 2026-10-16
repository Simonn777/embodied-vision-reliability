# Online BootsTAPIR runs on bounded cloud CPU development inputs

2026-10-16. AI-assisted implementation and measurement. **Measured feasibility,
not a reproduction of published accuracy or an evaluation of a reliability method.**

Real official weights (54,699,335 parameters) loaded with patched CPU PyTorch
2.10.0+cpu, weights_only=True and strict state-dict matching. The model explicitly
used causal convolution, eval mode, no trainable parameters, inference mode and
two intra/inter-op threads. Source origins and hashes, checkpoint digest and the
decoded development video/points/occlusion hashes were checked before inference.

## Development measurements

Host CPU: AMD EPYC 9V74. Container exposes an 8 GiB memory maximum and eight-CPU
quota; this experiment restricted CPU affinity to two logical CPUs, set all five
numerical-library thread variables to 2 and enforced a **6 GiB address-space cap**.
Available host memory was about 9 GiB; that is not the container's usable limit.
About 27 GiB disk was available before acquisition. Every experiment process used
an external `timeout 900`; peak RSS is the process high-water mark, not GPU memory.

Only whole video **archive index 0** is development (250 frames, 30 annotated
points). Only frames 0–15 were inferred; the two lowest point indices visible at
frames 0 and 4 were 0 and 1. The whole pickle container's other arrays were decoded
by a restricted loader but not inspected, visualized or inferred. They remain
unassigned, and scene relationships are unknown. Inputs were 256×256 RGB.

| Validated measurement | Actual result |
|---|---:|
| Cold model's first forward frame | 0.636 s |
| Cold four-frame run | 2.455 s |
| One query, 16 frames (after first call) | 8.276 s |
| Its subsequent-frame median | 0.518 s |
| Two simultaneous queries, 16 frames | 9.190 s |
| All main validation streams, wall time including setup | 63.504 s |
| Main validation process peak RSS | 746,092 KiB (728.6 MiB) |
| Restricted dataset decoding peak RSS | 2,572,184 KiB (2.45 GiB) |

Outputs contain `(x,y,occlusion_logit,expected_dist_logit)`. One-query output shape
is `[16,1,4]`, batch is `[16,2,4]`; active outputs are finite and pre-arrival outputs
are NaN. Last refinement scores already contain temporal history. About two
frames per second on this tiny clip is **not real time for a typical video** and
does not establish full-video/all-query throughput or a 64-frame memory bound.
The host/OS caches were not flushed; timing variation and a warm library/page
cache remain. No position-error accuracy was calculated.

## Adapter checks with real weights

All comparisons used the predeclared absolute tolerance `1e-4`, relative tolerance
zero. Raw scalar outputs and per-frame times are [published](../results/0002-development/).

| Check | Maximum absolute difference |
|---|---:|
| Invert frames 8+; compare frames 0–7 | 0 |
| Full run versus truncated eight-frame prefix | 0 |
| One query versus same query in a two-query batch | 0.0000276566 |
| Add future query at frame 4; compare earlier old-query outputs | 0 |
| Old query remains unchanged after second query arrives | 0 |
| Late query versus independent stream cropped at arrival frame 4 | 0 |
| Same-process repeated one-query 16-frame run | 0 |
| Fresh-process four-frame repeat | 0 |

Late-query pre-arrival masking also passed. These checks validate the inspected
adapter and small inputs, not every possible sequence, query count, numeric
platform or upstream model causality theorem. The independent repeat uses the
same host and data; it is not independent-data replication.

## Execution sequence, failures and review repairs

1. Development plan/code were published as `e9ca032d9fdc48b6b3d0ed9d0ac7c04c314a76ac`
   before acquisition, decoding or forward inference. `e11b70a8` added source
   digest verification and fresh-process repetition before initial inference.
2. The initial dependency choice was torch 2.5.1+cpu. Its dry-run and installation
   downloaded bytes, but **no checkpoint was loaded with it**. Review identified
   [GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p)
   (affected <=2.9.1, fixed >=2.10.0); actual inference uses exactly 2.10.0+cpu.
   Old package directories were retained for accounting and excluded as the torch
   import target by the preceding fixed-runtime path. Small pure/runtime packages
   still came from that setup directory; versions are recorded in each receipt.
3. Acquisition with an entirely cleared environment failed DNS because platform
   network configuration was absent. Retry used normal platform networking for
   public downloads only. Decode/inference used separate `env -i` processes with
   only PATH, PYTHONPATH and the five thread variables. No credential files were
   copied or used, and no network-policy bypass was attempted. The first failure
   receipt is retained with executor paths normalized.
4. Official object lengths, generations and MD5s matched. SHA-256 is
   `87c1e752cf5ce56e3e2f7da460aeb4d40fc826d04ef2939bade86a5c7495377f` for
   the checkpoint and `ba0cec640deb65ab510bc3920cdaf8b7a2e756fce9fb339199c557ade436b0cc`
   for the ZIP. Decoder inspected ZIP paths/2.46 GB uncompressed size and pickle
   opcodes, allowed only NumPy numeric reconstruction globals, and rejected
   arbitrary global imports and extension/persistent references. Numeric arrays
   were exported to non-pickle NPZ outside Git. No unrestricted pickle fallback.
5. Initial real-weight results under `e11b70a8` are retained as **preliminary**.
   Independent review found that failed numerical gates could still exit zero,
   decoded NPZ identity was unchecked, and actual module origins needed binding.
   It also requested strict arrival/dtype/coordinate/mask gates and the new late
   query crop comparison. Original tolerance, subset and results were preserved.
6. [Revision 1](../protocols/0002-development-revision-1.md) and repaired source
   were published as `d45c1bed4f82dd4237373567df1920a9a9ca7189` before a **serial**
   validated rerun. That run is the primary evidence above. Gate failures now set
   validation_failed and exit nonzero while retaining outputs; wrong manifest,
   data digest or module source fails before inference. A fresh-process minimal
   replay followed, without another model run executing concurrently.
7. An output-packaging command initially resolved the old editable repository and
   failed to import the new module. Explicit `PYTHONPATH=src` corrected packaging;
   no inference was repeated for that error. Initial full tests: **127 passed in 3.32s**; two additional provenance-gate tests brought the final suite to **129 passed in 3.07s**.
   Existing toy smoke: **1,240 rows in 2.84s**. This regression smoke is distinct
   from the real model measurement and does not rerun protocol 0001's full test.

The [transfer ledger](../results/0002-development/transfer-ledger.json) counts every
resolved full wheel for old dry-run + old install + small packages + fixed dry-run
+ fixed install (even where only metadata or cache hits occurred), plus official
model/data bodies, the existing pip cache and a 64 MiB metadata/source allowance.
Its conservative transfer ceiling is **1,454,567,918 bytes (1.35 GiB)**, below 10 GiB.
Expanded cache footprint is recorded separately (~2.0 GiB), not substituted for
cumulative downloads. No JAX/TensorFlow/CUDA inference stack was installed.

## Reproduction and next experiment

See [requirements](../requirements/0002-torch-cpu.txt) and
[small runtime pins](../requirements/0002-small-runtime.txt). Install them into an
isolated runtime using CPU wheels; retain full URL/hash receipts and account for
transfers. Do not reinstall the obsolete setup version. At a checked-out source
revision, with numeric thread variables all 2:

```bash
PYTHONPATH=src timeout 900 python scripts/run_bootstapir_dev.py acquire --cache <cache> --output <acquisition-output>
env -i PATH="$PATH" PYTHONPATH=<src> OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 timeout 900 python scripts/run_bootstapir_dev.py decode --cache <cache> --output <decode-output>
env -i PATH="$PATH" PYTHONPATH=<runtime>:<cache>/source:<src> OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 timeout 900 python scripts/run_bootstapir_dev.py measure --cache <cache> --output <measurement-output> --manifest <decode-output>/development-manifest.json --acquisition <acquisition-output>/acquire-receipt.json
# Add --minimal in a separate process for the four-frame replay.
PYTHONPATH=src python scripts/check_bootstapir_measurements.py
```

The [original development plan](../protocols/0002-development-feasibility.md),
revision, per-phase receipts, source/config/input hashes, every output scalar and
latencies are preserved. Binary inputs/weights/upstream code are not in Git.

The [future exploratory protocol draft](../protocols/0002-exploratory-pilot-draft.md)
narrows the comparison to native temporal confidence versus extra motion residual
on identical coordinates. It defines visible opportunities, missing/zero-accept
accounting, prospective paired matched-80%-coverage risk, a two-percentage-point
practical target and video-cluster uncertainty. It remains **unfrozen**: scene
relationships, full allocation manifest, 64-frame development profiling and metric
implementation are still required. No calibration/evaluation video was inferred.
