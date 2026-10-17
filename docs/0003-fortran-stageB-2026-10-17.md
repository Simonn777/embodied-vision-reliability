# 0003 cached Fortran audit: reader completed, geometry stopped

October 17, 2026. AI-assisted development-data audit, **one actual invocation**.
The bounded C/F reader completed all six required array prefixes. The first fixed
geometry attempt failed the frozen proper-rotation gate, before a pixel or error
was calculated. This is a preserved failed geometry audit, not a projection pass,
independent cohort, model result or reason to change the tolerance on these data.

## Published freeze before numeric access

Approved input commit: `93be3c95dee7cb5d23a5eae9b110a08cfce2a78d`.
Separate receipt/execution commit: `384489d94199a4d0ef316f85ea75684d60a44899`.
Canonical [v2 receipt](../protocols/0003-fortran-audit-freeze-v2.json) SHA256:
`ad6250bb93453b0ff8f0e9ef4e25b853cac99eb37f910ea007fa389f285f2da1`.
It binds the approved nine mandatory inputs, including the unchanged three prior
canonical receipts. Receipt publication and fresh fetch preceded the real frozen
`verify_published()` call: current/index/HEAD/input/remote blobs, remote receipt,
ancestry and all three previous maps (23 distinct inputs) passed. No mock gate.
The actual runner checks that gate again before obtaining its locks and marker.
The publication commit will differ from execution; source/protocol/config remain
those in the approved input, not rewritten to match the outcome.

Initial receipt preparation failed to import `evr` because PYTHONPATH was omitted.
That setup exception is retained in evidence; setting existing `PYTHONPATH=src`
resolved it without install, scientific edit, gate bypass or data access. It was
not a real audit attempt. The subsequent real audit ran once; its one-shot marker
remains in cache, with a derived copy archived. No retry or marker/lock deletion.

## Actual execution and scope

Command (relative to the isolated repository worktree):

```sh
PYTHONPATH=src timeout 900 python scripts/audit_0003_fortran_v2.py \
  --cache ../bootstapir-cache/0003-pointodyssey \
  --output results/0003-fortran-stageB-20261017/audit
```

The parent constructed an otherwise empty environment with inherited PATH (value
not published), PYTHONPATH=src and OMP/OPENBLAS/MKL/VECLIB/NUMEXPR thread variables
all 2. It enforced affinity [0,1] and 6,442,450,944-byte address space; command,
source hashes and one real wait4 are archived. Python 3.12.14, Linux 6.18.55 x86_64,
AMD EPYC 9V74; preflight observed 10,206,484 kB host memory / 9,483,336 kB available,
and 20,460,527,616 bytes free storage. The process limit is the stated 6 GiB AS,
not a claim that all host memory was available to this task.

Actual parent-observed UTC interval: **2026-10-17 23:14:14.644395–23:14:27.804052**.
Wall **13.159663 s**, user CPU **10.703756 s**, system CPU **2.430223 s**,
peak RSS **36,288 KiB**, exit **1**, raw wait **256**. This is the actual frozen
child's outcome, not a post-run verifier's exit. Atomic audit JSON and stdout match
byte for byte. Normal failure completed its finally summary; it is not a timeout.
Parent itself has no additional supervisor exit receipt.

Archive and the existing selected NPZ passed their frozen whole-byte hashes. No
archive GET, dependency install, re-extraction, RGB, model, old-test analysis,
new endpoint/score/threshold fit, fee/account/permission/schedule change occurred.
The previously disclosed acquisition ledger is not incremented by this cache-only
work; historical partial files remain preserved and unused.

## Access and read accounting

This accessed **all 11,721 points at original frames 0,1,2**, not six GT entries:
246,216 retained numeric scalars across trajectories, flags and camera matrices.
The fixed scene remains `sample/r4_new_f`; the other scene's annotation was not
numerically decoded. TAR gzip traversal skipped other bodies without semantic
GT/RGB interpretation. Full-file raw hashing and TAR skips are distinct from the
bounded numeric/metadata semantic-access ledger.

| Array | Retained scalars | Returned numeric span bytes | Discarded gap bytes |
|---|---:|---:|---:|
| trajs_2d (C) | 70,326 | 281,304 | 0 |
| trajs_3d (C) | 105,489 | 421,956 | 0 |
| visibs (F) | 35,163 | 9,950,283 | 9,915,120 |
| valids (F) | 35,163 | 9,950,283 | 9,915,120 |
| intrinsics (C) | 27 | 108 | 0 |
| extrinsics (C) | 48 | 192 | 0 |

Retained numeric bytes: **773,886**. Actual numeric stream spans: **20,604,126**,
including **19,830,240** discarded bytes. Gap values were not interpreted. Spans
are hashed including gaps; unread tails/full CRC are not claimed verified.
Independent arithmetic reconstructs each C/F maximum offset from saved shape/order,
without re-reading the NPZ or calling the candidate geometry helper.

| Cumulative budget component | Bytes |
|---|---:|
| Previously carried returned headers | 1,152 |
| New returned numeric spans + metadata/header/envelope bytes | 20,613,386 |
| Previously carried buffering reserve | 589,824 |
| New 24 × 64 KiB stream reserves | 1,572,864 |
| New implicit central-directory/EOCD reserves | 1,110 |
| **Total charged** | **22,778,336** |

Remaining under unchanged 64 MiB cap: **44,330,528 bytes**. Available budget is not
permission to retry or inspect more data. Counts are complete for this attempted
work, not for all planned checks. Atomic per-array checkpoints remain part of the
frozen runner; no source or tolerance was changed after observation.

## Fixed checks and strict stop

Frame-0-visible finite 2D/3D selection produced original point IDs **[0,1]**. It did
not screen future visibility, camera depth, validity or errors and did not replace
a point after failure. Planned **6**, attempted **1**, unattempted **5**.

| Frame | Original point ID | Validity | Visibility | Outcome | Projection consistency | Valid-visible GT support |
|---:|---:|---:|---:|---|---|---|
| 0 | 0 | 1 | 1 | proper-rotation gate failed | undefined | false |

No projected coordinates or finite reprojection error exists for this attempt;
finite-error count **0**, max/RMS **null**, not 0 px. Zero true consistency/support
rows are counters, not a measured zero error or evidence of a geometric mismatch
magnitude. The preset rotation gate checks both orthogonality products/determinant
at f32 tolerance 9.5367431640625e-7. It threw before returning its diagnostic values;
the offending matrix/statistics were not saved, so the precise magnitude/cause
cannot be independently recomputed from these small artifacts. We did not read
more GT, switch convention/axes, invert/transpose/orthogonalize, adjust tolerance
or rerun. Failure could not be promoted into a 0.5-pixel test outcome.

## Four-member metadata and independence

The exact two JSON members (3,105 and 604 bytes) were parsed as pure metadata and
the two info NPZ containers (1,208 and 1,205 bytes) inspected header-only. Total raw
containers **6,122 bytes**. All four member SHA256 values and parsed key coverage
are recorded; no raw JSON, paths/values or original metadata file is redistributed.
Each JSON has five inspected key paths, including one asset-related key. No
motion/session/story/view/camera matching key was reported. Equality of a hashed
key path proves neither matching asset values nor common/distinct capture identity.
The INFO files each have six C-order int64 NPY headers with shape lengths 2 or 3;
their numeric payloads were not interpreted, no pickle/object decode was attempted.

The graph's UNKNOWN cross-container edge is a **predeclared conservative rule**,
not an empirically verified shared-family identity. Story/session/assets/motion/
multiview relations remain UNKNOWN. Whole sample and unresolved linked family stay
permanently development. Established independent calibration/evaluation families:
**0**; two named containers do not satisfy the design's 10/20 independent families.
Units/FPS and complete checkpoint exposure remain unknown. This is synthetic-source
development evidence, not real-robot or physical-acceleration validation.

## Verification and next step

Independent saved-record stdlib verification passed: 0.205306 s, 16,436 KiB,
exit/raw 0/0. It checks hashes/logs/receipt/baseline, shape-derived spans, discarded
bytes, metadata totals, carried reserves and failed/undefined counts; no archive,
NPZ numeric access, new projection, model or old-result numerical analysis.
All pre-execution protected files remain unchanged. Related synthetic suite:
**76 stdlib tests passed**, 0.740961 s, 30,332 KiB, exit/raw 0/0. Local pytest absent
(0 pytest tests), no install. Exact publication-head public CI is checked separately.

[Raw stopped receipt](../results/0003-fortran-stageB-20261017/audit/audit.json)
and [commands, wait/log/source/hash evidence](../results/0003-fortran-stageB-20261017/evidence/)
preserve the actual outcome. The earlier geometry/layout failures and all stage-A
snapshots remain historical, unchanged. No raw data/weights were published.

Next is independent review of the strict rotation failure and evidence limitations.
Any further small diagnostic or revised camera eligibility needs explicit prospective
review/authorization; it must not silently retry this one-shot frozen audit or tune
on the observed outcome. Final cohort/pipeline, new-development precision feasibility
and independent family evidence remain absent. No model experiment is ready. PR17
stays unmerged for the parent review.
