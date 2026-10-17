# 0003 Fortran audit: stage A review candidate

October 17, 2026. AI-assisted implementation and synthetic software validation.
**No real GT numeric decoding, selected points or geometry checks. No v2 receipt
exists, and the real audit has not been invoked.** This candidate ends at review.

Base main: `863c7f43ee1ec2567126a79f15c6b48b0dac2d0e`. Isolated branch
`research/0003-fortran-design-20261017`; original worktrees, data and failures
preserved. Visible process inspection found no active research child; existing
cache-wide design/sample locks were obtainable. This is bounded visible-process/
lock evidence, not proof about every inaccessible prior session. Repository open-PR
search returned none before creation. No schedule/account/permission changes.

Raw-byte hashing reverified the existing 3,324,284,510-byte sample and the one
98,527,880-byte selected NPZ against published SHA256. It does not interpret GT
values. Old partial cache files remain untouched/unused. No download or install;
no additional acquisition beyond the previously disclosed conservative ledger.

## Candidate changes

- [Protocol](../protocols/0003-fortran-audit-v2.md) and
  [configuration](../configs/0003-fortran-audit-v2.json) specify a separately
  reviewed development audit, with no changes to old science.
- Versioned `point_geometry_v2.py`, `audit_0003_fortran_v2.py` and tests leave
  the old C-only reader/runner/tests byte-identical. Explicit physical C/F scalar
  offsets preserve logical time/point IDs, count discarded gaps and precheck spans.
- Carry prior 590,976 bytes into the unchanged 64 MiB total; all new headers,
  returned/discarded bytes, metadata containers and per-open reserves count.
- Prespecified rotation tolerance max(1e-8,8*eps), both orthogonality products and
  determinant, exact K/RT bottom rows, finite/positive focal/depth and unchanged
  0.5 original-pixel projection gate. No convention fitting or replacement points.
- Stage B would read all points' first-three-frame prefixes, not just six GT
  entries, then compute at most six fixed point/frame projections. Stage A read
  neither. Whole sample/unknown linked family remains development, not a cohort.
- Four exact small metadata members are prospectively bounded. Pure JSON parsing
  and info-NPZ headers may disclose relation-key coverage; object payloads remain
  uninterpreted. Unknown links keep the two named containers conservatively joined.

The sole source convention comes from cached pinned source, not data-driven choice:
[reprojection.py lines 26–33](https://github.com/y-zheng18/point_odyssey/blob/a053342a72fed32af4134b4ca0c44a78a3ac0fdb/utils/reprojection.py#L26-L33)
uses world homogeneous columns, saved RT, camera-z normalization, K and xy. Saved
arrays are read directly at lines 66–68; the v1.0 conversion is commented at 81–85.
[Generator lines 156–220](https://github.com/y-zheng18/point_odyssey/blob/a053342a72fed32af4134b4ca0c44a78a3ac0fdb/utils/gen_tracking_indoor.py#L156-L220)
apply export coordinate conversion before saving RT. Applying it again is forbidden.
Full source hashes/line ranges are in the [mandatory source evidence](../results/0003-fortran-stageA-20261017/source-convention.json).
Actual archive geometry, units/FPS, independent families and checkpoint exposure
remain unverified/unknown. No m/s², model improvement, robot or confirmation claim.

## Actual software checks

Python 3.12.14, Linux 6.18.55 x86_64, AMD EPYC 9V74; affinity slots [0,1], five
numeric thread variables=2, 6 GiB AS, external timeout900 s. Real parent wait4
records bind source snapshots, command, UTC, wall/CPU/RSS, raw exit and logs.
Parent's own exit has no further supervisor receipt. Runtime PATH values are
omitted publicly; the first parent receipt records its raw/public distinction.

Initial complete cloud stdlib suite: **75 tests**, including **19 new cases**, all passed
(0.920585 s outer wall, 31,104 KiB peak RSS, exit/raw 0/0). It also runs unchanged
35 design, 11 timing and 10 C-only resource fixtures. New cases cover asymmetric
C/F (4,2,2), both endian orders, exact/short physical budgets, carried reserves,
truncation/schema/unsafe members, zero/one queries and future visibility invariance,
missing receipt rejection before cache access, object header-only semantics,
float32-rounded rotation versus old tolerance, scale/shear/reflection/overflow,
exact homogeneous/depth/focal gates and fixed half-pixel/no-replacement behavior.
Earlier successful 15/17-case source states are retained. The latest semantic-review
rerun keeps the same 75/19 tests: 0.705281 s outer wall, 31,308 KiB, exit/raw 0/0.
An existing synthetic round-trip fixture now verifies six accurate projections
but only five valid-visible support rows after one later visibility flag becomes
occluded. These are distinct fields/counters; no accurate-but-invisible row is
counted as visible GT support, and no real-data point is selected or replaced.

Work pytest unavailable: **0 local pytest tests**, no install. Existing toy smoke
using the already cached runtime completed **1,240 synthetic frames in 3.18 s**
(3.555870 s outer wall, 90,776 KiB, exit/raw 0/0). No PointOdyssey/model run occurs
in that smoke. Standard public exact-head CI is the complete pytest gate.

[Evidence directory](../results/0003-fortran-stageA-20261017/) contains the protected
baseline, cached identity/source evidence, exact test source states and parent/log
records. Final publication audit verifies baseline bytes and append-only notes.
Only small original text/derived evidence is published, no original data or weights.

Next: independent review of this candidate. Only after approval publish the separate
canonical v2 receipt binding the real input commit and all nine mandatory files,
verify its published/index/HEAD/worktree/remote bindings, then consider the one
cached audit. Do not invoke the runner or create a receipt in stage A. The present
candidate is not a final cohort/pipeline or real-data execution authorization.

## Final pre-freeze review fixes

Projection consistency and valid-visible support now both require the unchanged
0.5-pixel bound; visibility/validity are separately explicit, and invalid validity
stops without replacement. A finally summary retains every attempted finite error,
including the failed value, with max/RMS/null and planned/attempted/unattempted
counts. The envelope additionally reserves the validated central-directory bytes
plus 22 implicit EOCD bytes, explicitly a conservative reserve, not measured reads.

Receipts now use same-directory exclusive temp + flush/fsync + atomic replace.
Metadata/member starts and numeric/array starts and completions are durable. An
interrupted record is incomplete with lower-bound/unknown counters, never proof of
zero GT. One added artificial subprocess fixture reads synthetic payload bytes then
is terminated by SIGTERM; the atomic JSON remains parseable and truthfully pending.
No real cache is read by that fixture. Latest source-bound parent/log records are
final-review; all earlier candidates, successful tests and publication checks remain
historical. The canonical v2 freeze receipt remains absent.

Latest final-review: 76 stdlib tests (20 new) passed, outer wall 0.797853 s,
peak RSS 31,064 KiB, exit/raw 0/0; synthetic SIGTERM child observed return -15.
