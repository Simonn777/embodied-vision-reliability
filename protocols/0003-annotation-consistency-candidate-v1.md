# 0003 candidate: descriptive annotation self-consistency after rigid failure

**UNFROZEN Stage A candidate; no execution authority.** October 17, 2026 client
date (the evidence preserves exact UTC timestamps). AI-assisted failure-follow-up
on permanently quarantined development data. It is prompted by the known failed
camera gate and is not an independent confirmation, new policy or improvement test.
Original designs, five receipts, all failures and thresholds remain immutable.

## What the source audit can establish

Use only the five already cached official text bodies and cached source-tree
listing at a053342a72fed32af4134b4ca0c44a78a3ac0fdb, bounded at 64 KiB per body
and 256 KiB aggregate; statically read, never execute. Check SHA256 and Git blob
identity against that cached tree. Existing local Git objects may be inspected
without fetching. Missing body/history means **not examined in this cache**, never
that a converter does not exist anywhere upstream. No new network data, archive,
NPZ, metadata member or RGB access is authorized in Stage A. Published camera
excerpt, diagnostic and Fortran JSON can be read, with their hashes recorded.
Source text and saved JSON are separately accounted from numeric-data reads.

## New endpoint; original rigid failure stays failed

The candidate asks only whether the published 2D annotations agree with the
algebraic projection using **unchanged stored** K, RT and 3D annotations. This is
**descriptive annotation self-consistency with a camera that may fail rigidity**.
It cannot certify a physical/rigid camera, correct GT, units, true generation
history, independent scenes or readiness of the frozen motion experiment.
A mutually wrong camera/annotation chain can be self-consistent.

Recompute and record the original finite, exact homogeneous rows, positive focal,
R Rᵀ, Rᵀ R and determinant gates, using the original f32 tau
9.5367431640625e-7. The known frame 0 `proper_rotation_fail` must remain true and
its K/RT exact bytes must match the saved 25-value excerpt; a mismatch stops as an
identity/inconsistency failure. For frames 1 and 2 report their own unchanged gates
without presuming the outcome. The rigid failure is **not changed to pass** and is
not silently ignored: it remains an explicit disqualifier for rigid-camera claims.
The new endpoint deliberately permits recording algebraic annotation consistency
under that failure; only separate prospective approval/freeze can authorize this.
No SVD/orthogonalization, normalization/repair of R, half roundtrip replacement,
transpose/inverse/axis/coordinate-convention search, tuning or best-result selection.

## Fixed tiny allocation and exact access

Same cached `sample/r4_new_f/anno.npz`, 98,527,880 bytes, SHA256
382d1edca0d6d88279ed1a61ee63c9ba27b7c5418f2269f4592c1658fd7d8e77.
Reuse **frames [0,1,2], original point IDs [0,1]**, already allocated in the immutable
Fortran audit JSON; no all-point rescreening or new visibility-dependent selection.
Order rows by frame then point ID, maximum six. No replacement after any failure,
zero/one-point fallback, additional scene, frame, RGB or metadata. Entire sample
and unknown linked families remain permanently development.

A future versioned sparse streaming reader must interpret only these fixed rows
and the same frames' original K/RT. It must explicitly support the verified C/f32
trajectories/cameras and F/byte flags with exact stride offsets, reject unsupported
layouts before payload access and charge every physical gap. It must never use
np.load/pickle or load whole arrays. The known [849,11721] flag layout physically
requires 852 bytes per flag member to retain six flags at t=0..2,p=0..1: offsets
0,1,2,849,850,851. The discarded bytes include other times of point 0; they are not
interpreted or used for selection. C trajectories similarly stream across other
points between the chosen rows. Claim neither that only six GT rows were physically
read nor that the discarded values were numerically inspected.

For the saved schema, maximum payload spans are 187,552 / 281,328 bytes for 2D/3D,
852 each for visibility/validity, and 108 / 192 for K/RT: **470,884 bytes**, of which
432 bytes / 117 scalars are retained (42 point/flag scalars plus 75 camera scalars).
All remaining physical bytes are charged and discarded. Required headers are
validated in the same six member opens, not a second payload pass. Inspect only
central-directory names for other entries; exact six-member schema required.
This is a prospective calculation from published shapes, not a new data read.

## Projection, flags, missingness and stopping

For each unchanged world point X, compute c=RT·[X,1], require finite c and c_z>0,
then q=c[:3]/c_z, uv=(K·q)[:2], in original pixel coordinates. No scaling to 256,
clipping, inverse operation or fitted convention. Preserve original 2D/3D finite
checks, exact affine/K rows and positive focal. Record stored flags; validity must
be exactly 1 or stop, and visibility must be one of 0,1,2 or stop. Known selected
IDs are never changed if flags differ. Visibility 0/2 does not get counted as
visible support; numerical projection consistency is separately reportable.

Descriptive discrepancy is the finite Euclidean distance from uv to stored GT2D;
unchanged criterion **<=0.5 original pixel**. Record signed dx/dy, projected uv,
stored camera depth, discrepancy, flags and original rigid-gate statistics. A
nonfinite/unsupported required input, nonpositive depth, overflow or >0.5 discrepancy
stops the one operation, saving the failed attempted row and all unattempted rows.
No rerun, threshold adjustment, alternative point or convention. Each completed
row has separate `projection_consistent`, `valid_visible_GT_support` (finite,
validity=1, visibility=1, positive depth and <=0.5), and
`rigid_qualified_support` (also original rigid gate passes). Frame 0 cannot qualify
for the last field even if projection agrees. Report both descriptive and rigid
outcomes rather than a single misleading `geometry_pass`.

Save planned/attempted/unattempted counts, finite-error count, all finite attempted
errors, max and RMS initialized null and finalized even on failure, and counts of
both support definitions. No frame-independent inference, CI or full-dataset 99.9%
claim can be made from at most six checks. No model effect or method comparison.

## Resources, freeze and durable failure evidence

Carry the latest exact ledger: **20,614,916 returned + 2,295,240 reserved =
22,910,156 bytes**, with directory-reserve subset 1,480; remaining 44,198,708 of
67,108,864. Do not revert to an earlier carry. Six 128-byte headers +22 envelope
+470,884 payload span yield 471,674 new returned bytes. Six 65,536 stream reserves
plus 348 directory +22 EOCD yield 393,586 new reserved bytes; prospective cumulative
charge **23,775,416**. All implicit buffering/header/reopen charges count. Extra
opens must fail unless explicitly accounted within this frozen prospective plan;
no silent budget reset or expansion. Unsupported/truncated layouts stop before
unbudgeted reads. No tail/whole-member CRC claim without actual consumption.

Before any future numeric access: independently review this candidate, implement
and synthetically validate the exact sparse reader/projection in new versioned
files (old helpers/runners remain immutable), publish complete inputs first, then
an independent canonical receipt binding their real commit, exact mandatory hashes,
all five prior canonical receipts, saved camera/diagnostic/Fortran JSON and this
source audit/config/protocol. No self-reference. Verify current/index/HEAD/input
Git blobs, published receipt/remote ancestry and prior chains before opening data.
**No canonical receipt is created this Stage A.** This text/config alone cannot
substitute for the complete reviewed executable freeze.

Use shared design/sample flock and new operation-specific lock, exclusive persistent
marker and unique output. Keep every old marker. One invocation, no automatic retry;
2 CPU slots/five thread variables 2, 6 GiB AS, external timeout 900 seconds. Atomic
start/member/payload checkpoints precede access; interrupted counts are incomplete
lower bounds/unknown. Preserve actual parent wait4 exit/raw status, UTC, wall/CPU/RSS,
commands/environment/source hashes and raw stdout/stderr, with no recursive
supervisor claim. Save actual errors/failures; exit semantics distinguish recording
from the failed rigid gate. No download/install/paid resource/permission/schedule
change. Public outputs are only small lawful derived evidence with PointOdyssey
CC BY-NC-SA 4.0 attribution, not original dataset/video/weights or repo-wide MIT.

Units/FPS, actual conversion chain and scene independence remain unknown. The
separate held-out study and its timing/GT gates remain blocked/unchanged. A possible
future parent-supplied rational-quaternion/half-cell compatibility certificate is
not executed here, is not a repaired camera and cannot prove historical generation.
