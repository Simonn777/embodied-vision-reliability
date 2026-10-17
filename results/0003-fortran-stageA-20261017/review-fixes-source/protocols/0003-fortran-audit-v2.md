# 0003 development geometry audit v2: C/F scalar streaming (review candidate)

October 17, 2026. AI-assisted **prospective stage A**, not yet a freeze or real-data
run. Preserve all v1 design, timing and resource inputs/receipts, failed runs and
old 0002 results byte-identically. This replaces only the C-only annotation reader
and strict rotation-storage precision gate for a separately authorized development
audit. No old test analysis, new model, predictor score/goal change or cohort.

## Order and exact inputs

Stage A reads only existing published headers/inventory, cached pinned upstream
source text and raw-file byte hashes. It implements and tests the new version
using artificial fixtures. **No real numeric GT decoding, query selection or
geometry calculation occurs in stage A.** No new canonical v2 receipt is created.
Publish the complete candidate input commit/PR and STOP for independent review.
After approval, publish a separate canonical receipt
`protocols/0003-fortran-audit-freeze-v2.json` referencing that actual input commit
and all nine mandatory input hashes in the versioned runner. Exclude the receipt
itself from its inputs and input commit. Verify current/index/HEAD/input blobs and
freshly fetched published branch before the one real audit. Bind the unchanged
three canonical receipts and every prior input blob/hash. Input changes during
review require a new candidate, not silent edits after receipt publication.

Use only the existing verified archive (3,324,284,510 bytes, SHA256
436e6a5a01702cebcf8be97254ea583457fd60177e2aba671b90ed578912167f) and one NPZ
`sample/r4_new_f/anno.npz` (98,527,880 bytes, SHA256
382d1edca0d6d88279ed1a61ee63c9ba27b7c5418f2269f4592c1658fd7d8e77).
No GET, archive retry/re-extraction, dependency install, RGB, model, old-test row
access, payment, schedule or permission change. A mismatch stops without fallback.
Old partial cache files are preserved, unused; this task does not rewrite the
historical acquisition ledger or claim that all historical HTTP attempts are known.

## Exact access and bounded reader

Known header schema: T=849, N=11,721, trajectories [T,N,2]/[T,N,3], byte visibs and
valids [T,N], intrinsics [T,3,3], extrinsics [T,4,4]. Require these six keys and
aligned shapes before numeric reads; camera constants [3,3]/[1,3,3] or [4,4]/[1,4,4]
are explicit supported layouts. Only f32/f64 camera/trajectory arrays; explicit
endian and C/F strides, no pickle/object/structured/complex arrays, Fortran reshape
shortcut, np.load or whole trajectory array in RAM. Metadata-only object headers
are recorded but never interpreted. Numeric camera convention is fixed below.

Read all points' original frames 0,1,2 from each required array into a small logical
C-order prefix. **This accesses many GT entries, not only six records.** C strides
multiply later dimensions; F strides multiply earlier dimensions, e.g. byte
[T,N] offset=t+T*p. Stream in increasing physical offset; count every byte returned
or discarded, all repeated headers and 64 KiB internal-buffer reserve PER open.
F flags may require scanning almost the full 9,951,129-byte payload to retain three
frames; discarded values are not semantically decoded. F trajectory/camera arrays
whose required physical span exceeds budget stop BEFORE their numeric read.
At most 1,000,000 retained scalars; max 64 KiB streaming block; no seek shortcut.
Returned-span hashes cover prefix PLUS discarded gaps, not unread tails/full CRC.

**Carry forward 590,976 bytes** from the previous audit/diagnosis: 1,152 header bytes
plus 589,824 reserved bytes. The NEW run's numeric/header/discard/container bytes and
reserves share the remaining **66,517,888 bytes** of the unchanged 64 MiB total.
No reset. Small raw metadata-container bytes/envelope reads are conservatively
charged too. Before each ZipFile construction, additionally reserve the validated
central-directory byte size plus 22 bytes for its implicit EOCD read; this is a
conservative metadata reserve, not a claim of measured implicit reads. Zip internal buffering is reserved, not claimed exactly measured.
Reject header >64 KiB, unsupported NPY versions/dtypes/shape arithmetic, truncation,
unsafe/duplicate/NUL/path/link/encrypted ZIP members, unsupported compression,
central directory >64 KiB or >64 members, ZIP64/comments/multiple disks, length
mismatch, unsafe TAR entries, >200,000 TAR members or >20 GiB declared sum. A
late failure remains a failed audit with prior attempts and partial access saved.

Select at most the first two ORIGINAL frame-0 visibility==1 point IDs with finite
GT2D/GT3D. No validity/depth/future-visibility/error screening, zero/one-point
replacement or additional scene. Preserve zero/one-query outcomes. For those fixed
IDs, attempt frames 0,1,2, at most six checks; selected invalid validity, nonfinite
GT, invalid camera, depth or projection error stops with the failed row retained.
No fallback points. Visibility at later times is recorded, not used to reselect.
Record projection consistency separately from valid-visible GT support: support
requires source validity==1 AND visibility==1 plus finite geometry/valid projection
AND error<=0.5 original px. Projection consistency and valid-visible support are
separate counters; a failed projection tolerance never supplies valid GT support.
An accurate projection at visibility 0/2 is not visible GT support. Invalid validity
stops with support=false and projection consistency undefined if not computed.
Counts refer only to attempted selected checks, not the whole-family support gate.

## One source convention, prospective arithmetic gate

Pinned generator revision a053342a72fed32af4134b4ca0c44a78a3ac0fdb:
`utils/reprojection.py` lines 26–33 forms homogeneous world points, applies RT,
normalizes camera XYZ by camera z, applies K, and keeps xy. Lines 66–68 read saved
3D/K/RT directly; the version-1.0 conversion is commented at lines 81–85. Pinned
`utils/gen_tracking_indoor.py` lines 156–157 construct the export transform before
saving; lines 186–220 save the transformed extrinsics and world tracks. Do NOT
apply that export conversion again to saved data. Cached source SHA256/line
ranges are in the mandatory source-convention JSON. Source inspection verifies
an implementation convention, not this archive's actual geometry/units/time scale.
No transposing, inversion, axis flips, orthogonalization or convention searches.

New `project_v2` validates finite f32/f64 stored matrices and intermediates, exact
RT row [0,0,0,1], exact K row [0,0,1], positive focal and camera depth. For BOTH
max abs(R^T R-I), max abs(R R^T-I), and abs(det(R)-1), tolerance is
**max(1e-8,8*eps(storage dtype))**, taking the less precise of K and RT: f32
9.5367431640625e-7, f64 1e-8. This prespecified rounding margin for nine order-one
rotation coefficients is independent of real GT results. No fitting/relaxation on
failure; the old 1e-8 helper remains historical and is not used as new execution
authority. A synthetic float32-rounded 45-degree fixture must expose the difference;
1e-4 scale/shear and reflections must fail. Overflow/nonfinite arithmetic fails.
The unchanged strict error gate is **<=0.5 original raster pixel**, no resize/255
scaling; save each error, including failures. In finally summarize ALL attempted
finite errors (passed or failed) for max/RMS, null when none, and record planned/
attempted/unattempted counts. Planned is unknown/null before selection, not zero. Small checks do NOT establish
the whole-dataset 99.9% gate. Units/FPS stay unknown absent explicit pinned evidence;
no m/s² or independent physical-motion result follows.

## Family metadata, separately disclosed access

After receipt publication only, read from the existing TAR the exact four members:
- sample/r4_new_f/scene_info.json, 604 bytes; info.npz, 1,205 bytes.
- sample/scene_d78_0318_3rd/scene_info.json, 3,105 bytes; info.npz, 1,208 bytes.

Hard caps 16 KiB/member, 64 KiB total raw containers, JSON depth 12 / 2,048 nodes;
all bytes count against the same 64 MiB carried budget. INFO NPZ <=32 members,
header-only NPY parsing; object payloads never pickle/guess-decode. Reject duplicate
JSON keys/nonfinite literals/unsafe containers. TAR streaming skips all other bodies
without semantic decoding, including the second scene GT; no RGB. Archive physical
hashing/TAR gzip skips are disclosed separately from numeric array interpretation.

Save member hashes, read-field/key-path hash coverage by asset/motion/session/story/
view/camera category and explicit unknowns, not raw JSON, private strings/paths,
source pages or original annotation arrays. Presence/absence of such fields does
not prove shared/unshared families. Graph has two named containers, conservatively
connected by UNKNOWN linkage; whole sample and unresolved linked family remain
DEVELOPMENT even if hashes differ. Verified independent calibration/evaluation
families=0 until external identity evidence establishes otherwise. Two containers
cannot satisfy 10 calibration / 20 evaluation families. Pretraining exposure stays
unknown. Metadata and numerical/RGB/prediction access are logged separately.

## Execution, preservation and readiness

Future one-shot runner: verify published receipt FIRST; cache-wide/design/sample
and group flock, exclusive persistent marker and brand-new output directory.
No deletion/retry on failure. Two affinity slots/five numeric variables=2, 6 GiB
AS, external timeout900 s, one real parent wait recording source/hash/command,
UTC/wall/CPU/RSS/raw wait/exit/stdout/stderr; parent itself has no extra supervisor.
Save every stopped/partial audit and exact access ledger. Freeze source/config/
protocol and old input/receipt bindings before reading any numeric data. Publish
only small original derived records. All previous protected bytes remain untouched.

The unchanged design/timing targets and future cohort gates remain binding. This
is development geometry/metadata evidence only; no final cohort, inference,
continuation result, new improvement policy or GT-quality pass is authorized.
Stage A ends at a reviewable PR with synthetic fixtures and candidate hashes.
