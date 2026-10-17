# 0003 rotation failure follow-up v1: prospective diagnostic access candidate

October 17, 2026. AI-assisted **failure-follow-up / exploratory diagnostic** prompted
by an already observed development camera-gate failure. Only access/output rules
are prospective; this is not a new confirmatory test. Stage A uses synthetic
fixtures and cached source text only. **No real camera decode or new receipt now.**

## Scope, attribution and publication order

Preserve all old code/protocols/tests/receipts/results and old one-shot marker.
The old audit remains stopped at frame0/point0 before any projected pixel/error.
New versioned helper/runner/tests perform a different operation: only original
frame 0's K (9 scalars) and RT (16) from the SAME cached r4_new_f annotation.
No trajectories/flags/RGB/second scene/later frames, JSON/INFO metadata re-read,
TAR scan/extraction, inference, projection/errors, matrix repair or threshold change.
No install/download/paid service/account/schedule/permission change.

Source NPZ is 98,527,880 bytes, SHA256
382d1edca0d6d88279ed1a61ee63c9ba27b7c5418f2269f4592c1658fd7d8e77,
member sample/r4_new_f/anno.npz from PointOdyssey v1.2 HF revision
4a2ec9492ca732dc3611adbb2a94248cc2bbe3eb. Whole NPZ raw hashing is permitted,
separate from semantic numeric access. No need to rehash the 3.3 GB TAR.

The user's new narrow permission permits publishing exactly these **25 stored
values**, with byte hex/float.hex/exact rational representation and no numeric
modification. This is a narrow new exception to prior no-raw-redistribution scope;
old documents stay unchanged. It does not permit arrays, trajectories, images or
additional frames. Attribute Yang Zheng, Adam W. Harley, Bokui Shen, Gordon
Wetzstein and Leonidas J. Guibas, dataset/version/HF revision/member/hash/source,
and [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Pinned
README line146 establishes dataset terms; the conflicting HF MIT card does not
relax them. The excerpt and derived numeric diagnostic retain these terms, **not
repository MIT**. New original code/tests/design prose remain repository MIT.

Publish the complete candidate input commit/PR and STOP for independent review.
After approval only, publish separate canonical
`protocols/0003-rotation-diagnostic-freeze-v1.json` binding the actual approved input
commit and exact 12-file mandatory hash map in the versioned runner: new protocol,
config, helper, runner, tests, source/rights evidence; four prior canonical receipts;
the prior audit JSON and prior source-convention evidence. No receipt in its own
input commit/map. Verify committed/published current/index/HEAD/input blobs,
ancestry, fixed remote branch and all four earlier chains (30 distinct inputs).
New receipt absent or any mismatch stops before camera/cache payload access.

## Exact bounded extraction and durable access ledger

Carry **20,614,538 returned + 2,163,798 reserved =22,778,336 bytes**, not the much
older 590,976. Historical directory/EOCD subset **1,110** stays separately labeled.
New returns, discarded physical gaps, headers, 64 KiB per camera-member open and
validated central directory +22 implicit EOCD reserves share the unchanged
64 MiB ceiling; 44,330,528 bytes remain before this diagnostic. No reset.

Central-directory names/types/duplicates/path safety may be inspected, but **only
intrinsics.npy and extrinsics.npy are opened**; no other NPY header/payload. Same
bounded header/envelope primitives are reused from the immutable prior helper,
not its three-frame/six-array reader or old budget initializer. Require observed
<f4 C [849,3,3] / [849,4,4] before numeric extraction. Generic extraction supports
explicit C/F and endian strides in synthetic fixtures, but no actual layout fallback.
Retain only (t=0,i,j). Count maximum physical offset span, not 25 kept values; gaps
are streamed/discarded without interpreting them and counted. The real known C
layout needs only 36+64 payload bytes. Per-open buffering may read beyond requested
bytes internally; its 64 KiB reserve is conservative, not measured/decoded values.

With the exact current two 128-byte headers /348-byte directory, planned NEW
returns are 22+256+100=378 bytes and reserves 2*65,536+348+22=131,442 bytes;
planned total 22,910,156 bytes. This is a prospective arithmetic check, not a real
measurement. Unsupported/malicious/oversized/truncated layouts fail closed before
unbudgeted payload; no seek/pickle/np.load/full-array/metadata fallback. No unread
tail/full member CRC claim. Retain each scalar's dtype/order/shape/endian/original
payload offset/stored hex/float.hex and numeric-span hash, including signed zero.

Future run uses shared design/sample locks plus a new diagnostic lock, NEW exclusive
persistent marker and brand-new output. Old marker hash is bound in config and
never deleted/rewritten. Two CPU slots/five numeric thread variables2, 6 GiB AS,
external timeout900 seconds, one real parent wait4 with env/command/source/hash,
UTC/elapsed/CPU/RSS/exit/raw wait/stdout/stderr. Parent itself has no further exit
supervisor. Atomic temp+fsync+replace initial receipt and camera-member/payload
start/done checkpoints precede access. Interruption means incomplete/lower-bound
or unknown counts, never false zero access. Preserve failures and marker; no retry.

## Fixed diagnostic arithmetic and outputs

Record complete exact 25-value excerpt BEFORE arithmetic, attributed as above.
No more data needed. Nonfinite coefficients/unsupported format stop with exact
raw representation retained. Finite input matrices must be source-f32 representable.
Compute the old double formulas with the SAME operation order as project_v2:
R R^T-I and R^T R-I, determinant and abs(det-1), full matrices, row-major first
argmax ties, row/column squared norms and off-diagonal dots. Keep exact homogeneous
K row [0,0,1], RT row [0,0,0,1], positive focal checks. Depth/projection are NOT
performed because world-point access is forbidden. Keep rotation gate separately
from these prefix checks and from diagnostic completion.

Original tau **9.5367431640625e-7=16u**, u=2^-24, remains unchanged for both Gram
maxabs and abs(det-1). Report each pass/fail, max(0,value-tau) and value/tau;
no new acceptance threshold. Independently recompute both Grams and determinant
using `Fraction.from_float` of the SAME stored dyadic values, exact integer
numerator/denominator arithmetic and determinant permutation expansion. Save all
exact statistics, exact gate result and double-minus-exact rational differences;
never refit the old gate from the crosscheck.

If the original rotation gate unexpectedly passes, record inconsistency with the
prior fixed failure and STOP (nonzero), no replacement/repeat. Diagnostic exit0
only means recording completed with the old rotation failure preserved; it is
not geometry passed. All finite checks and diagnostic statistics are saved without
asserting a unique physical/source cause.

## Predeclared narrow rounding hypothesis and invariants

Hypothesis H: Q is EXACT SO(3), each coefficient |Qij|<=1, and only ONE nearest
binary32 rounding occurs. Let R=Q+E, |Eij|<=u=2^-24 (a deliberately conservative
absolute bound, including subnormals). Each Gram-entry error is at most
sum over3 of (2u+u²)=**6u+3u²**. For determinant multilinearity, each error-column
norm <=sqrt(3)u<=3u. Hadamard bounds the three one-error terms by3u each, three
two-error terms by(3u)² each and one triple term by(3u)³, hence
**|det(R)-1|<=9u+27u²+27u³**. Both exact rational bounds are below original tau16u.

Compare exact Fraction statistics to these necessary bounds. Exceeding either
excludes ONLY H: it does not exclude upstream accumulated error, previous lower
precision, export semantics or a reader issue, and does not establish scale/shear
as a unique cause. Staying within bounds does not prove H caused the observation.
No old-data tuning or repaired matrix. The output retains row/column descriptive
statistics without automatic causal labels.

Transposition swaps the two exact Gram products and preserves determinant, so
combined old rotation qualification is unchanged. Orientation-preserving signed
permutations on rows/columns preserve max-entry Gram defects and determinant.
Orientation-reversing sign changes can flip determinant and CHANGE proper-rotation
qualification; this is NOT an invariant or permitted repair. These are analytic
statements and fixed synthetic fixtures only, not real-data transformed candidates
or searches for a passing coordinate convention. Generic orthogonal basis changes
need not preserve entrywise maxabs, so no broader invariant is claimed.

## Source and interpretation boundary

Only existing hash-pinned cached source text is read, never executed/fetched again.
`source-and-rights.json` fixes commit/hash/line ranges and original summaries.
The tracking path reads text K/RT, multiplies export transforms, then explicitly
casts K/RT to float16 before saving. This is a **source-supported possible mechanism**,
not proof that the v1.2 sample/actual frame0 used that code/configuration or an
unscaled Blender camera. Its saved keys differ from the selected schema; precise
generation/conversion history remains unknown. Renderer camera construction and
actual scene state are not verified from this cache. Separate three claims:
(1) directly measured old-gate trigger, (2) mechanisms supported by static source,
(3) unproven root cause/scene generation. Never infer (3) from (1)+(2) alone.

This single development matrix cannot verify full geometry, scene independence,
physical units/FPS, checkpoint exposure or model efficacy. Whole sample remains
permanently development, future cohort blocked. Stage A ends at a reviewable
candidate with artificial fixtures and source/log/wait evidence; no new canonical
receipt or real numeric camera read before independent approval.
