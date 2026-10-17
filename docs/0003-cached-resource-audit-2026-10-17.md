# 0003 cached-data resource audit: complete inventory, unsupported layout

October 17, 2026. AI-assisted bounded metadata audit under a prospectively published
resource-only addendum. **No model, numeric GT value/point selection, RGB, old-test
analysis or geometry result. Final cohort remains BLOCKED.** No archive was fetched
again, no package installed and no fee/resource/account/schedule change occurred.

## Prospectively published resource change

[Resource input](https://github.com/Simonn777/embodied-vision-reliability/commit/227972d345e8a2af6a40805e0ca5804770520e75)
was published first; separate receipt commit
`8e418de44e1040ce769781881d649a41d498e275` followed BEFORE any numeric GT access.
Canonical receipt `protocols/0003-resource-addendum-freeze-v1.json`, SHA256
`0e1b867fad0a3873ffb8cb6ec64335ce5c2c0dbf3ef779efd8cbe5c553452e04`, binds ten
full hashes including unchanged v1 and timing receipts. V1/timing inputs and all
prior failures remain byte-identical. The new [resource plan](../protocols/0003-resource-addendum-v1.md)
relaxes the 512 MiB *metadata-stop* only: skip large bodies during inventory,
at most one <=2 GiB annotation saved to exclusive temporary disk, <=64 MiB cumulative
uncompressed header/prefix reads including discarded bytes/buffering reserves.
Object/pickle/Fortran/unsupported layouts still reject. Targets, scores, sampling,
quality timing, quarantine and future-cohort requirements are unchanged.

## Actual cached audit

The unchanged cached archive was whole-hash verified again before use:
3,324,284,510 bytes, SHA256
`436e6a5a01702cebcf8be97254ea583457fd60177e2aba671b90ed578912167f`.
Safe stream inventory reached EOF with **14,415 members**, total declared member
bytes **3,334,075,426**, under unchanged 200,000 / 20 GiB gates. Eight derived CSV
parts contain the complete metadata inventory; no member bodies/data are published.
The original header ordinal 2,756 is now observed as
`sample/scene_d78_0318_3rd/anno.npz`, **705,904,917 bytes**. This is a NEW metadata
observation, not a repair or invented reconstruction of the previous failure log.

There are two named annotation parents in this sample, not two established
independent families. Whole sample/unknown linked story/session/asset/multiview
family remains permanently audit/development. Named scenes are not a test cohort.

Fixed lexicographic selection chose **sample/r4_new_f/anno.npz**, 98,527,880 bytes,
SHA256 `382d1edca0d6d88279ed1a61ee63c9ba27b7c5418f2269f4592c1658fd7d8e77`.
Free disk was checked; exactly one regular annotation streamed into an exclusive,
no-link cache file, below 2 GiB. No other annotation extracted; no archive-provided
path used as destination. Raw archive/NPZ stay untracked.

The frozen reader rejected the third NPY header, **valids.npy**, because
`fortran_order=True`. No required numeric prefix was read. A bounded *header-only*
diagnosis then identified the layouts below, without bypassing the rejection or
changing source, convention, points or reader. All earlier failures are retained.
The planned numeric prefix would cover **all points** in the first three frames
of each required array, followed by at most six selected point/frame projection
checks. Six planned checks must not be described as only six GT records accessed.
In this stopped run neither numeric prefix access nor projection selection occurred.

| Array | Shape | Dtype | Order |
|---|---|---|---|
| trajs_2d | 849 × 11721 × 2 | float32 little endian | C |
| trajs_3d | 849 × 11721 × 3 | float32 little endian | C |
| valids | 849 × 11721 | bool | Fortran — rejected |
| visibs | 849 × 11721 | bool | Fortran — rejected |
| intrinsics | 849 × 3 × 3 | float32 little endian | C |
| extrinsics | 849 × 4 × 4 | float32 little endian | C |

Cumulative charged metadata/buffering budget across the audit and header diagnosis
is **590,976 bytes**, well below 64 MiB: 1,152 header bytes returned plus 589,824
bytes of conservative internal buffering reserve. Numeric payload bytes returned
to an array parser: **0**; GT numeric values interpreted: **0**. Raw byte integrity
and ZipExtFile buffering are distinguished from semantic GT access. No whole array,
np.load, pickle, RGB or model. Frame-0 point IDs were **not selected**, not assumed
to be zero; 0 point checks attempted, max/RMS reprojection error **undefined/null**.
Units/FPS and actual sample geometry remain unverified. These headers are not a
99.9% projection/support check, reliable physical acceleration or efficacy result.

Outer resource child: **21.082816 s**, peak RSS **29,820 KiB**, observed exit/raw
wait **1/256** (preserved unsupported-layout stop). Header-only diagnosis:
**0.039508 s**, **11,520 KiB**, exit/raw **0/0**. Two affinity slots, five numeric
variables=2, 6 GiB AS, timeout 900 s. Python 3.12.14, Linux 6.18.55 x86_64; host
CPU recorded in prior audit, AMD EPYC 9V74. Commands/source/input hashes, UTC,
CPU/RSS/wait and logs are retained under
`results/0003-resource-audit-20261017/` and appended resource phase records under
`results/0003-network-followup-20261017/`. One parent wait layer; no further
supervisor receipt is claimed for that parent.

## Software evidence and next action

Independent final metadata/hash/log verification passed in 0.542274 s, peak RSS
26,024 KiB, observed exit/raw 0/0. It verified all 1,267 pre-resource baseline files
(except explicitly mutable state and append-only notes), the full inventory and all
ten resource inputs without GT decode. Its current parent logs/self are excluded
from its own closure map and bound by the publication tree.

Ten handcrafted header/shape/byte-budget/endian/visibility fixtures passed before
resource publication, with all ten resource input hashes recorded. The first
fixture-builder attempt failed on a structured dtype *test constructor*, before
that case reached the reader; original source/actual failed wait and publicly
redacted traceback/raw hash are preserved. Fixing that test is not a dataset rerun.
Work pytest is unavailable (0 tests), no install; final exact-head public CI is a
separate publication gate. No published v1/timing/resource science was edited after
its own input commit. Existing 20% relative planning goal/scales/sample limits stay.

The default network route and archive identity are no longer blockers. Current
blockers are explicit Fortran layout rejection, unverified numeric geometry/units,
unknown independent families and absence of a final cohort/pipeline freeze. No
more reader/resource relaxation or numeric decoding in this task. Reuse the cached
archive/one NPZ for later independently reviewed design decisions; do not retry
acquisition or invent independent groups. Any future permitted layout support must
be prospectively reviewed/frozen and preserve these records. Keep PR16 unmerged.
