# 0003 resource-only bounded cached-data audit addendum v1

October 17, 2026. AI-assisted prospective resource interpretation, before any
numeric 0003 GT decode. Preserve v1, its timing addendum, all receipts/failures.
Only the following audit resource behavior changes. No score, 20% goal, sample
count, timing/quarantine, future cohort rule or model experiment changes.

Use the existing whole-checksum-verified sample only; no further archive GET.
Stream safe TAR metadata through EOF without allocating member bodies. Keep
path/type/duplicate, 200,000-entry and 20 GiB total declared-size gates unchanged.
Large annotation headers are recorded/skipped rather than triggering the legacy
512 MiB immediate metadata-stop. Record exact previous header 2,756 name/size as
a NEW observation, without rewriting the original failure. Complete inventory
proves names/metadata, not independent source families.

Choose the lexicographically first canonical scene-parent path containing one
regular `anno.npz`, `annot.npz` or `annotations.npz`. Never inspect several scenes
based on errors. If multiple candidates for that scene are ambiguous, or the one
file lacks required numeric camera/2D/3D schema, stop rather than fallback. Entire
sample/unknown linked scene/story/session/multiview/asset family is development.
Metadata selection alone does not certify independent scenes. No RGB is needed.

Stream at most ONE selected regular annotation to an exclusive no-link temporary
cache file, maximum **2 GiB on disk**, with free space >=member size+512 MiB reserve
checked first. This is not a RAM allowance. Over-cap or insufficient disk stops;
no further resource expansion this task. Record full annotation member hash/bytes.
No extraction to archive-provided paths, links/devices or arbitrary executables.

Inspect NPZ central directory and bounded NPY headers only before values. Maximum
64 regular NPY members; reject encrypted entries, paths/links, unknown compression
(except stored/deflate), object/structured/complex dtype, Fortran order, unsupported
shape/layout, or headers >64 KiB. Real finite-layout numeric types f2/f4/f8 and
signed/unsigned 1/2/4/8-byte integers or bool are supported with explicit endian.
No pickle and no whole NPZ/array load. Only C-order time-major prefixes for first
THREE frames: trajectories [T,N,2]/[T,N,3], visibility [T,N]; optional validity
[T,N]. Camera K [T,3,3] or constant [3,3]/[1,3,3], T_cw [T,4,4] or constant
[4,4]/[1,4,4]. T>=3 and matching point/time counts required; unknown layouts stop.
Visibility aliases exactly visibs/visibilities/visibility (one only); value1 visible,
0/2 not visible. Optional validity alias valids/validities (one only). Required
trajs_2d/trajs_3d/intrinsics/extrinsics exact names; no outcome-driven substitutions.
All member headers are checked; unrelated numeric C-order headers are not decoded.

Cumulative uncompressed bytes returned/read, INCLUDING discarded/header bytes,
<=**64 MiB across all arrays/stream opens**, plus charge a conservative 64 KiB
internal ZipExtFile buffering reserve per open *within* that same limit. Read
only necessary prefixes; never np.load a full archive/array. Keep two CPU slots,
five numeric variables2, 6 GiB AS and external timeout900 s. No package install.

Select first two original point IDs with frame0 visibility==1 and finite frame0
GT2D/GT3D, solely those fields, without future visibility/error selection. Keep
zero/one-query outcomes; never replace a selected point after projection failure.
First three original frames only, no resampling; record all six or fewer checks,
validity flags, schema/dtype, original-raster reprojection errors/max/RMS and all
invalid reasons. K/T use the convention verified in pinned official reprojection
code. Run the unchanged v1 projection helper as its strict gate; if that gate or
projection tolerance fails, stop/record it, no orthogonalization/convention tuning.
Independent direct arithmetic may expose the reason but cannot certify a failed
helper as passed. Invalid/missing anchors are not replaced. These tiny checks are
not a whole-dataset 99.9% gate, model improvement or independent-cohort result.
World units/FPS stay unverified unless explicit source/metadata evidence establishes
them; no m/s² or predicted 3D drift claim. No GT enters model scores or inference.

Publish new inputs first, then independent canonical resource receipt referencing
the real commit and all hashes including unchanged v1 AND timing receipts. Numeric
GT decode starts only after that receipt is published and byte-verified. Bounded
reader/header/cap fixtures are software tests, not GT access. All new observations
and failures go to new append-only evidence paths; frozen earlier files stay intact.
