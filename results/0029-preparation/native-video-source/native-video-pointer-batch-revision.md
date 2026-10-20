# Prospective native-video pointer-batch check correction

2026-10-20. Source-only correction approved by the parent before scientific
video input or model execution. The unchanged earlier draft is
`native_video-before-pointer-batch-fix.py`, SHA256
`8f5b51e0a00f3063267b2fc634a21f2a12792082da7862076322d043045cf860`.

The earlier deletion comparison required retained projected pointer-position
hashes to match the full batch. Removing one pointer changes the projection's
batch shape, so identical time coordinates and weights need not produce
bit-identical floating-point results across those shapes. The existing
manufactured 7-to-6 pointer check passed; it did not establish this stronger
invariant for all batch sizes.

The revised observer still compares actual attention inputs exactly against
the unchanged native formula for each actual batch. Deletion checks still
require exact retained spatial features, spatial positions, pointer features,
frame identities, integer temporal distances and normalization denominators.
Cross-batch projected pointer-position hash differences are retained as
descriptive records; they no longer reject the operation. No tolerance,
projection rewrite, new time coordinates, backfill or model-setting change was
introduced. Same-batch baseline/no-op parity remains exact.

No module import, test, model call, image, witness, annotation or scientific
array access was performed for this source correction.
