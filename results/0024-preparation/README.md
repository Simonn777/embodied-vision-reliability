# 0024 source preparation: artificial checks only

The combined final source passed **40 manufactured tests** in 0.217 seconds
(unittest), 0.348 seconds including the enclosing test process. The source hashes
were unchanged before/after. These are geometry, failure-accounting, source-
binding, fake-inference and process-supervision tests; no scientific image,
model inference, GT, pose estimation or accuracy measurement was involved.

The first 11-test orchestration check failed one synthetic edge case: for an
invented image width greater than 2^25, NumPy float32 comparison could round the
integer bound and accept a coordinate rounded outward. The actual source
intrinsics parser already caps dimensions at 16,384. Nevertheless, the general
remap guard was fixed to compare float32-roundtripped coordinates in float64.
The same 11 tests then passed, followed by the combined 40-test check. Both
original attempts remain preserved with their original exit status. No camera
transform, depth estimator, threshold or scientific data changed.

The separate supervisor tests initially had a macOS /var versus /private/var
fixture-path mismatch; resolving the artificial fixture path produced 12
passing tests without a production-code change. The independent mathematical
review separately checked exact offset recovery, the correct zero-scale
boundary, the strong two-parameter numerical-rank baseline and homogeneous
remapping; see [review](../../docs/0024-protocol-review.md).

[Publication bindings](publication-bindings.json) record original/private and
public hashes with explicit literal path-prefix redaction. This preparation
supports a prospective source freeze, not the scientific or manuscript gate.
