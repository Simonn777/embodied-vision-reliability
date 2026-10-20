# 0025b prospective physical depth falsifier

The [protocol](../protocols/0025b-rs-depth-semantics.md) tests whether the already
qualified, unchanged MoGe-2 model admits a positive global affine approximation
to instantaneous row-camera depth on manufactured rolling-shutter images.
Independent rendered depth replaces the network-to-network consistency target
of the closed 0024 study. No accuracy result exists at source freeze.

The fixed population is eight structurally selected frames from each of two
official ICL-NUIM clean trajectory-2 layouts. These are development data, not
independent real-world final tests. One GS and four fixed rotational RS inputs
per frame require exactly 80 depth forwards. Complete raw predictions and clean
outer receipts precede any numeric GT depth decoding. Source poses are unused.

Four oracle diagnostic fits share one GT-only support and row-depth LS objective:
row-affine RS, GS-affine then row transport, same-point-middle-affine RS then
transport, and row-scale-only RS. Fixed checkerboard pixels split fitting from
evaluation; masks and prediction errors cannot remove samples. If any of the
64 required paired primary cases is unavailable, the complete primary aggregate
is unavailable. Both scenes must satisfy median AbsRel <=0.10 and median paired
RS-minus-GS AbsRel <=0.02. These are engineering qualification gates, not claims
of pose accuracy or unique depth semantics.

The exact protocol and all execution source bytes must be committed, reviewed
and pass CI before acquisition. Each acquisition, prediction and evaluation is
separately supervised for 1800 seconds, aggregate RSS <=8 GiB and four CPU
threads, preserving incomplete attempts. The two official archives require
731,653,920 response-body bytes within the remaining research allocation. No
new weights, packages, training, paid services or private uploads are admitted.

The [input-format audit](literature/0025b-icl-input-format.md) distinguishes
TUM-compatible axial PNG depth from native ray range and explains the negative-fy
coordinate convention. The exact original PNG-export converter was not located;
the official linked format specification is the documented basis. Original
full archives and image/depth arrays remain private with attribution; small
derived records may be published. Complete archive transport necessarily includes
pose bytes, but no pose text is extracted or interpreted.

A pass permits a new prospective translation/pose study. A failure closes the
tested unadapted approximation without changing these frames, motions, supports
or thresholds. Either outcome leaves the two-source held-out, matched-cost
substantial-gain requirement unchanged. No manuscript is admitted by this stage.
