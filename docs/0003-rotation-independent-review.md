# Independent review: frame-zero rotation failure

October 17, 2026. AI-assisted publication review of candidate
`bb5fe92fe38835266d660f63b159583151e64695`. **Approved for publication of this
bounded failure diagnostic. Geometry and the generating root cause remain unverified.**
The [cloud report](0003-rotation-diagnostic-stageB-2026-10-17.md) and every original
Stage A/B artifact are retained byte for byte.

## What is established

The prospective diagnostic input `8a7591b` was reviewed before the separate receipt
commit `8ffb8bc`. Its 12 mandatory inputs and four historical receipt chains
(30 distinct prior inputs) match the recorded preflight. The one cloud invocation
then read the same cached NPZ, decoding only frame-zero K9+RT16: **25 values /
100 numeric payload bytes**. It completed in 0.466789 seconds, peak RSS 16,128 KiB,
observed child exit/raw status 0/0. Exit 0 means recording complete; the original
proper-rotation gate still fails. There was no projection or model inference.

At unchanged tau `9.5367431640625e-7`, the RRt, RtR and determinant defects are
`0.0003642439842224121`, `0.0003985762596094844` and
`0.000009985611434615294`, respectively: **381.9375, 417.9375 and 10.4707 times tau**.
A separate local checker reproduced all 18 exact Gram entries and the determinant
from the published bytes, using Fraction Gaussian elimination rather than the
production determinant expansion. All reported double-minus-exact differences
are zero for this matrix. The local saved-record check took 0.081646 seconds,
returned 0 and produced empty stderr; it is not another cloud data run.

Independent rational lattice membership confirms K9/9, R9/9, RT14/16,
translation1/3 and homogeneous4/4 are exactly binary16-representable. Only
RT[0,3] and RT[2,3] fail. This agrees with the frozen roundtrip-based descriptor
without using its implementation or converting a matrix for evaluation.

The predeclared necessary bounds rule out only **a single final nearest-f32 rounding
of an exact proper rotation**. R's half-grid membership and the pinned source's
float16 saving operations support a possible earlier precision mechanism; neither
proves this sample's generation history. Key/schema differences and intervening
conversion or computation remain unresolved. The actual generating cause is
unknown, and no tolerance correction, matrix repair or performance claim follows.

## Provenance and preservation

The [parent review](../results/0003-rotation-publication-review/parent-review.json)
checks the complete 19-artifact Stage B map, source/log/wait bindings, receipt-only
freeze commit, and accounting. All 1,512 protected execution-baseline files,
including 1,390 result files and 37 Stage A artifacts, remain unchanged; the dated
note is append-only. This includes the original failed audit, thresholds and
freeze inputs. Cloud records retain both the old marker and new one-shot marker.
Cumulative charged bytes are 22,910,156 / 67,108,864, including conservative buffers.

The [numeric review](../results/0003-rotation-publication-review/numeric-review.json)
records the original-byte, offset, float.hex and rational checks, exact arithmetic,
independent half-grid result and source/output hashes. The [bundle](../results/0003-rotation-publication-review)
contains byte-identical review sources, inputs, outputs, actual execution receipt,
attribution and a manifest. No archive/cache/model access or new GT sample was
needed for independent review. Rights/source review also passed.

Candidate CI [39080399856](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39080399856)
passed at merge checkout `82c472b` containing candidate `bb5fe92`: **290 tests in
9.31 seconds** and the existing **1,240-frame synthetic smoke in 4.49 seconds**.
These are software checks, not additional PointOdyssey geometry evidence. Final
review-only publication and main CI are verified separately in PR18.

## Boundaries and next discriminating work

- The frozen runner retains a stale top-level STARTING label; completed member
  states, complete counts and overall completion are separately present. The
  original report explicitly discloses this; no artifact was rewritten to fix it.
- The report's phrase "no metadata was reread" means **no metadata JSON/INFO**.
  ZIP directory/envelope and the two NPY headers were read and charged, as the
  same report's accounting specifies. This clarification preserves the original.
- Public parent receipts normalize absolute repository source-key prefixes and
  retain hashes of the private originals. Current source/log bindings can be
  independently checked, but this review cannot verify the pre-normalization
  private bytes. Receipt chronology is recorded Git/run evidence, not an external
  cryptographic timestamp attestation.
- The setup import failure is preserved as narrative, without a raw wait/stderr
  receipt. No diagnostic retry is recorded. Child exits do not imply an additional
  observed outer-parent exit. No-fee/no-download scope is supported by reviewed
  code and cloud records, not a separate global billing or network audit.
- The entire sample remains development; scene/family independence, units/FPS,
  valid-visible geometry and checkpoint exposure remain unresolved. Original
  projection error statistics stay null, not zero.

The next useful step is a version-specific generation/conversion provenance review
that distinguishes earlier camera quantization from later arithmetic or convention
changes, starting with already cached source and published records. State what
would discriminate these mechanisms before new access. Any later numerical
geometry proposal needs a separate prospective freeze; do not loosen tau, repair
this matrix, retry the audit or tune the old test set automatically. A confirmatory
cohort is still unavailable.
