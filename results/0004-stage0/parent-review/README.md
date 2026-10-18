# Independent review of the saved Stage 0 result

2026-10-18. AI-assisted review of result commit
`ca68cfd56984dedf1cf875d9b4a6c68440565ee1`, relative to execution main
`ddadc5e17a6ceff3c358d3e3f440d80bf3fd0306`.

**Decision: accept the negative result; keep `stop_this_candidate`.** No inference,
cached dataset decoding, parameter fit or threshold change was performed locally.
This review is not the requested substantive tracking improvement or a paper.

## What was verified

Two separate read-only implementations checked the committed evidence. The
[provenance audit](provenance.json) verifies eight frozen inputs, all 1,691 original
files (including 1,528 prior result files), 32 indexed artifact hashes, raw logs and
source-bound execution receipts. The original result commit adds 33 files only.
The 4,488-row grid, 250 encoder / 561 refiner calls, 13 equality controls and recorded
zero child/wrapper exits agree. The original result files remain byte-identical.

The [metric audit](metrics.json) independently calculates native visibility from
raw logits and aggregates the published per-row errors/GT-visible flags. Both
summaries match on all 2,373 numeric leaves, maximum absolute difference
3.552713678800501e-15. The cloud checker separately used the cached numeric GT
arrays; this local review does **not** independently re-establish those coordinates.
Its source, hashes and recorded execution were examined instead.

The portable audit scripts are [check_metrics.py](check_metrics.py) and
[check_provenance.py](check_provenance.py). They print JSON to stdout. Run them from
a clean checkout of the reviewed result commit, keeping the review scripts available
outside that checkout if necessary. Both were actually run before the subsequent
state-document edits and exited 0. Their execution record is [execution.json](execution.json).

Recorded next-context digests cannot be reconstructed from saved context tensors,
which were not retained; equality is a checked runtime assertion. The outermost
supervisor has no further durable parent wait receipt. The desktop thread reader
also rejected the full completed cloud turn as oversized; this review uses the
published artifacts and the observed completed-turn result, not a separately
retrieved complete raw tool stream. These limits do not change the negative result.

Terminology correction to the preserved original README: its “predeclared checker
atol 1e-10” is the **saved checker's comparison tolerance**, not a threshold in the
scientific freeze. The frozen inference equality tolerance remains 1e-4 and all
observed control differences were exactly zero.

## Post-hoc interpretation, separate from the frozen decision

These additional counts are descriptive checks of already saved development rows.
They do not modify the continuation rule or select a deployable action policy.

- The 56 shadow episodes have five positive, thirteen negative and thirty-eight
  zero net returns. Positive episode gains sum to +11; negative gains to −37.
- The loss-count return at successive offsets is −3, −5, −3, −3, −3, −3, −3, −3.
  Aggregate damage therefore cannot be described as confined to the reset frame.
- Some positive episodes improve visibility only. In particular, start 160 / point
  7 is occluded throughout. Start 192 / point 3's positive change is also on an
  occluded frame. Start 192 / point 4 changes visibility while the baseline
  coordinate already lies within 4 px. These are not coordinate recovery evidence.
- A genuine sustained repair exists: start 128 / point 3 changes six visible frames
  from large baseline errors to within 4 px. Therefore “resets never help” would
  also overstate the evidence. The aggregate rule still fails and this selected
  example does not authorize advancing the current candidate.

The result rejects continuation of this declared operator under its fixed rule.
It does not establish that every conditional memory policy or every different
tracker must fail. Future work requires a distinct prospective mechanism and
appropriate development evidence, not reinterpretation of this run as success.
