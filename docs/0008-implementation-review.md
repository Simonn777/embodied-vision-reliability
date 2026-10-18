# 0008 prospective training implementation review

2026-10-18. AI-assisted design, implementation and independent review. **No
scientific training, inference or reference scoring occurred during this review.**
The goal remains task-quality improvement; software checks are not that result.

The [prospective protocol](../protocols/0008-antithetic-training.md) follows the
pre-outcome conditional training recommendation and 0007's complete local mechanism
pass. It fixes32 fresh trajectories and64 locked predictions,≥0.5dB at each of
two endpoints, texture/harm and full-cost checks. Stronger independent-data,
modern-baseline and paper gates remain unchanged.

## Independent core and metric review

Training core SHA256:
`79bbd008f6b2222ef8d4aafc05bedd5e492b769ff736f5bb64b358a157b84916`.
The independent reviewer approved its fixed R2R objective, equal512 forward
occurrences, original initialization, fresh Adam, three shared independent
streams, stable mean sort, detached endpoint snapshots and hard stop rules.
17 artificial fixtures passed with zero skips in0.858s on the existing local
Python3.9.6 / NumPy2.0.2 / Torch2.8.0 environment; warnings were errors.

Two pre-freeze issues were corrected: copying the500-update snapshot could cross
the time target and wrongly relabel that earlier update, and final receipt
serialization could cross the cooperative deadline. The final code locks update
time before copying and checks the deadline again after serialization. Virtual
clock fixtures cover both. No real outcome was used to choose the correction.

The inference module restores exact six-tensor weights and uses fixed MC50,
float64 averaging, no interpolation/clipping/direct pass, and the independent
inference seed. Eleven artificial fixtures passed, zero skips,0.667s; an
independent tiny MC oracle, a float64 cancellation case and weight-mutation
rejection check computation without calling the production interface.

The pure metric module was independently reviewed against the protocol. Its18
artificial tests and four additional independently constructed8×8-array checks
passed. Fixed oracle ties, both-endpoint rules, group harm, pooled first-difference
error, zero-error handling and the ratio of summed costs agree. A documentation-only
clarification names the supplied cost as allocated observed outer cost. The
mathematical implementation is unchanged. Scalar metrics alone cannot establish
actual training, inference or provenance; the wrapper verifies those separately.

## Independent wrapper findings and resolution

Final wrapper SHA256:
`c62e258f201e2c7b058a3c3f1233e90bf1351ede7ed9fb25ebd67d5abf7f20c1`.
Final supervision-test SHA256:
`f5faecfb30b038d4cde72d70bfbcc3d7559aa7f9b020bc8d25fb95dff8ad0160`.
The final independent decision is APPROVE within source-review/fixture scope.

- The first wrapper could import a same-named module from the dependency directory
  after that path was prepended. An artificial shadow module reproduced the defect.
  Modules now load explicitly from canonical hash-checked repository paths; the
  verified core and metric helper are registered before dependent imports.
- The original saved-output gate needed inference/state/content bindings. The
  final gate validates actual MC50 counts/seed, unchanged endpoint tensors and
  content hashes, finite prediction shape/content, exact runtime and shared input
  identities before any reference is opened.
- Command digests now also bind the exact frozen argv/cwd. Cost arithmetic is
  independently reconstructed. An intermediate version accepted a negative input
  time after all fixture hashes were rebound; the final guard rejects every
  nonpositive/nonfinite component and invalid nested clock. The same probe now
  fails as intended. All fixes preceded any source freeze or study run.

The reviewer reran five final wrapper tests with zero skips in0.146s, including a
synthetic32-trajectory/64-endpoint file grid, rebound-hash mutations, path/source
shadowing, method order and cost allocation. No production model was invoked.
Canonical freeze, exclusive directories/receipts, child wait4/log closure and
full-grid completion are mandatory; an incomplete run cannot be selectively scored.

## Accounting and execution limits

Allocated outer cost retains runtime startup, source checks, common setup, logs
and shutdown. Only measured other-endpoint extra training and disk/inference work
are subtracted. It is a shared-trajectory observed-cost allocation, not an
independently measured standalone endpoint runtime. Earlier snapshot overhead
and up-to50ms supervisor polling granularity remain. Both component sums and full
actual dual-endpoint wall must be published.

The integrated local51-test artificial suite passed, zero skips,3.488s; the initial
full repository suite passed607 tests in45.45s with physical TMPDIR. After the
last timing guard change, its five regression tests passed in0.143s; the independent
five-test replay above confirms the final guard. These local checks use Torch2.8.0,
so the exact cloud Torch2.10.0+cpu runtime still requires its own zero-skip fixtures
before scientific execution. Test filenames/import paths were adapted from the
temporary prototypes to the repository; scientific module bytes remain as reviewed.

The independent saved-array checker uses NumPy and compensated scalar arithmetic,
not production metric/Torch imports. Its source and artificial self-check are also
frozen before any predicted array exists. A future replay verifies saved evidence;
it never trains or regenerates predictions. No raw images, references, weights or
predictions are authorized for publication. No new assets, fee or schedule change.

## Final integrated evidence before freeze

The final repository suite passed607 tests (zero skips) in45.77s. The independent checker
passed16 self-test groups. A complete synthetic64-output file-contract replay
passed45,354 checks /5,159 numeric comparisons with maximum difference1.42e-14;
four deliberately inconsistent cases were rejected. These are artificial
software fixtures, not predictions from a trained research model.

The portable manual contract test is retained in
`tests/manual_antithetic_training_contract.py`; execute it with `--report NEW.json`.
It uses only temporary synthetic arrays and the production scalar metric code to
create comparison targets. The standalone checker itself has no production imports.
[Recorded self-test](0008-review/checker-self-test.json) and
[recorded synthetic contract](0008-review/synthetic-contract.json) retain actual
exits and source hashes. No scientific payload is published by these checks.

A separate reviewer approved checker source `37b40d62dbfc849a1eb579af007d081a1d83de79b01309190c8064b1a9f2c91a`, reran16 artificial groups in0.018s, and found no production/model/network import or metric/gate defect. The manual contract remains explicitly artificial.
