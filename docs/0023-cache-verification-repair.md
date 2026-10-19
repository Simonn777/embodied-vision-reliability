# 0023 technical amendment: exact calibration verification and input reuse

2026-10-19. This is a technical repair before any 0023 pose fit or GT access.
It does not change the scientific hypothesis, population, observations, solver,
initializations, loss, uncertainty operator, fallback, metric or decision gate in
[the original frozen protocol](../protocols/0023-mapfree-directional-instability.md).
That protocol and the successful acquisition/frontend sources remain unchanged.

## Preserved execution and failure

Source freeze `acf502b831ad9e3485ab56549217c8a603d56bf7` passed exact CI
[39341051566](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39341051566)
before allocation and payload access. Places s00513, s00475 and s00478, including
their unselected frames, are development forever. The other62 places remain
unallocated and their scientific payloads unread.

Acquisition01 completed all54 members with162 requests and11,062,865 response-body
bytes; its outer operation took75.975032791s, peak sampled group RSS122,336KiB,
exit0. Frontend01 completed all51 images and120 edges, with unchanged qualified
0022 models and settings, no offline violations, outer103.6027065s and2,142,832KiB,
exit0. These successful outputs are retained byte-for-byte.

Prediction01 stopped at `canonical_cache` with
`ValueError: calibration/coordinate metadata mismatch`. Its inner duration was
1.295762792s; outer1.409816375s,40,864KiB,exit1. It produced no pose fit, query
prediction or prediction seal, and no ground truth was acquired. Its failed
receipt, supervisor receipt and logs remain part of the record.

The frozen frontend computes `sx=w/ow` and `sy=h/oh` first, then
`cx'=(cx+.5)*sx-.5` and `cy'=(cy+.5)*sy-.5`. The adapter recomputed the principal
point as `(cy+.5)*h/oh-.5`, changing floating-point association before requiring
exact equality. In the first image,540x720 becomes384x512 and the recorded
`cy=349.9513` yields248.70981333333336 with the frontend order versus
248.70981333333333 with the adapter order. All recorded calibration metadata
and cached arrays are retained. This discrepancy is not an accuracy result.

## Narrow repair and admission for a fresh prediction operation

The adapter must use the frontend's precomputed scale factors and identical
operation order. Keep exact equality and every existing hash, dimension,
pixel-center and calibration check. Do not introduce a tolerance or alter any
cached calibration. Artificial regression tests must include a nontrivial resize
that distinguishes the two operation orders and reject a one-ULP tamper.

The new predictor accepts an explicit `--input-freeze`, meaning the0023
acquisition/frontend source commit, distinct from `frontend_freeze` (the0022
qualification commit). Omission defaults to the predictor's own expected commit.
For the repaired operation, input freeze is exactly
`acf502b831ad9e3485ab56549217c8a603d56bf7`. The successful old frontend and
acquisition commands, receipts, source bytes, protocol, all input/cache files,
supervision and logs must still verify against that original commit. The new
predictor's executed scientific sources and this amendment must independently
verify against its newly published expected commit and passing exact CI.

The prediction receipt records the input freeze. Evaluation derives it from the
sealed predictor command, requires receipt agreement, and rechecks the complete
original frontend chain. An arbitrary or changed cache cannot be admitted merely
by specifying another commit. Reusing the already successful bytes avoids a
second inference or dataset acquisition; the original failed prediction is not
resumed, overwritten or counted as a completed experiment.

Only after independent review, artificial checks and exact-commit CI may a fresh
prediction02 run under the original4-thread,8GiB,1800-second limits. Every24 query
and all seven estimators remain mandatory. GT stays inaccessible until that
complete prediction seal and successful outer operation exist. The first actual
evaluation then uses the unchanged fixed metric and development gate. All costs,
including the failed attempt, remain reported. No manuscript is admitted.
