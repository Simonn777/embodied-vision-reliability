# 0019 independent preparation review

Source and artificial-test review only, 2026-10-19. No cached arrays, saved
reference labels, images or native estimator calls were accessed. The reviewer
did not edit the experiment, its configuration, or its protocol.

The control changes only the nested final bundle loss to `TRUNCATED_CAUCHY`.
The native signature, observations, cameras, raw depth gauge, input order and
RANSAC options are retained. Successful calls must match the original pre-final
mask exactly and RANSAC statistics under the frozen componentwise tolerance;
final geometry is intentionally allowed to differ. Returned stale RANSAC scores
are not used to choose a winning method.

The supervisor verifies the complete original baseline and all 160 new records,
including sources, library, input declarations, process receipts and logs.
Only then does the evaluator decode the fixed-hash saved reference evaluation.
It retains all 80 pairs per arm and reports all five scenes, directed and
sign-folded metrics, pairwise benefits/harms, and charged outer-process time.
Historical baseline timing is explicitly not a matched-time comparison.

The reviewer raised the distinction between a thrown native error and an invalid
returned native value before any scientific call. The parent explicitly retained
the strict policy: genuine estimator exceptions/crashes/timeouts receive the
180-degree penalty in the full denominator; invalid/nonfinite returned geometry,
malformed masks/statistics or failed pre-final parity block the whole comparison.
This is an integrity/qualification stop, not permission to drop a case, reinterpret
the outcome or retry after seeing results. The protocol records that choice.

Observed independent command:

```text
<existing research venv>/bin/python -m unittest discover \
  -s experiments/0019-author-loss -p 'test_*.py' -v
```

All **14 artificial tests passed**, process exit code **0**, observed process wall
time **0.642323959 s**. Coverage includes the exact one-setting API change, mandatory
pre-final parity, eligible versus ineligible failure classes, 160-record rotating
schedule, full failure denominators, source/input/log/path tampering, timing
containment, and the no-label-read barrier on a failed seal. No additional broad
tests or scientific calls were run by the reviewer.

No remaining implementation blocker was identified. This is admission for a
bounded known-control comparison, not evidence of accuracy improvement, novelty,
held-out generalization or paper readiness. The failed 0018 mechanism diagnostic
remains ineligible and its gate remains unevaluated.
