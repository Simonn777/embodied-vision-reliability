# Independent saved-only review: study ineligible

The [audit](audit.json) independently verifies all 160 replay records, their
process/log/trace hashes, frozen sources and original 0017 bindings. All 4,640
floating pose/geometry/statistics components are **bitwise identical**; masks,
discrete statistics and observation declarations also agree. Trace payload totals
37,072,987 bytes. Worker logs are empty.

Across all 160 saved traces there are 19,343 required initial/accepted/exit states
and 829 rejected trials. Six required states are nonfinite: the initial and exit
of `proposal_lo`, `lo_id=0`, in each case below. Their raw rotations, quaternions,
translations, scales and native cost sums/means contain serialized nonfinite
values. These are invalid initial states, not rejected trial steps or merely
nonfinite auxiliary diagnostics.

| Pair index | Scene | Arm | Final LO ancestry | Invalid LO on final ancestry |
|---:|---|---|---|---|
| 44 | london_bridge | depth_scale | 10 → 2 | No |
| 61 | milan_cathedral | depth_scale | 7 → 2 | No |
| 64 | sagrada_familia | depth_scale | 5 → 2 | No |

Ancestry here is an identifier-only graph check, with no pose-error calculation.
No rejected trial contains a nonfinite state/cost. Each invalid proposal was
temporarily adopted as an incumbent and subsequently replaced; its associated
invalid LO is outside the final ancestry. That fact does not override the
frozen requirement to reconstruct every necessary state.

The diagnostic stopped after exactly 88 reconstructions (pairs 0–43, both arms),
with `TraceMismatch: q has invalid shape or nonfinite entries`. Its saved receipt
is **ineligible**. There is no reconstruction seal, label artifact or evaluation
output. The support-loss mechanism gate was **not evaluated**; this is not a
mechanism `NO_GO`, an improvement claim, or permission to relax the barrier.

The independent pass read saved prediction/trace JSON, source files and logs.
It did not read cached arrays or saved reference labels, execute estimation, or
rerun the diagnosis. Native replay time sums to 5.278665 s and worker outer times
sum to 58.165735207 s; these are saved instrumentation costs, not method speedups.

[Execution provenance](execution.json) preserves both saved-only review passes.
The [initial audit](initial-audit.json) is retained unchanged. A subsequent
dependency cleanup replaced the uninvoked label-auditor utility dependency with
the existing 0017 independent JSON/hash helpers and local raw-parity code. The
final compact reviewer and its four artificial tests passed in a combined shell
invocation with observed exit code 0 and wall time 1.574464667 s; that duration
includes file operations and tests, and is not an isolated audit duration.
The first audit's precise duration and exit metadata were not retained and remain
explicitly unknown. The prepared label/gate auditor entrypoint was never invoked.

Original study freeze: `4770af453750fce77dd0c655266c66530cc0138a`.
Raw evidence: [prediction seal](../predictions/seal.json),
[diagnosis failure](../diagnosis/operation.json),
[prediction process](../prediction-process.json), and
[diagnosis process](../diagnosis-process.json).
