# 0025b physical-depth qualification: pass, not a pose improvement

**Decision: permit a separately frozen translation/pose proposal. No manuscript
or research-goal completion.** The first fixed execution completed all 80 depth
forwards and all 64 required paired cases. The unchanged prospective engineering
gate passed, and an independently implemented saved-output arithmetic audit
agreed. This qualifies an oracle-assisted depth approximation in a simulated
pure-rotation regime; it does not establish a new method or improved camera pose.

## Frozen primary result

Each case first computes the uncapped mean AbsRel on its fixed held-out pixels.
The table then takes the median across the scene's fixed 32 cases. The paired
column is a median of per-case differences, not a difference of medians.

| Scene | Required paired cases | RS row-affine median AbsRel | GS transport median AbsRel | Median paired RS minus GS | Fixed gate |
|---|---:|---:|---:|---:|---|
| Living room | 32/32 | 2.93645% | 1.89201% | +0.97345 pp | Pass |
| Office | 32/32 | 2.29345% | 1.15471% | +1.09796 pp | Pass |

The thresholds were RS median <=10% and paired excess <=2 percentage points in
each scene, with all 64 primary pairs available. **The GS oracle control is
better in the paired median.** Passing the tolerance is not a gain over it.
All four arms use GT-assisted training-pixel fits; none is a deployable method. The RS row arm was worse than GS in 56/64 cases; the [descriptive interpretation](interpretation.md) retains tails and the inconclusive row/middle comparison.

## Complete coverage and controls

- `rs_row`, `gs_transport`, and `rs_scale_only`: 64/64 available. All 64 common
  GT-only supports meet the fixed coverage and split requirements.
- `rs_mid`: 63/64 available. Office frame index 10, numeric stem 251,
  `rs_y_positive` has an unconstrained fitted scale of −0.00232121. The fit is
  recorded unavailable, not clipped, retried, or excluded from the population.
  This explanatory-arm failure was prospectively not an extra primary veto.
- The [256-row table](all-cases.csv) includes every arm, all unavailable records,
  parameters and error summaries. Full per-case JSON and prediction metadata
  accompany the [aggregate](evaluation/aggregate.json). Raw prediction, support,
  design and pixel-error NPZ arrays remain privately preserved with public hashes.

These are two simulated development layouts. Frames share trajectories, pixels
are spatially dependent, and neither scene is an independent real-world final
test. Pure rotation has no translational parallax/disocclusion. The <=2% source
depth-neighbor spread restricts evaluation to sampled smooth-surface support.
The row and same-point middle-Z interpretations can be close at this fixed
12-degree readout motion; a pass does not identify a unique network time anchor.

## Execution and provenance

Protocol and all execution/audit sources were frozen at
`13d4b3259f8d12a1ba4d5d0de9ff56ec6527cf62` before data access.
[Exact-source CI 39353648975](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39353648975)
passed first; [source CI receipt](source-freeze-CI.json) preserves that evidence.
No execution retry, substitution or numerical-threshold change occurred.

The official clean TUM-compatible ICL-NUIM PNG archives contained 881 paired
numeric RGB/depth names each. The structural rule selected ranks/stems
`0,126,251,377,503,629,754,880` in each. The complete inventory was saved before
pixel decoding; no quality or result chose a frame. Acquisition transferred
731,653,920 response-body bytes, with TOFU archive hashes and exact HTTP validators.
Transport/decompression included text members, but source-pose text was never
extracted or parsed. Dataset attribution: Handa, Whelan, McDonald and Davison,
[ICL-NUIM](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html), CC BY 3.0.

| Operation | Outer wall seconds | Sampled group peak KiB | Completion |
|---|---:|---:|---|
| Acquisition/index | 57.4090 | 29,616 | Clean |
| 80 predictions, including loading and checks | 72.0348 | 2,083,040 | Clean |
| GT diagnosis | 6.7935 | 226,112 | Clean |
| Independent arithmetic audit | 0.9026 | 170,896 | Clean |

All operations used the frozen four-thread/8-GiB/1800-second supervisor. RSS is
sampled and can miss transient peaks. Prediction made zero feature, matcher or
native pose calls and zero attempted network calls. The unchanged backend also
loaded SuperPoint/LightGlue weights, without inference; that loading cost is
included. Each MoGe call emitted the same CPU-float32 autocast warning: the
unsupported autocast target was disabled. All 80 warnings are retained; float32
execution and frozen inference settings were unchanged.

All prediction outputs and the completed outer receipt preceded numeric GT
decoding. The GT stage then hashed/header-checked all 16 depth PNGs before its
first numeric decode. It completed 16/16 decodes and 64/64 cases. No ground-truth
pose or rotation entered an estimator; no pose estimator ran.

The [independent audit](audit/audit.json) reconstructed Rodrigues geometry,
inverse-depth truth interpolation, support, four designs, centered closed-form
calibration, held-out errors and the gate. All 256 arm records agreed; maximum
absolute numeric difference was 1.046e−13, with no tolerance change. This is an
independent implementation within the same AI-assisted program, not independent
team replication. It recomputed fits from saved outputs, without rerunning the
original model or original fitting pipeline.

## Artifact interpretation

`publication-bindings.json` maps each original private artifact hash to its public
copy. Local path prefixes are redacted where necessary; embedded scientific
bindings still refer to the immutable original artifacts. Redacted receipt copies
are an evidence view, not direct inputs to byte-exact local validators. PNG/NPZ
payloads and full archives are not republished here. No result was overwritten.

The next step must test actual motion estimation with within-readout translation,
realistic rotation input, strong same-information controls and complete-pipeline
costs. This pass does not reopen 0024 or admit same-data tuning as held-out
confirmation. The original substantial two-source held-out improvement gate
remains unmet.
