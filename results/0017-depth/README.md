# 0017: useful depth candidate headroom, no new-method improvement

All80 prospectively selected development pairs and320 known-method calls completed
once. Every prediction/process/log was sealed before one GT evaluation. The
additional-depth oracle headroom is9.202524pp and the predeclared scale-depth
omission trigger covers two scenes. These admit a specific follow-up audit;
**neither establishes a deployable improvement, novelty, or manuscript result**.

## Complete all-pair results

All values below are AUC percentage points. No cases were excluded; zero native
errors/timeouts. Directed translation is primary; the JSON also reports the
sign-folded secondary metric and separate successful-only coverage.

| Known method/control | AUC5 | AUC10 | AUC20 | Errors<5deg /80 |
|---|---:|---:|---:|---:|
| Current E normalized |71.488081|82.362408|88.575262|72|
| Current E Tangent Sampson |71.322710|82.265689|88.525076|72|
| RePoseD scale-only |62.383584|77.097927|85.932518|68|
| RePoseD affine depth |68.972945|80.716066|87.351192|72|
| External common pixel-MSAC selector |71.520213|82.371613|88.579865|72|
| GT oracle: two geometric outputs |71.640730|82.438733|88.613628|72|
| GT oracle: two depth outputs |74.210579|83.801742|89.284425|73|
| GT oracle: all four outputs |80.843254|87.296627|91.177630|75|

Oracle rows use GT and are **not deployable methods**. Additional candidate
headroom is80.843254−71.640730=9.202524pp, exceeding the2pp necessary opportunity
gate. The four-output oracle exceeds the best single method by9.355173pp. In
contrast, the ordinary score selector gains only0.032132pp over E normalized
while paying for all four calls. Direct depth fitting is worse on the aggregate.
This is a concrete unresolved use-of-information problem, not an achieved gain.

## All five scenes

Each row contains all16 allocated pairs. All five physical scenes are permanently
development. Key-derived image identifiers repeat within scenes; the selected
pairs involve23/25/29/26/25 distinct identifiers, respectively. No independent
80-trial inference or final-test claim is appropriate.

| Scene | E norm | E TS | Depth scale | Depth affine | External score | Oracle all |
|---|---:|---:|---:|---:|---:|---:|
| Florence cathedral |73.2290|73.1909|77.1215|75.5427|73.3238|80.0240|
| Lincoln memorial |83.8451|83.7905|66.0988|77.7580|83.8108|89.5225|
| London bridge |68.6436|68.9680|24.1777|57.8571|68.8740|75.1395|
| Milan cathedral |64.8858|63.9386|64.3119|60.1785|64.5783|74.4256|
| Sagrada familia |74.5443|74.4732|85.9846|82.3393|74.7508|89.6921|

Leave-one-scene-out additional-depth oracle headroom ranges7.605131–9.798635pp
(descriptive only). Against E normalized, scale-depth has8material benefits and
23material harms; affine-depth has13benefits and6harms. Scale-depth rescues3
geometric>=5degree cases but damages7geometric<5degree cases; affine rescues2
and damages2. Every pair/comparison, not only improvements, is retained.

## Specific mechanism trigger, not causal proof

The SAME scale-depth arm satisfies the fixed condition in Lincoln and London:
completed depth error materially worse than BOTH geometric arms, with at least
one transformed-z direction failing the native positive-z branch on the
nonempty pre-final-refinement mask. Affine depth qualifies in only London and
fails the two-scene trigger.

| Pair index | Scene | Arm | E norm error | E TS error | Depth error | Inactive directions / mask points |
|---:|---|---|---:|---:|---:|---:|
|21|Lincoln|scale|1.169760|1.179967|6.802746|15/1861|
|28|Lincoln|scale|3.786596|3.811392|5.414688|1/1725|
|44|London|scale|0.057930|0.056822|3.645266|9/1990|
|47|London|scale|0.318632|0.327761|4.738758|17/1963|
|44|London|affine|0.057930|0.056822|2.340960|8/1990|

There are two possible directional residual blocks per mask point. Across all80,
12scale and4affine outputs have any masked native inactive direction; neither
arm returns nonpositive scale. Qualifying inactive fractions are small. These
final-state associations do not show that optimization dropped terms or caused
the harm; the initial proposal could already be worse. The next separately
frozen trace must record the native objective's numerator AND active-count
denominator, accepted states and model lineage, and check unchanged-output
parity before interpretation. Known fixed-support/positivity constraints are
already present in CVPR2025 MADPose; an ordinary repair would not be novel.

## Execution, cost and provenance

Scientific freeze `5f267a3b657b38a7efe8381b9bb11f194e894355` and successful exact
[CI39316633022](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39316633022)
preceded all numerical inputs. Local62artificial tests and8manufactured native
calls passed. Source/input/config/library bindings were checked against the
published commit before execution. Official pinned PoseLib3.0.0, fixed seed0,
100–1000RANSACiterations,2px E and[16,2]px depth thresholds; no rescue tuning.

Prediction outer wall58.957799875s, actualexit0; internal completion58.694036292s.
Evaluation outer wall0.5380835s, actualexit0. All320worker logs are empty. Maximum
sampled worker RSS44,208KiB; sampling can miss peaks. No model inference, new
library build or paid services in this scientific stage.

| Arm | Input seconds | Native-call seconds | Diagnostic seconds | Charged outer-worker seconds |
|---|---:|---:|---:|---:|
| E norm |2.167502|0.594543|0.046913|11.274023|
| E TS |2.133875|0.957023|0.045789|12.701740|
| Depth scale |2.151625|2.238461|0.094919|16.777849|
| Depth affine |2.146435|2.989217|0.095349|17.234074|

Outer worker charges include imports, binding/hash and serialization overhead.
The external selector pays all57.987686worker seconds, not the selected arm's
time. Cached feature/depth generation is unavailable and excluded; equal
iteration limits are not equal runtime, and no complete-pipeline efficiency
claim follows. Do not compare these costs to0016 as if its settings/data matched.

Seal SHA256 `aef331923ced37edfb896ac070d954e3da6e08145145aa2bd3de636f9abce510`.
Evaluation SHA256 `66f3f693fa7473aba5371fe473c4f8c9df39a3dd711a32941f28f3e0380eb151`.
`prediction-process.json`/`evaluation-process.json` contain actual outer exits,
commands and log bindings. Every native final/raw worker/process record is in
`predictions/`; `evaluation.json` retains all80pair errors, masks' trigger counts,
fixed selections, scene and leave-one-scene-out summaries. The independent
[saved-output audit](independent-review/README.md) passed once: all320estimates,
1281bound files, independent angles/AUC/selections/costs/gates agree. It imports
no production code and reads no private cache. Geometric residual scores and
depth-inactivity counts are bound saved values, not independently reconstructed
from absent coordinates. Audit SHA256
`83cbfcfa0a3be6d480497d93783281aaf3493fdc3d2409861eeb16969f3792ae`.

The archived `../0017-preparation/outer-operation-source.py.txt` is the exact
outer receipt helper used from the sibling private preparation directory; it is
an archival source copy, not a runnable command from its new archive location.
Its actual predictor/evaluator commands are listed in the process receipts.

Source/headroom limitations remain: five development scenes, unverified exact
frontend checkpoint/exporter identity, no independent final validation, no
causal LO trace, no MADPose comparative run. Original research sources and
held-out0022numerical arrays remain unchanged. No manuscript is started.
