# 0012b: signed mean/variance follow-up, exploratory development only

The initial0012 result is retained unchanged:0.0884% fewer residual evaluations
than culling with slower measured runtime. It does not justify a compiled pose
implementation or manuscript. The follow-up changes the lower-bound derivation,
not the dataset partition, thresholds or historical result.

## Mechanism and falsifiable diagnostic

An axis-aligned box admits combinations of coordinates that are absent from
actual correspondences. Thus it can assign a huge maximum residual even when
all observed signed residuals are similar. Existing moments also retain the
signed residual mean because the last feature is1:

    mu = e^T M[:,8]/n; v = e^T M e/n - mu^2.

For a uniform random index over the fixed cell, Cantelli's inequality gives a
deterministic aggregate lower bound, with no iid sampling assumption. If
0<a<|mu| and d_i<=Dhi:

    S >= n * min(a^2/Dhi,t) * (|mu|-a)^2 / (v+(|mu|-a)^2).

Use a conservative lower bound on |mu| and upper bound on v,Dhi. Choose
a=min(m_lower/2,sqrt(t*Dhi)), fixed before any follow-up data read. Combine with
the previous bound by taking the maximum lower bound; keep its safe upper
bound. Symmetric signed residuals can make this addition vacuous. Derivation,
overflow/cancellation safeguards, zero denominators and a correlated-coordinate
counterexample require independent synthetic review first. The inequality is
classical; no claim of theoretical or computer-vision novelty is established.

## Fixed exploratory replay

- Reuse exactly the16 keys and1024 matrices already saved by0012. No resampling,
  changed threshold, new pair, new scene, pose label or calibration read.
- Preserve the original implementation and outputs byte-for-byte. New files and
  output directory identify this as a second, exploratory run on used development
  data. Read only the selected x1/x2 arrays and original saved measurements.
- Same leaf32, initial128 cells,128 refinement budget, threshold1pixel. Compare
  the original moment results, added mean/variance bound, spatial-culling saved
  results and direct reference. Check initial-cell and whole-score containment
  and exact argmin agreement; stop on an observed violation/wrong winner.
- Add a stronger simple control: ordinary partial exact MSAC, in blocks of256
  points after a deterministic shuffle seeded20261020; model order is the saved
  original order, initial incumbent model0. Stop a candidate only when its
  nonnegative partial sum strictly exceeds a guarded incumbent; retain exact
  ties with lowest original index. Charge all evaluated residuals, including
  repeated leaf work/fallback in bound-based methods.
- Record every pair's work counters, intervals, winner, time including setup,
  source/input hashes and failures. Initial lower-bound differences identify
  whether correlation information helps at all. Reused original timings are
  historical measurements, not a simultaneous controlled speed comparison.
- Before spending effort on a compiled backend, require at least20% fewer direct
  residual evaluations than the block-partial control, at least50% fewer than
  full scoring, correct winners for all16 pairs and zero observed containment
  violations. Passing is only numerical mechanism admission. Bound evaluation,
  memory and setup must still be cheaper in a comparable compiled implementation.

## Application boundary

Official DMS's main CCA arm uses center scoring and approximate final refinement,
not the surrogate scoring used here. A calibrated pose comparison must include
DDD,CCC,CCA and strong PoseLib/SupeRANSAC controls, shared five-point samples and
cheirality semantics. Plain-Sampson lower bounds remain conservative when
cheirality failures add cost; its upper bounds need modification/fallback.
No surrogate ranking disagreement establishes a published-method pose error.
The original held-out quality gate remains unchanged; no manuscript is started.

Resources: existing local runtime/data only, up to4 CPU threads, roughly8GiB,
30-minute hard process timeout. No new data/model/package transfer for this
numerical replay. The program remains active until substantial independently
confirmed task improvement, not merely a more favorable development diagnostic.
