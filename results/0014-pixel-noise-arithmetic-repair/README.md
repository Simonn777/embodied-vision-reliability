# 0014: known pixel-noise correction has negligible development effect

After published arithmetic-repair freeze `e5ed651b874cfc6286a6f46940f093470f5b8770`,
all16 selected pairs completed all three continuation branches. The separate
sealed evaluation exited0. This is a known-control study, not a new method or
independent validation. No held-out numeric values were read.

| Output | AUC@5 (%) | AUC@10 (%) | AUC@20 (%) |
|---|---:|---:|---:|
| Original official PoseLib |81.169622478|90.584811239|95.292405619|
| Official continuation |81.169622478|90.584811239|95.292405619|
| Independent normalized Sampson N |81.169622844|90.584811422|95.292405711|
| Known pixel Sampson P |81.176133020|90.588066510|95.294033255|

P gains only0.006510542 AUC@5 percentage points versus original, with9 improved
and7 worsened pair errors. All16 remain in every denominator. This negligible
change does not support advancing the noise-only route. Do not tune its robust
scale or select favorable pairs to claim progress.

The strict all-pair interpretation gate remains **FAILED (15/16)**. Pair13's N
objective exceeds original by4.547473508864641e-13, exactly one binary64 ULP
(relative2.056e-16); it accepted no steps. This rounding-sized discrepancy is
not material convergence failure, but the prospective strict gate is not
retrospectively relaxed. All local GN stationarity checks pass; no N/P branch
exhausts100 iterations. The independent review checks the literal gate and
accuracy arithmetic without another estimator or HDF5 read.

Supervisor wall time4.104904seconds, sampled peak workerRSS44992KiB (sampling may
miss peaks). Summed calls: native0.004751417s, N0.058260625s, P0.454022376s;
common-objective diagnostics add0.390833125s. These are additional continuation
costs after the already measured3.098394958s baseline API cost, not full-pipeline
competitive timings. No speed claim is made.

The [first attempt](../0014-pixel-noise/README.md) remains unchanged: one native
call0.0004535s, failed initial N computation0.000647125s, supervisor0.457464459s.
There were17 native continuation calls across both attempts,16 completed N and
16 completed P outputs. The failed native-only record is not mixed into this
cohort. Artificial diagnosis and arithmetic-only repair are disclosed there.
Logs record normal completion; evaluator log is empty. Historical sources and results remain unchanged.

Reproduction inputs and fixed definitions are in the
[protocol](../../protocols/0014-pixel-noise-controls.md). Run the frozen
`run_controls.py` with the0012 manifest, pinned H5 and0013 prediction seal into a
fresh directory; only after all16 predictions are sealed, run
`evaluate_controls.py` with the pinned0013 evaluation. The source and binary
bindings are retained in `predictions/seal.json` and each record. Actual execution
was serial and single-attempt after the disclosed technical freeze.54 relevant
artificial tests passed before this run; exact repair-commit CI39302911040 passed.

No manuscript follows this result. The next substantive question is whether
additional image information can identify correlated dense-matching errors;
ordinary covariance weighting or known affine extraction alone is not admitted
as novelty.
