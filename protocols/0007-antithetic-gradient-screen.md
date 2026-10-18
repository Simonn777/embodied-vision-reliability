# 0007 — prospective mean-paired R2R gradient screen

Status: prospective source/protocol freeze required before scientific execution.
AI-assisted research; no new method, denoising improvement or paper result is
claimed. This is one bounded falsification test, not a rescue of failed 0006.

## Question and prior art

At the same fixed R2R alpha0.5 objective and 512 evaluated occurrences, can pairing
patches by their observed mean before assigning opposite Gaussian perturbations
reduce gradient trace variance enough to justify a separate optimization study?
[Prior-art and mechanism audit](../docs/literature/0007-antithetic-prior-art.md)
records direct precedents for antithetic SGD and coupling. Broad novelty is
rejected; the exact mean rule has no established priority or theoretical benefit.
Same-patch pairing already cancels the leading small-alpha term, and even-noise
terms can make any shared-noise pairing worse. Raw mean is a weak proxy, not a
semantic representation or a guarantee of parameter-Jacobian similarity.

All eight colored R2R alpha0.5 final checkpoints from 0006 are included, ordered
d1seed0,d1seed1,...,d4seed1. None is selected by outcome. Each of these four image
substrates remains development forever; no scene-population claim is possible.
[Configuration](../configs/0007-gradient-screen.json) binds exact cached noisy+C,
weights and original identity receipts from published results commit `dabd5b9`.
The runner may read only those 24 cache files. It must not read clean references,
predictions or old scores, acquire data, install packages or download weights.
Existing identity receipts contain old training metadata; they are not used to
select parameters. Old results and the open 0006 stdout discrepancy are preserved.

## Fixed computation and estimators

Use CPU float32 49→64→32→1 ReLU MLP, all 5,313 parameters, unhalved mean MSE,
alpha0.5, no clipping/normalization, the original float64 Cholesky cast to float32.
Record the resulting covariance error; no jitter/repair. Extract all 122² valid
7×7 noisy patches. B=512 uniform draws **with replacement**, not unique patches.
No optimizer, update, inference reconstruction, early stopping or checkpoint
selection. Parameters must hash identically before/after. Every replicate uses
one vectorized 512-row forward and backward for each of exactly four estimators:

1. Independent: 512 independent Gaussian patch noises eta ~ N(0,C).
2. Random antithetic: fresh uniform permutation of 512 occurrence slots, adjacent
   pairs, 256 independent eta vectors assigned +eta/−eta to each pair.
3. Mean antithetic: stable ascending sort of the float64 arithmetic means of the
   49 original noisy pixels, occurrence-slot order breaks ties, adjacent pairs
   assigned +eta/−eta. Sorting precedes fresh noise; no extra representation pass.
4. Same patch: first 256 base occurrences, each duplicated with +eta/−eta. This
   spends the same forwards on fewer underlying sampled occurrences.

All methods use input p+0.5 eta and target p_center−2 eta_center. Pairings do not
alter a full batch's occurrence weights. Each occurrence keeps its Gaussian
marginal, so the expected full-batch gradient is unchanged, conditional on the
fixed model/batch. Numerical finite precision remains a limitation. No partner
is imported from outside the batch, duplicates dropped or favorable batches sought.

## Two regimes and seed schedule

A_fixed uses one predetermined 512-occurrence batch per case for 64 repetitions.
It measures augmentation plus random-pairing variation conditional on that batch.
Same-patch's duplicated first half has a different conditional empirical objective;
report its own variance but exclude it from A's gradient-mean equality checks.
B_redraw uses 64 fresh base batches; all four methods share each replicate's base
index draws, and have the same dataset-gradient expectation. Its variance includes
both occurrence sampling and augmentation. B is the primary continuation endpoint.

For case c=0..7 and regime q=0,1, three CPU Torch generators have master seeds
700000+10000c+1000q+s, s=0(indices),1(augmentation),2(random pairing). Each first
produces 64 int64 replicate seeds using randint(0,2**63−1). Each method reconstructs
fresh generators from the same replicate tuple. A's batch uses index seed[0]
once, then methods clone its fixed indices; B redraws indices inside each method.
Independent draws 512×49 standard normals; paired methods draw 256×49, sharing
that prefix where the pinned Torch implementation confirms it in a tiny fixture.
Rotate execution order by case and replicate; no hidden warmup. Master schedules,
indices/noise hashes, per-step losses and gradient hashes are retained.

## Timing and resources

Each estimator step timer includes generator creation/seeding, B index draw (or
A fixed-index clone), row selection, means/sort/permutation, Gaussian generation,
factor multiplication, zero_grad, forward/backward, finite checks and gradient
flatten/copy. Regenerate shared random values separately inside each method's
timer. No free cached augmentation or feature computation. The ordinary setup
(model/input load, factor and all-patch preparation), master seed creation,
A batch preparation, and post-step audit hashing are separately timed and reported;
none is candidate-specific work excluded from its step timer. Startup/whole-case
wall and peak RSS are also reported. The B time gate sums all 64 B steps, not A,
model load or evidence serialization. V×time uses each case's corresponding B sum.
This is measured estimator-step cost including first-use overhead; full study
cost is disclosed separately.

Use cached Python3.12.14, NumPy2.0.2, Torch2.10.0+cpu, four CPU cores/threads, one
Torch interop thread, deterministic algorithms, 8GiB address space. Each case has
90-second cooperative core deadline and 120-second external supervisor. A study
has 900-second hard wall limit. Scoring saved gradients has a separate 120-second
hard cap. Exactly 8×2×64×4=4,096 batch forward/backward calls are planned, zero
optimizer steps. This runtime is prospective, not a performance measurement.
The unique output is the original cache parent / `antithetic-gradient-0007`;
exclusive output-directory and case markers prevent reruns. Child cases require
a matching started-study record. A hard interruption can leave only markers,
partial files and outer command status; absence of a case receipt is a failure. Incomplete/failed
cases stop the study; preserve all partials and do not lower repetitions, retry
or select successful cases. Software fixture tests may run before the science;
cloud must execute Torch fixtures with no skips. No paid service or added fees.
New asset bytes=0; conservative historical ceiling remains 7,532,690,072 bytes.

## Measurements and fixed decision

Save float32 gradients privately in [2,4,64,5313] per case and analyze in float64.
For every case/regime/method report trace covariance
V=Σ_r ||g_r−mean(g)||²/(64−1), mean-gradient norm, finite status and elapsed time.
This is trace variance, not covariance ordering in every direction. A zero V or
nonfinite quantity makes required ratios undefined and blocks continuation.

Create eight unit Rademacher parameter projections with NumPy default_rng seed
709999. For every A pair among the first three methods (3 pairs) and B pair among
all four (6 pairs), calculate replicate-paired projected differences d_r. Require
|mean(d)| ≤ 5×sample_sd(d)/sqrt(64) + 1e−7×max(1,|mean projected gradient1|,
|mean projected gradient2|). Failure flags unresolved estimator integrity; it is
not proof of bias and must not be erased by adding replicates. The mathematical
marginal argument, not a noisy finite check, justifies equal expectations.

Use 2,000 bootstrap replicate-index bundles per case from NumPy default_rng
seed2000000+c. The same index draw applies to all methods and both regimes;
cases have separate random streams. Recompute V from the resampled gradients,
then average log ratios with equal weights over the eight cases. Use NumPy linear
quantile at 0.9833333333333333 for each of three B comparisons. These are approximate
Monte Carlo uncertainty bounds conditional on the eight fixed saved models/data;
they are neither independent-scene intervals nor guaranteed simultaneous coverage.

Only continue this exact rule to a separately frozen optimization study when all
cases and integrity checks pass and every following condition holds:

- A geometric-mean V(mean)/V(random) <1, the direct mechanism check.
- B geometric-mean V(mean)/V(control) ≤0.90 for each of independent, random and
  same-patch, and each corresponding bootstrap upper bound <1.
- B mean pairing beats the smallest control V in at least six of eight cases.
- Total B mean-pair step time / independent step time ≤1.10.
- Geometric-mean B V×time ratio ≤0.90 against each of the three controls.

The 10% variance/cost-adjusted threshold screens out small fragile wins; it does
not substitute for a task-quality threshold. Report all A/B results and failed
gates. Failure stops this exact mean-sort candidate: no unplanned feature/alpha/
seed search to rescue it. A pass is only local estimator evidence, not convergence,
PSNR, real-camera efficacy, novelty or a paper. The later denoising paper gate
(≥0.5dB over strong matched-resource baselines, independent data/scene uncertainty,
controls and replication) is not weakened. A distinct optimization claim would
need a new justified practical-performance gate before optimization outcomes.

## Publication and integrity

Freeze protocol/config/source/tests in a source commit, then create a separate
canonical receipt binding all eight required files. Verify receipt bytes against
its explicit commit and each file against both current and source Git bytes
before every stage. Freeze publication and CI precede science. Validate original
cached hashes before/after, preserve all old evidence, and review commands/exit
statuses and saved-array scoring independently. Publish original concise receipts,
logs, summaries, code and research interpretation; actual command argv remain
private with their recorded SHA and a published path-placeholder recipe. Keep images, weights, full
gradients and any reconstructable outputs in the private cache. No paper until
substantial reproducible task progress and novelty survive independent checks.
