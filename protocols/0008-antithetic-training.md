# 0008: does unchanged mean pairing improve actual denoising?

Prospective development protocol, 2026-10-18. **No results and no paper claim.**
Scientific execution requires the canonical source/hash freeze published on main,
passing pinned-runtime fixtures, an unused output directory and complete parent
review. This follows 0007's local gradient screen; it does not reinterpret that
screen as optimization or image-quality evidence. AI-assisted design and code.

## Question and prior-art boundary

Does the exact raw-patch-mean pairing rule tested in 0007 improve denoising from
fresh initialization, at fixed optimizer updates **and** approximately fixed
training wall time? Lower final-checkpoint gradient variance need not improve Adam
convergence, texture or the finite-image objective. Generic antithetic SGD,
matching and Gaussian coupling are prior work; see the [primary-source audit](../docs/literature/0007-antithetic-prior-art.md).
There is no claim that joining existing ideas constitutes a new contribution.

The conditional design was written before the 0007 scientific outcomes were
inspected; this executable protocol follows its complete local pass. No learning
rate, architecture, pairing feature, subset, alpha or budget sweep is admitted.
A failure stops this exact mean-sort proposal, without turning a favorable seed,
training-loss curve or variance result into task improvement.

## Existing development inputs and exposure

Use all eight already cached d1–d4 × corruption seeds0,1 colored-noise inputs from
0006, with exact byte identities in `configs/0008-training.json`. The four
metadata-selected PolyU mean-image JPEG substrates remain development forever.
They are not pristine camera ground truth, four independent representative scene
samples, or new untouched test data. No old 0002 evaluation data is used.
Known synthetic 49×49 covariance is supplied to every method. No new download,
package install, model purchase or paid API. New asset bytes0; conservative total
acquisition ceiling remains7,532,690,072 bytes without refunding missing files.

Only noisy128×128 float32 arrays and covariance49×49 float64 arrays are loaded by
training/inference. No old final model, optimizer, prediction or quality score is
used. Cached clean-reference substrates are first opened by a separate scoring
process after all32 successful trajectories and64 endpoint predictions are locked
and integrity-checked. Reference hashes are pinned from existing public preparation
metadata; they do not choose checkpoints, pairs, seeds or hyperparameters.

## Four equal-forward estimators and fresh initialization

For each case run `independent`, `random_antithetic`, `mean_antithetic`, and
`same_patch`. Rotate execution order by case index: method `(case_index+j)%4`,
j0…3. Case order is d1seed0,d1seed1,…,d4seed1. Every method receives fresh identical
49→64→32→1 ReLU weights, initialization seed
`620000+100*(case_index//2)+10*(case_index%2)+1`, and fresh Adam state. Adam is
lr0.001, betas(0.9,0.999), epsilon1e-8, weight decay0, amsgradFalse, foreachFalse;
CPU float32, no scheduler, clipping or warmup. Exactly5,313 parameters.

Each update draws512 patch occurrences with replacement. Unhalved R2R MSE uses
input `p+0.5*eta` and target `p_center−2*eta_center`; eta has the supplied covariance
through an unjittered float64 Cholesky cast to float32. Independent uses512 Gaussian
rows. Random pairing uniformly permutes occurrences; mean pairing stable-sorts
the raw49-pixel float64 means, tying by occurrence slot. Both use adjacent ±eta
with256 Gaussian rows. Same-patch duplicates the first256 base occurrences into
adjacent pairs and uses ±eta. Thus all evaluate512 examples but same-patch has
only256 original draws; do not claim equal distinct-patch coverage.

For step1…100000, derive three independent stream seeds from exact ASCII
`evr0008|{case_index}|{stream}|{step}`, with unpadded decimal integers and literal
stream `indices`, `augmentation`, or `pairing`: first8 SHA256 bytes interpreted
little-endian, masked by `(1<<63)-1`. Same case/step tuple is shared by all methods;
generators are recreated when used. Hash construction and generators are timed.
Rolling digests cover equal-sized actual base/used/order indices, first256 normal
rows and all three seeds. No gradient flattening, extra forward, learned pairing,
future information, reference access or post-result seed choice. Record loss every
50 completed updates and at the last step; losses do not select checkpoints.

## Two prospectively fixed endpoints from each trajectory

1. Exactly500 completed optimizer updates.
2. The first completed optimizer update at or after5.00 seconds on the training
   clock; copied checkpoint must finish by5.05 seconds (at most1% tolerance).

Continue the same fresh trajectory only until both are captured. Lock optimizer
completion time **before any endpoint copy**. If copying the500-update snapshot
crosses5 seconds after the update completed earlier, the next completed update
triggers the wall endpoint. If both triggers coincide, reuse an immutable copy
and disclose sharing. A greater-than5.05-second capture invalidates the trajectory;
retain its failure, do not retime, retry or extend the budget.

The training clock includes seed generation, batch draws, sorting/permutation,
Gaussian draws/factor products, equal-sized stream hashes, forward/backward,
Adam, finite checks, loss recording and snapshot copying/content hashing. Common
input/model/Adam/factor setup is separately timed without method-specific cached
work. The earlier endpoint's snapshot overhead remains charged to the later
clock. Full-core cooperative90 seconds and safety100000 updates; outer120 seconds
per trajectory,900 seconds for the full study; four CPU slots/numeric threads,
one Torch interop thread,8GiB address-space limit. Do not relax bounds after failure.

## Locked inference and cost accounting

Save both endpoint weights before inference. For every method and endpoint use
identical50-draw R2R inference at alpha0.5, chunk1024, full122×122 valid region,
float64 MC accumulation cast to float32, no output clipping/correction or extra
direct inference. Seed is first8LE SHA256 of exact ASCII
`evr0008|{case_index}|inference|0`, masked to63 bits. All four methods and both
endpoints share draws within case; these streams are distinct from training.
Exactly750 forwards and36,465,800 normal scalar draws per endpoint. Inference
order is updates500 then wall5s; save/hash outputs and full receipts.

Report setup, training elapsed/update count, checkpoint copying and disk write,
weight restore, complete MC50 inference and prediction saving. Also report actual
full-trajectory outer wait4 wall including source checks, Python/runtime loading,
shared setup, both endpoints, output/checks and shutdown. No unmeasured GPU claim.

For an endpoint's **allocated outer pipeline cost**, start with that actual outer
wall and subtract only (a) measured training-clock time after the endpoint's
capture and (b) the other endpoint's separately timed weight-disk-write and
restore/infer/save intervals. All shared runtime/setup/checks/receipt/shutdown
costs and earlier snapshot overhead remain charged. Require finite positive costs
at least as large as the separately recorded input+shared-setup+endpoint-training+
its-own-weight-write+restore/infer/save cost. Report both numbers. This is an
explicit allocation of one shared trajectory, not two independently timed runs;
endpoint outcomes are correlated. The outer measurement includes up-to50ms wait4
polling granularity. Full actual consumed compute is reported too.
The cost gate below uses allocated outer costs, not training time alone.

## Fixed scoring and demanding development continuation rule

Use the original independently tested unclipped PSNR and first-difference metrics
on all64 saved arrays. Data range1; crop references `[3:-3,3:-3]`. Preserve per-case
scores and8×8 block SSE/counts. At each endpoint, use per-case best PSNR of the
three controls, ties ordered independent→random_antithetic→same_patch. This is an
oracle comparator, not a deployable selector. Publish every control, seed and harm.

All of the following must hold, separately at **both** endpoints:

- Mean paired PSNR gain over this best-control comparator≥0.5dB.
- At least3 of4 image-token mean gains>0.
- Every image-token mean gain≥−0.2dB.
- Mean pooled first-difference MSE≤1.05×that of the corresponding PSNR-best control.

Additionally, total mean-pair allocated outer cost divided by independent's at the
wall5s endpoint≤1.10. All32 trajectories/64 predictions, unchanged-input/freeze
bindings, matched initialization and500-update shared streams must pass; all
predictions finite, all reference identities verified, independent saved-array
arithmetic replay must agree. Incomplete runs are invalid, not selected-case
results. Undefined finite PSNR gains are not evaluable, never silently infinite
successes. No population CI from four reused tokens or two noise draws.

A complete pass means **continue_to_novelty_review_only**. A valid failure means
**stop_this_candidate**. No repeated LR/alpha/feature/budget tuning to rescue it.
Only a separately reviewed rationale can justify a distinct mechanism/protocol.
A pass prompts denoising-specific novelty and stronger-baseline reassessment
before any new data acquisition. It does not automatically justify a manuscript.

## Unchanged paper gate and publication

Still require≥0.5dB over strong matched-resource baselines, independent datasets
and physically separated held-out groups, paired95% scene-cluster lower bound>0,
gains on≥2/3 held-out groups, white-noise harm≤0.1dB, texture/failure accounting,
noise/covariance regimes, ablations and independent replication. Matched-accuracy
runtime≤1.25× strongest comparator with full cost curves where needed. This
small known-covariance MLP screen has neither those baselines nor new scenes.

Publish original code, freeze, actual environment/commands/exits/times/hashes,
compact measurements, failures and limitations only. Raw images/references,
weights and reconstructable prediction arrays stay in the existing cloud cache.
One-shot directory is the existing project root's `antithetic-training-0008`;
never remove/recreate it or restart an incomplete run. Preserve all older data,
markers, freezes, 0006's open stdout gap and 0003's deferred missing-cache state.
No schedule change, additional fee, external submission or institutional claim.
