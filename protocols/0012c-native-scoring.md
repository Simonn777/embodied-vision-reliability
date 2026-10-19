# 0012c: native verification-time diagnostic

0012b reduced direct residual work45.38% versus block-partial scoring, but its
Python implementation was44.65times slower. This protocol tests whether that
mechanism translates to actual scoring speed before any pose-accuracy study.
It is further exploratory use of the same development scene, not confirmation.

## Fixed workload and implementation

- Same16 pair keys,10,000 matches each and64 stored fundamental matrices from
  the original0012 JSONL. No new hypotheses, poses, calibration, labels, images
  or held-out coordinates. Threshold1pixel, leaf32, initial128 cells,
  maximum128 refinements and unchanged mean/variance threshold formula.
- One dependency-free C++17 backend implements five modes:0 full scoring,
  1 ordinary block256 partial scoring,2 the moment+Cantelli selector with full
  fallback,3 partial scoring with the same moment-derived initial incumbent and
  order as mode2, and4 the bound selector with block-partial fallback instead of
  full fallback for overlapping intervals. Mode4 is the primary candidate;
  mode2 retains the original selection structure as an ablation. Mode3 charges
  its entire tree/coarse-bound ordering cost, and mode4 charges repeated leaf
  plus partial work without pretending the repeated work is cached. This hybrid
  change is specified before native data execution. Point permutations are
  generated as in0012b with seed20261020. Strict guarded comparisons and smaller
  index tie-breaking apply to every mode. Zero denominators cost the cap.
- Compile with system clang++, `-O3 -std=c++17 -fno-fast-math -ffp-contract=off`.
  Single native thread; record compiler/version/command and source/library hashes.
  No CPU/GPU mixing, approximate fast-math, or per-method compiler flags.
- Build candidate tree in C++ with the same deterministic4D median partition.
  Compute moments at leaves and combine parent moments/extrema bottom-up. Share
  quadratic/denominator calculations between original and added bounds. These
  implementation changes reduce repeated work but do not change the formulas.
  Floating-point guard compatibility requires synthetic independent review.
- Store45 unique symmetric moment entries; reuse already evaluated coarse cell
  intervals during refinement and build the refinement heap only when needed.
  Native node counters therefore differ from the unoptimized Python path;
  every actual new bound evaluation and repeated direct residual is charged.
- No oracle score or winner enters selection. Stored references enter validation
  only. Baselines do not construct unused candidate trees.

## Validation and timing

First validate native full scores against all saved NumPy scores, requiring
absolute difference <=1e-8 +1e-12*abs(saved_score). Scalar/reduction differences
motivate this numerical comparison tolerance; **winner indices must agree
exactly**, without a ranking tolerance. Check every initial candidate cell and
aggregate interval against native direct scoring with no test-side containment
tolerance. Any observed violation, nonfinite value, wrong winner or source
mismatch stops the run, retaining all output.

Run three timed repetitions per method and pair. For pair index i and repetition
j (both zero-based), mode order is `(i+j+offset)%5` for offsets0 through4.
Each call starts with fresh method setup; no cached
candidate tree or matrices across repetitions. Record native setup/scoring times
and Python call duration; fixed dataset loading and validation are separate.
The primary total is the outer ctypes call duration, plus permutation setup for
modes1,3,4. Generate the shared permutation once per pair, measure that generation,
and charge the same measured cost to each corresponding fresh-call repetition;
this prevents a baseline from receiving free preparation. Internal native phases
are also reported. Report every repetition,
each pair's median, all work counters and ratios, then sum per-pair medians.
No best-of-run timing, omitted pair or claimed independent n=16 confidence interval.

Before calibrated-pose integration, require primary mode4 aggregate median time
<=2/3 of **each** mode0,1,3 aggregate median times, correct winners
on every repeat and pair, and zero observed containment violations. Passing is
only a scorer-level speed result on used development data. It is not complete
estimator speed, task accuracy, generalization, floating certification or CVPR
readiness. The original independent task-quality gate remains in force.

Resources: existing data only, no external library/model/package downloads,
single native thread, <=4 Python/BLAS threads, roughly8GiB,30-minute hard process
timeout. Build products stay outside Git; source, complete small measurements,
failures and limitations are published. Freeze source/protocol after synthetic
tests and independent review before the one real-data diagnostic invocation.
