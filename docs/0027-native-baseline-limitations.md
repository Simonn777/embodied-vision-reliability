# Pinned native three-point baseline: source-limited review

Reviewed before the 0027 manufactured cohort. No solver, test, build, import, fixture generation, or benchmark was executed for this review. This establishes a conditional source defect, not its frequency or effect on any physical input.

## Exact source and branch

Pinned PoseLib commit `6a636b1d84273d5e3ba97676463476569611572a`, file `PoseLib/solvers/relpose_monodepth_3pt.cc`, SHA256 `6687da6a6f37f30f65aeb0223bc0853040ade8c476ac342bf9b35e3e4f7ccb04`. Local source is under `work/0016-preparation/inputs/PoseLib-6a636b1d84273d5e3ba97676463476569611572a/`. The existing 0018 copy has the identical file hash, so it is not an alternative implementation.

At lines 88–101, let `n` be the returned quartic-root count. The code allocates `sols(3,n)` and sets `m=0`. For root index `ii`, it computes `ss`; `ss < 0.001` skips the root. Otherwise it writes all three entries to column **ii**, then increments **m**. At the end it retains only columns `[0,m)` with `conservativeResize(3,m)`.

Consequently, let `I` be the indices that pass this branch and `m=|I|`. The result contains exactly the intended initialized roots iff `I={0,...,m−1}`. Whenever a skipped index precedes a retained index, `I` is not that prefix: at least one skipped/unassigned column is retained and the same number of accepted columns outside the prefix are discarded. For example, with two roots and pass pattern `[false,true]`, column 1 is populated, then resizing to one column retains unassigned column 0 and discards column 1. This is an indexing argument, not an executed numerical example.

The matrix is not explicitly initialized at line 91. The inspected build flags do not declare `EIGEN_INITIALIZE_MATRICES_BY_ZERO` or `...BY_NAN`. Unassigned entries therefore cannot be treated as meaningful solutions; even under an initialization macro they would not be the omitted root. The downstream loop, lines 144–181, consumes every retained column, checks pre-refinement depths, refines its parameters, constructs a pose, and emits it. External finite/proper/positive checks can reject bad returned candidates, but cannot restore a discarded legitimate candidate.

The quartic routine `PoseLib/misc/univariate.cc:211–265` emits roots from two quadratic factors followed by Newton updates; it neither receives `k6,k7,k8` nor sorts by this caller's `ss` predicate. There is no source contract establishing the required pass-prefix ordering. This review does **not** prove that a positive-depth physical three-point fixture reaches a non-prefix pattern, that omitted algebraic roots would survive the later cheirality checks, or that a previous result was affected. Those stronger claims would require separate evidence.

The `ss < 0.001` cutoff itself is distinct from this compaction defect. It excludes sufficiently small squared relative scales and is not a scale-gauge-invariant physical criterion. It should remain disclosed as an author implementation policy; silently removing it would change the baseline beyond a bookkeeping correction. NaN comparisons also do not constitute a finite-value guard. Our wrapper retains explicit invalid outputs, but this does not establish root completeness.

## Decision before outcomes

The current V/D8 comparison must be qualified prospectively. A passing single physical API test establishes the interface and that fixture only; it cannot establish absence of this branch. The unchanged native binary can remain a labeled reproducibility control, but it should not be the sole sufficient basis for claiming that T generates better hypotheses than the established three-point construction.

The narrowest trustworthy *candidate correction* is to write the accepted root to column `m` at lines 96–98, then increment `m`, retaining the original root order, cutoff, polynomial, refinement and pose recovery. This preserves the intended mathematical procedure. Before using it, separately freeze the source delta and qualify compaction on fixed pass patterns plus physical API behavior; do not search the future cohort for a favorable fix. A paired unmodified control may remain descriptive. No such correction/build has been performed in this review, and the corrected implementation is not yet qualified.

An independently implemented three-point distance/quartic solver would be another possible comparator, but has a larger qualification burden. No already-qualified, corrected alternative was established from the available local sources. Shared-/varying-focal solver variants are different models, not automatic replacements. No current-upstream fix or external implementation correctness is claimed.

If the immediate screen proceeds unchanged, its advancement rule must explicitly remain provisional with respect to V/D8: T failing its own exact/noisy necessary conditions can close T, but T beating these pinned controls cannot by itself establish a gain or admit a contribution claim. The existing requirement for a full nonminimal same-observation control remains necessary independently of this source defect.

Root's pre-outcome decision adopts that limited option: retain the unchanged V/D8 implementation for this necessary screen, disclose the compaction limitation, and require an independently qualified corrected or alternative three-point control **before** advancement to F-all. Thus passing the present numerical gate alone is not advancement. This review accepts that interpretation; it does not upgrade the pinned baseline to a complete root enumerator.
