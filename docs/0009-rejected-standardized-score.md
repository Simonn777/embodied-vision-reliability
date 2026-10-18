# 0009 independent design review — NO-GO for pure z before performance

2026-10-18. Read the lane-A memo/addendum, primary-followup memo and geometry
prototype/source/tests. No performance execution, real data, network, installation
or repository changes. This is a design judgment, not an experimental result.

**Reject the current pure-z homography ranker before a performance trial.** Keep
the narrower nonuniform-accidental-outlier question open. Do not add a floor,
variance epsilon or hybrid score after this rejection, and do not acquire a dataset
to compensate for a weak mechanism. The arithmetic prototype remains valid as a
prototype; its 18 artificial tests are not evidence of task improvement.

## Direct mathematical problem, and its limit

For n verification pairs, let a true model's compatibility matrix be a k-by-k
all-one principal block and zero elsewhere, 1<=k<n. Then

    N=k; mu=k²/n;
    var=k²(n-k)²/[n²(n-1)]; z=sqrt(n-1).

Its score is independent of the amount k of true support. For a competing matrix
with q isolated diagonal ones and zero elsewhere,

    N=q; mu=q/n;
    var=q(n²-2n+q)/[n²(n-1)];
    z²=q(n-1)³/(n²-2n+q).

For n>2,q>1, this is strictly larger than n-1. Thus four isolated wrong supports
can outrank a compact true cluster of 32 at n=128. This is a symbolic consequence,
not a mock numerical result. It can arise geometrically with one common endpoint
set: identity and a noncollapsing translation; true endpoints in a tight cluster,
separated erroneous pairs following the translation, and remaining endpoints
outside both tolerance neighborhoods. Unique coordinates do not remove it.

A dense cluster also gives weak global extrapolation constraints, so this does
not establish that raw count always identifies the true model. It establishes
that the z objective measures standardized rarity, rather than reliably rewarding
stronger geometric consensus. Real feature-spacing rules may make this extreme
less common; their residual headroom is empirical. This is not a proof that z
can never improve any accidental-outlier subset.

Zero variance correctly means all assignments have the same support; the observed
assignment adds no information. Neutral zero with an explicit noninformative flag
is appropriate, not infinity. More permutations or an epsilon cannot fix that.
Likewise, coherent periodic shifts do not refute the different problem of
nonuniform *independent accidental* matches. The rejection is of the present
ranker and its investment case, not all density-aware geometry.

## True tails and likelihood: distinguish the example, not a ready new method

For the k-block, T has a hypergeometric law. At observed maximum k,

    P(T>=k)=1/binomial(n,k).

For q specified isolated diagonal supports, at maximum q,

    P(T>=q)=(n-q)!/n!.

For n=128,k=32,q=4 the block has the smaller tail, unlike the z ranking. These are
exact laws for special matrices. The current moment implementation computes no
general tail. First two moments do not determine one; general compatibility
patterns require additional counting/approximation. Zero exceedances in a small
Monte Carlo budget cannot order extremely rare events reliably. No <=900-second
general exact-tail capability is established.

A fully specified alternative likelihood can retain support and geometry discarded
by z, but needs an explicit alternative/noise model and principled treatment of
fitted parameters. A likelihood based only on z keeps the same information loss.
Neither a tail nor a likelihood automatically supplies exchangeability, posterior
correctness or correction for selecting the best hypothesis. These observations
are **not authorization to substitute another candidate**.

## Prior art and a possible future falsification boundary

GMS CVPR2017 already uses local target-feature density in its wrong-match model
(the parent's official indexed §2.1–2.2 check: beta*m/M). ORSA's IPOL2012
registration work already uses a uniform-point/random-coupling null (§1.2).
AdaLAM normalizes support by expected accidental support and discusses textured
clustering. Thus first density correction, first random-coupling significance,
and new exact-moment mathematics are unavailable novelty claims. The newly
reported GMS/ORSA excerpts are parent-verified primary evidence, not a claim that
I completed both PDFs. Sources: [GMS](https://openaccess.thecvf.com/content_cvpr_2017/papers/Bian_GMS_Grid-based_Motion_CVPR_2017_paper.pdf),
[ORSA](https://www.ipol.im/pub/art/2012/mmm-oh/).

Only if a genuinely separate mechanism later merits testing, a bounded question
would be: **Does nonuniform accidental support leave useful model-selection
headroom after simple expected-support corrections?** No pure-z performance
screen is recommended now. A future prospective geometry-only screen could fix
128 proposal/128 verification pairs, 32 identical hypotheses, tau=3 pixels,
independent seeds, and <=900 seconds / four CPU threads / 8GiB. It must include
uniform accidental outliers, clustered accidental outliers, clustered true inliers
and isolated small false support; periodic ambiguity is a separate negative
control. Proposal construction cannot inspect verification labels. Planted-H
coverage, if used, must be disclosed as an oracle-pool ranking test.

Necessary same-pool controls are raw support, MSAC at the same residual tolerance,
N-mu and N/mu (mu=0 implies N=0; neutral zero), using the same empirical radius
counts. A future conditional-variance claim also needs the independent-row
variance sum_i p_i(1-p_i). These simple controls separate density correction from
without-replacement variance. They are not stand-ins for actual GMS/MAGSAC++.
Charge radius-graph construction to graph-dependent methods, not raw/MSAC; freeze
all ties, thresholds and seeds before outcomes. Evaluate fixed-grid transfer error,
not support. An exploratory target might require >=8 net successes in 64 clustered
accidental cases over the strongest fixed density control, with <=2 additional
failures/64 in uniform and true-cluster controls. Insufficient headroom, sparse
support failure or failure to beat simple correction means stop, not a variant
sweep. This is a conditional design sketch, **not another protocol to freeze now**.

## Minimum real-data qualification

The parent reports an official HPatches-hosted route with declared MIT metadata
and range-capable archive. Preserve that possibility for a suitable method;
metadata inventory, bounded exact file selection (e.g. <=100MiB), and applicable
image/data reuse terms still need a separate documented qualification before any
acquisition. HEB licensing remains unresolved. No current dataset is approved by
this review; synthetic-city trajectories or warped photographs do not replace
independent real-view confirmation.

Require coordinates, tentative-match identities and matching provenance, actual
identical mutual/ratio filtering, independent planar GT plus valid-overlap regions,
and physical-scene-separated development/held-out groups. Unique synthetic
endpoints alone do not establish post-filtering headroom. The eventual complete
pipeline needs pinned legitimate MAGSAC++ and relevant spatial-filter controls,
matched total CPU budgets, identical matches/sampling information and all failures.
SciPy availability with no cv2 does not provide that comparator. Real-scene and
novelty evidence remain missing; no manuscript claim follows from this review.
