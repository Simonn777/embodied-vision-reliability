# 0009: conditional assignment evidence for geometric verification

2026-10-18. **Candidate under review, not an established method or improvement.**
The research question is whether the structure of accidental geometric matches
can help choose a more accurate image transformation than strong residual and
simple density scores. This is a new direction after the failed 0008 quality
screen, not a continuation that changes its outcome.

## Mechanism to test

For a fixed proposed homography, build a binary compatibility graph between
the observed source and target feature positions at one fixed pixel tolerance.
Compare the observed matched-pair support with the support obtained by uniformly
permuting target labels. The proposed score is the exact upper tail under this
specified combinatorial model. A component-based rook-polynomial calculation
may make some graphs tractable on CPU. Its general counting problem is hard;
unsupported graphs must remain explicit failures rather than disappear from
the denominator.

The combinatorial identities and component factorization are classical. Prior
art already covers a-contrario geometry, random-coupling nulls and local density
correction. A short literature search has not established the originality of
this particular vision application. Moreover, descriptor filtering does not
make endpoint assignments uniform: this is a **candidate ranking statistic**,
not a calibrated probability that a homography is wrong or correct.

The earlier moment-only standardized score was rejected before performance
testing. It can assign the same score to dense support blocks of different sizes
and prefer a few isolated false matches. Exact tails retain more information but
still need not prefer the true model. The rejected variant will not be revived
by tuning its threshold, adding a support floor or omitting counterexamples.

## Evidence required to continue

The immediate investment is one bounded, prospectively specified synthetic
headroom check, after independent arithmetic and design review. It asks whether
there is a task-quality effect worth testing on real matches. It must include
uniform and clustered accidental matches, concentrated true support, identical
deduplication and a fixed shared hypothesis pool. Raw support, MSAC, a fixed
spatial-cell normalizer and expected-support excess/ratio are necessary controls.
Higher statistical scores, new software and tiny tail values do not meet the gate.
An oracle-covered hypothesis pool, if used, is only a ranking diagnostic.

A positive diagnostic would permit a small, separately frozen real-view
development experiment. It would not permit a paper. That experiment must verify
feature identities and scene grouping, reveal whether simple location deduplication
already explains any gain, measure unsupported graphs and count all computation.
Official MAGSAC++, relevant GMS and a-contrario implementations are required for
any later complete-pipeline claim; informal substitutes with familiar names are
insufficient. Current NumPy/SciPy availability does not establish those baselines.

Only subsequent independent, untouched real-scene evaluation could support a
method contribution. Before those outcomes, a full protocol must fix meaningful
absolute improvement, scene-level uncertainty, comparable CPU budgets, controls,
ablations and failure limits. The intended improvement is better geometric
accuracy or an explicit accuracy/cost tradeoff against the strongest relevant
baseline, not statistical exactness alone. No manuscript starts until those
predeclared criteria and an independent reproduction pass, and the novelty
claim survives a deeper primary-source audit.

## Resource and evidence boundaries

No paid API, GPU, service or additional fee. Scientific work remains on the
existing cloud CPU, at most four threads and 8 GiB, with a 900-second screen cap.
Old inspected tests are excluded. The 0003 missing-cache constraint, failed
0004/0006/0008 results and every existing marker/freeze remain unchanged.

An HPatches archive-directory inspection obtained filenames and sizes but failed
its strict HTTP request/body accounting because a redirect was not fully metered.
Its failure and original artifacts must be preserved. No image or homography
value was read. A corrected, separately reviewed transfer design is necessary
before any future member acquisition; directory metadata alone does not establish
scene independence, payload integrity or performance headroom.

This work is AI-assisted. No institutional affiliation, conference acceptance,
new mathematical identity or practical improvement is claimed.
