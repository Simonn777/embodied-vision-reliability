# 0023: directional reference instability failed the development gate

**CLOSED / no manuscript.** The candidate achieved 56.497091 AUC@5 versus 61.984444
for the strongest same-observation geometry control: **−5.487353 percentage points**.
All four efficacy conditions failed. No subset is excluded and no parameter will
be tuned and re-tested on these 24 queries.

| Method | AUC@5 (%) | AUC@10 (%) | AUC@20 (%) | Native-E fallbacks /24 |
|---|---:|---:|---:|---:|
| Geometry multiview BA | 61.984444 | 75.643333 | 83.655000 | 8 |
| Joint depth-field BA | 57.863271 | 73.582746 | 82.624706 | 8 |
| Residual scalar | 56.642436 | 72.972329 | 82.319498 | 8 |
| Hessian prior | 56.549094 | 72.925658 | 82.296162 | 8 |
| Directional candidate | 56.497091 | 72.899656 | 82.283162 | 8 |
| Trace-matched deletion scalar | 56.008201 | 72.655211 | 82.160939 | 8 |
| Native E (reference-filtered query) | 52.122880 | 68.629218 | 78.064609 | 0 |

Every method retains 24 outcomes across three preselected places. The primary
error is max(rotation angle, directed translation-direction angle). The seventh
estimator uses reference-track-filtered query matches, so it is not strictly
query-only. AUC uses the frozen trapezoidal convention. This is a custom
multiview development diagnostic, not the official Map-free leaderboard task.

## Prespecified decision

| Condition | Required | Observed |
|---|---|---|
| Overall AUC@5 over best control | ≥+2 pp | −5.487353 pp |
| Positive scene margins over best scene control | ≥2/3 | 0/3 |
| Worst scene margin | ≥−1 pp | −9.666212 pp |
| Overall AUC@20 loss versus best control | ≤0.5 pp | 1.371838 pp |

The candidate is +4.374211 pp over native E, but −1.366180 pp versus full joint
depth-field BA and −0.052003 pp versus the Hessian control. These comparisons do
not establish a novel gain. Geometry BA is an existing method, so its improvement
over the two-view estimator must not be presented as our contribution.

| Place | Candidate AUC@5 | Best control AUC@5 | Margin (pp) |
|---|---:|---:|---:|
| s00475 | 83.307601 | 92.973813 | -9.666212 |
| s00478 | 77.031977 | 81.910014 | -4.878036 |
| s00513 | 15.360294 | 15.360294 | +0.000000 |

## Execution and provenance

- Original input freeze: `acf502b831ad9e3485ab56549217c8a603d56bf7`; exact CI 39341051566 passed before allocation. All three places and their unselected frames remain development forever.
- Prediction/evaluation freeze: `cb520839515381db125f64ea3bcceb434444a85b`; exact CI 39342283046 passed before fitting. The narrow calibration-order repair and original pre-fit failure remain preserved.
- Frontend: all 51 images and 120 match caches completed once; outer 103.6027065 s, 2,142,832 KiB sampled peak.
- Prediction02: all 24 query rows and seven estimators sealed before labels; outer 298.587250167 s, 1,173,520 KiB, exit 0, no offline violations or remaining descendants.
- Evaluation02: outer 7.046813708 s, 340,128 KiB, exit 0. Only the three planned pose members were acquired after the seal: 117,919 response-body bytes, 9 requests; 27 selected numeric pose rows parsed.
- The earlier prediction01 failed before fitting in 1.409816375 s and is not hidden. Every operation stayed under 4 threads, 8 GiB and 1800 s.
- These are full study-operation costs covering shared work and all arms, not isolated matched-cost method benchmarks. No final held-out or two-source generalization claim is made.

The [public evaluation](evaluation-public.json) retains every estimated pose,
error, status, aggregate and gate. Authored GT poses are omitted from that public
derivative. [Publication bindings](publication-bindings.json) preserve its original
and public hashes, byte-identical receipts, log redactions and all private sealed
output hashes. Original 57.98 MB scene records, input arrays and dataset payloads
remain private, including every failed initialization and attempted fit.

See the [frozen protocol](../../../protocols/0023-mapfree-directional-instability.md),
[technical amendment](../../../docs/0023-cache-verification-repair.md), and
[pre-result limitations](../../../docs/0023-pre-result-limitations.md).
The [independent result audit](../../../docs/0023-result-audit.md) reproduces all
168 errors and aggregate metrics; the
[saved-result mechanism diagnosis](../../../docs/0023-mechanism-diagnosis.md)
records initialization, convergence and cost limitations without refitting.
The broader research objective remains open; this hypothesis is closed.
