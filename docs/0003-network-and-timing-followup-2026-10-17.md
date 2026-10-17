# 0003 follow-up: default runtime route works; inventory and cohort still blocked

October 17, 2026. AI-assisted bounded infrastructure/data audit and prospective
timing clarification for PR #16. All published v1 inputs/receipt remain unchanged.
No model, old-test analysis, threshold fit, numeric GT decode or new effect result.

## Network diagnosis and acquisition

The prior v1 audit stripped the runtime environment with env-i. A names/booleans-only
check found standard HTTP/HTTPS proxy and CA variables present (uppercase/lowercase
HTTP_PROXY, HTTPS_PROXY, NO_PROXY; SSL_CERT_FILE and REQUESTS_CA_BUNDLE). ALL_PROXY
was absent. No values, credentials, private hostnames, new proxy/direct IP or changed
security settings were used/logged. One tiny official HF metadata request using
unchanged runtime defaults returned HTTP 200: 2,328 bytes in 0.209065 s. It confirmed
the requested revision's sample size and LFS SHA256. The comparison establishes that
the default approved route works; it does not isolate which individual proxy/CA
variable caused the earlier stripped-environment failure. V1's DNS record remains
accurate for that environment, not evidence of permanent default-route unavailability.

Exactly one sample GET followed, without retries, on the unchanged configured route.
Full size **3,324,284,510 bytes**, SHA256
`436e6a5a01702cebcf8be97254ea583457fd60177e2aba671b90ed578912167f`, matches the
pinned official HF metadata. Actual parent wait: **82.836491 s**, peak RSS
**18,092 KiB**, child exit/raw wait **0/0**. Numeric variables=2, two CPU affinity
slots, 6 GiB AS, external timeout 900 s; no installs or new fees. The archive stays
untracked in cloud cache; no raw data/source pages are committed.

Pinned generator README/reprojection/tree and the export/tracking sources were
retrieved as bounded public source files, hashed and read without executing them.
The reprojection implementation applies RT to homogeneous world coordinates, then
K and perspective divide; this supports the proposed convention in source, but
not its correctness on actual sample annotations. Units/FPS and sample geometry
remain unverified. Source hashes/URLs and HTTP byte accounting are retained in
`results/0003-network-followup-20261017/`; no complete source pages are republished.

## Safe inventory stopped at the existing frozen gate

The metadata-only tar walk applied unchanged v1 path/type/duplicate/count/size
limits. It stopped with observed exit/raw wait **1/256** after **2,756 headers**
and **1,789,467,656 bytes of member sizes encountered**, because an annotation
member exceeded the frozen **512 MiB** single-annotation cap. The encountered
size includes the offending header's declared size; it is not bytes decoded into
GT arrays. The prefix includes root/video/scene labels, not a complete scene count.
Inventory completeness is false; it took **2.904198 s**, peak **14,080 KiB**.
No cap relaxation, second archive GET, extraction, RGB decode, NPZ/GT array read,
frame-0 GT query read, model or geometry computation followed. The original
partial inventory/failure receipt is retained. Exact failing filename/size was
not saved separately by that first walk; the report does not invent either or
claim a complete member manifest. Further content access requires resolving the
cap under a separately explicit revision; this follow-up does not do so.

All sample material is conservatively **AUDIT_UNKNOWN_FAMILY / development** while
story/session/assets/multiview relationships are unresolved. No independent held-out
family or 20-group cohort is established. Different file names/hashes do not prove
independence or checkpoint non-exposure. Final cohort/confirmation remain BLOCKED.
Acquisition/body identity is now verified; geometry, units, family provenance and
precision feasibility are not. Access accounting separates whole-object bytes,
member metadata, RGB, GT arrays/frame-0 fields and predictions (the last four are 0).

## Prospective timing clarification, not changed targets

The [separate timing addendum](../protocols/0003-timing-addendum-v1.md) resolves a
v1 ambiguity BEFORE any new cohort GT arrays/predictions. Before final cohort freeze:
metadata, quarantined development convention verification and indispensable frame-0
query fields only, truthfully logged as GT access when used. Freeze whole family/
point/window allocation, source and manifest; collect and freeze ALL predictions
before held-out GT support/projection adjudication. Unchanged >=95% GT support and
99.9% within 0.5 original px gates then apply. Required-family failure/undefined
support makes continuation INCONCLUSIVE; retain all allocated and zero-query units,
no drop/refill/refit/convention switch, no diagnostic-anchor replacement or primary
denominator change. Pre-inference quality is PENDING_POST_PREDICTION, not passed.

V1 readiness() remains a legacy design diagnostic with no execution authority.
The future separately frozen cohort/pipeline must bind v1 plus this addendum and
implement both phases. The 20% relative goal, 8 px/weight 1/cap 10 reference policies,
64-frame/two-query budget, statistical gates, old results and resources stay unchanged.
The new typed phase helper has hand-computed fixtures, not a model/data runner.
New addendum inputs are published first, then an independent receipt including the
unchanged canonical v1 receipt and full eight-file hashes, without self reference.

## Historical provenance limits clarified by parent review

The earliest 27-test helper receipt is not fully source-bound. The 29/31-test
snapshots preserve code/tests and hashes, but their exact earlier protocol text
bytes were not separately retained. Do not call those records complete nine-file
reproductions. The final 35-test pre-publication check binds all nine hashes, which
match input commit `66e5330ad7404b4e7678d006e2537e4703f4683d`. The original v1 report,
receipts and earlier limitations are preserved, not retroactively repaired. The
new timing fixtures are separately source-bound. This is a provenance clarification,
not an old-result reanalysis. A local parent invocation argument error occurred
before starting any sample child/GET; its setup correction is disclosed in new
follow-up evidence, not confused with a download retry.
