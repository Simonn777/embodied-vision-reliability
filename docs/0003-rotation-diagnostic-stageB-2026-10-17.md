# Frame 0 rotation failure: recorded diagnostic

October 17, 2026. AI-assisted **failure-follow-up / exploratory diagnostic** on
permanently quarantined development data. The one frozen invocation completed its
recording, with the **original rotation gate still failed**. No projection, geometry
repair, model inference, old-test analysis or new improvement result was attempted.

## Freeze before access

Independent review approved input commit
`8a7591b822e59c52a70dc4643f8e9cfcfa368c97`. The separate canonical
[receipt](../protocols/0003-rotation-diagnostic-freeze-v1.json) was published in
`8ffb8bcfbc5c7a9fe87e5a624d3f3dc1e0d97072` at 23:44:44 UTC, SHA256
`957f830c6a67e6923ed82cba90d1ddf9d74558f03f93763abd82e01aad095820`.
The input commit contains no receipt. A fresh fetch and the unchanged verifier
checked all 12 mandatory current/index/HEAD/input/remote blobs and ancestry, four
prior canonical chains (30 distinct inputs), old failure/carry and old marker.
Preflight recorded verification at 23:45:14 UTC before diagnostic start. The actual
execution source is the receipt commit above, distinct from later publication.

No active candidate Python process, output or new marker was visible before work;
shared locks were available. This is visible process/reservation evidence, not a
claim about inaccessible sessions. The unchanged runner acquired all shared/new
locks and created an exclusive persistent new marker; no retry was performed.

## Direct observations

Only the existing selected NPZ was raw-hashed and its two camera members opened.
Its identity is 98,527,880 bytes / SHA256
`382d1edca0d6d88279ed1a61ee63c9ba27b7c5418f2269f4592c1658fd7d8e77`,
member `sample/r4_new_f/anno.npz`, PointOdyssey v1.2. The observed arrays remain
C-order little-endian f32, shapes [849,3,3] and [849,4,4]. Exactly original frame 0
K9+RT16 was decoded: **25 scalars / 100 payload bytes**. The complete lossless,
attributed [excerpt](../results/0003-rotation-stageB-20261017/audit/camera-excerpt.json)
contains original stored bytes, float.hex, rational values and offsets. No
trajectories, flags, RGB, other frames/scenes, metadata or TAR was reread. Buffer
reserves are conservative possible physical reads, not extra decoded values.

All coefficients are finite, both exact homogeneous rows pass and focal lengths
are positive. With unchanged tau **9.5367431640625e-7**, all three rotation checks
fail:

| Original check | Observed defect | Excess above tau | Defect / tau |
|---|---:|---:|---:|
| max abs(R Rᵀ − I) | 0.0003642439842224121 | 0.00036329030990600586 | 381.9375 |
| max abs(Rᵀ R − I) | 0.0003985762596094844 | 0.00039762258529307815 | 417.9374999962747 |
| abs(det(R) − 1) | 0.000009985611434615294 | 0.000009031937118209044 | 10.470672495663166 |

Both Gram maxima occur at zero-based [1,1]. det(R) is 1.0000099856114346.
Full signed Gram defects, row/column norms/dots, every original gate, exact Fraction
statistics and double-minus-exact differences are in
[diagnostic.json](../results/0003-rotation-stageB-20261017/audit/diagnostic.json).
This identifies the numerical trigger: stored R fails the fixed proper-rotation
requirements. It does not identify the actual generating cause as scale, shear,
precision conversion or another mechanism.

The exact necessary bounds for **one final nearest-f32 rounding of an exact proper
rotation**, Gram 6u+3u² and determinant 9u+27u²+27u³ with u=2^-24, are exceeded.
That narrow hypothesis is excluded; upstream accumulated error, earlier precision
steps, export semantics and generation/reader history are not thereby excluded.

| Exact binary16 descriptor | Representable / total |
|---|---:|
| K | 9 / 9 |
| R | 9 / 9 |
| RT | 14 / 16 |
| Translation | 1 / 3 |
| Homogeneous row | 4 / 4 |

The two nonrepresentable values are RT[0,3] and RT[2,3]. Every R value lies on the
half grid. This is compatible with direct half storage and widening of R, but does
not prove it: identity/integer values also qualify. The translation mismatches
exclude only an all-values direct f16→f32 path without intervening arithmetic for
this entire RT. An earlier half step followed by computation remains possible.
No converted matrix was evaluated, repaired or projected.

## Source-supported mechanism versus unverified history

The already cached source at revision a053342a72fed32af4134b4ca0c44a78a3ac0fdb,
`utils/gen_tracking_indoor.py`, explicitly casts K/RT to float16 before saving
(lines 198–220, including 200/201 and 209/212). Its hash and exact reviewed ranges
remain in the immutable [source/rights record](../results/0003-rotation-stageA-20261017/source-and-rights.json).
This supports a **possible precision mechanism**, not the actual generation chain
for this sample. Saved key/schema differences, subsequent computation/conversion,
actual renderer/scene configuration and version history remain unverified. Source
plus half-grid membership does not establish causation or causal probability.

## Actual execution, accounting and independent check

| Phase | UTC start → end | Wall seconds | Peak RSS KiB | Observed exit / raw status |
|---|---|---:|---:|---:|
| Published-chain preflight | 23:45:14.156576 → 23:45:14.800450 | 0.643879 | 13,952 | 0 / 0 |
| One diagnostic | 23:45:14.801012 → 23:45:15.267788 | 0.466789 | 16,128 | 0 / 0 |
| Saved-record verification | 23:46:05.973923 → 23:46:06.006900 | 0.033001 | 11,264 | 0 / 0 |

Cloud CPU: AMD EPYC 9V74, Python 3.12.14, Linux 6.18.55 x86_64. Each child used
env-i-equivalent explicit PATH/PYTHONPATH plus five numeric thread variables 2,
affinity [0,1], 6 GiB address-space limit and external timeout 900 seconds. Actual
commands, CPU time, source/config/protocol/receipt hashes, stdout/stderr and raw
wait4 records are in [execution evidence](../results/0003-rotation-stageB-20261017).
Each parent has no further observed supervisor exit receipt. Public parent records
normalize only the absolute repository source-key prefix; original receipt hashes
are retained, measurements/logs unchanged, raw originals remain private scratch.

Carry was 20,614,538 returned + 2,163,798 reserved = 22,778,336 bytes. New returns
are **378 bytes** (22 envelope + 256 headers + 100 payload); new reserves are
**131,442** (two 65,536 streams + 348 directory + 22 EOCD). Final charged total is
**22,910,156 / 67,108,864 bytes**, including directory-reserve subset 1,480. Counts
are complete for attempted work and both camera-member statuses are completed.
The frozen runner's top-level `camera_access_state` retains its last STARTING label;
completion is recorded in overall status, member statuses and counts, not that label.
No unread tail/full-member CRC verification is claimed.

The independent saved-record verifier redecoded only the published 25-value excerpt,
not the cache. It checked bit/hex/rational/offset/span hashes, 18 exact Gram entries,
determinant via **Gaussian elimination** (production uses permutation expansion),
and half-grid flags via an **exact integer-significand lattice** (production uses
half roundtrip), with zero tolerance. It also checked budgets and source/log/wait
bindings. No real cache access, new policy or threshold resulted. Full verifier
source, logs and actual parent exit are published.

An initial receipt-construction attempt omitted PYTHONPATH and failed before any
receipt/cache access; the setup history preserves it. The corrected explicit src
path produced the reviewed canonical receipt. No diagnostic failed or was retried.
Unchanged source already passed 19 targeted fixtures and 290-test candidate CI;
no synthetic suite was rerun locally just to manufacture another record. Final
publication-head standard CI is verified separately in the PR.

## Rights, preservation and next step

The 25-value excerpt and derived numeric diagnostics retain **PointOdyssey
CC BY-NC-SA 4.0**, with authors, revision, source/member hash, license link and
unchanged-values statement in the excerpt. They are not repository MIT; new
original software/prose retains its original terms. No full array, RGB, weight or
third-party source page is redistributed. No transfer/download/install, paid
service/API/GPU/credits, account/permission or schedule change occurred.

All existing tracked historical files except current/next/README and appended note
remain byte-identical, including frozen input chains, prior failures, both Stage A
phases and old marker; the new marker is retained. See publication closure for
counts/digests. This is a recorded failure diagnostic, **not geometry passed**, a
model improvement, independent cohort, robot result or confirmed root cause.
Whole sample remains development, scene/family relations and units/FPS unknown,
and the confirmatory cohort remains blocked. Next is independent review of these
saved values and source-generation evidence; any new access/geometry proposal must
be separately justified and prospectively frozen. Do not repair this matrix, loosen
tau or rerun the old audit automatically.
