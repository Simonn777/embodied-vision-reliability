# 0022 independent asset-acquisition review

**Decision: the fixed acquisition is admissible after its exact source freeze
and CI pass.** This decision does not admit installation, model loading,
manufactured inference, benchmark inputs, or a scientific experiment. Those
implementations require their own complete freeze and review. No research gain
or manuscript follows from acquiring these assets.

The review read source, manifests, protocol, and saved artificial-test evidence;
it did not make HTTP requests, download payloads, import numerical/model
runtimes, install packages, or read benchmark values.

## Verified scope and provenance

- Independently summed **105 source/model files, 194,142,010 bytes**, and
  **14 wheels, 76,809,693 bytes**. Response-body limits are separately
  210,000,000 and 90,000,000 bytes; the additional 20,000,000-byte metadata
  reserve is an accounting reservation, not another payload allowance.
- All 102 source-layout entries map uniquely to safe relative destinations and
  to their declared immutable upstream revision URLs. All 13 utils3d lazy export
  maps are present. This checks the selected layout, not runtime completeness.
- Source Git blob identities, the MoGe publisher SHA256, and wheel publisher
  SHA256s precede acquisition. All 14 wheel URL/size/SHA tuples agree with the
  saved dependency metadata manifest. The two exact official LightGlue release
  URLs explicitly use first-download hashes because publisher digests are
  absent; their successful acquisition will not establish publisher-hash parity.
- The helper enforces HTTPS/host restrictions, exact size/hash checks, bounded
  redirect/error bodies, total body accounting, exclusive partials, atomic
  no-overwrite publication, and failure retention. It contains no installation
  or model execution. Signed redirect URLs are redacted.

## Freeze and supervision

Two source-binding weaknesses were corrected before acquisition: the wrapper
now compares every bound file with `git show <commit>:<path>` (so an untracked
manifest cannot pass), and the supervisor requires the child's expected commit
to equal its own. The wrapper retains the verified bytes and parses both
download manifests from them; receipt hashes refer to those same bytes.

The supervisor checks its own and child source bytes, launches the allowlisted
acquisition operation in a new session, limits the operation to 1800 seconds,
samples descendant/group RSS against 8 GiB, and records process/log outcomes.
Cleanup failures cannot produce a completed receipt. RSS is sampled, so transient
peaks can be missed; HTTP accounting measures observed response bodies, not
transport overhead. Abrupt termination can leave an inner started/partial
receipt; the outer failed supervision receipt must govern eligibility.

The saved acquisition receipt reports **38 invented-byte/fake-HTTP tests passed**,
exit 0, wall time 0.124124667 seconds. This review independently matched its log
SHA256 and both recorded source SHA256s to the current files. It did not rerun
those tests or treat their duration as acquisition performance. Supervisor source
was reviewed separately; that 38-test count does not cover the supervisor.
Its author also records passing mocked commit/launch, descendant-RSS,
normal/nonzero-exit, RSS-limit and interruption checks in
[runtime-artificial-checks.json](../results/0022-preparation/runtime-artificial-checks.json).
That source hash matches this review. Those checks were supplied as an author
summary, without a persisted test script or process-time receipt; this review
does not claim to have reproduced them. Real process killing and the timeout
path remain untested. Installer checks listed in that record do not extend this
acquisition-only admission.

The public 0022 preparation files were scanned for local absolute user paths and
credential-like strings; none were found. Upstream payloads remain private and
must not be committed. Existing shared Python/Torch/NumPy installations remain
untouched by this acquisition.

## Reviewed byte identities

| File | SHA256 |
|---|---|
| `acquire.py` | `5b09d48b3b81fcf1d62b90871475813e26544934711244dcff2d429d4ad7fa11` |
| `acquire_operation.py` | `28aed9e86d764ca3f336d7eab3b0ae001b261b21cbfb551350af2a6c0a1e0540` |
| `supervise.py` | `76c7291004d181150a2d1a0ea5d287fffec3cb41f07645ea5bd1409ef241ace5` |
| `configs/0022-source-model-assets.json` | `30d4474b3372c8cea8aed86f67620425146303c0fae52b652b37062936691944` |
| `configs/0022-wheel-assets.json` | `fc8bea8344e061857dbf025f80915f8d757a18a2f960a92d490349cbb12d0f14` |
| `protocols/0022-traceable-frontend-assets.md` | `68f916d10232a339d00a4e69789b186653cd2070150fcd2c77e8689342793d22` |

Evidence: [acquisition receipt](../results/0022-preparation/acquisition-tests.json),
[test log](../results/0022-preparation/acquisition-tests.log),
[dependency metadata](../results/0022-preparation/dependency-wheels.json), and
[protocol](../protocols/0022-traceable-frontend-assets.md). Runtime compatibility,
checkpoint contents, actual memory and timing remain unmeasured.
