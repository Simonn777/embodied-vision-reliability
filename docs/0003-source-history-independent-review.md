# Independent review and frozen descriptive follow-up

October 17, 2026, America/New_York. AI-assisted review of the cached-source audit,
the one saved-R compatibility analysis and the prospective annotation-consistency
implementation. The numerical finding is **INCONCLUSIVE**, and the original
rotation failure remains unchanged. This review adds no new dataset access or
geometry execution.

## Findings supported by the saved evidence

The cloud inspected five cached source bodies plus their cached Git-tree listing,
39,425 bytes. Their recorded content hashes and tree blob identities were checked.
The static generator writes `intrinsics` and `extrinsics` and casts cameras to
float16; the example consumer casts to float32 in memory without resaving an
archive. Earlier wording must not be read as evidence that these camera keys need
renaming. Other field names, filenames, persisted dtype and sample-specific
conversion history remain unconnected. Missing cached bodies/history do not prove
that a converter does not exist upstream.

The separately prepublished single-proposal analysis ran in the cloud in 0.034895 s,
peak RSS 11,392 KiB, exit/raw 0/0, using only the already published R9 bytes. Its
fixed z-branch hypothetical rotation is exact SO(3), but only 8/9 binary16 rounding
cells match; [0,1] fails. The result is INCONCLUSIVE. This does not exclude every
possible precursor and cannot prove or disprove the actual float16 generation
history. There was no second proposal, optimization or matrix replacement.

A prewritten independent local output verifier used exact rational nearest-even
half rounding and Gaussian elimination, without importing the proposal program or
constructing another proposal. It reproduced the recorded result, including all
18 Gram entries, determinant +1 and the failed [0,1] cell. The saved-output check
ran in 0.075051459 s, returned 0 and produced empty stderr. This is a separate local
arithmetic review of saved JSON, not a second cloud analysis or a new data sample.
See the publication-review bundle for exact sources, inputs, outputs and execution
receipts; data and derived numeric outputs retain PointOdyssey CC BY-NC-SA 4.0.

## Pre-freeze defect and correction

The first complete candidate `ec26b8f` bound 44 inputs but omitted
`src/evr/__init__.py`, which Python executes before its submodules. An isolated
synthetic Git probe reproduced that an initializer edit could escape the gate.
The candidate was not frozen or run on real data. The correction `84dd6aa` adds
that unchanged initializer to the 45-file mandatory set and checks its actual
import path and source hash. Two new synthetic tests cover current/index/input
commit mismatches and imported-package identity. Earlier maps, tests and source
snapshots remain historical; the latest map is v3.

Source-bound cloud attempt3 passed 17 synthetic tests in 0.394826 s, peak RSS
16,128 KiB, exit/raw 0/0. Candidate CI 39089968641 independently passed 320 tests
and 4 subtests in 10.50 s, plus the existing 1,240-frame synthetic smoke in 4.61 s.
These are software checks. None constitutes a real PointOdyssey projection result.
The protected baseline of 1,544 files, 1,420 previous result files, five prior
freeze chains and 38 distinct historical inputs remains unchanged. Notes append.

## What the next frozen operation means

The versioned sparse reader, runner and measurement parent fix frame indices
[0,1,2] and previously allocated point IDs [0,1], at most six checks. They use only
the original stored RT/K and the unchanged 0.5 original-pixel criterion. Original
rigid-camera qualification, descriptive projection agreement, visible annotation
support and rigid-qualified support are separate fields. Even perfect agreement
cannot certify rigid geometry, independent GT, physical units or scene independence.

The plan has fixed sparse physical spans, countable gaps and retained values,
latest carried charge 22,910,156 bytes and prospective total 23,775,416 bytes.
It uses a fresh one-shot marker/output, old-marker checks, strict header/layout
checks, atomic interruption records, bounded resources and stop-on-failure rules.
No new camera coefficients, original point coordinates, projected coordinates or
depth are published. Additional raw dataset values stay outside the public repository. A hypothetical
half-cell matrix can never enter the projection or qualification path.

The original threshold and failure, source audit, single-proposal result, prior
snapshots and both earlier operation markers are preserved. No data/source
redownload, model inference, old-test tuning, paid service or permission/schedule
change was needed. The next operation remains unexecuted in this task.

## Evidence boundaries

The source-audit `execution_source=2826fac` is the Git baseline at its run, not a
claim that this baseline contained the new audit utility. Pre-start source hashes
and later published byte identity bind that utility. The saved-R analysis uses
committed execution source `565c1aa` with the same checker bytes as its prepublished
method `c1bcc59`. Parent receipts observe child exits, not recursive outer
supervisor exits. A local review cannot independently attest the entire cloud
network, billing system or unavailable source-generation history.

The local Git fixtures initially encountered macOS temporary-directory symlink
handling; the same tests passed using a canonical temporary path. This is a local
fixture/environment detail, not a cloud experiment failure. Source/provenance and
math reviewers inspected different evidence; their approvals are AI-assisted
checks, not an external human peer review or dataset validation.

## Final freeze and publication gate

Independent code/provenance review approved exact input commit
`84dd6aa71027ab3d5c18e3d3fdef5a59ccdeb035`. The separate receipt-only commit
`48ff18ff8e54be8a57bda5741a008b04582e3afc` publishes
[the canonical freeze](../protocols/0003-annotation-consistency-freeze-v1.json),
SHA256 `fe7f949bb227ff0824fdee71cdabd29129ac4ec09fe39d3858466c0bc937ee44`.
It binds all 45 required inputs; the receipt is absent from its own input commit.
The actual local `verify_published()` call then passed current/index/HEAD/input/
published blob checks, imported-module identity, all five prior chains and their
38 distinct inputs. This took 2.774887 s and returned 0, using no runner main,
cache, dataset or projection. The full actual command/source/output record is in
[the review bundle](../results/0003-source-publication-review).

The final parent review preserves all 1,544 original protected paths, all 1,469
previous-candidate result files, the 49 earlier review artifacts and the original
append-only note prefix. Its complete 63-artifact since-baseline map includes the
14 added initializer-fix result artifacts. Final publication and main CI are checked
separately in [PR19](https://github.com/Simonn777/embodied-vision-reliability/pull/19).
The next research step can verify this merged freeze and execute the bounded
descriptive operation once. This task stops at the reviewed freeze: no new
PointOdyssey geometry, matrix repair or old-test tuning has been performed.
