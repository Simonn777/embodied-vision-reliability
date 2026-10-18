# Prospective 0010: baseline and data qualification

2026-10-18. Source/filename-metadata evidence only. No image, embedding, weight, package,
encoder execution or accuracy result. The self-inclusion hypothesis remains
[unproven](0010-self-inclusion-candidate.md); no manuscript is justified.

## Strong baseline, now pinned

StatA author revision
[`2635f9355f9389756064becb5d9ed54c2756809c`](https://github.com/MaxZanella/StatA/commit/2635f9355f9389756064becb5d9ed54c2756809c)
binds the inspected implementation. The complete
[solver](https://github.com/MaxZanella/StatA/blob/2635f9355f9389756064becb5d9ed54c2756809c/solvers/StatA.py)
SHA256 is `9fcd2eb8a22cabc2c7eb43823625cb04557476c637a24c8a12eb3115ede2b962`.
Thirteen source files were checked against Git blob identities and byte lengths;
fifteen direct metadata/source responses returned 132,829 body bytes.
This source is AGPL-3.0, not MIT. It was read, not executed during this audit.

The default finite schedule has 11 blocks of five responsibility updates and ten
moment updates; there is no convergence test. Initial logits are 100 times cosine,
with unit-normalized images and text. Hard-cardinality anchoring uses alpha1.
The graph uses signed-cosine top-three neighbors, discarding the first top-k
entry rather than masking sample IDs; tied/repeated features can retain a self
edge. Covariance/zero-mass boundaries are unguarded. These details cannot be
silently repaired and still called an unchanged baseline.

The official solver unconditionally calls CUDA; the full entrypoint also loads
CLIP even with cached-image-feature options and eagerly imports other baselines.
An isolated device-only CPU adaptation therefore needs explicit parity tests.
For a matched one-way readout, save responsibilities immediately before the
fifth update of the final block. Rescoring returned responsibilities instead
would add an extra update. A held-out query additionally needs a declared graph
extension; the author implementation has no such API. Neither portability nor
benefit is established by this source review.

## Data route: reject ambiguous cached embeddings

The [pinned Lance provider card](https://huggingface.co/datasets/lance-format/eurosat-lance/blob/0a38598495784cb5bd9fc70f5ac89add4f6fd9b8/README.md)
describes OpenCLIP ViT-B-32 embeddings but lacks an exact checkpoint/tag/hash,
preprocessing/conversion record and original filename/hash mapping. Architecture
alone does not identify a joint image/text space. Equal split counts do not
establish a spatial-filename join. These embeddings are not admitted.

A possible alternative is our own matched extraction using official CLIP
ViT-B/32 and the RGB EuroSAT archive at
[TorchGeo revision f9fa29f](https://huggingface.co/datasets/torchgeo/eurosat/tree/f9fa29ffb3980d9959c3d177dd2bc7c71ffeddcc).
The provider advertises 94,280,567 bytes and SHA256
`8ebea626349354c5328b142b96d0430e647051f26efc2dc974c843f25ecf70bd`.
The pinned spatial train/validation/test filename bodies now match their Git
blob identities and SHA256:16,200 /5,400 /5,400 unique names,27,000 in their
union and zero pairwise overlap. Archive membership and pixels remain unchecked.
Longitude partitioning is not proof of independent
physical scenes, buffers, or exclusion from CLIP pretraining. Original EuroSAT
and relevant Copernicus attribution must be retained; no image rehosting is planned.

## Matched encoder, not a guessed text anchor

[OpenAI CLIP revision d05afc4](https://github.com/openai/CLIP/tree/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6)
is the prospective encoder. Five minimal source/license files match their fixed
Git tree, sizes and SHA256. The BPE vocabulary is only tree-qualified so far:
1,356,917-byte blob `7b5088a527f720063f044eb928eee315f63b2fc0`; body/SHA256 unverified.
The official ViT-B/32 checkpoint advertises 353,976,522 bytes and SHA256
`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`.
HEAD size and provider hashes are not a downloaded-file verification.

Ordinary `import clip` immediately opens the local BPE vocabulary. Its tokenizer
requires ftfy and regex; actual dependency compatibility remains to be checked.
`jit=False` still attempts TorchScript deserialization first. The proposed loader
will accept only the verified local official archive, reconstruct the official
eager CPU model, and forbid automatic download and generic pickle fallback.
The source returns unnormalized encoder features; its forward uses learned
`exp(logit_scale)`, distinct from StatA's fixed-100 convention.

The first proposed operation is eight fixed spatial-train images and the ten
fixed author-template prompts, CPU only, four threads, 8 GiB, one attempt and
300 seconds including first load. This proposal has not yet passed dependency,
acquisition or runtime admission. No throughput or quality result is claimed.

The filename-only qualification used a prospectively saved selector:
sort train names by SHA256 of `0010-clip-capability-v1|` plus the basename.
The first eight are River_362.jpg, AnnualCrop_1607.jpg, Pasture_919.jpg,
Residential_1561.jpg, Industrial_1563.jpg, AnnualCrop_1012.jpg, Highway_1291.jpg,
River_2042.jpg. No balancing or replacement was applied. The selection JSON
SHA256 is `75e31cc9742f6def800019dfd66009abb5294a9eafa7b8ef2d5130501617375c`.
This qualification ran locally as metadata inspection, not cloud inference.

## Accounting scopes remain explicit

Initial direct source/metadata scouting measured 31,868 + 132,829 + 28,255 =
**192,952 B**. Subsequent filename qualification read **523,363 B** in six
physical GETs (three text bodies total522,430 B plus three redirect bodies933 B),
within its fixed1 MiB cap, with no retry; command exit0,0.993169167s. Browser
retrieval bytes remain unknown. No image/embedding/weight/package body was acquired.
The earlier 0009 unmetered metadata redirect remains a strict-accounting failure;
the historical all-HTTP total remains UNKNOWN. A prospective stage can separately
enforce the original 10 GiB data/weight budget and a new 512 MiB cap on all new
application-read response bodies, after classifying/reusing existing assets.
That does not retroactively certify the old HTTP total or erase its failure.
No additional paid service was used or proposed.
