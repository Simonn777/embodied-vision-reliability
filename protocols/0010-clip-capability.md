# 0010: one official CLIP CPU capability operation

Prospective protocol. Execution requires the separate published freeze receipt; no model has run.

Purpose: determine whether one pinned official CLIP ViT-B/32 can produce matched
image/text features on the existing free cloud CPU. This is not an accuracy
comparison or an improvement claim. A separate exploratory development protocol
must define the StatA comparison before its outcomes are inspected.

## Required before admission

1. 0009 publication is complete through PR 35 and exact main CI. Check that no prior 0010 acquisition/probe
   attempt or active model process exists; the archived read-only preflight is expected. Inventory named cache paths once; reuse exact verified
   files, preserve partial/mismatched files, and never replace them implicitly.
2. The saved cloud preflight imports Torch 2.10.0+cpu, NumPy 2.0.2, torchvision
   0.25.0+cpu, Pillow 13.0.0 and packaging 26.3. It lacks ftfy, regex, tqdm and
   wcwidth. Acquire only their four hash-pinned wheels and install offline into
   a fresh dedicated overlay; no dependency resolution, extras, replacement
   Torch/NumPy/CUDA, or changes to historical overlays. Check distribution
   versions and record module versions separately before the model operation.
3. Freeze the acquisition/runner/test sources, asset manifest, configuration,
   this protocol, and the unchanged reused Linux supervisor in a Git commit.
   Verify exact published bytes and passing relevant tests before acquisition.
4. Filename qualification is complete locally: 16,200/5,400/5,400 names, no
   intersections, and a fixed eight-name selection saved before any pixel/model
   access. Reuse the published exact lists and selection; no repeat metadata
   acquisition or pixel-based replacement. Filenames contain class names: this
   was not literally label-free, although no accuracy or image outcome was read.

## Immutable route

OpenAI CLIP source d05afc436d78f1c48dc0dbf8e5980a9d471f35f6. Required source
identities come from its verified Git tree; retain the MIT notice. BPE vocabulary
is the 1,356,917-byte blob 7b5088a527f720063f044eb928eee315f63b2fc0; its
SHA256 is not known before acquisition. Verify its pinned Git blob identity,
then record the actual SHA256. Do not substitute a tokenizer.

ViT-B/32 checkpoint: 353,976,522 bytes, SHA256
40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af.
TorchGeo EuroSAT revision f9fa29ffb3980d9959c3d177dd2bc7c71ffeddcc;
EuroSAT.zip: 94,280,567 bytes, SHA256
8ebea626349354c5328b142b96d0430e647051f26efc2dc974c843f25ecf70bd.
No other model or dataset is admitted. Metadata-advertised sizes/hashes are
prospective identities, not a statement that payloads have been verified.

Sort all valid spatial-train basenames by
`(SHA256(UTF8("0010-clip-capability-v1|" + basename)), basename)` and take eight.
Verify the three filename lists are unique and pairwise disjoint; each selected
name must map to exactly one safe regular JPEG member. Decode only those eight
images, require RGB-compatible 64x64 dimensions. The ZIP includes other split
bytes; their pixels remain unopened. Longitude partitioning does not prove
scene independence, geographic buffers, or absence from CLIP pretraining.

Use the single pinned StatA EuroSAT template `a centered satellite photo of {}.`
and its literal NEW_CNAMES values, with fixed key order AnnualCrop, Forest,
HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential,
River, SeaLake. Read literals from verified source, without executing the dataset
package. No prompt ensemble or result-driven prompt selection.

## Resource and acquisition accounting

No added fees, paid API or GPU. The original 10 GiB data/weight ceiling remains.
Keep the historical conservative scientific-asset bound 7,532,690,072 bytes and
known 163,286-byte directory increment distinct from UNKNOWN total HTTP traffic.
The 0009 redirect-accounting failure remains failed. Known direct preparation reads total 5,235,758 bytes: 192,952 source/metadata
scouting, 523,363 filename qualification, 4,252,063 first PyPI metadata reads
and 267,380 narrower release metadata. The all-history regex JSON reached its
4 MiB cap and remains a preserved failure; only the narrower release endpoint
was subsequently read. Browser-tool retrieval bytes are unmeasured. No missing
historical response is assigned an invented size.

Enforce a 512 MiB allowance for direct preparation response reads, conservatively
charging the 5,235,758 known bytes above. Allocate exactly 4,194,304 bytes to the
four-wheel acquisition and 527,440,850 bytes to remaining fixed external assets.
Their sum is the remaining 531,635,154 bytes. Unused wheel allowance is not
automatically reassigned. Each downloader counts every body it reads, including
redirects, errors and partials. This is not a bound on browser traffic, historical
all-HTTP traffic or network-wire bytes. The separate data/weight ceiling remains
valid: the two proposed payloads add 448,257,089 bytes to the historical bound,
well below 10 GiB. No scientific assets are charged as already downloaded.
No automatic retries, HEAD-to-GET fallbacks, cache overwrites or optional assets.
Use exact manifests, verified TLS/proxy defaults, at most four allowed HTTPS
redirects per object, each redirect/error body capped at 64 KiB. Every physical
request and actual body read is recorded; signed URLs/credentials are omitted.
Failed or incomplete attempts remain terminal under this version. A separate
reviewed correction may address infrastructure, never silently repeat science.

## Single capability operation

Linux supervisor: four available CPU affinity slots, four numerical/Torch
threads, 8 GiB address-space bound and actual cgroup limit checked; 300-second
wall deadline including imports/load/output. Record cgroup memory.current,
memory.stat and memory.events before launch: the 8 GiB ceiling is not a claim of
8 GiB currently free memory, and file cache differs from anonymous allocations.
Do not purge historical caches or alter the limit. Outer stage records wait status,
peak RSS and timeout. Exclusive operation marker and fresh output directory.

Verify all source/asset bytes before loading. Use only the hash-verified local
official TorchScript archive on CPU to obtain the state dictionary, reconstruct
the verified official architecture, and convert to float32/eval. No automatic
download, generic pickle fallback or scripted forward. Official native-resolution
bicubic resize/center crop/RGB/tensor/normalization; exact dependency versions.

In inference-only mode: eight image encoder calls, batch size one in fixed order,
and one text encoder call for the ten fixed prompts. No warm-up, augmentation,
optimizer, adaptation, accuracy calculation or hidden extra call. Record attempts
before calls, completions after calls, and retain failures. Save raw/normalized
image (8x512) and text (10x512) features, actual learned logit_scale, finite/norm
checks, shapes/dtypes/hashes, token IDs, selected archive member hashes and timings.
StatA's later fixed scale100 is a separate method convention, not assumed equal
to CLIP's learned scale. No forward timing estimate is asserted before execution.

Stop on unavailable enforcement/dependencies, any asset/source/split identity
failure, ambiguous member, nonfinite/zero output, wrong shape/call count, timeout
or nonzero exit. No image replacement or rerun. Passing only establishes this
eight-image capability and permits designing the next bounded quality screen.
The original inspected tests remain excluded and no manuscript follows this stage.

Preparation and probe have separate one-shot markers. An acquisition/install
failure prevents the probe from starting; preserve the partial files and logs.
Before launch, verify source/config/selection hashes against the published freeze.
Saved capability vectors may be reused in later software checks, but these eight
images are excluded from the future quality cohort. No classification score is
computed in this operation.
