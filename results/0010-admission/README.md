# 0010 preparation evidence

These are source, filename, dependency and cache qualifications. **No image
encoder, learned checkpoint, classification comparison or improvement has run.**
They support a prospective bounded CPU operation, not a paper claim.

## Existing cloud CPU

The original workspace's saved [preflight record](cloud-preflight/record.json)
reports six successful imports: Torch 2.10.0+cpu, NumPy 2.0.2, torchvision
0.25.0+cpu, Pillow 13.0.0, packaging 26.3 and SciPy 1.17.0. Four imports fail
with `ModuleNotFoundError`: ftfy, regex, tqdm and wcwidth. The parent directly
retrieved the [actual command returns](preflight-parent-observation.json).

The observed outer command exits 0 in 4.756 seconds at the tool boundary;
the saved wait4 boundary is 4.732260914 seconds, 291,812 KiB peak RSS. These
boundaries differ. Each missing-package import exits 1; outer exit 0 means the
inventory completed, not that all dependencies exist. CPU affinity is [0,1,2,3],
numerical threads 4 and AS/cgroup limits 8 GiB. Torch's default inter-op value
was 9; the actual model runner must explicitly set and verify its planned value.

A bounded search visited 149 entries in four named cache roots, depth at most
four. It found no matching CLIP checkpoint, EuroSAT ZIP or BPE/source file.
Pruned package trees and other workspace locations were not searched. This
does not establish universal absence. The operation directory did not exist
before its exclusive claim. No model, data decode, package install or download
occurred. Original markers and historical caches were retained.

All 30 original preflight files (37,391 bytes) were transported without rerunning
imports. A plain JSON export exceeded the parent tool's 20,000-character item
limit; subsequent gzip chunks were reassembled and verified. The 16,715-byte
gzip hash is `b96fc21a6ac643ba41e15a542e7af1d898ce55f4256ac55238b3239d236a87fa`;
the 53,660-byte JSON hash is
`9ddc2e2a208f9d849491edcb42fdf7db9ccfa15d4ec0777c57398f49429189c4`.
Every exported file's size/hash matched. Original record SHA256:
`45b12bba336b453d338b1cafd3007a121b902457b6f0054a672284e93f4a3b5d`.
These exports were serialization only, not repeated preflight executions.

## Filename and source metadata

The [independent retained-file review](metadata/independent-review.md) passes
156 checks. Three pinned EuroSAT spatial lists contain 16,200/5,400/5,400 names,
27,000 unique names and no intersections. The fixed eight training names match
the pre-access SHA256-order selector. Acquisition read 523,363 bytes in six
physical GETs: 522,430 final text bytes plus 933 redirect bytes.

**Correction to the retained original plan:** filenames contain class names,
including validation/test filenames. Class-bearing metadata was read. No pixels,
image predictions, accuracy or outcome-driven selection were examined. Filename
disjointness does not establish independent physical scenes or pretraining
non-exposure. Original plan bytes remain unchanged.

Five CLIP source/license bodies match the pinned Git tree and SHA256 records.
The BPE and model bodies remain unacquired. The local command's home path is
replaced in a [clearly marked derivative](metadata/command.redacted.json), with
[original/public byte lengths and hashes](metadata/redaction.json). All other
copies are bound by the [copy manifest](metadata/copy-manifest.json). Planned
source identity matches the retained utility; the execution receipt itself does
not attest a loaded-source hash at runtime. This qualification is not described
as stronger evidence than was captured.

## Missing package metadata

The first PyPI metadata attempt read 4,252,063 bytes and failed when the complete
regex release-history endpoint reached its fixed 4 MiB body cap. Its partial
body is preserved locally; [failed ledger](pypi-metadata/all-history-failed-ledger.json)
records its exact digest and count. The complete `info` object in that retained
prefix identifies regex 2026.9.29. A separately planned narrower release-only
operation completed three requests / 267,380 bytes without retrying the old URL.
No wheel body was acquired in either operation.

The prospective wheels are ftfy 6.3.1, regex 2026.9.29 (CPython 3.12 manylinux
x86_64), tqdm 4.70.1 and wcwidth 0.9.2; total advertised size 1,234,470 bytes.
Exact URLs, hashes, requirements and yanked flags are in the
[ftfy selection](pypi-metadata/ftfy-selected.json) and
[other three selections](pypi-metadata/release-only-selected.json).
Linux without extras needs only these four additions. Package metadata is
sourced from [PyPI](https://pypi.org/), not evidence of installed compatibility.
The wcwidth release's license field is null, while its published classifier
identifies MIT; retain the wheel's own license when installing. No package code
is republished here.

Known direct preparation reads sum to 5,235,758 bytes. Browser retrieval bytes
and historical all-HTTP totals remain unknown. The prospective downloader cap
applies to its direct body reads, not those unknown totals. The original 0009
accounting failure remains failed; no added paid service was used.

The fixed EuroSAT URL's [single HEAD](hf-route/record.json) returned HTTP 302
to HTTPS `us.aws.cdn.hf.co`, with the exact repository commit, linked size and
linked SHA256. Zero body bytes were read; its Content-Length describes the
corresponding response representation and is not counted as a HEAD body. The
signed Location query is not published. Only `huggingface.co` and that observed
CDN host may serve the future ZIP; any new host stops acquisition. This qualifies
a route, not payload integrity.

Software integration: 54 artificial tests passed; a separate CLI/capability
review passed 20 checks. CI includes the dedicated suite because the existing
pytest configuration only collects tests/. The unchanged upstream CLIP source
contains one trailing-whitespace line and its LICENSE has a final blank line;
those two Git whitespace warnings are retained to preserve exact upstream bytes.
The staged owned-file whitespace check passes with that vendor directory excluded.
