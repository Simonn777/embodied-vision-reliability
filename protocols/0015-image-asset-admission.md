# 0015: bounded development-image acquisition

This freezes acquisition only. It authorizes no estimator, photometric fit,
new quality score, held-out image or manuscript. Existing0013/0014 development
results remain exploratory; their frozen sources and records stay unchanged.

## Why these inputs

The next scientific question requires actual image evidence: covariance changes
alone cannot identify pose-aligned matching bias. Retrieve only the23 unique
images already named by the16-pair0012 development manifest, from the official
DMS-paired PoseBench image ZIP. Its pinned download script and demo load stored
x1/x2 against these images directly. Do not apply the separate DKM benchmark's
max-dimension1200 resizing rule to this cache. Header dimensions and CRC/SHA
checks establish acquired identity, not physical correctness of the poses.

Archive URL,623957998-byte length and exact selected central-directory records
are frozen in `configs/0015/posebench-images-archive.json`. Selected compressed
payload22491600 bytes; JPEG file bytes22559656. Metadata comparison found the
same filenames, uncompressed sizes and CRCs in the ETH copy, but acquisition
uses the directly paired archive. CRC agreement is not a cryptographic proof.

## Acquisition and preservation

Use the separately frozen stdlib range extractor. Require HTTP206, exact
Content-Range and bounded Content-Length/body reads; reject a full-archive
response. Verify local ZIP signature, flags, compression, names, sizes and CRC
against the frozen inventory. Accept only the23 manifest-derived names. No
unsafe paths, symlink output, encrypted members or unexpected descriptors.
Require complete deflate consumption, bounded decompression, expected file
length and CRC; record per-image SHA256 and image dimensions. Keep originals
without resizing, color transforms or metadata rewriting. Do not rehost images.

The run uses a fresh private `work/assets/0015-development-images` directory and
new public `results/0015-admission/acquisition.json` receipt. Use exclusive
creation; an incomplete transfer remains visible. Per-request timeout30seconds,
aggregate body-read cap24MiB, aggregate extracted-file cap32MiB. Do not hide
retries or skip a failed member. A failure requires a disclosed technical
resolution; already downloaded bytes and failed records are not overwritten.
No network call in artificial tests; fake-response tests cover range refusal,
header mismatch, bounds, corruption, unsafe paths and fresh-output protection.

Existing numeric H5 coordinates/GT are not needed for this acquisition. No
H5 numeric reads, weights, package installation or paid service occurs. Decode
only enough to validate JPEG identity and dimensions; image-based experiments
need a separate prospective protocol and method review. All selected images
belong to scene0015, already permanently development. Other scenes stay unread.

## Accounting and rights

Carry the previous conservative asset balance2901471736bytes. Reserve24MiB
for this acquisition and1MiB for newly inspected archive metadata (721577bytes
observed before freeze). Record actual bodies, including failures, without
calling source-page/header accounting an exact total-network measurement.
Remain below the existing10GiB data/weight budget. This acquisition is covered
by the owner's ongoing research authorization and creates no fee.

Use the official publicly distributed research subset under its applicable
terms. MegaDepth derived outputs are CC BY4.0; original photos retain their
respective licenses. Public code/receipts contain filenames, sizes/hashes and
source attribution only. No claim that all photographs are MIT or a proposal
to redistribute them. No external communication or additional approval is
introduced by this protocol.
