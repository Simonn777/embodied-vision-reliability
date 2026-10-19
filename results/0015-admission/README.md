# 0015 development images acquired, no image experiment yet

After published freezeb608ae81d04ace0f95dfe3212d3681b971828476, the one bounded
acquisition completed23/23 images in50.727468seconds, exit0. All69 requests
returned the required206 ranges, reading22493826 body bytes. Raw JPEG files
sum to22559656bytes. No retries, pixel decoding, HDF5 reads, weights or
scientific estimator calls occurred. Private image files are not redistributed.

The extractor checked ZIP metadata/CRC and JPEG dimensions. A separate local
byte verification recomputed every file's size, CRC and SHA256 without pixel
decoding; all agree with the receipt. This verifies acquisition integrity,
not coordinate accuracy or a research improvement. After the conservative
1MiB metadata allowance, the remaining asset budget is2877929334bytes.

[Receipt](acquisition.json), [actual process](acquisition.process.json),
[output log](acquisition.log), [independent byte checks](local-byte-verification.json),
[ledger](transfer-ledger.json), [frozen asset protocol](../../protocols/0015-image-asset-admission.md).

The separate photometric diagnostic is still being prepared and reviewed.
It will use the existing16 development pairs, explicit coordinate conventions,
identical-point controls and the full10000-match baseline. No accuracy outcome
is implied by these assets or passing artificial tests.
