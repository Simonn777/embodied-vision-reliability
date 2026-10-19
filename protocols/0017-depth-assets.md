# 0017: prospective cached-depth acquisition and metadata allocation

2026-10-19. This plan is frozen before new asset acquisition and numerical
inspection. It admits no accuracy claim and no manuscript. The completed0016
restart-bank diagnostic failed its2pp headroom gate; a distinct information
source must first be examined. The source-reviewed resource and consumer
contracts are in `docs/literature/0016-monodepth-resources.md` and
`docs/literature/0016-monodepth-consumer-contract.md`.

Acquire exactly the five author-hosted outdoor RoMa/MoGe caches below, from
`https://vision.maths.lth.se/viktor/posebench/relative/monodepth/`.
No images, weights, ScanNet files, alternative matchers or alternative depths.

| Filename | Bytes |
|---|---:|
| florence_cathedral_roma_moge.h5 |10675000|
| lincoln_memorial_roma_moge.h5 |10678136|
| london_bridge_roma_moge.h5 |10486136|
| milan_cathedral_roma_moge.h5 |10636784|
| sagrada_familia_roma_moge.h5 |10636488|

Total53,112,544bytes. Previous conservative remaining asset allowance is
2,825,233,268bytes. One GET per file, no retry or alternate source; stream opaque
bytes with expected content-length and bounded body, SHA256 each complete file.
Stop on mismatch/transfer failure and preserve receipts/partial bytes. Private
Each file's request is bounded by60seconds and metadata inspection by30seconds.
arrays remain outside Git. The upstream cache exporter/checkpoint identities and
specific derived-cache redistribution license are not established; no raw data
will be republished. No additional approval requirement was found for these five
public outdoor author-hosted caches. Their use does not verify front-end training
provenance or independence from other landmark photograph collections.

After successful byte acquisition, read HDF5 names, link types, shapes and dtypes
only. Do not read dataset values or attributes. Require at least16 pair groups
per scene. Rank keys by SHA256 of UTF-8
`evr0017:development:2026-10-19\0<scene>\0<pair-key>` where each `\0` is one NUL
byte, then lexicographic key for a tie. Select the first16 in each fixed scene
order shown above, concatenate to80pairs, and publish a manifest containing all
source/file hashes and metadata. All five physical scenes, including unused
pairs, become permanently development. Repeated images are not independent
replicates. No pair replacement based on schema/value/pose quality is allowed.

The acquisition/allocator source and this plan must be committed and published
before acquisition. A separate complete `0017-depth-information.md` protocol,
all prediction/evaluation sources, library pins and selected manifest must then
be frozen and pass artificial tests before any scientific array/native read.
Before that second freeze only manufactured arrays may exercise geometry code.
Ground truth R/t remain unread until every baseline output is terminal and sealed.

The intended next diagnostic compares four known native methods on all80pairs:
current normalized and tangent-Sampson E estimation, scale-only and affine-depth
RePoseD. No new-method novelty follows from those methods or their combination.
The review in `docs/literature/0017-depth-diagnostic-review.md` is a design input;
the separate executable scientific protocol will fix the actual choices.
