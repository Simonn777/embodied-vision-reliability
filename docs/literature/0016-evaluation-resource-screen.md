# 0016: evaluation resource admission screen

2026-10-19. **Metadata/source review only; no resource acquisition or evaluation
admitted.** No correspondence, calibration, ground-truth, confidence, or image
pixels were read. HTTP HEADs, an archive central directory, official split-name
text files, and public source/documents were inspected. No dataset asset was
saved. The stated remaining download allowance was **2,877,929,334 bytes**;
small metadata/document transfers must also remain in the parent ledger.

## Recommendation

**ScanNet1500 from the official DMS/PoseBench host is the strongest feasible
external capture domain.** Its DKM and RoMa coordinate caches plus all paired
images total **909,531,382 bytes**, leaving **1,968,397,952 bytes** before further
metadata overhead. Cached matching avoids model downloads and learned inference.
Processing one pair at a time is compatible with the memory budget in principle;
runtime and actual peak memory have not been measured here.

Prefer keeping **all 100 ScanNet physical scenes as a locked external test**, so
the canonical 1,500-pair benchmark remains test-only. Retain MegaDepth scene 0015
as development and 0022 as the existing downstream held-out scene. This provides
an indoor/outdoor capture change and 100 external scene units, but does not turn
one development landmark into adequate diverse development evidence. Before
learning or tuning a scene-dependent method, obtain an additional development
pool with verified provenance. If indoor development is essential, prospectively
allocate 20 physical scenes to development and 80 to final test using a fixed
hash of scene IDs, before reading any values; explicitly call that a custom
split, and never describe the full 1,500 pairs as untouched final evaluation.
This alternative is not allocated by this note.

Two admission conditions remain: establish the user's ScanNet permission under
the primary terms, and bind the exact cache/export provenance and pair-name
mapping. Public availability is not proof of permission or of training-set
independence. This source supports indoor two-view pose evaluation, not an
autonomous-driving deployment claim.

## Authoritative resources and verified sizes

The [DMS ScanNet download script](https://github.com/jastermark/DMS/blob/master/data/download_scannet.sh)
and [PoseBench download script](https://github.com/PoseLib/posebench/blob/main/download_data.sh)
both identify the same ScanNet1500 caches and the SuperGlue evaluation protocol.
The former also supplies paired images. All three HEAD requests returned 200
and `Accept-Ranges: bytes` on 2026-10-19; sizes are bytes, not rounded GiB.

| Official asset | Content-Length | Last-Modified | ETag |
|---|---:|---|---|
| [scannet1500_dkm.h5](https://vision.maths.lth.se/viktor/posebench/relative/scannet1500_dkm.h5) | 322,764,856 | 2024-04-15 15:34:56 GMT | `133d0038-616245d0c8800` |
| [scannet1500_roma.h5](https://vision.maths.lth.se/viktor/posebench/relative/scannet1500_roma.h5) | 322,764,856 | 2024-10-10 19:18:22 GMT | `133d0038-624243bdc9b80` |
| [scannet1500-images.zip](https://vision.maths.lth.se/viktor/posebench/relative/scannet1500-images.zip) | 264,001,670 | 2024-01-04 12:26:21 GMT | `fbc5886-60e1dd6cf1540` |

DKM alone costs 322,764,856 bytes; DKM plus images costs 586,766,526 bytes.
Coordinate-only methods need no images. Both matchers are useful robustness
checks but share images and ground truth, so they are not independent datasets.
ETags are server metadata, not cryptographic content hashes. Freeze actual
SHA-256 values after separately authorized acquisition. Avoid the unrestricted
`posebench` command: its [README](https://github.com/PoseLib/posebench) says the
full benchmark downloads its dataset on first use.

## Independent units: directly checked without labels

The [SuperGlue README](https://github.com/magicleap/SuperGluePretrainedNetwork#evaluation-on-scannet)
describes 1,500 test pairs from 100 scans, with intrinsics and relative pose in
its evaluation format. Its numerical pair/GT file was **not opened**. The
[ScanNet README](https://github.com/ScanNet/ScanNet) defines
`scene<spaceId>_<scanId>`: repeated scans with the same space ID are the same
physical location. It describes indoor RGB-D acquisition using an iPad/Structure
sensor, distinct from MegaDepth's Internet landmark photography. This establishes
different capture provenance; it does not prove universal statistical independence.

Two range requests read only the ZIP's 22-byte end record and 259,677-byte central
directory, with exact 206/Content-Range checks: **259,699 bytes total**, no member
payload. There are 2,596 JPEG entries, 100 distinct scan IDs, and 100 physical
space IDs, with 19–30 images per space. No repeated-space scan group occurs in
this archive. Total uncompressed image size is 267,603,213 bytes. Selective
image retrieval is technically possible from ZIP member offsets; it was not done.

Only the names in the official
[train](https://raw.githubusercontent.com/ScanNet/ScanNet/master/Tasks/Benchmark/scannetv2_train.txt),
[validation](https://raw.githubusercontent.com/ScanNet/ScanNet/master/Tasks/Benchmark/scannetv2_val.txt),
and [test](https://raw.githubusercontent.com/ScanNet/ScanNet/master/Tasks/Benchmark/scannetv2_test.txt)
lists were parsed. Train has 1,201 scans/565 spaces, validation 312/142, test
100/100. Test physical-space IDs have zero intersection with train or validation.
The ZIP and official test lists give **exactly the same physical-space set**:
SHA-256 of sorted space IDs joined by newline, without trailing newline,
`613d4d66bed4d3dfa43ecd867e302781d0d9af291a92f1614b8cf670e187212c`.
The exact downloaded test-list text SHA-256 is
`0214c6a3b1ee516ad653393b0321e7c0394c7662a4b3702eac1ddd7fbc00f7e0`.
The HDF5 pair-to-scene distribution and every pair's membership remain unverified;
check their names/shapes before any later numerical/GT read. Shared frames and
scan trajectories require scene-clustered analysis, not 1,500 independent trials.

## Calibration, metric, and exposure limits

The [DMS paper, arXiv:2506.02893v1, §4/§4.5](https://arxiv.org/html/2506.02893v1)
uses ScanNet1500, DKM/RoMa with 10,000 sampled matches, and a 2.5-pixel ScanNet
inlier threshold, versus 1 pixel for MegaDepth. Its
[loader at commit 7d5b228](https://github.com/jastermark/DMS/blob/7d5b228de9b6fab460f43e466fb0ed8565ffe137/src/dms/dataloader.py)
reads raw `x1/x2`, named images, camera dictionaries, and `R/t`; this is source
evidence of the expected interface, not validation of unread files. Check each
camera model and coordinate convention explicitly; do not inherit MegaDepth's
one-pixel threshold merely because the existing worker uses it.

[DMS metrics at that commit](https://github.com/jastermark/DMS/blob/7d5b228de9b6fab460f43e466fb0ed8565ffe137/src/dms/evaluation/metrics.py)
use directed translation angle and max(rotation, translation) error with
trapezoidal AUC@5/10/20. In contrast,
[SuperGlue `compute_pose_error`](https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/utils.py#L362-L382)
folds translation error to `min(angle, 180-angle)`. Predeclare and report both
with distinct names if comparing both literatures. Preserve failure coverage;
do not mix these conventions in a single table. Use physical-scene clustered
uncertainty and paired scene changes; matcher variants are repeated observations.

[DKM v3, §4.1](https://arxiv.org/html/2202.00667v3) describes MegaDepth training
and an indoor checkpoint additionally trained on ScanNet. Its
[training instructions](https://github.com/Parskatt/DKM/blob/main/docs/training.md)
provide separate indoor/outdoor scripts. [RoMa v2, §4.2](https://arxiv.org/html/2305.15404v2)
states that test scenes are excluded from its MegaDepth/ScanNet training and
uses the indoor checkpoint on ScanNet. Therefore this is new capture-domain
evaluation for a downstream method developed on MegaDepth, **not** an unseen
domain for the indoor matcher. Exact checkpoint/export hashes for the hosted
caches were not established here; paper-level split descriptions are not a
cache provenance certificate. RoMa's frozen DINOv2 pretraining also prevents a
blanket claim of no upstream image exposure.

RoMa §4.1 explicitly uses MegaDepth 0015/0022 for validation; §4.3 names the two
landmarks as St Peter's Basilica and Brandenburg Gate. Thus 0022 can remain
unseen by our downstream method without being unseen by matcher development.
Do not claim otherwise.

## Terms and alternatives

The full one-page [primary ScanNet Terms of Use](https://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf)
was read. It describes permission after approval, restricts use to noncommercial
research/education, and conditions colleague access on agreement to the terms.
The official README directs applicants to submit the agreement. No evidence of
this user's completed agreement/approval was examined. Code MIT licenses in
ScanNet/DMS do not replace the data terms. Do not put ScanNet images/cache arrays
in the public research repository; preserve provenance and aggregate results.

The [official PoseBench script](https://github.com/PoseLib/posebench/blob/main/download_data.sh)
also lists nine IMC2021 landmark caches and two fisheye scenes from Terekhov and
Larsson (ICCV 2023). Neither is admitted as the requested independent domain:

- **IMC:** useful potential additional outdoor development scenes, but a separate
  benchmark name does not exclude MegaDepth/Flickr/SfM image overlap. Establish
  authoritative landmark/scene-ID mapping and original image IDs or hashes
  before assigning any such scene to held-out evaluation. Classical SIFT, if
  its exact cache extractor is verified, avoids learned-extractor training
  exposure; it does not eliminate duplicate photographs. Exact IMC extractor
  provenance and licensing were not resolved in this screen.
- **Fisheye:** potentially different capture/lens conditions, but only two
  listed physical scenes and a different camera model. This does not supply
  many independent scenes, and the current pinhole pipeline is not automatically
  applicable. No numerical contents or model compatibility were inspected.
- **WxBS/absolute-pose caches:** DMS describes WxBS as lacking intrinsics and
  evaluates fundamental-matrix matching recall. PoseBench's 7Scenes/Cambridge/
  ETH3D absolute-pose caches solve a different task. Neither is a drop-in
  calibrated two-view test merely because it is small or familiar.

Read depth: relevant DMS/DKM/RoMa full-text sections and author source files were
read, along with official README/split text and the complete ScanNet terms.
HEAD/range findings above are direct metadata observations. No benchmark result,
expected gain, cache numerical integrity, or permission status is inferred from
those observations. This note admits a feasible resource route, not a research
mechanism or an experiment.
