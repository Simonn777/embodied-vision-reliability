# 0016: bounded monocular-depth development resources

2026-10-19; source and HTTP metadata only. No HDF5 body, numerical array,
calibration value, ground-truth value, image, or model was read/downloaded. No
development or final-test allocation is made here. The separate ScanNet ZIP
directory inspection previously transferred 259,699 bytes, to be included in
the parent's resource ledger.

**A feasible small route exists:** the official PoseBench cache provides five
Phototourism scenes with RoMa matches and MoGe predictions in **53,112,544 bytes
total**. All five scenes with both MoGe and UniDepth cost **106,500,224 bytes**.
This is a useful prospective development-only source for auditing depth/geometry
consistency, without images or neural inference. It is not yet sufficient to
claim measured nonlinear depth error: exact checkpoint/export provenance and
geometric reference construction remain to be resolved.

## Frozen source and exact files

PoseBench source was read at commit
`7b494256e89a13c47ab37b6917c2e0d3dae0c2e4`, specifically its
[dataset configuration/loader](https://github.com/PoseLib/posebench/blob/7b494256e89a13c47ab37b6917c2e0d3dae0c2e4/src/posebench/monodepth_relative_pose.py)
and [download script](https://github.com/PoseLib/posebench/blob/7b494256e89a13c47ab37b6917c2e0d3dae0c2e4/download_data.sh).
The six configured bases are listed below. Each has four files named exactly
`<base>_<matcher>_<depth>.h5`, where matcher is `roma` or `splg`, and depth is
`moge` or `unidepth`. The base URL is
[the author's monodepth directory](https://vision.maths.lth.se/viktor/posebench/relative/monodepth/).

All 24 HEADs returned HTTP 200, `Accept-Ranges: bytes`, and Last-Modified on
2025-11-20. These are exact Content-Length values, not filesystem estimates.

| Base / physical interpretation | roma_moge | roma_unidepth | splg_moge | splg_unidepth |
|---|---:|---:|---:|---:|
| florence_cathedral | 10,675,000 | 10,677,160 | 3,276,072 | 3,288,408 |
| lincoln_memorial | 10,678,136 | 10,678,472 | 2,678,984 | 2,679,032 |
| london_bridge | 10,486,136 | 10,677,848 | 1,819,928 | 1,923,040 |
| milan_cathedral | 10,636,784 | 10,677,008 | 2,418,744 | 2,543,808 |
| sagrada_familia | 10,636,488 | 10,677,192 | 3,341,856 | 3,467,264 |
| scannet: pooled dataset, **not one physical scene** | 53,377,472 | 53,377,472 | 11,396,316 | 11,396,316 |

All 24 files total **263,484,936 bytes**. Five outdoor scenes with SP+LG and both
depth methods total **27,437,136 bytes**. All outdoor combinations total
**133,937,360 bytes**. Avoid treating the 24 files as 24 independent scenes.

If admitted later, a single technical starting asset is
[florence_cathedral_roma_moge.h5](https://vision.maths.lth.se/viktor/posebench/relative/monodepth/florence_cathedral_roma_moge.h5)
(10,675,000 bytes; ETag `a2e338-6440797aede68`). The proposed substantive
development pool is all five outdoor RoMa/MoGe files, not a result-selected
subset. This consumes under 2% of the stated remaining 2,877,929,334-byte
allowance. No allocation is executed here. Hash each acquired file separately;
ETags are not content-hash certificates. The full PoseBench CLI automatically
downloads additional resources and should not be used for this narrow task.

## Interface and known controls

The loader constructs each pair from `x1`, `x2`, `K1`, `K2`, `depth1`, `depth2`,
`camera1`, `camera2`, `R`, and `t`. The
[estimator wrapper](https://github.com/PoseLib/posebench/blob/7b494256e89a13c47ab37b6917c2e0d3dae0c2e4/src/posebench/estimators.py#L224-L243)
passes point coordinates, per-point depths, and camera dictionaries directly to
`estimate_monodepth_relative_pose`. Its configured comparison includes ordinary
essential-matrix pose, RePoseD scale-only, and RePoseD scale-plus-shift; thresholds
are 2 pixels for epipolar error and 16 pixels for reprojection error, with
100–1,000 RANSAC iterations. These are existing controls, not proposed novelty.
The current API is not assumed compatible with our pinned older runtime; no
package installation or upgrade is authorized by this note.

No true per-point depth field is used by this loader. The source lists expected
names but does not establish the unread files' shapes, pair counts, camera
models, point order, image identifiers, or shared matches across model variants.
Different file byte lengths do not establish the cause of a difference, but
neither may equal-point coverage be assumed. Verify keys/IDs/shapes first under
a prospective read protocol, and explicitly compare correspondence identity
before interpreting differences between two depth models. Do not assume the
full paper's pair count applies to these small regression caches.

## Depth provenance: established and missing

[RePoseD, ICCV 2025; arXiv:2501.07742v3, §4](https://arxiv.org/html/2501.07742v3#S4)
identifies RoMa and SuperPoint+LightGlue matching, MoGe and UniDepth depth
predictions, and five Phototourism test scenes. Its model references distinguish
MoGe's scale/affine-invariant geometry from UniDepth's metric-depth method.
The author's [scene-name source](https://github.com/kocurvik/mdrp/blob/7955c0761a2fa7de30d18aa03dd888321bf2a0dc/utils/data.py#L1-L8)
lists the same five outdoor bases. The
[official repository](https://github.com/kocurvik/mdrp)
publishes extracted matches/depths and distinguishes original ICCV evaluation
branches from later improved joint local optimization.

However, no reviewed source binds the hosted PoseBench files to exact model
checkpoint hashes, inference resize/crop settings, point-depth interpolation,
invalid-depth filtering, or an exporter revision. `moge` is not proof of MoGe-1
or MoGe-2; `unidepth` does not identify V1 versus V2 or backbone. The current
[RePoseD demo](https://github.com/kocurvik/mdrp/blob/7955c0761a2fa7de30d18aa03dd888321bf2a0dc/make_pair.py#L19-L25)
loads `Ruicheng/moge-2-vitl-normal`, but is **not** evidence that this generated
these caches. Its separate
[Phototourism preparation script](https://github.com/kocurvik/mdrp/blob/7955c0761a2fa7de30d18aa03dd888321bf2a0dc/prepare_pt_test.py#L82-L95)
samples previously saved depth arrays; this also does not certify the cache
export path. Do not call raw fields metric range, inverse depth, or a specific
checkpoint output until their semantics are confirmed. PoseLib's model uses
depth along calibrated rays, so this distinction materially affects the audit.

## What the proposed diagnostic can identify

With calibrated rays `x,y` and a known relative pose, a later development
diagnostic can solve `z2*y = R*(z1*x) + t` for geometric depths. Normalizing
translation fixes a common positive baseline scale; ordinal comparisons within
an image and a free positive-scale calibration do not require metric units.
This is **triangulation from matched points and reference pose**, not measured
ground-truth depth. Low parallax, mismatches, intrinsics errors, and reference
pose uncertainty can generate apparent nonlinear depth residuals. ScanNet's
`scannet` cache likewise must not be assumed to contain sensor-depth labels.

A credible first question is whether a fixed monotone calibration improves
out-of-sample geometric depth agreement over scale-only and affine-depth
calibration **on the same independently fixed correspondences**, with parallax
and reprojection conditioning declared in advance and every excluded/invalid
case counted. Fit/evaluate splits must group images or connected scene units;
reused pixels across overlapping pairs are not independent observations.
This is a diagnostic of recoverable relation, not a deployable method if it
uses reference poses. A monotone map preserves depth ordering; if the dominant
error is rank reversals or correspondence mismatch, it cannot repair that error.
Using the evaluated points to both fit and judge a flexible map would merely
measure interpolation capacity. No such diagnostic ran here.

## Data terms and permissible claim scope

The [official IMC data page](https://www.cs.ubc.ca/research/image-matching-challenge/2021/data/)
describes Phototourism's Flickr/YFCC/reconstruction provenance. The
[updated UBC download page](https://www.cs.ubc.ca/~kmyi/imw2020/data.html)
states that the full Phototourism test set with ground truth became public in
2022. No account approval requirement for those public outdoor resources was
found in these pages. This is not a blanket redistribution license: the reviewed
PoseBench and RePoseD trees have no top-level LICENSE file, and no specific
derived-cache license was located. Original photograph terms, code terms, model
terms, and derived numeric caches must not be conflated. Keep acquired arrays
out of the public repository; document sources and publish code/aggregate
findings within their applicable terms.

[MoGe's repository](https://github.com/microsoft/MoGe#%EF%B8%8F-license)
describes MIT code with an Apache-2.0 DINOv2 component;
[UniDepth's repository](https://github.com/lpiccinelli-eth/UniDepth#license)
states CC BY-NC 4.0 for its software. Neither identifies the separate cache
license. For ScanNet, the specific approval/terms issue documented in
`0016-evaluation-resource-screen.md` remains; the outdoor option avoids relying
on an unverified ScanNet permission status.

These outdoor scenes are candidates for **our development use**, even though
the source paper calls them test scenes. They would then be unavailable as our
untouched final evidence. No MegaDepth image overlap audit or upstream model
training-exposure certificate was established. Scene-name differences alone
must not support an independence or unseen-data claim.

Read depth: complete relevant author loader/wrapper/configuration files,
RePoseD full-text §3/§4 and cited preparation/scene source, official data/model
license sections, and 24 HTTP HEADs. No inference was made from numerical cache
contents, and no real experiment or artifact acquisition ran.
