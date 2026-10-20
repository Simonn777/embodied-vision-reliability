# RS RGB pose data: metadata-only feasibility screen

Historical metadata snapshot. Mission-specific timing exceptions and the current budget assessment are in [the later budget screen](0025-rs-dataset-budget.md) and [current admission](../0025-admission.md); in particular ETH-1/2/3 HDR timestamps do not qualify.

Status: one strong conditional new source, **not two verified independent qualifying datasets**. No image, depth, IMU observation, pose/GT value, archive member or model payload was downloaded or read. No allocation is made. Public source code, paper text and directory/HTTP metadata only. The approximately 2.43 GB remaining payload budget is not a reason to weaken the timing or GT requirement.

## 1. Best new candidate: GrandTour / Boxi (2026)

The [primary paper, arXiv:2602.18164v1](https://arxiv.org/html/2602.18164v1), §§3.1–3.2.3, 4.1–4.2, specifies three TierIV C1 **RGB rolling-shutter** cameras, 1920×1280, with hardware timestamps at the middle row's exposure midpoint. Intrinsic and camera/IMU extrinsic calibration is released. Its outdoor reference uses independent NovAtel CPT7/HG4930 GNSS/INS; the separate Holistic Fusion reference adds total-station measurements. Neither is visual SfM truth. GT is still an estimate, and the paper explicitly describes drift when external references disappear. The paper describes 49 missions; this is one acquisition source, not 49 independent datasets. The ZED2i branch has unreliable USB timestamps and no per-line timing, so it is unsuitable here.

**Exact timing source:** [`hdr_timestamp_adjuster.py` at 7c7b7884e06a2082ac11f7b8a2ef3b7d25de5062](https://github.com/leggedrobotics/grand_tour_box/blob/7c7b7884e06a2082ac11f7b8a2ef3b7d25de5062/box_utils/box_auto/python/box_auto/scripts/camera/hdr_timestamp_adjuster.py), lines 16–36 locally, sets `VMAX=1400`, `FPS=30`, hence line interval `1/(30*1400)` s = 23.8095238 µs. It applies a 40-line trigger delay, 1280 image rows and 11 ms exposure, and moves the timestamp to the middle-row exposure midpoint. These are vendor/settings constants used by the authors, not a newly performed timing calibration. Exposure duration is not frame readout duration.

**Producer contract:** [release config](https://github.com/leggedrobotics/grand_tour_box/blob/7c7b7884e06a2082ac11f7b8a2ef3b7d25de5062/box_utils/box_converter/huggingface-dataset-conversion/configs/grandtour_release.yaml) names `hdr_front` as Color, the `/boxi/hdr/front/image_raw/compressed` topic, and its camera-info source. It names `cpt7_ie_tc_odometry` as the tightly coupled Inertial Explorer pose, distinct from DLIO and ZED VIO. [The parser](https://github.com/leggedrobotics/grand_tour_box/blob/7c7b7884e06a2082ac11f7b8a2ef3b7d25de5062/box_utils/box_converter/huggingface-dataset-conversion/dataset_builder/message_parsing.py) preserves `header.stamp.to_sec()` and `header.seq`.

**Access/license:** the [official HF dataset card](https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset) at revision `daf91c5e2b6db564822a35cc58222d73f8e75bab` declares MIT; API reports `gated=false`, `private=false`. The [official access page](https://grand-tour.leggedrobotics.com/dataset) distinguishes registration-free HF from registered ROS bag access. No registration or paid service was used. Runtime GNSS postprocessing is not required: the author-derived reference is already published.

**Budget evidence, not a selected scene:** for the first lexicographic mission `2024-10-01-11-29-55` (ETH-1), the pinned public directory metadata reports:

| File | Bytes | Purpose |
|---|---:|---|
| `images/hdr_front.tar` | 1,051,596,800 | One RGB RS camera |
| `data/hdr_front.tar` | 2,191,360 | Frame timestamps / IDs |
| `data/cpt7_ie_tc_odometry.tar` | 15,738,880 | Independent GNSS/INS reference; keep sealed until evaluation |
| `data/tf.tar` | 30,720 | Transform data |
| All 66 metadata YAML files combined | 30,327 | Upper bound, not a recommendation to read all |
| **Illustrative total** | **1,069,588,087** | **1.070 GB decimal; within 2.43 GB** |

The image TAR has publisher LFS SHA256 `d397885cdbb50431df264bac6516ff2d4e90047db277109d8e257a94f9a38211`. A HEAD-only request returned 200, exact Content-Length 1,051,596,800 and `Accept-Ranges: bytes`. This supports a future bounded range implementation, but **no 206 response or TAR member index has been tested**. Current publication packages JPEGs in TARs, despite the paper's logical JPEG layout; there is no verified per-JPEG URL. A whole single-camera TAR already fits. Acquiring two missions is not independent-source validation.

**Unresolved before scientific use:** release config describes HDR as 10 Hz whereas sensor paper and timestamp code use 30 Hz. This might be downsampling or stale prose; it is not established. Do not infer line time from released frame spacing. The HF publication has not yet been traced to an execution receipt proving the exact timestamp-correction version. The named GNSS/INS output is public, but the Holistic Fusion output was not identified in the inspected mission directory. Future protocol must specify which reference, its validity/quality support, the camera-to-IMU transform convention and first/center row indexing without looking at outcome. This is a feasible resource lead, not final dataset admission.

## 2. Independent possible sources that do not yet satisfy the contract

| Source | Established primary evidence | Blocking or unresolved contract |
|---|---|---|
| **PennCOSYVIO, 2016** | [Official overview](https://daniilidis-group.github.io/penncosyvio/) and [paper](https://daniilidis-group.github.io/penncosyvio/docs/penncosyvio.pdf): 3 GoPro RGB RS plus Tango RGB RS; independent surveyed-AprilTag rig pose reference, approximately decimeter accuracy. Four traversals of one ~150 m indoor/outdoor route; A-S/A-F public training GT, B-S/B-F hidden test GT. [Camera calibration](https://daniilidis-group.github.io/penncosyvio/intrinsic_calib/) includes GoPro OCamCalib and Tango pinhole model; rig extrinsics supplied. | No per-row timing/readout calibration located in paper/pages. Explicit dataset license not found in inspected sources. [Download page](https://daniilidis-group.github.io/penncosyvio/download/) advertises GoPro TARs around 29.4 GB and Tango Bottom 9.2 GB; [author download script](https://github.com/daniilidis-group/penncosyvio/blob/master/tools/download.bash) requests whole per-sensor TARs. A bounded subset/range route has not been verified. Do not count it as usable now. |
| **Rimac Technology RT, ECMR 2025** | [Official page](https://unizgfer-lamor.github.io/rt-dataset/) describes 6 sequences, 3 laps on each of 2 tracks, RS stereo 1920×1208 at 30 Hz, independent SBG Ellipse-D dual-antenna RTK INS at 200 Hz. Public per-lap/per-camera ZIP share pages exist. | Timing is an author **assumption** of approximately 98% scanning duty, not a verified readout calibration. RGB encoding and camera–INS GT mapping were not established. [Repository](https://github.com/unizgfer-lamor/rt-dataset), tree `0ba538f5941ce3fca7a4eb5cc7c161fddc9eff17`, contains website files; README calibration section is `...`. ZIP sizes/ranges unverified. CC BY-SA 4.0 is explicitly under “Website License”; it cannot silently be extended to data. |
| **SenseTime / ISMAR 2019 RS-VI** | [Official competition page](https://ismar2019.ismar.net/newenweb/news/254.html): Xiaomi Mi8 images/IMU, VICON reference, calibrated/synchronized; training data announced public. | [Ctrl-VIO primary paper §IV-C](https://arxiv.org/html/2208.12008v1) explicitly says no reference line delay is given. Later method's estimate is not independent timing calibration. Current public archive access, RGB encoding, size and data license not verified. |
| **Kerl et al., ICCV 2015 RS RGB-D** | [Primary paper §4.2](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Kerl_Dense_Continuous-Time_Tracking_ICCV_2015_paper.pdf) describes six PrimeSense Carmine real sequences with mocap trajectory; separate synthetic ICL-NUIM extension. | Author's linked dataset URL `https://cvg.vision.in.tum.de/~kerl/kerl_etal_iccv2015_webpage/` currently returned 404. No verified current archive, row calibration release, size or data license. |

These are evidence gaps, not conclusions that access or use would be unlawful. No requests were sent to maintainers and no account gate was bypassed.

## 3. Avoid false second-source matches

- [TUM Rolling-Shutter / TUM-RSVI](https://cvg.cit.tum.de/data/datasets/rolling-shutter-dataset) is one dataset: 10 indoor sequences, **monochrome**, GS left + RS right, independent OptiTrack, camera/IMU calibration and manufacturer-derived approximate row timing. It is a strong RS geometry reference if grayscale is prospectively allowed, but not a second RGB capture. TUM-VI is global shutter.
- [WHU-RSVI primary paper](https://mee-jieling.github.io/journals/2020-IEEE-ACCESS.pdf) uses rendered RGB trajectories, not real independent capture. It explicitly specifies dataset CC BY-SA 3.0 and known row timing; useful for a synthetic falsifier only. Do not label it real data because a later paper does.
- [IndoorMCD primary paper](https://elib.dlr.de/194916/2/2022_IROS_Multicam_Dataset.release_copyright.pdf) offers RS RGB-D and Vicon, but no readout calibration was found in the inspected calibration/synchronization sections. [OpenLORIS](https://lifelong-robotic-vision.github.io/dataset/scene.html) offers D435i color and mocap/laser references, but frame synchronization does not establish line timing. Neither is counted.
- RACECAR IROS 2023 is a different project from Rimac RT and uses global-shutter RGB cameras. Fastec's reconstruction-based pseudo-GT is not an independent measured reference; this was screened separately by the parent team.

## Decision

Advance **GrandTour HDR** only to a narrowly scoped source/metadata admission check if desired. Keep PennCOSYVIO or Rimac as unresolved independent-source leads, not a promised second benchmark. Under the requested strict real RS RGB + usable row timing + independent 6DoF + bounded public acquisition contract, two datasets have **not** been established. There is no authorization here to acquire scientific payloads or relax that contract.

## Audit trail

Direct metadata/source GETs and the HEAD-only request are retained in the private source directory as `http-receipt*.json`, with body sizes and hashes; `http-accounting-addendum.json` accounts for the first capped TUM author-page read (250,001 bytes) without altering its original receipt. Total direct response bodies: **1,546,798 bytes**, including both intentionally capped metadata reads. Web-tool retrieval is provider-managed; exact transferred bytes are unavailable and are not falsely recorded as zero. All source/HTML snapshots remain in the private source directory; no third-party content is proposed for redistribution.
