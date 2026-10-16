# Candidate data and model provenance

Checked October 16, 2026. **No external dataset or model weight has been downloaded or executed.** This is a source/size audit, not a completed evaluation. The first experiment uses original generated scenes.

## TAP-Vid / TAPIR family

Official repository checked at commit [`730cda1c730877cfedbe01bf87fb1cadb78a565d`](https://github.com/google-deepmind/tapnet/tree/730cda1c730877cfedbe01bf87fb1cadb78a565d). Sources: [dataset instructions](https://github.com/google-deepmind/tapnet/blob/730cda1c730877cfedbe01bf87fb1cadb78a565d/tapnet/tapvid/README.md), [license and coordinate notes](https://github.com/google-deepmind/tapnet/blob/730cda1c730877cfedbe01bf87fb1cadb78a565d/README.md).

This benchmark evaluates tracking of an initialized image point and its visibility over a video. It is relevant to the planned move from synthetic target centers to real image points, but those two targets are not interchangeable.

| Candidate archive | Bytes reported by HTTP HEAD | Use decision |
|---|---:|---|
| [TAP-Vid RGB-Stacking](https://storage.googleapis.com/dm-tapnet/tapvid_rgb_stacking.zip) | 187,291,581 | Smallest practical external diagnostic: 50 synthetic robotics videos. Still synthetic; not evidence of real-video transfer. |
| [TAP-Vid DAVIS](https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip) | 1,668,710,491 | Candidate real-video validation set of 30 videos. Review original per-video terms and define an untouched evaluation subset before acquisition/use. |
| [RoboTAP](https://storage.googleapis.com/dm-tapnet/robotap/robotap.zip) | 13,558,087,507 | Exceeds this project's current 10 GiB total download/storage budget before extraction; do not download the full archive by default. |

All three returned HTTP 200 in a header-only check. Sizes are compressed archive sizes, not extracted memory requirements. These values can change; verify again before retrieval. HTTP ETags are not a substitute for a recorded local SHA-256 and provenance.

The upstream README identifies the non-TAPVid-3D software and linked checkpoints as Apache 2.0. It identifies TAP-Vid annotations, RGB-Stacking video, and RoboTAP video as CC BY 4.0; DAVIS original videos retain creator-specific terms, and Kinetics videos retain their individual terms. TAPVid-3D has a separate license and is outside this audit. A license on annotations does not automatically cover source video.

## Before the next experiment

1. Pin the exact reader/model revision, preserve notices if code is reused, and record the weight URL, license, size, and SHA-256. No checkpoint has been selected yet and MPS compatibility is untested.
2. Treat serialized dataset files as executable-risk inputs: verify the upstream download and inspect the loader before deserialization. Keep downloads and extraction out of Git; do not commit video or pickle archives.
3. Match point and coordinate conventions explicitly. The source documentation distinguishes normalized image coordinates from pixel coordinates and orders query coordinates differently from point coordinates. Add a known-coordinate test before evaluating a model.
4. Prefer the first-visible-point query mode for a causal online study. Offline/bidirectional trackers can use future frames and cannot serve as a causal method without being identified as a separate offline control.
5. Separate development/calibration/final evaluation by full video. If repurposing part of a standard evaluation set for calibration, label the result a custom research split and do not compare it to an official full-benchmark leaderboard score.
6. Preserve the original benchmark's visibility and localization metrics alongside any new selective-risk endpoint. The source warns that predictions for one queried point must not exploit other ground-truth queries; keep that constraint in the adapter and tests.

Next decision: assess whether the small RGB-Stacking archive supports a meaningful failure-detection pilot with a transparent local estimator, then audit a compact causal point-tracker for real-video work. Do not infer learned-model or real-robot performance from the current hand-specified diagnostic.
