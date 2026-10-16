# Dataset/checkpoint exposure and shared capture scope

2026-10-16. Primary-source review, not new model evaluation. Only the cited
sections/official code were checked; no whole-paper reproduction is claimed.

- [TAP-Vid, arXiv 2211.03726v2, section 3.1](https://arxiv.org/html/2211.03726v2):
  RGB-Stacking comprises 50 teleoperated simulated episodes using the same
  triplet-4 object set and front-left camera. Thirty first-frame points include
  20 moving-object and 10 static/background points. Our first two visible indices
  are fixed annotated points, not a representative random sample. Shared assets
  and viewpoint prevent claims of new-object/new-view/new-scene generalization.
- [BootsTAP, arXiv 2402.00847v2, sections 3–5](https://arxiv.org/html/2402.00847v2):
  supervised synthetic training and real-video self-training precede reported
  zero-shot benchmark evaluation; the released model has additional changes.
  Section 3 uses a six-pixel target for the localization-uncertainty head. Our
  strict >4px endpoint stays fixed; native scores are not calibrated probabilities
  of that endpoint. No six-pixel secondary endpoint is added. A paper's benchmark
  description does not certify complete non-exposure for the later causal file.
- [TAPIR, arXiv 2306.08637v2, sections 4 and 6](https://arxiv.org/abs/2306.08637v2):
  previously reviewed methods record Kubric training, DAVIS-informed selection
  and released-model modifications; see the [earlier reading note](literature/2026-10-16-confidence-methods.md).
  This does not establish the selected checkpoint's full selection history.
- [Pinned dataset schema/reader](https://github.com/google-deepmind/tapnet/blob/730cda1c730877cfedbe01bf87fb1cadb78a565d/tapnet/tapvid/README.md)
  gives a list of video/points/occluded arrays without video names or scene/episode
  identifiers. [Pinned bootstrap configuration](https://github.com/google-deepmind/tapnet/blob/730cda1c730877cfedbe01bf87fb1cadb78a565d/configs/tapir_bootstrap_config.py)
  includes eval_robotics_points; configuration is not proof that a particular
  run executed it or selected a checkpoint using its outcomes. Checked source
  SHA-256: `66cf230ee20067d23651c06284fc5bb32334ba31ef944c7abd61b0f205c6142f`.
- [Pinned root README](https://github.com/google-deepmind/tapnet/blob/730cda1c730877cfedbe01bf87fb1cadb78a565d/README.md)
  identifies Online BootsTAPIR's released file, Apache-2.0 code/weights and CC BY
  4.0 RGB-Stacking video/annotations. Those rights remain separate. This repository
  publishes original scripts and derived hashes/numbers, not video or checkpoints.

The frozen protocol defines an exploratory comparison on held-out published
videos from a shared object set and camera viewpoint. Exact duplicate groups
remain within one partition. Scene independence, near-duplicate independence and
complete checkpoint training/model-selection non-exposure are not established.
Evaluation videos are untouched by this project's inference and score tuning at
freeze time; they are not claimed to be globally unseen benchmark data.
