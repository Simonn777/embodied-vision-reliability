# RGB-Stacking and Online BootsTAPIR: acquisition decision

2026-10-16. AI-assisted source/code and HTTP-header audit. **No external video, annotation archive, or checkpoint body was downloaded, deserialized, or evaluated.** This advances NEXT_ACTIONS item 3; it does not reproduce published tracking performance. The [metadata receipt](data-model-audit-2026-10-16.json) records inspected source hashes and observed object versions.

## Decision

Select **TAP-Vid RGB-Stacking + Online BootsTAPIR PyTorch** as a conditional, small external **synthetic** pilot. The source and licensing checks support preparation, but acquisition integrity, decoding, CPU resource use, adapter causality, and scene independence still need gates before a frozen evaluation. Do not describe this as real-video transfer or deployment on a physical robot. Do not silently substitute offline TAPIR or a newer checkpoint.

| Item | Pinned source / object | Evidence and limit |
|---|---|---|
| Code and reader | TAPNet commit `730cda1c730877cfedbe01bf87fb1cadb78a565d` | Seven source files retrieved as text and SHA-256 hashed; no upstream code executed |
| Dataset | `https://storage.googleapis.com/dm-tapnet/tapvid_rgb_stacking.zip` | HEAD 200; 187,291,581 bytes; generation `1676313001737086`; 50 synthetic robotics videos according to upstream documentation, not locally counted |
| Model | `https://storage.googleapis.com/dm-tapnet/bootstap/causal_bootstapir_checkpoint.pt` | HEAD 200; 218,887,028 bytes; generation `1732921823937966`; exact URL matches pinned notebook |
| Combined transfer | 406,178,609 bytes (about 387.36 MiB) | Fits 10 GiB download budget before dependencies/extraction; neither extracted size nor peak RAM is measured |

Object generations, lengths and server MD5 metadata are identification aids, **not local SHA-256 checksums**. Before loading, download once with a byte limit, verify object metadata has not changed, calculate full SHA-256, inspect ZIP paths/uncompressed sizes, and reserve storage for extraction plus model dependencies. Abort on mismatch or budget overrun. Keep all bodies outside Git. Model and dataset URLs are mutable despite pinned source code.

## Rights and redistribution

The [pinned root README](https://github.com/google-deepmind/tapnet/blob/730cda1c730877cfedbe01bf87fb1cadb78a565d/README.md#license-and-disclaimer) assigns Apache-2.0 to non-TAPVid-3D software and linked checkpoints, and CC BY 4.0 to TAP-Vid annotations and RGB-Stacking videos. These are separate grants. Preserve Apache license/notices and identify modifications if adapting code; give CC attribution, source, license link and change indication for reused data material. This repository will publish original adapter code, manifests and derived aggregate measurements rather than rehost the archive/weights. DAVIS creator-specific video rights and TAPVid-3D terms are outside this selection. This is a source-terms audit, not legal advice or a guarantee that every upstream provenance issue has been resolved.

## Annotation and partition implications

The [dataset documentation](https://github.com/google-deepmind/tapnet/blob/730cda1c730877cfedbe01bf87fb1cadb78a565d/tapnet/tapvid/README.md) describes a list of examples, without DAVIS-style video names. Fields are uint8 RGB video `[T,H,W,3]`, float32 points `[N,T,2]` in normalized `(x,y)`, and Boolean occlusion `[N,T]`. The [reader](https://github.com/google-deepmind/tapnet/blob/730cda1c730877cfedbe01bf87fb1cadb78a565d/tapnet/tapvid/evaluation_datasets.py#L535-L563) multiplies points by `(width,height)`, rescales frames to 256×256 by default, and normalizes pixels to `[-1,1]`. Do not multiply by `(width-1,height-1)` or swap axes silently. Query order is `(t,y,x)`; predicted tracks are `(x,y)`.

`sample_queries_first` initializes each point at its first visible frame, not necessarily frame 0, and drops tracks with no visible frame. `compute_tapvid_metrics` excludes the query and all preceding frames in first-query mode. Its position thresholds are strict `<` at 1/2/4/8/16 pixels; they do not automatically equal protocol 0001's normalized-error endpoint. Keep official AJ/occlusion metrics separate from the new risk/coverage endpoint. For selective localization, define visible/annotated opportunities, missing predictions and all-rejected risk explicitly; do not assume occluded coordinates are valid merely because an array contains them.

All queries from one video must share a partition. Archive list indices plus video-content hashes can identify examples, but **video disjointness does not establish scene disjointness**. Audit related simulator scenes/duplicates without using evaluation labels before assigning complete groups. A calibration subset drawn from this published evaluation collection creates a custom research split, not the official benchmark protocol or completely unseen upstream data. Freeze the full identity manifest before final inference. No split is assigned here.

The loader uses `pickle.load`: inspect the serialization and restrict/sandbox decoding before use. A checksum confirms bytes, not safe deserialization. Avoid executing an uninspected pickle in a credential-bearing connector or publisher process.

## Frozen model interface and causal boundary

The [pinned causal notebook](https://github.com/google-deepmind/tapnet/blob/730cda1c730877cfedbe01bf87fb1cadb78a565d/colabs/torch_causal_tapir_demo.ipynb) instantiates `TAPIR(pyramid_level=1, use_casual_conv=True)` (the public parameter is spelled `casual`), loads the selected checkpoint, calls `.eval()`, and uses CPU if CUDA is absent. CPU is an implemented path, **not a measured runtime guarantee**. MPS is not selected or validated by this notebook. The project cloud pilot should use CPU; no new GPU service is needed.

The notebook's frame-by-frame path passes only `frames[:, i:i+1]`, saved query features and recurrent context to `estimate_trajectories(..., query_points_in_video=None, get_causal_context=True)`. It chooses the last refinement output. In [nets.py](https://github.com/google-deepmind/tapnet/blob/730cda1c730877cfedbe01bf87fb1cadb78a565d/tapnet/torch/nets.py#L107-L186), causal temporal convolutions use left padding and past context. This supports a **zero additional video-frame lookahead design** after query initialization; it does not establish wall-clock latency or a passing numerical prefix test.

Do not copy the notebook's evaluation shortcut that extracts all query descriptors from the whole video before starting the loop. For queries arriving after frame 0, initialize each at its arrival frame, feed no future descriptors into earlier outputs, and mask pre-query outputs. Test changing future images and future queries independently; require already emitted predictions and scores to remain unchanged. Test one query alone versus batched queries because upstream evaluation forbids information from other ground-truth query locations influencing a queried point.

Native outputs are `tracks`, `occlusion` and `expected_dist`. The latter two are **logits**, not pixel distances or calibrated failure probabilities. Preserve raw final-refinement values. The demo collapses them to Boolean visibility via the product of their sigmoid complements; that loses ranking information. Both refined logits contain temporal context and cannot be labeled single-frame confidence. Initializer-only confidence refers to initializer coordinates; pairing it with refined coordinates requires an explicitly justified control. Compare visibility, localization confidence and their product separately, then temporal augmentation on exactly the same frozen coordinates. This audit does not resolve the strict current-image-only baseline for refined outputs.

Freeze weights (`requires_grad=False`), use inference mode, load on CPU with a weights-only loader if compatible, and check state-dict keys strictly. Do not automatically fall back to unrestricted deserialization after a safe-loader failure. The upstream package declares a broad JAX stack plus optional PyTorch/torchvision; do not install all training dependencies by default. Audit the minimal inference dependency closure and sizes first.

The logical causal-context tensor entries sum to 983,040 float32 bytes per query at one refinement resolution (12 blocks × two history steps × (512+2048) channels × four refinement iterations × four bytes). This is not a measured allocation or memory bound: the initial constructor repeats a shared dictionary, while subsequent refinement updates create new context. Parameters, activations, feature grids, temporary copies and allocator overhead also require memory. Use a few queries and one frame at a time for the first CPU profile; query chunk size 64 is not a bound on total recurrent-state memory. No CPU/MPS speed, full RAM footprint, checkpoint checksum or accuracy was measured in this audit.

## Required next step

Draft protocol 0002's acquisition/development stage: bounded download and serialization inspection, video/group manifest, coordinate checks, streaming adapter and prefix/query-independence checks, and a two-thread/6 GiB/15-minute CPU profile on development-only inputs. If it cannot fit, record that failure and choose a smaller valid baseline before freezing any final evaluation. Only after these gates should the complete split, practical effect threshold, matched-coverage comparison and sequence-level uncertainty plan be frozen and published for a later run. Existing final measurements and protocol 0001 are unchanged.
