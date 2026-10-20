# 0028 normal observation contract — source and metadata only

2026-10-20. **The existing 0023 cache already contains predicted MoGe2 surface normals for all 51 images. No fresh forward pass, checkpoint, image acquisition or model download is needed to expose that existing observation.** This establishes input feasibility for a possible known-baseline diagnostic, not normal accuracy, statistical independence, a new solver, or experimental admission. The ICCV-2019 paired-normal relative-pose precedent remains controlling.

Scope: read producer/source text, saved JSON metadata and manifests only. No NPZ opened, tensor values decoded, weights loaded, images/GT read, model called or fitting performed. Public primary README/normal documentation/training source were read through the browser; no payload downloaded to the workspace. Cache archive integrity was not freshly rehashed here; the finding concerns the producer and recorded cache inventory, whose original hashes are retained.

## What was actually saved

`work/0023-preparation/frontend-01/receipt.json` is completed and eligible. Its 51 image entries agree exactly with all 51 `image-NNN.json` cache declarations. Every entry declares:

- `normal`: float32 `[512,384,3]`;
- `depth`: float32 `[512,384]`, `mask`: bool `[512,384]`;
- `points`, predicted normalized `intrinsics`, authored `K_original/K_resized`, SuperPoint `keypoints/xy/descriptors`, and depth samples/validity.

The maps reside in `image-000.npz` through `image-050.npz`. Normals are full image maps, not previously sampled at keypoints. The 120 completed edge caches preserve `source/target` image identities, canonical `match_ids/match_scores`, and `xy0/xy1`, but **do not contain per-match surface normals** (`depth*_normalized` is a depth field, not a normal). A later authorized adapter could sample the original per-image normal and mask maps at these frozen locations; that would be new array processing, not a new model observation. No sampling rule has been tested or admitted by this note.

Producer `experiments/0023-mapfree/frontend.py` lines 376–379 converts **every** `moge.infer` return tensor to NumPy; lines 425–441 merge those outputs into the immutable per-image cache. Lines 221–222 validate normal shape and finiteness on the model mask. It does not differentiate the final depth to obtain these normals. It does not check nonzero normal magnitude at all pixels or establish normal correctness.

Input preprocessing is ordinary original-view RGB resizing to max edge 512, with half-up dimensions and half-pixel intrinsic rescaling. It rejects nonidentity EXIF orientation, applies no EXIF transpose, virtual rotation, crop or horizontal flip. Thus the saved normals and depths describe the same **unrotated resized input view**. All 51 recorded dimensions happen to be width 384, height 512; this is metadata, not a tensor-value finding.

## Pinned model and inference semantics

The existing asset is `Ruicheng/moge-2-vits-normal`, revision `26b477f41595707c5db6770294c0d1721e8ed4ed`, local name `moge2-small-model.pt`, **140,550,416 bytes**, SHA256 `79a16621928c2bf0ed04659218c55c01075e950507f40bb3332fb4c873d3e1dc`. The manifest's exact [official checkpoint URL](https://huggingface.co/Ruicheng/moge-2-vits-normal/resolve/26b477f41595707c5db6770294c0d1721e8ed4ed/model.pt) is a provenance reference; it was not fetched here.

The completed frontend used `installation-05` under `work/0022-frontend-preparation`, frontend qualification freeze `fa454585e67a8f0395ad5f67a03d5a28cc33a66a`, and 0023 producer freeze `acf502b831ad9e3485ab56549217c8a603d56bf7`. MoGe source pin is `925b8ed835a7a9cdb7578ba15c658a0afc969030`. Recorded settings: CPU float32, 1,200 tokens, `use_fp16=False`, **`apply_mask=False`**, `force_projection=True`; no supplied `fov_x`. Torch 2.8.0, NumPy 2.0.2, four Torch intra-op threads and one inter-op thread; the existing dependency/runtime and strict-load qualification suffice for these already cached outputs.

[MoGe v2 source](https://github.com/microsoft/MoGe/blob/925b8ed835a7a9cdb7578ba15c658a0afc969030/moge/model/v2.py), read locally in full, has a separate learned `normal_head` sharing the encoder/neck with geometry heads. It bilinearly resizes the decoded vectors with `align_corners=False`, then applies vector normalization. `infer` returns this normal directly. The subsequent focal/axial-shift recovery, point reprojection and metric scaling operate on points/depth, not on normals. Therefore normals are distinct learned outputs, but shared features and training prohibit assuming independent error. This does not claim every predicted vector is nonzero or correctly oriented.

## Coordinate, sign and validity contract

The pinned [official README](https://raw.githubusercontent.com/microsoft/MoGe/925b8ed835a7a9cdb7578ba15c658a0afc969030/README.md), lines 141–148, explicitly declares camera-frame normals in **OpenCV axes: x right, y down, z forward**. Maps have the input image dimensions; predicted intrinsics are normalized. Do not interpret the raw normals as world coordinates or multiply them by the normal visualization RGB conversion.

The [normal documentation](https://raw.githubusercontent.com/microsoft/MoGe/925b8ed835a7a9cdb7578ba15c658a0afc969030/docs/normal.md) states that the normal head was added after the original MoGe-2 submission, uses squared angular supervision, and derives targets from training depth/intrinsics. This is a software-feature contract, not an inference from the original paper.

The pinned [training loader](https://raw.githubusercontent.com/microsoft/MoGe/925b8ed835a7a9cdb7578ba15c658a0afc969030/moge/train/dataloader.py), lines 130–165, constructs depth-derived normals, spatially warps them and applies `normal @ R.T`; horizontal flips negate the x component. The locally installed utils3d pin `3fab839f0be9931dac7c8488eb0e1600c236e183`, `numpy/maps.py` lines 343–416, forms crosses in the order up×left, left×down, down×right, right×up. **Algebraic sign implication:** a frontoparallel positive-depth plane has expected normal `(0,0,-1)`, toward the camera. This is a source-derived orientation expectation, not a measured check of the checkpoint. The precise dependency revision used during original checkpoint training is not independently established here. The normal decoder does not enforce view-facing sign on each prediction; sign errors remain possible. The existing `vis.py` multiplies components by `[0.5,-0.5,-0.5]` and adds 0.5 only for color display; that transform is not part of the cached array.

The `infer` mask is `(sigmoid(mask_head)>0.5)` intersected with positive shifted point z. It is **one shared predicted geometry-validity mask**, not a normal-specific angular-confidence map, visibility certificate or correspondence-inlier label. Since 0023 used `apply_mask=False`, invalid locations were not zeroed in the normal map and depth/points were not replaced with infinity. A plausible-looking finite/nonzero vector alone is not admission evidence. A future sampler must explicitly check the recorded mask, finiteness and nonzero magnitude, define interpolation and normalization, and preserve unsupported observations in coverage accounting. No numerical pixel-validity counts or normal-distribution analysis was performed here.

## Feasibility consequence and binding evidence

Existing original-view normals are available at zero additional inference/download cost. Different virtual views, input sizes, model settings or checkpoint outputs would be new observations and normally require fresh inference; none is authorized here. The inherited 0023 set remains development data, not an independent held-out normal benchmark. Accuracy and usefulness would require a separately frozen known-baseline comparison, with complete frontend costs reported. Simply aligning paired normals and solving translation is already covered by [Cui et al., ICCV 2019, Polarimetric Relative Pose Estimation](https://openaccess.thecvf.com/content_ICCV_2019/html/Cui_Polarimetric_Relative_Pose_Estimation_ICCV_2019_paper.html); this note admits no novel-solver claim.

Read-byte hashes (JSON/source only):

| Artifact | SHA256 |
|---|---|
| 0023 frontend receipt | `4bc0e5b7b47a5c1e0266e645115f3b71f92c099042ec83106ccd7b12148f8259` |
| frozen frontend.py | `5076452d2e69645745eb68675a40bb47473ec4a4a28db87cf6f9652229614bf7` |
| qualification-05 receipt | `88dfc856c4824f20d4969acd1c56afdba3e3c26381a0ca6527283ba13018fd64` |
| installation-05 receipt | `0e51d4995c91e0b446daedf640c0d8549e9683b81906c76d7d438cccc0796b3e` |
| installed MoGe v2.py | `e736ed59fdb8ad89ec0c29ddfeeccd6a4246c387b3f423b922708366b69758fd` |
| installed utils3d numpy/maps.py | `06489ac97c737b6fb25308ca0f078f80a0eba1f3b1bc507c15ea9bef18d6d212` |
| source/model asset manifest | `30d4474b3372c8cea8aed86f67620425146303c0fae52b652b37062936691944` |

Example declared cache binding, not a fresh NPZ read: `image-000.npz` size 6,526,647 bytes, archive SHA256 `c34ea991d8f96e5c36a793252c01fa8d545e66dd6e7094f710c382bb1b157b0d`; normal raw-C-order SHA256 `9abf368c3c19f9b788029924dcdaa151e7a67bd5d9d3ab36afbd35943e27b05f`. The full frontend receipt binds the corresponding identities for all 51 caches.
