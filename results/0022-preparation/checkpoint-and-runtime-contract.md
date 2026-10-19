# 0022: MoGe2-S checkpoint and runtime metadata contract

Metadata/source review only, 2026-10-19. No checkpoint/wheel download, model import, installation or inference was performed by this review. The ETH3D scene totals in published 0021 remain feasibility examples, not development/test allocations.

## Artifact identity and licensing

Use the author [Ruicheng/moge-2-vits-normal](https://huggingface.co/Ruicheng/moge-2-vits-normal/tree/26b477f41595707c5db6770294c0d1721e8ed4ed) checkpoint at revision `26b477f41595707c5db6770294c0d1721e8ed4ed`: `model.pt`, **140,550,416 B**, publisher LFS SHA256 `79a16621928c2bf0ed04659218c55c01075e950507f40bb3332fb4c873d3e1dc`. Validate size and SHA256 before loading. The official HF card metadata explicitly declares MIT and `facebook/dinov2-small` as base model. The metadata exposes no complete `model_config`; the model name and byte size cannot establish its exact architecture or head configuration.

Use pre-MoGe3 source [925b8ed835a7a9cdb7578ba15c658a0afc969030](https://github.com/microsoft/MoGe/tree/925b8ed835a7a9cdb7578ba15c658a0afc969030), package version 2.0.0. The code is MIT; included DINOv2 code has its separate Apache-2.0 license. Preserve notices for any permitted redistribution. The [pinned v2 training configuration](https://github.com/microsoft/MoGe/blob/925b8ed835a7a9cdb7578ba15c658a0afc969030/configs/train/v2.json) lists 24 datasets, including MegaDepth, ScanNet++, ARKitScenes and Waymo; it does not name ETH3D or Map-free. This is not a checkpoint-specific training manifest or an audit of DINOv2 pretraining. Do not claim benchmark-image exclusion or upstream training independence. Independent new inference is a different claim from unseen training data.

## Small, complete declared dependency additions

`dependency-wheels.json` contains exact PyPI URLs, filenames, byte counts, SHA256 values, full `Requires-Dist`, active requirements and Python constraints for **14 wheels totaling 76,809,693 B**:

| Packages | Frozen versions |
|---|---|
| SciPy, OpenCV, torchvision | 1.13.1, 4.11.0.86, 0.23.0 |
| Kornia, kornia-rs | 0.8.1, 0.1.9 |
| huggingface-hub, hf-xet | 0.34.4, 1.1.9 |
| PyYAML, tqdm | 6.0.2, 4.67.1 |
| requests, urllib3, certifi | 2.32.5, 2.5.0, 2025.10.5 |
| charset-normalizer, idna | 3.4.4, 3.11 |

All selected wheels match the actual Python 3.9/macOS ARM64 interpreter tags; the pure-Python charset-normalizer wheel was chosen explicitly. `torchvision==0.23.0` requires `torch==2.8.0`; SciPy accepts the installed `numpy==2.0.2`. Recursively active declared requirements are satisfied by these additions plus 12 recorded installed distributions, including Torch 2.8.0, NumPy 2.0.2, Pillow 11.3.0 and packaging 26.3. Their local distribution-metadata hashes are in the manifest. No upgrades to Torch or NumPy are needed. This verifies packaging compatibility, not successful native import or numerical correctness.

Hugging Face Hub is imported even with a local checkpoint; its base macOS dependency on hf-xet is therefore represented rather than silently omitted. Installing no extras avoids CLI/UI/training dependencies. The source import-closure audit is separate: preserve `utils3d` at `3fab839f0be9931dac7c8488eb0e1600c236e183` and its checked-in lazy `__all__` files; ordinary LightGlue package imports also need torchvision because of eager ALIKED imports. Do not install all MoGe pyproject extras/dependencies or a current MoGe3 distribution indiscriminately. The exact source closure and later artifact acquisition are owned by the frontend preparation protocol.

PyPI metadata bodies total **536,976 B in 14 requests**; no wheel body was transferred. Raw JSON and its append-only `http.jsonl` are retained privately in chat `work/0022-frontend-preparation/dependency-metadata`. These transfers are separate from the published 0021 HTTP ledger. The offline summarizer checks wheel tags, Python constraints and recursive active requirements before emitting the manifest.

## Fail-closed checkpoint admission

The pinned [loader source](https://github.com/microsoft/MoGe/blob/925b8ed835a7a9cdb7578ba15c658a0afc969030/moge/model/v2.py) uses `torch.load(..., map_location='cpu', weights_only=True)`, reads `model_config`, constructs the model and calls `load_state_dict(..., strict=False)`. The runtime adapter must instead **record the exact config and reject any missing/unexpected state keys or ignored/deprecated configuration entries**. Do not infer/replace the encoder, override heads, call `init_weights()`, or silently allow defaults to compensate for incompatible weights. Capture config/state-key hashes and the full load report. Checkpoint acquisition alone does not admit inference.

The bundled DINO constructor uses `pretrained=False`; calling its separate `init_weights()` would request a second pretrained backbone. Keep CPU float32, `eval()`/inference mode, explicit `use_fp16=False`, native PyTorch SDPA and `XFORMERS_DISABLED=1`. Require local verified weight paths, offline Hub settings and an actual network-denial check during artificial qualification; environment flags alone are not evidence that every downloader is disabled. Official LightGlue release metadata provides sizes but no publisher SHA for its two weight assets: any later exact-URL/size acquisition must be marked trust-on-first-use, with a newly measured local hash, rather than mislabeled publisher-verified.

Retain the full authored `infer()` postprocessing: focal/shift recovery, axial `depth=points[...,2]`, metric scaling and explicit mask semantics. Raw network points are not the same depth input. `fov_x` assumes the model's centered/isotropic camera form and is not an arbitrary full-K override. Keep authored camera calibration for geometry; record resize/pixel-center transforms and fixed token count. Expected infinity at a deliberately masked pixel must be distinguished from nonfinite arithmetic at a valid pixel. The actual checkpoint config, admitted tensor shape/token budget, CPU time and peak memory remain unknown until controlled artifact-only and artificial qualification. This review makes no scientific accuracy claim.
