# 0022 CPU runtime: source contract, not measured qualification

The pinned sources support a local CPU float32 path in principle. No model or image was loaded, no package was installed, and no inference was run for this audit. Checkpoint compatibility, actual memory, timing, valid depth and matching outputs remain unqualified.

Pins are MoGe `925b8ed835a7a9cdb7578ba15c658a0afc969030`, utils3d `3fab839f0be9931dac7c8488eb0e1600c236e183`, and LightGlue `eb42fee2d71449efb0aa5c10549752b5d75384d8`. Sources were reused from the private preparation directory before requesting missing import dependencies.

## Actual import closure

| Dependency | Why the selected path needs it |
|---|---|
| PyTorch, NumPy, Pillow | Existing runtime; models, geometry, artificial RGB image handling. |
| SciPy, OpenCV | MoGe `geometry_numpy` imports `cv2` and `scipy.signal`; full `infer` recovers focal/shift with `scipy.optimize.least_squares`, including shift fitting when focal is supplied. Returning only raw network output would change the method. |
| Hugging Face Hub and its declared dependencies | MoGe v2 imports `hf_hub_download` at module import even when the checkpoint is local. Offline loading can avoid calling it. |
| Torchvision | LightGlue package initialization imports ALIKED, whose module imports torchvision. Importing only `lightglue.superpoint` still executes package initialization. |
| Kornia, kornia-rs | SuperPoint uses Kornia grayscale conversion; LightGlue utilities import Kornia. Kornia initializes its I/O module, which imports `kornia_rs` eagerly. |
| Packaging, typing-extensions | Existing runtime; LightGlue SIFT module and utils3d helpers import these even though SIFT is not instantiated. |

These facts follow the pinned [MoGe model](https://github.com/microsoft/MoGe/blob/925b8ed835a7a9cdb7578ba15c658a0afc969030/moge/model/v2.py), [geometry code](https://github.com/microsoft/MoGe/blob/925b8ed835a7a9cdb7578ba15c658a0afc969030/moge/utils/geometry_numpy.py), [LightGlue package initialization](https://github.com/cvg/LightGlue/blob/eb42fee2d71449efb0aa5c10549752b5d75384d8/lightglue/__init__.py), and [Kornia I/O import](https://github.com/kornia/kornia/blob/v0.8.1/kornia/io/io.py).

The complete declared wheel dependency closure is recorded separately in `results/0022-preparation/dependency-wheels.json`: 14 additions, 76,809,693 bytes, with exact filenames, SHA256s, Python/macOS ARM tags and active requirements. Versions are scipy 1.13.1, opencv-python 4.11.0.86, torchvision 0.23.0, kornia 0.8.1, kornia-rs 0.1.9, huggingface-hub 0.34.4, hf-xet 1.1.9, PyYAML 6.0.2, requests 2.32.5, urllib3 2.5.0, certifi 2025.10.5, charset-normalizer 3.4.4, idna 3.11 and tqdm 4.67.1. This is a conservative declared-dependency installation closure, not a claim that every wheel is exercised by local inference. Existing torch 2.8.0 matches torchvision's exact requirement. No additional missing hard dependency was identified in the inspected inference import paths.

Do not install all broad research/demo requirements. Matplotlib, moderngl, pycolmap, xformers, ONNX runtimes, transformers, diffusers and CUDA are not used by this selected path. Crucially, retain utils3d's checked-in `_*.__all__.py` files: its [lazy-import helper](https://github.com/EasternJournalist/utils3d/blob/3fab839f0be9931dac7c8488eb0e1600c236e183/utils3d/helpers.py) imports modules eagerly and writes metadata if those files are missing. A shortened source tree could therefore change the dependency closure.

## No implicit network or package installation

Kornia's [default installation mode](https://github.com/kornia/kornia/blob/v0.8.1/kornia/config.py) is **ASK**. Its [optional-module LazyLoader](https://github.com/kornia/kornia/blob/v0.8.1/kornia/core/external.py) can call `pip install` in ASK/AUTO modes. Qualification must deny process creation and socket/network connections before third-party imports, then set `kornia_config.lazyloader.installation_mode = InstallationMode.RAISE`. Set Hugging Face offline variables, disable xformers before import, and reject any attempted network/process operation. These are fail-closed checks, not a reason to install optional models.

MoGe's bundled [DINO constructor](https://github.com/microsoft/MoGe/blob/925b8ed835a7a9cdb7578ba15c658a0afc969030/moge/model/modules.py) calls its local backbone with `pretrained=False`; never call `init_weights()`, which would request pretrained DINO weights. Load the preverified local MoGe checkpoint with `torch.load(..., map_location='cpu', weights_only=True)`, validate the constructor configuration including nested encoder arguments, and require strict state compatibility. Its upstream convenience loader uses `strict=False`, so a successful return alone would not establish a valid checkpoint. Unknown or ignored configuration is a qualification failure; actual checkpoint configuration has not been inspected here.

[SuperPoint](https://github.com/cvg/LightGlue/blob/eb42fee2d71449efb0aa5c10549752b5d75384d8/lightglue/superpoint.py) and [LightGlue](https://github.com/cvg/LightGlue/blob/eb42fee2d71449efb0aa5c10549752b5d75384d8/lightglue/lightglue.py) constructors request hard-coded release URLs. A scoped adapter may map **only those two exact URLs** to previously verified local `weights_only=True` loads, restoring the original function afterward. Preserve `features='superpoint'` and the official legacy attention-key renaming. After construction independently check every learned key, shape and loaded value. The only prospective missing-key exception is LightGlue's `confidence_thresholds`: a deterministic registered buffer generated by the pinned constructor's formula, not a learned weight. If present, it must agree with that formula; unknown/missing learned keys fail. Do not misinterpret the constructor's initial permissive legacy load as final strict compatibility.

## Prospective manufactured-input qualification

Use CPU float32, four threads, no compile, no mixed precision, no pretrained DINO call. Fixed artificial RGB image: 384×512, seed 0 noise plus declared smooth gradients and squares; second image is a fixed 12-pixel horizontal shift with zero fill. This is a runtime fixture, not a geometric accuracy benchmark.

Run complete MoGe `infer` once per image with `num_tokens=1200`, `use_fp16=False`, `apply_mask=False`, `force_projection=True`; retain full raw output depth and mask, including invalid-region values. Validate shape and finite positive depth on the valid mask. Preserve focal/shift recovery, projection and metric-scale postprocessing. Run SuperPoint once per image with at most 2048 points and `extract(..., resize=None)`, then LightGlue with `flash=False`, `mp=False`, `depth_confidence=-1`, `width_confidence=-1`. Verify output shapes, descriptor finiteness and match index bounds; zero usable depth or zero matches is an ineligible qualification. No thresholds or fixture may be tuned after output inspection.

The pinned extractor's pixel-center scale conversion must remain intact. No resize is used for this fixture. This qualification cannot establish real-image pose quality, external generalization, or acceptable downstream full-dataset cost.

## Transfer receipt

This audit added **393,924 response-body bytes in 48 requests**, bounded by 1 MiB total and 256 KiB per response. Two guessed source paths returned 404 and remain in the accounting: utils3d `interface.py` (its absence is handled by the pinned import guard) and Kornia `transpiler.py` (the package instead exposes a subpackage; no file payload was obtained at that URL). Other responses were official GitHub source or tree metadata, with no weights, images or wheels. Private receipt: `work/0022-frontend-preparation/dependency-audit/http-receipt.json` (project-root relative), SHA256 `969ea5081cce83e8b055e4ce61e7f04b1c8fc6fe04e4c8ff54d1cc67d2ad4fce`. Reused source downloads and the independent 536,976-byte wheel metadata audit are accounted separately; do not add them twice.
