"""First-stage SP/LG evidence only; no work or ML imports on module import.

run(inputs, output, record) requires the parent's completed source/assets/data
admission, offline guard and sys.path setup. Inputs: qualify_helper,
diagnostic_core, lightglue_source, runtime, superpoint_checkpoint,
lightglue_checkpoint, source_sha256, and cohort. Each cohort row is
{sequence, frame_count, initial_mask:{path,sha256,bytes},
 images:[{frame,path,sha256,bytes}]}; images are exactly frame 0 + eight queries.
No noninitial annotation path is accepted or opened here. No SAM is imported.
"""
import contextlib
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bound_file(item):
    path = Path(item["path"])
    require(path.is_file() and not path.is_symlink()
            and type(item["bytes"]) is int and path.stat().st_size == item["bytes"]
            and sha(path) == item["sha256"], "verified media input changed")
    return path


def module_origins(inputs):
    checked = {}
    runtime = Path(inputs["runtime"]).resolve(strict=True)
    packages = {
        "lightglue": ("lightglue_source", None), "numpy": ("runtime", "numpy"),
        "torch": ("runtime", "torch"), "torchvision": ("runtime", "torchvision"),
        "cv2": ("runtime", "opencv-python"), "PIL": ("runtime", "Pillow"),
        "kornia": ("runtime", "kornia"), "kornia_rs": ("runtime", "kornia_rs"),
    }
    for name, (key, distribution) in packages.items():
        loaded = importlib.import_module(name)
        root = Path(inputs[key]).resolve(strict=True)
        filename = getattr(loaded, "__file__", None)
        paths = [Path(filename).resolve(strict=True)] if filename else [
            Path(x).resolve(strict=True) for x in getattr(loaded, "__path__", ())]
        require(paths and all(x.is_relative_to(root) for x in paths), "wrong import origin: " + name)
        item = {"root_key": key, "relative_origins": [str(x.relative_to(root)) for x in paths]}
        if filename:
            item["module_sha256"] = sha(paths[0])
        if distribution:
            dist = importlib.metadata.distribution(distribution)
            require(Path(dist.locate_file("")).resolve().is_relative_to(runtime),
                    "wrong distribution metadata: " + name)
            item.update(distribution=distribution, version=dist.version)
        checked[name] = item
    return checked


def canonical_matches(arrays, counts, np):
    """Preserve native fields; add stable Nx2/N compact fields, including empty."""
    pairs = arrays["matches"][0] if arrays["matches"].ndim == 3 else arrays["matches"]
    scores = arrays["scores"][0] if arrays["scores"].ndim == 2 else arrays["scores"]
    require(all(np.isfinite(x).all() for x in arrays.values())
            and pairs.ndim == 2 and pairs.shape[1] == 2
            and pairs.dtype.kind in "iu" and scores.shape == (len(pairs),),
            "native match fields/nonfinite")
    require(np.all(pairs >= 0) and all(np.all(pairs[:, i] < counts[i]) for i in (0, 1))
            and len(np.unique(pairs[:, 0])) == len(pairs) == len(np.unique(pairs[:, 1])),
            "native compact match indices/mutuality")
    require(arrays["matches0"].shape == (1, counts[0])
            and arrays["matches1"].shape == (1, counts[1])
            and np.array_equal(arrays["matches0"][0, pairs[:, 0]], pairs[:, 1])
            and np.array_equal(arrays["matches1"][0, pairs[:, 1]], pairs[:, 0]),
            "native dense/compact matches disagree")
    arrays["match_ids"], arrays["match_scores"] = pairs, scores
    return pairs, scores


def run(inputs, output, record):
    """Prepare all 48 fixed witness rows; coverage failure is a completed result."""
    output = Path(output)
    require(not output.exists() and not output.is_symlink(), "fresh output required")
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema": "evr0029.frontend-evidence.v1", "status": "started",
              "complete": False, "warnings": [], "timings": [], "loads": {},
              "local_hub_calls": [], "images": [], "queries": [], "initial_masks": [],
              "source_sha256": inputs["source_sha256"], "coverage": None,
              "calls": {"superpoint_intent": 0, "superpoint_completed": 0,
                        "lightglue_intent": 0, "lightglue_completed": 0},
              "sam_calls": 0, "noninitial_annotation_access": False,
              "settings": {"device": "cpu", "dtype": "float32", "resize": 512,
                           "resize_side": "long", "maximum_keypoints": 2048,
                           "coordinate_convention": "original-image zero-based pixel centers"},
              "sam_sequence_admitted": False}
    record["frontend_evidence"] = result
    began = time.perf_counter()

    def save():
        result["wall_seconds"] = time.perf_counter() - began
        pending = output / "evidence-record.pending"
        pending.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        os.replace(pending, output / "evidence-record.json")

    @contextlib.contextmanager
    def timed(stage, row=None):
        result["stage"] = stage
        item = {"stage": stage, "status": "started"}
        result["timings"].append(item)
        if row is not None:
            row["status"] = "started"
        save()
        start = time.perf_counter()
        try:
            yield
            item["status"] = "completed"
            if row is not None:
                row["status"] = "completed"
        except BaseException as error:
            item.update(status="failed", error_type=type(error).__name__)
            if row is not None:
                row.update(status="failed", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            item["seconds"] = time.perf_counter() - start
            save()

    save()
    try:
        q = module("_evr0029_evidence_pinned_q", inputs["qualify_helper"])
        with timed("imports"), q.capture_warnings(result):
            import numpy as np
            import torch
            import cv2
            from PIL import Image
            from kornia.config import InstallationMode, kornia_config
            kornia_config.lazyloader.installation_mode = InstallationMode.RAISE
            from lightglue import SuperPoint, LightGlue
            core = module("_evr0029_evidence_arithmetic", inputs["diagnostic_core"])
            result["module_origins"] = module_origins(inputs)
        cohort = inputs["cohort"]
        require([(x["sequence"], x["frame_count"]) for x in cohort] == list(core.COHORT),
                "fixed six-video cohort changed")
        image_rows, query_rows = {}, {}
        for video in cohort:
            sequence = video["sequence"]
            require(set(video) == {"sequence", "frame_count", "initial_mask", "images"},
                    "unexpected video input fields")
            require(set(video["initial_mask"]) == {"path", "sha256", "bytes"}, "initial mask contract")
            expected = [0, *core.query_schedule(video["frame_count"])]
            require([x["frame"] for x in video["images"]] == expected
                    and all(set(x) == {"frame", "path", "sha256", "bytes"} for x in video["images"]),
                    "RGB inputs must be exactly initial plus eight fixed queries")
            initial = {"sequence": sequence, "frame": 0, "status": "not_attempted",
                       "input": {**video["initial_mask"], "path": str(video["initial_mask"]["path"])}}
            result["initial_masks"].append(initial)
            for media in video["images"]:
                row = {"sequence": sequence, "frame": media["frame"], "status": "not_attempted",
                       "input": {**media, "path": str(media["path"])}, "output": None}
                result["images"].append(row)
                image_rows[(sequence, media["frame"])] = row
            for frame in expected[1:]:
                row = {"sequence": sequence, "frame": frame, "status": "not_attempted",
                       "actionable": None, "matches": None, "witnesses": None}
                result["queries"].append(row); query_rows[(sequence, frame)] = row
        require(len(result["images"]) == 54 and len(result["queries"]) == 48, "fixed cardinalities")
        save()
        with timed("execution_configuration"):
            torch.set_num_threads(4); torch.set_num_interop_threads(1)
            torch.manual_seed(0); cv2.setNumThreads(0)
            result["execution"] = {"torch_intraop": torch.get_num_threads(),
                                   "torch_interop": torch.get_num_interop_threads(),
                                   "opencv_threads": cv2.getNumThreads(), "autocast": False}
            require(tuple(result["execution"][k] for k in
                          ("torch_intraop", "torch_interop", "opencv_threads")) == (4, 1, 1),
                    "thread configuration")
        with timed("strict_sp_lg_loading"), q.capture_warnings(result):
            paths = {q.SP_URL: Path(inputs["superpoint_checkpoint"]),
                     q.LG_URL: Path(inputs["lightglue_checkpoint"])}
            result["checkpoint_sha256"] = {url: sha(path) for url, path in paths.items()}
            with q.local_hub(torch, paths, result["local_hub_calls"]):
                sp = SuperPoint(**q.SP_OPTIONS).eval().to(device="cpu", dtype=torch.float32)
                raw = torch.load(paths[q.SP_URL], map_location="cpu", weights_only=True)
                result["loads"]["superpoint"] = q.strict_state(sp, raw, torch)
                lg = LightGlue(**q.LG_OPTIONS).eval().to(device="cpu", dtype=torch.float32)
                raw = q.rename_lightglue(torch.load(paths[q.LG_URL], map_location="cpu",
                                                   weights_only=True), lg.conf.n_layers)
                result["loads"]["lightglue"] = q.strict_state(lg, raw, torch, allow_confidence=True)
                del raw
            require([x["url"] for x in result["local_hub_calls"]] == [q.SP_URL, q.LG_URL],
                    "unexpected local weight loading")
            require(all(p.device.type == "cpu" and p.dtype == torch.float32
                        for model in (sp, lg) for p in model.parameters()), "model device/dtype")
            result["settings"].update(superpoint=dict(q.SP_OPTIONS), lightglue=dict(q.LG_OPTIONS))

        def extract(sequence, media):
            row = image_rows[(sequence, media["frame"])]
            with timed("extract:%s:%d" % (sequence, media["frame"]), row):
                path = bound_file(media)
                with Image.open(path) as image:
                    require(image.format == "JPEG", "expected admitted JPEG")
                    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
                h, w = rgb.shape[:2]
                row.update(shape=[h, w, 3], decoded_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
                tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255.
                result["calls"]["superpoint_intent"] += 1
                row["superpoint_call_intent"] = True; save()
                # The pinned Extractor restores (kp+.5)/scales-.5 to original xy.
                feature = sp.extract(tensor, resize=512)
                result["calls"]["superpoint_completed"] += 1
                arrays = {key: value.detach().cpu().numpy() for key, value in feature.items()}
                require(set(arrays) == {"keypoints", "keypoint_scores", "descriptors", "image_size"}
                        and all(np.isfinite(x).all() for x in arrays.values()), "feature fields/nonfinite")
                keypoints = arrays["keypoints"]
                require(keypoints.ndim == 3 and keypoints.shape[0] == 1
                        and keypoints.shape[2] == 2 and keypoints.shape[1] <= 2048,
                        "feature point shape/count")
                count = keypoints.shape[1]
                require(arrays["keypoint_scores"].shape == (1, count)
                        and arrays["descriptors"].shape == (1, count, 256)
                        and np.array_equal(arrays["image_size"], [[w, h]]), "feature dimensions/original image size")
                row["keypoint_count"] = count
                row["output"] = q.save_arrays(output / ("features-%s-%05d.npz" %
                                                        (sequence, media["frame"])), arrays, np)
                bound_file(media)
                return feature, arrays, (h, w)

        def match(sequence, frame, first, current, initial_labels, target_id, shape):
            row = query_rows[(sequence, frame)]
            with timed("match:%s:%d" % (sequence, frame), row):
                result["calls"]["lightglue_intent"] += 1
                row["lightglue_call_intent"] = True; save()
                matched = lg({"image0": first, "image1": current})
                result["calls"]["lightglue_completed"] += 1
                arrays = {key: (value[0].detach().cpu().numpy() if isinstance(value, list)
                                else value.detach().cpu().numpy())
                          for key, value in matched.items() if key != "stop"}
                arrays["stop"] = np.asarray(matched["stop"], dtype=np.int64)
                counts = [first["keypoints"].shape[1], current["keypoints"].shape[1]]
                pairs, scores = canonical_matches(arrays, counts, np)
                row["matches"] = q.save_arrays(output / ("matches-%s-%05d.npz" % (sequence, frame)), arrays, np)
                witnesses = core.select_witnesses(initial_labels,
                    first["keypoints"][0].detach().cpu().numpy(),
                    current["keypoints"][0].detach().cpu().numpy(), pairs, scores, shape,
                    target_id=target_id)
                witnesses.update(sequence=sequence, frame=frame)
                path = output / ("witnesses-%s-%05d.json" % (sequence, frame))
                with path.open("x") as stream:
                    json.dump(witnesses, stream, indent=2, allow_nan=False); stream.write("\n")
                row.update(target_id=target_id, match_count=len(pairs),
                           actionable=witnesses["actionable"], witness_counts=witnesses["counts"],
                           witnesses={"filename": path.name, "sha256": sha(path), "bytes": path.stat().st_size})

        with torch.inference_mode(), q.capture_warnings(result):
            for video, mask_row in zip(cohort, result["initial_masks"]):
                sequence = video["sequence"]
                with timed("initial_mask:" + sequence, mask_row):
                    path = bound_file(video["initial_mask"])
                    with Image.open(path) as image:
                        require(image.format == "PNG" and image.mode in ("P", "L", "I", "I;16"),
                                "initial annotation must retain authored integer IDs")
                        labels = np.asarray(image).copy()
                    target_id = core.choose_target(labels)
                    require(target_id is not None, "invalid initial annotation: no foreground target")
                    mask_row.update(target_id=target_id, shape=list(labels.shape),
                                    decoded_ids_sha256=hashlib.sha256(labels.tobytes()).hexdigest())
                    bound_file(video["initial_mask"])
                initial, _, initial_shape = extract(sequence, video["images"][0])
                require(labels.shape == initial_shape, "initial mask/RGB shape mismatch")
                for media in video["images"][1:]:
                    current, _, shape = extract(sequence, media)
                    match(sequence, media["frame"], initial, current, labels, target_id, shape)
                    del current
                del initial, labels
        require(result["calls"] == {"superpoint_intent": 54, "superpoint_completed": 54,
                                    "lightglue_intent": 48, "lightglue_completed": 48}
                and all(x["status"] == "completed" for x in result["images"] + result["queries"]),
                "incomplete fixed frontend")
        require({url: sha(path) for url, path in paths.items()} == result["checkpoint_sha256"],
                "checkpoint bytes changed")
        result["coverage"] = core.evidence_coverage(result["queries"])
        require(time.perf_counter() - began <= 1800, "frontend wall limit")
        result.update(status="completed", complete=True, stage="completed")
    except BaseException as error:
        result.update(status="failed", complete=False, error_type=type(error).__name__,
                      error=str(error), failed_stage=result.get("stage"))
        raise
    finally:
        save()
    return result
