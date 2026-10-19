"""Offline, once-per-image depth/features and once-per-edge matching caches.

Only source preparation/artificial tests are admitted until a separate protocol
freezes the plan, source and assets. No scientific work happens on import.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MAX_EDGE = 512
RANGE_LIMITS = dict(body_bytes=20 * 1024**2, requests=162, wall_seconds=1800, request_seconds=30)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


q = module("_evr0023_frozen_qualifier", ROOT / "experiments/0022-frontend/qualify.py")
ranges = module("_evr0023_frontend_ranges", HERE / "range_inputs.py")
acquisition = module("_evr0023_frontend_acquisition", HERE / "acquire_operation.py")


def require(value, message):
    if not value:
        raise ValueError(message)


def raw_sha(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def write_json_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def safe_input(root, name):
    ranges.frontend_kind(name)  # Explicitly excludes every pose/GT path.
    root = Path(root).resolve()
    out = root / name
    require(out.resolve().is_relative_to(root), "frontend input escapes private root")
    parent = out
    while parent != root.parent:
        require(not parent.is_symlink(), "symlink frontend input")
        parent = parent.parent
    return out


def validate_plan(plan):
    require(plan.get("schema") == "evr0023.proposed-plan.v1", "plan schema")
    scenes = plan.get("scenes")
    require(isinstance(scenes, list) and len(scenes) == 3, "three planned places required")
    names, edges, image_names, k_names = [], [], [], []
    for scene in scenes:
        selected = scene["selected_seq1_chronological"]
        require(len(selected) == 16, "sixteen planned seq1 images required")
        images = [scene["anchor"]] + [row["filename"] for row in selected]
        k = scene["intrinsics_member"]["filename"]
        require(len(set(images)) == 17 and ranges.frontend_kind(k) == "intrinsics", "scene frontend members")
        require(all(ranges.frontend_kind(name) == "image" and name.startswith("val/" + scene["scene"] + "/")
                    for name in images), "scene image paths")
        require([m["filename"] for m in scene["image_members"]] == images, "image metadata order")
        require(len(scene["reference_edges"]) == 32 and len(scene["query_edges"]) == 8, "edge counts")
        for edge in scene["reference_edges"] + scene["query_edges"]:
            require(edge["source"] in images and edge["target"] in images
                    and edge["source"] != edge["target"], "edge image identity")
            edges.append(dict(edge, scene=scene["scene"]))
        names.extend(images + [k]); image_names.extend(images); k_names.append(k)
    require(len(set(names)) == 54, "frontend member uniqueness")
    require(len({(e["source"], e["target"]) for e in edges}) == 120, "edge uniqueness")
    frontend = plan["frontend_acquisition"]
    require(frontend["member_names"] == names and frontend["image_names"] == image_names
            and frontend["intrinsics_names"] == k_names, "frontend plan list mismatch")
    return image_names, edges


def verify_inputs(args):
    require(q.sha(args.plan) == args.plan_sha256, "plan hash mismatch")
    plan = q.read_json(args.plan)
    validate_plan(plan)
    require(q.sha(args.input_receipt) == args.input_receipt_sha256, "input receipt hash mismatch")
    receipt = q.read_json(args.input_receipt)
    require(receipt.get("schema") == ranges.SCHEMA and receipt.get("status") == "completed"
            and receipt.get("response_body_bytes_exact") is True
            and receipt.get("pose_gt_members_allowed") is False, "incomplete frontend acquisition")
    require(receipt.get("source_sha256") == q.sha(HERE / "range_inputs.py"), "range helper source changed")
    for key in ("url", "etag", "archive_bytes"):
        require(receipt.get(key) == plan["source"][key], "frontend archive identity mismatch")
    require(receipt.get("index_sha256") == plan["source"]["cached_index_sha256"], "frontend index fingerprint")
    require(receipt.get("selected_names") == plan["frontend_acquisition"]["member_names"], "acquisition plan coverage")
    limits = receipt["limits"]
    budget = plan["frontend_acquisition"]
    fixed_limits = dict(RANGE_LIMITS, output_bytes=budget["uncompressed_bytes"],
                        member_bytes=budget["maximum_member_uncompressed_bytes"])
    require(limits == fixed_limits, "acquisition caps differ from frozen protocol/plan")
    require(type(receipt.get("wall_seconds")) in (int, float)
            and math.isfinite(receipt["wall_seconds"]) and 0 <= receipt["wall_seconds"] <= limits["wall_seconds"]
            and type(receipt.get("response_body_bytes")) is int
            and 0 <= receipt["response_body_bytes"] <= min(budget["response_body_upper_bound"], limits["body_bytes"])
            and receipt["output_bytes"] == budget["uncompressed_bytes"]
            and len(receipt["requests"]) == budget["range_requests"] == 162, "frontend acquisition limits")
    require(all(type(row.get("body_bytes")) is int and row["body_bytes"] >= 0 and row.get("status") == 206
                for row in receipt["requests"])
            and sum(row["body_bytes"] for row in receipt["requests"]) == receipt["response_body_bytes"],
            "acquisition response-body ledger/count mismatch")
    records = receipt["members"]
    require([r["filename"] for r in records] == receipt["selected_names"], "acquisition record order")
    expected = {r["filename"]: r for scene in plan["scenes"]
                for r in scene["image_members"] + [scene["intrinsics_member"]]}
    minimum_body = sum(r["compressed_bytes"] + 30 + len(r["filename"].encode("ascii")) for r in expected.values())
    require(receipt["response_body_bytes"] >= minimum_body, "acquisition body undercounts required members")
    verified = {}
    for row in records:
        name = row["filename"]
        require(all(row[k] == expected[name][k] for k in ranges.CORE_FIELDS), "acquisition ZIP metadata changed")
        path = safe_input(args.inputs, name)
        require(path.is_file() and path.stat().st_size == row["uncompressed_bytes"]
                and q.sha(path) == row["sha256"], "frontend member bytes changed")
        verified[name] = row["sha256"]
    return plan, verified


def parse_intrinsics(text, selected_paths):
    """Author README c3290ec5: frame_path fx fy cx cy frame_width frame_height.

    Only requested rows' numeric columns are parsed. No pose file is accepted.
    Frames may be skipped; filenames, not row position, identify calibration.
    """
    requested, seen, result = set(selected_paths), set(), {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        columns = line.split()
        require(len(columns) == 7, "intrinsics row must have seven columns")
        name = columns[0]
        require(re.fullmatch(r"seq[01]/frame_[0-9]{5}\.jpg", name) is not None and name not in seen,
                "invalid or duplicate intrinsics image path")
        seen.add(name)
        if name not in requested:
            continue
        numbers = [float(v) for v in columns[1:]]
        require(all(math.isfinite(v) for v in numbers), "nonfinite intrinsics")
        fx, fy, cx, cy, w, h = numbers
        require(fx > 0 and fy > 0 and w.is_integer() and h.is_integer()
                and 1 <= w <= 16384 and 1 <= h <= 16384 and w * h <= 32000000,
                "intrinsics dimensions/focal invalid")
        result[name] = dict(fx=fx, fy=fy, cx=cx, cy=cy, width=int(w), height=int(h))
    require(set(result) == requested, "selected image calibration missing")
    return result


def resize_geometry(calibration, np):
    w, h = calibration["width"], calibration["height"]
    scale = min(1.0, MAX_EDGE / max(w, h))
    rw, rh = max(1, math.floor(w * scale + .5)), max(1, math.floor(h * scale + .5))
    sx, sy = rw / w, rh / h
    original = np.array([[calibration["fx"], 0, calibration["cx"]],
                         [0, calibration["fy"], calibration["cy"]], [0, 0, 1]], dtype=np.float64)
    resized = original.copy()
    resized[0, 0] *= sx; resized[1, 1] *= sy
    resized[0, 2] = (original[0, 2] + .5) * sx - .5
    resized[1, 2] = (original[1, 2] + .5) * sy - .5
    return original, resized, (rw, rh), (sx, sy)


def prepare_image(path, calibration, np, Image):
    original_k, resized_k, size, scales = resize_geometry(calibration, np)
    started = time.perf_counter()
    with Image.open(path) as image:
        require(image.format in ("JPEG", "PNG") and image.size == (calibration["width"], calibration["height"]),
                "decoded image dimensions disagree with authored calibration")
        require(image.getexif().get(274, 1) == 1, "nonidentity EXIF orientation requires separate source review")
        original = image.convert("RGB")
        raw = np.asarray(original, dtype=np.uint8).copy()
        decode_seconds = time.perf_counter() - started
        started = time.perf_counter()
        resized = original if original.size == size else original.resize(size, resample=Image.Resampling.BILINEAR)
        pixels = np.asarray(resized, dtype=np.uint8).copy()
    values = np.ascontiguousarray(pixels.astype(np.float32) / np.float32(255))
    resize_seconds = time.perf_counter() - started
    arrays = dict(K_original=original_k, K_resized=resized_k,
                  image_size_original=np.asarray([calibration["width"], calibration["height"]], dtype=np.int64),
                  image_size_resized=np.asarray(size, dtype=np.int64))
    metadata = dict(original_encoded_sha256=q.sha(path), original_rgb_uint8_sha256=raw_sha(raw),
                    resized_rgb_uint8_sha256=raw_sha(pixels), resized_rgb_float32_sha256=raw_sha(values),
                    original_shape_hwc=list(raw.shape), resized_shape_hwc=list(values.shape),
                    scales_xy=list(scales), resampling="Pillow RGB bilinear; no EXIF transpose",
                    authored_K_original=original_k.tolist(), authored_K_resized=resized_k.tolist(),
                    coordinate_convention="zero-based pixel centers; K' principal point=(c+.5)*scale-.5",
                    decode_seconds=decode_seconds, resize_seconds=resize_seconds)
    return values, arrays, metadata


def validate_depth(raw, shape, np):
    h, w = shape
    require({"depth", "mask", "points", "intrinsics"} <= set(raw), "missing MoGe outputs")
    d, mask = raw["depth"], raw["mask"]
    require(d.shape == mask.shape == (h, w) and d.dtype == np.float32 and mask.dtype == np.bool_,
            "depth/mask shape or dtype")
    require(raw["points"].shape == (h, w, 3) and raw["intrinsics"].shape == (3, 3)
            and np.isfinite(raw["intrinsics"]).all(), "depth geometry shape/finiteness")
    require(np.isfinite(d[mask]).all() and (d[mask] > 0).all()
            and np.isfinite(raw["points"][mask]).all(), "invalid arithmetic at model-valid depth pixel")
    if "normal" in raw:
        require(raw["normal"].shape == (h, w, 3) and np.isfinite(raw["normal"][mask]).all(), "invalid model-valid normal")


def validate_features(raw, shape, np):
    h, w = shape
    require({"keypoints", "descriptors", "keypoint_scores", "image_size"} <= set(raw), "missing SuperPoint outputs")
    xy, desc, scores = raw["keypoints"], raw["descriptors"], raw["keypoint_scores"]
    require(xy.ndim == 3 and xy.shape[0] == 1 and xy.shape[2] == 2 and 0 <= xy.shape[1] <= 2048,
            "SuperPoint keypoint shape/count")
    n = xy.shape[1]
    require(desc.shape == (1, n, 256) and scores.shape == (1, n)
            and xy.dtype == desc.dtype == np.float32
            and all(np.isfinite(v).all() for v in raw.values()), "SuperPoint shape/finiteness")
    require((xy >= 0).all() and (xy[..., 0] < w).all() and (xy[..., 1] < h).all()
            and np.array_equal(raw["image_size"], [[w, h]]), "SuperPoint coordinates/image_size")
    return xy[0]


def sample_depth(depth, mask, xy, np):
    """Four distinct, in-bounds positive-valid neighbors; never removes xy."""
    h, w = depth.shape
    positive = mask & np.isfinite(depth) & (depth > 0)
    median = float(np.median(depth[positive].astype(np.float64))) if positive.any() else float("nan")
    values = np.full(len(xy), np.nan, dtype=np.float64)
    valid = np.zeros(len(xy), dtype=np.bool_)
    interior = np.isfinite(xy).all(axis=1) & (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < w - 1) & (xy[:, 1] < h - 1)
    ids = np.flatnonzero(interior)
    if len(ids):
        xx, yy = xy[ids, 0].astype(np.float64), xy[ids, 1].astype(np.float64)
        x, y = np.floor(xx).astype(np.int64), np.floor(yy).astype(np.int64)
        okay = positive[y, x] & positive[y, x + 1] & positive[y + 1, x] & positive[y + 1, x + 1]
        ids, x, y, xx, yy = ids[okay], x[okay], y[okay], xx[okay], yy[okay]
        dx, dy = xx - x, yy - y
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            values[ids] = ((1 - dx) * (1 - dy) * depth[y, x].astype(np.float64)
                           + dx * (1 - dy) * depth[y, x + 1].astype(np.float64)
                           + (1 - dx) * dy * depth[y + 1, x].astype(np.float64)
                           + dx * dy * depth[y + 1, x + 1].astype(np.float64))
        valid[ids] = True
    normalized = np.full(len(xy), np.nan, dtype=np.float64)
    if math.isfinite(median):
        require(median > 0, "positive-valid depth median must be positive")
        normalized[valid] = values[valid] / median
        require(np.isfinite(normalized[valid]).all(), "nonfinite normalized sampled depth")
    else:
        require(not valid.any(), "sample available without image median")
    return dict(depth_sample_raw=values, depth_sample_normalized=normalized,
                depth_sample_valid=valid, depth_positive_valid_median=np.asarray(median, dtype=np.float64))


def validate_matches(raw, n0, n1, np):
    require({"matches", "scores", "matches0", "matches1", "matching_scores0", "matching_scores1", "stop"} <= set(raw),
            "missing LightGlue outputs")
    require(all(np.isfinite(v).all() for v in raw.values()), "nonfinite LightGlue output")
    pairs, scores = raw["matches"], raw["scores"]
    if pairs.ndim == 3:
        require(pairs.shape[0] == 1, "LightGlue batch size")
        pairs, scores = pairs[0], scores[0]
    require(pairs.ndim == 2 and pairs.shape[1] == 2 and np.issubdtype(pairs.dtype, np.integer)
            and scores.shape == (len(pairs),), "compact match shape")
    require((pairs >= 0).all() and (pairs[:, 0] < n0).all() and (pairs[:, 1] < n1).all(), "compact match bounds")
    require(len(np.unique(pairs[:, 0])) == len(pairs) == len(np.unique(pairs[:, 1])), "nonmutual compact matches")
    for name, scores_name, n, other in (("matches0", "matching_scores0", n0, n1), ("matches1", "matching_scores1", n1, n0)):
        dense = raw[name]
        require(dense.shape == (1, n) and np.issubdtype(dense.dtype, np.integer)
                and raw[scores_name].shape == (1, n) and (dense >= -1).all() and (dense < other).all(), "dense match bounds")
    require(np.array_equal(raw["matches0"][0, pairs[:, 0]], pairs[:, 1])
            and np.array_equal(raw["matches1"][0, pairs[:, 1]], pairs[:, 0])
            and int((raw["matches0"] >= 0).sum()) == len(pairs)
            and int((raw["matches1"] >= 0).sum()) == len(pairs), "compact/dense matching disagreement")
    return pairs


def edge_arrays(raw, source, target, np):
    pairs = validate_matches(raw, len(source["xy"]), len(target["xy"]), np)
    # Authored raw fields remain unchanged. These two aliases always have rank
    # 2/1, including LightGlue's different tensor-shaped empty-output branch.
    arrays = dict(raw, match_ids=pairs, match_scores=raw["scores"].reshape(-1),
                  xy0=source["xy"][pairs[:, 0]], xy1=target["xy"][pairs[:, 1]])
    for end, cache in enumerate((source, target)):
        ids = pairs[:, end]
        arrays["depth%d_raw" % end] = cache["depth_sample_raw"][ids]
        arrays["depth%d_normalized" % end] = cache["depth_sample_normalized"][ids]
        arrays["depth_valid%d" % end] = cache["depth_sample_valid"][ids]
    return arrays


class Backend:
    """Original0022 loader sequence, using its unchanged strict helper functions."""
    def __init__(self, args, files, wheels, record):
        os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          XFORMERS_DISABLED="1", PYTHONDONTWRITEBYTECODE="1")
        sys.dont_write_bytecode = True
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = "4"
        sys.path[:0] = [str(args.installation / "runtime")] + [str(args.installation / "sources" / k) for k in q.PINS]
        q.install_guard(record["offline_attempts"], record["compatibility_probes"])
        started = time.perf_counter()
        import numpy as np
        import torch
        import cv2
        import kornia
        from kornia.config import InstallationMode, kornia_config
        kornia_config.lazyloader.installation_mode = InstallationMode.RAISE
        record["typing_compatibility"] = q.prepare_utils3d_typing(args.installation)
        from moge.model.v2 import MoGeModel
        from moge.model.modules import DINOv2Encoder, ConvStack, MLP
        from lightglue import SuperPoint, LightGlue
        from PIL import Image
        torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.manual_seed(0)
        versions = {x["name"]: x["version"] for x in wheels["wheels"]}
        reuse = wheels["reuse_installed"]
        reuse = reuse.values() if isinstance(reuse, dict) else reuse
        versions.update({x["name"]: x["version"] for x in reuse})
        q.require(all(importlib.metadata.version(k) == v for k, v in versions.items()), "runtime_version")
        record["runtime_versions"] = versions
        for package in q.PINS:
            q.require(Path(sys.modules[package].__file__).resolve().is_relative_to(
                (args.installation / "sources" / package).resolve()), "source_import_origin")
        record["timings"]["imports_seconds"] = time.perf_counter() - started
        q.require(not record["offline_attempts"], "blocked_import_operation")
        with q.capture_warnings(record):
            started = time.perf_counter()
            checkpoint = torch.load(files["moge2-small-model.pt"], map_location="cpu", weights_only=True)
            q.require(isinstance(checkpoint, dict) and "model_config" in checkpoint and "model" in checkpoint, "moge_checkpoint_structure")
            config = checkpoint["model_config"]
            q.canonical(config)
            q.validate_config(config, dict(model=MoGeModel, encoder=DINOv2Encoder, conv=ConvStack, mlp=MLP))
            record["moge_config"], record["moge_config_sha256"] = config, q.canonical(config)
            moge = MoGeModel(**config)
            record["loads"] = {"moge": q.strict_state(moge, checkpoint["model"], torch)}
            del checkpoint
            record["timings"]["moge_load_seconds"] = time.perf_counter() - started
            paths = {q.SP_URL: files["superpoint_v1.pth"], q.LG_URL: files["superpoint_lightglue.pth"]}
            with q.local_hub(torch, paths, record["local_hub_calls"]):
                started = time.perf_counter(); sp = SuperPoint(**q.SP_OPTIONS)
                state = torch.load(paths[q.SP_URL], map_location="cpu", weights_only=True)
                record["loads"]["superpoint"] = q.strict_state(sp, state, torch); del state
                record["timings"]["superpoint_load_seconds"] = time.perf_counter() - started
                started = time.perf_counter(); lg = LightGlue(**q.LG_OPTIONS)
                state = q.rename_lightglue(torch.load(paths[q.LG_URL], map_location="cpu", weights_only=True), lg.conf.n_layers)
                record["loads"]["lightglue"] = q.strict_state(lg, state, torch, allow_confidence=True); del state
                record["timings"]["lightglue_load_seconds"] = time.perf_counter() - started
        q.require(not any("ignored" in w["message"].lower() or "invalid arguments" in w["message"].lower() for w in record["warnings"]), "ignored_model_config")
        q.require([x["url"] for x in record["local_hub_calls"]] == [q.SP_URL, q.LG_URL], "hub_call_count")
        for model in (moge, sp, lg):
            model.eval().to(device="cpu", dtype=torch.float32)
            q.require(all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters()), "model_device_dtype")
        record["execution_configuration"] = q.configure_execution(moge, torch, cv2)
        self.np, self.Image, self.torch, self.moge, self.sp, self.lg = np, Image, torch, moge, sp, lg

    def tensor(self, image):
        return self.torch.from_numpy(image.transpose(2, 0, 1).copy())

    def depth(self, image):
        with self.torch.inference_mode():
            output = self.moge.infer(self.tensor(image), **q.MOGE_OPTIONS)
        return {k: v.detach().cpu().numpy() for k, v in output.items()}

    def features(self, image):
        with self.torch.inference_mode():
            output = self.sp.extract(self.tensor(image), resize=None)
        return {k: v.detach().cpu().numpy() for k, v in output.items()}

    def match(self, feature0, feature1):
        features = [{k: self.torch.from_numpy(v.copy()) for k, v in feature.items()} for feature in (feature0, feature1)]
        with self.torch.inference_mode():
            output = self.lg(dict(image0=features[0], image1=features[1]))
        raw = {k: (v[0].detach().cpu().numpy() if isinstance(v, list) else v.detach().cpu().numpy())
               for k, v in output.items() if k != "stop"}
        raw["stop"] = self.np.asarray(output["stop"], dtype=self.np.int64)
        return raw


def process(plan, inputs, output, backend, record, save):
    """Artificial tests inject a fake backend; the real run uses the verified Backend."""
    np, Image = backend.np, backend.Image
    names, edges = validate_plan(plan)
    record["images"] = [dict(image_index=i, filename=name, status="not_attempted") for i, name in enumerate(names)]
    record["edges"] = [dict(edge_index=i, **edge, status="not_attempted") for i, edge in enumerate(edges)]
    save()
    calibration, calibration_errors = {}, {}
    for scene in plan["scenes"]:
        prefix = "val/" + scene["scene"] + "/"
        selected = [row["filename"][len(prefix):] for row in scene["image_members"]]
        try:
            rows = parse_intrinsics(safe_input(inputs, scene["intrinsics_member"]["filename"]).read_text("utf-8"), selected)
            calibration.update({prefix + key: value for key, value in rows.items()})
        except Exception as exc:
            for key in selected:
                calibration_errors[prefix + key] = str(exc)
    feature_cache, image_cache = {}, {}
    for row in record["images"]:
        began, arrays = time.perf_counter(), {}
        row.update(status="started", stage="decode", timings={})
        record["stage"] = "image:%03d" % row["image_index"]; save()
        try:
            require(row["filename"] not in calibration_errors, calibration_errors.get(row["filename"], "calibration"))
            image, arrays, metadata = prepare_image(safe_input(inputs, row["filename"]), calibration[row["filename"]], np, Image)
            row["input"] = metadata
            row["timings"].update(decode_seconds=metadata["decode_seconds"], resize_seconds=metadata["resize_seconds"])
            row["stage"] = "moge"; save(); started = time.perf_counter()
            try:
                depth = backend.depth(image)
            finally:
                row["timings"]["moge_seconds"] = time.perf_counter() - started
            arrays.update(depth)
            validate_depth(depth, image.shape[:2], np)
            row["stage"] = "superpoint"; save(); started = time.perf_counter()
            try:
                feature = backend.features(image)
            finally:
                row["timings"]["superpoint_seconds"] = time.perf_counter() - started
            require(not set(arrays) & set(feature), "model output key collision")
            arrays.update(feature)
            xy = validate_features(feature, image.shape[:2], np)
            sampled = sample_depth(depth["depth"], depth["mask"], xy, np)
            arrays.update(sampled, xy=xy)
            row["stage"] = "cache"; save(); started = time.perf_counter()
            row["cache"] = q.save_arrays(output / ("image-%03d.npz" % row["image_index"]), arrays, np)
            row["timings"]["cache_seconds"] = time.perf_counter() - started
            median = float(sampled["depth_positive_valid_median"])
            row.update(status="completed", stage="completed", keypoint_count=len(xy),
                       sampled_depth_valid_count=int(sampled["depth_sample_valid"].sum()),
                       positive_valid_depth_median=median if math.isfinite(median) else None,
                       model_valid_depth_pixels=int(depth["mask"].sum()))
            feature_cache[row["filename"]] = feature
            image_cache[row["filename"]] = dict(xy=xy, **sampled)
        except BaseException as exc:
            row.update(status="error", failure_stage=row["stage"], error=dict(type=type(exc).__name__, message=str(exc)))
            if arrays and "cache" not in row:
                row["partial_cache"] = q.save_arrays(output / ("image-%03d-partial.npz" % row["image_index"]), arrays, np)
            if not isinstance(exc, Exception):
                raise
        finally:
            row["timings"]["total_seconds"] = time.perf_counter() - began
            write_json_new(output / ("image-%03d.json" % row["image_index"]), row)
            save()
    for row in record["edges"]:
        began, arrays = time.perf_counter(), {}
        row.update(status="started", stage="inputs", timings={})
        record["stage"] = "edge:%03d" % row["edge_index"]; save()
        try:
            require(row["source"] in feature_cache and row["target"] in feature_cache, "source_or_target_frontend_failed")
            row["stage"] = "lightglue"; save(); started = time.perf_counter()
            try:
                raw = backend.match(feature_cache[row["source"]], feature_cache[row["target"]])
            finally:
                row["timings"]["lightglue_seconds"] = time.perf_counter() - started
            arrays.update(raw)
            a, b = image_cache[row["source"]], image_cache[row["target"]]
            arrays = edge_arrays(raw, a, b, np)
            row["stage"] = "cache"; save(); started = time.perf_counter()
            row["cache"] = q.save_arrays(output / ("edge-%03d.npz" % row["edge_index"]), arrays, np)
            row["timings"]["cache_seconds"] = time.perf_counter() - started
            row.update(status="completed", stage="completed", match_count=len(arrays["match_ids"]),
                       both_depth_valid_count=int((arrays["depth_valid0"] & arrays["depth_valid1"]).sum()),
                       geometry_observations_removed_for_depth=0)
        except BaseException as exc:
            row.update(status="error", failure_stage=row["stage"], error=dict(type=type(exc).__name__, message=str(exc)))
            if arrays and "cache" not in row:
                row["partial_cache"] = q.save_arrays(output / ("edge-%03d-partial.npz" % row["edge_index"]), arrays, np)
            if not isinstance(exc, Exception):
                raise
        finally:
            row["timings"]["total_seconds"] = time.perf_counter() - began
            write_json_new(output / ("edge-%03d.json" % row["edge_index"]), row)
            save()


def qualification_args(args):
    return SimpleNamespace(installation=args.installation, installation_supervision=args.installation_supervision,
                           asset_manifest=args.asset_manifest, wheel_manifest=args.wheel_manifest,
                           wheel_assets_manifest=args.wheel_assets_manifest,
                           expected_commit=args.frontend_freeze, acquisition_receipt=args.acquisition_receipt,
                           assets_dir=args.assets_dir, protocol=args.frontend_protocol)


def verify_qualification(args, bindings, frozen):
    previous, outer = q.read_json(args.qualification_receipt), q.read_json(args.qualification_supervision)
    require(previous.get("schema") == "evr0022.qualification.v1" and previous.get("status") == "completed"
            and previous.get("eligible") is True and previous.get("expected_commit") == args.frontend_freeze
            and previous.get("bindings") == bindings and previous.get("frozen_files") == frozen,
            "completed0022qualification binding required")
    require(not previous.get("offline_attempts") and previous.get("source_sha256") == q.sha(q.__file__), "qualification source/offline guard")
    require(outer.get("schema") == "evr0022.supervision.v1" and outer.get("operation") == "qualification"
            and outer.get("status") == "completed" and outer.get("exit_code") == 0
            and outer.get("expected_commit") == args.frontend_freeze and outer.get("termination") is None
            and outer.get("remaining_descendant_pids") == [] and outer.get("cleanup_errors") == [], "qualification supervision")
    outer_keys = ("experiments/0022-frontend/qualify.py", "experiments/0022-frontend/supervise.py")
    require(outer.get("source_sha256") == {key: frozen[key] for key in outer_keys}, "qualification supervisor source binding")
    require(type(outer.get("wall_seconds")) in (int, float) and math.isfinite(outer["wall_seconds"])
            and 0 <= outer["wall_seconds"] < 1800 and outer.get("hard_timeout_seconds") == 1800
            and type(outer.get("peak_group_rss_kib")) is int and 0 <= outer["peak_group_rss_kib"] <= 8 * 1024**2
            and outer.get("group_rss_limit_kib") == 8 * 1024**2, "qualification resource limits")
    logs = {}
    for name in ("stdout", "stderr"):
        path = args.qualification_supervision.parent / (name + ".log")
        logs[name] = q.sha(path)
        require(logs[name] == outer.get(name + "_sha256"), "qualification log hash")
    command = outer.get("command")
    require(isinstance(command, list) and len(command) >= 4 and command[0] == "$PYTHON"
            and command[1] == "$REPOSITORY/experiments/0022-frontend/qualify.py"
            and len(command[2:]) % 2 == 0, "qualification command")
    flags = dict(zip(command[2::2], command[3::2]))
    require(len(flags) * 2 == len(command[2:]) and flags.get("--expected-commit") == args.frontend_freeze,
            "qualification command bindings")
    def expanded(value):
        path = Path(value.replace("$REPOSITORY", str(ROOT)).replace("$PRIVATE", str(ROOT.parent)))
        return (path if path.is_absolute() else ROOT / path).resolve()
    for key, path in {"--output-dir": args.qualification_receipt.parent, "--installation": args.installation,
                      "--installation-supervision": args.installation_supervision, "--assets-dir": args.assets_dir,
                      "--acquisition-receipt": args.acquisition_receipt}.items():
        require(key in flags and expanded(flags[key]) == Path(path).resolve(), "qualification child input mismatch")
    lines = (args.qualification_supervision.parent / "stdout.log").read_text().splitlines()
    require(bool(lines), "qualification stdout missing")
    summary = json.loads(lines[-1])
    require(summary == dict(status="completed", eligible=True, wall_seconds=previous.get("wall_seconds")),
            "qualification stdout/receipt mismatch")
    return dict(qualification_receipt_sha256=q.sha(args.qualification_receipt),
                qualification_supervision_sha256=q.sha(args.qualification_supervision), qualification_log_sha256=logs)


def run(args):
    require(not args.output_dir.exists(), "fresh frontend output directory required")
    require(not args.output_dir.resolve().is_relative_to(ROOT), "frontend array caches must remain private outside public repository")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    record = dict(schema="evr0023.frontend.v1", status="started", eligible=False, stage="verification",
                  expected_commit=args.expected_commit, frontend_freeze=args.frontend_freeze,
                  plan_sha256=args.plan_sha256, input_receipt_sha256=args.input_receipt_sha256,
                  settings=dict(moge=q.MOGE_OPTIONS, superpoint=q.SP_OPTIONS, lightglue=q.LG_OPTIONS,
                                max_edge=512, extract_resize=None, device="cpu", dtype="float32", threads=4),
                  timings={}, warnings=[], offline_attempts=[], compatibility_probes=[], local_hub_calls=[])
    record["warnings_scope"] = "Explicit model-loading/inference capture contexts; import warnings remain in supervisor-bound raw stderr."
    def save():
        temp = args.output_dir / "receipt.pending"
        with temp.open("w") as stream:
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
        os.replace(temp, args.output_dir / "receipt.json")
    save()
    try:
        supervisor = acquisition.supervisor
        record["source_sha256"] = supervisor.verify_frozen(args.expected_commit,
            [Path(__file__), HERE / "range_inputs.py", HERE / "propose_plan.py", HERE / "acquire_operation.py",
             HERE / "supervise.py", args.protocol])
        qa = qualification_args(args)
        frozen = q.frozen_bindings(qa)
        files, _, wheels, bindings = q.verify_files(qa, frozen)
        record["frontend_bindings"] = dict(bindings, **verify_qualification(args, bindings, frozen))
        record["input_acquisition_bindings"] = acquisition.verify_completed_chain(args)
        plan, input_hashes = verify_inputs(args)
        record["input_sha256"] = input_hashes
        names, edges = validate_plan(plan)
        record["images"] = [dict(image_index=i, filename=name, status="not_attempted") for i, name in enumerate(names)]
        record["edges"] = [dict(edge_index=i, **edge, status="not_attempted") for i, edge in enumerate(edges)]
        record["timings"]["verification_seconds"] = time.perf_counter() - began
        record["stage"] = "model_loading"; save()
        backend = Backend(args, files, wheels, record)
        record["offline_guard_scope"] = "Python socket/DNS/process/input audit events; not an OS sandbox for arbitrary native syscalls"
        with q.capture_warnings(record):
            process(plan, args.inputs, args.output_dir, backend, record, save)
        record["stage"] = "postcheck"; save()
        require(not record["offline_attempts"], "offline operation attempted")
        _, final_hashes = verify_inputs(args)
        require(final_hashes == input_hashes, "frontend inputs changed during execution")
        require(acquisition.verify_completed_chain(args, verified_bindings=record["source_sha256"])
                == record["input_acquisition_bindings"],
                "acquisition admission chain changed during frontend execution")
        _, _, _, final_bindings = q.verify_files(qa, frozen)
        require(final_bindings == bindings, "frontend assets changed during execution")
        require(all(q.sha(ROOT / path) == digest for path, digest in {**frozen, **record["source_sha256"]}.items()), "frozen source changed")
        completed = all(row["status"] == "completed" for row in record["images"] + record["edges"])
        record.update(status="completed" if completed else "failed", eligible=completed, stage="completed")
    except BaseException as exc:
        record.update(status="failed", eligible=False, failed_stage=record["stage"],
                      error=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        record["wall_seconds"] = time.perf_counter() - began
        save()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "input-receipt", "input-operation", "input-supervision", "inputs", "output-dir", "protocol", "installation",
                 "installation-supervision", "acquisition-receipt", "assets-dir", "qualification-receipt",
                 "qualification-supervision"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("plan-sha256", "input-receipt-sha256", "expected-commit", "frontend-freeze"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--frontend-protocol", type=Path, default=ROOT / "protocols/0022-traceable-frontend-assets.md")
    parser.add_argument("--asset-manifest", type=Path, default=ROOT / "configs/0022-source-model-assets.json")
    parser.add_argument("--wheel-manifest", type=Path, default=ROOT / "results/0022-preparation/dependency-wheels.json")
    parser.add_argument("--wheel-assets-manifest", type=Path, default=ROOT / "configs/0022-wheel-assets.json")
    result = run(parser.parse_args())
    print(json.dumps(dict(status=result["status"], eligible=result["eligible"], wall_seconds=result["wall_seconds"])))
    return 0 if result["eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
