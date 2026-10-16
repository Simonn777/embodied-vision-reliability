"""Bounded development-only acquisition and streaming adapter (no model vendoring)."""
from __future__ import annotations

import base64
import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import pickle
import pickletools
import time
import urllib.request
import zipfile

import numpy as np

UPSTREAM = "730cda1c730877cfedbe01bf87fb1cadb78a565d"
OBJECTS = {
    "weights.pt": ("bootstap/causal_bootstapir_checkpoint.pt", 218887028,
                   "1732921823937966", "1C+KZQL2VuSdEJGxGwEHSA=="),
    "rgb.zip": ("tapvid_rgb_stacking.zip", 187291581,
                "1676313001737086", "RuxWCiOHqBmkhWpKxbLifQ=="),
}
MAX_BYTES = 10 * 1024**3
UPSTREAM_HASHES = {
    "tapnet/torch/tapir_model.py": "87c05fb17d48d0f4353a00b67c9d40d3a264e13e43d88e3f7b5829119845e3ba",
    "tapnet/torch/nets.py": "4961a8e4882e8c9ac219c4714f6f2c811cc1ab665a439d4d2e79942002a9517c",
    "tapnet/torch/utils.py": "a4267b3a6bd8ecf2dfe5effacab9f7a43664dc5d08966c98f6a0cc68f5ccedd3",
}


def verify_upstream(source):
    for path, expected in UPSTREAM_HASHES.items():
        if sha256(Path(source) / path) != expected:
            raise ValueError("Pinned upstream source digest mismatch: " + path)


def verify_module_paths(source):
    """Resolve each actual import target before executing model modules."""
    source = Path(source).resolve()
    verify_upstream(source)
    paths = {}
    for path in UPSTREAM_HASHES:
        module = path.removesuffix(".py").replace("/", ".")
        spec = importlib.util.find_spec(module)
        actual = Path(spec.origin).resolve() if spec and spec.origin else None
        if actual != source / path:
            raise ValueError("Unexpected model module path: " + module)
        paths[module] = {"relative_path": path, "sha256": sha256(actual)}
    return paths


def verify_development(npz_path, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("count") != 50 or manifest.get("development_indices") != [0]:
        raise ValueError("Manifest is not audited development index zero")
    expected = {
        "video": "0ac2e86352c9d58deb2eb7713fa0757891922c434cd157750ef86343a71effc2",
        "points": "a7e4b1b69d96857285d54e6460d959eff13e4b9dc34e27f7e41b8b64136871f7",
        "occluded": "162ebe9043605cdca3caedae67d0ce1562cf795e979cd18cef1eadf2c27a2ec5",
    }
    data = np.load(npz_path, allow_pickle=False)
    arrays = {k: data[k] for k in expected}
    validate_example(arrays)
    for key, digest in expected.items():
        if (manifest.get(key + "_sha256") != digest or
                hashlib.sha256(arrays[key].tobytes()).hexdigest() != digest):
            raise ValueError("Development data/manifest digest mismatch: " + key)
    return arrays, {"npz_sha256": sha256(npz_path),
                    "manifest_sha256": sha256(manifest_path),
                    "array_sha256": expected, "development_indices": [0]}


def validation_status(checks):
    return "completed" if checks and all(v["passed"] for v in checks.values()) else "validation_failed"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def tree_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())


def download_object(cache, name):
    """Exact official object, length/generation and full MD5 check; never resume."""
    cache = Path(cache)
    path, length, generation, md5 = OBJECTS[name]
    url = "https://storage.googleapis.com/dm-tapnet/" + path
    target = cache / name
    if not target.exists():
        if tree_bytes(cache) + length > MAX_BYTES:
            raise ValueError("Cache budget exceeded before download")
        with urllib.request.urlopen(url, timeout=60) as response:
            if (int(response.headers.get("Content-Length", -1)) != length or
                    response.headers.get("x-goog-generation") != generation):
                raise ValueError("Official object length/generation changed")
            temporary = target.with_suffix(".partial")
            try:
                with temporary.open("wb") as f:
                    total = 0
                    for block in iter(lambda: response.read(1024**2), b""):
                        total += len(block)
                        if total > length:
                            raise ValueError("Response exceeds declared length")
                        f.write(block)
                if total != length:
                    raise ValueError("Truncated response")
                temporary.rename(target)
            finally:
                temporary.unlink(missing_ok=True)
    if target.stat().st_size != length:
        raise ValueError("Cached object length mismatch")
    digest = hashlib.md5()
    with target.open("rb") as f:
        for block in iter(lambda: f.read(1024**2), b""):
            digest.update(block)
    if base64.b64encode(digest.digest()).decode() != md5:
        raise ValueError("Official object MD5 mismatch")
    return {"url": url, "bytes": length, "generation": generation,
            "official_md5_base64": md5, "sha256": sha256(target)}


class NumpyOnlyUnpickler(pickle.Unpickler):
    """Allow NumPy numeric reconstruction only; no arbitrary imports/callables.

    Run only in a credential-free, resource-limited process. This allowlist does
    not replace the official-byte digest check or prevent all allocation DoS.
    """
    def find_class(self, module, name):
        if module in ("numpy.core.multiarray", "numpy._core.multiarray"):
            if name == "_reconstruct":
                return np._core.multiarray._reconstruct
            if name == "scalar":
                return np._core.multiarray.scalar
        if module == "numpy" and name in ("ndarray", "dtype"):
            return getattr(np, name)
        raise pickle.UnpicklingError(f"Forbidden global: {module}.{name}")

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("Persistent references forbidden")


def inspect_pickle(stream):
    """Inventory opcodes without executing; unpickler checks STACK_GLOBAL too."""
    counts, globals_seen = {}, []
    for opcode, argument, _ in pickletools.genops(stream):
        counts[opcode.name] = counts.get(opcode.name, 0) + 1
        if opcode.name == "GLOBAL":
            globals_seen.append(argument)
        if opcode.name in ("EXT1", "EXT2", "EXT4", "PERSID", "BINPERSID"):
            raise pickle.UnpicklingError("Extension/persistent opcode forbidden")
    return {"opcodes": counts, "literal_globals": sorted(set(globals_seen)),
            "stack_globals_checked_by": "NumpyOnlyUnpickler.find_class"}


def validate_example(example):
    if not isinstance(example, dict) or not {"video", "points", "occluded"} <= example.keys():
        raise ValueError("Unexpected example schema")
    video, points, occluded = (example[k] for k in ("video", "points", "occluded"))
    if video.dtype != np.uint8 or video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError("Expected uint8 THWC RGB")
    if points.ndim != 3 or points.shape[-1] != 2 or points.dtype.kind != "f":
        raise ValueError("Expected floating normalized NT2 points")
    if occluded.dtype != np.bool_ or occluded.shape != points.shape[:2]:
        raise ValueError("Invalid visibility")
    if points.shape[1] != video.shape[0] or not np.isfinite(points).all():
        raise ValueError("Invalid time axis or coordinate values")
    return video, points, occluded


def decode_development(cache, output):
    """Decode official archive; access only predeclared entire video index zero.

    The pickle container is materialized, including other videos' arrays. Only
    index zero is inspected/exported; others are never visualized/inferred.
    """
    cache = Path(cache)
    with zipfile.ZipFile(cache / "rgb.zip") as z:
        members = z.infolist()
        for info in members:
            p = Path(info.filename)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError("Unsafe ZIP member")
        if sum(x.file_size for x in members) > 4 * 1024**3:
            raise ValueError("Uncompressed archive exceeds 4 GiB gate")
        candidates = [x for x in members if x.filename.endswith(".pkl")]
        if len(candidates) != 1:
            raise ValueError("Expected exactly one pickle")
        with z.open(candidates[0]) as f:
            audit = inspect_pickle(f)
        with z.open(candidates[0]) as f:
            data = NumpyOnlyUnpickler(f).load()
        if not isinstance(data, list) or len(data) != 50:
            raise ValueError("Unexpected archive list")
        video, points, occluded = validate_example(data[0])
        np.savez(cache / "development-000.npz", video=video, points=points,
                 occluded=occluded)
        receipt = {"count": len(data), "development_indices": [0],
                   "others": "1..49 unassigned; container decoded, no inspection/inference",
                   "scene_independence": "unknown", "pickle_audit": audit,
                   "zip_members": [{"name": x.filename, "bytes": x.file_size}
                                   for x in members],
                   "video_shape": list(video.shape), "points_shape": list(points.shape),
                   "video_sha256": hashlib.sha256(video.tobytes()).hexdigest(),
                   "points_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
                   "occluded_sha256": hashlib.sha256(occluded.tobytes()).hexdigest()}
    Path(output).write_text(json.dumps(receipt, indent=2) + "\n")


def pixel_query(points_xy, arrival, height, width):
    xy = np.asarray(points_xy, dtype=np.float32)
    if (xy.shape != (2,) or not np.isfinite(xy).all() or
            np.any(xy < 0) or np.any(xy > 1) or
            not isinstance(arrival, (int, np.integer)) or isinstance(arrival, (bool, np.bool_)) or arrival < 0):
        raise ValueError("Invalid query")
    # Local frame index is zero because descriptor extraction sees one frame.
    return np.array([0.0, xy[1] * height, xy[0] * width], np.float32)


def validate_run(frames, queries, chunk_size):
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("query_chunk_size must be a positive integer")
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (256, 256, 3):
        raise ValueError("Development frames must be 256px RGB")
    for q in queries:
        if not isinstance(q["arrival"], (int, np.integer)) or isinstance(q["arrival"], (bool, np.bool_)):
            raise ValueError("Arrival must be an integer, never bool")
        if not 0 <= q["arrival"] < len(frames):
            raise ValueError("Arrival outside clip")
        pixel_query(q["xy"], q["arrival"], 256, 256)


def stream(model, frames, queries, chunk_size=8, on_frame=None):
    """One state per arrival group; no future frame/query descriptors supplied."""
    import torch
    validate_run(frames, queries, chunk_size)
    out = np.full((len(frames), len(queries), 4), np.nan, np.float32)
    groups, timings = [], []
    with torch.inference_mode():
        for t, image in enumerate(frames):
            started = time.perf_counter()
            arriving = [i for i, q in enumerate(queries) if q["arrival"] == t]
            if not arriving and not groups:
                timings.append(time.perf_counter() - started)
                if on_frame is not None:
                    on_frame(t, out[t].copy(), timings[-1])
                continue
            tensor = torch.from_numpy(image.copy()).float()[None, None] / 255 * 2 - 1
            grids = model.get_feature_grids(tensor, is_training=False)
            if arriving:
                local = np.stack([pixel_query(queries[i]["xy"], t, 256, 256) for i in arriving])
                query_feats = model.get_query_features(tensor, is_training=False,
                    query_points=torch.from_numpy(local)[None], feature_grids=grids)
                state = model.construct_initial_causal_state(
                    len(arriving), len(query_feats.resolutions) - 1)
                groups.append([arriving, query_feats, state])
            for group in groups:
                indices, query_feats, state = group
                result = model.estimate_trajectories((256, 256), is_training=False,
                    feature_grids=grids, query_features=query_feats,
                    query_points_in_video=None, query_chunk_size=chunk_size,
                    causal_context=state, get_causal_context=True)
                group[2] = result["causal_context"]
                coords = result["tracks"][-1][0, :, 0].cpu().numpy()
                occ = result["occlusion"][-1][0, :, 0].cpu().numpy()
                dist = result["expected_dist"][-1][0, :, 0].cpu().numpy()
                values = np.column_stack((coords, occ, dist))
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite active output")
                out[t, indices] = values
            timings.append(time.perf_counter() - started)
            if on_frame is not None:
                on_frame(t, out[t].copy(), timings[-1])
    expected = np.arange(len(frames))[:, None] >= np.array([q["arrival"] for q in queries])[None, :]
    if not np.array_equal(np.isfinite(out).all(axis=-1), expected):
        raise ValueError("Output active mask mismatch")
    return out, timings


def compare(a, b, atol=1e-4):
    if a.shape != b.shape or not np.array_equal(np.isfinite(a), np.isfinite(b)):
        return {"passed": False, "reason": "shape or active-mask mismatch", "atol": atol}
    finite = np.isfinite(a)
    diff = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
    return {"passed": diff <= atol, "max_abs_diff": diff, "atol": atol,
            "rtol": 0.0, "finite_values": int(finite.sum())}
