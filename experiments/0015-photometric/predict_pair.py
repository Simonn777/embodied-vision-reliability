"""0015 image-only correspondence correction; no pose estimation or labels."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import zlib

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

SCHEMA = "evr0015.photometric.v1"
SOURCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SOURCE_DIR.parents[1]
BASELINE_DIR = SOURCE_DIR.parent / "0013-pose-headroom"
PROTOCOL_PATH = REPO_ROOT / "protocols/0015-photometric-information-development.md"
ARCHIVE_PATH = REPO_ROOT / "configs/0015/posebench-images-archive.json"
ACQUISITION_PATH = REPO_ROOT / "results/0015-admission/acquisition.json"
SETTINGS = {"maximum_queries": 256, "maximum_batch": 64,
    "cache_to_array_offset": -0.5, "array_to_cache_offset": 0.5,
    "pixel_conversion": "PIL.convert('L'), float64 / 255",
    "resize": False, "exif_transpose": False, "native_threads": 1,
    "failure_policy": "retain original target; all selected queries reported",
    "pose_estimation": False, "calibration": "original stored K unchanged"}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_dependencies():
    return (load_module("_evr0013_base_for0015_photo", BASELINE_DIR / "predict_pair.py"),
        load_module("_evr0013_seal_for0015_photo", BASELINE_DIR / "evaluate_sealed.py"),
        load_module("_evr0015_acquisition_for_photo", SOURCE_DIR / "acquire_images.py"),
        load_module("_evr0015_selector_for_photo", SOURCE_DIR / "select_queries.py"),
        load_module("_evr0015_kernel_for_photo", SOURCE_DIR / "refine_patch.py"))


def write_record(stream, record):
    encoded = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
    stream.seek(0)
    stream.write(encoded)
    stream.truncate()
    stream.flush()
    os.fsync(stream.fileno())


def selected_baseline(base, verifier, manifest_path, seal_path, index):
    manifest, seal, rows, mh, sh = verifier.verify_seal(manifest_path, seal_path)
    keys = manifest["selected_pair_keys"]
    if (manifest.get("schema") != "evr0012.development.v1"
            or manifest.get("coordinate_convention") != "raw_pixels"
            or manifest.get("development_scene") != "0015"
            or len(keys) != 16 or keys != sorted(set(keys))
            or any(not key.startswith("0015-") for key in keys)):
        raise ValueError("baseline manifest allocation differs")
    row, baseline = rows[index]
    if baseline.get("status") != "completed" or baseline.get("settings") != base.SETTINGS:
        raise ValueError("selected baseline must be completed with fixed settings")
    mask = baseline.get("returned_ransac_stats", {}).get("inliers")
    if (not isinstance(mask, list) or len(mask) != 10000
            or any(type(x) is not bool for x in mask) or sum(mask) < 6):
        raise ValueError("baseline mask must be 10000 booleans with at least six inliers")
    return manifest, baseline, base.np.asarray(mask, dtype=bool), {
        "key": keys[index], "manifest_sha256": mh, "baseline_seal_sha256": sh,
        "baseline_prediction_sha256": row["sha256"],
        "dataset_sha256_declared": manifest["dataset_sha256"],
        "mask_sha256": hashlib.sha256(bytes(mask)).hexdigest(), "kept_count": sum(mask),
        "mask_encoding": "10000 uint8 values in original correspondence order"}


def image_binding(base, verifier, acq, manifest, manifest_hash, receipt_path, key):
    """Validate every receipt entry; this function never reads an image file."""
    archive, ah = verifier.load_hashed_json(ARCHIVE_PATH)
    receipt, rh = verifier.load_hashed_json(receipt_path)
    if (manifest_hash != acq.EXPECTED_MANIFEST_SHA256
            or ah != acq.EXPECTED_ARCHIVE_MANIFEST_SHA256
            or rh != base.sha256(ACQUISITION_PATH)):
        raise ValueError("image inputs differ from pinned manifests or acquisition receipt")
    expected = acq.selected_members(manifest, archive)
    if (receipt.get("schema") != acq.SCHEMA or receipt.get("status") != "completed"
            or receipt.get("stage") != "completed" or receipt.get("selected_images") != 23
            or receipt.get("url") != acq.URL or receipt.get("archive_bytes") != acq.ARCHIVE_BYTES
            or receipt.get("manifest_sha256") != manifest_hash
            or receipt.get("expected_manifest_sha256") != manifest_hash
            or receipt.get("archive_manifest_sha256") != ah
            or receipt.get("expected_archive_manifest_sha256") != ah
            or receipt.get("source_sha256") != base.sha256(SOURCE_DIR / "acquire_images.py")):
        raise ValueError("image acquisition identity or completion mismatch")
    images = receipt.get("images", [])
    if (not isinstance(images, list) or len(images) != 23
            or [x.get("filename") for x in images] != [x["filename"] for x in expected]):
        raise ValueError("receipt must contain exactly the 23 frozen filenames in order")
    for item, entry in zip(images, expected):
        if any(item.get(k) != v for k, v in entry.items()):
            raise ValueError("receipt CRC, sizes or archive member identity differs")
        w, h = item.get("width"), item.get("height")
        if (type(w) is not int or type(h) is not int or not 0 < w <= 16384
                or not 0 < h <= 16384 or w*h > acq.MAX_IMAGE_PIXELS
                or item.get("channels") not in (1, 3, 4) or item.get("precision_bits") != 8
                or item.get("pixel_decoding_performed") is not False
                or re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256"))) is None):
            raise ValueError("invalid image hash or dimension receipt")
    if receipt.get("output_bytes") != sum(x["uncompressed_bytes"] for x in expected):
        raise ValueError("acquisition output-byte total differs")
    names = re.fullmatch(r"0015-([^/\\]+\.jpg)_0015-([^/\\]+\.jpg)", key)
    if names is None:
        raise ValueError("selected pair does not encode safe development image names")
    lookup = {x["filename"]: x for x in images}
    selected = [lookup["images/0015/" + name] for name in names.groups()]
    return selected, {"image_acquisition_receipt_sha256": rh,
        "image_acquisition_source_sha256": receipt["source_sha256"],
        "archive_manifest_sha256": ah, "images": selected}


def decode_image(base, acq, root, entry):
    """Read exactly one allowlisted JPEG, verifying bytes before pixel decoding."""
    from PIL import Image, ImageFile
    if ImageFile.LOAD_TRUNCATED_IMAGES:
        raise ValueError("Pillow truncated-image decoding must remain disabled")
    root = root.resolve()
    path = root / entry["filename"]
    if path.resolve() != path or root not in path.parents:
        raise ValueError("image symlink or path escape is not permitted")
    expected = entry["uncompressed_bytes"]
    if path.stat().st_size != expected or expected > acq.MAX_JPEG_BYTES:
        raise ValueError("selected JPEG size mismatch")
    with path.open("rb") as stream:
        data = stream.read(expected + 1)
    if (len(data) != expected or hashlib.sha256(data).hexdigest() != entry["sha256"]
            or "%08x" % (zlib.crc32(data) & 0xffffffff) != entry["crc32_hex"]):
        raise ValueError("selected JPEG size, SHA256 or CRC32 mismatch")
    header = acq.jpeg_header(data)
    if any(entry.get(k) != v for k, v in header.items()):
        raise ValueError("selected JPEG dimension header differs from receipt")
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "JPEG" or image.size != (entry["width"], entry["height"]):
            raise ValueError("Pillow JPEG dimensions differ from receipt")
        image.load()
        result = base.np.asarray(image.convert("L"), dtype=base.np.float64) / 255.0
    if result.shape != (entry["height"], entry["width"]) or not base.np.isfinite(result).all():
        raise ValueError("decoded image shape or finite check failed")
    return result


def initialize_queries(base, selector, x1, x2, mask, width, height):
    selected = base.json_safe(selector.select_and_initialize(x1, x2, mask, width, height))
    if not isinstance(selected, list) or len(selected) > SETTINGS["maximum_queries"]:
        raise ValueError("selector exceeded query limit")
    seen, queries = set(), []
    for item in selected:
        i = item.get("index")
        if type(i) is not int or not 0 <= i < len(mask) or not mask[i] or i in seen:
            raise ValueError("selector returned a duplicate or non-inlier index")
        seen.add(i)
        good = item.get("status") == "initialized"
        if not good and item.get("status") != "initialization_failed":
            raise ValueError("unexpected initializer status")
        if good:
            base.finite_array(item.get("A0"), (2, 2), "initial affine")
        queries.append({"index": i, "cell": item["cell"], "initialization": item,
            "original_source_cache": x1[i].tolist(), "original_target_cache": x2[i].tolist(),
            "target_cache": x2[i].tolist(), "delta_cache": [0.0, 0.0],
            "status": "not_attempted" if good else "initialization_failed",
            "admitted": False, "target_coordinate_changed": False,
            "reason": None if good else item.get("reason"), "photometric": None})
    return queries


def apply_batch(base, kernel, queries, ids, image1, image2):
    np = base.np
    p = np.asarray([queries[i]["original_source_cache"] for i in ids]) - 0.5
    q = np.asarray([queries[i]["original_target_cache"] for i in ids]) - 0.5
    A = np.asarray([queries[i]["initialization"]["A0"] for i in ids])
    result = base.json_safe(kernel.refine_batch(image1, image2, p.copy(), q.copy(), A.copy()))
    if (result.get("schema") != "evr0015.patch-refinement.v1"
            or result.get("settings") != kernel.SETTINGS or len(result.get("patches", [])) != len(ids)):
        raise ValueError("kernel schema, settings or batch length differs")
    updated = []
    for j, i in enumerate(ids):
        patch = result["patches"][j]
        if (patch.get("patch_index") != j or patch.get("source") != p[j].tolist()
                or patch.get("original_target") != q[j].tolist()
                or patch.get("original_A") != A[j].tolist()
                or type(patch.get("admitted")) is not bool):
            raise ValueError("kernel output identity differs from batch input")
        # Only the kernel's declared metadata schema is exportable, never image/residual arrays.
        allowed = {"patch_index", "source", "original_target", "original_A", "initial_cost",
            "final_cost", "admitted", "changed", "status", "reason", "termination",
            "accepted_steps", "attempts", "final", "initial_diagnostic", "final_diagnostic",
            "batch_wall_elapsed_at_finalization"}
        if set(patch) - allowed:
            raise ValueError("unexpected kernel output fields")
        row = dict(queries[i], photometric=patch, admitted=patch["admitted"],
            status="admitted" if patch["admitted"] else "fit_failed", reason=patch.get("reason"))
        if patch["admitted"]:
            final = patch.get("final", {})
            delta = base.finite_array(final.get("delta"), (2,), "fitted delta")
            target = base.finite_array(final.get("target"), (2,), "fitted target")
            if (not np.array_equal(target, q[j] + delta)
                    or math.hypot(*delta.tolist()) > kernel.SETTINGS["maximum_delta"]):
                raise ValueError("fitted target or displacement differs from kernel convention")
            target_cache = target + 0.5
            row.update(target_cache=target_cache.tolist(),
                delta_cache=(target_cache - np.asarray(row["original_target_cache"])).tolist(),
                target_coordinate_changed=bool(np.any(target_cache != row["original_target_cache"])))
        updated.append((i, row))
    # Validate the whole batch before changing any previously persisted query row.
    for i, row in updated:
        queries[i] = row
    return {k: v for k, v in result.items() if k != "patches"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "manifest", "baseline-seal", "images-root", "image-receipt", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--pair-index", type=int, choices=range(16), required=True)
    args = parser.parse_args(argv)
    try:
        stream = args.output.open("x+")
    except FileExistsError:
        print(json.dumps({"status": "refused_existing_output", "pair_index": args.pair_index}))
        return 2
    started = time.perf_counter()
    record = {"schema": SCHEMA, "status": "started", "phase": "binding",
        "pair_index": args.pair_index, "utc": datetime.now(timezone.utc).isoformat(),
        "settings": SETTINGS, "timing_seconds": {}, "queries": [], "batches": []}
    failure_kind = "binding"
    with stream:
        write_record(stream, record)  # Before imports, manifest, receipt or data access.
        try:
            base, verifier, acq, selector, kernel = load_dependencies()
            record.update(source_sha256=base.sha256(__file__),
                kernel_sha256=base.sha256(SOURCE_DIR / "refine_patch.py"),
                selector_sha256=base.sha256(SOURCE_DIR / "select_queries.py"),
                protocol_sha256=base.sha256(PROTOCOL_PATH),
                kernel_settings=base.json_safe(kernel.SETTINGS),
                selector_settings=base.json_safe(selector.SETTINGS))
            manifest, baseline, mask, identity = selected_baseline(base, verifier,
                args.manifest, args.baseline_seal, args.pair_index)
            record.update(identity)
            entries, binding = image_binding(base, verifier, acq, manifest,
                identity["manifest_sha256"], args.image_receipt, identity["key"])
            record.update(binding)
            if args.data.stat().st_size != manifest["dataset_bytes"]:
                raise ValueError("dataset byte size differs from frozen declaration")
            import h5py
            import poselib
            import PIL
            library = {**base.library_metadata(poselib), "h5py_version": h5py.__version__}
            if library != baseline.get("library"):
                raise ValueError("library differs from sealed baseline provenance")
            record.update(library=library, pillow={"version": PIL.__version__,
                "decode": "JPEG without resize or EXIF transpose; truncated decoding disabled"})
            failure_kind, record["phase"] = "input", "coordinates"
            write_record(stream, record)
            stamp = time.perf_counter()
            try:
                with h5py.File(args.data, "r") as handle:
                    x1, x2, calibration = base.read_pair(handle, identity["key"])
                if base.json_safe(calibration) != baseline.get("calibration"):
                    raise ValueError("calibration differs from sealed baseline")
                record["calibration"] = base.json_safe(calibration)
            finally:
                record["timing_seconds"]["coordinates"] = time.perf_counter() - stamp
            record["phase"] = "images"
            write_record(stream, record)
            stamp = time.perf_counter()
            try:
                image1, image2 = [decode_image(base, acq, args.images_root, e) for e in entries]
            finally:
                record["timing_seconds"]["images"] = time.perf_counter() - stamp
            failure_kind, record["phase"] = "selector", "selection"
            write_record(stream, record)
            stamp = time.perf_counter()
            try:
                record["queries"] = initialize_queries(base, selector, x1, x2, mask,
                    entries[0]["width"], entries[0]["height"])
            finally:
                record["timing_seconds"]["selection"] = time.perf_counter() - stamp
            record["selected_indices"] = [x["index"] for x in record["queries"]]
            record["selected_count"] = len(record["queries"])
            ids = [i for i, x in enumerate(record["queries"]) if x["status"] == "not_attempted"]
            record["initialized_count"] = len(ids)
            failure_kind = "photometric"
            for batch, offset in enumerate(range(0, len(ids), SETTINGS["maximum_batch"])):
                subset = ids[offset:offset + SETTINGS["maximum_batch"]]
                record["phase"] = "photometric_batch_%03d" % batch
                write_record(stream, record)
                stamp = time.perf_counter()
                try:
                    info = apply_batch(base, kernel, record["queries"], subset, image1, image2)
                    record["batches"].append({"batch": batch, "query_positions": subset, **info})
                finally:
                    record["timing_seconds"][record["phase"]] = time.perf_counter() - stamp
                write_record(stream, record)
            record["admitted_count"] = sum(x["admitted"] for x in record["queries"])
            record["target_coordinate_changed_count"] = sum(
                x["target_coordinate_changed"] for x in record["queries"])
            record.update(status="completed", phase="completed")
        except BaseException as error:
            message = str(error)
            for name, path in vars(args).items():
                if isinstance(path, Path):
                    message = message.replace(str(path), "<" + name + ">")
            record.update(status="error", failure_kind=failure_kind,
                error={"type": type(error).__name__, "message": message})
            if not isinstance(error, Exception):
                record["timing_seconds"]["total"] = time.perf_counter() - started
                write_record(stream, record)
                raise
        record["timing_seconds"]["total"] = time.perf_counter() - started
        write_record(stream, record)
    print(json.dumps({"schema": SCHEMA, "status": record["status"],
        "pair_index": args.pair_index, "selected": record.get("selected_count"),
        "admitted": record.get("admitted_count")}))
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
