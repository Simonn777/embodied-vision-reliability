"""Development DAVIS-formula J/F; admission/GT-access authorization is caller-owned.

Reference: davisvideochallenge/davis2017-evaluation commit
ac7c43fca936f9722837b7fbd337d284ba37004b, metrics.py/evaluation.py/davis.py.
Reimplementation: original resolution, explicit object IDs, no object matching,
no union, no missing-frame padding. Semi-supervised aggregation excludes BOTH
endpoints. All-frame raw metrics remain visible. 255 is background as in the
official semi-supervised evaluation path (which passes void_pixels=None).
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REFERENCE = "ac7c43fca936f9722837b7fbd337d284ba37004b"
BOUNDARY_FRACTION = 0.008


def boundary_map(mask):
    """Official same-resolution half-pixel-towards-origin boundary convention."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or min(m.shape) < 1:
        raise ValueError("mask must be nonempty H,W")
    east = np.zeros_like(m)
    south = np.zeros_like(m)
    southeast = np.zeros_like(m)
    east[:, :-1] = m[:, 1:]
    south[:-1, :] = m[1:, :]
    southeast[:-1, :-1] = m[1:, 1:]
    b = (m ^ east) | (m ^ south) | (m ^ southeast)
    b[-1, :] = m[-1, :] ^ east[-1, :]
    b[:, -1] = m[:, -1] ^ south[:, -1]
    b[-1, -1] = False
    return b


def boundary_radius(shape):
    return int(math.ceil(BOUNDARY_FRACTION * math.hypot(*shape)))


def jf(prediction, truth):
    """Two boolean H,W object masks -> (J,F); no resizing or void suppression."""
    p, t = np.asarray(prediction), np.asarray(truth)
    if p.dtype != np.bool_ or t.dtype != np.bool_ or p.shape != t.shape:
        raise ValueError("J/F require identically shaped boolean masks")
    pb, tb = boundary_map(p), boundary_map(t)
    union = int(np.count_nonzero(p | t))
    j = float(np.count_nonzero(p & t) / union) if union else 1.0
    radius = boundary_radius(p.shape)
    axis = np.arange(-radius, radius + 1, dtype=np.int64)
    disk = (axis[:, None] ** 2 + axis[None, :] ** 2 <= radius ** 2).astype(np.uint8)
    # Same disk and zero exterior as the upstream skimage disk + cv2.dilate.
    pd = cv2.dilate(pb.astype(np.uint8), disk, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    td = cv2.dilate(tb.astype(np.uint8), disk, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    npred, ntruth = int(pb.sum()), int(tb.sum())
    if npred == 0 or ntruth == 0:
        f = 1.0 if npred == ntruth else 0.0
    else:
        precision = np.count_nonzero(pb & (td != 0)) / npred
        recall = np.count_nonzero(tb & (pd != 0)) / ntruth
        f = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return j, float(f)


def _descriptor(d):
    p = Path(d["path"])
    if not p.is_file() or p.stat().st_size != d["bytes"]:
        raise ValueError(f"file size mismatch: {p}")
    with p.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if digest != d["sha256"]:
        raise ValueError(f"file hash mismatch: {p}")
    return p


def _labels(path):
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode not in ("P", "L"):
            raise ValueError("annotations must be indexed or 8-bit grayscale PNG")
        a = np.array(image)
    if a.ndim != 2 or a.dtype != np.uint8:
        raise ValueError("annotation must be uint8 H,W")
    return a


def _prediction(path, ids, shape):
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {"masks", "object_ids"}:
            raise ValueError("NPZ must contain exactly masks and object_ids")
        masks, found = z["masks"], z["object_ids"]
    if (found.ndim != 1 or found.dtype.kind not in "iu" or
            found.tolist() != ids or masks.dtype != np.bool_ or
            masks.shape != (len(ids), *shape)):
        raise ValueError("prediction object IDs, boolean dtype, or original shape mismatch")
    if len(ids) > 1 and np.any(np.count_nonzero(masks, axis=0) > 1):
        raise ValueError("overlapping multi-object masks need a frozen upstream label decision")
    return masks


def _means(rows):
    n = len(rows)
    if not n:
        raise ValueError("no included frames")
    j, f = (math.fsum(r[k] for r in rows) / n for k in ("J", "F"))
    return {"J": j, "F": f, "JF": (j + f) / 2}


def evaluate_sequence(spec):
    """Full sequence manifest -> raw frames and per-object interior means.

    Required: sequence,parent_id,frame_ids,shape_hw,object_mode,predictions,
    annotations. target mode additionally target_id; all mode object_ids.
    predictions/annotations are full ordered descriptor lists {path,bytes,sha256}.
    This verifies bytes, not the parent's operation seals or split admission.
    """
    frames = spec["frame_ids"]
    shape = tuple(spec["shape_hw"])
    if (not isinstance(frames, list) or len(frames) < 3 or
            frames != [f"{i:05d}" for i in range(len(frames))] or
            len(shape) != 2 or min(shape) != 480 or
            any(type(x) is not int or x <= 0 for x in shape)):
        raise ValueError("need complete ordered frame stems and original 480p dimensions")
    if spec["object_mode"] == "target":
        ids = [spec["target_id"]]
    elif spec["object_mode"] == "all":
        ids = spec["object_ids"]
    else:
        raise ValueError("object_mode must explicitly be target or all; no union mode")
    if (not isinstance(ids, list) or not ids or ids != sorted(set(ids)) or
            any(type(x) is not int or not 0 < x < 255 for x in ids)):
        raise ValueError("invalid object ID declaration")
    if any(len(spec[k]) != len(frames) for k in ("predictions", "annotations")):
        raise ValueError("missing frame; successful subsets are not evaluated")
    # All encoded file hashes before the first annotation decode.
    pp = [_descriptor(d) for d in spec["predictions"]]
    gt = [_descriptor(d) for d in spec["annotations"]]
    records = {i: [] for i in ids}
    initial_ids = None
    for index, (frame, pred_path, gt_path) in enumerate(zip(frames, pp, gt)):
        masks = _prediction(pred_path, ids, shape)
        labels = _labels(gt_path)
        if labels.shape != shape:
            raise ValueError("annotation dimensions differ from original prediction dimensions")
        observed = set(int(i) for i in np.unique(labels) if 0 < i < 255)
        if index == 0:
            initial_ids = observed
            if not set(ids).issubset(observed):
                raise ValueError("declared target absent from initial supervision")
            if spec["object_mode"] == "all" and set(ids) != observed:
                raise ValueError("all mode omitted initial objects")
        elif spec["object_mode"] == "all" and not observed.issubset(initial_ids):
            raise ValueError("new undeclared object ID in semi-supervised annotation")
        for channel, object_id in enumerate(ids):
            j, f = jf(masks[channel], labels == object_id)
            records[object_id].append({"frame_id": frame, "index": index,
                "included": 0 < index < len(frames) - 1,
                "J": j, "F": f, "JF": (j + f) / 2,
                "void_pixels_as_background": int(np.count_nonzero(labels == 255))})
    for d in spec["predictions"] + spec["annotations"]:
        _descriptor(d)
    objects = [{"object_id": i, "frames": records[i],
                "mean": _means([r for r in records[i] if r["included"]])} for i in ids]
    return {"schema": "evr0030.development-jf.sequence.v1",
        "sequence": spec["sequence"], "parent_id": spec["parent_id"],
        "scope": "object-subset development J&F" if spec["object_mode"] == "target"
                 else "all-object development J&F",
        "official_benchmark_result": False, "reference_commit": REFERENCE,
        "endpoint_policy": "exclude_first_and_last", "void_policy": "official_background",
        "boundary_fraction": BOUNDARY_FRACTION, "boundary_radius_pixels": boundary_radius(shape),
        "shape_hw": list(shape), "frame_count": len(frames),
        "included_frame_count": len(frames) - 2, "initial_object_ids": sorted(initial_ids),
        "object_ids": ids, "objects": objects, "mean": _means([o["mean"] for o in objects])}


def aggregate(reports):
    """Equal parent mean; also expose equal-object mean, not pooled frame mean."""
    if not reports or len({r["sequence"] for r in reports}) != len(reports):
        raise ValueError("empty or duplicate sequence cohort")
    policies = {(r["scope"], r["endpoint_policy"], r["void_policy"], r["reference_commit"])
                for r in reports}
    if len(policies) != 1:
        raise ValueError("mixed evaluation contracts")
    parents = {}
    for r in reports:
        parents.setdefault(r["parent_id"], []).append(r)
    rows = [{"parent_id": p, "sequences": [r["sequence"] for r in values],
             "mean": _means([r["mean"] for r in values])} for p, values in sorted(parents.items())]
    return {"schema": "evr0030.development-jf.cohort.v1", "official_benchmark_result": False,
        "metric_scale": "[0,1]; multiply by 100 for percentage points",
        "aggregation": "equal objects within sequence; equal sequences within parent; equal parents",
        "parents": rows, "mean": _means([p["mean"] for p in rows]),
        "equal_object_mean": _means([o["mean"] for r in reports for o in r["objects"]]),
        "sequences": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cv2.setNumThreads(0)
    specs = json.loads(Path(args.manifest).read_text())["sequences"]
    report = aggregate([evaluate_sequence(s) for s in specs])
    with Path(args.output).open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
