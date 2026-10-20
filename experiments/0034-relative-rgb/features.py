"""Pure, label-free feature construction for the fixed 0034 screen.

Inputs are already admitted original-resolution observations. This module does
not load files, decode labels, resize RGB, fit models, or read native features.
"""

from functools import lru_cache

import cv2
import numpy as np


BAND_RADIUS = 8
REFERENCE_RADIUS = 32
REFERENCE_STRIDE = 4
REFERENCE_CAP = 32
MIN_REFERENCES = 4
SAMPLE_CAP = 1024
TILE_SIZE = 512
EVIDENCE_EPSILON = 1e-6
AXIAL_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))
GRAM_PAIRS = tuple((i, j) for i in range(4) for j in range(i, 4))
FLOAT_WIDTHS = {
    "c": 3,
    "absolute_prefix": 18,
    "proto": 5,
    "nn": 5,
    "ordinary": 17,
    "sym": 14,
    "sym_coarse": 14,
    "bilateral": 4,
}


def _validate(rgb, logits, native, coarse_rgb):
    for name, value in (("rgb", rgb), ("logits", logits),
                        ("native", native), ("coarse_rgb", coarse_rgb)):
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError("rgb must have nonempty H,W,3 shape")
    shape = rgb.shape[:2]
    if logits.shape != shape or native.shape != shape or coarse_rgb.shape != rgb.shape:
        raise ValueError("all observation shapes must match")
    if native.dtype != np.bool_:
        raise TypeError("native must be boolean")
    for name, value in (("rgb", rgb), ("coarse_rgb", coarse_rgb), ("logits", logits)):
        if value.dtype != np.float32:
            raise TypeError(f"{name} must be float32")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains nonfinite values")
    for name, value in (("rgb", rgb), ("coarse_rgb", coarse_rgb)):
        if np.any(value < 0) or np.any(value > 1):
            raise ValueError(f"{name} must be in [0,1]")
    return shape


def _empty_arrays():
    arrays = {name: np.empty((0, width), dtype=np.float32)
              for name, width in FLOAT_WIDTHS.items()}
    arrays.update(coords=np.empty((0, 2), dtype=np.int64),
                  e=np.empty(0, dtype=np.float32),
                  e_coarse=np.empty(0, dtype=np.float32),
                  sample_indices=np.empty(0, dtype=np.int64))
    return arrays


def _summary(values):
    values = np.asarray(values, dtype=np.int64)
    if values.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "histogram": {}}
    keys, counts = np.unique(values, return_counts=True)
    return {
        "count": int(values.size),
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": float(values.mean(dtype=np.float64)),
        "histogram": {str(int(k)): int(n) for k, n in zip(keys, counts)},
    }


def _uniform_sample(count):
    """Exact integer round-half-up ranks, including endpoints when count > 1."""
    if not isinstance(count, (int, np.integer)) or isinstance(count, (bool, np.bool_)):
        raise TypeError("count must be an integer")
    if count < 0:
        raise ValueError("count must be nonnegative")
    size = min(int(count), SAMPLE_CAP)
    if size == 0:
        return np.empty(0, dtype=np.int64)
    if size == 1:
        return np.array([0], dtype=np.int64)
    k = np.arange(size, dtype=np.int64)
    denominator = size - 1
    return (2 * k * (int(count) - 1) + denominator) // (2 * denominator)


def _native_probability(logits):
    """Stable sigmoid and sign-symmetric uncertainty, accumulated in float64."""
    values = logits.astype(np.float64)
    magnitude = np.abs(values)
    decay = np.zeros_like(values)
    # Beyond this point exp(-|l|) cannot affect a returned float32 feature.
    # Avoid evaluating an underflowing exponential, rather than hiding warnings.
    ordinary = magnitude <= -np.log(np.finfo(np.float64).tiny)
    decay[ordinary] = np.exp(-magnitude[ordinary])
    probability = np.where(values >= 0, 1.0 / (1.0 + decay), decay / (1.0 + decay))
    uncertainty = 4.0 * decay / np.square(1.0 + decay)
    return probability, uncertainty


def _opposite_band(native):
    """Caller excludes the empty/full cases before invoking the native EDT."""
    fg_distance = cv2.distanceTransform(
        native.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    bg_distance = cv2.distanceTransform(
        (~native).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    distance = np.where(native, fg_distance, bg_distance)
    if not np.isfinite(distance).all():
        raise FloatingPointError("nonfinite opposite-class distance")
    return distance <= BAND_RADIUS


@lru_cache(maxsize=16)
def _grid_offsets(x_phase, y_phase):
    """Distance-ordered grid offsets; (dy,dx) breaks spatial-distance ties."""
    offsets = []
    for gy in range(-8, 9):
        dy = REFERENCE_STRIDE * gy - y_phase
        for gx in range(-8, 9):
            dx = REFERENCE_STRIDE * gx - x_phase
            if abs(dx) <= REFERENCE_RADIUS and abs(dy) <= REFERENCE_RADIUS:
                offsets.append((dx * dx + dy * dy, dy, dx))
    offsets.sort()
    result = np.array([(dx, dy) for _, dy, dx in offsets], dtype=np.int64)
    result.setflags(write=False)
    return result


def _reference_indices(coords, native, band):
    """Return capped flat indices (-1 padding) and uncapped local counts.

    Temporary tables are limited to the caller's tile times 289 offsets.
    Grid coordinates are generated exactly; out-of-image entries are discarded.
    """
    height, width = native.shape
    total = len(coords)
    indices = {c: np.full((total, REFERENCE_CAP), -1, dtype=np.int64)
               for c in ("fg", "bg")}
    counts = {c: np.zeros(total, dtype=np.int64) for c in ("fg", "bg")}
    flat_native = native.reshape(-1)
    flat_outside = (~band).reshape(-1)
    for y_phase in range(REFERENCE_STRIDE):
        for x_phase in range(REFERENCE_STRIDE):
            rows = np.flatnonzero(
                (coords[:, 0] % REFERENCE_STRIDE == x_phase)
                & (coords[:, 1] % REFERENCE_STRIDE == y_phase)
            )
            if rows.size == 0:
                continue
            offsets = _grid_offsets(x_phase, y_phase)
            xx = coords[rows, 0, None] + offsets[None, :, 0]
            yy = coords[rows, 1, None] + offsets[None, :, 1]
            in_bounds = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
            # Zero is an indexing placeholder only. Its out-of-bounds entries
            # remain false in allowed and can never become reference samples.
            safe_index = np.where(in_bounds, yy * width + xx, 0)
            allowed = in_bounds & flat_outside[safe_index]
            for name, class_mask in (
                ("fg", flat_native[safe_index]),
                ("bg", ~flat_native[safe_index]),
            ):
                accepted = allowed & class_mask
                counts[name][rows] = accepted.sum(axis=1, dtype=np.int64)
                ordinal = np.cumsum(accepted, axis=1, dtype=np.int64)
                chosen = accepted & (ordinal <= REFERENCE_CAP)
                local_row, local_column = np.nonzero(chosen)
                slot = ordinal[local_row, local_column] - 1
                indices[name][rows[local_row], slot] = safe_index[local_row, local_column]
    return indices, counts


def _appearance(rgb, coords, reference_indices):
    """All callers provide rows with at least four references in both classes."""
    height, width, _ = rgb.shape
    del height
    flat_rgb = rgb.reshape(-1, 3)
    center_index = coords[:, 1] * width + coords[:, 0]
    center = flat_rgb[center_index].astype(np.float64)
    prototype = []
    nearest = []
    for name in ("fg", "bg"):
        ids = reference_indices[name]
        present = ids >= 0
        number = present.sum(axis=1, dtype=np.int64)
        if np.any(number < MIN_REFERENCES):
            raise ValueError("appearance received an insufficient support row")
        colors = flat_rgb[np.maximum(ids, 0)].astype(np.float64)
        means = np.where(present[:, :, None], colors, 0.0).sum(axis=1) / number[:, None]
        prototype.append(np.square(center - means).sum(axis=1) / 3.0)
        distances = np.square(center[:, None, :] - colors).sum(axis=2) / 3.0
        nearest.append(np.where(present, distances, np.inf).min(axis=1))
    prototype = np.stack(prototype, axis=1)
    nearest = np.stack(nearest, axis=1)
    stencil = [center]
    for dx, dy in AXIAL_OFFSETS:
        ids = (coords[:, 1] + dy) * width + coords[:, 0] + dx
        stencil.append(flat_rgb[ids].astype(np.float64))
    differences = [neighbor - center for neighbor in stencil[1:]]
    gram = np.stack(
        [(differences[i] * differences[j]).sum(axis=1) / 3.0
         for i, j in GRAM_PAIRS],
        axis=1,
    )
    evidence = (nearest[:, 1] - nearest[:, 0]) / (
        nearest.sum(axis=1) + EVIDENCE_EPSILON
    )
    return prototype, nearest, gram, evidence, np.concatenate(stencil, axis=1)


def _symmetric(uncertainty, nearest, gram):
    return np.column_stack((
        np.ones(len(nearest), dtype=np.float64),
        uncertainty,
        nearest.sum(axis=1),
        np.abs(nearest[:, 1] - nearest[:, 0]),
        gram,
    ))


def _bilateral(coords, rgb, probability):
    height, width, _ = rgb.shape
    offsets = np.array(
        [(dx, dy) for dy in range(-3, 4) for dx in range(-3, 4)
         if dx != 0 or dy != 0], dtype=np.int64,
    )
    xx = coords[:, 0, None] + offsets[None, :, 0]
    yy = coords[:, 1, None] + offsets[None, :, 1]
    inside = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
    ids = np.where(inside, yy * width + xx, 0)
    center_ids = coords[:, 1] * width + coords[:, 0]
    colors = rgb.reshape(-1, 3)
    center = colors[center_ids].astype(np.float64)
    color_distance = np.square(center[:, None] - colors[ids]).sum(axis=2)
    spatial_distance = np.square(offsets).sum(axis=1).astype(np.float64)
    weights = np.exp(
        -spatial_distance[None] / (2.0 * 2.0**2)
        - color_distance / (2.0 * 0.1**2)
    )
    weights *= inside
    denominator = weights.sum(axis=1)
    if not np.isfinite(denominator).all() or np.any(denominator <= 0):
        raise FloatingPointError("invalid bilateral denominator")
    flat_probability = probability.reshape(-1)
    return ((weights * flat_probability[ids]).sum(axis=1) / denominator
            - flat_probability[center_ids])


def construct(rgb, logits, native, coarse_rgb):
    """Return (arrays, JSON-compatible metadata), without reading any labels.

    coords is int64 [Q,2] (x,y), in row-major (y,x) order. Floating arrays are
    float32. sample_indices indexes coords, not the image's flattened pixels.
    """
    height, width = _validate(rgb, logits, native, coarse_rgb)
    foreground_count = int(native.sum(dtype=np.int64))
    metadata = {
        "schema": "evr0034.features.v1",
        "status": "completed",
        "shape": [int(height), int(width)],
        "coordinate_order": "xy_row_major",
        "native_foreground_pixels": foreground_count,
        "candidate_band_count": 0,
        "inbounds_stencil_count": 0,
        "eligible_before_minimum_count": 0,
        "valid_query_count": 0,
        "sample_count": 0,
        "native_noop_reason": None,
        "failure_counts": {
            "out_of_bounds_stencil": 0,
            "foreground_support_lt4": 0,
            "background_support_lt4": 0,
            "either_support_lt4": 0,
            "both_support_lt4": 0,
            "minimum_query_rule_discarded": 0,
        },
        "support_counts": {
            name: _summary(np.empty(0, dtype=np.int64))
            for name in (
                "uncapped_fg_checked", "uncapped_bg_checked",
                "selected_fg_checked", "selected_bg_checked",
                "selected_fg_eligible", "selected_bg_eligible",
            )
        },
    }
    if foreground_count == 0 or foreground_count == height * width:
        metadata["native_noop_reason"] = "empty_native" if foreground_count == 0 else "full_native"
        return _empty_arrays(), metadata

    band = _opposite_band(native)
    yy, xx = np.nonzero(band)
    metadata["candidate_band_count"] = int(len(xx))
    stencil_ok = (xx >= 1) & (xx < width - 1) & (yy >= 1) & (yy < height - 1)
    metadata["failure_counts"]["out_of_bounds_stencil"] = int((~stencil_ok).sum())
    coords = np.column_stack((xx[stencil_ok], yy[stencil_ok])).astype(np.int64)
    count = len(coords)
    metadata["inbounds_stencil_count"] = int(count)
    if count == 0:
        metadata["native_noop_reason"] = "no_inbounds_stencil"
        return _empty_arrays(), metadata

    result = {name: np.zeros((count, columns), dtype=np.float32)
              for name, columns in FLOAT_WIDTHS.items()}
    result["e"] = np.zeros(count, dtype=np.float32)
    result["e_coarse"] = np.zeros(count, dtype=np.float32)
    fg_counts = np.zeros(count, dtype=np.int64)
    bg_counts = np.zeros(count, dtype=np.int64)
    probability, uncertainty = _native_probability(logits)
    for start in range(0, count, TILE_SIZE):
        stop = min(start + TILE_SIZE, count)
        tile_coords = coords[start:stop]
        refs, counts = _reference_indices(tile_coords, native, band)
        fg_counts[start:stop] = counts["fg"]
        bg_counts[start:stop] = counts["bg"]
        tile_valid = ((counts["fg"] >= MIN_REFERENCES)
                      & (counts["bg"] >= MIN_REFERENCES))
        if not tile_valid.any():
            continue
        positions = start + np.flatnonzero(tile_valid)
        selected_coords = tile_coords[tile_valid]
        selected_refs = {name: refs[name][tile_valid] for name in ("fg", "bg")}
        x, y = selected_coords[:, 0], selected_coords[:, 1]
        u = uncertainty[y, x]
        c = np.column_stack((
            np.ones(len(x), dtype=np.float64),
            np.clip(logits[y, x].astype(np.float64) / 8.0, -1.0, 1.0),
            u,
        ))
        proto, nearest, gram, evidence, stencil = _appearance(
            rgb, selected_coords, selected_refs
        )
        _, coarse_nearest, coarse_gram, coarse_evidence, _ = _appearance(
            coarse_rgb, selected_coords, selected_refs
        )
        fields = {
            "c": c,
            "absolute_prefix": np.column_stack((c, stencil)),
            "proto": np.column_stack((c, proto)),
            "nn": np.column_stack((c, nearest)),
            "ordinary": np.column_stack((c, proto, nearest, gram)),
            "sym": _symmetric(u, nearest, gram),
            "e": evidence,
            "sym_coarse": _symmetric(u, coarse_nearest, coarse_gram),
            "e_coarse": coarse_evidence,
            "bilateral": np.column_stack((c, _bilateral(selected_coords, rgb, probability))),
        }
        for name, values in fields.items():
            if not np.isfinite(values).all():
                raise FloatingPointError(f"nonfinite constructed {name}")
            result[name][positions] = values.astype(np.float32)

    fg_bad = fg_counts < MIN_REFERENCES
    bg_bad = bg_counts < MIN_REFERENCES
    eligible = ~(fg_bad | bg_bad)
    eligible_count = int(eligible.sum())
    metadata["eligible_before_minimum_count"] = eligible_count
    metadata["failure_counts"].update({
        "foreground_support_lt4": int(fg_bad.sum()),
        "background_support_lt4": int(bg_bad.sum()),
        "either_support_lt4": int((fg_bad | bg_bad).sum()),
        "both_support_lt4": int((fg_bad & bg_bad).sum()),
    })
    for name, raw_counts in (("fg", fg_counts), ("bg", bg_counts)):
        selected_counts = np.minimum(raw_counts, REFERENCE_CAP)
        metadata["support_counts"][f"uncapped_{name}_checked"] = _summary(raw_counts)
        metadata["support_counts"][f"selected_{name}_checked"] = _summary(selected_counts)
        metadata["support_counts"][f"selected_{name}_eligible"] = _summary(selected_counts[eligible])
    if eligible_count < 2:
        metadata["failure_counts"]["minimum_query_rule_discarded"] = eligible_count
        metadata["native_noop_reason"] = "fewer_than_two_eligible_queries"
        return _empty_arrays(), metadata

    result = {name: np.ascontiguousarray(values[eligible]) for name, values in result.items()}
    result["coords"] = np.ascontiguousarray(coords[eligible])
    result["sample_indices"] = _uniform_sample(eligible_count)
    metadata["valid_query_count"] = eligible_count
    metadata["sample_count"] = int(len(result["sample_indices"]))
    for name, values in result.items():
        if not np.isfinite(values).all():
            raise FloatingPointError(f"nonfinite output {name}")
    return result, metadata
