"""Pure fixed-cell features/actions/counts; no I/O, fitting, tracking or model.

Ordinary noisy label propagation is evidence, not certified material identity.
Feature extraction never accepts truth. Label counts are separate evaluation/
other-clip training arithmetic. Initial-frame action preservation is caller-owned.
NumPy/OpenCV and the sibling features module are imported only on invocation.
"""

FIXED_ARMS = ("B", "B+M1", "B+M2", "B+M1+M2")
POLICIES = ("author_background", "void_ignore")
FEATURE_NAMES = (
    "rgb_r_mean", "rgb_g_mean", "rgb_b_mean",
    "rgb_r_std", "rgb_g_std", "rgb_b_std",
    "logit_B_mean", "logit_M1_mean", "logit_M2_mean",
    "logit_B_std", "logit_M1_std", "logit_M2_std",
    "support_B", "support_BM1", "support_BM2", "support_union",
    "cell_x_center", "cell_y_center",
    "foreground_fraction", "background_fraction", "unknown_fraction",
    "valid_fraction", "fb_clipped_div10", "photo_div255",
    "backward_dx_div512", "backward_dy_div512", "frame_age_div200",
)
TRACK_COLUMNS = tuple(range(18, 27))


def _controls(controls, np):
    from collections.abc import Mapping
    if isinstance(controls, Mapping):
        if set(controls) != set(FIXED_ARMS):
            raise ValueError("exact four fixed controls required")
        values = np.stack([np.asarray(controls[key]) for key in FIXED_ARMS])
    else:
        values = np.asarray(controls)
    if (values.ndim != 3 or values.shape[0] != 4 or 0 in values.shape
            or values.dtype.kind not in "biu"
            or not np.all((values == 0) | (values == 1))):
        raise ValueError("controls must be binary [4,H,W], ID1 only")
    masks = values.astype(bool, copy=False)
    if (np.any(masks[0] & ~masks[1]) or np.any(masks[0] & ~masks[2])
            or not np.array_equal(masks[3], masks[1] | masks[2])):
        raise ValueError("controls must have the declared nested union support")
    return masks


def build_features(rgb_small, evidence, original_logits, controls, frame):
    """Return X[N,27], original-pixel ids[H,W], union_counts[N], metadata.

    RGB moments are divided by255; logit moments use INTER_AREA resize followed
    by clip(logit/32,-1,1). Evidence photo is ALREADY divided by255. Cell centers
    are means of normalized pixel-center coordinates, including partial cells.
    All returned arrays are new; inputs are never mutated.
    """
    import numpy as np
    import cv2
    import features

    rgb, logits = np.asarray(rgb_small), np.asarray(original_logits)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb_small must be uint8 [h,w,3]")
    if type(frame) is not int or frame < 0:
        raise ValueError("frame must be a nonnegative integer")
    masks = _controls(controls, np)
    if (logits.shape != (3, *masks.shape[1:]) or logits.dtype != np.float32
            or not np.isfinite(logits).all()):
        raise ValueError("logits must be finite float32 [3,H,W]")
    small_shape = tuple(rgb.shape[:2])
    ids = features.cell_ids(tuple(masks.shape[1:]), small_shape)
    small_ids = features.cell_ids(small_shape, small_shape)
    n = features.cell_count(small_shape)

    def mean(value, index=small_ids):
        return features.cell_means(value, index, n_cells=n)[0]

    def moments(value):
        value = np.asarray(value, dtype=np.float64)
        first = mean(value)
        second = mean(value * value)
        # Nonnegative mathematical variance; cancellation may give tiny negatives.
        return first, np.sqrt(np.maximum(second - first * first, 0.0))

    fields = {}
    for key in ("fg", "bg", "unknown", "valid", "fb", "photo", "flow_dx", "flow_dy"):
        value = np.asarray(evidence[key])
        if (value.shape != small_shape or value.dtype.kind not in "bifu"
                or not np.isfinite(value).all()):
            raise ValueError("invalid finite evidence field: " + key)
        if key in ("fg", "bg", "unknown", "valid") and value.dtype != np.bool_:
            raise ValueError("identity/validity evidence must be boolean")
        fields[key] = value
    if not np.all(fields["fg"].astype(np.int8) + fields["bg"] + fields["unknown"] == 1):
        raise ValueError("FG/BG/UNKNOWN must partition the small grid")
    if (np.any(fields["fb"] < 0) or np.any(fields["photo"] < 0)
            or np.any(fields["photo"] > 1)):
        raise ValueError("invalid evidence error units/range")
    rgb_mean, rgb_std = moments(rgb.astype(np.float64) / 255.0)
    h, w = small_shape
    small_logits = np.stack([
        cv2.resize(logits[k], (w, h), interpolation=cv2.INTER_AREA) for k in range(3)
    ], axis=-1)
    logit_mean, logit_std = moments(np.clip(small_logits.astype(np.float64) / 32, -1, 1))
    support = mean(np.moveaxis(masks, 0, -1), ids)
    yy, xx = np.indices(small_shape, dtype=np.float64)
    centers = mean(np.stack(((xx + 0.5) / w, (yy + 0.5) / h), axis=-1))
    state = mean(np.stack([fields[k] for k in ("fg", "bg", "unknown")], axis=-1))
    diagnostics = mean(np.stack((
        fields["valid"], np.clip(fields["fb"] / 10, 0, 1), fields["photo"],
        np.clip(fields["flow_dx"].astype(np.float64) / 512, -1, 1),
        np.clip(fields["flow_dy"].astype(np.float64) / 512, -1, 1),
    ), axis=-1))
    X = np.column_stack((rgb_mean, rgb_std, logit_mean, logit_std,
                         support, centers, state, diagnostics,
                         np.full(n, min(frame / 200, 1.0)))).astype(np.float32)
    if X.shape != (n, len(FEATURE_NAMES)) or not np.isfinite(X).all():
        raise ValueError("nonfinite or wrong-dimensional final cell features")
    union_counts = np.bincount(ids[masks[3]], minlength=n).astype(np.int64)
    return X, ids, union_counts, {
        "feature_names": list(FEATURE_NAMES), "track_columns": list(TRACK_COLUMNS),
        "original_shape": list(masks.shape[1:]), "small_shape": list(small_shape),
        "cells": n, "nonempty_union_cells": int(np.count_nonzero(union_counts)),
        "union_pixels": int(union_counts.sum()), "frame": frame,
        "cell_centers": "mean normalized small-grid pixel centers",
        "logit_order": "resize INTER_AREA, divide32, clip[-1,1], moments",
        "photo_input_already_divided_by255": True,
    }


def label_counts(truth, controls, ids, *, n_cells=None):
    """Flat policy-specific count arrays; pass len(X) when upsampling leaves holes."""
    import numpy as np
    masks, gt, index = _controls(controls, np), np.asarray(truth), np.asarray(ids)
    if (gt.shape != masks.shape[1:] or gt.dtype.kind not in "iu"
            or not np.all((gt == 0) | (gt == 1) | (gt == 255))):
        raise ValueError("truth must be integer ID0/1/255 with original shape")
    if index.shape != gt.shape or index.dtype.kind not in "iu" or np.any(index < 0):
        raise ValueError("invalid original-pixel cell IDs")
    minimum = int(index.max()) + 1
    n = minimum if n_cells is None else n_cells
    if type(n) is not int or n < minimum:
        raise ValueError("n_cells does not cover all original pixel IDs")
    result = {}
    for policy in POLICIES:
        valid = gt != 255 if policy == "void_ignore" else np.ones(gt.shape, bool)
        positive = gt == 1
        row = {"total_gt": int(np.count_nonzero(positive & valid))}
        for name, mask in (("union", masks[3]), ("native", masks[0]),
                           ("extra", masks[3] & ~masks[0])):
            for suffix, target in (("tp", positive), ("fp", ~positive)):
                row[name + "_" + suffix] = np.bincount(
                    index[mask & target & valid], minlength=n).astype(np.int64)
        row["fixed_tp"] = np.array([np.count_nonzero(m & positive & valid) for m in masks], np.int64)
        row["fixed_fp"] = np.array([np.count_nonzero(m & ~positive & valid) for m in masks], np.int64)
        result[policy] = row
    return result


def score_actions(counts, keep_native, keep_extra):
    """Exact original-resolution J from counts; return JSON-safe removal totals."""
    import numpy as np
    first, second = np.asarray(keep_native), np.asarray(keep_extra)
    if (first.ndim != 1 or first.dtype != np.bool_
            or second.shape != first.shape or second.dtype != np.bool_):
        raise ValueError("two same-length boolean action vectors required")
    vectors = {}
    for prefix in ("native", "extra", "union"):
        for kind in ("tp", "fp"):
            key = prefix + "_" + kind
            value = np.asarray(counts[key])
            if value.shape != first.shape or value.dtype.kind not in "iu" or np.any(value < 0):
                raise ValueError("invalid count vector: " + key)
            vectors[key] = value
    for kind in ("tp", "fp"):
        if not np.array_equal(vectors["native_" + kind] + vectors["extra_" + kind],
                              vectors["union_" + kind]):
            raise ValueError("native/extra counts must partition union")
    total_gt = counts["total_gt"]
    if isinstance(total_gt, (bool, np.bool_)) or not isinstance(total_gt, (int, np.integer)) or total_gt < 0:
        raise ValueError("total_gt must be a nonnegative integer")
    if int(vectors["union_tp"].sum()) > total_gt:
        raise ValueError("union TP exceeds total ground truth")
    tp = int(vectors["native_tp"][first].sum() + vectors["extra_tp"][second].sum())
    fp = int(vectors["native_fp"][first].sum() + vectors["extra_fp"][second].sum())
    denominator = int(total_gt) + fp
    result = {"J": tp / denominator if denominator else 1.0, "tp": tp, "fp": fp}
    for prefix, keep in (("native", first), ("extra", second)):
        for kind in ("tp", "fp"):
            result["removed_" + prefix + "_" + kind] = int(vectors[prefix + "_" + kind][~keep].sum())
    return result


def fixed_actions(X):
    """Negative-veto acts on all union; positive-required always retains native."""
    import numpy as np
    values = np.asarray(X)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES) or not np.isfinite(values).all():
        raise ValueError("finite fixed-order 27-column features required")
    keep = values[:, 19] < 0.75
    return {"fixed_negative": (keep.copy(), keep.copy()),
            "positive_required": (np.ones(len(values), bool), values[:, 18] >= 0.75)}


actions_fixed = fixed_actions


def sample_indices(union_counts):
    """Up to512 equally spaced nonempty-cell positions, row-major; no truth."""
    import numpy as np
    values = np.asarray(union_counts)
    if values.ndim != 1 or values.dtype.kind not in "iu" or np.any(values < 0):
        raise ValueError("nonnegative integer union counts required")
    nonempty = np.flatnonzero(values)
    if not len(nonempty):
        return nonempty
    positions = np.linspace(0, len(nonempty) - 1, min(512, len(nonempty))).astype(np.int64)
    return nonempty[positions]
