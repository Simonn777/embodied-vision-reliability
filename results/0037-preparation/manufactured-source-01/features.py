"""Causal, non-reseeding initial-label transport; no SAM or label-file access.

All NumPy/OpenCV imports are lazy. Coordinates denote zero-based pixel centers.
initialize/advance/propagate return (state, current_small_rgb, evidence).
State is uint8: 0 unknown, 1 initial target (ID 1), 2 initial background (ID 0).
Evidence arrays use the current small grid; flow_dx/dy are backward displacements.
"""

from types import MappingProxyType


CONFIG = MappingProxyType({
    "schema": "evr0037.label_transport.v1",
    "longest_side": 512,
    "minimum_side": 4,
    "dimension_rounding": "nearest_ties_to_even",
    "rgb_resize": "opencv_INTER_AREA",
    "initial_resize": "opencv_INTER_NEAREST",
    "initial_erosion_size": 3,
    "initial_erosion_border": "constant_zero",
    "label_unknown": 0,
    "label_foreground": 1,
    "label_background": 2,
    "flow_algorithm": "opencv_DIS_PRESET_MEDIUM",
    "flow_initialization": "None_fresh_instance_each_direction",
    "finest_scale": 1,
    "patch_size": 8,
    "patch_stride": 3,
    "gradient_descent_iterations": 25,
    "variational_refinement_iterations": 5,
    "variational_refinement_alpha": 20.0,
    "variational_refinement_delta": 5.0,
    "variational_refinement_gamma": 10.0,
    "use_mean_normalization": True,
    "use_spatial_propagation": True,
    "opencv_threads": 4,
    "rgb_to_gray": "opencv_COLOR_RGB2GRAY",
    "flow_coordinate_convention": "zero_based_pixel_centers",
    "flow_map_direction": "current_to_previous",
    "backcoordinate_domain": "closed_pixel_center_rectangle",
    "flow_and_photo_sampling": "opencv_INTER_LINEAR",
    "label_sampling": "opencv_INTER_NEAREST",
    "forward_backward_threshold": 1.5,
    "photo_raw_rgb_threshold": 20.0,
    "photo_report_divisor": 255.0,
    "cell_side": 4,
    "cell_order": "row_major",
    "empty_cell_mean": 0.0,
})


def configuration():
    """Return a JSON-safe copy of every explicitly fixed public setting."""
    return dict(CONFIG)


def _libraries():
    import numpy as np
    import cv2

    cv2.setNumThreads(CONFIG["opencv_threads"])
    if cv2.getNumThreads() != CONFIG["opencv_threads"]:
        raise RuntimeError("OpenCV did not accept the fixed four-thread setting")
    return np, cv2


def _shape(shape, name, *, small=False):
    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
        raise ValueError(name + " must be a (height, width) pair")
    if any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError(name + " dimensions must be positive Python integers")
    h, w = shape
    if small and (min(h, w) < 4 or max(h, w) > 512):
        raise ValueError(name + " must have both sides in [4,512]")
    return h, w


def _rgb(value, name, np, *, small=False):
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
        raise ValueError(name + " must be a uint8 NumPy array")
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(name + " must have shape (H,W,3)")
    _shape(tuple(value.shape[:2]), name, small=small)
    return value


def _state(value, shape, np):
    if (not isinstance(value, np.ndarray) or value.dtype != np.uint8
            or value.shape != shape):
        raise ValueError("state must be uint8 with the small RGB spatial shape")
    if not np.all((value == 0) | (value == 1) | (value == 2)):
        raise ValueError("state contains a label outside {0,1,2}")
    return value


def _target_shape(shape):
    h, w = _shape(shape, "original shape")
    longest = max(h, w)
    return (
        max(4, int(round(h * 512.0 / longest))),
        max(4, int(round(w * 512.0 / longest))),
    )


def resize_rgb(rgb):
    """Resize RGB to longest side 512, including upsampling smaller inputs."""
    np, cv2 = _libraries()
    _rgb(rgb, "rgb", np)
    h, w = _target_shape(tuple(rgb.shape[:2]))
    result = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    _rgb(result, "resized RGB", np, small=True)
    return np.ascontiguousarray(result).copy()


def _metadata(state, np, cv2, phase, **extra):
    return {
        "phase": phase,
        "config": configuration(),
        "numpy_version": str(np.__version__),
        "opencv_version": str(cv2.__version__),
        "small_shape": [int(v) for v in state.shape],
        "pixels": int(state.size),
        "foreground": int(np.count_nonzero(state == 1)),
        "background": int(np.count_nonzero(state == 2)),
        "unknown": int(np.count_nonzero(state == 0)),
        **extra,
    }


def initialize(rgb, initial):
    """Seed once from ID 1 / background 0; ID 255 stays unknown.

    Resize the annotation with INTER_NEAREST, then erode the two known classes
    separately with a 3x3 all-ones kernel and a zero border. Other IDs fail closed.
    No subsequent operation can seed labels from a current SAM prediction.
    """
    np, cv2 = _libraries()
    _rgb(rgb, "rgb", np)
    if (not isinstance(initial, np.ndarray) or initial.dtype != np.uint8
            or initial.shape != rgb.shape[:2]):
        raise ValueError("initial must be uint8 with the original RGB shape")
    if not np.all((initial == 0) | (initial == 1) | (initial == 255)):
        raise ValueError("initial labels must be exactly 0, 1, or 255")
    small = resize_rgb(rgb)
    h, w = small.shape[:2]
    labels = cv2.resize(initial, (w, h), interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((3, 3), dtype=np.uint8)
    fg = cv2.erode(
        (labels == 1).astype(np.uint8), kernel, iterations=1,
        borderType=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(bool)
    bg = cv2.erode(
        (labels == 0).astype(np.uint8), kernel, iterations=1,
        borderType=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(bool)
    if np.any(fg & bg):
        raise RuntimeError("eroded initial classes unexpectedly overlap")
    state = np.zeros((h, w), dtype=np.uint8)
    state[fg], state[bg] = 1, 2
    known = state != 0
    evidence = {
        "fg": fg.copy(),
        "bg": bg.copy(),
        "unknown": ~known,
        "valid": known.copy(),
        "reliable": known.copy(),
        "in_bounds": np.ones((h, w), dtype=bool),
        "fb": np.zeros((h, w), dtype=np.float64),
        "photo": np.zeros((h, w), dtype=np.float64),
        "flow_dx": np.zeros((h, w), dtype=np.float32),
        "flow_dy": np.zeros((h, w), dtype=np.float32),
        "source_label": state.copy(),
        "metadata": _metadata(
            state, np, cv2, "initial",
            original_shape=[int(v) for v in rgb.shape[:2]],
            initial_void_small=int(np.count_nonzero(labels == 255)),
            initial_foreground_before_erosion=int(np.count_nonzero(labels == 1)),
            initial_background_before_erosion=int(np.count_nonzero(labels == 0)),
            flow_estimation_calls=0,
        ),
    }
    return state, small, evidence


_DIS_PARAMETERS = (
    ("FinestScale", "finest_scale"),
    ("PatchSize", "patch_size"),
    ("PatchStride", "patch_stride"),
    ("GradientDescentIterations", "gradient_descent_iterations"),
    ("VariationalRefinementIterations", "variational_refinement_iterations"),
    ("VariationalRefinementAlpha", "variational_refinement_alpha"),
    ("VariationalRefinementDelta", "variational_refinement_delta"),
    ("VariationalRefinementGamma", "variational_refinement_gamma"),
    ("UseMeanNormalization", "use_mean_normalization"),
    ("UseSpatialPropagation", "use_spatial_propagation"),
)


def _make_dis(cv2):
    estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    for suffix, key in _DIS_PARAMETERS:
        getattr(estimator, "set" + suffix)(CONFIG[key])
        if getattr(estimator, "get" + suffix)() != CONFIG[key]:
            raise RuntimeError("DIS rejected fixed parameter " + key)
    return estimator


def _estimate_flows(previous, current, cv2):
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    # Independent estimators and None initialization prevent previous-flow reuse.
    forward = _make_dis(cv2).calc(previous_gray, current_gray, None)
    backward = _make_dis(cv2).calc(current_gray, previous_gray, None)
    return forward, backward


def propagate(previous, current, state, forward, backward):
    """Transport labels with injected flows, without estimating or fitting flow.

    forward is previous->current and backward is current->previous, both float32
    HxWx2. Forward flow and previous RGB are bilinearly sampled at the backward
    coordinate. A coordinate must lie in [0,W-1] x [0,H-1]. Invalid coordinates
    are never clamped into eligibility. UNKNOWN sources remain UNKNOWN.
    """
    np, cv2 = _libraries()
    _rgb(previous, "previous small RGB", np, small=True)
    _rgb(current, "current small RGB", np, small=True)
    if previous.shape != current.shape:
        raise ValueError("previous and current small RGB shapes differ")
    h, w = current.shape[:2]
    _state(state, (h, w), np)
    for value, name in ((forward, "forward"), (backward, "backward")):
        if (not isinstance(value, np.ndarray) or value.dtype != np.float32
                or value.shape != (h, w, 2) or not np.isfinite(value).all()):
            raise ValueError(name + " flow must be finite float32 HxWx2")

    yy, xx = np.indices((h, w), dtype=np.float64)
    bx = xx + backward[..., 0].astype(np.float64)
    by = yy + backward[..., 1].astype(np.float64)
    in_bounds = (bx >= 0) & (bx <= w - 1) & (by >= 0) & (by <= h - 1)
    # Sample a harmless coordinate for rejected sites, avoiding large-coordinate
    # conversion inside OpenCV. in_bounds still excludes every such site.
    mx = np.where(in_bounds, bx, 0.0).astype(np.float32)
    my = np.where(in_bounds, by, 0.0).astype(np.float32)
    sampled_forward = cv2.remap(
        forward, mx, my, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(np.float64)
    consistency = sampled_forward + backward.astype(np.float64)
    fb = np.hypot(consistency[..., 0], consistency[..., 1])
    previous_warp = cv2.remap(
        previous.astype(np.float32), mx, my, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(np.float64)
    photo_raw = np.mean(
        np.abs(current.astype(np.float64) - previous_warp), axis=2,
        dtype=np.float64,
    )
    source = cv2.remap(
        state, mx, my, cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    source = source.copy()
    source[~in_bounds] = 0
    if not np.isfinite(fb).all() or not np.isfinite(photo_raw).all():
        raise ValueError("nonfinite transport diagnostic")
    fb_pass = fb <= CONFIG["forward_backward_threshold"]
    photo_pass = photo_raw <= CONFIG["photo_raw_rgb_threshold"]
    reliable = in_bounds & fb_pass & photo_pass
    valid = reliable & (source != 0)
    result = np.where(valid, source, 0).astype(np.uint8)
    evidence = {
        "fg": result == 1,
        "bg": result == 2,
        "unknown": result == 0,
        "valid": valid,
        "reliable": reliable,
        "in_bounds": in_bounds,
        "fb": fb,
        "photo": photo_raw / CONFIG["photo_report_divisor"],
        "flow_dx": backward[..., 0].copy(),
        "flow_dy": backward[..., 1].copy(),
        "source_label": source,
        "metadata": _metadata(
            result, np, cv2, "injected_flow_transport",
            flow_estimation_calls=0,
            in_bounds=int(np.count_nonzero(in_bounds)),
            fb_pass_in_bounds=int(np.count_nonzero(in_bounds & fb_pass)),
            photo_pass_in_bounds=int(np.count_nonzero(in_bounds & photo_pass)),
            reliable=int(np.count_nonzero(reliable)),
            reliable_unknown_source=int(np.count_nonzero(reliable & (source == 0))),
            out_of_bounds=int(np.count_nonzero(~in_bounds)),
            fb_failed_in_bounds=int(np.count_nonzero(in_bounds & ~fb_pass)),
            photo_failed_in_bounds=int(np.count_nonzero(in_bounds & ~photo_pass)),
        ),
    }
    return result, np.ascontiguousarray(current).copy(), evidence


def advance(prev_rgb_small, current_rgb, state):
    """One causal step: resize current RGB, estimate both flows, then transport.

    The resized current lattice must match the previous lattice. The caller owns
    the original-resolution video-shape contract. No current annotation or SAM
    output is accepted by this API.
    """
    np, cv2 = _libraries()
    _rgb(prev_rgb_small, "previous small RGB", np, small=True)
    _state(state, prev_rgb_small.shape[:2], np)
    current = resize_rgb(current_rgb)
    if current.shape != prev_rgb_small.shape:
        raise ValueError("current resized lattice differs from previous lattice")
    forward, backward = _estimate_flows(prev_rgb_small, current, cv2)
    result, small, evidence = propagate(
        prev_rgb_small, current, state, forward, backward,
    )
    evidence["metadata"].update({
        "phase": "advance",
        "original_shape": [int(v) for v in current_rgb.shape[:2]],
        "flow_estimation_calls": 2,
    })
    return result, small, evidence


def cell_count(small_shape):
    h, w = _shape(small_shape, "small shape", small=True)
    return ((h + 3) // 4) * ((w + 3) // 4)


def cell_ids(original_shape, small_shape):
    """Disjoint row-major 4x4 small-grid cells, mapped to original pixels.

    Exact integer arithmetic implements floor(original_coord*small_dim/orig_dim)
    //4, without overlapping windows or clipping the final ceil-edge blocks.
    """
    np, _ = _libraries()
    h, w = _shape(original_shape, "original shape")
    sh, sw = _shape(small_shape, "small shape", small=True)
    ys = (np.arange(h, dtype=np.int64) * sh // h) // 4
    xs = (np.arange(w, dtype=np.int64) * sw // w) // 4
    return ys[:, None] * ((sw + 3) // 4) + xs[None, :]


def cell_means(values, ids, *, n_cells=None):
    """Return (float64 means, int64 counts), retaining empty cells explicitly.

    values is finite HxW or HxWxC; ids is nonnegative int64 HxW. Pass
    n_cells=cell_count(small_shape) to retain unoccupied trailing cells, which can
    occur when the original image is smaller than the resized tracking grid.
    Empty-cell means are zero and MUST be interpreted together with counts.
    """
    np, _ = _libraries()
    if (not isinstance(ids, np.ndarray) or ids.dtype != np.int64
            or ids.ndim != 2 or min(ids.shape) <= 0):
        raise ValueError("ids must be nonempty int64 HxW")
    if (not isinstance(values, np.ndarray) or values.ndim not in (2, 3)
            or values.shape[:2] != ids.shape or values.dtype.kind not in "buif"):
        raise ValueError("values must be numeric HxW or HxWxC matching ids")
    if values.ndim == 3 and values.shape[2] <= 0:
        raise ValueError("values must have a nonempty channel dimension")
    if not np.isfinite(values).all() or np.any(ids < 0):
        raise ValueError("nonfinite values or negative cell IDs")
    minimum_cells = int(ids.max()) + 1
    if n_cells is None:
        n_cells = minimum_cells
    if type(n_cells) is not int or not minimum_cells <= n_cells <= 128 * 128:
        raise ValueError("n_cells must cover IDs and fit the maximum 512 grid")
    flat_ids = ids.ravel()
    counts = np.bincount(flat_ids, minlength=n_cells).astype(np.int64)
    matrix = values.reshape(ids.size, -1).astype(np.float64)
    means = np.zeros((n_cells, matrix.shape[1]), dtype=np.float64)
    for channel in range(matrix.shape[1]):
        totals = np.bincount(
            flat_ids, weights=matrix[:, channel], minlength=n_cells,
        )
        np.divide(totals, counts, out=means[:, channel], where=counts > 0)
    if not np.isfinite(means).all():
        raise ValueError("nonfinite cell mean")
    return (means[:, 0] if values.ndim == 2 else means), counts
