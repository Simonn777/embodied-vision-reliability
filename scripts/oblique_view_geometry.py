"""Fixed-canvas geometry for prospective oblique-view tracking diagnostics.

Pure NumPy operations only: no files, models, random samples or inference.
Coordinates are [x, y], with x right and y down. Array fields are [row_y,
column_x]. The 256-square canvas has centers 0.5, ..., 255.5, and transforms
rotate about (128, 128). Positive angles turn the rightward axis downward.

Raster interpolation uses zero-valued missing neighbor taps, as in ordinary
zero-padded bilinear sampling. Accumulation is float64, then converted to
float32. The public uint8 renderer rounds that float32 result with np.rint
(ties to even), clips to [0, 255], and converts to uint8. The input is unchanged.
"""

import math

import numpy as np


CANVAS_SIZE = 256
CENTER = 128.0
PROBABILITY_FLOOR = 1e-12
FIXED_VIEW_SPECS = (
    ("original", 0.0, 1.0),
    ("reference", 0.0, 0.7),
    ("oblique", 45.0, 0.7),
    ("axis_control", 90.0, 0.7),
)


def _finite_real(value, name):
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(name + " must be real numeric values")
    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(name + " must be finite")
    return array


def _affine(matrix):
    result = _finite_real(matrix, "affine")
    if result.shape != (3, 3) or not np.array_equal(result[2], [0.0, 0.0, 1.0]):
        raise ValueError("affine must be 3x3 with final row [0, 0, 1]")
    if np.linalg.det(result[:2, :2]) == 0:
        raise ValueError("affine must be invertible")
    return result


def view_affine(angle_degrees, scale):
    """Return original-to-view homogeneous affine, computed in float64.

    R = [[cos, -sin], [sin, cos]]. Exact multiples of 90 degrees use exact
    zero/one entries, avoiding a numerical artifact in the axis control.
    """
    angle = float(angle_degrees)
    scale = float(scale)
    if not math.isfinite(angle) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("angle must be finite and scale finite and positive")
    quarter_turn = angle / 90.0
    if quarter_turn.is_integer():
        cosine, sine = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))[
            int(quarter_turn) % 4
        ]
    else:
        radians = math.radians(angle)
        cosine, sine = math.cos(radians), math.sin(radians)
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :2] = scale * np.array([[cosine, -sine], [sine, cosine]])
    center = np.array([CENTER, CENTER], dtype=np.float64)
    matrix[:2, 2] = center - matrix[:2, :2] @ center
    return _affine(matrix)


def fixed_view_affines():
    """Fresh original/reference/oblique/axis_control matrices; no shared mutation."""
    return {name: view_affine(angle, scale) for name, angle, scale in FIXED_VIEW_SPECS}


def invert_affine(matrix):
    """Return an inverse affine in float64 without changing its argument."""
    matrix = _affine(matrix)
    inverse = np.eye(3, dtype=np.float64)
    inverse[:2, :2] = np.linalg.inv(matrix[:2, :2])
    inverse[:2, 2] = -inverse[:2, :2] @ matrix[:2, 2]
    return _affine(inverse)


def _map_float64(points_xy, matrix):
    points = _finite_real(points_xy, "points")
    if points.ndim < 1 or points.shape[-1] != 2:
        raise ValueError("points must have final dimension [x, y]")
    matrix = _affine(matrix)
    mapped = points @ matrix[:2, :2].T + matrix[:2, 2]
    if not np.all(np.isfinite(mapped)):
        raise ValueError("mapped coordinates are not finite")
    return mapped


def map_points(points_xy, matrix):
    """Map query/display coordinates, returning float32 [x, y] values."""
    mapped = _map_float64(points_xy, matrix).astype(np.float32)
    if not np.all(np.isfinite(mapped)):
        raise ValueError("mapped query exceeds float32 range")
    return mapped


def pixel_center_grid():
    """Return float64 [256, 256, 2] centers; grid[y,x] contains [x+.5,y+.5]."""
    centers = np.arange(CANVAS_SIZE, dtype=np.float64) + 0.5
    x, y = np.meshgrid(centers, centers, indexing="xy")
    return np.stack([x, y], axis=-1)


def warp_rgb_float(frame_rgb, original_to_view):
    """Inverse-warp one uint8 RGB fixture/frame to float32 RGB in [0,255].

    Bilinear samples combine four center samples. Each out-of-bounds sample
    contributes zero, including the ordinary half-pixel boundary transition.
    This is interpolation, not area averaging or an antialiasing filter.
    """
    frame = np.asarray(frame_rgb)
    if frame.shape != (CANVAS_SIZE, CANVAS_SIZE, 3) or frame.dtype != np.uint8:
        raise ValueError("frame must be uint8 RGB with shape (256, 256, 3)")
    source = _map_float64(pixel_center_grid(), invert_affine(original_to_view))
    # Bound indices before integer conversion; unsupported huge coordinates
    # cannot overflow an int64 index or accidentally wrap into the image.
    supported = np.all((source > -0.5) & (source < CANVAS_SIZE + 0.5), axis=-1)
    index = np.clip(source - 0.5, -1.0, float(CANVAS_SIZE))
    low = np.floor(index).astype(np.int64)
    fraction = index - low
    output = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.float64)
    for offset_y in (0, 1):
        for offset_x in (0, 1):
            x_index = low[..., 0] + offset_x
            y_index = low[..., 1] + offset_y
            valid = supported & (x_index >= 0) & (x_index < CANVAS_SIZE)
            valid &= (y_index >= 0) & (y_index < CANVAS_SIZE)
            wx = fraction[..., 0] if offset_x else 1.0 - fraction[..., 0]
            wy = fraction[..., 1] if offset_y else 1.0 - fraction[..., 1]
            output[valid] += frame[y_index[valid], x_index[valid]].astype(np.float64) * (
                wx[valid] * wy[valid]
            )[:, None]
    return output.astype(np.float32)


def warp_rgb(frame_rgb, original_to_view):
    """Render uint8 RGB with fixed float32-then-round-to-even quantization."""
    output = warp_rgb_float(frame_rgb, original_to_view)
    return np.clip(np.rint(output), 0.0, 255.0).astype(np.uint8)


def reference_to_view_grid(reference_affine, view_affine):
    """Map reference bin centers through T_view @ inverse(T_reference).

    Returns float64 [row_y, column_x, xy]. The fixed equally scaled views have
    no relative scale, but their absolute input-rendering shrink is retained.
    """
    relative = _affine(view_affine) @ invert_affine(reference_affine)
    return _map_float64(pixel_center_grid(), relative)


def _probability_vector(probability, name):
    probability = _finite_real(probability, name)
    if probability.shape != (CANVAS_SIZE,) or np.any(probability < 0):
        raise ValueError(name + " must contain 256 nonnegative probabilities")
    if not np.isclose(probability.sum(), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError(name + " must sum to one within 1e-6")
    return probability


def map_marginal_probabilities(prob_x, prob_y, reference_affine, view_affine,
                               *, floor=PROBABILITY_FLOOR):
    """Evaluate auxiliary x/y marginals on reference coordinates, not a PDF warp.

    Inputs are DISPLAY x and y probabilities, not the raw model's y/x chunks.
    Linear interpolation uses centers .5 through 255.5; locations outside this
    closed interval receive floor. All values are floored, and nothing is
    renormalized. These are likelihood fields [row_y, column_x], not densities
    with a Jacobian or a claim of calibrated independent observations.
    """
    px = _probability_vector(prob_x, "prob_x")
    py = _probability_vector(prob_y, "prob_y")
    floor = float(floor)
    if not math.isfinite(floor) or not 0 < floor < 1.0 / CANVAS_SIZE:
        raise ValueError("probability floor must be positive and below 1/256")
    coordinates = reference_to_view_grid(reference_affine, view_affine)
    centers = np.arange(CANVAS_SIZE, dtype=np.float64) + 0.5
    mapped = []
    for probability, axis in ((px, 0), (py, 1)):
        field = np.interp(coordinates[..., axis].ravel(), centers, probability,
                          left=floor, right=floor)
        mapped.append(np.maximum(field, floor).reshape(CANVAS_SIZE, CANVAS_SIZE))
    return tuple(mapped)
