"""Unfrozen NumPy-only 7x7 conditional-residual patch prototype.

This is an oracle-covariance mathematical component, not a denoiser or a result.
Rows index independent valid centers; downstream processing must be pointwise.
"""
import math
import numpy as np

RADIUS = 3
PATCH_SIDE = 7
CENTER = 24
OFFSETS = tuple((dy, dx) for dy in range(-3, 4) for dx in range(-3, 4))
NEIGHBORS = tuple(i for i in range(49) if i != CENTER)
FEATURE_OFFSETS = tuple(OFFSETS[i] for i in NEIGHBORS)
DTYPES = (np.dtype('float32'), np.dtype('float64'))


def _float_array(value, ndim, name):
    array = np.asarray(value)
    if array.ndim != ndim or array.dtype not in DTYPES:
        raise ValueError(name + ' must have the required rank and float32/64 dtype')
    if not np.isfinite(array).all():
        raise ValueError(name + ' must be finite')
    return array


def validate_covariance(covariance):
    """Validate a full [49,49] patch covariance in the documented raster order.

    Tiny eigenvalue/symmetry residuals within 64*eps*scale are accepted as floating
    arithmetic error. The supplied matrix is never symmetrized or repaired.
    """
    c = _float_array(covariance, 2, 'covariance')
    if c.shape != (49, 49):
        raise ValueError('covariance must be [49,49], including center index 24')
    if c[CENTER, CENTER] <= 0:
        raise ValueError('center noise variance sigma_squared must be positive')
    scale = float(np.max(np.abs(c)))
    tolerance = 64 * np.finfo(c.dtype).eps * scale
    if np.max(np.abs(c.astype(np.float64) - c.T.astype(np.float64))) > tolerance:
        raise ValueError('covariance must be symmetric')
    if np.linalg.eigvalsh(c.astype(np.float64))[0] < -tolerance:
        raise ValueError('covariance must be positive semidefinite')
    return c


def default_kernel(dtype=np.float64):
    """L2-unit h=g outer g, g=[1,2,1]/sqrt(6); not a fitted kernel."""
    dtype = np.dtype(dtype)
    if dtype not in DTYPES:
        raise ValueError('kernel dtype must be float32/64')
    g = np.array([1, 2, 1], dtype=np.float64) / math.sqrt(6)
    return np.outer(g, g).astype(dtype)


def covariance_from_kernel(kernel, sigma=1.0):
    """Covariance of n[p]=sigma*sum_u kernel[u]*epsilon[p-u].

    epsilon is unit-variance IID Gaussian on an infinite lattice. No wrapping,
    finite-image padding, or kernel normalization is implied. General kernels
    have C00=sigma**2*sum(kernel**2); the default kernel has unit squared norm.
    Returned float32/64 dtype equals kernel dtype. This computes covariance only.
    """
    h = _float_array(kernel, 2, 'kernel')
    if not h.size or min(h.shape) == 0:
        raise ValueError('kernel must be nonempty')
    if isinstance(sigma, (bool, np.bool_)) or np.ndim(sigma) != 0:
        raise ValueError('sigma must be a positive finite scalar')
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError('sigma must be positive and finite')
    h64 = h.astype(np.float64)
    cache = {}
    for dy in range(-6, 7):
        for dx in range(-6, 7):
            products = [float(h64[y, x]) * float(h64[y+dy, x+dx])
                        for y in range(h.shape[0]) for x in range(h.shape[1])
                        if 0 <= y+dy < h.shape[0] and 0 <= x+dx < h.shape[1]]
            cache[(dy, dx)] = math.fsum(products) * sigma * sigma
    result = np.array([[cache[(b[0]-a[0], b[1]-a[1])] for b in OFFSETS]
                       for a in OFFSETS], dtype=np.float64)
    if not np.isfinite(result).all() or np.max(np.abs(result)) > np.finfo(h.dtype).max:
        raise ValueError('kernel covariance is not representable in its dtype')
    return validate_covariance(result.astype(h.dtype))


def linear_operator(covariance):
    """Return A[48,49]: each row is e_j - (C[j,24]/C[24,24])*e_24."""
    c = validate_covariance(covariance)
    a = np.zeros((48, 49), dtype=c.dtype)
    a[np.arange(48), NEIGHBORS] = 1
    a[:, CENTER] = -c[list(NEIGHBORS), CENTER] / c[CENTER, CENTER]
    return a


def analytic_statistics(covariance, operator=None):
    """Return Cov(A n, n_center) and Cov(A n) using explicit matrix algebra.

    An optional alternative operator permits negative-control leakage checks.
    Zero cross-covariance implies joint independence only for Gaussian noise.
    """
    c = validate_covariance(covariance)
    a = linear_operator(c) if operator is None else _float_array(operator, 2, 'operator')
    if a.shape != (48, 49) or a.dtype != c.dtype:
        raise ValueError('operator must be [48,49] and match covariance dtype')
    # Small explicit reductions avoid backend BLAS/FPU warning differences.
    # ac[i,l] = sum_k A[i,k]*C[k,l], then sum_l ac[i,l]*A[j,l].
    ac = np.sum(a[:, :, None] * c[None, :, :], axis=1)
    return {'cross_covariance_with_center': np.sum(a * c[None, :, CENTER], axis=1),
            'residual_noise_covariance': np.sum(ac[:, None, :] * a[None, :, :], axis=2)}


def valid_centers(image_shape):
    """Return [N,2] integer (row,column), row-major, with no padded centers."""
    if (len(image_shape) != 2 or any(isinstance(n, (bool, np.bool_))
            or not isinstance(n, (int, np.integer)) or n < 7 for n in image_shape)):
        raise ValueError('image must have two dimensions, each at least 7')
    height, width = map(int, image_shape)
    return np.array([(y, x) for y in range(3, height-3)
                     for x in range(3, width-3)], dtype=np.int64)


def extract_valid_patches(image):
    """Return [N,49] rasterized raw patches from one finite grayscale image."""
    y = _float_array(image, 2, 'image')
    valid_centers(y.shape)
    patches = np.lib.stride_tricks.sliding_window_view(y, (7, 7))
    return patches.reshape(-1, 49)


def residual_features(image, covariance):
    """Return [N,48] features; every row uses that patch's SAME center value.

    Image and covariance must share dtype, so no implicit precision change occurs.
    Do not reshape these rows into feature maps and apply a spatial convolution:
    each row is independent of its own center noise only, not another center's.
    """
    y = _float_array(image, 2, 'image')
    c = validate_covariance(covariance)
    if y.dtype != c.dtype:
        raise ValueError('image and covariance must have the same dtype')
    patches = extract_valid_patches(y)
    gains = c[list(NEIGHBORS), CENTER] / c[CENTER, CENTER]
    result = patches[:, NEIGHBORS] - patches[:, CENTER, None] * gains
    if not np.isfinite(result).all():
        raise ValueError('transformed features overflowed their dtype')
    return result
