"""Prospective reference-information controls; no data loading or fitting.

The delete-view matrix is a stability proxy, not a calibrated covariance.
All matrices use the same fixed local pose/field chart and column units.
"""
import numpy as np


RANK_RTOL = 1e-8


def mm(a, b):
    """Explicit rank-two contraction; no warning suppression/large projector."""
    return np.einsum("ik,kj->ij", a, b, optimize=False)


def finite_matrix(value, name):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError(name + " must be a finite matrix")
    return a


def column_space(a, rtol=RANK_RTOL):
    a = finite_matrix(a, "matrix")
    if not min(a.shape):
        return np.empty((a.shape[0], 0)), np.empty(0)
    u, s, _ = np.linalg.svd(a, full_matrices=False)
    keep = s > (rtol * s[0] if s.size else 0)
    return u[:, keep], s[keep]


def marginal_information(jacobian, keep, rtol=RANK_RTOL):
    """Project nuisance columns out before extracting identifiable information.

    The nullspace has zero *precision*. Its pseudoinverse is never interpreted
    as a zero-variance covariance. The caller must retain every kept variable
    in the query problem, including those in this nullspace.
    """
    j = finite_matrix(jacobian, "jacobian")
    keep = np.asarray(keep)
    if keep.ndim != 1 or keep.dtype.kind not in "iu" or len(set(keep.tolist())) != len(keep):
        raise ValueError("keep must contain distinct integer column indices")
    if not len(keep) or np.any(keep < 0) or np.any(keep >= j.shape[1]):
        raise ValueError("invalid kept columns")
    other = np.setdiff1d(np.arange(j.shape[1]), keep)
    basis, _ = column_space(j[:, other], rtol)
    effective = j[:, keep].copy()
    if basis.shape[1]:
        effective -= mm(basis, mm(basis.T, effective))
    _, s, vt = np.linalg.svd(effective, full_matrices=effective.shape[0] < effective.shape[1])
    # Use the complete kept-block scale, not tiny projection roundoff, as floor.
    original = np.linalg.svd(j[:, keep], compute_uv=False)
    scale = original[0] if original.size else 0.0
    rank = int(np.count_nonzero(s > rtol * scale)) if scale > 0 else 0
    u = vt[:rank].T
    singular = s[:rank]
    null = vt[rank:].T
    b = u / singular if rank else np.empty((len(keep), 0))
    return {"basis": u, "singular": singular, "null": null, "root": b,
            "rank": rank, "dimension": len(keep),
            "precision_factor": singular[:, None] * u.T,
            "projected_jacobian": effective,
            "rank_threshold": float(rtol * scale)}


def delete_view_variability(deltas):
    """Eight fixed, dependent group deletions in one reference-local chart."""
    d = finite_matrix(deltas, "deletion deltas")
    if d.shape[0] != 8:
        raise ValueError("all eight predeclared view deletions are required")
    centered = d - d.mean(axis=0)
    return (7.0 / 8.0) * mm(centered.T, centered)


def positive_part(matrix):
    a = finite_matrix(matrix, "symmetric matrix")
    if a.shape[0] != a.shape[1]:
        raise ValueError("matrix must be square")
    values, vectors = np.linalg.eigh((a + a.T) / 2.0)
    return mm(vectors * np.maximum(values, 0.0), vectors.T)


def shape_derivative(phi_anchor, phi_query, theta_anchor, theta_query,
                     valid_anchor=None, valid_query=None):
    """Depth-functional support derivative and its unprojected scale reference.

    Coordinates are [query pose(6), anchor field excluding fixed a(5), query
    log(a) and other field coefficients(6)]. Phi uses whole-image normalized
    raw depth. This quotient is a comparison convention, not a physical gauge
    or a claim of pose observability. Invalid source depth rows are excluded
    only here; their matches must remain in the BA fixed-mass loss.
    """
    parts = []; original = []
    for phi, theta, anchor, valid in ((phi_anchor, theta_anchor, True, valid_anchor),
                                      (phi_query, theta_query, False, valid_query)):
        phi = np.asarray(phi, dtype=np.float64)
        if phi.ndim != 2 or phi.shape[1] != 6:
            raise ValueError("phi must have six columns")
        if valid is None:
            valid = np.ones(len(phi), dtype=bool)
        valid = np.asarray(valid)
        if valid.shape != (len(phi),) or valid.dtype != bool:
            raise ValueError("boolean depth-valid mask must match phi")
        phi = finite_matrix(phi[valid], "valid phi")
        theta = np.asarray(theta, dtype=np.float64)
        if phi.shape[1] != 6 or theta.shape != (6,) or not np.isfinite(theta).all() or theta[0] <= 0:
            raise ValueError("invalid field")
        d = np.zeros((len(phi), 17))
        if anchor:
            if theta[0] != 1.0:
                raise ValueError("anchor depth scale must be exactly one")
            d[:, 6:11] = phi[:, 1:]
        else:
            d[:, 11:17] = phi
            d[:, 11] *= theta[0]
        original.append(d.copy())
        nuisance, _ = column_space(np.column_stack((np.einsum("ni,i->n", phi, theta), np.ones(len(phi)))))
        # This small rank-two projection uses explicit contractions. It also
        # avoids the platform BLAS matmul warning observed for this finite,
        # manufactured fixture; no floating-point warnings are suppressed.
        coefficients = np.einsum("ni,nj->ij", nuisance, d, optimize=False)
        d -= np.einsum("ni,ij->nj", nuisance, coefficients, optimize=False)
        parts.append(d)
    return {"projected": np.vstack(parts), "unprojected": np.vstack(original)}


def covariance_controls(information, deltas, shape):
    b = finite_matrix(information["root"], "information root")
    u = finite_matrix(information["basis"], "information basis")
    singular = np.asarray(information["singular"], dtype=np.float64)
    projected = finite_matrix(shape["projected"], "projected depth derivative")
    unprojected = finite_matrix(shape["unprojected"], "unprojected depth derivative")
    v = delete_view_variability(deltas)
    if (v.shape != (len(b), len(b)) or projected.shape[1] != len(b)
            or unprojected.shape != projected.shape):
        raise ValueError("state dimension mismatch")
    rank = b.shape[1]
    whiten = singular[:, None] * u.T
    z = mm(mm(whiten, v), whiten.T)
    excess = positive_part(z - np.eye(rank))
    weighted = mm(projected, b)
    unprojected_weighted = mm(unprojected, b)
    _, row_singular, row_vt = np.linalg.svd(weighted, full_matrices=False)
    original_singular = np.linalg.svd(unprojected_weighted, compute_uv=False)
    original_scale = float(original_singular[0]) if original_singular.size else 0.
    threshold = RANK_RTOL * original_scale
    retained = row_singular > threshold if original_scale > 0 else np.zeros(len(row_singular), dtype=bool)
    row_basis = row_vt[retained].T
    projector = mm(row_basis, row_basis.T)
    shaped = mm(mm(projector, excess), projector)
    shaped = (shaped + shaped.T) / 2.0
    shape_rank = row_basis.shape[1]
    gamma = max(float(np.trace(shaped)), 0.0) / shape_rank if shape_rank else 0.0
    inflation = {"hessian": np.eye(rank),
                 "scalar_deletion": np.eye(rank) + gamma * projector,
                 "directional": np.eye(rank) + shaped}
    factors = {}
    for name, covariance in inflation.items():
        values, vectors = np.linalg.eigh((covariance + covariance.T) / 2.0)
        if np.any(values <= 0):
            raise ValueError("nonpositive identifiable covariance")
        # ||factor @ local_delta||^2 supplies the compressed prior exactly once.
        factors[name] = mm((vectors / np.sqrt(values)).T, whiten)
    null = finite_matrix(information["null"], "reference nullspace")
    return {"precision_factors": factors, "deletion_variability": v,
            "whitened_variability": z, "positive_excess": excess,
            "shape_excess": shaped, "shape_projector": projector,
            "shape_rank": shape_rank, "scalar_inflation": gamma,
            "shape_excess_trace": float(np.trace(shaped)),
            "shape_null_norm": float(np.linalg.norm(mm(projected, null))),
            "shape_rank_threshold": threshold,
            "shape_unprojected_scale": original_scale,
            "identifiable_inflation": inflation}


def residual_inflation_factor(information, squared_pixel_blocks):
    errors = np.asarray(squared_pixel_blocks, dtype=np.float64)
    if errors.ndim != 1 or not errors.size or not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError("fixed complete residual block population required")
    # Invalid blocks enter as 16^2, matching the same fixed capped objective.
    alpha = max(1.0, float(np.median(np.minimum(errors, 256.0))) / (4.0 * 2.0 * np.log(2.0)))
    return information["precision_factor"] / np.sqrt(alpha), alpha
