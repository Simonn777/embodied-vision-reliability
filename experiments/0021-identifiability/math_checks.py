"""Fixed manufactured geometry; NumPy only, with no file or model access."""
import numpy as np


RANK_RTOL = 1e-10
RANK_SENSITIVITY = (1e-12, 1e-10, 1e-8)
FD_STEP = 1e-6
FD_ATOL = 5e-8
FD_RTOL = 1e-6
FIXTURE_SPEC = {
    "reference_uv": [[-.65, -.45], [.5, -.4], [-.4, .55], [.6, .5],
                     [.1, -.05], [-.15, .25]],
    "generic_depth": [2., 3.1, 4.6, 2.7, 5.3, 3.8],
    "generic_theta": [1.1, .12, -.08, .2, -.06, .09],
    "rotation_vector": [.13, -.19, .07],
    "translation_before_normalization": [.9, .2, .35],
    "plane_alpha_beta_gamma": [.2, -.15, .8],
    "plane_theta": [1., 0., 0., 0., 0., 0.],
    "plane_null_epsilon": .1,
    "query_uv": [[-.55, .10], [.25, .65], [.72, -.15], [-.75, -.65],
                 [.05, .45], [.35, -.55], [-.22, -.12], [.48, .12]],
    "query_depth": [2.2, 3.6, 4.1, 2.5, 5., 3.3, 4.7, 2.9],
    "finite_scale_factor": 1.7,
    "finite_additive_shift": .15,
    "calibration": "Normalized camera coordinates, K_source=K_target=I; no distortion.",
    "interpretation": "Query-depth shape ambiguity only; no query pose is fitted or evaluated.",
}


def finite_array(value, shape=None):
    a = np.asarray(value, dtype=np.float64)
    if shape is not None and a.shape != shape:
        raise ValueError("unexpected shape")
    if not np.isfinite(a).all():
        raise ValueError("nonfinite input")
    return a


def phi(uv, depth):
    uv = finite_array(uv)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("uv must be N by 2")
    d = finite_array(depth, (len(uv),))
    return np.column_stack((d, uv[:, 0] * d, uv[:, 1] * d,
                            np.ones(len(d)), uv[:, 0], uv[:, 1]))


def skew(v):
    x, y, z = finite_array(v, (3,))
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def rotation_exp(omega):
    omega = finite_array(omega, (3,))
    angle = np.linalg.norm(omega)
    if angle == 0:
        return np.eye(3)
    w = skew(omega)
    return (np.eye(3) + (np.sin(angle) / angle) * w
            + (2 * np.sin(angle / 2) ** 2 / angle ** 2) * (w @ w))


def tangent_basis(t):
    t = finite_array(t, (3,))
    if not np.isclose(np.linalg.norm(t), 1., rtol=0, atol=1e-12):
        raise ValueError("translation must have unit norm")
    axis = np.eye(3)[np.argmin(np.abs(t))]
    b1 = np.cross(t, axis)
    b1 /= np.linalg.norm(b1)
    return np.column_stack((b1, np.cross(t, b1)))


def project(points):
    p = finite_array(points)
    if p.ndim != 2 or p.shape[1] != 3 or np.any(p[:, 2] <= 0):
        raise ValueError("projection requires positive finite Z")
    return p[:, :2] / p[:, 2, None]


def projection_jacobian(point):
    x, y, z = finite_array(point, (3,))
    if z <= 0:
        raise ValueError("projection requires positive finite Z")
    return np.array([[1 / z, 0., -x / z ** 2],
                     [0., 1 / z, -y / z ** 2]])


def reference_prediction(uv, depth, theta, R, t):
    f = phi(uv, depth) @ finite_array(theta, (6,))
    if np.any(f <= 0):
        raise ValueError("corrected source depth must be positive")
    rays = np.column_stack((finite_array(uv), np.ones(len(f))))
    rotated = (finite_array(R, (3, 3)) @ (rays * f[:, None]).T).T
    return project(rotated + finite_array(t, (3,)))


def reference_jacobian(uv, depth, theta, R, t, unit_translation=True):
    """Columns theta[6], left rotation[3], tangent translation[2] or free[3]."""
    design = phi(uv, depth)
    theta, R, t = (finite_array(theta, (6,)), finite_array(R, (3, 3)),
                   finite_array(t, (3,)))
    f = design @ theta
    if np.any(f <= 0):
        raise ValueError("corrected source depth must be positive")
    rays = np.column_stack((finite_array(uv), np.ones(len(f))))
    rotated_rays = (R @ rays.T).T
    rotated_points = rotated_rays * f[:, None]
    basis = tangent_basis(t) if unit_translation else np.eye(3)
    rows = []
    for i in range(len(f)):
        jpi = projection_jacobian(rotated_points[i] + t)
        rows.append(np.column_stack((
            np.outer(jpi @ rotated_rays[i], design[i]),
            jpi @ -skew(rotated_points[i]), jpi @ basis)))
    return np.concatenate(rows, axis=0)


def rank_info(matrix):
    matrix = finite_array(matrix)
    s = np.linalg.svd(matrix, compute_uv=False)
    scale = float(s[0]) if len(s) else 0.
    ranks = {str(tol): int(np.count_nonzero(s > tol * scale))
             for tol in RANK_SENSITIVITY}
    return {"shape": list(matrix.shape), "singular_values": s.tolist(),
            "rank": ranks[str(RANK_RTOL)], "rtol": RANK_RTOL,
            "absolute_threshold": RANK_RTOL * scale,
            "ranks_at_fixed_relative_tolerances": ranks}


def nullspace(matrix):
    matrix = finite_array(matrix)
    _, _, vh = np.linalg.svd(matrix, full_matrices=True)
    return vh[rank_info(matrix)["rank"]:].T


def orthogonal_residual(basis, values):
    """Remove a specified linear nuisance span; this does not fit any pose."""
    basis, values = finite_array(basis), finite_array(values)
    u, _, _ = np.linalg.svd(basis, full_matrices=False)
    q = u[:, :rank_info(basis)["rank"]]
    return values - q @ (q.T @ values)


def fixtures():
    spec = FIXTURE_SPEC
    uv = np.asarray(spec["reference_uv"], dtype=float)
    t = np.asarray(spec["translation_before_normalization"], dtype=float)
    t /= np.linalg.norm(t)
    alpha, beta, gamma = spec["plane_alpha_beta_gamma"]
    return {"uv": uv, "depth": np.array(spec["generic_depth"]),
            "theta": np.array(spec["generic_theta"]),
            "R": rotation_exp(spec["rotation_vector"]), "t": t,
            "plane_depth": 1 / (alpha * uv[:, 0] + beta * uv[:, 1] + gamma),
            "plane_theta": np.array(spec["plane_theta"]),
            "plane_null": np.array([gamma, alpha, beta, -1., 0., 0.]),
            "query_uv": np.array(spec["query_uv"]),
            "query_depth": np.array(spec["query_depth"])}


def evaluate():
    f = fixtures()
    uv, d, theta, R, t = (f[k] for k in ("uv", "depth", "theta", "R", "t"))
    design = phi(uv, d)
    j = reference_jacobian(uv, d, theta, R, t)
    jf = reference_jacobian(uv, d, theta, R, t, False)
    gauge = np.r_[theta, np.zeros(3), t]
    add = np.array([0., 0., 0., 1., 0., 0.])
    b = j[:, :6] @ add
    base = reference_prediction(uv, d, theta, R, t)
    c = FIXTURE_SPEC["finite_scale_factor"]
    scaled = reference_prediction(uv, d, c * theta, R, c * t)
    shifted = reference_prediction(uv, d, theta + .15 * add, R, t)
    pd, pt, pn = (f[k] for k in ("plane_depth", "plane_theta", "plane_null"))
    pp = phi(uv, pd)
    jp = reference_jacobian(uv, pd, pt, R, t)
    qphi = phi(f["query_uv"], f["query_depth"])
    query_depth = qphi @ pt
    nuisance = np.column_stack((query_depth, np.ones(len(query_depth))))
    visible = orthogonal_residual(nuisance, qphi @ pn)
    ref_null = nullspace(jp)
    null_visibility = orthogonal_residual(nuisance, qphi @ ref_null[:6])
    epsilon = FIXTURE_SPEC["plane_null_epsilon"]
    moved_pt = pt + epsilon * pn
    ref_before = reference_prediction(uv, pd, pt, R, t)
    ref_after = reference_prediction(uv, pd, moved_pt, R, t)
    jzero = reference_jacobian(uv, d, theta, R, np.zeros(3), False)
    zero_before = reference_prediction(uv, d, theta, R, np.zeros(3))
    zero_after = reference_prediction(uv, d, theta + .15 * add, R, np.zeros(3))
    return {
        "schema": "evr0021.identifiability.v1", "fixtures": FIXTURE_SPEC,
        "generic": {"Phi": design.tolist(), "theta": theta.tolist(),
                    "R": R.tolist(), "unit_t": t.tolist(),
                    "target_uv": base.tolist(), "unit_tangent_basis": tangent_basis(t).tolist(),
                    "phi_rank": rank_info(design), "pose_rank": rank_info(j[:, 6:]),
                    "unit_joint_rank": rank_info(j), "free_joint_rank": rank_info(jf),
                    "analytic_unit_jacobian": j.tolist()},
        "gauge": {"free_scale_null_vector": gauge.tolist(),
                  "free_scale_J_times_null_norm": float(np.linalg.norm(jf @ gauge)),
                  "finite_scale_projection_max_difference": float(np.max(np.abs(base - scaled))),
                  "finite_shift_projection_max_difference": float(np.max(np.abs(base - shifted))),
                  "additive_shift_J_norm": float(np.linalg.norm(b)),
                  "additive_shift_after_pose_projection_norm": float(np.linalg.norm(
                      orthogonal_residual(j[:, 6:], b)))},
        "plane": {"depth": pd.tolist(), "Phi": pp.tolist(), "phi_rank": rank_info(pp),
                  "unit_joint_rank": rank_info(jp), "known_null_vector": pn.tolist(),
                  "Phi_times_null_norm": float(np.linalg.norm(pp @ pn)),
                  "J_times_known_null_norm": float(np.linalg.norm(jp @ np.r_[pn, np.zeros(5)])),
                  "finite_epsilon_reference_projection_max_difference": float(np.max(np.abs(
                      ref_before - ref_after))),
                  "moved_theta": moved_pt.tolist(),
                  "reference_corrected_depth_min": float(np.min(pp @ moved_pt)),
                  "query_corrected_depth_min": float(np.min(qphi @ moved_pt)),
                  "slope_min_on_fixed_reference_and_query_support": float(np.min(
                      np.column_stack((np.ones(len(uv) + len(f["query_uv"])),
                                       np.vstack((uv, f["query_uv"])))) @ moved_pt[:3])),
                  "svd_reference_nullspace": ref_null.tolist(),
                  "svd_nullspace_reference_residual_norm": float(np.linalg.norm(jp @ ref_null))},
        "query_depth_shape_only": {
            "coordinates_disjoint_from_reference": not any(np.array_equal(x, y)
                for x in uv for y in f["query_uv"]),
            "Phi": qphi.tolist(), "base_depth": query_depth.tolist(),
            "known_null_raw_depth_change": (qphi @ pn).tolist(),
            "known_null_after_affine_quotient": visible.tolist(),
            "known_null_visible_norm": float(np.linalg.norm(visible)),
            "svd_reference_null_after_affine_quotient_norm": float(np.linalg.norm(null_visibility)),
            "quotient_annihilates_scale_norm": float(np.linalg.norm(
                orthogonal_residual(nuisance, query_depth))),
            "quotient_annihilates_shift_norm": float(np.linalg.norm(
                orthogonal_residual(nuisance, np.ones(len(query_depth))))),
            "no_pose_fit_or_pose_error_computation": True},
        "pure_rotation": {"free_joint_rank": rank_info(jzero),
                          "theta_block_norm": float(np.linalg.norm(jzero[:, :6])),
                          "finite_shift_projection_max_difference": float(np.max(
                              np.abs(zero_before - zero_after)))},
        "limitations": ["No real data or shared-depth-realization qualification.",
                        "Rank is computed only from analytic Jacobians, never finite differences.",
                        "Local rank is not uniqueness, accuracy, conditioning assurance, or novelty.",
                        "Visible query depth ambiguity is not an observed pose error."]}
