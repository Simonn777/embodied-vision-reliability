"""Fixed two-start query updates; no input selection, I/O, native calls or GT."""
import time
import numpy as np
import field_ba as field


def alternative_chart(reference, query_id, pair_pose):
    """Query-only direction aligned to the reference estimate's arbitrary norm."""
    base = reference["state"]["poses"][query_id]
    r, t = pair_pose
    r = field.finite(r, (3, 3)); t = field.finite(t, (3,))
    if not np.allclose(np.einsum("ji,jk->ik", r, r), np.eye(3), atol=1e-8, rtol=0) or abs(np.linalg.det(r)-1) > 1e-8:
        raise ValueError("alternative rotation must be SO(3)")
    norm = float(np.linalg.norm(t)); target_norm = float(np.linalg.norm(base[1]))
    if norm <= 1e-6 or target_norm <= 1e-6:
        raise ValueError("undefined translation direction for alternative start")
    w = np.zeros(17)
    w[:3] = field.rotation_log(np.einsum("ij,kj->ik", r, base[0]))
    w[3:6] = t * (target_norm / norm) - base[1]
    return w


def pose_observability(reference, edge, w, precision_factor):
    """Check rotation and translation *direction*, allowing a length-only null.

    Uses the query IRLS information, not tangential derivatives of capped loss.
    This is a local numerical rank diagnostic, never a calibrated error bound.
    """
    problem = reference["problem"]; keep = problem.query_columns(edge["j"])
    delta = np.zeros(problem.dimension); delta[keep] = w
    state = problem.retract(reference["state"], delta)
    qj = problem.local(state, [edge])[2][:, keep].toarray()
    left = field.rotation_left_jacobian(w[:3])
    left_singular = np.linalg.svd(left, compute_uv=False)
    if left_singular[-1] <= 1e-8 * left_singular[0]:
        return dict(observable=False, rank=None, dimension=17, pose_null_ratio=None,
                    reason="singular fixed rotation chart")
    qj[:, :3] = np.einsum("ni,ij->nj", qj[:, :3], left)
    j = np.vstack((qj, precision_factor))
    _, singular, vt = np.linalg.svd(j, full_matrices=j.shape[0] < j.shape[1])
    rank = int(np.count_nonzero(singular > 1e-8 * singular[0])) if len(singular) and singular[0] > 0 else 0
    null = vt[rank:].T
    t = state["poses"][edge["j"]][1]; norm = float(np.linalg.norm(t))
    if not np.isfinite(norm) or norm <= 1e-6:
        return dict(observable=False, rank=rank, dimension=17, pose_null_ratio=None)
    unit = t / norm
    # Redundant 3D tangent projector has rank2 and is coordinate-independent.
    g = np.zeros((6, 17)); g[:3, :3] = left
    g[3:, 3:6] = (np.eye(3) - np.outer(unit, unit)) / norm
    product = np.einsum("ij,jk->ik", g, null)
    ratios = [float(np.linalg.norm(product[s:s+3]) / np.linalg.norm(g[s:s+3])) for s in (0, 3)]
    return dict(observable=max(ratios) <= 1e-8, rank=rank, dimension=17,
                pose_null_ratio=max(ratios), rotation_null_ratio=ratios[0],
                direction_null_ratio=ratios[1], null_dimension=17-rank)


def fit_conditional(reference, edge, precision_factor, pair_pose=None, options=None):
    """One reference prior plus one query factor; all starts charged and saved."""
    began = time.perf_counter()
    factor = field.finite(precision_factor, name="prior information factor")
    if factor.ndim != 2 or factor.shape[1] != 17:
        raise ValueError("prior factor must have17 columns")
    starts = [np.zeros(17)]
    if pair_pose is not None:
        starts.append(alternative_chart(reference, edge["j"], pair_pose))
    def evaluate(w):
        residual, jac = field.query_residual(reference, edge, w)
        return (np.concatenate((residual, np.einsum("ij,j->i", factor, w))),
                np.vstack((jac, factor)))
    attempts = []
    for start in starts:
        result = field.solve_trf(evaluate, start, **(options or {}))
        objective = float(np.dot(result["residual"], result["residual"]))
        attempts.append(dict(x=result["x"], objective=objective,
                             converged=result["converged"], termination_reason=result["termination_reason"],
                             accepted_steps=result["accepted_steps"], evaluations=result["evaluations"],
                             seconds=result["seconds"]))
    selected = min(range(len(attempts)), key=lambda i:(attempts[i]["objective"], i))
    w = attempts[selected]["x"]
    p = reference["problem"]; delta = np.zeros(p.dimension); delta[p.query_columns(edge["j"])] = w
    state = p.retract(reference["state"], delta)
    return dict(pose=state["poses"][edge["j"]], state=state, x=w,
                attempts=attempts, selected=selected, seconds=time.perf_counter()-began,
                observability=pose_observability(reference, edge, w, factor),
                information_count=dict(reference_prior=1, query_edge=1))


def fit_joint(reference, edge, pair_pose=None, options=None):
    """Full nonlinear same-field control with exactly the same two starts."""
    began = time.perf_counter(); problem = reference["problem"]
    images = {i: dict(image, theta=reference["state"]["theta"][i].copy())
              for i, image in problem.images.items()}
    starts = [reference["state"]["poses"]]
    if pair_pose is not None:
        w = alternative_chart(reference, edge["j"], pair_pose)
        delta = np.zeros(problem.dimension); delta[problem.query_columns(edge["j"])] = w
        starts.append(problem.retract(reference["state"], delta)["poses"])
    active_edges = [*problem.edges, edge] if len(edge["ids_i"]) else list(problem.edges)
    attempts = [field.fit_field(images, active_edges, poses, options, problem.anchor)
                for poses in starts]
    selected = min(range(len(attempts)), key=lambda i:(attempts[i]["objective"], i))
    return dict(pose=attempts[selected]["poses"][edge["j"]], attempts=attempts,
                selected=selected, seconds=time.perf_counter()-began,
                information_count=dict(reference_edges=len(problem.edges), query_edge=1))
