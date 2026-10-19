"""Six-coefficient depth-field BA on a supplied, fixed correspondence graph.

No I/O, model loading, feature selection, initialization or labels. Pixels and K
use array coordinates (first center 0,0); only the field basis uses xy + .5.
Depth has already been normalized once by its whole-image valid median.

The optimizer uses the exact derivative of a continuous radial embedding of
the capped block objective. Its tangential derivative at a capped block is NOT
information: robustJ is a separate IRLS information proxy, zero for every
capped/invalid block. Neither matrix is a calibrated uncertainty estimate.
"""
import time
import numpy as np


SCALE2 = 4.0
CAP2 = 256.0
CAP_COST = float(np.log1p(CAP2 / SCALE2))
DEPTH_EPS = 1e-6
SOLVER_OPTIONS = dict(max_accepted=30, max_evaluations=100,
                      ftol=1e-8, xtol=1e-8, gtol=1e-8)


def finite(value, shape=None, name="array"):
    a = np.asarray(value, dtype=np.float64)
    if (shape is not None and a.shape != shape) or not np.isfinite(a).all():
        raise ValueError(name + " has invalid shape or nonfinite entries")
    return a


def skew(v):
    x, y, z = finite(v, (3,), "rotation vector")
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def batch_skew(v):
    v = finite(v)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("expected Nx3 vectors")
    out = np.zeros((len(v), 3, 3))
    out[:, 0, 1] = -v[:, 2]; out[:, 0, 2] = v[:, 1]
    out[:, 1, 0] = v[:, 2]; out[:, 1, 2] = -v[:, 0]
    out[:, 2, 0] = -v[:, 1]; out[:, 2, 1] = v[:, 0]
    return out


def rotation_exp(v):
    v = finite(v, (3,)); a = float(np.linalg.norm(v)); k = skew(v)
    if a < 1e-5:
        aa = a*a
        c1, c2 = 1-aa/6+aa*aa/120, .5-aa/24+aa*aa/720
    else:
        c1, c2 = np.sin(a)/a, (1-np.cos(a))/(a*a)
    return np.eye(3) + c1*k + c2*np.einsum("ij,jk->ik", k, k)


def rotation_left_jacobian(v):
    v = finite(v, (3,)); a = float(np.linalg.norm(v)); k = skew(v)
    if a < 1e-5:
        aa = a*a
        c1, c2 = .5-aa/24+aa*aa/720, 1/6-aa/120+aa*aa/5040
    else:
        c1, c2 = (1-np.cos(a))/(a*a), (a-np.sin(a))/(a*a*a)
    return np.eye(3) + c1*k + c2*np.einsum("ij,jk->ik", k, k)


def rotation_log(r):
    # SciPy's quaternion branch handles near-pi rotations; no rotvec subtraction.
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(finite(r, (3, 3))).as_rotvec()


def robust_blocks(errors, valid):
    """Exact radial residual/Jacobian and distinct zero-at-cap IRLS weight."""
    e = np.asarray(errors, dtype=np.float64)
    valid = np.asarray(valid)
    if e.ndim != 2 or e.shape[1] != 2 or valid.shape != (len(e),) or valid.dtype != bool:
        raise ValueError("expected Nx2 errors and N boolean validity")
    if not np.isfinite(e[valid]).all():
        raise ValueError("valid pixel errors must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        s = np.einsum("ni,ni->n", e, e)
    good = valid & np.isfinite(s)
    # A finite coordinate whose squared norm overflowed is an invalid full cap.
    sq = np.where(good, s, CAP2)
    capped = good & (sq >= CAP2)
    rho = np.log1p(np.minimum(sq, CAP2) / SCALE2)
    residual = np.tile([np.sqrt(CAP_COST), 0.], (len(e), 1))
    transform = np.zeros((len(e), 2, 2))
    weight = np.zeros(len(e))
    zero = good & (sq == 0)
    residual[zero] = 0.; transform[zero] = .5*np.eye(2); weight[zero] = .5
    active = good & (sq > 0)
    if active.any():
        ss, rr = sq[active], rho[active]
        alpha = np.sqrt(rr / ss)
        unit = e[active] / np.sqrt(ss)[:, None]
        radial = np.where(ss < CAP2, 1. / ((SCALE2 + ss)*alpha), 0.)
        transform[active] = (alpha[:, None, None]*np.eye(2)
                            + (radial-alpha)[:, None, None]*np.einsum("ni,nj->nij", unit, unit))
        residual[active] = alpha[:, None]*e[active]
        weight[active] = np.where(ss < CAP2, 1/np.sqrt(SCALE2+ss), 0.)
    return dict(residual=residual, transform=transform, sqrt_weight=weight,
                valid=good,
                squared_pixel_errors=sq, cost=float(rho.sum()),
                invalid_count=int((~good).sum()), capped_count=int(capped.sum()))


class _FitLimit(Exception):
    pass


def solve_trf(evaluate, x0, max_accepted=30, max_evaluations=100,
              ftol=1e-8, xtol=1e-8, gtol=1e-8):
    """Sparse TRF with hard objective/accepted-step limits and exact retention.

    SciPy 1.13 calls analytic jac only at initial/accepted points. Initial jac
    counts as zero accepted steps. A limit stops before another iteration and
    returns the last accepted point, never the most recent rejected trial.
    """
    from scipy.optimize import least_squares
    from scipy.sparse import csr_matrix
    x0 = finite(x0, name="initial chart")
    if x0.ndim != 1 or not len(x0) or max_accepted < 1 or max_evaluations < 1:
        raise ValueError("invalid fit dimensions or limits")
    cache = {}; accepted = {}; count = 0; steps = 0; start = time.perf_counter()

    def calculate(x):
        nonlocal count
        if "x" not in cache or not np.array_equal(cache["x"], x):
            if count >= max_evaluations:
                raise _FitLimit("evaluation_limit")
            residual, jacobian = evaluate(x)
            residual = finite(residual, name="residual")
            jacobian = csr_matrix(jacobian)
            if residual.ndim != 1 or jacobian.shape != (len(residual), len(x)) or not np.isfinite(jacobian.data).all():
                raise ValueError("nonfinite or incompatible analytic Jacobian")
            cache.update(x=x.copy(), residual=residual.copy(), jacobian=jacobian.copy())
            count += 1
        return cache

    def fun(x):
        return calculate(x)["residual"]

    def jac(x):
        nonlocal steps
        current = calculate(x)
        if accepted and not np.array_equal(accepted["x"], x):
            steps += 1
        accepted.update({k: v.copy() for k, v in current.items()})
        if steps >= max_accepted:
            raise _FitLimit("accepted_step_limit")
        return current["jacobian"]

    converged = False
    try:
        outcome = least_squares(fun, x0, jac=jac, method="trf", tr_solver="lsmr",
                                x_scale=1.0, max_nfev=max_evaluations,
                                ftol=ftol, xtol=xtol, gtol=gtol)
        reason = str(outcome.message)
        converged = bool(outcome.success)
        # A returned accepted point must agree with the last analytic-jac point.
        if not np.array_equal(outcome.x, accepted["x"]):
            raise RuntimeError("SciPy accepted-state contract changed")
    except _FitLimit as exc:
        reason = str(exc)
    if not accepted:
        raise RuntimeError("no accepted initial state")
    return dict(**accepted, converged=converged, termination_reason=reason,
                accepted_steps=steps, evaluations=count, seconds=time.perf_counter()-start)


class FieldProblem:
    """Owns copied fixed inputs and one shared anchor-scale state layout."""
    def __init__(self, images, edges, poses, anchor="A0"):
        from scipy.sparse import csr_matrix
        self.csr_matrix = csr_matrix
        if anchor not in images or set(images) != set(poses) or len(images) < 2:
            raise ValueError("anchor or pose/image mismatch")
        self.anchor = anchor
        self.ids = [anchor] + sorted(k for k in images if k != anchor)
        self.images = {}; self.edges = []; self.layout = {}; dimension = 0
        self.initial = {"poses": {}, "theta": {}}
        for key in self.ids:
            item = images[key]; xy = finite(item["xy"], name="pixels").copy()
            if xy.ndim != 2 or xy.shape[1] != 2:
                raise ValueError("pixels must have shape Nx2")
            raw_depth = np.asarray(item["depth"], dtype=np.float64)
            depth_valid = np.asarray(item.get("depth_valid", np.ones(len(xy), dtype=bool)))
            if (raw_depth.shape != (len(xy),) or depth_valid.shape != (len(xy),)
                    or depth_valid.dtype != bool):
                raise ValueError("depth and boolean depth_valid must match pixels")
            if not np.isfinite(raw_depth[depth_valid]).all() or np.any(raw_depth[depth_valid] <= 0):
                raise ValueError("valid normalized depth samples must be positive and finite")
            # Placeholder is used only to keep algebra finite; its source
            # directions always retain a full capped cost and zero information.
            d = np.where(depth_valid, raw_depth, 1.).copy()
            k = finite(item["K"], (3, 3), "calibration").copy()
            if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.array_equal(k[2], [0., 0., 1.]) or k[0, 1] != 0 or k[1, 0] != 0:
                raise ValueError("canonical zero-skew calibrated PINHOLE required")
            width, height = item["width"], item["height"]
            if type(width) is not int or type(height) is not int or min(width, height) <= 0:
                raise ValueError("positive integer image dimensions required")
            uv = 2*(xy+.5)/np.array([width, height])-1
            phi = np.column_stack((d, uv[:, 0]*d, uv[:, 1]*d, np.ones(len(d)), uv))
            rays = np.column_stack(((xy[:, 0]-k[0, 2])/k[0, 0],
                                    (xy[:, 1]-k[1, 2])/k[1, 1], np.ones(len(xy))))
            theta = finite(item.get("theta", [1., 0, 0, 0, 0, 0]), (6,), "field").copy()
            if theta[0] <= 0 or (key == anchor and theta[0] != 1.):
                raise ValueError("positive field scale; anchor scale exactly one")
            r, t = poses[key]; r = finite(r, (3, 3)).copy(); t = finite(t, (3,)).copy()
            if not np.allclose(np.einsum("ji,jk->ik", r, r), np.eye(3), atol=1e-9, rtol=0) or abs(np.linalg.det(r)-1) > 1e-9:
                raise ValueError("rotation must be SO(3)")
            if key == anchor and (not np.array_equal(r, np.eye(3)) or not np.array_equal(t, np.zeros(3))):
                raise ValueError("anchor pose must be exactly identity")
            pose_columns = None if key == anchor else np.arange(dimension, dimension+6)
            if pose_columns is not None:
                dimension += 6
            field_columns = np.arange(dimension, dimension+(5 if key == anchor else 6))
            dimension += len(field_columns)
            self.layout[key] = dict(pose=pose_columns, field=field_columns)
            self.images[key] = dict(xy=xy, depth=d, raw_depth=raw_depth.copy(),
                                    depth_valid=depth_valid.copy(), K=k, phi=phi, rays=rays,
                                    width=width, height=height)
            self.initial["poses"][key] = (r, t); self.initial["theta"][key] = theta
        self.dimension = dimension
        for edge in edges:
            i, j = edge["i"], edge["j"]
            if i not in images or j not in images or i == j:
                raise ValueError("invalid edge endpoints")
            ids = []
            for name, key in (("ids_i", i), ("ids_j", j)):
                a = np.asarray(edge[name])
                if a.ndim != 1 or a.dtype.kind not in "iu" or np.any(a < 0) or np.any(a >= len(self.images[key]["xy"])):
                    raise ValueError("invalid feature indices")
                ids.append(a.astype(np.int64, copy=True))
            if len(ids[0]) != len(ids[1]) or not len(ids[0]):
                raise ValueError("empty or incompatible edge")
            self.edges.append(dict(i=i, j=j, ids_i=ids[0], ids_j=ids[1]))
        if not self.edges:
            raise ValueError("at least one fixed edge required")

    def retract(self, base, delta):
        delta = finite(delta, (self.dimension,), "state chart")
        state = {"poses": {}, "theta": {}}
        for key in self.ids:
            r, t = base["poses"][key]; theta = base["theta"][key].copy()
            pose, field = self.layout[key]["pose"], self.layout[key]["field"]
            if pose is not None:
                r = np.einsum("ij,jk->ik", rotation_exp(delta[pose[:3]]), r)
                t = t + delta[pose[3:]]
                with np.errstate(over="raise", invalid="raise"):
                    theta[0] *= np.exp(delta[field[0]])
                theta[1:] += delta[field[1:]]
            else:
                theta[1:] += delta[field]
            state["poses"][key] = (r.copy(), t.copy()); state["theta"][key] = theta
        return state

    def query_columns(self, key):
        if key == self.anchor or key not in self.layout:
            raise ValueError("query must be a nonanchor image")
        return np.concatenate((self.layout[key]["pose"], self.layout[self.anchor]["field"],
                               self.layout[key]["field"]))

    def local(self, state, edges=None):
        """Residual, exact local Jacobian, IRLS local Jacobian, block metadata."""
        for key in self.ids:
            finite(state["poses"][key][0], (3, 3), "state rotation")
            finite(state["poses"][key][1], (3,), "state translation")
            finite(state["theta"][key], (6,), "state field")
        rows = []; columns = []; values = []; info_values = []; residuals = []
        squared = []; invalid = 0; capped = 0; row_base = 0
        for edge in self.edges if edges is None else edges:
            for reverse in (False, True):
                i, j, si, sj = ((edge["j"], edge["i"], edge["ids_j"], edge["ids_i"])
                                 if reverse else (edge["i"], edge["j"], edge["ids_i"], edge["ids_j"]))
                source, target = self.images[i], self.images[j]
                ri, ti = state["poses"][i]; rj, tj = state["poses"][j]
                rji = np.einsum("ij,kj->ik", rj, ri)
                phi, rays = source["phi"][si], source["rays"][si]
                # Finite states can still overflow in projection arithmetic.
                # Such blocks retain full cap; no nonfinite product is reused
                # in their derivatives. Nonfinite state parameters are errors.
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    depth = np.einsum("ni,i->n", phi, state["theta"][i])
                    a = rays*depth[:, None]-ti
                    v = np.einsum("ij,nj->ni", rji, a)
                    point = v+tj
                    good = (source["depth_valid"][si] & np.isfinite(depth) & (depth > DEPTH_EPS)
                            & (point[:, 2] > DEPTH_EPS) & np.isfinite(point).all(axis=1))
                    z = np.where(good, point[:, 2], 1.)
                    k = target["K"]; pixel = point[:, :2]/z[:, None]*[k[0, 0], k[1, 1]]+[k[0, 2], k[1, 2]]
                    error = pixel-target["xy"][sj]
                good &= np.isfinite(error).all(axis=1)
                robust = robust_blocks(error, good)
                good = robust["valid"]  # Includes squared-error overflow.
                safe_point = np.where(good[:, None], point, 0.)
                safe_a = np.where(good[:, None], a, 0.)
                safe_v = np.where(good[:, None], v, 0.)
                z = np.where(good, point[:, 2], 1.)
                residuals.append(robust["residual"].ravel()); squared.append(robust["squared_pixel_errors"])
                invalid += robust["invalid_count"]; capped += robust["capped_count"]
                p = np.zeros((len(si), 2, 3))
                p[:, 0, 0] = k[0, 0]/z; p[:, 1, 1] = k[1, 1]/z
                p[:, 0, 2] = -(k[0, 0]/z)*(safe_point[:, 0]/z)
                p[:, 1, 2] = -(k[1, 1]/z)*(safe_point[:, 1]/z)
                p[~good] = 0
                blocks = []
                if self.layout[i]["pose"] is not None:
                    pr = np.einsum("nij,jk->nik", p, rji)
                    di = np.concatenate((np.einsum("nij,njk->nik", pr, batch_skew(safe_a)), -pr), axis=2)
                    blocks.append((self.layout[i]["pose"], di))
                if self.layout[j]["pose"] is not None:
                    dj = np.concatenate((-np.einsum("nij,njk->nik", p, batch_skew(safe_v)), p), axis=2)
                    blocks.append((self.layout[j]["pose"], dj))
                field_phi = np.where(good[:, None], phi[:, 1:] if i == self.anchor else phi, 0.)
                if i != self.anchor:
                    field_phi[:, 0] *= state["theta"][i][0]
                direction = np.einsum("ij,nj->ni", rji, np.where(good[:, None], rays, 0.))
                projected_direction = np.einsum("nij,nj->ni", p, direction)
                blocks.append((self.layout[i]["field"], projected_direction[:, :, None]*field_phi[:, None, :]))
                for cols, raw in blocks:
                    exact = np.einsum("nij,njk->nik", robust["transform"], raw)
                    information = robust["sqrt_weight"][:, None, None]*raw
                    rr = np.broadcast_to((row_base+np.arange(2*len(si))).reshape(-1, 2, 1), exact.shape)
                    cc = np.broadcast_to(cols, exact.shape)
                    rows.append(rr.ravel()); columns.append(cc.ravel()); values.append(exact.ravel()); info_values.append(information.ravel())
                row_base += 2*len(si)
        shape = (row_base, self.dimension)
        indices = (np.concatenate(rows), np.concatenate(columns))
        exact = self.csr_matrix((np.concatenate(values), indices), shape=shape)
        irls = self.csr_matrix((np.concatenate(info_values), indices), shape=shape)
        if not np.isfinite(exact.data).all() or not np.isfinite(irls.data).all():
            raise FloatingPointError("nonfinite field Jacobian")
        return np.concatenate(residuals), exact, irls, dict(squared_pixel_errors=np.concatenate(squared),
                                                           invalid_count=invalid, capped_count=capped)

    def evaluate(self, delta, base=None):
        from scipy.sparse import lil_matrix
        base = self.initial if base is None else base
        state = self.retract(base, delta)
        residual, local, _, _ = self.local(state)
        chart = lil_matrix(np.eye(self.dimension))
        for key in self.ids:
            cols = self.layout[key]["pose"]
            if cols is not None:
                chart[np.ix_(cols[:3], cols[:3])] = rotation_left_jacobian(delta[cols[:3]])
        return residual, local @ chart.tocsr()


def fit_field(images, edges, poses, options=None, anchor="A0"):
    start = time.perf_counter(); problem = FieldProblem(images, edges, poses, anchor)
    settings = dict(SOLVER_OPTIONS)
    if options:
        if set(options)-set(settings):
            raise ValueError("unknown optimizer settings")
        settings.update(options)
    optimized = solve_trf(problem.evaluate, np.zeros(problem.dimension), **settings)
    state = problem.retract(problem.initial, optimized["x"])
    residual, jac, robust_j, meta = problem.local(state)
    return dict(problem=problem, state=state, poses=state["poses"], theta=state["theta"],
                residual=residual, jacobian=jac, robustJ=robust_j, block_errors=meta["squared_pixel_errors"],
                objective=float(np.dot(residual, residual)), diagnostics=meta,
                converged=optimized["converged"], termination_reason=optimized["termination_reason"],
                accepted_steps=optimized["accepted_steps"], evaluations=optimized["evaluations"],
                settings=settings, seconds=time.perf_counter()-start,
                chart="left rotation; additive W2C translation; log(a), five additive coefficients; A0 fixed pose/a",
                information="IRLS proxy sqrt(1/(4+s))*Jpixel for valid s<256, zero otherwise")


def jacobian_at(solution):
    """Final local-chart IRLS information (not the fixed optimizer chart)."""
    return solution["problem"].local(solution["state"])[2]


def state_delta(base_solution, other_solution):
    """Other state in base local chart; permits a deletion state missing views."""
    problem = base_solution["problem"]; base = base_solution["state"]; other = other_solution["state"]
    delta = np.full(problem.dimension, np.nan)
    for key in problem.ids:
        if key not in other["poses"] or key not in other["theta"]:
            continue
        cols = problem.layout[key]["pose"]; field = problem.layout[key]["field"]
        rb, tb = base["poses"][key]; ro, to = other["poses"][key]
        if cols is not None:
            delta[cols[:3]] = rotation_log(np.einsum("ij,kj->ik", ro, rb))
            delta[cols[3:]] = to-tb
            delta[field[0]] = np.log(other["theta"][key][0]/base["theta"][key][0])
            delta[field[1:]] = other["theta"][key][1:]-base["theta"][key][1:]
        else:
            delta[field] = other["theta"][key][1:]-base["theta"][key][1:]
    return delta


def query_residual(solution, edge, w17):
    """Query residual and exact derivative in the shared reference-local chart."""
    problem = solution["problem"]
    if edge["i"] != problem.anchor or edge["j"] == problem.anchor:
        raise ValueError("query edge must be anchor -> query")
    keep = problem.query_columns(edge["j"]); w17 = finite(w17, (17,), "query chart")
    delta = np.zeros(problem.dimension); delta[keep] = w17
    state = problem.retract(solution["state"], delta)
    residual, local, _, _ = problem.local(state, [edge])
    jac = local[:, keep].toarray()
    jac[:, :3] = np.einsum("ni,ij->nj", jac[:, :3], rotation_left_jacobian(w17[:3]))
    return residual, jac
