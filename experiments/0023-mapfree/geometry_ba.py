"""Original bounded calibrated track/point BA control for 0023.

No I/O, label access, feature extraction or automatic native import. Native
initializers require an explicitly injected PoseLib-3 backend. Pixels/K use
array-center coordinates. Depth is used only once to fix arbitrary global scale;
its whole-image normalization belongs to the caller, never to this module.
"""
import time
import numpy as np
from field_ba import (finite, skew, rotation_exp, rotation_left_jacobian,
                      robust_blocks, solve_trf)

NATIVE_OPTIONS = {"max_error": 2.0, "ransac": {"seed": 0, "min_iterations": 100,
                                                            "max_iterations": 1000}}
DEPTH_EPS = 1e-6
RANK_RTOL = 1e-8


class InitializationFailure(RuntimeError):
    """A declared geometric failure; never replaced using a query or labels."""


def _images(images):
    out = {}
    if not images or any(not isinstance(k, str) for k in images):
        raise ValueError("nonempty string image IDs required")
    for key, image in images.items():
        xy = finite(image["xy"], name="pixels").copy()
        k = finite(image["K"], (3, 3), "K").copy()
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("pixels must be Nx2")
        if min(k[0, 0], k[1, 1]) <= 0 or k[0, 1] != 0 or k[1, 0] != 0 or not np.array_equal(k[2], [0., 0., 1.]):
            raise ValueError("canonical PINHOLE K required")
        w, h = image["width"], image["height"]
        if type(w) is not int or type(h) is not int or min(w, h) <= 0:
            raise ValueError("positive integer image dimensions required")
        out[key] = dict(xy=xy, K=k, width=w, height=h)
    return out


def _edge_rows(images, edges):
    rows = []
    for n, edge in enumerate(edges):
        i, j = edge["i"], edge["j"]
        if i == j or i not in images or j not in images:
            raise ValueError("invalid edge endpoints")
        ii, jj = np.asarray(edge["ids_i"]), np.asarray(edge["ids_j"])
        scores = finite(edge["scores"], name="confidence")
        if ii.ndim != 1 or jj.shape != ii.shape or scores.shape != ii.shape:
            raise ValueError("incompatible edge arrays")
        if any(a.dtype.kind not in "iu" or np.any(a < 0) or np.any(a >= len(images[k]["xy"]))
               for a, k in ((ii, i), (jj, j))):
            raise ValueError("feature ID out of range")
        for m, (a, b, s) in enumerate(zip(ii, jj, scores)):
            p, q = (i, int(a)), (j, int(b))
            lo, hi = sorted((p, q))
            rows.append((-float(s), lo, hi, n, m, p, q))
    return sorted(rows)


def build_tracks(images, edges, frozen_reference=None):
    """Confidence-sorted union; reject duplicate-image component conflicts.

    With frozen_reference, only the NEW query edges are sorted: reference
    components/retained edges cannot be retroactively changed by query scores.
    Each leave-view fold must instead build from its own raw fold edges.
    """
    clean = _images(images)
    parent, members = {}, {}

    def add(p):
        if p not in parent:
            parent[p] = p; members[p] = {p[0]: p[1]}

    def find(p):
        add(p)
        while parent[p] != p:
            parent[p] = parent[parent[p]]; p = parent[p]
        return p

    def join(p, q):
        a, b = find(p), find(q)
        if a == b:
            return True
        if any(k in members[b] and v != members[b][k] for k, v in members[a].items()):
            return False
        a, b = sorted((a, b))
        parent[b] = a; members[a].update(members.pop(b))
        return True

    kept = []; seen = set(); conflicts = duplicates = 0
    if frozen_reference is not None:
        # Validate the saved reference edges using this image set, then replay
        # exactly their already accepted connectivity without confidence sorting.
        for edge in frozen_reference["retained_edges"]:
            _edge_rows(clean, [edge])
            copy = {k: (np.asarray(v).copy() if k in ("ids_i", "ids_j", "scores") else v)
                    for k, v in edge.items()}
            kept.append(copy)
            for a, b in zip(edge["ids_i"], edge["ids_j"]):
                p, q = (edge["i"], int(a)), (edge["j"], int(b))
                if not join(p, q):
                    raise ValueError("frozen reference has a track conflict")
                seen.add(tuple(sorted((p, q))))
    accepted = [[] for _ in edges]
    for _, lo, hi, n, m, p, q in _edge_rows(clean, edges):
        if (lo, hi) in seen:
            duplicates += 1; continue
        if not join(p, q):
            conflicts += 1; continue
        seen.add((lo, hi)); accepted[n].append(m)
    for edge, indexes in zip(edges, accepted):
        if indexes:
            indexes = np.asarray(sorted(indexes), dtype=int)
            kept.append(dict(i=edge["i"], j=edge["j"],
                             ids_i=np.asarray(edge["ids_i"], dtype=int)[indexes],
                             ids_j=np.asarray(edge["ids_j"], dtype=int)[indexes],
                             scores=np.asarray(edge["scores"], dtype=float)[indexes]))
    groups = {}
    for p in sorted(parent):
        groups.setdefault(find(p), []).append(p)
    observations = sorted(tuple(sorted(g)) for g in groups.values() if len(g) >= 2)
    tracks = [dict(id=i, observations=list(g)) for i, g in enumerate(observations)]
    return dict(tracks=tracks, retained_edges=kept, conflict_count=conflicts,
                duplicate_count=duplicates, observation_count=sum(len(g) for g in observations),
                match_count=sum(len(e["ids_i"]) for e in kept))


def _pose(value):
    r, t = value
    r = finite(r, (3, 3), "R").copy(); t = finite(t, (3,), "t").copy()
    if not np.allclose(r.T @ r, np.eye(3), rtol=0, atol=1e-8) or abs(np.linalg.det(r)-1) > 1e-8:
        raise ValueError("pose rotation must be SO(3)")
    return r, t


def _camera(image):
    k = image["K"]
    return dict(model="PINHOLE", width=image["width"], height=image["height"],
                params=[float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])])


def _native_result(result, n, absolute=False):
    pose, info = result
    if absolute:
        pose = pose.pose  # Pinned PoseLib-3 absolute wrapper returns Image.
    r = pose.R() if callable(pose.R) else pose.R
    r, t = _pose((r, pose.t))
    mask = np.asarray(info["inliers"])
    if mask.shape != (n,) or mask.dtype != bool:
        raise ValueError("native inlier mask must contain one bool per input")
    return (r, t), mask.copy()


def estimate_pair(images, edge, backend):
    """Supplied query/reference support; no further filtering or depth use.

    A caller may already have filtered query support using reference tracks;
    that use must not be described as strictly query-only information.
    """
    im = _images(images); _edge_rows(im, [edge])
    i, j = edge["i"], edge["j"]
    a = im[i]["xy"][np.asarray(edge["ids_i"], dtype=int)]
    b = im[j]["xy"][np.asarray(edge["ids_j"], dtype=int)]
    if len(a) < 8:
        raise InitializationFailure("relative initializer has fewer than eight matches")
    start = time.perf_counter()
    options = {"max_error": 2.0, "ransac": dict(NATIVE_OPTIONS["ransac"])}
    result = backend.estimate_relative_pose(a, b, _camera(im[i]), _camera(im[j]), options)
    pose, mask = _native_result(result, len(a))
    if np.linalg.norm(pose[1]) <= DEPTH_EPS or int(mask.sum()) < 8:
        raise InitializationFailure("relative initializer has no supported baseline")
    return dict(pose=pose, inliers=mask, input_count=len(a), native_calls=1,
                seconds=time.perf_counter()-start, options=options)


def _ray(image, feature):
    p = image["xy"][feature]; k = image["K"]
    return np.array([(p[0]-k[0, 2])/k[0, 0], (p[1]-k[1, 2])/k[1, 1], 1.])


def triangulate_track(images, observations, poses):
    """Linear multiview triangulation with explicit positive-depth validity.

    Uses every currently registered observation, not monocular depth. Failed
    tracks remain fixed full-cap observations in the BA denominator.
    """
    selected = [(i, int(f)) for i, f in observations if i in poses]
    if len(selected) < 2:
        return None
    a = []
    for i, f in selected:
        r, t = poses[i]; p = np.column_stack((r, t)); ray = _ray(images[i], f)
        a.extend((ray[0]*p[2]-p[0], ray[1]*p[2]-p[1]))
    try:
        _, singular, vh = np.linalg.svd(np.asarray(a), full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if len(singular) < 4 or singular[-2] <= 1e-12*singular[0] or abs(vh[-1, 3]) <= 1e-12:
        return None
    point = vh[-1, :3]/vh[-1, 3]
    if not np.isfinite(point).all() or any((poses[i][0] @ point + poses[i][1])[2] <= DEPTH_EPS for i, _ in selected):
        return None
    return point


def initialize_reference(images, edges, backend, anchor="A0"):
    """Deterministic anchor-pair -> triangulation -> PnP; no fallback edges."""
    start = time.perf_counter(); im = _images(images)
    if anchor not in im:
        raise ValueError("missing anchor")
    graph = build_tracks(images, edges)
    adjacent = [e for e in graph["retained_edges"] if anchor in (e["i"], e["j"])]
    if not adjacent:
        raise InitializationFailure("no retained anchor reference edge")
    # Aggregate all observations of each anchor/partner, independent of how an
    # input file happened to split that edge into rows.
    partners = {}
    for e in adjacent:
        j = e["j"] if e["i"] == anchor else e["i"]
        ai, aj = (e["ids_i"], e["ids_j"]) if e["i"] == anchor else (e["ids_j"], e["ids_i"])
        partners.setdefault(j, []).extend(zip(ai.tolist(), aj.tolist(), e["scores"].tolist()))
    second = min(partners, key=lambda j: (-len(partners[j]), j))
    rows = sorted(partners[second])
    pair = dict(i=anchor, j=second, ids_i=np.array([r[0] for r in rows]),
                ids_j=np.array([r[1] for r in rows]), scores=np.array([r[2] for r in rows]))
    result = estimate_pair(images, pair, backend)
    poses = {anchor: (np.eye(3), np.zeros(3)), second: result["pose"]}
    trusted = {(anchor, int(a), second, int(b)) for a, b, ok in zip(pair["ids_i"], pair["ids_j"], result["inliers"]) if ok}
    points = {}; calls = [dict(kind="relative", image=second, input_count=result["input_count"],
                               inlier_count=int(result["inliers"].sum()), seconds=result["seconds"])]
    for tr in graph["tracks"]:
        obs = dict(tr["observations"])
        if anchor in obs and second in obs and (anchor, obs[anchor], second, obs[second]) in trusted:
            point = triangulate_track(im, tr["observations"], poses)
            if point is not None:
                points[tr["id"]] = point
    while len(poses) < len(im):
        supports = {}
        for key in sorted(set(im)-set(poses)):
            supports[key] = [(tr["id"], dict(tr["observations"])[key]) for tr in graph["tracks"]
                             if tr["id"] in points and key in dict(tr["observations"])]
        key = min(supports, key=lambda k: (-len(supports[k]), k))
        support = supports[key]
        if len(support) < 6:
            raise InitializationFailure("incremental PnP has fewer than six 3D tracks: "+key)
        xy = np.array([im[key]["xy"][f] for _, f in support]); xyz = np.array([points[t] for t, _ in support])
        call_start = time.perf_counter()
        options = {"max_error": 2., "ransac": dict(NATIVE_OPTIONS["ransac"])}
        pose, mask = _native_result(backend.estimate_absolute_pose(xy, xyz, _camera(im[key]), options), len(xy), True)
        if int(mask.sum()) < 6:
            raise InitializationFailure("PnP insufficient inliers: "+key)
        poses[key] = pose
        calls.append(dict(kind="absolute", image=key, input_count=len(xy), inlier_count=int(mask.sum()),
                          seconds=time.perf_counter()-call_start))
        for tr in graph["tracks"]:
            if tr["id"] not in points:
                point = triangulate_track(im, tr["observations"], poses)
                if point is not None:
                    points[tr["id"]] = point
    # Reconstruct all fixed tracks, retaining failure status rather than removing
    # their observations. Initialization-only RANSAC rejection is not BA filtering.
    points = {tr["id"]: triangulate_track(im, tr["observations"], poses) for tr in graph["tracks"]}
    depth = np.asarray(images[anchor]["depth"], dtype=float)
    depth_valid = np.asarray(images[anchor].get("depth_valid", np.ones(len(depth), dtype=bool)))
    if depth.shape != (len(im[anchor]["xy"]),) or depth_valid.shape != depth.shape or depth_valid.dtype != bool:
        raise ValueError("anchor depth and validity shape mismatch")
    if not np.isfinite(depth[depth_valid]).all() or np.any(depth[depth_valid] <= 0):
        raise ValueError("valid anchor depth must be positive and finite")
    z, d = [], []
    for tr in graph["tracks"]:
        point = points[tr["id"]]; obs = dict(tr["observations"])
        if point is not None and anchor in obs and depth_valid[obs[anchor]]:
            z.append(point[2]); d.append(depth[obs[anchor]])
    depth_scale_available = len(z) >= 3
    if depth_scale_available:
        z, d = np.array(z), np.array(d)
        scale = float(np.dot(z, d)/np.dot(z, z))
        if not np.isfinite(scale) or scale <= 0:
            raise InitializationFailure("invalid global depth-unit alignment")
        scale_status = "anchor_depth_ls"
    else:
        # Monocular depth availability cannot disqualify a geometric direction
        # estimate. Use one common positive rescaling of ALL translations and
        # points; the caller separately admits depth-field fits only when the
        # anchor-depth scale is available. Malformed input checks above remain
        # technical failures, not a reason to enter this branch.
        baseline = float(np.linalg.norm(poses[second][1]))
        if not np.isfinite(baseline) or baseline <= DEPTH_EPS:
            raise InitializationFailure("invalid geometric baseline for unit gauge")
        scale = 1.0 / baseline
        scale_status = "unit_baseline_insufficient_anchor_depth"
    poses = {i: (r, t*scale) for i, (r, t) in poses.items()}
    points = {i: (None if p is None else p*scale) for i, p in points.items()}
    return dict(poses=poses, points=points, graph=graph, anchor=anchor, gauge_image=second,
                baseline_norm=float(np.linalg.norm(poses[second][1])),
                depth_scale_available=depth_scale_available, scale_status=scale_status,
                receipt=dict(native_calls=len(calls), calls=calls, global_scale=scale,
                             depth_scale_available=depth_scale_available, scale_status=scale_status,
                             scale_anchor_points=len(z), seconds=time.perf_counter()-start,
                             initialized_tracks=sum(p is not None for p in points.values()),
                             fixed_track_observations=graph["observation_count"]))


def _tangent(unit):
    axis = np.eye(3)[int(np.argmin(np.abs(unit)))]; a = np.cross(unit, axis); a /= np.linalg.norm(a)
    return np.column_stack((a, np.cross(unit, a)))


class GeometryProblem:
    def __init__(self, images, graph, poses, points, gauge_image, anchor="A0"):
        self.images = _images(images); self.graph = graph; self.anchor = anchor; self.gauge_image = gauge_image
        if set(poses) != set(images) or anchor not in poses or gauge_image == anchor or gauge_image not in poses:
            raise ValueError("pose/image/gauge mismatch")
        self.poses = {i: _pose(p) for i, p in poses.items()}
        if not np.array_equal(self.poses[anchor][0], np.eye(3)) or not np.array_equal(self.poses[anchor][1], np.zeros(3)):
            raise ValueError("anchor must be exactly identity")
        self.baseline = float(np.linalg.norm(self.poses[gauge_image][1]))
        if not np.isfinite(self.baseline) or self.baseline <= DEPTH_EPS:
            raise ValueError("invalid scale-fixing baseline")
        self.unit = self.poses[gauge_image][1]/self.baseline; self.basis = _tangent(self.unit)
        self.layout = {}; self.point_layout = {}; self.points = {}; n = 0
        for i in sorted(set(poses)-{anchor}):
            size = 5 if i == gauge_image else 6
            self.layout[i] = np.arange(n, n+size); n += size
        self.pose_dimension = n
        self.observations = []
        for tr in graph["tracks"]:
            tid = tr["id"]
            if tid in self.points:
                raise ValueError("duplicate track ID")
            point = points.get(tid)
            self.points[tid] = None if point is None else finite(point, (3,), "point").copy()
            if point is not None:
                self.point_layout[tid] = np.arange(n, n+3); n += 3
            seen = set()
            for i, f in tr["observations"]:
                if i in seen or i not in images or type(f) not in (int, np.int64) or f < 0 or f >= len(self.images[i]["xy"]):
                    raise ValueError("invalid fixed track observation")
                seen.add(i); self.observations.append((i, int(f), tid))
        if not self.point_layout or not self.observations:
            raise InitializationFailure("no triangulated tracks for BA")
        self.dimension = n

    def state(self, x):
        x = finite(x, (self.dimension,), "chart")
        poses = {self.anchor: self.poses[self.anchor]}; derivatives = {}
        for i, cols in self.layout.items():
            r0, t0 = self.poses[i]; rotation_vector = x[cols[:3]]
            r = rotation_exp(rotation_vector) @ r0
            if i == self.gauge_image:
                v = x[cols[3:]]; radius = float(np.hypot(v[0], v[1])); tangent = self.basis @ v
                if radius < 1e-5:
                    rr = radius*radius
                    sinc = 1-rr/6+rr*rr/120
                    cosine = 1-rr/2+rr*rr/24
                    coefficient = -1/3+rr/30-rr*rr/840
                else:
                    sinc = np.sin(radius)/radius; cosine = np.cos(radius)
                    coefficient = (radius*cosine-np.sin(radius))/(radius**3)
                # Sphere exponential: both hemispheres are reachable. Like all
                # fixed 2D manifold charts it has singular points (k*pi), not a
                # hidden hemisphere restriction. The initializer sets its pole.
                t = self.baseline*(cosine*self.unit+sinc*tangent)
                dt = self.baseline*(sinc*self.basis-sinc*np.outer(self.unit,v)
                                    +coefficient*np.outer(tangent,v))
            else:
                t = t0+x[cols[3:]]; dt = np.eye(3)
            poses[i] = (r, t); derivatives[i] = (rotation_left_jacobian(rotation_vector), dt)
        points = {t: (None if p is None else p+x[self.point_layout[t]]) for t, p in self.points.items()}
        return poses, points, derivatives

    def evaluate(self, x, details=False, information=False):
        from scipy.sparse import csr_matrix
        poses, points, derivatives = self.state(x)
        errors = np.zeros((len(self.observations), 2)); valid = np.zeros(len(errors), dtype=bool)
        blocks = []
        for row, (i, f, tid) in enumerate(self.observations):
            p = points[tid]
            if p is None:
                continue
            r, t = poses[i]; k = self.images[i]["K"]
            with np.errstate(over="ignore", invalid="ignore"):
                rotated = r @ p; q = rotated+t
            if not np.isfinite(q).all() or q[2] <= DEPTH_EPS:
                continue
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                predicted = np.array([k[0, 0]*(q[0]/q[2])+k[0, 2], k[1, 1]*(q[1]/q[2])+k[1, 2]])
                jp = np.array([[k[0, 0]/q[2], 0, -(k[0, 0]/q[2])*(q[0]/q[2])],
                               [0, k[1, 1]/q[2], -(k[1, 1]/q[2])*(q[1]/q[2])]])
                error = predicted-self.images[i]["xy"][f]
            if not np.isfinite(predicted).all() or not np.isfinite(jp).all() or not np.isfinite(error).all():
                continue
            valid[row] = True; errors[row] = error
            blocks.append((row, tid, i, jp, r, rotated))
        robust = robust_blocks(errors, valid)
        rowids, colids, values = [], [], []
        for row, tid, i, jp, r, rotated in blocks:
            if not robust["valid"][row]:
                continue  # Includes squared-error overflow; never 0 times Inf.
            if information and robust["sqrt_weight"][row] == 0:
                continue  # Capped losses carry zero IRLS information.
            local = [(self.point_layout[tid], jp @ r)]
            if i != self.anchor:
                jl, dt = derivatives[i]
                local.append((self.layout[i], np.column_stack((jp @ (-skew(rotated)) @ jl, jp @ dt))))
            for cols, raw in local:
                block = (robust["sqrt_weight"][row] * raw if information
                         else robust["transform"][row] @ raw)
                rowids.extend(np.repeat([2*row, 2*row+1], len(cols)))
                colids.extend(np.tile(cols, 2)); values.extend(block.ravel())
        jac = csr_matrix((values, (rowids, colids)), shape=(2*len(errors), self.dimension))
        output = (robust["residual"].ravel(), jac)
        return (*output, robust) if details else output


def fit_geometry(images, graph, initial, options=None):
    """Fit supplied fixed tracks using common robust loss and sparse solver.

    Query extension may merge reference tracks. Re-triangulate its tracks only
    from the supplied (reference-only initialized) poses, never from labels.
    All failed triangulations keep their full fixed observation cost.
    """
    start = time.perf_counter()
    poses = initial["poses"]; im = _images(images)
    points = {tr["id"]: triangulate_track(im, tr["observations"], poses) for tr in graph["tracks"]}
    problem = GeometryProblem(images, graph, poses, points, initial["gauge_image"], initial.get("anchor", "A0"))
    outcome = solve_trf(problem.evaluate, np.zeros(problem.dimension), **(options or {}))
    fitted_poses, fitted_points, _ = problem.state(outcome["x"])
    _, _, stats = problem.evaluate(outcome["x"], details=True)
    return dict(poses=fitted_poses, points=fitted_points, problem=problem,
                cost=stats["cost"], invalid_count=stats["invalid_count"], capped_count=stats["capped_count"],
                observation_count=len(problem.observations), baseline_norm=problem.baseline,
                converged=outcome["converged"], termination_reason=outcome["termination_reason"],
                accepted_steps=outcome["accepted_steps"], evaluations=outcome["evaluations"],
                seconds=time.perf_counter()-start)


def pose_observability(solution, query_id):
    """Local query rotation/direction identifiability after eliminating points.

    Each track contributes an independent three-column landmark block. Its
    column space is projected out locally, never through a dense all-points
    matrix. Only the remaining pose columns enter a tall, narrow SVD. Capped
    and invalid observations carry zero IRLS information. This is a numerical
    rank diagnostic, not a calibrated uncertainty or a convergence claim.

    Recenter at the final state so a singular fixed optimization chart is not
    mistaken for missing geometric information. Translation columns, including
    the scale-fixing camera's two tangent columns, use normalized-depth units;
    rotation columns use radians. The query functional discards translation
    length, so an otherwise harmless length-only null direction is allowed.
    """
    began = time.perf_counter()
    original = solution["problem"]
    if query_id == original.anchor or query_id not in original.layout:
        raise ValueError("query must be a non-anchor camera")
    problem = GeometryProblem(original.images, original.graph,
                              solution["poses"], solution["points"],
                              original.gauge_image, original.anchor)
    _, jac, robust = problem.evaluate(np.zeros(problem.dimension),
                                     details=True, information=True)
    finite(jac.data, name="geometry IRLS Jacobian")
    n = problem.pose_dimension
    pose = jac[:, :n].toarray()
    # Sphere coordinates in the optimizer are angles. Here one tangent unit
    # means one translation unit, matching every other camera's translation.
    scale = np.ones(n)
    scale[problem.layout[problem.gauge_image][3:]] = 1 / problem.baseline
    pose *= scale[None, :]
    original_singular = np.linalg.svd(pose, compute_uv=False)
    original_scale = float(original_singular[0]) if original_singular.size else 0.
    rows_by_track = {}
    for row, (_, _, tid) in enumerate(problem.observations):
        rows_by_track.setdefault(tid, []).extend((2*row, 2*row+1))
    ranks = [0, 0, 0, 0]
    for tid, rows in rows_by_track.items():
        if tid not in problem.point_layout:
            # Such observations already have zero information and fixed cost.
            ranks[0] += 1
            continue
        point = jac[rows][:, problem.point_layout[tid]].toarray()
        u, singular, _ = np.linalg.svd(point, full_matrices=False)
        rank = int(np.count_nonzero(singular > RANK_RTOL * singular[0])) if len(singular) and singular[0] > 0 else 0
        ranks[rank] += 1
        if rank:
            basis = u[:, :rank]
            block = pose[rows]
            # Explicit contractions avoid a residual-by-residual projector.
            coordinates = np.einsum("ri,rj->ij", basis, block, optimize=False)
            pose[rows] = block - np.einsum("ri,ij->rj", basis, coordinates, optimize=False)
    _, singular, vt = np.linalg.svd(pose, full_matrices=pose.shape[0] < n)
    threshold = RANK_RTOL * original_scale
    rank = int(np.count_nonzero(singular > threshold)) if original_scale > 0 else 0
    null = vt[rank:].T
    t = problem.poses[query_id][1]
    norm = float(np.linalg.norm(t))
    report = dict(rank=rank, dimension=n, null_dimension=n-rank,
                  rank_relative_threshold=RANK_RTOL, rank_threshold=threshold,
                  point_block_rank_counts=ranks, track_count=len(rows_by_track),
                  observation_count=len(problem.observations),
                  informative_observation_count=int(np.count_nonzero(robust["sqrt_weight"])),
                  chart="final left rotation / translation-unit tangent")
    if not np.isfinite(norm) or norm <= DEPTH_EPS:
        return dict(report, observable=False, pose_null_ratio=None,
                    rotation_null_ratio=None, direction_null_ratio=None,
                    combined_pose_null_ratio=None,
                    reason="undefined query translation direction", seconds=time.perf_counter()-began)
    cols = problem.layout[query_id]
    functional = np.zeros((6, n)); functional[:3, cols[:3]] = np.eye(3)
    unit = t / norm
    tangent = (np.eye(3) - np.outer(unit, unit)) / norm
    translation = problem.basis if query_id == problem.gauge_image else np.eye(3)
    functional[3:, cols[3:]] = tangent @ translation
    visible_null = np.einsum("ij,jk->ik", functional, null, optimize=False)
    # Normalize the two output blocks separately: a long translation must not
    # hide a direction null behind the unit-sized rotation derivative.
    rotation_ratio = float(np.linalg.norm(visible_null[:3]) / np.linalg.norm(functional[:3]))
    direction_ratio = float(np.linalg.norm(visible_null[3:]) / np.linalg.norm(functional[3:]))
    combined_ratio = float(np.linalg.norm(visible_null) / np.linalg.norm(functional))
    ratio = max(rotation_ratio, direction_ratio)
    return dict(report, observable=ratio <= RANK_RTOL, pose_null_ratio=ratio,
                rotation_null_ratio=rotation_ratio, direction_null_ratio=direction_ratio,
                combined_pose_null_ratio=combined_ratio,
                reason="query functional null test", seconds=time.perf_counter()-began)
