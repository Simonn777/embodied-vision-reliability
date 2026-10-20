"""Pure numerical implementation of the prospective 0024 falsifier.

No model, file, image-decoder, GT, or network dependencies. Coordinates denote
zero-based pixel centers; depth is axial, not unit-ray range. Invalid sampled
values have a zero sentinel AND an explicit false validity flag.
"""
import math

import numpy as np


GRID = 64
MIN_SUPPORT = 256
RANK_RATIO = 1e-5
DIRECTIONS = ("yaw_to_pitch", "pitch_to_yaw")


def _size(size):
    if len(size) != 2 or any(isinstance(x, (bool, np.bool_)) for x in size):
        raise ValueError("size must contain two integer dimensions")
    w, h = int(size[0]), int(size[1])
    if (w, h) != tuple(size) or min(w, h) < 2:
        raise ValueError("dimensions must be integers >= 2")
    return w, h


def _camera(K):
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("invalid camera")
    if (K[0, 1] != 0 or K[1, 0] != 0 or
            not np.array_equal(K[2], [0., 0., 1.]) or
            K[0, 0] <= 0 or K[1, 1] <= 0):
        raise ValueError("only finite zero-skew pinhole cameras are supported")
    return K


def _rotation(R):
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3) or not np.isfinite(R).all():
        raise ValueError("invalid rotation")
    gram = np.einsum("ji,jk->ik", R, R, optimize=False)
    if not np.allclose(gram, np.eye(3), rtol=0, atol=1e-10) or abs(np.linalg.det(R)-1) > 1e-10:
        raise ValueError("rotation must be proper orthogonal")
    return R


def _rays(K, xy):
    return np.column_stack(((xy[:, 0]-K[0, 2])/K[0, 0],
                            (xy[:, 1]-K[1, 2])/K[1, 1], np.ones(len(xy))))


def _project(K, xyz):
    z = xyz[:, 2]
    xy = np.full((len(xyz), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(xyz).all(axis=1) & (z > 0)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        xy[valid, 0] = K[0, 0]*xyz[valid, 0]/z[valid]+K[0, 2]
        xy[valid, 1] = K[1, 1]*xyz[valid, 1]/z[valid]+K[1, 2]
    return xy, valid


def remap_coordinates(Korig, Kc, R, sizeout):
    """Map every virtual output pixel directly into the original image."""
    Korig, Kc, R = _camera(Korig), _camera(Kc), _rotation(R)
    w, h = _size(sizeout)
    yy, xx = np.indices((h, w), dtype=np.float64)
    rays = _rays(Kc, np.column_stack((xx.ravel(), yy.ravel())))
    source_rays = np.einsum("ji,nj->ni", R, rays, optimize=False)
    xy, valid = _project(Korig, source_rays)
    if not valid.all() or not np.isfinite(xy).all():
        raise ValueError("virtual remap has a nonpositive or nonfinite source ray")
    return xy.reshape(h, w, 2)


def geometry(Korig, sizeorig, sizeout=None):
    """Deterministic canonical camera and identity,+yaw,-yaw,+pitch,-pitch."""
    Korig = _camera(Korig)
    ow, oh = _size(sizeorig)
    if sizeout is None:
        scale = min(1., 512./max(ow, oh))
        sizeout = (int(math.floor(ow*scale+.5)), int(math.floor(oh*scale+.5)))
    w, h = _size(sizeout)
    fx, fy, cx, cy = Korig[0, 0], Korig[1, 1], Korig[0, 2], Korig[1, 2]
    distances = ((cx-2)/fx, (ow-3-cx)/fx, (cy-2)/fy, (oh-3-cy)/fy)
    if min(distances) <= 0 or not np.isfinite(distances).all():
        raise ValueError("original camera lacks the prescribed two-pixel margin")
    angle = min(math.radians(8.), .25*min(math.atan(x) for x in distances))
    c, s = math.cos(angle), math.sin(angle)
    yaw = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    pitch = np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
    rotations = np.stack((np.eye(3), yaw, yaw.T, pitch, pitch.T))
    corners = np.array([[0., 0.], [w-1., 0.], [0., h-1.], [w-1., h-1.]])

    def camera(f):
        return np.array([[f, 0., (w-1)/2], [0., f, (h-1)/2], [0., 0., 1.]])

    def feasible(f):
        rays = _rays(camera(f), corners)
        for R in rotations:
            xyz = np.einsum("ji,nj->ni", R, rays, optimize=False)
            xy, good = _project(Korig, xyz)
            if not good.all() or not ((xy[:, 0] >= 2).all() and
                    (xy[:, 0] <= ow-3).all() and (xy[:, 1] >= 2).all() and
                    (xy[:, 1] <= oh-3).all()):
                return False
        return True

    lo = hi = max(w, h)/100.
    for attempt in range(65):
        if feasible(hi):
            break
        if attempt == 64:
            raise ValueError("no feasible canonical focal after 64 doublings")
        hi *= 2.
    for _ in range(60):
        mid = (lo+hi)/2.
        if feasible(mid):
            hi = mid
        else:
            lo = mid
    focal = hi*(1.+1e-12)
    Kc = camera(focal)
    # Full maps are checked, not only the four projective corner extrema.
    for R in rotations:
        xy = remap_coordinates(Korig, Kc, R, (w, h))
        if not ((xy[..., 0] >= 2).all() and (xy[..., 0] <= ow-3).all() and
                (xy[..., 1] >= 2).all() and (xy[..., 1] <= oh-3).all()):
            raise ValueError("full canonical remap violates the prescribed margin")
    return {"Kc": Kc, "size": (w, h), "rotations": rotations,
            "angle_radians": angle, "focal": focal,
            "fov_x_degrees": math.degrees(2*math.atan(w/(2*focal))),
            "focal_doublings": attempt, "focal_bisections": 60}


def base_grid(size):
    w, h = _size(size)
    jj, ii = np.indices((GRID, GRID))
    xy = np.column_stack((((ii.ravel()+.5)*w/GRID)-.5,
                          ((jj.ravel()+.5)*h/GRID)-.5))
    return xy, ((ii+jj).ravel() % 2 == 0)


def sample(depth, mask, xy):
    """Bilinear sample; all four distinct neighbors must be positive-valid."""
    depth = np.asarray(depth, dtype=np.float64)
    mask = np.asarray(mask)
    xy = np.asarray(xy, dtype=np.float64)
    if depth.ndim != 2 or min(depth.shape) < 2 or mask.shape != depth.shape or mask.dtype != bool:
        raise ValueError("depth must be HxW and mask a same-shaped boolean array")
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must be Nx2")
    h, w = depth.shape
    valid = (np.isfinite(xy).all(axis=1) & (xy[:, 0] >= 0) &
             (xy[:, 0] < w-1) & (xy[:, 1] >= 0) & (xy[:, 1] < h-1))
    values = np.zeros(len(xy), dtype=np.float64)
    ids = np.flatnonzero(valid)
    ix = np.floor(xy[ids, 0]).astype(np.int64)
    iy = np.floor(xy[ids, 1]).astype(np.int64)
    neighbors = np.column_stack((depth[iy, ix], depth[iy, ix+1],
                                 depth[iy+1, ix], depth[iy+1, ix+1]))
    good = (np.isfinite(neighbors).all(axis=1) & (neighbors > 0).all(axis=1) &
            mask[iy, ix] & mask[iy, ix+1] & mask[iy+1, ix] & mask[iy+1, ix+1])
    valid[ids[~good]] = False
    ids, ix, iy, neighbors = ids[good], ix[good], iy[good], neighbors[good]
    dx, dy = xy[ids, 0]-ix, xy[ids, 1]-iy
    weights = np.column_stack(((1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        values[ids] = np.einsum("ni,ni->n", neighbors, weights, optimize=False)
    if not np.isfinite(values).all():
        raise FloatingPointError("nonfinite bilinear arithmetic")
    return values, valid


def _scaled_svd(X):
    norms = np.hypot.reduce(X, axis=0)
    if not np.isfinite(X).all() or not np.isfinite(norms).all():
        raise FloatingPointError("nonfinite least-squares design")
    if (norms == 0).any():
        return None
    U, sigma, Vt = np.linalg.svd(X/norms, full_matrices=False)
    if not (np.isfinite(U).all() and np.isfinite(sigma).all() and np.isfinite(Vt).all()):
        raise FloatingPointError("nonfinite SVD")
    return norms, U, sigma, Vt


def _svd_solution(decomposition, target):
    norms, U, sigma, Vt = decomposition
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        projection = np.einsum("ni,n->i", U, target, optimize=False)/sigma
        result = np.einsum("ji,j->i", Vt, projection, optimize=False)/norms
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite least-squares solution")
    return result


def fit_offset(d0, pair_depths, pair_h):
    """Five-unknown shared-offset OLS on supplied TRAINING samples only."""
    d0 = np.asarray(d0, dtype=np.float64)
    d = np.asarray(pair_depths, dtype=np.float64)
    h = np.asarray(pair_h, dtype=np.float64)
    if d0.ndim != 1 or d.shape != (2, len(d0)) or h.shape != d.shape or len(d0) < 3:
        raise ValueError("invalid shared-offset input shapes")
    if not (np.isfinite(d0).all() and np.isfinite(d).all() and np.isfinite(h).all()) or (h <= 0).any():
        raise ValueError("shared-offset samples must be finite with positive h")
    n = len(d0)
    X = np.zeros((2*n, 5), dtype=np.float64)
    X[:, 0] = -1.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        X[:n, 1], X[:n, 2] = d[0]/h[0], 1./h[0]
        X[n:, 3], X[n:, 4] = d[1]/h[1], 1./h[1]
    decomposition = _scaled_svd(X)
    result = {"usable": False, "reason": "rank_deficient", "coefficients": None,
              "singular_values": [], "singular_ratio": 0., "training_rows": 2*n}
    if decomposition is None:
        return result
    sigma = decomposition[2]
    ratio = float(sigma[-1]/sigma[0]) if sigma[0] > 0 else 0.
    result.update(singular_values=sigma.tolist(), singular_ratio=ratio)
    if len(sigma) != 5 or not (sigma > 0).all() or ratio < RANK_RATIO:
        return result
    coeff = _svd_solution(decomposition, np.tile(d0, 2))
    result["coefficients"] = {"t0": float(coeff[0]),
                              "scales": [float(coeff[1]), float(coeff[3])],
                              "shifts": [float(coeff[2]), float(coeff[4])]}
    if coeff[1] <= 0 or coeff[3] <= 0:
        result["reason"] = "nonpositive_scale"
        return result
    result.update(usable=True, reason=None)
    return result


def fit_nuisance(d0, dj, h, t0):
    """Constrained s>=0 OLS; the boundary prediction is t/h, not mean(y)."""
    d0, dj, h = (np.asarray(v, dtype=np.float64) for v in (d0, dj, h))
    if d0.ndim != 1 or dj.shape != d0.shape or h.shape != d0.shape or not len(d0):
        raise ValueError("invalid nuisance input shapes")
    if not (np.isfinite(d0).all() and np.isfinite(dj).all() and np.isfinite(h).all() and math.isfinite(t0)) or (h <= 0).any():
        raise ValueError("invalid nuisance samples")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        v = 1./h
        X = np.column_stack((dj/h, v))
        y = d0+t0
    decomposition = _scaled_svd(X)
    sigma = [] if decomposition is None else decomposition[2].tolist()
    full_rank = (decomposition is not None and len(sigma) == 2 and
                 sigma[-1] > max(X.shape)*np.finfo(np.float64).eps*sigma[0])
    boundary = "rank_deficient" if not full_rank else None
    coeff = _svd_solution(decomposition, y) if full_rank else None
    if full_rank and coeff[0] < 0:
        boundary = "negative_scale"
    if boundary is not None:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            intercept = np.einsum("n,n->", v, y, optimize=False)/np.einsum("n,n->", v, v, optimize=False)
        coeff = np.array([0., intercept])
    if not np.isfinite(coeff).all():
        raise FloatingPointError("nonfinite constrained nuisance solution")
    return {"scale": float(coeff[0]), "shift": float(coeff[1]),
            "boundary": boundary, "singular_values": sigma,
            "training_rows": len(d0)}


def _mae(d0, dj, h, offset, nuisance):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        residual = (nuisance["scale"]*dj+nuisance["shift"])/h-offset-d0
        value = float(np.mean(np.abs(residual)))
    if not math.isfinite(value):
        raise FloatingPointError("nonfinite held-out error")
    return value


def analyze_samples(values, h, common, train, base_positive):
    """Numerical kernel exposed for manufactured identifiability tests.

    values are already normalized, shape(5,N); h has shape(4,N).
    base_positive contains ALL positive-valid normalized base-image pixels.
    """
    values, h = np.asarray(values, dtype=np.float64), np.asarray(h, dtype=np.float64)
    common, train = np.asarray(common), np.asarray(train)
    base_positive = np.asarray(base_positive, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 5 or h.shape != (4, values.shape[1]):
        raise ValueError("invalid sample matrices")
    n = values.shape[1]
    if common.shape != (n,) or train.shape != (n,) or common.dtype != bool or train.dtype != bool:
        raise ValueError("invalid sample masks")
    if base_positive.ndim != 1 or not len(base_positive) or not np.isfinite(base_positive).all() or (base_positive <= 0).any():
        raise ValueError("invalid full-image positive depth support")
    if not np.isfinite(values[:, common]).all() or not np.isfinite(h[:, common]).all() or (h[:, common] <= 0).any():
        raise ValueError("nonfinite or nonpositive common sample geometry")
    fit = common & train
    test = common & ~train
    result = {"usable_offset": False, "unavailable_reason": None,
              "support": {"total": n, "common": int(common.sum()),
                          "train": int(fit.sum()), "evaluation": int(test.sum())},
              "offsets": {}, "offset_disagreement": None, "directions": {}}
    if fit.sum() < MIN_SUPPORT or test.sum() < MIN_SUPPORT:
        return _unavailable(result, "insufficient_common_support")
    for axis, ids in (("yaw", [0, 1]), ("pitch", [2, 3])):
        fitted = fit_offset(values[0, fit], values[np.array(ids)+1][:, fit], h[ids][:, fit])
        fitted["positive_fraction"] = None
        if fitted["coefficients"] is not None:
            with np.errstate(over="raise", invalid="raise"):
                corrected = base_positive+fitted["coefficients"]["t0"]
            fraction = float(np.count_nonzero(corrected > 0)/len(corrected))
            fitted["positive_fraction"] = fraction
            if fitted["usable"] and fraction < .99:
                fitted.update(usable=False, reason="full_image_positivity")
        result["offsets"][axis] = fitted
    result["usable_offset"] = all(v["usable"] for v in result["offsets"].values())
    coeffs = [result["offsets"][a]["coefficients"] for a in ("yaw", "pitch")]
    if all(v is not None for v in coeffs):
        result["offset_disagreement"] = abs(coeffs[0]["t0"]-coeffs[1]["t0"])
        if not math.isfinite(result["offset_disagreement"]):
            raise FloatingPointError("nonfinite offset disagreement")
    if not result["usable_offset"]:
        result["unavailable_reason"] = "offset_fit_unavailable"
    for name, source, ids in ((DIRECTIONS[0], "yaw", [2, 3]),
                              (DIRECTIONS[1], "pitch", [0, 1])):
        offset = result["offsets"][source]["coefficients"]
        offset = offset["t0"] if result["usable_offset"] else None
        per_view = []
        for j in ids:
            baseline = fit_nuisance(values[0, fit], values[j+1, fit], h[j, fit], 0.)
            error0 = _mae(values[0, test], values[j+1, test], h[j, test], 0., baseline)
            candidate = None
            error = error0
            if offset is not None:
                candidate = fit_nuisance(values[0, fit], values[j+1, fit], h[j, fit], offset)
                error = _mae(values[0, test], values[j+1, test], h[j, test], offset, candidate)
            per_view.append({"view_index": j+1, "zero_offset_fit": baseline,
                             "candidate_fit": candidate, "zero_offset_error": error0,
                             "candidate_error": error})
        result["directions"][name] = {"transferred_offset": offset,
            "zero_offset_error": sum(v["zero_offset_error"] for v in per_view)/2.,
            "candidate_error": sum(v["candidate_error"] for v in per_view)/2.,
            "fallback_to_baseline": offset is None, "views": per_view}
    return result


def _unavailable(result, reason):
    result["unavailable_reason"] = reason
    result["usable_offset"] = False
    result["directions"] = {name: {"transferred_offset": None,
        "candidate_error": 1., "zero_offset_error": 1.,
        "fallback_to_baseline": True, "views": []} for name in DIRECTIONS}
    return result


def analyze_image(depths, masks, Kc, rotations):
    """Compute the complete cross-axis diagnostic, without GT or I/O."""
    depths = np.asarray(depths, dtype=np.float64)
    masks = np.asarray(masks)
    Kc = _camera(Kc)
    rotations = np.asarray(rotations, dtype=np.float64)
    if depths.ndim != 3 or depths.shape[0] != 5 or min(depths.shape[1:]) < 2 or masks.shape != depths.shape or masks.dtype != bool:
        raise ValueError("expected five equal-sized depth maps and boolean masks")
    if rotations.shape != (5, 3, 3) or not np.allclose(rotations[0], np.eye(3), rtol=0, atol=1e-12):
        raise ValueError("five rotations must start with identity")
    for R in rotations:
        _rotation(R)
    height, width = depths.shape[1:]
    if Kc[0, 0] != Kc[1, 1] or Kc[0, 2] != (width-1)/2 or Kc[1, 2] != (height-1)/2:
        raise ValueError("analysis requires the declared canonical camera")
    positive = masks & np.isfinite(depths) & (depths > 0)
    medians = [float(np.median(depths[j][positive[j]])) if positive[j].any() else None for j in range(5)]
    metadata = {"medians": medians, "positive_valid_pixel_counts": [int(v.sum()) for v in positive],
                "size": [width, height]}
    if any(m is None for m in medians):
        return _unavailable(dict(metadata, support={"total": GRID*GRID, "common": 0,
            "train": 0, "evaluation": 0}, offsets={}, offset_disagreement=None), "positive_valid_median_unavailable")
    if any(not math.isfinite(m) or m <= 0 for m in medians):
        raise FloatingPointError("nonfinite or nonpositive median arithmetic")
    xy, train = base_grid((width, height))
    rays = _rays(Kc, xy)
    values, validity, hs = [], [], []
    for j in range(5):
        if j == 0:
            coords, geometric = xy, np.ones(len(xy), dtype=bool)
        else:
            rotated = np.einsum("ij,nj->ni", rotations[j], rays, optimize=False)
            coords, geometric = _project(Kc, rotated)
            hs.append(rotated[:, 2])
        sampled, good = sample(depths[j], masks[j], coords)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            values.append(sampled/medians[j])
        validity.append(good & geometric)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        base_positive = depths[0][positive[0]]/medians[0]
    result = analyze_samples(np.stack(values), np.stack(hs), np.logical_and.reduce(validity), train, base_positive)
    result.update(metadata)
    return result


def _relative_gain(candidate, baseline):
    if baseline == 0:
        return 0. if candidate == 0 else None
    return 1.-candidate/baseline


def summarize(rows):
    """Exactly 27 images/three places/nine images each, 54 directional errors."""
    if len(rows) != 27:
        raise ValueError("the frozen cohort contains exactly 27 images")
    grouped, identities = {}, set()
    for row in rows:
        scene, image_id, analysis = row["scene"], row["image_id"], row["analysis"]
        if not isinstance(scene, str) or not isinstance(image_id, str) or not scene or not image_id or (scene, image_id) in identities:
            raise ValueError("invalid or duplicate image identity")
        identities.add((scene, image_id))
        if not isinstance(analysis["usable_offset"], (bool, np.bool_)):
            raise ValueError("usable_offset must be boolean")
        errors = []
        for name in DIRECTIONS:
            d = analysis["directions"][name]
            candidate, baseline = float(d["candidate_error"]), float(d["zero_offset_error"])
            if not math.isfinite(candidate) or not math.isfinite(baseline) or min(candidate, baseline) < 0:
                raise ValueError("directional errors must be finite and nonnegative")
            if not analysis["usable_offset"] and candidate != baseline:
                raise ValueError("unavailable images must exactly tie their baseline")
            errors.append((candidate, baseline))
        grouped.setdefault(scene, []).append((bool(analysis["usable_offset"]), errors))
    if len(grouped) != 3 or any(len(v) != 9 for v in grouped.values()):
        raise ValueError("the frozen cohort requires three scenes of nine images")
    scenes = {}
    for scene, items in grouped.items():
        candidate = math.fsum(e[0] for _, pair in items for e in pair)/18.
        baseline = math.fsum(e[1] for _, pair in items for e in pair)/18.
        gain = _relative_gain(candidate, baseline)
        scenes[scene] = {"image_count": 9, "direction_count": 18,
            "usable_offset_count": sum(v[0] for v in items),
            "candidate_mean": candidate, "zero_offset_mean": baseline,
            "relative_gain": gain, "strictly_improved": candidate < baseline,
            "floor_pass": gain is not None and gain >= -.02}
    usable = sum(v["usable_offset_count"] for v in scenes.values())
    candidate = math.fsum(e[0] for items in grouped.values() for _, pair in items for e in pair)/54.
    baseline = math.fsum(e[1] for items in grouped.values() for _, pair in items for e in pair)/54.
    gain = _relative_gain(candidate, baseline)
    gates = {"coverage": usable >= 19 and all(v["usable_offset_count"] >= 5 for v in scenes.values()),
             "pooled_gain": gain is not None and gain >= .10,
             "scene_improvement": sum(v["strictly_improved"] for v in scenes.values()) >= 2,
             "scene_floor": all(v["floor_pass"] for v in scenes.values())}
    return {"image_count": 27, "direction_count": 54, "usable_offset_count": usable,
            "candidate_mean": candidate, "zero_offset_mean": baseline,
            "relative_gain": gain, "scenes": scenes, "gates": gates,
            "pass": all(gates.values()), "interpretation": "necessary_predictor_model_transfer_test_only"}
