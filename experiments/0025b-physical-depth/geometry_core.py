"""Pure geometry/calibration helpers for a prospective RS-depth diagnostic.

No files, models, inference, image remapping library or network access. Coordinates
are zero-based pixel centers. Input truth must already be source-camera axial z,
not Euclidean range. Calibration uses the fixed training half only; prediction
failure never removes points from the independently defined evaluation support.
"""
import math

import numpy as np


SOURCE_WIDTH, SOURCE_HEIGHT = 640, 480
WIDTH, HEIGHT = 448, 336
SOURCE_K = np.array([[481.2, 0., 319.5], [0., 480., 239.5], [0., 0., 1.]])
OUTPUT_K = np.array([[481.2, 0., 223.5], [0., 481.2, 167.5], [0., 0., 1.]])
READOUT_SECONDS = 0.04
TOTAL_ANGLE_RADIANS = math.radians(12.)
VIEWS = ("gs", "rs_x_positive", "rs_x_negative", "rs_y_positive", "rs_y_negative")
ARMS = ("rs_row", "rs_mid", "gs_transport", "rs_scale_only")


def _array(value, shape, name):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape:
        raise ValueError(name+" has incorrect shape")
    return a


def bilinear_inbounds(xy, width, height):
    """All four neighbors must exist, including at integer sample coordinates."""
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape[-1] != 2:
        raise ValueError("coordinate last dimension must be 2")
    return (np.all(np.isfinite(xy), axis=-1) & (xy[..., 0] >= 0.)
            & (xy[..., 0] < width-1) & (xy[..., 1] >= 0.) & (xy[..., 1] < height-1))


def make_view_geometry(view):
    """Return full output-to-source maps; reject any RGB padding requirement.

    Q is W2C row relative to middle. source_ray=Q.T*r_row, h=source_ray.z,
    source z_mid=h*z_row, hence row truth is sampled source axial z divided by h.
    Readout time is retained explicitly; fixed total rotation is 12 degrees.
    """
    if view not in VIEWS:
        raise ValueError("unknown fixed view")
    v, u = np.indices((HEIGHT, WIDTH), dtype=np.float64)
    x, y = (u-OUTPUT_K[0,2])/OUTPUT_K[0,0], (v-OUTPUT_K[1,2])/OUTPUT_K[1,1]
    row_phase = v/(HEIGHT-1)-.5
    row_seconds = READOUT_SECONDS*row_phase
    angle = np.zeros_like(v)
    if view != "gs":
        sign = 1. if view.endswith("positive") else -1.
        angle = sign*TOTAL_ANGLE_RADIANS*row_phase
    c, s = np.cos(angle), np.sin(angle)
    if view.startswith("rs_x"):
        sx, sy, h = x, c*y+s, c-s*y
    elif view.startswith("rs_y"):
        sx, sy, h = c*x-s, y, c+s*x
    else:
        sx, sy, h = x, y, np.ones_like(x)
    if not (np.all(np.isfinite(h)) and np.all(h > 0.)):
        raise ValueError("nonpositive/nonfinite source-ray axial factor")
    rx, ry = sx/h, sy/h
    source_xy = np.stack((SOURCE_K[0,0]*rx+SOURCE_K[0,2], SOURCE_K[1,1]*ry+SOURCE_K[1,2]), axis=-1)
    gs_xy = np.stack((OUTPUT_K[0,0]*rx+OUTPUT_K[0,2], OUTPUT_K[1,1]*ry+OUTPUT_K[1,2]), axis=-1)
    if view == "gs":
        # This is an identity pixel map, exactly; avoid artificial roundoff at 0.
        gs_xy = np.stack((u,v),axis=-1)
    if not np.all(bilinear_inbounds(source_xy, SOURCE_WIDTH, SOURCE_HEIGHT)):
        raise ValueError("fixed full RGB map requires padding")
    return dict(view=view, source_xy=source_xy, canonical_gs_xy=gs_xy, h=h,
                source_ray=np.stack((sx,sy,h),axis=-1), row_angle_radians=angle,
                row_seconds=row_seconds, full_rgb_four_neighbor_inbounds=True)


def fixed_lattice():
    """Exact integer half-up 64x64 lattice; even checker parity is training."""
    i,j = np.indices((64,64),dtype=np.int64)
    u = (2*j*(WIDTH-1)+63)//126
    v = (2*i*(HEIGHT-1)+63)//126
    return dict(u=u.reshape(-1),v=v.reshape(-1),
                train=((i+j)%2 == 0).reshape(-1),
                eval=((i+j)%2 == 1).reshape(-1),count=4096)


def _neighbors(array, xy):
    if array.ndim != 2:
        raise ValueError("bilinear input must be a scalar image")
    valid = bilinear_inbounds(xy,array.shape[1],array.shape[0])
    # Placeholder indices permit vectorized gathering; their values never become
    # valid support or a padded interpolation outside the actual four-neighbor domain.
    safe_xy = np.where(valid[...,None],xy,0.)
    lo = np.floor(safe_xy).astype(np.int64)
    x,y = lo[...,0],lo[...,1]
    dx,dy = safe_xy[...,0]-x,safe_xy[...,1]-y
    neighbors = np.stack((array[y,x],array[y,x+1],array[y+1,x],array[y+1,x+1]),axis=-1)
    weights = np.stack(((1.-dx)*(1.-dy),dx*(1.-dy),(1.-dx)*dy,dx*dy),axis=-1)
    return neighbors,weights,valid


def sample_truth(source_axial_z, geometry, *, max_relative_depth_spread):
    """Model-independent truth/support; depth-gap threshold has no default.

    Spread=(max-min)/min over four positive finite source axial depths. Inverse
    depth is interpolated only after this test. It is exact for a local planar
    surface in a pinhole image, but does not resolve occlusions/curved surfaces.
    """
    if (not np.isscalar(max_relative_depth_spread) or
            not math.isfinite(float(max_relative_depth_spread)) or max_relative_depth_spread < 0.):
        raise ValueError("depth spread threshold must be explicit finite nonnegative")
    truth = _array(source_axial_z,(SOURCE_HEIGHT,SOURCE_WIDTH),"source axial depth")
    lattice = fixed_lattice()
    u,v = lattice["u"],lattice["v"]
    source_xy = _array(geometry["source_xy"],(HEIGHT,WIDTH,2),"source map")[v,u]
    gs_xy = _array(geometry["canonical_gs_xy"],(HEIGHT,WIDTH,2),"GS map")[v,u]
    h = _array(geometry["h"],(HEIGHT,WIDTH),"axial factor")[v,u]
    if not (np.all(np.isfinite(h)) and np.all(h > 0.)):
        raise ValueError("invalid fixed axial factor")
    neighbors,weights,inside = _neighbors(truth,source_xy)
    positive_four = np.all(np.isfinite(neighbors) & (neighbors > 0.),axis=1)
    safe = np.where(positive_four[:,None],neighbors,1.)
    lo,hi = np.min(safe,axis=1),np.max(safe,axis=1)
    with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
        spread = (hi-lo)/lo
    gs_inside = bilinear_inbounds(gs_xy,WIDTH,HEIGHT)
    support = inside & positive_four & np.isfinite(spread) & (spread <= max_relative_depth_spread) & gs_inside
    mid_z = np.full(4096,np.nan)
    with np.errstate(over="ignore",invalid="ignore",divide="ignore",under="ignore"):
        inverse = np.sum(weights[support]/neighbors[support],axis=1)
        mid_z[support] = 1./inverse
        row_z = mid_z/h
    derived_ok = np.all(np.isfinite(row_z[support]) & (row_z[support] > 0.))
    counts = dict(total=4096,support=int(np.count_nonzero(support)),
        train=int(np.count_nonzero(support & lattice["train"])),
        eval=int(np.count_nonzero(support & lattice["eval"])),
        source_positive_four=int(np.count_nonzero(inside & positive_four)),
        canonical_gs_inbounds=int(np.count_nonzero(gs_inside)))
    coverage_ok = counts["support"] >= 2048 and counts["train"] >= 256 and counts["eval"] >= 256
    reason = None if derived_ok and coverage_ok else ("invalid_derived_truth" if not derived_ok else "insufficient_fixed_support")
    return dict(status="usable" if reason is None else "unavailable",reason=reason,
        lattice=lattice,support=support,counts=counts,h=h,source_xy=source_xy,
        canonical_gs_xy=gs_xy,source_mid_z=mid_z,row_z=row_z,
        relative_depth_spread=spread,max_relative_depth_spread=float(max_relative_depth_spread),
        support_uses_model_values=False)


def build_designs(d_rs, d_gs, truth_record):
    """Four row-z LS designs, with the same truth-defined support and no mask.

    Nonfinite model values are retained as invalid predictions, never used to
    shrink support. Finite negative/zero raw values may be repaired by the affine
    offset; only calibrated depth must be positive. GS uses ordinary bilinear
    interpolation, deliberately distinct from inverse-depth truth interpolation.
    """
    rs = _array(d_rs,(HEIGHT,WIDTH),"RS model depth")
    gs = _array(d_gs,(HEIGHT,WIDTH),"GS model depth")
    lattice = truth_record["lattice"]
    dr = rs[lattice["v"],lattice["u"]].copy()
    ng,wg,inside = _neighbors(gs,np.asarray(truth_record["canonical_gs_xy"]))
    valid_gs = inside & np.all(np.isfinite(ng),axis=1)
    dg = np.full(4096,np.nan)
    with np.errstate(over="ignore",invalid="ignore"):
        dg[valid_gs] = np.sum(wg[valid_gs]*ng[valid_gs],axis=1)
    dr[~np.isfinite(dr)] = np.nan
    h = np.asarray(truth_record["h"])
    with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
        designs = dict(rs_row=np.column_stack((dr,np.ones(4096))),
            rs_mid=np.column_stack((dr/h,1./h)),
            gs_transport=np.column_stack((dg/h,1./h)),
            rs_scale_only=dr[:,None])
    return designs


def _fit_failure(reason,count,**details):
    return dict(status="unavailable",reason=reason,training_count=int(count),**details)


def fit_on_training(design, truth_record):
    """Unweighted row-z LS; machine rank only, positive scale, fixed training set."""
    X = np.asarray(design,dtype=np.float64)
    if X.ndim != 2 or X.shape[0] != 4096 or X.shape[1] not in (1,2):
        raise ValueError("design must be 4096x1 or 4096x2")
    use = np.asarray(truth_record["support"]) & np.asarray(truth_record["lattice"]["train"])
    count = int(np.count_nonzero(use))
    if truth_record["status"] != "usable":
        return _fit_failure(truth_record["reason"],count)
    A,y = X[use],np.asarray(truth_record["row_z"])[use]
    if not np.all(np.isfinite(A)):
        return _fit_failure("invalid_training_model_depth",count)
    if not np.all(np.isfinite(y) & (y > 0.)):
        return _fit_failure("invalid_training_truth",count)
    details = {}
    try:
        with np.errstate(over="raise",invalid="raise",divide="raise"):
            norms = np.sqrt(np.sum(A*A,axis=0))
            if np.all(np.isfinite(norms)):
                details["column_norms"] = norms.tolist()
            if not np.all(np.isfinite(norms) & (norms > 0.)):
                return _fit_failure("invalid_column_norm",count,**details)
            normalized = A/norms
            U,s,Vh = np.linalg.svd(normalized,full_matrices=False)
            if np.all(np.isfinite(s)):
                details["singular_values"] = s.tolist()
            threshold = max(normalized.shape)*np.finfo(np.float64).eps*s[0]
            rank = int(np.count_nonzero(s > threshold))
            details["rank"] = rank
            if np.isfinite(threshold):
                details["rank_threshold"] = float(threshold)
            if rank != X.shape[1]:
                return _fit_failure("rank_deficient",count,**details)
            uy = np.einsum("ij,i->j",U,y,optimize=False)
            beta = np.einsum("ji,j->i",Vh,uy/s,optimize=False)/norms
            if not np.all(np.isfinite(beta)):
                return _fit_failure("nonfinite_coefficients",count,**details)
            details["coefficients"] = beta.tolist()
            if beta[0] <= 0.:
                return _fit_failure("nonpositive_scale",count,**details)
            prediction = np.einsum("ij,j->i",A,beta,optimize=False)
            if not np.all(np.isfinite(prediction) & (prediction > 0.)):
                return _fit_failure("nonpositive_or_nonfinite_training_prediction",count,**details)
            residual = prediction-y
            squared_error = float(np.sum(residual*residual))
    except (np.linalg.LinAlgError,FloatingPointError):
        return _fit_failure("numerical_fit_failure",count,**details)
    return dict(status="usable",reason=None,coefficients=beta.tolist(),training_count=count,
                rank=rank,rank_threshold=float(threshold),singular_values=s.tolist(),
                column_norms=norms.tolist(),training_squared_row_z_error=squared_error)


def evaluate_fixed_holdout(fit,design,truth_record):
    """Return every fixed holdout error, or one explicit unavailable-arm record.

    Relative errors are descriptive outputs, not a gate. No penalty, clipping,
    subset retry or replacement of failed points is invented here.
    """
    use = np.asarray(truth_record["support"]) & np.asarray(truth_record["lattice"]["eval"])
    indices = np.flatnonzero(use)
    base = dict(holdout_count=int(indices.size),holdout_indices=indices,
                original_lattice_count=4096,support_uses_model_values=False)
    if fit["status"] != "usable":
        return dict(base,status="unavailable",reason=fit["reason"],invalid_model_count=None)
    X = np.asarray(design,dtype=np.float64)
    if X.shape != (4096,len(fit["coefficients"])):
        raise ValueError("heldout design/coefficient shape mismatch")
    A,y = X[use],np.asarray(truth_record["row_z"])[use]
    valid = np.all(np.isfinite(A),axis=1)
    if not np.all(valid):
        return dict(base,status="unavailable",reason="invalid_holdout_model_depth",invalid_model_count=int(np.count_nonzero(~valid)))
    try:
        with np.errstate(over="raise",invalid="raise",divide="raise"):
            prediction = np.einsum("ij,j->i",A,np.asarray(fit["coefficients"]),optimize=False)
            valid_prediction = np.isfinite(prediction) & (prediction > 0.)
            if not np.all(valid_prediction):
                return dict(base,status="unavailable",reason="nonpositive_or_nonfinite_holdout_prediction",invalid_model_count=int(np.count_nonzero(~valid_prediction)))
            residual = prediction-y
            relative = np.abs(residual)/y
    except FloatingPointError:
        return dict(base,status="unavailable",reason="numerical_holdout_failure",invalid_model_count=None)
    return dict(base,status="usable",reason=None,invalid_model_count=0,
                predicted_row_z=prediction,truth_row_z=y,residual_row_z=residual,
                absolute_relative_error=relative)
