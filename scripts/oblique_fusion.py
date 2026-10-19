"""Prospective single-point fusion of four already-computed view outputs.

No inference, images, labels, random sampling, files, clipping, or filtering.
Joints are [y,x]; coordinates are [x,y] in original-canvas units on return.
The caller must bind these three source modules in its execution freeze.
Fixed-marginal transport is a standard solver, not a novelty/result claim.
"""

from collections.abc import Mapping

import numpy as np

import marginal_coupling as coupling
import oblique_view_geometry as geometry


VIEWS = ("original", "reference", "oblique", "axis_control")
METHODS = (
    "original_native", "reference_native", "oblique_native", "axis_control_native",
    "reference_independent", "coordinate_average", "entropy_select", "product_joint",
    "mixture_joint", "oblique_coupling", "axis_coupling",
)
MAX_ITERATIONS = 1000
MARGINAL_TOLERANCE = 1e-9
AXIS_JOINT_TOLERANCE = 1e-7
GLOBAL_MEAN_TOLERANCE = 5e-5
READOUT_RADIUS = 20


def _inputs(values, shape, name):
    if not isinstance(values, Mapping) or set(values) != set(VIEWS):
        raise ValueError(name + " must have exactly the four fixed view keys")
    copied = {}
    for view in VIEWS:
        value = np.asarray(values[view])
        if value.dtype.kind not in "fiu" or value.shape != shape:
            raise ValueError(name + " has invalid real array shape for " + view)
        value = np.array(value, dtype=np.float64, copy=True)
        if not np.isfinite(value).all():
            raise ValueError(name + " must be finite for " + view)
        copied[view] = value
    return copied


def _inverse_point(point, inverse):
    # geometry.map_points intentionally returns float32 for model queries.
    # This display/readout path retains float64, using its validated inverse.
    with np.errstate(over="raise", invalid="raise"):
        result = point @ inverse[:2, :2].T + inverse[:2, 2]
    if not np.isfinite(result).all():
        raise ValueError("inverse-warped coordinate is not finite")
    return result


def _solver_record(result):
    return {"iterations": int(result.iterations), "row_error": float(result.row_error),
            "column_error": float(result.column_error), "residual": float(result.residual),
            "converged": bool(result.converged)}


def _distribution_diagnostic(joint, independent):
    return {"entropy": float(-np.sum(joint * np.log(joint))),
            "l1_change_from_independent": float(np.abs(joint - independent).sum()),
            "kl_to_independent": float(np.sum(joint * (np.log(joint) - np.log(independent)))),
            "exact_mode_tie_count": int(np.count_nonzero(joint == joint.max())),
            "total_mass": float(joint.sum())}


def _mean_diagnostic(joint, independent, inverse_reference):
    # Centering tightens the L1 bound; the mass term retains validity even
    # when the two total masses differ by roundoff. The mean gate below is in
    # reference coordinates and is checked separately from solver convergence.
    centers = np.arange(256, dtype=np.float64) + 0.5
    dx = joint.sum(axis=0) - independent.sum(axis=0)
    dy = joint.sum(axis=1) - independent.sum(axis=1)
    delta = np.array([dx @ centers, dy @ centers])
    bound = np.array([np.abs(dx) @ np.abs(centers - 128) + 128 * abs(dx.sum()),
                      np.abs(dy) @ np.abs(centers - 128) + 128 * abs(dy.sum())])
    original_delta = inverse_reference[:2, :2] @ delta
    original_bound = np.abs(inverse_reference[:2, :2]) @ bound
    return {"reference_delta_xy": delta.tolist(), "reference_bound_xy": bound.tolist(),
            "reference_max_abs_difference": float(np.max(np.abs(delta))),
            "original_delta_xy": original_delta.tolist(), "original_bound_xy": original_bound.tolist(),
            "original_max_abs_difference": float(np.max(np.abs(original_delta))),
            "mean_check_passed": bool(np.max(np.abs(delta)) <= GLOBAL_MEAN_TOLERANCE)}


def fuse(logits_by_view, native_xy_by_view):
    """Fuse one point, with exact keys original/reference/oblique/axis_control.

    Each view supplies a finite 512-vector ordered [y256,x256] and a finite xy
    native coordinate. Native controls keep the supplied coordinate, even if
    it differs from the floored distribution's reconstructed point. All joints
    use reference-canvas bins, followed by the reference inverse affine.

    Returns status, methods, solvers, diagnostics, and failure_reasons. methods
    is all 11 float64 xy arrays on success, or None on any numerical failure.
    Invalid inputs raise ValueError. A capped solver never yields a fallback
    prediction. No mean/coordinate agreement tolerance silently merges ties.
    """
    logits = _inputs(logits_by_view, (512,), "logits_by_view")
    native = _inputs(native_xy_by_view, (2,), "native_xy_by_view")
    transforms = geometry.fixed_view_affines()
    inverses = {view: geometry.invert_affine(transforms[view]) for view in VIEWS}
    probabilities = {view: coupling.native_marginals(logits[view]) for view in VIEWS}
    px, py = probabilities["reference"]
    independent = coupling.independent_joint(px, py)
    methods = {view + "_native": _inverse_point(native[view], inverses[view]) for view in VIEWS}
    methods["reference_independent"] = _inverse_point(
        coupling.mode_local_mean(independent, radius=READOUT_RADIUS), inverses["reference"])
    methods["coordinate_average"] = .5 * methods["reference_native"] + .5 * methods["oblique_native"]
    entropies = {view: float(sum(-np.sum(p * np.log(p)) for p in probabilities[view]))
                 for view in ("reference", "oblique")}
    chosen = "reference" if entropies["reference"] <= entropies["oblique"] else "oblique"
    methods["entropy_select"] = methods[chosen + "_native"].copy()
    diagnostics = {"marginal_entropies": entropies, "entropy_selected_view": chosen,
                   "reference_native_parity_max_abs_difference_original": float(np.max(np.abs(
                       methods["reference_native"] - methods["reference_independent"]))),
                   "axis_joint_max_abs_difference": None,
                   "axis_coordinate_max_abs_difference_original": None,
                   "axis_joint_check_passed": None, "global_means": {},
                   "joint_distributions": {"reference_independent": _distribution_diagnostic(independent, independent)}}
    solvers, failures = {}, []
    for view, method in (("oblique", "oblique_coupling"), ("axis_control", "axis_coupling")):
        fields = geometry.map_marginal_probabilities(
            *probabilities[view], transforms["reference"], transforms[view],
            floor=coupling.PROBABILITY_FLOOR)
        # Probability interpolation precedes multiplication/logging. A constant
        # normalization of this likelihood field cancels under fixed marginals.
        cost = -np.log(fields[0]) - np.log(fields[1])
        if view == "oblique":
            auxiliary = coupling.normalize_joint(fields[0] * fields[1])
            for name, joint in (("product_joint", coupling.poe_joint(independent, auxiliary)),
                                ("mixture_joint", coupling.mixture_joint(independent, auxiliary))):
                methods[name] = _inverse_point(
                    coupling.mode_local_mean(joint, radius=READOUT_RADIUS), inverses["reference"])
                diagnostics["joint_distributions"][name] = _distribution_diagnostic(joint, independent)
        try:
            result = coupling.entropic_coupling(
                px, py, cost, max_iterations=MAX_ITERATIONS, tolerance=MARGINAL_TOLERANCE)
        except (ValueError, FloatingPointError) as exc:
            solvers[method] = {"converged": False, "error": str(exc)}
            failures.append(method + ": numerical failure")
            continue
        solvers[method] = _solver_record(result)
        if not result.converged:
            failures.append(method + ": maximum iterations reached")
            continue
        methods[method] = _inverse_point(
            coupling.mode_local_mean(result.joint, radius=READOUT_RADIUS), inverses["reference"])
        diagnostics["joint_distributions"][method] = _distribution_diagnostic(result.joint, independent)
        means = _mean_diagnostic(result.joint, independent, inverses["reference"])
        diagnostics["global_means"][method] = means
        if not means["mean_check_passed"]:
            failures.append(method + ": fixed-marginal global mean check failed")
        if view == "axis_control":
            difference = float(np.max(np.abs(result.joint - independent)))
            diagnostics["axis_joint_max_abs_difference"] = difference
            diagnostics["axis_coordinate_max_abs_difference_original"] = float(np.max(
                np.abs(methods[method] - methods["reference_independent"])))
            diagnostics["axis_joint_check_passed"] = difference <= AXIS_JOINT_TOLERANCE
            if difference > AXIS_JOINT_TOLERANCE:
                failures.append("axis_coupling: separable-cost joint check failed")
    return {"status": "failed" if failures else "completed",
            "methods": None if failures else {name: methods[name] for name in METHODS},
            "solvers": solvers, "diagnostics": diagnostics, "failure_reasons": failures}
