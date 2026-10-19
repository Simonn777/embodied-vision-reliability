"""Read-only reconstruction of native calibrated monodepth LM block costs.

The separate cohort pass reads only the frozen observation/depth allowlist after
the complete supervisor input/seal/parity barrier. It loads saved reference poses
only after all 160 cost reconstructions pass. No optimizer or native call occurs.
Cost accumulation and native Jacobian block counts are deliberately distinct.
Numerical tolerances were fixed before any real trace reconstruction; they are
cross-implementation comparison tolerances, not formal floating-point bounds.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np


ATOL = 1e-12
RTOL = 1e-9
KINDS = ("sampson", "forward", "backward")
ARMS = ("depth_scale", "depth_affine")
STAGES = ("proposal_lo", "terminal_lo", "final_refinement")
CASE_SECONDS = 90.
TOTAL_SECONDS = 1800.
OUTPUT_BYTES_LIMIT = 512*1024*1024


class TraceMismatch(ValueError):
    """The entire case is ineligible; do not retain a successful subset."""


def tolerance(a, b):
    return ATOL + RTOL * max(abs(float(a)), abs(float(b)))


def close(a, b):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b)) <= tolerance(a, b)


def _finite(value, shape, name):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise TraceMismatch(name + " has invalid shape or nonfinite entries")
    return a


def _scalar(value, name, nonnegative=False, positive=False):
    if isinstance(value, (bool, np.bool_)):
        raise TraceMismatch(name + " must be numeric")
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0) or (positive and value <= 0):
        raise TraceMismatch(name + " is invalid")
    return value


def _count(value, name):
    if type(value) is not int or value < 0:
        raise TraceMismatch(name + " must be a nonnegative integer")
    return value


def support_indices(value, n):
    if not isinstance(value, (list, tuple)) or any(type(v) is not int for v in value):
        raise TraceMismatch("Support must explicitly list integer original-point indices")
    if any(v < 0 or v >= n for v in value) or len(set(value)) != len(value):
        raise TraceMismatch("Support has duplicate/out-of-range indices")
    return tuple(value)


def prepare_inputs(pair):
    """Take only the already-verified 0017 read_pair payload, never an HDF5 handle."""
    x1 = np.asarray(pair["x1"], dtype=np.float64)
    if x1.ndim != 2 or x1.shape[1] != 2 or len(x1) == 0:
        raise TraceMismatch("Expected nonempty Nx2 observations")
    n = len(x1)
    output = {"n": n}
    cal = pair["calibration"]
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        for side in (1, 2):
            x = _finite(pair["x"+str(side)], (n, 2), "points")
            camera = cal["camera"+str(side)]
            if camera.get("model") != "PINHOLE":
                raise TraceMismatch("Only the frozen PINHOLE camera contract is supported")
            p = _finite(camera["params"], (4,), "camera params")
            if np.any(p[:2] <= 0):
                raise TraceMismatch("Nonpositive focal length")
            K = _finite(cal["K"+str(side)], (3, 3), "K")
            expected = np.array([[p[0], 0., p[2]], [0., p[1], p[3]], [0., 0., 1.]])
            # Same authored-camera crosscheck as 0017; arithmetic uses params.
            if not np.allclose(K, expected, rtol=1e-6, atol=1e-6):
                raise TraceMismatch("Camera parameters and authored K disagree")
            h = np.column_stack(((x[:, 0]-p[2])/p[0], (x[:, 1]-p[3])/p[1], np.ones(n)))
            # Match the native Point2D unprojection route: unit ray then hnormalized.
            norm = np.sqrt(np.einsum("ni,ni->n", h, h, optimize=False))
            unit = h / norm[:, None]
            h = unit / unit[:, 2, None]
            output["h"+str(side)] = _finite(h, (n, 3), "calibrated rays")
            output["d"+str(side)] = _finite(pair["depth"+str(side)], (n,), "depth")
            output["f"+str(side)] = float(p[0]/2.0 + p[1]/2.0)
    return output


def quaternion_matrix(q):
    """Eigen's w,x,y,z expression, without quaternion normalization/sign fixing."""
    w, x, y, z = _finite(q, (4,), "quaternion")
    tx, ty, tz = 2*x, 2*y, 2*z
    twx, twy, twz = tx*w, ty*w, tz*w
    txx, txy, txz = tx*x, ty*x, tz*x
    tyy, tyz, tzz = ty*y, tz*y, tz*z
    return _finite([[1-(tyy+tzz), txy-twz, txz+twy],
                    [txy+twz, 1-(txx+tzz), tyz-twx],
                    [txz-twy, tyz+twx, 1-(txx+tyy)]], (3, 3), "raw quaternion R")


def expected_effective(prepared, arm, stage):
    if arm not in ("depth_scale", "depth_affine"):
        raise TraceMismatch("Only the two frozen depth arms are admitted")
    if stage not in ("proposal_lo", "terminal_lo", "final_refinement"):
        raise TraceMismatch("Unknown refinement stage")
    factor = .5*(1.0/prepared["f1"] + 1.0/prepared["f2"])
    final = stage == "final_refinement"
    return {"loss_type": "CAUCHY" if final else "TRUNCATED",
            "loss_scale": factor if final else 2.0*factor,
            "max_iterations": 100 if final else 25,
            "scale_reproj": 4.0/256.0, "weight_sampson": 1.0,
            "refine_shift": arm == "depth_affine", "weights": "uniform"}


def validate_effective(actual, expected):
    for key, value in expected.items():
        if key not in actual:
            raise TraceMismatch("Missing effective setting: " + key)
        if type(value) is float:
            # Settings are simple deterministic scalings, not cost reductions.
            a = _scalar(actual[key], key)
            if abs(a-value) > 1e-12*max(abs(a), abs(value), 1e-300):
                raise TraceMismatch("Effective setting differs: " + key)
        elif type(actual[key]) is not type(value) or actual[key] != value:
            raise TraceMismatch("Effective setting differs: " + key)


@dataclass
class Blocks:
    indices: tuple
    active: np.ndarray
    losses: np.ndarray
    numerator: float
    count: int
    mean: float
    cost_formula_nonzero_weight_count: int

    def summary(self):
        # This is the weight test applied to independently reconstructed COST
        # residuals. The native Jacobian Sampson formula has different rounding
        # (C*(1/sqrt(den)) versus C/sqrt(den)); near a TRUNCATED threshold, do not
        # equate this counter to the actual logged Jacobian-construction count.
        return {"numerator": self.numerator, "count": self.count, "mean": self.mean,
                "cost_formula_nonzero_weight_count": self.cost_formula_nonzero_weight_count,
                "kind_counts": {k: int(self.active[:, j].sum()) for j, k in enumerate(KINDS)}}


def reconstruct_blocks(prepared, state, effective, indices):
    """Recompute one state's cost blocks; do not simulate a Jacobian or LM step."""
    indices = support_indices(indices, prepared["n"])
    n = len(indices)
    idx = np.asarray(indices, dtype=np.int64)
    h1, h2 = prepared["h1"][idx], prepared["h2"][idx]
    d1, d2 = prepared["d1"][idx], prepared["d2"][idx]
    R = quaternion_matrix(state["q"])
    if "R" in state:
        actual_R = _finite(state["R"], (3, 3), "native R")
        if not np.allclose(R, actual_R, rtol=1e-12, atol=1e-12):
            raise TraceMismatch("Native R and raw quaternion disagree")
        R = actual_R
    t = _finite(state["t"], (3,), "raw translation")
    scale = _scalar(state["scale"], "raw scale")
    shift1, shift2 = (_scalar(state[k], k) for k in ("shift1", "shift2"))
    sr2 = _scalar(effective["scale_reproj"], "scale_reproj", nonnegative=True)
    ws = _scalar(effective["weight_sampson"], "weight_sampson", nonnegative=True)
    tau = _scalar(effective["loss_scale"], "loss_scale", positive=True)
    if effective.get("weights") != "uniform":
        raise TraceMismatch("Unexpected residual weights")
    loss_type = effective["loss_type"]
    if loss_type not in ("TRUNCATED", "CAUCHY"):
        raise TraceMismatch("Unsupported native loss")
    active = np.zeros((n, 3), dtype=bool)
    squared = np.zeros((n, 3), dtype=np.float64)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        tau2 = tau*tau
        if not math.isfinite(tau2) or tau2 <= 0:
            raise TraceMismatch("Loss-scale square collapsed")
        if ws > 0:
            tx = np.array([[0., -t[2], t[1]], [t[2], 0., -t[0]], [-t[1], t[0], 0.]])
            E = np.einsum("ij,jk->ik", tx, R, optimize=False)
            Eh1 = np.einsum("ij,nj->ni", E, h1, optimize=False)
            Eth2 = np.einsum("ji,nj->ni", E, h2, optimize=False)
            C = np.einsum("ni,ni->n", h2, Eh1, optimize=False)
            den = np.einsum("ni,ni->n", Eh1[:, :2], Eh1[:, :2], optimize=False) + np.einsum("ni,ni->n", Eth2[:, :2], Eth2[:, :2], optimize=False)
            if np.any(den <= 0) or not np.isfinite(den).all():
                raise TraceMismatch("Native Sampson denominator is degenerate/nonfinite")
            residual = C/np.sqrt(den)
            squared[:, 0] = residual*residual
            active[:, 0] = True
        if sr2 > 0:
            Z1 = np.einsum("ij,nj->ni", R, (d1+shift1)[:, None]*h1, optimize=False) + t
            Z2 = np.einsum("ji,nj->ni", R, (scale*(d2+shift2))[:, None]*h2-t, optimize=False)
            # Nonfinite accepted/initial/exit arithmetic blocks the case. This
            # never replaces the native z>0 test with a source-depth sign test.
            if not np.isfinite(Z1).all() or not np.isfinite(Z2).all():
                raise TraceMismatch("Nonfinite transformed depth in a required state")
            for column, Z, target in ((1, Z1, h2), (2, Z2, h1)):
                mask = Z[:, 2] > 0.0
                active[:, column] = mask
                inv_z = 1.0/Z[mask, 2]
                residual = (Z[mask, :2]*inv_z[:, None]-target[mask, :2])*math.sqrt(sr2)
                squared[mask, column] = np.einsum("ni,ni->n", residual, residual, optimize=False)
        if not np.isfinite(squared).all():
            raise TraceMismatch("Nonfinite squared residual")
        if loss_type == "TRUNCATED":
            losses = np.minimum(squared, tau2)
            eligible = active & (squared < tau2)
        else:
            losses = tau2*np.log1p(squared*(1.0/tau2))
            eligible = active.copy()  # finite Cauchy robust weight is positive
        losses[:, 0] *= ws
        losses[~active] = 0.
        if not np.isfinite(losses).all():
            raise TraceMismatch("Nonfinite robust loss")
    # The native residual pass visits Sampson, forward, backward for each point.
    numerator, count = 0.0, 0
    for row in range(n):
        for col in range(3):
            if active[row, col]:
                numerator += float(losses[row, col])
                count += 1
    if not math.isfinite(numerator):
        raise TraceMismatch("Cost accumulation overflowed")
    return Blocks(indices, active, losses, numerator, count, numerator*(1.0/max(1, count)), int(eligible.sum()))


def reconcile(blocks, native):
    count = _count(native["count"], "native cost count")
    numerator = _scalar(native["numerator"], "native cost numerator", nonnegative=True)
    mean = _scalar(native["mean"], "native cost mean", nonnegative=True)
    if count != blocks.count or not close(numerator, blocks.numerator) or not close(mean, blocks.mean):
        raise TraceMismatch("Native/reconstructed block cost or count mismatch")
    if not close(mean, numerator*(1.0/max(1, count))):
        raise TraceMismatch("Native numerator/count/mean are inconsistent")
    return {"count_exact": True, "numerator_difference": blocks.numerator-numerator,
            "mean_difference": blocks.mean-mean,
            "numerator_tolerance": tolerance(numerator, blocks.numerator),
            "mean_tolerance": tolerance(mean, blocks.mean)}


def decompose_transition(old, new):
    if old.indices != new.indices:
        raise TraceMismatch("Cannot compare different fixed LO supports")
    common = old.active & new.active
    lost, gained = old.active & ~new.active, new.active & ~old.active
    D = math.fsum(float(v) for v in (new.losses-old.losses)[common])
    L = math.fsum(float(v) for v in old.losses[lost])
    G = math.fsum(float(v) for v in new.losses[gained])
    n0, n1 = max(1, old.count), max(1, new.count)
    denominator = old.numerator*(1.0/n1-1.0/n0)
    parts = {"common_change": D/n1, "lost": -L/n1, "gained": G/n1,
             "denominator_change": denominator}
    sum_delta = D-L+G
    mean_delta = math.fsum(parts.values())
    if not close(sum_delta, new.numerator-old.numerator) or not close(mean_delta, new.mean-old.mean):
        raise TraceMismatch("Transition decomposition failed to reconcile")
    propagated = tolerance(old.numerator, old.numerator)/n0 + tolerance(new.numerator, new.numerator)/n1
    support_contribution = math.fsum((parts["lost"], parts["gained"], parts["denominator_change"]))
    by_kind = {}
    for col, kind in enumerate(KINDS):
        by_kind[kind] = {"common": int(common[:, col].sum()), "lost": int(lost[:, col].sum()), "gained": int(gained[:, col].sum())}
    return {"old_count": old.count, "new_count": new.count,
            "old_denominator": n0, "new_denominator": n1,
            "common_count": int(common.sum()), "lost_count": int(lost.sum()), "gained_count": int(gained.sum()),
            "kind_counts": by_kind, "D": D, "L": L, "G": G,
            "numerator_delta": new.numerator-old.numerator,
            "native_mean_delta_reconstructed": new.mean-old.mean,
            "mean_components": parts, "mean_components_sum": mean_delta,
            "support_contribution": support_contribution,
            "propagated_comparison_tolerance": propagated,
            "resolved_decrease": bool(new.mean-old.mean < -10.0*propagated),
            "support_contribution_resolved_negative": bool(support_contribution < -10.0*propagated),
            "common_change_only_fixed_old_denominator": D/n0,
            "numerical_scope": "comparison tolerance; not a formal bound on per-block arithmetic"}


def _require(condition, message):
    if not condition:
        raise TraceMismatch(message)


def _same_state(a, b):
    for key, shape in (("q", (4,)), ("t", (3,)), ("R", (3, 3))):
        if key == "R" and key not in a and key not in b:
            continue
        av, bv = _finite(a[key], shape, key), _finite(b[key], shape, key)
        _require(np.all(np.abs(av-bv) <= 1e-12*np.maximum(1., np.maximum(np.abs(av), np.abs(bv)))), "State lineage changed: "+key)
    for key in ("scale", "shift1", "shift2"):
        av, bv = _scalar(a[key], key), _scalar(b[key], key)
        _require(abs(av-bv) <= 1e-12*max(1., abs(av), abs(bv)), "State lineage changed: "+key)


def diagnose_native_trace(prepared, trace, arm, final_support, deadline=None):
    """Audit every LO and accepted step; retain rejected decisions without replay."""
    begun = time.perf_counter()
    deadline = min(deadline if deadline is not None else math.inf, begun+CASE_SECONDS)
    _require(trace.get("schema") == "evr0018.native-trace.v1" and trace.get("complete") is True, "Incomplete/unknown native trace")
    events = trace.get("events")
    _require(isinstance(events, list) and 0 < len(events) <= 50000, "Invalid event coverage")
    _require(trace.get("capacity") == 50000 and trace.get("event_count") == len(events) and trace.get("dropped_events") == 0, "Trace capacity/count differs")
    _require(isinstance(trace.get("proposal_trace_hash_hex"), str) and re.fullmatch(r"[0-9a-f]{16}", trace["proposal_trace_hash_hex"]) is not None, "Missing diagnostic proposal hash")
    for key in ("sampled_draws", "scored_proposals"):
        _count(trace[key], key)
    support_indices(final_support, prepared["n"])
    los, proposals, incumbent_events, pipeline = {}, {}, [], {}
    current, live, last_state, last_cost, seen_exit = None, None, None, None, False
    for event_index, event in enumerate(events):
        _require(time.perf_counter() <= deadline, "Case/cohort reconstruction time limit exceeded")
        kind = event.get("kind")
        _require(type(event.get("finite")) is bool and type(event.get("lo_id")) is int, "Malformed event flags")
        if kind in ("proposal", "incumbent", "pipeline"):
            _require(event["lo_id"] == -1, "Outer event has an optimizer id")
        if kind == "lo_begin":
            _require(current is None, "Overlapping LO invocations")
            lo_id = _count(event["lo_id"], "lo_id")
            _require(lo_id not in los, "Duplicate LO id")
            stage = event["stage"]
            _require(stage in STAGES, "Invalid LO stage/entry")
            indices = support_indices(event["support"], prepared["n"])
            wanted = tuple(final_support) if stage == "final_refinement" else tuple(range(prepared["n"]))
            _require(indices == wanted, "Stage support differs from exact all-input/final-mask support")
            validate_effective(event["effective"], expected_effective(prepared, arm, stage))
            parent = (event["parent_kind"], event["parent_id"])
            _require(parent[0] in ("proposal", "lo", "none") and type(parent[1]) is int, "Invalid parent reference")
            if parent[0] == "proposal":
                _require(parent[1] in proposals, "Unknown parent proposal")
                _same_state(event["state"], proposals[parent[1]]["state"])
            elif parent[0] == "lo":
                _require(parent[1] in los and los[parent[1]].get("exit") is not None, "Unknown/unclosed parent LO")
                _same_state(event["state"], los[parent[1]]["exit"]["state"])
            current = {"lo_id": lo_id, "stage": stage, "parent_kind": parent[0], "parent_id": parent[1],
                "support_count": len(indices), "support_sha256": hashlib.sha256(json.dumps(indices,separators=(",",":")).encode()).hexdigest(),
                "effective": event["effective"], "begin_state": event["state"], "states": [], "transitions": [],
                "attempt_counts": [], "rejected_count": 0, "nonfinite_rejected_count": 0, "exit": None}
            los[lo_id] = current
            live, last_state, last_cost, seen_exit = None, None, None, False
        elif kind in ("initial", "attempt", "exit", "lo_end"):
            _require(current is not None and event.get("lo_id") == current["lo_id"], "Optimizer event outside its LO")
            _require(event.get("stage") == current["stage"], "Optimizer stage changed")
            if kind == "lo_end":
                _require(seen_exit and event.get("stage") == current["stage"], "Missing/malformed LO exit")
                _same_state(event["state"], last_state)
                current = live = last_state = last_cost = None
                continue
            _require(not seen_exit, "Optimizer event after exit")
            if kind == "attempt":
                _require(live is not None and type(event.get("accepted")) is bool, "Attempt before initial state or missing decision")
                iteration = _count(event["iteration"], "iteration")
                _require(iteration < current["effective"]["max_iterations"], "Attempt exceeds native iteration budget")
                if current["attempt_counts"]:
                    _require(iteration > current["attempt_counts"][-1]["iteration"], "Attempt order changed")
                counts = {"iteration": iteration, "accepted": event["accepted"], "finite": event.get("finite"),
                          "jacobian_count": _count(event["jacobian_count"], "jacobian_count"),
                          "solve_count": _count(event["solve_count"], "solve_count"),
                          "candidate_cost_count": _count(event["native_cost"]["count"], "candidate cost count")}
                _require(all(counts[k] <= 3*current["support_count"] for k in ("jacobian_count","solve_count","candidate_cost_count")), "Impossible native block count")
                current["attempt_counts"].append(counts)
                if not event["accepted"]:
                    current["rejected_count"] += 1
                    current["nonfinite_rejected_count"] += int(event.get("finite") is False)
                    continue
                _require(_scalar(event["native_cost"]["mean"],"accepted native mean",nonnegative=True) < last_cost["mean"], "Accepted native cost did not strictly decrease")
            elif kind == "initial":
                _require(live is None and not current["states"], "Duplicate initial state")
                _same_state(event["state"], current["begin_state"])
            else:
                _require(live is not None, "Exit before initial state")
                _same_state(event["state"], last_state)
                _require(_count(event["invalid_steps"], "invalid_steps") == current["rejected_count"], "Native rejected-step count differs")
                _require(_count(event["iterations"], "iterations") <= current["effective"]["max_iterations"], "Native exit iteration exceeds budget")
                _require(isinstance(event.get("reason"),str) and bool(event["reason"]), "Missing termination reason")
            blocks = reconstruct_blocks(prepared,event["state"],current["effective"],list(indices))
            check = reconcile(blocks,event["native_cost"])
            expected_counts = {"sampson_blocks":int(blocks.active[:,0].sum()),
                "forward_active":int(blocks.active[:,1].sum()),"backward_active":int(blocks.active[:,2].sum()),
                "forward_nonfinite_z":0,"backward_nonfinite_z":0}
            _require(event.get("blocks") == expected_counts,"Native per-kind block/visibility count mismatch")
            item = {"kind":kind,"event_index":event_index,"state":event["state"],
                    "native_cost":event["native_cost"],"reconstructed":blocks.summary(),"reconciliation":check,
                    "event_all_floats_finite":event["finite"],
                    "auxiliary_diagnostics":{k:event[k] for k in ("lambda_before","lambda_after","nu","grad_norm","step_norm") if k in event},
                    "auxiliary_nonfinite_with_valid_state_and_cost":not event["finite"]}
            if kind == "attempt":
                item["iteration"] = event["iteration"]
                transition = decompose_transition(live,blocks)
                transition.update(from_state_index=len(current["states"])-1,to_state_index=len(current["states"]),
                                  iteration=event["iteration"],native_count_roles=current["attempt_counts"][-1])
                current["transitions"].append(transition)
            if kind == "exit":
                current["exit"] = item
                current["termination"] = {k:event[k] for k in ("iterations","invalid_steps","reason","live_count","jacobian_count")}
                seen_exit = True
            else:
                current["states"].append(item)
            live,last_state,last_cost = blocks,event["state"],event["native_cost"]
        elif kind == "proposal":
            _require(current is None, "Proposal interleaved with active LO")
            proposal_id = _count(event["proposal_id"], "proposal_id")
            _require(proposal_id not in proposals, "Duplicate proposal id")
            _require(not proposals or proposal_id > max(proposals), "Proposal ids reordered")
            proposals[proposal_id] = event
        elif kind == "incumbent":
            _require(current is None, "Incumbent change interleaved with LO")
            source = (event["source_kind"], event["source_id"])
            source_state = _source_state(source, los, proposals)
            _same_state(event["state"], source_state)
            if incumbent_events:
                previous = incumbent_events[-1]
                _require((event["previous_source_kind"],event["previous_source_id"]) ==
                         (previous["source_kind"],previous["source_id"]), "Incumbent history changed")
            incumbent_events.append(event)
        elif kind == "pipeline":
            _require(current is None and event.get("phase") in ("begin","pre_final","returned"), "Invalid pipeline event")
            _require(event["phase"] not in pipeline, "Duplicate pipeline phase")
            pipeline[event["phase"]] = event
        else:
            raise TraceMismatch("Unknown native trace event kind: "+str(kind))
    _require(current is None and set(pipeline) == {"begin","pre_final","returned"}, "Incomplete optimizer/pipeline closure")
    _require(events[0].get("phase") == "begin" and events[-1].get("phase") == "returned", "Pipeline boundary reordered")
    returned = pipeline["returned"]
    enabled = pipeline["pre_final"]["final_refinement_enabled"]
    performed = returned["final_refinement_performed"]
    _require(type(enabled) is bool and type(performed) is bool and enabled == performed, "Final-stage enable/performed flags differ")
    _require(enabled == (len(final_support) > 3), "Final wrapper branch differs from native epipolar mask threshold")
    _require(sum(lo["stage"] == "final_refinement" for lo in los.values()) == int(performed), "Final-stage invocation coverage differs")
    pre_final = pipeline["pre_final"]
    _same_state(pre_final["state"], _source_state((pre_final["source_kind"],pre_final["source_id"]),los,proposals))
    _same_state(returned["state"], _source_state((returned["source_kind"],returned["source_id"]),los,proposals))
    if performed:
        final = next(lo for lo in los.values() if lo["stage"] == "final_refinement")
        _require((final["parent_kind"],final["parent_id"]) == (pre_final["source_kind"],pre_final["source_id"]), "Final LO ancestry changed")
        _require((returned["source_kind"],returned["source_id"]) == ("lo",final["lo_id"]), "Final returned LO differs")
    else:
        _require((returned["source_kind"],returned["source_id"]) == (pre_final["source_kind"],pre_final["source_id"]), "Skipped final stage changed ancestry")
    ancestry = []
    source = (returned["source_kind"], returned["source_id"])
    while source[0] == "lo":
        _require(source[1] in los and source[1] not in ancestry, "Missing/cyclic final ancestry")
        lo = los[source[1]]
        if not ancestry:
            _same_state(returned["state"],lo["exit"]["state"])
        ancestry.append(source[1])
        source = (lo["parent_kind"],lo["parent_id"])
    _require(source[0] in ("proposal","none"), "Unknown final ancestry root")
    if source[0] == "proposal":
        _require(source[1] in proposals, "Final ancestry proposal missing")
    adopted = {e["source_id"] for e in incumbent_events if e["source_kind"] == "lo"}
    for lo in los.values():
        lo["on_final_ancestry"] = lo["lo_id"] in ancestry
        lo["adopted_as_ransac_incumbent"] = lo["lo_id"] in adopted
    return {"schema":"evr0018.case-reconstruction.v1","status":"completed","arm":arm,
            "event_count":len(events),"lo_count":len(los),"accepted_transition_count":sum(len(v["transitions"]) for v in los.values()),
            "final_ancestry_lo_ids":ancestry,"final_ancestry_root":{"kind":source[0],"id":source[1]},
            "returned_state":returned["state"],"final_refinement_performed":performed,
            "proposal_trace_hash_hex":trace["proposal_trace_hash_hex"],"sampled_draws":trace["sampled_draws"],"scored_proposals":trace["scored_proposals"],
            "los":list(los.values()),"reconstruction_seconds":time.perf_counter()-begun,
            "tolerance":{"absolute":ATOL,"relative":RTOL,"interpretation_multiplier":10.}}


def _source_state(source, los, proposals):
    kind, index = source
    _require(type(index) is int, "Noninteger lineage source")
    if kind == "proposal":
        _require(index in proposals, "Unknown source proposal")
        return proposals[index]["state"]
    if kind == "lo":
        _require(index in los and los[index].get("exit") is not None, "Unknown source LO")
        return los[index]["exit"]["state"]
    raise TraceMismatch("Returned/adopted source cannot be an unobserved model")


def material_harm(before, after):
    return bool(after-before >= 1.0 or (before < 5.0 and after >= 5.0))


def annotate_case(case, reference, metrics):
    """Label a fully reconstructed case; callers enforce the cohort barrier first."""
    gt = reference["ground_truth"]
    def error(state):
        R = state.get("R", quaternion_matrix(state["q"]))
        e, _ = metrics.pose_errors(R, state["t"], gt["R_original"], gt["t_original"])
        folded = min(e["translation_degrees"], 180.0-e["translation_degrees"])
        return {**e, "translation_signfolded_degrees": folded,
                "max_signfolded_degrees": max(e["rotation_degrees"], folded)}
    los = []
    for lo in case["los"]:
        state_errors = [error(s["state"]) for s in lo["states"]]
        exit_error = error(lo["exit"]["state"])
        before, after = state_errors[0]["max_degrees"], exit_error["max_degrees"]
        transitions = []
        for tr in lo["transitions"]:
            e0, e1 = (state_errors[tr[key]] for key in ("from_state_index", "to_state_index"))
            lost, gained = tr["lost_count"] > 0, tr["gained_count"] > 0
            support_kind = "mixed" if lost and gained else "loss_only" if lost else "gain_only" if gained else "unchanged"
            harms = material_harm(e0["max_degrees"], e1["max_degrees"])
            qualifies = lost and harms and tr["support_contribution_resolved_negative"]
            transitions.append({**tr, "before_error": e0, "after_error": e1,
                "max_error_delta_degrees": e1["max_degrees"]-e0["max_degrees"],
                "material_pose_harm": harms, "support_change_kind": support_kind,
                "same_step_support_harm_condition": bool(qualifies),
                "common_change_alone_resolved_decrease": bool(tr["common_change_only_fixed_old_denominator"] < -10.*tr["propagated_comparison_tolerance"])})
        net_harm = material_harm(before,after)
        weaker = any(t["lost_count"] > 0 and t["support_contribution_resolved_negative"] for t in transitions)
        strict = any(t["same_step_support_harm_condition"] for t in transitions)
        los.append({"lo_id":lo["lo_id"], "stage":lo["stage"],
            "parent_kind":lo["parent_kind"], "parent_id":lo["parent_id"],
            "on_final_ancestry":lo["on_final_ancestry"],
            "adopted_as_ransac_incumbent":lo["adopted_as_ransac_incumbent"],
            "state_errors":state_errors, "exit_error":exit_error,
            "net_max_error_delta_degrees":after-before, "net_material_pose_harm":net_harm,
            "strict_condition":bool(lo["on_final_ancestry"] and net_harm and strict),
            "weaker_cooccurrence_only":bool(lo["on_final_ancestry"] and net_harm and weaker and not strict),
            "transitions":transitions, "rejected_count":lo["rejected_count"],
            "nonfinite_rejected_count":lo["nonfinite_rejected_count"], "termination":lo["termination"]})
    returned = error(case["returned_state"])
    original = reference["outcomes"][case["arm"]]
    _require(original["status"] == "completed", "Reference arm was not completed")
    angular_differences = {}
    for key, value in returned.items():
        baseline = _scalar(original["error"][key],"saved angular error",nonnegative=True)
        _require(baseline <= 180. and math.isfinite(value),"Invalid saved/recomputed angular error")
        angular_differences[key] = value-baseline
    # Mandatory raw-state parity is established before label access. The acos
    # metric is ill-conditioned near +/-1: the COST comparison tolerance is not
    # an angular-parity tolerance. Preserve these differences descriptively.
    return {"schema":"evr0018.case-labels.v1", "status":"completed",
        **{key:case[key] for key in ("pair_index","scene","key","arm")},
        "returned_error":returned,"frozen0017_returned_error":original["error"],
        "returned_error_minus_frozen0017_degrees":angular_differences,
        "strict_condition":any(lo["strict_condition"] for lo in los),
        "weaker_cooccurrence_only":any(lo["weaker_cooccurrence_only"] for lo in los),
        "los":los, "interpretation":"Trajectory association, not an intervention or a deployable GT-selected output."}


def _distribution(values):
    if not values:
        return {"count":0,"minimum":None,"median":None,"maximum":None,"mean":None}
    a = np.asarray(values,dtype=float)
    _require(np.isfinite(a).all(), "Nonfinite descriptive distribution")
    return {"count":len(values),"minimum":float(a.min()),"median":float(np.median(a)),
            "maximum":float(a.max()),"mean":float(a.mean())}


def summarize_cases(cases, metrics):
    _require(len(cases) == 160 and [(c["pair_index"],c["arm"]) for c in cases] ==
             [(i,a) for i in range(80) for a in ARMS], "Incomplete labelled cohort")
    scenes = sorted({c["scene"] for c in cases})
    _require(len(scenes) == 5, "Expected all five development scenes")
    def one(subset):
        arms = {}
        for arm in ARMS:
            selected = [c for c in subset if c["arm"] == arm]
            stages = {}
            for stage in STAGES:
                los = [lo for c in selected for lo in c["los"] if lo["stage"] == stage]
                groups = {}
                for group in ("all","final_ancestry","adopted","not_final_ancestry"):
                    ll = [lo for lo in los if group == "all" or
                          (group == "final_ancestry" and lo["on_final_ancestry"]) or
                          (group == "adopted" and lo["adopted_as_ransac_incumbent"]) or
                          (group == "not_final_ancestry" and not lo["on_final_ancestry"])]
                    tt = [t for lo in ll for t in lo["transitions"]]
                    groups[group] = {"lo_count":len(ll),"accepted_transition_count":len(tt),
                        "rejected_count":sum(lo["rejected_count"] for lo in ll),
                        "nonfinite_rejected_count":sum(lo["nonfinite_rejected_count"] for lo in ll),
                        "net_material_harm_lo_count":sum(lo["net_material_pose_harm"] for lo in ll),
                        "strict_condition_lo_count":sum(lo["strict_condition"] for lo in ll),
                        "support_change_counts":{k:sum(t["support_change_kind"] == k for t in tt) for k in ("loss_only","gain_only","mixed","unchanged")},
                        "same_step_support_harm_count":sum(t["same_step_support_harm_condition"] for t in tt),
                        "net_error_delta_degrees":_distribution([lo["net_max_error_delta_degrees"] for lo in ll]),
                        "step_error_delta_degrees":_distribution([t["max_error_delta_degrees"] for t in tt]),
                        "support_contribution":_distribution([t["support_contribution"] for t in tt])}
                stages[stage] = groups
            errors = [c["returned_error"]["max_degrees"] for c in selected]
            folded = [c["returned_error"]["max_signfolded_degrees"] for c in selected]
            frozen = [c["frozen0017_returned_error"]["max_degrees"] for c in selected]
            arms[arm] = {"pair_denominator":len(selected),
                "strict_condition_pair_count":sum(c["strict_condition"] for c in selected),
                "weaker_cooccurrence_only_pair_count":sum(c["weaker_cooccurrence_only"] and not c["strict_condition"] for c in selected),
                "returned_auc_fractions_5_10_20":metrics.dms_auc(errors),
                "frozen0017_returned_auc_fractions_5_10_20":metrics.dms_auc(frozen),
                "returned_max_error_difference_degrees":_distribution([c["returned_error_minus_frozen0017_degrees"]["max_degrees"] for c in selected]),
                "returned_signfolded_auc_fractions_5_10_20":metrics.dms_auc(folded), "stages":stages}
        return arms
    qualifying = sorted({c["scene"] for c in cases if c["arm"] == "depth_scale" and c["strict_condition"]})
    return {"all_pairs":one(cases), "scenes":{s:one([c for c in cases if c["scene"] == s]) for s in scenes},
        "primary_gate":{"passed":len(qualifying) >= 2,"qualifying_scenes":qualifying,
            "required_scene_count":2,"arm":"depth_scale",
            "criterion":"A materially harmful LO on final ancestry contains an accepted support-losing step with material pose harm in that same step and a support contribution below minus ten propagated comparison tolerances.",
            "scope":"Necessary admission condition for a separately frozen intervention; not causal proof, novel-method success, or paper readiness."}}


def _module(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _dependencies():
    here = Path(__file__).resolve().parent
    c = _module("evr0018_diagnosis_common",here/"common.py")
    with c.aliases({"common":c}):
        worker = _module("evr0018_diagnosis_worker",here/"predict_pair.py")
    with c.aliases({"common":c,"predict_pair":worker}):
        runner = _module("evr0018_diagnosis_runner",here/"run_diagnostic.py")
    return c,runner


def _load_observations(c,path,key):
    import h5py
    with h5py.File(path,"r") as handle:
        return c.read_pair(handle,key)


def _mask(record,n):
    _require(record["mask_length"] == n, "Final support length differs")
    raw = bytes.fromhex(record["native_inliers_packbits_big_hex"])
    _require(len(raw) == (n+7)//8, "Final packed support length differs")
    bits = np.unpackbits(np.frombuffer(raw,dtype=np.uint8),bitorder="big")
    _require(not bits[n:].any(), "Nonzero unused mask padding")
    mask = bits[:n].astype(bool)
    _require(hashlib.sha256(mask.tobytes()).hexdigest() == record["mask_sha256"] and
             int(mask.sum()) == record["kept_count"], "Final support binding differs")
    return [int(i) for i in np.flatnonzero(mask)]


def _state_matches_output(state,geometry):
    R = state.get("R",quaternion_matrix(state["q"]))
    for key,value in (("R",R),("t",state["t"]),("scale",state["scale"]),
                      ("shift1",state["shift1"]),("shift2",state["shift2"])):
        a,b = np.asarray(value,dtype=float),np.asarray(geometry[key],dtype=float)
        _require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all() and
            np.all(np.abs(a-b) <= 1e-12*np.maximum(1.,np.maximum(np.abs(a),np.abs(b)))),
            "Trace return and frozen final geometry differ: "+key)


def _load_reference(c,config,manifest,manifest_hash,baseline_hash):
    binding = config["baseline_evaluation"]
    relative = Path(binding["path"])
    _require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe reference path")
    path = c.ROOT/relative
    _require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(c.ROOT.resolve()), "Unsafe reference file")
    _require(c.sha256(path) == binding["sha256"], "Saved reference evaluation hash differs")
    evaluation = json.loads(path.read_text())
    _require(evaluation.get("schema") == "evr0017.evaluation.v1" and evaluation.get("status") == "completed" and
        evaluation.get("pair_count") == 80 and evaluation.get("manifest_sha256") == manifest_hash and
        evaluation.get("seal_sha256") == baseline_hash, "Saved reference evaluation binding differs")
    rows = evaluation.get("rows",[])
    _require(len(rows) == 80, "Incomplete saved reference coverage")
    for row,expected in zip(rows,manifest["selected_pairs"]):
        _require(all(row[k] == expected[k] for k in ("pair_index","scene","key")), "Saved reference identity differs")
    return rows,path


def run(data_dir,manifest_path,baseline_seal,seal_path,output_dir,*,_modules=None,_pair_loader=None):
    """All 160 parity -> all 160 reconstruction -> saved-label evaluation, once."""
    begun = time.perf_counter()
    out = Path(output_dir)
    out.mkdir(parents=True,exist_ok=False)
    c,runner = _modules if _modules is not None else _dependencies()
    loader = _pair_loader if _pair_loader is not None else _load_observations
    byte_count = 0
    def write(name,value):
        nonlocal byte_count
        encoded = (json.dumps(c.json_safe(value),sort_keys=True,allow_nan=False)+"\n").encode()
        _require(byte_count+len(encoded) <= OUTPUT_BYTES_LIMIT,"Serialized diagnostic output exceeds cohort cap")
        with (out/name).open("xb") as stream:
            stream.write(encoded)
        byte_count += len(encoded)
        return c.sha256(out/name)
    def deadline():
        _require(time.perf_counter()-begun <= TOTAL_SECONDS,"Whole diagnostic time limit exceeded")
    phase,completed = "parity",[]
    write("started.json",{"schema":"evr0018.diagnosis-operation.v1","status":"started",
        "case_seconds_limit":CASE_SECONDS,"total_seconds_limit":TOTAL_SECONDS,"output_bytes_limit":OUTPUT_BYTES_LIMIT})
    try:
        # This verifies all source, binary, baseline, process, log and trace hashes
        # and every final raw-gauge/mask/stat parity before observation access.
        manifest,seal,records = runner.verify_seal(manifest_path,baseline_seal,seal_path)
        _require(len(records) == 160 and [(r["pair_index"],r["arm"]) for r in records] ==
                 [(i,a) for i in range(80) for a in ARMS],"Incomplete/reordered replay coverage")
        sources,config = c.source_hashes(),c.library_config()
        mh,bh,sh = c.sha256(manifest_path),c.sha256(baseline_seal),c.sha256(seal_path)
        binding = {"manifest_sha256":mh,"baseline_seal_sha256":bh,"trace_seal_sha256":sh,
                   "source_sha256":sources,"baseline_evaluation":config["baseline_evaluation"]}
        assets = {}
        for row in manifest["selected_pairs"]:
            if row["filename"] in assets:
                _require(assets[row["filename"]] == {"bytes":row["dataset_bytes"],"sha256":row["dataset_sha256"]},"Inconsistent asset declaration")
                continue
            path = c.bound_file(data_dir,row["filename"])
            _require(path.stat().st_size == row["dataset_bytes"] and c.sha256(path) == row["dataset_sha256"],"Observation asset hash/size differs")
            assets[row["filename"]] = {"bytes":row["dataset_bytes"],"sha256":row["dataset_sha256"]}
        _require(len(assets) == 5,"Expected five observation assets")
        deadline()
        phase = "reconstruction"
        reconstruction_started = time.perf_counter()
        for i,row in enumerate(manifest["selected_pairs"]):
            deadline()
            input_started = time.perf_counter()
            pair = loader(c,c.bound_file(data_dir,row["filename"]),row["key"])
            prepared = prepare_inputs(pair)
            input_seconds = time.perf_counter()-input_started
            for arm in ARMS:
                case_started = time.perf_counter()
                record = records[2*i+ARMS.index(arm)]
                _require(all(record[k] == row[k] for k in ("pair_index","scene","key")),"Observation identity differs")
                _require(c.input_hashes(pair) == record["input_arrays"] and
                         c.json_safe(pair["calibration"]) == record["calibration"] and
                         c.json_safe(pair["input_diagnostics"]) == record["input_diagnostics"],"Observation content changed")
                runner.check_trace(Path(seal_path).parent,record)
                trace_path = c.bound_file(Path(seal_path).parent,record["trace_path"])
                envelope = json.loads(trace_path.read_text())
                case = diagnose_native_trace(prepared,envelope["native"],arm,_mask(record,prepared["n"]),
                    deadline=min(begun+TOTAL_SECONDS,case_started+CASE_SECONDS-input_seconds))
                _state_matches_output(case["returned_state"],record["geometry"])
                case.update({k:row[k] for k in ("pair_index","scene","key")})
                case.update(bindings={**binding,"trace_sha256":record["trace_sha256"],
                    "input_arrays":record["input_arrays"],"dataset_sha256":row["dataset_sha256"]},
                    timings={"shared_pair_input_seconds":input_seconds,"case_reconstruction_seconds":time.perf_counter()-case_started,
                             "original_replay":record["timings"]})
                name = f"pair-{i:02d}-{arm}.reconstruction.json"
                digest = write(name,case)
                _require(time.perf_counter()-case_started+input_seconds <= CASE_SECONDS,"Case diagnostic time limit exceeded")
                completed.append({"pair_index":i,"scene":row["scene"],"key":row["key"],"arm":arm,"path":name,"sha256":digest})
            del pair,prepared
        _require(len(completed) == 160,"Incomplete cost reconstruction")
        reconstruction_seconds = time.perf_counter()-reconstruction_started
        # Recheck the complete barrier and opaque inputs before any label access.
        runner.verify_seal(manifest_path,baseline_seal,seal_path)
        _require(c.source_hashes() == sources and c.sha256(manifest_path) == mh and c.sha256(baseline_seal) == bh and c.sha256(seal_path) == sh,"Bindings changed during reconstruction")
        for filename,expected in assets.items():
            path = c.bound_file(data_dir,filename)
            _require(path.stat().st_size == expected["bytes"] and c.sha256(path) == expected["sha256"],"Observation cache changed during reconstruction")
        deadline()
        reconseal = {"schema":"evr0018.reconstruction-seal.v1","status":"completed","case_count":160,
            "all_160_parity_and_cost_checks_passed":True, "rows":completed,"bindings":binding,
            "verified_assets":assets,"reconstruction_seconds":reconstruction_seconds}
        reconseal_hash = write("reconstruction-seal.json",reconseal)
        phase = "saved_labels"
        label_started = time.perf_counter()
        references,reference_path = _load_reference(c,config,manifest,mh,bh)
        metrics = _module("evr0018_frozen_metrics",c.ROOT/"experiments/0013-pose-headroom/evaluate_sealed.py")
        labelled, label_rows = [],[]
        for row in completed:
            deadline()
            path = c.bound_file(out,row["path"])
            _require(c.sha256(path) == row["sha256"],"Reconstruction artifact changed")
            case = annotate_case(json.loads(path.read_text()),references[row["pair_index"]],metrics)
            name = f"pair-{row['pair_index']:02d}-{row['arm']}.labels.json"
            digest = write(name,{**case,"reconstruction_sha256":row["sha256"],"baseline_evaluation":config["baseline_evaluation"]})
            label_rows.append({**{k:row[k] for k in ("pair_index","scene","key","arm")},"path":name,"sha256":digest})
            labelled.append(case)
        summary = summarize_cases(labelled,metrics)
        _require(c.sha256(reference_path) == config["baseline_evaluation"]["sha256"] and c.source_hashes() == sources,
                 "Saved reference/source changed during labels")
        _require(c.sha256(manifest_path) == mh and c.sha256(baseline_seal) == bh and c.sha256(seal_path) == sh,"Input seal changed during labels")
        deadline()
        output = {"schema":"evr0018.diagnosis-evaluation.v1","status":"completed","pair_count":80,"case_count":160,
            "bindings":binding,"reconstruction_seal_sha256":reconseal_hash,"rows":label_rows,
            "all_160_parity_and_cost_checks_passed":True,"all_160_returned_raw_state_parity_passed":True,
            "all_160_returned_angular_differences_recorded":True,
            "summary":summary,"timings":{"reconstruction_seconds":reconstruction_seconds,
                "saved_label_seconds":time.perf_counter()-label_started,"total_diagnostic_seconds":time.perf_counter()-begun,
                "replay_outer_process_seconds":math.fsum(r["timings"]["charged_total_seconds"] for r in records),
                "replay_instrumented_native_seconds":math.fsum(r["timings"]["call_seconds"] for r in records)},
            "scope":"All five scenes are permanently development and labels were previously exposed. Diagnostic association only; no optimizer or native call in this pass. LO counts are descriptive, not independent samples.",
            "failure_policy":"Any missing case, parity/cost mismatch or exceeded cap blocks the entire diagnostic."}
        write("evaluation.json",output)
        write("operation.json",{"schema":"evr0018.diagnosis-operation.v1","status":"completed",
            "case_count":160,"bindings":binding,"total_seconds":time.perf_counter()-begun,
            "output_bytes_before_receipt":byte_count,"evaluation_sha256":c.sha256(out/"evaluation.json")})
        return 0
    except BaseException as exc:
        message = type(exc).__name__+": "+str(exc)
        for private in (str(Path(data_dir).resolve()),str(out.resolve()),str(c.ROOT.resolve())):
            message = message.replace(private,"<local-path>")
        # Even an output-cap failure retains a small explicit ineligible receipt.
        c.write_new(out/"operation.json",{"schema":"evr0018.diagnosis-operation.v1","status":"ineligible",
            "failed_phase":phase,"completed_reconstructions":len(completed),"error":message,
            "total_seconds":time.perf_counter()-begun,"output_bytes_before_receipt":byte_count})
        if not isinstance(exc,Exception):
            raise
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-dir","manifest","baseline-seal","seal","output-dir"):
        parser.add_argument("--"+name,required=True,type=Path)
    args = parser.parse_args()
    return run(args.data_dir,args.manifest,args.baseline_seal,args.seal,args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
