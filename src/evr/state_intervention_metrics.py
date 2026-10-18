"""Fixed 0004 development diagnostics from completed predictions; no inference.

The single video provides no population CI. Shadow returns are retrospective
paired measurements, never an online action rule or deployed oracle.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


FIELDS = {"branch", "frame", "point_index", "shadow_start", "shadow_offset",
          "x", "y", "occlusion_logit", "expected_distance_logit"}
VALUES = ("x", "y", "occlusion_logit", "expected_distance_logit")


def _int(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} below minimum")
    return value


def _float(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _mean(values):
    if not values.size:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(np.mean(values))
    if not math.isfinite(result):
        raise ValueError("computed aggregate is nonfinite")
    return result


def _ordered_ints(values, name, upper):
    result = tuple(_int(value, name) for value in values)
    if list(result) != sorted(set(result)) or any(value >= upper for value in result):
        raise ValueError(f"{name} must be sorted, unique and within frame range")
    return result


def summarize(rows, gt_points_xy, gt_occluded, point_ids, thresholds, *,
              shadow_starts, shadow_horizon, control_frames, control_atol=1e-4):
    """Validate complete branch grids, then compute protocol 0004 quantities.

    GT is the full original point-index N,T,2 normalized-xy array. Point IDs are
    original indices in the declared query order. Numeric CSV fields must be
    explicitly parsed before calling. Frame zero is required but never scored.
    Every per-branch/frame row group must preserve the declared point order;
    groups themselves may be interleaved. Undefined rates/means are None.
    """
    gt = np.asarray(gt_points_xy)
    occluded = np.asarray(gt_occluded)
    if (gt.ndim != 3 or gt.shape[2] != 2 or gt.dtype.kind != "f" or
            gt.shape[0] == 0 or gt.shape[1] < 2 or not np.isfinite(gt).all()):
        raise ValueError("GT must be finite floating N,T,2 with T >= 2")
    if occluded.dtype != np.bool_ or occluded.shape != gt.shape[:2]:
        raise ValueError("GT occluded must be boolean N,T")
    total_frames = gt.shape[1]
    ids = tuple(_int(value, "point_id") for value in point_ids)
    if not ids or len(set(ids)) != len(ids) or max(ids) >= gt.shape[0]:
        raise ValueError("point_ids must be unique nonempty original indices")
    if occluded[list(ids), 0].any():
        raise ValueError("declared queries must be visible at frame zero")
    cuts = tuple(_float(value, "threshold") for value in thresholds)
    if not cuts or list(cuts) != sorted(set(cuts)) or min(cuts) <= 0 or 4.0 not in cuts:
        raise ValueError("thresholds must be sorted unique positive values including 4")
    starts = _ordered_ints(shadow_starts, "shadow_starts", total_frames)
    controls = _ordered_ints(control_frames, "control_frames", total_frames)
    horizon = _int(shadow_horizon, "shadow_horizon", 1)
    if any(start == 0 or start + horizon > total_frames for start in starts):
        raise ValueError("shadow windows must be post-query and entirely present")
    if any(b < a + horizon for a, b in zip(starts, starts[1:])):
        raise ValueError("shadow windows must not overlap")
    atol = _float(control_atol, "control_atol")
    if atol < 0:
        raise ValueError("control_atol must be nonnegative")

    expected = {(branch, frame, -1, point) for branch in ("keep", "zero")
                for frame in range(total_frames) for point in ids}
    expected.update(("clone_control", frame, -1, point)
                    for frame in controls for point in ids)
    expected.update(("shadow", frame, start, point) for start in starts
                    for frame in range(start, start + horizon) for point in ids)
    observed, order = {}, defaultdict(list)
    for row in rows:
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError("prediction row schema mismatch")
        branch = row["branch"]
        if not isinstance(branch, str) or branch not in ("keep", "zero", "clone_control", "shadow"):
            raise ValueError("unknown branch")
        frame = _int(row["frame"], "frame")
        point = _int(row["point_index"], "point_index")
        start = _int(row["shadow_start"], "shadow_start", -1)
        offset = _int(row["shadow_offset"], "shadow_offset", -1)
        if (branch == "shadow" and (start < 0 or offset != frame - start or offset < 0)):
            raise ValueError("shadow offset inconsistent with frame/start")
        if branch != "shadow" and (start != -1 or offset != -1):
            raise ValueError("non-shadow rows require -1 start/offset")
        key = (branch, frame, start, point)
        if key not in expected or key in observed:
            raise ValueError("unexpected or duplicate prediction grid row")
        observed[key] = np.array([_float(row[name], name) for name in VALUES], dtype=np.float64)
        order[(branch, frame, start)].append(point)
    if set(observed) != expected:
        raise ValueError("missing prediction grid rows")
    if any(tuple(points) != ids for points in order.values()):
        raise ValueError("point order differs from declared point_ids")

    def values(branch, frame, point, start=-1):
        return observed[(branch, frame, start, point)]

    maxima = {"clone_control": 0.0, "frame_zero": 0.0, "shadow_start": 0.0}
    comparisons = []
    comparisons.extend(("frame_zero", values("keep", 0, point), values("zero", 0, point))
                       for point in ids)
    comparisons.extend(("clone_control", values("keep", frame, point),
                        values("clone_control", frame, point))
                       for frame in controls for point in ids)
    comparisons.extend(("shadow_start", values("zero", start, point),
                        values("shadow", start, point, start))
                       for start in starts for point in ids)
    for name, left, right in comparisons:
        with np.errstate(over="ignore"):
            difference = float(np.max(np.abs(left - right)))
        maxima[name] = max(maxima[name], difference)
        if not math.isfinite(difference) or difference > atol:
            raise ValueError(f"{name} integrity comparison failed")

    selected_gt = gt[list(ids)].astype(np.float64) * 256.0
    visible = ~occluded[list(ids)]

    def arrays(branch, frames, start=-1):
        raw = np.array([[values(branch, frame, point, start) for frame in frames]
                        for point in ids])
        with np.errstate(over="ignore", invalid="ignore"):
            delta = raw[..., :2] - selected_gt[:, frames]
            error = np.hypot(delta[..., 0], delta[..., 1])
        if not np.isfinite(error).all():
            raise ValueError("computed coordinate error is nonfinite")
        # exp(-logaddexp(0,z)) = sigmoid(-z), stable even for extreme logits.
        confidence = np.exp(-np.logaddexp(0.0, raw[..., 2])) * np.exp(
            -np.logaddexp(0.0, raw[..., 3]))
        return error, confidence > 0.5

    frames = list(range(1, total_frames))
    gt_visible = visible[:, frames]
    keep_error, keep_visible = arrays("keep", frames)
    zero_error, zero_visible = arrays("zero", frames)

    def branch_summary(error, predicted_visible):
        count = int(gt_visible.sum())
        result = {
            "post_query_rows": int(gt_visible.size), "gt_visible_count": count,
            "predicted_visible_count": int(predicted_visible.sum()),
            "predicted_visible_fraction": float(predicted_visible.mean()),
            "visibility_accuracy": float((predicted_visible == gt_visible).mean()),
            "mean_visible_epe_px": _mean(error[gt_visible]),
            "thresholds": [],
        }
        for threshold in cuts:
            coordinate_correct = gt_visible & (error <= threshold)
            tp = int((coordinate_correct & predicted_visible).sum())
            fp = int((predicted_visible & ~coordinate_correct).sum())
            correct = int(coordinate_correct.sum())
            result["thresholds"].append({
                "threshold_px": threshold, "coordinate_correct_count": correct,
                "visible_coordinate_error_count": count - correct,
                "visible_coordinate_error_rate": _ratio(count - correct, count),
                "coordinate_correct_fraction_among_gt_visible": _ratio(correct, count),
                "visible_correct_fraction_among_gt_visible": _ratio(tp, count),
                "true_positive_count": tp, "false_positive_count": fp,
                "jaccard": _ratio(tp, count + fp),
            })
        jaccards = [entry["jaccard"] for entry in result["thresholds"]]
        result["mean_threshold_jaccard"] = (float(np.mean(jaccards))
                                              if all(v is not None for v in jaccards) else None)
        return result

    keep_wrong = gt_visible & (keep_error > 4)
    zero_wrong = gt_visible & (zero_error > 4)
    recovered = keep_wrong & ~zero_wrong
    harmed = ~keep_wrong & zero_wrong
    per_query = []
    for index, point in enumerate(ids):
        mask = gt_visible[index]
        n = int(mask.sum())
        per_query.append({
            "point_index": point, "gt_visible_count": n,
            "mean_keep_epe_px": _mean(keep_error[index, mask]),
            "mean_zero_epe_px": _mean(zero_error[index, mask]),
            "mean_epe_zero_minus_keep_px": _mean(
                zero_error[index, mask] - keep_error[index, mask]),
            "coordinate_recoveries_at_4px": int(recovered[index].sum()),
            "coordinate_harms_at_4px": int(harmed[index].sum()),
        })

    def loss(error, predicted_visible, is_visible):
        return np.where(is_visible, ~(predicted_visible & (error <= 4)), predicted_visible)

    episodes, beneficial_ids, total_return = [], set(), 0
    for start in starts:
        episode_frames = list(range(start, start + horizon))
        shadow_error, shadow_visible = arrays("shadow", episode_frames, start)
        # Post-query arrays start at absolute frame 1.
        keep_slice = slice(start - 1, start - 1 + horizon)
        k_loss = loss(keep_error[:, keep_slice], keep_visible[:, keep_slice], visible[:, episode_frames])
        s_loss = loss(shadow_error, shadow_visible, visible[:, episode_frames])
        delta_loss = k_loss.astype(np.int64).sum(axis=1) - s_loss.astype(np.int64).sum(axis=1)
        for index, point in enumerate(ids):
            gain = int(delta_loss[index])
            total_return += gain
            if gain > 0:
                beneficial_ids.add(point)
            episodes.append({"shadow_start": start, "point_index": point,
                             "frames": horizon, "keep_loss_count": int(k_loss[index].sum()),
                             "shadow_loss_count": int(s_loss[index].sum()),
                             "frame_indices": episode_frames,
                             "keep_losses": k_loss[index].astype(int).tolist(),
                             "shadow_losses": s_loss[index].astype(int).tolist(),
                             "frame_loss_deltas": (k_loss[index].astype(int) -
                                                   s_loss[index].astype(int)).tolist(),
                             "loss_delta_count": gain, "return_keep_minus_shadow": gain / horizon,
                             "outcome": "beneficial" if gain > 0 else "harmful" if gain < 0 else "neutral"})
    aggregate_return = _ratio(total_return, len(episodes) * horizon)
    baseline_errors = int(keep_wrong.sum())
    recoveries = int(recovered.sum())
    criteria = {"baseline_errors_at_least_20": baseline_errors >= 20,
                "coordinate_recoveries_at_least_5": recoveries >= 5,
                "aggregate_shadow_return_positive": bool(episodes) and total_return > 0,
                "at_least_2_queries_with_beneficial_shadow": len(beneficial_ids) >= 2}
    status = ("undersensitive" if baseline_errors < 20 else
              "continue_independent_development" if all(criteria.values()) else "stop_this_candidate")
    return {
        "scope": "single-video development mechanism diagnostic; no CI or deployed selector",
        "query_frame_excluded": 0, "point_ids": list(ids), "frames": total_frames,
        "validated_rows": len(observed), "control_max_absolute_differences": maxima,
        "branch_summaries": {"keep": branch_summary(keep_error, keep_visible),
                             "zero": branch_summary(zero_error, zero_visible)},
        "per_query": per_query,
        "paired_coordinate_changes": {"threshold_px": 4, "baseline_visible_errors": baseline_errors,
                                      "recoveries": recoveries, "harms": int(harmed.sum()),
                                      "net_recoveries_minus_harms": recoveries - int(harmed.sum())},
        "shadow_episodes": episodes,
        "shadow_aggregate": {"episode_count": len(episodes), "loss_delta_count": total_return,
                             "mean_return_keep_minus_shadow": aggregate_return,
                             "beneficial_episode_count": sum(e["outcome"] == "beneficial" for e in episodes),
                             "harmful_episode_count": sum(e["outcome"] == "harmful" for e in episodes),
                             "neutral_episode_count": sum(e["outcome"] == "neutral" for e in episodes),
                             "queries_with_beneficial_shadow": sorted(beneficial_ids)},
        "continuation": {"status": status, "criteria": criteria,
                         "permits_independent_development_only": status == "continue_independent_development"},
    }
