"""Pure arithmetic for the prospective 0029 fixed development diagnostic.

No decoder, tracker, matching, file access, or ground-truth-dependent selection.
The aggregation entry point consumes already sealed selections and evaluated J.
"""
import hashlib
import math

import numpy as np

COHORT = (
    ("dogs-scale", 83), ("upside-down", 65), ("dog-agility", 25),
    ("cat-girl", 89), ("planes-water", 38), ("dog-gooses", 86),
)
BANK = ("B", "D1", "D3")
CONTROLS = (
    "baseline", "max_predicted_iou", "max_unsigned_change",
    "always_D1", "always_D3", "random_deletion", "random_on_signed",
    "native_bank", "prompt_direct", "prompt_signed",
    "reset_direct", "reset_signed",
)
METHODS = ("signed_deletion", *CONTROLS)
DESCRIPTIVE_CONTROLS = ("ungated_D1", "ungated_D3")
MARGIN = 1e-6
RANDOM_SALT = "evr0029:random-deletion:2026-10-20"


def require(value, message):
    if not value:
        raise ValueError(message)


def _finite(value, name):
    require(type(value) in (float, int) and math.isfinite(value), name + " must be finite")
    return float(value)


def _shape(shape):
    require(len(shape) == 2 and all(type(x) is int and x > 0 for x in shape), "shape must be positive (height,width) integers")
    return tuple(shape)


def _points(value, name):
    a = np.asarray(value, dtype=np.float64)
    require(a.ndim == 2 and a.shape[1] == 2, name + " must be N by 2")
    return a


def _labels(value, void_id):
    a = np.asarray(value)
    require(a.ndim == 2 and a.size > 0 and a.dtype.kind in "biu" and np.all(a >= 0), "annotation must be a nonnegative integer image")
    require(type(void_id) is int and void_id > 0, "invalid void ID")
    return a


def choose_target(initial_labels, void_id=255):
    """Smallest positive non-void ID; None means an invalid cohort initial mask."""
    a = _labels(initial_labels, void_id)
    ids = np.unique(a[(a > 0) & (a != void_id)])
    return int(ids[0]) if ids.size else None


def query_schedule(frame_count):
    require(type(frame_count) is int and frame_count >= 12, "eight distinct query times require at least twelve frames")
    result = [4 + (2 * j * (frame_count - 5) + 7) // 14 for j in range(8)]
    require(len(set(result)) == 8, "query schedule is not unique")
    return result


QUERY_IDENTITIES = tuple((name, frame) for name, n in COHORT for frame in query_schedule(n))


def _morphology(target):
    # Fixed 3x3 square / Chebyshev radius 1; outside-image target is false.
    h, w = target.shape
    padded = np.pad(target, 1, constant_values=False)
    neighbors = [padded[dy:dy+h, dx:dx+w] for dy in range(3) for dx in range(3)]
    return np.logical_and.reduce(neighbors), np.logical_or.reduce(neighbors)


def select_witnesses(initial_labels, keypoints_initial, keypoints_current,
                     match_ids, match_scores, current_shape, target_id=None,
                     void_id=255):
    """Native accepted matches only; nonfinite/OOB coordinates are discarded.

    Scores are ordered without any additional confidence threshold. Match index
    means its original row in match_ids, not a keypoint ID or reordered position.
    """
    labels = _labels(initial_labels, void_id)
    chosen = choose_target(labels, void_id)
    require(chosen is not None, "no initial target; record a cohort failure")
    if target_id is not None:
        require(type(target_id) is int and target_id == chosen, "target differs from smallest positive initial ID")
    h, w = _shape(current_shape)
    p0 = _points(keypoints_initial, "initial keypoints")
    pt = _points(keypoints_current, "current keypoints")
    ids = np.asarray(match_ids)
    scores = np.asarray(match_scores, dtype=np.float64)
    require(ids.ndim == 2 and ids.shape[1] == 2 and ids.dtype.kind in "iu", "match IDs must be integer N by 2")
    require(scores.shape == (len(ids),) and np.all(np.isfinite(scores)), "match confidence shape/nonfinite")
    require(np.all(ids >= 0) and np.all(ids[:, 0] < len(p0)) and np.all(ids[:, 1] < len(pt)), "match keypoint ID out of range")
    positive, dilated = _morphology(labels == chosen)
    negative = (~dilated) & (labels != void_id)
    ih, iw = labels.shape
    counts = dict(matches=len(ids), invalid_coordinates=0, boundary_or_void=0,
                  labeled=0, opposite_collision_points=0, same_label_discarded=0,
                  after_collisions=0)
    pixels = {}
    for index, (i0, it) in enumerate(ids):
        source, target = p0[i0], pt[it]
        if not (np.all(np.isfinite(source)) and np.all(np.isfinite(target))
                and 0 <= source[0] < iw and 0 <= source[1] < ih
                and 0 <= target[0] < w and 0 <= target[1] < h):
            counts["invalid_coordinates"] += 1
            continue
        sx, sy = np.floor(source + .5).astype(np.int64)
        tx, ty = np.floor(target + .5).astype(np.int64)
        if not (0 <= sx < iw and 0 <= sy < ih and 0 <= tx < w and 0 <= ty < h):
            counts["invalid_coordinates"] += 1
            continue
        label = 1 if positive[sy, sx] else (0 if negative[sy, sx] else None)
        if label is None:
            counts["boundary_or_void"] += 1
            continue
        cx = min(7, max(0, int(math.floor(8 * target[0] / w))))
        cy = min(7, max(0, int(math.floor(8 * target[1] / h))))
        row = {"match_index": index, "initial_keypoint_index": int(i0),
               "current_keypoint_index": int(it), "label": label,
               "source_xy": source.tolist(), "target_xy": target.tolist(),
               "rounded_current_xy": [int(tx), int(ty)],
               "cell": [cy, cx], "confidence": float(scores[index])}
        pixels.setdefault((int(tx), int(ty)), []).append(row)
        counts["labeled"] += 1
    survivors = []
    for rows in pixels.values():
        if len({x["label"] for x in rows}) > 1:
            counts["opposite_collision_points"] += len(rows)
        else:
            survivors.append(min(rows, key=lambda x: (-x["confidence"], x["match_index"])))
            counts["same_label_discarded"] += len(rows) - 1
    counts["after_collisions"] = len(survivors)
    selected = {}
    for label, name in ((1, "positive"), (0, "negative")):
        cells = [[] for _ in range(64)]
        for row in survivors:
            if row["label"] == label:
                cells[8 * row["cell"][0] + row["cell"][1]].append(row)
        for cell in cells:
            cell.sort(key=lambda x: (-x["confidence"], x["match_index"]))
        result, depth = [], 0
        while len(result) < 32:
            added = False
            for cell in cells:
                if depth < len(cell):
                    result.append(cell[depth]); added = True
                    if len(result) == 32:
                        break
            if not added:
                break
            depth += 1
        selected[name] = result
        counts[name + "_before_cap"] = sum(len(cell) for cell in cells)
        counts[name + "_retained"] = len(result)
    return {"target_id": chosen, "initial_shape": [ih, iw], "current_shape": [h, w],
            "counts": counts, **selected,
            "actionable": len(selected["positive"]) >= 4 and len(selected["negative"]) >= 4}


def sample_logits(logits, xy, original_shape):
    """Bilinear sampled logits, align_corners=False, border replication.

    Input xy must already be valid original-image continuous coordinates.
    Outside points are rejected, never moved into the image by this helper.
    """
    h, w = _shape(original_shape)
    values = np.asarray(logits, dtype=np.float64)
    points = _points(xy, "sample coordinates")
    require(values.ndim == 2 and values.size > 0 and np.all(np.isfinite(values)), "invalid logits")
    require(np.all(np.isfinite(points)) and np.all((points >= 0) & (points < [w, h])), "sample coordinates outside image/nonfinite")
    lh, lw = values.shape
    mapped = (points + .5) * np.array([lw / w, lh / h]) - .5
    mapped = np.clip(mapped, [0., 0.], [lw - 1., lh - 1.])
    left = np.floor(mapped).astype(np.int64)
    right = np.minimum(left + 1, [lw - 1, lh - 1])
    dx, dy = (mapped - left).T
    x0, y0 = left.T; x1, y1 = right.T
    with np.errstate(over="raise", invalid="raise"):
        result = ((1-dx)*(1-dy)*values[y0, x0] + dx*(1-dy)*values[y0, x1]
                  + (1-dx)*dy*values[y1, x0] + dx*dy*values[y1, x1])
    require(np.all(np.isfinite(result)), "nonfinite interpolated logits")
    return result


def balanced_logloss(logits, witnesses):
    require(witnesses["actionable"] is True, "balanced evidence unavailable: fewer than four per class")
    losses = []
    for name, sign in (("positive", -1), ("negative", 1)):
        rows = witnesses[name]
        require(len(rows) >= 4, "inconsistent actionable witness record")
        sampled = sample_logits(logits, [row["target_xy"] for row in rows], witnesses["current_shape"])
        # Divide before summing: avoids overflow for finite extreme logits.
        terms = np.logaddexp(0., sign * sampled)
        losses.append(float(np.sum(terms / len(terms), dtype=np.float64)))
    loss = .5 * losses[0] + .5 * losses[1]
    require(math.isfinite(loss), "nonfinite balanced loss")
    return loss


def random_branch(sequence, object_id, frame):
    require(isinstance(sequence, str) and sequence and "\0" not in sequence, "invalid sequence identity")
    require(type(object_id) is int and object_id > 0 and type(frame) is int and frame >= 0, "invalid random identity")
    text = "\0".join((RANDOM_SALT, sequence, str(object_id), str(frame)))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {"branch": ("D1", "D3")[int(digest[-1], 16) & 1], "sha256": digest}


def select_controls(*, sequence, object_id, frame, actionable, losses=None,
                    predicted_ious=None, unsigned_changes=None,
                    native_losses=None, prompt_loss=None, reset_loss=None):
    """Only image/matcher/model observations enter. There is no J or GT input."""
    require(type(actionable) is bool, "actionable must be Boolean")
    random = random_branch(sequence, object_id, frame)
    choices = {name: "B" for name in METHODS}
    if not actionable:
        return {"actionable": False, "choices": choices, "signed_action": False,
                "random": random, "bank_losses": None}
    def bank_values(value, name):
        require(isinstance(value, dict) and set(value) == set(BANK), name + " must contain exactly B,D1,D3")
        return {key: _finite(value[key], name) for key in BANK}
    losses = bank_values(losses, "losses")
    ious = bank_values(predicted_ious, "predicted IoUs")
    changes = bank_values(unsigned_changes, "unsigned changes")
    require(all(x >= 0 for x in losses.values()), "negative loss")
    require(all(0 <= x <= 1 for x in changes.values()) and changes["B"] == 0, "invalid unsigned change")
    require(isinstance(native_losses, (list, tuple)) and len(native_losses) >= 1, "native alternatives must be supplied")
    native = [_finite(x, "native loss") for x in native_losses]
    prompt = _finite(prompt_loss, "prompt loss"); reset = _finite(reset_loss, "reset loss")
    require(min([*native, prompt, reset]) >= 0, "negative alternative loss")
    def signed(names, values):
        best = min(range(len(names)), key=lambda i: (values[i], i))
        return names[best] if values[0] - values[best] > MARGIN else "B"
    chosen = signed(BANK, [losses[x] for x in BANK])
    choices.update(signed_deletion=chosen,
                   max_predicted_iou=max(BANK, key=lambda x: ious[x]),
                   max_unsigned_change=max(BANK, key=lambda x: changes[x]),
                   always_D1="D1", always_D3="D3",
                   random_deletion=random["branch"],
                   random_on_signed=random["branch"] if chosen != "B" else "B",
                   native_bank=signed(["B", *["M" + str(i) for i in range(len(native))]], [losses["B"], *native]),
                   prompt_direct="P", prompt_signed=signed(["B", "P"], [losses["B"], prompt]),
                   reset_direct="R", reset_signed=signed(["B", "R"], [losses["B"], reset]))
    return {"actionable": True, "choices": choices, "signed_action": chosen != "B",
            "random": random, "bank_losses": losses}


def _binary(value):
    a = np.asarray(value)
    require(a.ndim == 2 and a.size > 0 and a.dtype.kind in "bif" and np.all(np.isfinite(a)), "invalid mask/logits")
    return a if a.dtype.kind == "b" else a > 0


def foreground_j(prediction, annotation, target_id, void_id=255):
    predicted = _binary(prediction)
    labels = _labels(annotation, void_id)
    require(labels.shape == predicted.shape and type(target_id) is int and target_id > 0 and target_id != void_id, "mask/annotation identity mismatch")
    valid = labels != void_id
    a, b = predicted & valid, (labels == target_id) & valid
    union = int(np.count_nonzero(a | b))
    return float(np.count_nonzero(a & b) / union) if union else 1.


def unsigned_change(candidate, baseline):
    a, b = _binary(candidate), _binary(baseline)
    require(a.shape == b.shape, "candidate/baseline shape mismatch")
    union = int(np.count_nonzero(a | b))
    return 1. - float(np.count_nonzero(a & b) / union) if union else 0.


def evidence_coverage(rows):
    """Requires all 48 predetermined identities, including ordinary no-witnesses."""
    expected = set(QUERY_IDENTITIES)
    require(all(isinstance(row["sequence"], str) and type(row["frame"]) is int for row in rows), "invalid query identity types")
    seen = [(row["sequence"], row["frame"]) for row in rows]
    require(len(seen) == 48 and len(set(seen)) == 48 and set(seen) == expected, "fixed 48-query cohort missing/duplicate/changed")
    require(all(type(row["actionable"]) is bool for row in rows), "invalid actionable field")
    counts = {name: sum(row["actionable"] for row in rows if row["sequence"] == name) for name, _ in COHORT}
    videos = sum(value > 0 for value in counts.values())
    return {"queries": 48, "videos": 6, "actionable_queries": sum(counts.values()),
            "actionable_queries_by_video": counts, "videos_with_evidence": videos,
            "pass": videos >= 4}


def summarize(rows):
    """Post-seal descriptive evaluation; never selects a deployable branch.

    Each row: sequence/frame/status/actionable/selection/candidate_j. A completed
    row's selection is select_controls output; candidate_j contains B,D1,D3 and
    every selected M*/P/R. Failed rows stay present; any failure makes the whole
    result ineligible and prevents successful-subset metric aggregation.
    """
    coverage = evidence_coverage(rows)
    require(all(row.get("status") in ("completed", "failed") for row in rows), "unknown query status")
    output = {"denominator": 48, "video_count": 6, "queries_per_video": 8,
              "coverage": coverage, "eligible": all(x["status"] == "completed" for x in rows),
              "rows": [], "pooled": None, "videos": None,
              "gate": {"pass": False, "coverage": coverage["pass"],
                       "candidate_information": False, "pooled_selection": False,
                       "four_positive_videos": False, "no_video_below_minus_002": False}}
    order = {identity: i for i, identity in enumerate(QUERY_IDENTITIES)}
    for row in sorted(rows, key=lambda r: order[(r["sequence"], r["frame"])]):
        result = {"sequence": row["sequence"], "frame": row["frame"],
                  "status": row["status"], "actionable": row["actionable"], "j": None}
        if row["status"] == "completed":
            selection = row["selection"]; choices = selection["choices"]
            require(selection["actionable"] is row["actionable"] and set(choices) == set(METHODS), "selection identity/method mismatch")
            require(choices["baseline"] == "B" and all(choices[name] in BANK for name in METHODS[:8]), "invalid fixed-bank selector")
            require(row["actionable"] or all(x == "B" for x in choices.values()), "non-actionable selector must be baseline")
            candidate = {key: _finite(value, "candidate J") for key, value in row["candidate_j"].items()}
            require(all(0 <= x <= 1 for x in candidate.values()) and set(BANK) <= set(candidate), "invalid/missing bank J")
            require(all(value in candidate for value in choices.values()), "selected candidate J missing")
            result["j"] = {key: candidate[choices[key]] for key in METHODS}
            result["j"].update(ungated_D1=candidate["D1"], ungated_D3=candidate["D3"],
                               oracle_admissible=max(candidate[x] for x in BANK) if row["actionable"] else candidate["B"],
                               oracle_unrestricted=max(candidate[x] for x in BANK))
            result["selected_branches"] = dict(choices)
        else:
            result["reason"] = row.get("reason", "technical query failure")
        output["rows"].append(result)
    if not output["eligible"]:
        return output
    keys = tuple(output["rows"][0]["j"])
    videos = {}
    for name, _ in COHORT:
        group = [r for r in output["rows"] if r["sequence"] == name]
        means = {key: math.fsum(r["j"][key] for r in group) / 8 for key in keys}
        best = max(CONTROLS, key=lambda key: means[key])
        videos[name] = {"means": means, "best_control": best,
                        "signed_margin": means["signed_deletion"] - means[best]}
    pooled = {key: math.fsum(videos[name]["means"][key] for name, _ in COHORT) / 6 for key in keys}
    best = max(CONTROLS, key=lambda key: pooled[key])
    margins = [videos[name]["signed_margin"] for name, _ in COHORT]
    candidate_gain = pooled["oracle_admissible"] - pooled["baseline"]
    signed_gain = pooled["signed_deletion"] - pooled[best]
    output.update(pooled=pooled, videos=videos, best_pooled_control=best,
                  candidate_gain=candidate_gain, signed_gain=signed_gain,
                  positive_videos=sum(x > 0 for x in margins),
                  worst_video_margin=min(margins),
                  signed_actions=sum(r["selected_branches"]["signed_deletion"] != "B" for r in output["rows"]))
    gate = output["gate"]
    gate.update(candidate_information=candidate_gain >= .02,
                pooled_selection=signed_gain >= .02,
                four_positive_videos=sum(x > 0 for x in margins) >= 4,
                no_video_below_minus_002=min(margins) >= -.02)
    gate["pass"] = all(gate[key] for key in gate if key != "pass")
    return output
