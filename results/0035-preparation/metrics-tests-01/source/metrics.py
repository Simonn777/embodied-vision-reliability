"""Pure categorical J/Jtr arithmetic; no I/O, model, selection, or efficacy gate.

Author convention: TRI-ML/VOST fe274574cb03c8a3ea83e121dd76e20b703781fd
evaluation/source/{dataset,evaluation,metrics,utils}.py and evaluation_method.py.
255 becomes background in its actual caller. ``void_ignore`` is a named
sensitivity only. Oracle outputs use truth and are never deployed controls.
"""

import math
from collections.abc import Mapping

import numpy as np

CONTROLS = ("B", "B+M1", "B+M2", "B+M1+M2")
ORACLES = ("O_B", "O")
POLICIES = ("author_background", "void_ignore")
_SLOTS = ((0,), (0, 1), (0, 2), (0, 1, 2))


def _ids(object_ids):
    values = tuple(object_ids)
    if (not values or any(isinstance(x, (bool, np.bool_)) or
                          not isinstance(x, (int, np.integer)) for x in values)):
        raise ValueError("object_ids must be nonempty contiguous integers 1..K")
    if values != tuple(range(1, len(values) + 1)) or len(values) > 254:
        raise ValueError("object_ids must be contiguous 1..K, excluding 255")
    return tuple(int(value) for value in values)


def _labels(value, name):
    result = np.asarray(value)
    if (result.ndim != 2 or 0 in result.shape or
            result.dtype.kind not in "iu" or
            np.any(result < 0) or np.any(result > 255)):
        raise ValueError(f"{name} must be a nonempty 2D integer label image")
    return result


def initial_object_ids(initial):
    """Discover all initial IDs; reject sparse IDs rather than silently pad."""
    image = _labels(initial, "initial")
    return _ids(tuple(int(x) for x in np.unique(image) if x not in (0, 255)))


def _truth(truth, ids, void_policy):
    if void_policy not in POLICIES:
        raise ValueError("unknown void policy")
    image = _labels(truth, "truth")
    if np.any((image != 255) & (image > len(ids))):
        raise ValueError("truth contains an object absent from initial IDs")
    valid = (image != 255) if void_policy == "void_ignore" else np.ones(image.shape, bool)
    clean = np.where(image == 255, 0, image)
    return clean, valid


def _bank(logits, object_ids):
    ids = _ids(object_ids)
    values = np.asarray(logits)
    if (values.ndim != 4 or values.shape[:2] != (len(ids), 3) or
            0 in values.shape or values.dtype.kind not in "fiu" or
            not np.isfinite(values).all()):
        raise ValueError("logits must be finite real [K,3,H,W], slots B/M1/M2")
    return values, ids


def compose_controls(logits, object_ids):
    """Four categorical controls. No mutation, reordering, or truth input."""
    values, ids = _bank(logits, object_ids)
    outputs = {}
    for name, slots in zip(CONTROLS, _SLOTS):
        scores = np.max(values[:, slots, :, :], axis=1)
        winner = np.argmax(scores, axis=0)  # first == lowest original ID
        maximum = np.max(scores, axis=0)
        outputs[name] = np.where(maximum > 0, winner + 1, 0).astype(np.int64)
    return outputs


def support_oracles(logits, truth, object_ids, *, void_policy="author_background"):
    """Privileged simultaneous per-object J maxima in support-relaxed families."""
    values, ids = _bank(logits, object_ids)
    clean, valid = _truth(truth, ids, void_policy)
    if clean.shape != values.shape[-2:]:
        raise ValueError("truth/logit spatial shape mismatch")
    outputs = {}
    for name, support in (("O_B", values[:, 0] > 0),
                          ("O", np.max(values, axis=1) > 0)):
        output = np.zeros(clean.shape, dtype=np.int64)
        for index, object_id in enumerate(ids):
            output[(clean == object_id) & support[index] & valid] = object_id
        outputs[name] = output
    return outputs


def score_frame(truth, predictions, object_ids, *, void_policy="author_background"):
    """JSON-safe {method: [J for each ID]} for one frame; retain endpoint rows."""
    ids = _ids(object_ids)
    clean, valid = _truth(truth, ids, void_policy)
    if (not isinstance(predictions, Mapping) or not predictions or
            any(not isinstance(name, str) or not name for name in predictions)):
        raise ValueError("predictions must be a nonempty named mapping")
    result = {}
    for name, prediction in predictions.items():
        pred = _labels(prediction, name)
        if pred.shape != clean.shape or np.any(pred > len(ids)):
            raise ValueError("prediction shape/identity mismatch")
        values = []
        for object_id in ids:
            gt, mask = clean == object_id, pred == object_id
            intersection = int(np.count_nonzero(gt & mask & valid))
            union = int(np.count_nonzero((gt | mask) & valid))
            values.append(intersection / union if union else 1.0)
        result[name] = values
    return result


def frame_intervals(frame_count):
    """Zero-based positions. Jtr is the final quarter AFTER endpoint exclusion."""
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 3:
        raise ValueError("at least three complete frames are required")
    primary = list(range(1, frame_count - 1))
    return {"J": primary, "Jtr": primary[(3 * len(primary)) // 4:]}


def _mean(values):
    return math.fsum(values) / len(values)


def _decomposition(methods):
    if not all(name in methods for name in ("B", "O_B", "O")):
        return None
    return {
        metric: {
            "total_O_minus_B": methods["O"][metric] - methods["B"][metric],
            "native_cleanup_and_ownership_O_B_minus_B":
                methods["O_B"][metric] - methods["B"][metric],
            "additional_support_O_minus_O_B":
                methods["O"][metric] - methods["O_B"][metric],
        }
        for metric in ("J", "Jtr")
    }


def summarize_sequence(sequence, frame_scores, object_ids, *,
                       void_policy="author_background"):
    """Accumulate tiny score rows, not video tensors; every frame must be present.

    Caller is responsible for chronological complete-frame admission and for
    obtaining object_ids with initial_object_ids before scoring later frames.
    """
    ids = _ids(object_ids)
    if not isinstance(sequence, str) or not sequence or void_policy not in POLICIES:
        raise ValueError("invalid sequence or policy")
    rows = list(frame_scores)
    intervals = frame_intervals(len(rows))
    if not isinstance(rows[0], Mapping) or not rows[0]:
        raise ValueError("missing frame methods")
    names = tuple(rows[0])
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != set(names):
            raise ValueError("missing/extra method in a frame")
        for name in names:
            values = np.asarray(row[name])
            if (values.shape != (len(ids),) or values.dtype.kind not in "fiu" or
                    not np.isfinite(values).all() or
                    np.any(values < 0) or np.any(values > 1)):
                raise ValueError("invalid per-object J; no successful-subset averaging")
    methods = {}
    for name in names:
        objects = []
        for index, object_id in enumerate(ids):
            raw = [float(row[name][index]) for row in rows]
            objects.append({
                "object_id": object_id, "frame_J": raw,
                **{metric: _mean([raw[i] for i in positions])
                   for metric, positions in intervals.items()},
            })
        methods[name] = {
            "per_object": objects,
            **{metric: _mean([obj[metric] for obj in objects]) for metric in intervals},
        }
    return {
        "schema": "evr0035.sequence-metrics.v1", "sequence": sequence,
        "void_policy": void_policy, "frame_count": len(rows),
        "object_ids": list(ids), "frame_positions": intervals,
        "methods": methods, "oracle_decomposition": _decomposition(methods),
    }


def score_sequence(sequence, truth, predictions, object_ids=None, *,
                   void_policy="author_background"):
    """Convenience stacked-array API; streaming score_frame is equally supported."""
    gt = np.asarray(truth)
    if gt.ndim != 3:
        raise ValueError("truth must be [N,H,W]")
    frame_intervals(len(gt))
    found = initial_object_ids(gt[0])
    ids = found if object_ids is None else _ids(object_ids)
    if ids != found:
        raise ValueError("requested IDs differ from complete initial-mask IDs")
    if not isinstance(predictions, Mapping) or not predictions:
        raise ValueError("missing predictions")
    pred = {name: np.asarray(value) for name, value in predictions.items()}
    if any(value.shape != gt.shape for value in pred.values()):
        raise ValueError("every method must contain every complete frame")
    rows = [score_frame(gt[i], {name: value[i] for name, value in pred.items()},
                        ids, void_policy=void_policy) for i in range(len(gt))]
    return summarize_sequence(sequence, rows, ids, void_policy=void_policy)


def aggregate_sequences(reports):
    """Author primary: equal sequence-object means, never pooled pixels/frames."""
    reports = list(reports)
    if not reports or len({r["sequence"] for r in reports}) != len(reports):
        raise ValueError("reports must contain distinct complete sequences")
    policy, names = reports[0]["void_policy"], tuple(reports[0]["methods"])
    if policy not in POLICIES or not names:
        raise ValueError("invalid report policy/methods")
    for report in reports:
        if (report["schema"] != "evr0035.sequence-metrics.v1" or
                report["void_policy"] != policy or set(report["methods"]) != set(names)):
            raise ValueError("inconsistent sequence reports")
        ids = _ids(report["object_ids"])
        for name in names:
            objects = report["methods"][name]["per_object"]
            if tuple(obj["object_id"] for obj in objects) != ids:
                raise ValueError("incomplete sequence-object population")
            for obj in objects:
                for metric in ("J", "Jtr"):
                    if not math.isfinite(obj[metric]) or not 0 <= obj[metric] <= 1:
                        raise ValueError("nonfinite/missing result")
    methods = {
        name: {
            metric: _mean([obj[metric] for r in reports
                           for obj in r["methods"][name]["per_object"]])
            for metric in ("J", "Jtr")
        } for name in names
    }
    return {
        "schema": "evr0035.aggregate-metrics.v1", "void_policy": policy,
        "aggregation": "equal_sequence_object_means", "sequence_count": len(reports),
        "object_count": sum(len(r["object_ids"]) for r in reports),
        "methods": methods, "oracle_decomposition": _decomposition(methods),
        "per_sequence": reports,
    }
