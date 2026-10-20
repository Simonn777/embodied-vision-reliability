"""Pure saved-record metrics for the fixed 0028 development opportunity test.

No scientific files, models, or ground-truth sources are opened here. The
caller must establish the prediction seal and supply the already admitted
truth mapping. A successful development gate is never manuscript admission.
"""
from bisect import bisect_left
import importlib.util
import math
from numbers import Real
from pathlib import Path


# Load only the sibling's stdlib constants; avoid another experiment's generic
# ``common`` module being present in sys.modules.
_spec = importlib.util.spec_from_file_location(
    "_evr0028_metrics_common", Path(__file__).with_name("common.py"))
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)
ARMS = _common.ARMS
SCENES = _common.SCENES
QUERIES = tuple("B" + str(i) for i in range(1, 9))
THRESHOLDS = (5, 10, 20)
STATUSES = ("completed", "fallback", "unavailable")
ROTATION_TOLERANCE = 1e-5  # Same proper-rotation validation as frozen 0023.


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value, label):
    _require(isinstance(value, Real) and not isinstance(value, bool),
             label + " must be a real number")
    result = float(value)
    _require(math.isfinite(result), label + " must be finite")
    return result


def _vector(value, label):
    _require(isinstance(value, (list, tuple)) and len(value) == 3,
             label + " must be a three-vector")
    return [_finite(x, label) for x in value]


def _rotation(value):
    _require(isinstance(value, (list, tuple)) and len(value) == 3,
             "R must be three by three")
    r = [_vector(row, "R") for row in value]
    # Reject gross magnitudes before forming products, rather than accepting
    # overflow or projecting a submitted matrix to SO(3).
    _require(max(abs(v) for row in r for v in row) <= 1 + ROTATION_TOLERANCE,
             "R entry outside proper-rotation range")
    gram = [[math.fsum(r[i][k] * r[j][k] for k in range(3))
             for j in range(3)] for i in range(3)]
    determinant = (
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]))
    _require(max(abs(gram[i][j] - float(i == j))
                 for i in range(3) for j in range(3)) <= ROTATION_TOLERANCE
             and abs(determinant - 1) <= ROTATION_TOLERANCE,
             "R is not a proper rotation")
    return r


def _unit_translation(value):
    t = _vector(value, "t")
    scale = max(abs(v) for v in t)
    _require(scale > 0, "translation direction is undefined")
    scaled = [v / scale for v in t]
    norm = math.hypot(*scaled)
    return [v / norm for v in scaled]


def checked_pose(value):
    """Validate a pose without fixing its rotation or translation sign."""
    _require(isinstance(value, dict), "pose must be a mapping")
    return _rotation(value.get("R")), _unit_translation(value.get("t"))


def _angle(cosine):
    return math.degrees(math.acos(max(-1., min(1., cosine))))


def _checked_error(pose, truth):
    r, t = pose
    rg, tg = truth
    rotation = _angle((math.fsum(r[i][j] * rg[i][j]
                                for i in range(3) for j in range(3)) - 1) / 2)
    directed = _angle(math.fsum(x * y for x, y in zip(t, tg)))
    return dict(rotation_degrees=rotation, translation_degrees=directed,
                max_degrees=max(rotation, directed))


def pose_error(prediction, truth):
    """Directed translation angle; antipodal translation has 180 degree error."""
    return _checked_error(checked_pose(prediction), checked_pose(truth))


def dms_auc_percent(errors):
    """DMS left-search trapezoidal convention, expressed in percentage points.

    Exact-threshold observations do not enter the recall before that threshold.
    This is not the area of an empirical step-CDF.
    """
    _require(isinstance(errors, (list, tuple)) and len(errors) > 0,
             "nonempty error sequence required")
    values = [_finite(v, "error") for v in errors]
    _require(all(0 <= v <= 180 for v in values), "error outside [0,180]")
    x = [0.] + sorted(values)
    recall = [i / len(values) for i in range(len(values) + 1)]
    output = {}
    for threshold in THRESHOLDS:
        end = bisect_left(x, threshold)
        xx = x[:end] + [float(threshold)]
        yy = recall[:end] + [recall[end - 1]]
        area = math.fsum((xx[i + 1] - xx[i]) * (yy[i + 1] + yy[i]) / 2
                         for i in range(len(xx) - 1))
        output[str(threshold)] = 100 * area / threshold
    return output


def _summarize(rows):
    result = {}
    for arm in ARMS:
        outcomes = [row["arms"][arm] for row in rows]
        result[arm] = {
            "denominator": len(rows),
            "auc_percent": dms_auc_percent([v["max_degrees"] for v in outcomes]),
            "status_counts": {status: sum(v["status"] == status for v in outcomes)
                              for status in STATUSES},
            "valid_pose_count": sum(v["pose_valid"] for v in outcomes),
            "genuine_completed": sum(v["genuine_completed"] for v in outcomes),
            "penalized_count": sum(v["penalized"] for v in outcomes),
        }
    return result


def evaluate_gate(pooled, scenes, *, comparator_valid):
    """Apply only the frozen four clauses to already computed summaries."""
    _require(type(comparator_valid) is bool, "comparator_valid must be boolean")
    _require(set(pooled) == set(ARMS) and set(scenes) == set(SCENES),
             "gate requires all arms and all three scenes")
    for summaries in [pooled] + [scenes[scene] for scene in SCENES]:
        _require(set(summaries) == set(ARMS), "incomplete arm summaries")
        for arm in ARMS:
            for threshold in THRESHOLDS:
                auc = _finite(summaries[arm]["auc_percent"][str(threshold)], "AUC")
                _require(0 <= auc <= 100, "AUC outside [0,100]")
    genuine = pooled["ANG"]["genuine_completed"]
    _require(type(genuine) is int and 0 <= genuine <= 24,
             "invalid ANG genuine-completed count")
    controls = tuple(arm for arm in ARMS if arm != "ANG")
    # Python max retains the first equal key; ARMS is the fixed tie order.
    strongest = max(controls, key=lambda a: pooled[a]["auc_percent"]["5"])
    strongest20 = max(controls, key=lambda a: pooled[a]["auc_percent"]["20"])
    margin5 = pooled["ANG"]["auc_percent"]["5"] - pooled[strongest]["auc_percent"]["5"]
    margin20 = (pooled["ANG"]["auc_percent"]["20"]
                - pooled[strongest20]["auc_percent"]["20"])
    scene_margins = {
        scene: (scenes[scene]["ANG"]["auc_percent"]["5"]
                - scenes[scene][strongest]["auc_percent"]["5"])
        for scene in SCENES}
    positive_scenes = sum(value > 0 for value in scene_margins.values())
    clauses = {
        "trustworthy_comparators": comparator_valid,
        "auc5_gain_and_coverage": margin5 >= 2. and genuine >= 20,
        "scene_consistency": (positive_scenes >= 2
                              and min(scene_margins.values()) >= -2.),
        "auc20_noninferiority": margin20 >= -.5,
    }
    return {
        "passed": all(clauses.values()),
        "clauses": clauses,
        "comparator_valid": comparator_valid,
        "strongest_control_auc5": strongest,
        "strongest_control_auc20": strongest20,
        "auc5_margin_pp": margin5,
        "auc20_margin_pp": margin20,
        "ang_genuine_completed": genuine,
        "positive_scene_count": positive_scenes,
        "scene_auc5_margins_pp": scene_margins,
        "all_control_margins_pp": {
            arm: {str(threshold): (pooled["ANG"]["auc_percent"][str(threshold)]
                                  - pooled[arm]["auc_percent"][str(threshold)])
                  for threshold in THRESHOLDS}
            for arm in controls},
        "manuscript_admission": False,
        "interpretation": (
            "A pass supports only a separately designed larger development "
            "validation and contribution assessment; no novelty or held-out claim."),
    }


def evaluate_records(prediction_rows, truth_map, *, comparator_valid):
    """Evaluate exactly B1..B8 in each fixed scene, retaining failed outcomes.

    Invalid prediction poses are penalized, never projected, sign-flipped or
    excluded. Invalid truth is an integrity failure. Fallback poses can be
    scored but do not count as genuine ANG results. Identity/provenance beyond
    this fixed 24-row key set is the caller's sealed-input responsibility.
    """
    _require(type(comparator_valid) is bool, "comparator_valid must be boolean")
    _require(isinstance(prediction_rows, (list, tuple)) and len(prediction_rows) == 24,
             "exactly 24 prediction rows required")
    _require(isinstance(truth_map, dict), "truth_map must be a mapping")
    expected = {(scene, query) for scene in SCENES for query in QUERIES}
    seen = set()
    anchors = {scene: set() for scene in SCENES}
    targets = {scene: set() for scene in SCENES}
    for row in prediction_rows:
        _require(isinstance(row, dict), "prediction row must be a mapping")
        key = (row.get("scene"), row.get("query"))
        _require(key in expected and key not in seen, "unknown or duplicate query")
        seen.add(key)
        for name in ("anchor", "target"):
            _require(isinstance(row.get(name), str) and row[name],
                     name + " identity must be nonempty")
        _require(row["anchor"] != row["target"], "query equals its anchor")
        anchors[key[0]].add(row["anchor"])
        targets[key[0]].add(row["target"])
        _require(isinstance(row.get("arms"), dict) and set(row["arms"]) == set(ARMS),
                 "each query requires all eleven arms")
        for arm in ARMS:
            outcome = row["arms"][arm]
            _require(isinstance(outcome, dict) and outcome.get("status") in STATUSES,
                     "invalid arm status")
    _require(seen == expected and set(truth_map) == expected,
             "prediction/truth query coverage differs")
    _require(all(len(anchors[s]) == 1 and len(targets[s]) == 8 for s in SCENES),
             "each scene requires one anchor and eight distinct targets")
    truths = {key: checked_pose(truth_map[key]) for key in expected}
    rows = []
    for source in prediction_rows:
        key = (source["scene"], source["query"])
        row = {name: source[name] for name in ("scene", "query", "anchor", "target")}
        row["arms"] = {}
        for arm in ARMS:
            outcome = source["arms"][arm]
            status = outcome["status"]
            reason = "unavailable_outcome" if status == "unavailable" else None
            error = None
            if status != "unavailable":
                try:
                    error = _checked_error(checked_pose(outcome), truths[key])
                except ValueError as exc:
                    reason = "invalid_prediction_pose: " + str(exc)
            valid = error is not None
            if error is None:
                error = dict(rotation_degrees=180., translation_degrees=180.,
                             max_degrees=180.)
            row["arms"][arm] = {
                "status": status, "pose_valid": valid, "penalized": not valid,
                "genuine_completed": status == "completed" and valid,
                "penalty_reason": reason, **error,
            }
        rows.append(row)
    pooled = _summarize(rows)
    scenes = {scene: _summarize([r for r in rows if r["scene"] == scene])
              for scene in SCENES}
    return {
        "schema": "evr0028.metrics.v1",
        "query_count": 24, "scene_count": 3,
        "arm_order": list(ARMS), "scene_order": list(SCENES),
        "primary_error": "max(rotation_angle,directed_translation_angle)",
        "failure_penalty_degrees": 180.,
        "auc_convention": "DMS_left_search_trapezoid_percent",
        "rows": rows, "pooled": pooled, "scenes": scenes,
        "gate": evaluate_gate(pooled, scenes, comparator_valid=comparator_valid),
        "manuscript_admission": False,
    }
