"""Pure saved-J/F aggregation for the fixed 0034 exposed-development screen."""
from copy import deepcopy
import json
import math


ARMS = ("native", "logit", "absolute", "prototype", "nn", "ordinary_relations",
        "bilateral", "signed_no_gram", "signed_gram", "displaced_gram", "rgb256")
CONTROLS = ("native", "logit", "absolute", "prototype", "nn",
            "ordinary_relations", "signed_no_gram", "bilateral")
PARENTS = (("dog-agility", 25), ("planes-water", 38),
           ("dogs-scale", 83), ("dog-gooses", 86))
KEYS = ("J", "F", "JF")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def means(values):
    require(bool(values), "empty metric population")
    return {key: math.fsum(item[key] for item in values) / len(values) for key in KEYS}


def differences(arms):
    return {arm: {key: arms[arm][key] - arms["native"][key] for key in KEYS} for arm in ARMS}


def aggregate(selected, targets):
    """Equal frames/events within parent; equal means across the four parents."""
    parents = []
    for sequence, frames in PARENTS:
        chosen = [row for row in selected if row["sequence"] == sequence]
        arms = {arm: means([row["arms"][arm] for row in chosen]) for arm in ARMS}
        parents.append(dict(sequence=sequence, frames=frames, target_id=targets[sequence],
                            included_count=len(chosen), arms=arms, delta_from_native=differences(arms)))
    pooled = {arm: means([parent["arms"][arm] for parent in parents]) for arm in ARMS}
    return dict(parents=parents, mean=pooled, delta_from_native=differences(pooled),
                weighting="equal observations within each parent, then four equal parent means")


def build_report(rows, fit_failures):
    """Validate all 232 saved rows and aggregate; never decode labels or masks.

    Additional row fields and failure records are copied unchanged. Malformed or
    incomplete inputs raise ValueError; declared fit failures retain full coverage
    but make continuation false. No numerical tolerance is added to any gate.
    """
    require(isinstance(rows, list) and isinstance(fit_failures, list), "rows/failures must be lists")
    try:
        json.dumps((rows, fit_failures), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("records must be finite JSON-compatible values") from error
    rows, failures = deepcopy(rows), deepcopy(fit_failures)
    lengths, targets, indexed = dict(PARENTS), {}, {}
    for row in rows:
        require(isinstance(row, dict), "row must be a dictionary")
        require({"sequence", "frame", "frames", "target_id", "arms"} <= row.keys(), "missing row fields")
        sequence, frame, frames, target = (row[k] for k in ("sequence", "frame", "frames", "target_id"))
        require(isinstance(sequence, str) and sequence in lengths, "unexpected parent")
        require(type(frame) is int and type(frames) is int and frames == lengths[sequence]
                and 0 <= frame < frames, "frame identity/range")
        require(type(target) is int and target > 0 and target != 255, "invalid target identity")
        require(sequence not in targets or targets[sequence] == target, "target changes within parent")
        targets[sequence] = target
        identity = (sequence, frame)
        require(identity not in indexed, "duplicate frame")
        require(isinstance(row["arms"], dict) and set(row["arms"]) == set(ARMS), "exact eleven arms required")
        for arm in ARMS:
            metric = row["arms"][arm]
            require(isinstance(metric, dict) and set(KEYS) <= metric.keys(), "missing arm metric")
            for key in KEYS:
                value = metric[key]
                require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                        "metric outside finite [0,1]")
            require(metric["JF"] == (metric["J"] + metric["F"]) / 2, "JF is not (J+F)/2")
        indexed[identity] = row
    expected = [(sequence, frame) for sequence, frames in PARENTS for frame in range(frames)]
    require(len(rows) == 232 and set(indexed) == set(expected), "complete 232-frame population required")
    ordered = [indexed[identity] for identity in expected]
    interiors = [row for row in ordered if 0 < row["frame"] < row["frames"] - 1]
    primary = aggregate(interiors, targets)
    primary.update(frame_count=len(interiors), endpoint_policy="exclude first and last of each parent")
    require(len(interiors) == 224, "primary population mismatch")

    events = []
    for sequence, frames in PARENTS:
        anchors = [1 + i * (frames - 5) // 11 for i in range(12)]
        require(len(set(anchors)) == 12, "duplicate anchor")
        for anchor in anchors:
            for horizon in (1, 3):
                frame = anchor + horizon
                row = indexed[sequence, frame]
                events.append(dict(sequence=sequence, frame=frame, frames=frames,
                    target_id=targets[sequence], anchor=anchor, horizon=horizon,
                    final_frame=frame == frames - 1, arms=deepcopy(row["arms"])))
    unique = len({(row["sequence"], row["frame"]) for row in events})
    final_events = sum(row["final_frame"] for row in events)
    require(len(events) == 96 and unique == 87 and final_events == 4, "secondary population mismatch")
    secondary = aggregate(events, targets)
    secondary.update(events=events, event_count=96, unique_frames=unique, final_frame_events=final_events,
        scope="descriptive old horizon population; repeated frames and four final-frame events retained")

    pooled = primary["mean"]
    # max retains the first item at an exact tie: the frozen CONTROLS order.
    strongest = max(CONTROLS, key=lambda arm: pooled[arm]["JF"])
    parent_margins = []
    for parent in primary["parents"]:
        score = parent["arms"]["signed_gram"]["JF"]
        parent_margins.append(dict(sequence=parent["sequence"],
            vs_native=score - parent["arms"]["native"]["JF"],
            vs_ordinary_relations=score - parent["arms"]["ordinary_relations"]["JF"],
            vs_signed_no_gram=score - parent["arms"]["signed_no_gram"]["JF"],
            vs_strongest_pooled_control=score - parent["arms"][strongest]["JF"]))
    positive_ordinary = sum(row["vs_ordinary_relations"] > 0 for row in parent_margins)
    positive_no_gram = sum(row["vs_signed_no_gram"] > 0 for row in parent_margins)
    delta = primary["delta_from_native"]["signed_gram"]
    clauses = dict(
        primary_JF_gain_at_least_1pp=delta["JF"] >= .01,
        margin_over_strongest_control_at_least_quarter_pp=
            pooled["signed_gram"]["JF"] - pooled[strongest]["JF"] >= .0025,
        beats_ordinary_relations_on_at_least_three_parents=positive_ordinary >= 3,
        beats_signed_no_gram_on_at_least_three_parents=positive_no_gram >= 3,
        no_parent_native_JF_loss_exceeds_half_pp=all(row["vs_native"] >= -.005 for row in parent_margins),
        mean_native_J_loss_at_most_half_pp=delta["J"] >= -.005,
        mean_native_F_loss_at_most_half_pp=delta["F"] >= -.005,
        mean_JF_above_displaced_gram=pooled["signed_gram"]["JF"] > pooled["displaced_gram"]["JF"],
        mean_JF_above_rgb256_by_at_least_tenth_pp=
            pooled["signed_gram"]["JF"] - pooled["rgb256"]["JF"] >= .001,
        no_fit_failures=not failures,
        full_coverage=True)
    passed = all(clauses.values())
    return dict(schema="evr0034.relative-rgb-metrics.v1", official_benchmark_result=False,
        scope="four exposed TRAIN parents; out-of-parent development comparison, not final held-out evidence",
        metric_scale="[0,1]; multiply differences by100 for absolute percentage points",
        arm_order=list(ARMS), rows=ordered, fit_failures=failures,
        coverage=dict(complete=True, raw_frames=232, noninitial_frames=228, primary_frames=224,
                      parents=4, arms=11, raw_arm_rows=232 * 11),
        primary=primary, secondary=secondary,
        decision=dict(strongest_control=dict(arm=strongest, JF=pooled[strongest]["JF"],
                                            tie_order=list(CONTROLS)),
            parent_margins=parent_margins,
            positive_parent_counts=dict(ordinary_relations=positive_ordinary, signed_no_gram=positive_no_gram),
            candidate_JF=pooled["signed_gram"]["JF"],
            margin_over_strongest_control=pooled["signed_gram"]["JF"] - pooled[strongest]["JF"],
            clauses=clauses, passed=passed,
            action="development_continuation_only" if passed else "close_fixed_signed_rgb_gram_recipe"),
        goal_achieved=False, manuscript_claim=False)
