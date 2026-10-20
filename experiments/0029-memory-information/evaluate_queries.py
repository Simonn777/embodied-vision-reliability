"""Pure post-seal evaluation of the fixed 0029 queries.

The caller owns prediction/GT acquisition, source and artifact admission. This
module does no I/O and never chooses a deployed branch. All identities, sealed
choices and prediction arrays are checked before annotations are accessed.
"""
import math

import numpy as np

SCHEMA = "evr0029.query-evaluation.v1"
NATIVE = ("M0", "M1", "M2")


def require(value, message):
    if not value:
        raise ValueError(message)


def finite_number(value, name):
    require(type(value) in (int, float) and math.isfinite(value), name + " must be finite")
    return float(value)


def costs(value):
    require(isinstance(value, dict) and set(value) == {"seconds", "counts"},
            "costs require separate seconds/counts maps")
    result = {"seconds": {}, "counts": {}}
    for name, number in value["seconds"].items():
        require(isinstance(name, str) and bool(name), "invalid timing name")
        result["seconds"][name] = finite_number(number, "timing")
        require(number >= 0, "negative timing")
    for name, number in value["counts"].items():
        require(isinstance(name, str) and bool(name) and type(number) is int and number >= 0,
                "invalid call count")
        result["counts"][name] = number
    return result


def validate_completed(row, core):
    target = row["target_id"]
    require(type(target) is int and target > 0 and target != 255, "invalid fixed target ID")
    evidence = row["witnesses"]
    require(evidence["target_id"] == target and evidence["actionable"] is row["actionable"],
            "witness identity/actionability mismatch")
    shape = evidence["current_shape"]
    require(len(shape) == 2 and all(type(x) is int and x > 0 for x in shape),
            "invalid original-image shape")
    positive, negative = evidence["positive"], evidence["negative"]
    require(len(positive) <= 32 and len(negative) <= 32
            and row["actionable"] == (len(positive) >= 4 and len(negative) >= 4),
            "witness-count actionability mismatch")
    rounded = set()
    for name, label in (("positive", 1), ("negative", 0)):
        for witness in evidence[name]:
            xy = np.asarray(witness["target_xy"], dtype=np.float64)
            require(xy.shape == (2,) and np.all(np.isfinite(xy))
                    and 0 <= xy[0] < shape[1] and 0 <= xy[1] < shape[0],
                    "invalid retained witness coordinates")
            px, py = map(int, np.floor(xy + .5))
            require(0 <= px < shape[1] and 0 <= py < shape[0]
                    and witness["rounded_current_xy"] == [px, py]
                    and witness["label"] == label, "retained witness lookup mismatch")
            require((px, py) not in rounded, "retained witness collision")
            rounded.add((px, py))
    candidates = row["candidates"]
    mandatory = set(core.BANK) | set(NATIVE)
    require(isinstance(candidates, dict) and mandatory <= set(candidates)
            and set(candidates) <= mandatory | {"P", "R"}, "candidate bank changed")
    require(not row["actionable"] or {"P", "R"} <= set(candidates),
            "actionable prompt/reset candidate missing")
    masks = {}
    for name, value in candidates.items():
        array = np.asarray(value)
        require(array.shape == tuple(shape) and array.dtype.kind in "bif"
                and np.all(np.isfinite(array)), "invalid original-resolution prediction")
        masks[name] = array if array.dtype.kind == "b" else array > 0
    selection = row["selection"]
    choices = selection["choices"]
    require(selection["actionable"] is row["actionable"]
            and isinstance(choices, dict) and set(choices) == set(core.METHODS),
            "sealed selection schema changed")
    require(choices["baseline"] == "B" and all(x in masks for x in choices.values()),
            "selected candidate missing")
    bank_methods = ("signed_deletion", "baseline", "max_predicted_iou",
                    "max_unsigned_change", "always_D1", "always_D3",
                    "random_deletion", "random_on_signed")
    require(all(choices[name] in core.BANK for name in bank_methods),
            "fixed-bank selector escaped its bank")
    require(choices["native_bank"] in ("B", *NATIVE)
            and choices["prompt_signed"] in ("B", "P")
            and choices["reset_signed"] in ("B", "R"), "control bank changed")
    require(selection["signed_action"] is (choices["signed_deletion"] != "B"),
            "sealed action mismatch")
    if row["actionable"]:
        require(choices["prompt_direct"] == "P" and choices["reset_direct"] == "R"
                and choices["always_D1"] == "D1" and choices["always_D3"] == "D3",
                "direct-control choice changed")
        require(set(selection["bank_losses"]) == set(core.BANK), "sealed loss bank changed")
        for number in selection["bank_losses"].values():
            require(finite_number(number, "sealed loss") >= 0, "negative sealed loss")
    else:
        require(all(x == "B" for x in choices.values()) and selection["bank_losses"] is None,
                "nonactionable deployed choices must remain baseline")
    return masks, costs(row["costs"])


def witness_precision(evidence, annotation, target_id):
    result = {}
    for name, positive in (("positive", True), ("negative", False)):
        total = len(evidence[name])
        known = correct = void = 0
        for witness in evidence[name]:
            x, y = witness["rounded_current_xy"]
            observed = int(annotation[y, x])
            if observed == 255:
                void += 1
            else:
                known += 1
                correct += int((observed == target_id) == positive)
        result[name] = {"retained": total, "known": known, "void_unknown": void,
                        "correct": correct, "incorrect": known - correct,
                        "precision": correct / known if known else None}
    return result


def sign(value):
    return 1 if value > 0 else (-1 if value < 0 else 0)


def sum_costs(rows):
    """Sum each named column only; never add wall and stage times together."""
    result = {"seconds": {}, "counts": {}}
    for unit in result:
        names = sorted({name for row in rows for name in row["costs"][unit]})
        for name in names:
            present = [row["costs"][unit][name] for row in rows if name in row["costs"][unit]]
            result[unit][name] = {"sum": math.fsum(present) if unit == "seconds" else sum(present),
                                 "reported_queries": len(present)}
    return result


def describe(rows, core):
    result = {"queries": len(rows), "precision": {}, "native_bank": {},
              "method_choices": {}, "loss_j_sign": {}, "costs": sum_costs(rows)}
    for label in ("positive", "negative"):
        counts = {name: sum(row["witness_precision"][label][name] for row in rows)
                  for name in ("retained", "known", "void_unknown", "correct", "incorrect")}
        counts["precision"] = counts["correct"] / counts["known"] if counts["known"] else None
        result["precision"][label] = counts
    for mode in ("admissible", "unrestricted"):
        gains = [row["native_oracle"][mode + "_gain"] for row in rows]
        result["native_bank"][mode + "_mean_gain"] = math.fsum(gains) / len(rows)
    for method in core.METHODS:
        names = sorted({row["selected_branches"][method] for row in rows})
        result["method_choices"][method] = {
            "branch_counts": {name: sum(row["selected_branches"][method] == name for row in rows)
                              for name in names},
            "selected_baseline": sum(row["selected_branches"][method] == "B" for row in rows),
            "selected_mask_equals_baseline": sum(row["selected_mask_noop"][method] for row in rows)}
    for branch in ("D1", "D3", "selected"):
        values = [row["loss_j_sign"][branch] for row in rows
                  if row["loss_j_sign"] is not None]
        result["loss_j_sign"][branch] = {
            "reported_queries": len(values),
            "sign_pair_counts": {"%d,%d" % (a, b): sum(
                item["loss_gain_sign"] == a and item["j_gain_sign"] == b for item in values)
                for a in (-1, 0, 1) for b in (-1, 0, 1)},
            "strict_nonzero_concordant": sum(
                item["loss_gain_sign"] == item["j_gain_sign"] != 0 for item in values),
            "strict_nonzero_discordant": sum(
                item["loss_gain_sign"] * item["j_gain_sign"] == -1 for item in values)}
    harmed = [row for row in rows if row["signed_harm"]["drop_gt_005"]]
    worst = max(harmed, key=lambda row: row["signed_harm"]["j_drop"]) if harmed else None
    result["harmful_signed_actions"] = {
        "threshold_strict_j_drop": .05, "count": len(harmed),
        "sum_drop": math.fsum(row["signed_harm"]["j_drop"] for row in harmed),
        "mean_drop_among_harmful": math.fsum(row["signed_harm"]["j_drop"] for row in harmed) / len(harmed)
        if harmed else None,
        "worst": {key: worst[key] for key in ("sequence", "frame")} |
                 {"j_drop": worst["signed_harm"]["j_drop"]} if worst else None}
    return result


def evaluate_queries(rows, annotations, *, core):
    """Evaluate sealed choices; ``annotations`` is indexed only after preflight.

    ``rows``: all 48 fixed identities, statuses, frozen target/action/selection,
    witnesses, original-resolution candidates and costs={seconds:{},counts:{}}.
    ``annotations``: mapping (sequence, frame) to the admitted integer GT array.
    No oracle is a selectable method or part of the named-control maximum.
    """
    rows = list(rows)
    coverage = core.evidence_coverage(rows)  # no annotations touched
    require(all(row.get("status") in ("completed", "failed") for row in rows),
            "unknown query prediction status")
    if any(row["status"] != "completed" for row in rows):
        return {"schema": SCHEMA, "eligible": False, "denominator": 48,
                "coverage": coverage, "summary": None, "diagnostics": None,
                "gate": {"pass": False}, "gt_accessed": False,
                "rows": [{key: row[key] for key in ("sequence", "frame", "status")}
                         | {"reason": row.get("reason")} for row in rows],
                "reason": "incomplete technical prediction cohort; no successful-subset means"}
    order = {identity: i for i, identity in enumerate(core.QUERY_IDENTITIES)}
    rows.sort(key=lambda row: order[(row["sequence"], row["frame"])])
    prepared = [validate_completed(row, core) for row in rows]
    for sequence, _ in core.COHORT:
        require(len({row["target_id"] for row in rows if row["sequence"] == sequence}) == 1,
                "target changed within video")
    # Coverage, identity, choices, candidate shapes, witnesses and costs have all
    # been validated above. Ground truth begins here, never in branch selection.
    require(set(annotations) == set(core.QUERY_IDENTITIES), "GT identity coverage mismatch")
    metric_rows, output_rows = [], []
    for row, (masks, row_costs) in zip(rows, prepared):
        identity = row["sequence"], row["frame"]
        annotation = np.asarray(annotations[identity])
        candidate_j = {name: core.foreground_j(mask, annotation, row["target_id"])
                       for name, mask in masks.items()}
        choices = dict(row["selection"]["choices"])
        metric_rows.append({key: row[key] for key in ("sequence", "frame", "status", "actionable",
                                                     "selection")} | {"candidate_j": candidate_j})
        baseline = candidate_j["B"]
        native_best = max(candidate_j[name] for name in ("B", *NATIVE))
        selected = choices["signed_deletion"]
        j_delta = candidate_j[selected] - baseline
        relationships = None
        if row["actionable"]:
            loss = row["selection"]["bank_losses"]
            relationships = {}
            for name, branch in (("D1", "D1"), ("D3", "D3"), ("selected", selected)):
                loss_gain = loss["B"] - loss[branch]
                gain = candidate_j[branch] - baseline
                relationships[name] = {"branch": branch, "loss_gain": loss_gain, "j_gain": gain,
                                       "loss_gain_sign": sign(loss_gain), "j_gain_sign": sign(gain)}
        output_rows.append({key: row[key] for key in ("sequence", "frame", "target_id", "status",
                                                     "actionable")} | {
            "candidate_j": candidate_j, "selected_branches": choices,
            "witness_precision": witness_precision(row["witnesses"], annotation, row["target_id"]),
            "loss_j_sign": relationships,
            "signed_harm": {"action": selected != "B", "j_delta": j_delta,
                            "j_drop": max(0., -j_delta),
                            "drop_gt_005": selected != "B" and -j_delta > .05},
            "selected_mask_noop": {method: bool(np.array_equal(masks[branch], masks["B"]))
                                   for method, branch in choices.items()},
            "candidate_mask_noop": {name: bool(np.array_equal(mask, masks["B"]))
                                    for name, mask in masks.items()},
            "native_oracle": {"unrestricted_j": native_best,
                              "unrestricted_gain": native_best - baseline,
                              "admissible_j": native_best if row["actionable"] else baseline,
                              "admissible_gain": native_best - baseline if row["actionable"] else 0.},
            "costs": row_costs})
    summary = core.summarize(metric_rows)
    return {"schema": SCHEMA, "eligible": summary["eligible"], "denominator": 48,
            "coverage": coverage, "gt_accessed": True, "summary": summary,
            "gate": dict(summary["gate"]), "rows": output_rows,
            "diagnostics": {"pooled": describe(output_rows, core),
                "videos": {name: describe([row for row in output_rows if row["sequence"] == name], core)
                           for name, _ in core.COHORT}},
            "scope": {"metric": "custom fixed-query foreground J; not official DAVIS J&F",
                      "oracle_not_deployed": True,
                      "sign_agreement": "Exact zero/nonzero signs; descriptive, no threshold selection",
                      "costs": "Named-column sums only; overlapping wall/stage columns are not added",
                      "admission": "Prediction/GT artifact and outer-seal admission is caller-owned"}}
