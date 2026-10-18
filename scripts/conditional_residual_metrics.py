"""Independent NumPy scoring only. No trainer imports, file reads, or fitting."""
import hashlib
import math

import numpy as np


def _digest(array, dtype):
    return hashlib.sha256(np.asarray(array, dtype=dtype, order="C").tobytes()).hexdigest()


def _error_summary(error):
    with np.errstate(over="raise", invalid="raise"):
        sse = float(np.sum(np.square(error), dtype=np.float64))
    if not math.isfinite(sse):
        raise ValueError("nonfinite accumulated error")
    return {"sse": sse, "count": int(error.size), "mse": sse / error.size}


def score_pair(clean_center, prediction):
    """Score matching HxW arrays (H,W>=8), without clipping; data range is1.

    Production caller crops a128x128 clean image as clean[3:-3,3:-3].
    Prediction must already be the saved float32 valid122x122 array.
    This function also permits smaller >=8x8 arrays for analytical fixtures.
    """
    clean = np.asarray(clean_center)
    pred = np.asarray(prediction)
    if clean.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("clean must be float32 or float64")
    if pred.dtype != np.dtype("float32"):
        raise ValueError("saved prediction must be float32")
    if clean.ndim != 2 or clean.shape != pred.shape or min(clean.shape) < 8:
        raise ValueError("matching two-dimensional arrays of at least8x8 required")
    if not np.isfinite(clean).all() or not np.isfinite(pred).all():
        raise ValueError("nonfinite input")
    clean64, pred64 = clean.astype(np.float64), pred.astype(np.float64)
    try:
        with np.errstate(over="raise", invalid="raise"):
            error = pred64 - clean64
            pixel = _error_summary(error)
            horizontal = _error_summary(np.diff(error, axis=1))
            vertical = _error_summary(np.diff(error, axis=0))
            blocks = []
            for row, rows in enumerate(np.array_split(np.arange(clean.shape[0]), 8)):
                for col, cols in enumerate(np.array_split(np.arange(clean.shape[1]), 8)):
                    block = _error_summary(error[np.ix_(rows, cols)])
                    blocks.append({"row": row, "col": col,
                                   "row_start": int(rows[0]), "row_stop": int(rows[-1]) + 1,
                                   "col_start": int(cols[0]), "col_stop": int(cols[-1]) + 1,
                                   "sse": block["sse"], "count": block["count"]})
    except FloatingPointError as exc:
        raise ValueError("numeric overflow while scoring") from exc
    fd_sse = horizontal["sse"] + vertical["sse"]
    if not math.isfinite(fd_sse):
        raise ValueError("nonfinite combined first-difference error")
    fd_count = horizontal["count"] + vertical["count"]
    return {"shape": list(clean.shape), **pixel,
            "psnr_db": None if pixel["mse"] == 0 else -10.0 * math.log10(pixel["mse"]),
            "data_range": 1, "clipped": False,
            "first_difference": {"horizontal": horizontal, "vertical": vertical,
                                 "combined": {"sse": fd_sse, "count": fd_count,
                                              "mse": fd_sse / fd_count}},
            "blocks": blocks,
            "clean_sha256_f64le": _digest(clean, "<f8"),
            "prediction_sha256_f32le": _digest(pred, "<f4")}


def _close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)


def _validate_score(score, shape):
    if tuple(score["shape"]) != tuple(shape) or score["data_range"] != 1 or score["clipped"] is not False:
        raise ValueError("score geometry/range/clipping mismatch")
    h, w = shape
    def check(part, count):
        if part["count"] != count:
            raise ValueError("metric count mismatch")
        if any(not math.isfinite(part[k]) or part[k] < 0 for k in ("sse", "mse")):
            raise ValueError("invalid error metric")
        if not _close(part["mse"], part["sse"] / count):
            raise ValueError("SSE/MSE mismatch")
    check(score, h * w)
    expected = None if score["mse"] == 0 else -10 * math.log10(score["mse"])
    if expected is None and score["psnr_db"] is not None:
        raise ValueError("zero-error PSNR must be null")
    if expected is not None and (score["psnr_db"] is None or not _close(score["psnr_db"], expected)):
        raise ValueError("PSNR mismatch")
    fd = score["first_difference"]
    check(fd["horizontal"], h * (w - 1))
    check(fd["vertical"], (h - 1) * w)
    check(fd["combined"], h * (w - 1) + (h - 1) * w)
    if not _close(fd["combined"]["sse"], fd["horizontal"]["sse"] + fd["vertical"]["sse"]):
        raise ValueError("first-difference sum mismatch")
    rows, cols = np.array_split(np.arange(h), 8), np.array_split(np.arange(w), 8)
    if len(score["blocks"]) != 64:
        raise ValueError("64 rectangular blocks required")
    for block, (r, c) in zip(score["blocks"], ((r, c) for r in range(8) for c in range(8))):
        expected_bounds = (r, c, int(rows[r][0]), int(rows[r][-1]) + 1,
                           int(cols[c][0]), int(cols[c][-1]) + 1, len(rows[r]) * len(cols[c]))
        actual = tuple(block[k] for k in ("row", "col", "row_start", "row_stop", "col_start", "col_stop", "count"))
        if actual != expected_bounds or not math.isfinite(block["sse"]) or block["sse"] < 0:
            raise ValueError("invalid block geometry or SSE")
    if not _close(sum(b["sse"] for b in score["blocks"]), score["sse"]):
        raise ValueError("block SSE does not cover image")
    for key in ("clean_sha256_f64le", "prediction_sha256_f32le"):
        value = score[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("invalid array digest")


def _gain(candidate, baseline):
    """Both numerically perfect => zero harm; one perfect => undefined finite dB gain."""
    if candidate["mse"] == 0 and baseline["mse"] == 0:
        return 0.0
    if candidate["mse"] == 0 or baseline["mse"] == 0:
        return None
    return candidate["psnr_db"] - baseline["psnr_db"]


def aggregate_cases(cases, *, groups, seeds, candidate="residual",
                    baselines=("blindspot", "safe_blindspot", "r2r_alpha05"), white_reference="blindspot",
                    expected_shape=(122, 122)):
    """Require complete4x2x2x4 case grid; records contain group/condition/seed/method/score.

    Conditions are exactly 'white' and 'colored'. Ties in the prespecified
    oracle-best-PSNR baseline use the caller's baseline order. No cases drop.
    Zero-error one-sided gains make the finite-dB gate not evaluable.
    """
    groups, seeds, baselines = tuple(groups), tuple(seeds), tuple(baselines)
    if len(groups) != 4 or len(set(groups)) != 4 or len(seeds) != 2 or len(set(seeds)) != 2:
        raise ValueError("exactly4 unique groups and2 seeds required")
    if len(baselines) != 3 or len(set(baselines)) != 3 or candidate in baselines or white_reference not in baselines or "safe_blindspot" not in baselines:
        raise ValueError("candidate plus3 unique baselines including safe_blindspot and a valid white reference required")
    methods = (candidate,) + baselines
    expected = {(g, c, s, m) for g in groups for c in ("white", "colored") for s in seeds for m in methods}
    indexed, clean_hash = {}, {}
    for row in cases:
        key = (row["group"], row["condition"], row["seed"], row["method"])
        if key not in expected or key in indexed:
            raise ValueError("unexpected or duplicate case")
        _validate_score(row["score"], expected_shape)
        digest = row["score"]["clean_sha256_f64le"]
        if key[0] in clean_hash and clean_hash[key[0]] != digest:
            raise ValueError("clean image differs across cases of a group")
        clean_hash[key[0]] = digest
        indexed[key] = row["score"]
    if set(indexed) != expected:
        raise ValueError("incomplete64-case grid")
    colored, white = [], []
    for group in groups:
        for seed in seeds:
            cs = indexed[group, "colored", seed, candidate]
            chosen = min(baselines, key=lambda m: indexed[group, "colored", seed, m]["mse"])
            bs = indexed[group, "colored", seed, chosen]
            colored.append({"group": group, "seed": seed, "oracle_best_baseline": chosen,
                            "candidate_psnr_db": cs["psnr_db"], "baseline_psnr_db": bs["psnr_db"],
                            "gain_db": _gain(cs, bs),
                            "candidate_first_difference_mse": cs["first_difference"]["combined"]["mse"],
                            "baseline_first_difference_mse": bs["first_difference"]["combined"]["mse"]})
            ws, wb = indexed[group, "white", seed, candidate], indexed[group, "white", seed, white_reference]
            white.append({"group": group, "seed": seed,
                          "prediction_bitwise_equal": ws["prediction_sha256_f32le"] == wb["prediction_sha256_f32le"],
                          "gain_db": _gain(ws, wb), "candidate_mse": ws["mse"], "reference_mse": wb["mse"]})
            if "safe_blindspot" in baselines:
                safe = indexed[group, "white", seed, "safe_blindspot"]
                white[-1]["safe_blindspot_bitwise_equal_to_reference"] = safe["prediction_sha256_f32le"] == wb["prediction_sha256_f32le"]
                white[-1]["safe_blindspot_gain_db"] = _gain(safe, wb)
    finite_gain = all(r["gain_db"] is not None for r in colored)
    group_gains = {}
    for group in groups:
        gains = [r["gain_db"] for r in colored if r["group"] == group]
        group_gains[group] = sum(gains) / 2 if all(v is not None for v in gains) else None
    mean_gain = sum(r["gain_db"] for r in colored) / 8 if finite_gain else None
    fd_candidate = sum(r["candidate_first_difference_mse"] for r in colored) / 8
    fd_baseline = sum(r["baseline_first_difference_mse"] for r in colored) / 8
    gates = {"colored_pair_mean_ge_0_3_db": mean_gain >= 0.3 if finite_gain else None,
             "colored_at_least3_group_means_gt0": sum(v > 0 for v in group_gains.values()) >= 3 if finite_gain else None,
             "colored_each_group_mean_ge_minus0_2_db": all(v >= -0.2 for v in group_gains.values()) if finite_gain else None,
             "colored_first_difference_le_1_05": fd_candidate <= 1.05 * fd_baseline,
             "white_predictions_bitwise_equal": all(r["prediction_bitwise_equal"] for r in white),
             "white_safe_predictions_bitwise_equal": all(r["safe_blindspot_bitwise_equal_to_reference"] for r in white),
             "white_psnr_harm_zero": all(r["gain_db"] == 0 and r["candidate_mse"] == r["reference_mse"] for r in white)}
    status = "not_evaluable" if any(v is None for v in gates.values()) else ("continue_mechanism_exploration" if all(gates.values()) else "stop_this_candidate")
    ordered_cases = [{"group": g, "condition": c, "seed": s, "method": m, "score": indexed[g, c, s, m]}
                     for g in groups for c in ("white", "colored") for s in seeds for m in methods]
    return {"status": status, "development_only": True, "paper_success": False,
            "case_count": len(indexed), "oracle_comparator": "best PSNR baseline per colored group/seed; ties use declared baseline order",
            "groups": list(groups), "seeds": list(seeds), "methods": list(methods),
            "mean_colored_gain_db": mean_gain, "group_mean_colored_gain_db": group_gains,
            "mean_candidate_first_difference_mse": fd_candidate,
            "mean_corresponding_baseline_first_difference_mse": fd_baseline,
            "gates": gates, "colored_pairs": colored, "white_pairs": white,
            "cases": ordered_cases}
