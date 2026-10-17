#!/usr/bin/env python3
"""Independent saved-matrix review; stdlib only, never imports project code.

Input JSON is either {"K": nine numbers or 3x3, "RT": sixteen numbers or 4x4}
or {"K_hex": "...", "RT_hex": "...", "byte_order": "little" | "big"}.
Numeric inputs must already be exact decoded float32 values: no silent rounding.
Read JSON with --input FILE or from stdin. --self-test uses artificial data only.
Output goes to stdout. No repository, archive, model, network, or cache access.
"""

import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

U = F(1, 2**24)
TAU = 16 * U  # max(1e-8, 8*epsilon_f32) = 2**-20
GRAM_BOUND = 6 * U + 3 * U**2
DET_BOUND = 9 * U + 27 * U**2 + 27 * U**3


def unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key: " + key)
        out[key] = value
    return out


def reject_constant(value):
    raise ValueError("nonfinite JSON literal: " + value)


def flatten(value, dimension):
    if not isinstance(value, list):
        raise ValueError("matrix must be a list")
    if len(value) == dimension and all(isinstance(row, list) for row in value):
        if any(len(row) != dimension for row in value):
            raise ValueError("matrix row shape")
        return [x for row in value for x in row]
    if len(value) != dimension * dimension or any(isinstance(x, list) for x in value):
        raise ValueError("flat matrix shape")
    return value


def checked_f32(value):
    if type(value) not in (int, float):
        raise ValueError("numeric matrices require real JSON numbers, not booleans")
    try:
        number = float(value)
        encoded = struct.pack("<f", number)
    except (OverflowError, struct.error) as exc:
        raise ValueError("value is outside finite float32") from exc
    if not math.isfinite(number):
        raise ValueError("nonfinite matrix coefficient")
    if type(value) is int and F.from_float(number) != value:
        raise ValueError("integer loses precision before float32 checking")
    if struct.unpack("<f", encoded)[0] != number:
        raise ValueError("not an exact decoded float32; supply original byte hex or exact decoded values")
    return number


def decode_input(source):
    if type(source) is not dict:
        raise ValueError("input must be a JSON object")
    if set(source) == {"K", "RT"}:
        byte_order = "little"
        values = {
            "K": [checked_f32(x) for x in flatten(source["K"], 3)],
            "RT": [checked_f32(x) for x in flatten(source["RT"], 4)],
        }
        payloads = {key: struct.pack("<" + str(len(v)) + "f", *v) for key, v in values.items()}
        input_kind = "exact_decoded_float32_numbers"
    elif set(source) in ({"K_hex", "RT_hex"}, {"K_hex", "RT_hex", "byte_order"}):
        byte_order = source.get("byte_order", "little")
        if byte_order not in ("little", "big"):
            raise ValueError("byte_order must be little or big")
        prefix = "<" if byte_order == "little" else ">"
        payloads = {}
        values = {}
        for key, n in (("K", 9), ("RT", 16)):
            text = source[key + "_hex"]
            if not isinstance(text, str):
                raise ValueError("byte hex must be text")
            try:
                payload = bytes.fromhex(text)
            except ValueError as exc:
                raise ValueError("invalid byte hex") from exc
            if len(payload) != 4 * n:
                raise ValueError("matrix byte length")
            payloads[key] = payload
            values[key] = list(struct.unpack(prefix + str(n) + "f", payload))
            if not all(math.isfinite(x) for x in values[key]):
                raise ValueError("nonfinite matrix coefficient; retain the input byte record, no arithmetic verdict")
        input_kind = "original_float32_bytes"
    else:
        raise ValueError("use exactly K/RT numeric fields or K_hex/RT_hex with optional byte_order")
    matrices = {}
    for key, n in (("K", 3), ("RT", 4)):
        matrices[key] = [[F.from_float(x) for x in values[key][i*n:(i+1)*n]] for i in range(n)]
    return matrices, payloads, byte_order, input_kind


def transposed(matrix):
    return [list(column) for column in zip(*matrix)]


def multiply(left, right):
    columns = transposed(right)
    return [[sum((a*b for a, b in zip(row, column)), F(0)) for column in columns] for row in left]


def determinant_elimination(matrix):
    """Exact pivoted elimination, independent of the frozen cofactor expansion."""
    work = [row[:] for row in matrix]
    answer = F(1)
    for column in range(len(work)):
        pivot = next((row for row in range(column, len(work)) if work[row][column]), None)
        if pivot is None:
            return F(0)
        if pivot != column:
            work[pivot], work[column] = work[column], work[pivot]
            answer = -answer
        divisor = work[column][column]
        answer *= divisor
        for row in range(column + 1, len(work)):
            factor = work[row][column] / divisor
            for j in range(column, len(work)):
                work[row][j] -= factor * work[column][j]
    return answer


def rotation_statistics(rotation):
    rr = multiply(rotation, transposed(rotation))
    rt = multiply(transposed(rotation), rotation)
    rr_error = [[x - int(i == j) for j, x in enumerate(row)] for i, row in enumerate(rr)]
    rt_error = [[x - int(i == j) for j, x in enumerate(row)] for i, row in enumerate(rt)]
    det = determinant_elimination(rotation)
    rr_max = max(abs(x) for row in rr_error for x in row)
    rt_max = max(abs(x) for row in rt_error for x in row)
    return {"RRt": rr, "RtR": rt, "RRt_minus_I": rr_error, "RtR_minus_I": rt_error,
            "RRt_max": rr_max, "RtR_max": rt_max, "det": det, "det_minus_one_abs": abs(det - 1)}


def exact(value):
    if isinstance(value, F):
        return {"numerator": str(value.numerator), "denominator": str(value.denominator),
                "decimal_display_only": float(value)}
    if isinstance(value, list):
        return [exact(x) for x in value]
    if isinstance(value, dict):
        return {key: exact(item) for key, item in value.items()}
    return value



def power_of_two(exponent):
    return F(2**exponent) if exponent >= 0 else F(1, 2**(-exponent))


def half_grid(value):
    """Exact binary16 membership via a rational lattice, with no half converter."""
    magnitude = abs(value)
    if magnitude == 0:
        return {"representable": True, "region": "zero", "exponent": None, "grid_step": None}
    if magnitude > 65504:
        return {"representable": False, "region": "outside_finite_half_range", "exponent": None, "grid_step": None}
    if magnitude < F(1, 2**14):
        step = F(1, 2**24)
        return {"representable": (magnitude / step).denominator == 1,
                "region": "subnormal_range", "exponent": None, "grid_step": step}
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if magnitude < power_of_two(exponent):
        exponent -= 1
    assert -14 <= exponent <= 15
    assert power_of_two(exponent) <= magnitude < power_of_two(exponent + 1)
    step = power_of_two(exponent - 10)
    return {"representable": (magnitude / step).denominator == 1,
            "region": "normal_range", "exponent": exponent, "grid_step": step}


def half_fingerprint(matrices, payloads, byte_order):
    entries = []
    for name, dimension in (("K", 3), ("RT", 4)):
        for row in range(dimension):
            for column in range(dimension):
                index = dimension * row + column
                raw = payloads[name][4 * index:4 * (index + 1)]
                bits = int.from_bytes(raw, byte_order)
                value = matrices[name][row][column]
                entries.append({"matrix": name, "row": row, "column": column,
                                "input_float32_hex": raw.hex(), "input_float32_bits": f"{bits:08x}",
                                "negative_zero": bits == 0x80000000,
                                "exact_value": value, **half_grid(value)})
    selectors = {
        "K9": lambda item: item["matrix"] == "K",
        "RT16": lambda item: item["matrix"] == "RT",
        "R9": lambda item: item["matrix"] == "RT" and item["row"] < 3 and item["column"] < 3,
        "all25": lambda item: True,
    }
    summaries = {}
    for name, selected in selectors.items():
        group = [item for item in entries if selected(item)]
        count = sum(item["representable"] for item in group)
        summaries[name] = {"count": len(group), "representable": count, "not_representable": len(group) - count,
                           "all_representable": count == len(group),
                           "outside_finite_half_range": sum(item["region"] == "outside_finite_half_range" for item in group),
                           "negative_zero": sum(item["negative_zero"] for item in group)}
    return {
        "arithmetic": "Exact rational binary16 grid membership, independent of a binary16 conversion routine.",
        "rule": "Finite abs<=65504; zero accepted; subnormal x*2^24 integral; normal floorlog2(x)=e and x/2^(e-10) integral.",
        "summaries": summaries,
        "entries": entries,
        "interpretation": "All entries on the half grid are compatible with direct binary16-to-binary32 widening, not proof of that history. A nonrepresentable entry excludes only an all-values direct widening path with no intervening arithmetic. Original input bytes and signed zero are preserved; no replacement matrix or new gate is produced.",
    }


def review(source):
    matrices, payloads, byte_order, input_kind = decode_input(source)
    K, RT = matrices["K"], matrices["RT"]
    rotation = [row[:3] for row in RT[:3]]
    stats = rotation_statistics(rotation)
    transpose_stats = rotation_statistics(transposed(rotation))
    predicates = {key: stats[key] > TAU for key in ("RRt_max", "RtR_max", "det_minus_one_abs")}
    transpose_predicates = {key: transpose_stats[key] > TAU for key in predicates}
    ratios = {}
    exclusions = {}
    for key in predicates:
        bound = DET_BOUND if key == "det_minus_one_abs" else GRAM_BOUND
        ratios[key] = {"over_original_tolerance": stats[key] / TAU, "over_storage_only_bound": stats[key] / bound}
        exclusions[key] = stats[key] > bound
    preconditions = {
        "all_25_coefficients_finite_float32": True,
        "K_bottom_row_exact": K[2] == [0, 0, 1],
        "RT_bottom_row_exact": RT[3] == [0, 0, 0, 1],
        "positive_fx_fy": K[0][0] > 0 and K[1][1] > 0,
    }
    invariant = {
        "RRt_becomes_RtR": transpose_stats["RRt"] == stats["RtR"],
        "RtR_becomes_RRt": transpose_stats["RtR"] == stats["RRt"],
        "determinant_unchanged": transpose_stats["det"] == stats["det"],
        "overall_gate_unchanged": any(transpose_predicates.values()) == any(predicates.values()),
    }
    assert all(invariant.values())
    return exact({
        "reviewer": "independent stdlib saved-float32 matrix verifier",
        "arithmetic": "exact Fraction coefficients; determinant via pivoted elimination",
        "scope": "original predicate definitions under exact arithmetic, not an instruction-level binary64 replay; no projection/depth/GT-support verdict",
        "input_kind": input_kind,
        "byte_order": byte_order,
        "inputs": {key: {"bytes": len(payload), "hex": payload.hex(), "sha256": hashlib.sha256(payload).hexdigest(),
                         "matrix": matrices[key]} for key, payload in payloads.items()},
        "half_grid_fingerprint": half_fingerprint(matrices, payloads, byte_order),
        "matrix_preconditions": preconditions,
        "matrix_preconditions_pass": all(preconditions.values()),
        "statistics": stats,
        "original_f32_tolerance": TAU,
        "strict_greater_than_tolerance": predicates,
        "proper_rotation_gate_fails": any(predicates.values()),
        "transpose_invariant": invariant,
        "ratios": ratios,
        "storage_rounding_test": {
            "unit_roundoff": U, "gram_bound": GRAM_BOUND, "det_bound": DET_BOUND,
            "assumptions": "An exact proper orthogonal Q is rounded once to float32 round-to-nearest; |Qij|<=1 and each entry error<=u.",
            "derivation": "Gram uses 6u+3u^2; determinant multilinearity and column perturbation norm<=3u give 9u+27u^2+27u^3.",
            "strict_exceeds_bound": exclusions,
            "single_final_storage_rounding_hypothesis_excluded": any(exclusions.values()),
            "interpretation": "Exceeding a bound excludes only this narrow single-final-rounding hypothesis. Staying within does not establish rounding as the cause. No upstream-chain, export, reader, scale, shear, or reflection root cause is identified.",
        },
    })


def fixture(rotation):
    return {"K": [[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]],
            "RT": [list(row) + [0.] for row in rotation] + [[0., 0., 0., 1.]]}


def self_test():
    def rational(record):
        return F(int(record["numerator"]), int(record["denominator"]))
    identity = fixture([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    report = review(identity)
    assert not report["proper_rotation_gate_fails"]
    assert rational(report["statistics"]["det"]) == 1
    c = struct.unpack("<f", struct.pack("<f", math.sqrt(.5)))[0]
    rounded = review(fixture([[c,-c,0.],[c,c,0.],[0.,0.,1.]]))
    assert not rounded["proper_rotation_gate_fails"]
    assert rational(rounded["statistics"]["RRt_max"]) > F(1,10**8)
    assert not rounded["storage_rounding_test"]["single_final_storage_rounding_hypothesis_excluded"]
    reflection = review(fixture([[1.,0.,0.],[0.,1.,0.],[0.,0.,-1.]]))
    assert reflection["strict_greater_than_tolerance"] == {"RRt_max": False, "RtR_max": False, "det_minus_one_abs": True}
    assert rational(reflection["statistics"]["det"]) == -1
    uniform = review(fixture([[1.25,0.,0.],[0.,1.25,0.],[0.,0.,1.25]]))
    assert rational(uniform["statistics"]["det"]) == F(125,64)
    anisotropic = review(fixture([[1.25,0.,0.],[0.,.75,0.],[0.,0.,1.]]))
    assert rational(anisotropic["statistics"]["det"]) == F(15,16)
    shear = review(fixture([[1.,.125,0.],[0.,1.,0.],[0.,0.,1.]]))
    assert rational(shear["statistics"]["RRt_max"]) == F(1,8)
    assert rational(shear["statistics"]["det"]) == 1
    singular = review(fixture([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]))
    assert rational(singular["statistics"]["det"]) == 0
    a = [[F(x) for x in row] for row in [[1,2,3],[0,4,5],[1,0,6]]]
    assert determinant_elimination(a) == 22
    assert determinant_elimination([a[1],a[0],a[2]]) == -22
    assert GRAM_BOUND < TAU and DET_BOUND < TAU
    for byte_order, code in (("little","<"), ("big",">")):
        source = {"K_hex": struct.pack(code+"9f", *flatten(identity["K"],3)).hex(),
                  "RT_hex": struct.pack(code+"16f", *flatten(identity["RT"],4)).hex(), "byte_order": byte_order}
        assert review(source)["statistics"] == report["statistics"]
    failures = 0
    bad = fixture([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    for value in (True, .1, float("nan"), float("inf"), 2**100 + 1):
        bad["K"][0][0] = value
        try:
            review(bad)
        except ValueError:
            failures += 1
    assert failures == 5
    # A deliberately separate conversion oracle is used only for artificial tests.
    def half_converter_accepts(number):
        try:
            return struct.unpack("<e", struct.pack("<e", number))[0] == number
        except (OverflowError, struct.error):
            return False
    boundaries = [
        (0.0, True), (-0.0, True), (1.0, True), (-1.0, True),
        (2.0**-25, False), (2.0**-24, True), (3 * 2.0**-25, False),
        (1023 * 2.0**-24, True), (2.0**-14 - 2.0**-25, False),
        (2.0**-14, True), (2.0**-14 + 2.0**-25, False),
        (1.0 + 2.0**-11, False), (1.0 + 2.0**-10, True),
        (65504.0, True), (65505.0, False), (65520.0, False),
        (65536.0, False), (c, False),
    ]
    for number, expected in boundaries:
        assert half_grid(F.from_float(number))["representable"] == expected
        assert half_converter_accepts(number) == expected
        assert half_grid(F.from_float(-number))["representable"] == expected
    exhaustive_finite = 0
    for bits in range(2**16):
        number = struct.unpack("<e", bits.to_bytes(2, "little"))[0]
        if math.isfinite(number):
            assert half_grid(F.from_float(number))["representable"]
            assert half_converter_accepts(number)
            exhaustive_finite += 1
    assert exhaustive_finite == 63488
    for byte_order, code in (("little", "<"), ("big", ">")):
        signed = fixture([[1., -0., 0.], [0., 1., 0.], [0., 0., 1.]])
        signed["K"][0][1] = -0.0
        source = {"K_hex": struct.pack(code+"9f", *flatten(signed["K"], 3)).hex(),
                  "RT_hex": struct.pack(code+"16f", *flatten(signed["RT"], 4)).hex(), "byte_order": byte_order}
        fingerprint = review(source)["half_grid_fingerprint"]
        summaries = fingerprint["summaries"]
        assert {name: item["representable"] for name, item in summaries.items()} == {"K9": 9, "RT16": 16, "R9": 9, "all25": 25}
        assert {name: item["negative_zero"] for name, item in summaries.items()} == {"K9": 1, "RT16": 1, "R9": 1, "all25": 2}
        assert sum(item["input_float32_bits"] == "80000000" for item in fingerprint["entries"]) == 2
    rounded_counts = rounded["half_grid_fingerprint"]["summaries"]
    assert rounded_counts["R9"]["representable"] == 5
    assert rounded_counts["RT16"]["representable"] == 12
    assert rounded_counts["K9"]["representable"] == 9
    return {"status": "passed", "scope": "artificial fixtures only", "matrix_cases": 7,
            "half_grid_boundary_cases": len(boundaries), "half_grid_exhaustive_finite_patterns": exhaustive_finite,
            "additional_checks": ["pivot swap determinant", "big/little endian byte equivalence", "transpose invariants in every case",
                                  "storage bounds below original tolerance", "reject bool/non-f32/nonfinite numeric inputs", "half grid rational versus artificial struct-e oracle",
                                  "signed-zero bits retained in both endian formats", "separate R9/K9/RT16 counts"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="saved JSON matrix record; otherwise stdin")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.input:
            parser.error("self-test does not accept input data")
        result = self_test()
    else:
        raw = args.input.read_text() if args.input else sys.stdin.read()
        if len(raw) > 65536:
            raise ValueError("matrix input JSON exceeds 64 KiB")
        result = review(json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
