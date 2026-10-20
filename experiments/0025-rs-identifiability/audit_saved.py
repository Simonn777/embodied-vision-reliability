#!/usr/bin/env python3
"""Read saved manufactured matrices only; never import or rerun their generator."""
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
SOURCE_HASH = "0b058405fbc151b8765dafdb4e4a81f8c7e48f515da3d217b2df33b9a9565683"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def close(a, b, atol=1e-12, rtol=1e-12):
    aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert aa.shape == bb.shape
    assert np.all(np.isfinite(aa)) and np.all(np.isfinite(bb))
    assert np.allclose(aa, bb, atol=atol, rtol=rtol), float(np.max(np.abs(aa-bb)))
    return float(np.max(np.abs(aa-bb))) if aa.size else 0.


def cross(w):
    return np.array([[0., -w[2], w[1]], [w[2], 0., -w[0]], [-w[1], w[0], 0.]])


def q_and_integral(w, s):
    # Direct closed form, independent of the generator's small-angle branch.
    a = np.asarray(w, dtype=float) * s
    theta = np.linalg.norm(a)
    if theta == 0.:
        return np.eye(3), s*np.eye(3)
    W = cross(a)
    Q = np.eye(3) + np.sin(theta)/theta*W + (1.-np.cos(theta))/theta**2*(W@W)
    V = np.eye(3) + (1.-np.cos(theta))/theta**2*W + (theta-np.sin(theta))/theta**3*(W@W)
    return Q, s*(Q.T@V)


def check_record(matrix, record):
    spectrum = np.linalg.svd(matrix, compute_uv=False)
    difference = close(spectrum, record["singular_values"])
    assert record["shape"] == list(matrix.shape)
    rank = int(np.count_nonzero(spectrum > record["threshold"]))
    assert rank == record["rank"]
    close(spectrum[0], record["maximum"])
    close(spectrum[-1], record["minimum"])
    return dict(rank=rank, threshold=record["threshold"],
                minimum=float(spectrum[-1]), maximum=float(spectrum[0]),
                condition=float(spectrum[0]/spectrum[-1]), spectrum_max_difference=difference)


def main():
    folder = BASE/"manufactured-01"
    receipt, values = read(folder/"receipt.json"), read(folder/"mathematics.json")
    execution = read(BASE/"execution-01/receipt.json")
    assert digest(BASE/"check_manufactured.py") == receipt["source_sha256"] == execution["source_sha256"] == SOURCE_HASH
    assert digest(folder/"mathematics.json") == receipt["result_sha256"]
    assert receipt["status"] == execution["status"] == "completed"
    assert execution["exit_code"] == 0 and execution["termination_reason"] is None
    assert execution["no_real_data_or_model_inputs"] and receipt["no_real_inputs"]
    for name, entry in execution["files"].items():
        path = BASE/"execution-01"/name
        assert path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"]
    assert execution["files"]["stderr.log"]["bytes"] == 0
    stdout = read(BASE/"execution-01/stdout.log")
    for key in ("status", "wall_seconds", "full_unknown_ranks", "known_R_omega_ranks"):
        assert stdout[key] == receipt[key]

    full = values["full_unknown_checks"]
    assert [(x["kind"], x["count"]) for x in full] == [
        ("generic", 7), ("generic", 8), ("zero_motion", 8),
        ("zero_readout", 8), ("vertical", 8), ("equal_row", 8)]
    scales = np.asarray(receipt["parameter_column_scales"])
    close(scales, [1.]*6+[25.]*12+[1.]*3)
    full_rows = []
    observations = {}
    for row in full:
        J = np.asarray(row["jacobian_scaled"])
        physical = J/scales
        close(physical, row["jacobian_physical"])
        close(np.linalg.svd(physical, compute_uv=False), row["physical_spectrum"])
        maximum = np.linalg.svd(J, compute_uv=False)[0]
        floor = 100.*max(row["richardson_spectral_disagreements"] + [np.finfo(float).eps*max(1., maximum)])
        threshold = max(1e-8*maximum, floor)
        close(floor, row["finite_difference_floor"])
        close(threshold, row["rank_record"]["threshold"])
        summary = check_record(J, row["rank_record"])
        for r in row["richardson"]:
            close(r["threshold"], threshold)
            assert int(np.count_nonzero(np.asarray(r["singular_values"]) > threshold)) == r["rank"]
        assert row["full_rank_witness"] == all(r["rank"] == 21 for r in row["richardson"])
        if "analytic_check" in row:
            A = np.asarray(row["analytic_check"]["jacobian_scaled"])
            check_record(A, row["analytic_check"]["rank_record"])
            close(np.linalg.norm(A-J, 2), row["analytic_check"]["spectral_difference"])
            assert np.linalg.norm(A-J, 2) <= row["analytic_check"]["tolerance"]
            for key in ("known_offset_null", "known_common_twist_nulls"):
                if key+"_physical" in row:
                    calculated = (A/scales)@np.asarray(row[key+"_physical"])
                    residual_key = "known_common_twist_null_residual" if key.endswith("nulls") else key+"_residual"
                    close(calculated, row[residual_key])
                    assert np.max(np.abs(calculated)) < 1e-10
        summary.update(kind=row["kind"], count=row["count"], witness=row["full_rank_witness"])
        full_rows.append(summary)
        observations[(row["kind"], row["count"])] = row["observations"]
    assert [{k:r[k] for k in ("count", "kind", "rank", "witness")} for r in full_rows] == receipt["full_unknown_ranks"]

    observations[("identity_rotation", 8)] = values["additional_identity_rotation_fixture"]
    linear = values["known_R_omega_linear_checks"]
    assert [(x["kind"], x["count"]) for x in linear] == [(k,n) for k in ("generic", "zero_motion", "vertical", "identity_rotation") for n in (4,5)]
    linear_rows = []
    for row in linear:
        obs = observations[(row["kind"], 8)]
        q, n = obs["truth"], row["count"]
        R = np.asarray(q["R"])
        blocks, rhs = [], []
        for i in range(n):
            Q1, B1 = q_and_integral(q["xi1"][:3], obs["s1"][i])
            Q2, B2 = q_and_integral(q["xi2"][:3], obs["s2"][i])
            r1, r2 = Q1.T@np.asarray(obs["r1"][i]), Q2.T@np.asarray(obs["r2"][i])
            blocks.append(np.column_stack((-np.eye(3), R@B1, -B2, -R@r1, obs["d2"][i]*r2, r2)))
            rhs.extend(R@(obs["d1"][i]*r1))
        A, rhs = np.vstack(blocks), np.asarray(rhs)
        matrix_difference = close(A, row["matrix_physical"])
        close(rhs, row["rhs"])
        beta = np.r_[q["t"], q["xi1"][3:], q["xi2"][3:], q["b1"], q["a2"], q["b2"]]
        close(beta, row["truth_beta"])
        truth_error = float(np.max(np.abs(A@beta-rhs)))
        assert truth_error <= 1e-11
        close(truth_error, row["truth_max_abs_residual"])
        scaling = np.asarray(row["column_scales"])
        close(scaling, [1.]*3+[25.]*6+[1.]*3)
        record = check_record(A*scaling, row["rank_record"])
        close(record["threshold"], 1e-8*record["maximum"])
        close(np.linalg.svd(A, compute_uv=False), row["physical_spectrum"])
        assert row["full_rank_witness"] == (record["rank"] == 12)
        _, _, Vh = np.linalg.svd(A*scaling, full_matrices=True)
        null = Vh[record["rank"]:].T
        translation = np.asarray(q["t"])
        unit = translation/np.linalg.norm(translation)
        direction_derivative = (np.eye(3)-np.outer(unit,unit))/np.linalg.norm(translation)
        functional = np.zeros((3,12))
        functional[:,:3] = direction_derivative
        null_sensitivity = functional@null
        record["translation_direction_null_analysis"] = dict(
            null_dimension=int(null.shape[1]),
            null_basis_physical=(scaling[:,None]*null).tolist(),
            matrix_null_max_abs=float(np.max(np.abs((A*scaling)@null))) if null.size else 0.,
            projected_direction_derivative=null_sensitivity.tolist(),
            relative_functional_null_norm=float(np.linalg.norm(null_sensitivity,2)/np.linalg.norm(functional,2)) if null.size else 0.,
            interpretation="Continuous sensitivity using the same saved-matrix rank threshold; no new fixture or rank threshold.")
        if "known_offset_null_physical" in row:
            null_error = A@np.asarray(row["known_offset_null_physical"])
            close(null_error, row["known_offset_null_residual"])
            assert np.max(np.abs(null_error)) < 1e-10
        if "translation_elimination" in row:
            e = row["translation_elimination"]
            L = np.column_stack((np.ones(n), obs["s1"][:n], obs["s2"][:n]))
            M = np.asarray(e["left_null_basis"])
            close(L, e["row_time_design"])
            close(M.T@L, np.zeros((n-3, 3)))
            close(M.T@M, np.eye(n-3))
            P = np.einsum("ik,icj->kcj", M, A.reshape(n, 3, 12)).reshape(-1, 12)
            close(P, e["projected_matrix"])
            offsets = P[:, [9,11]].reshape(n-3,3,2)
            assert np.max(np.abs(offsets[:,1:,:])) < 1e-12
            if n == 4:
                assert record["rank"] <= 11 and e["analytic_four_point_rank_upper_bound"] == 11
        record.update(kind=row["kind"], count=n, witness=row["full_rank_witness"],
                      truth_max_abs_residual=truth_error, independent_matrix_max_difference=matrix_difference)
        linear_rows.append(record)
    assert [{k:r[k] for k in ("count", "kind", "rank", "witness")} for r in linear_rows] == receipt["known_R_omega_ranks"]
    zero = observations[("zero_motion", 8)]["truth"]
    assert np.linalg.norm(zero["xi1"]) == np.linalg.norm(zero["xi2"]) == 0.
    assert np.linalg.norm(np.asarray(zero["R"])-np.eye(3)) > 0. and np.linalg.norm(zero["t"]) > 0.
    result = dict(schema="evr0025.independent-saved-audit.v1", status="passed",
        source_sha256=SOURCE_HASH, audited_result_sha256=digest(folder/"mathematics.json"),
        result_receipt_sha256=digest(folder/"receipt.json"), execution_receipt_sha256=digest(BASE/"execution-01/receipt.json"),
        auditor_sha256=digest(Path(__file__)), full_unknown_checks=full_rows,
        known_R_omega_linear_checks=linear_rows,
        zero_motion_display_name="zero intra-readout twists, nonzero inter-frame pose",
        zero_motion_interframe_translation_norm=float(np.linalg.norm(zero["t"])),
        limitations=["Saved-array audit only; no observation regeneration, derivative rerun, solver or model execution.",
                     "Three Richardson rank counts checked from saved spectra; only the final saved Jacobian spectrum can be independently recomputed.",
                     "Cross-step differences are not rigorous derivative error bounds; full rank is numerical local evidence, not a generic/global theorem or accuracy result."])
    output = BASE/"audit-01"
    output.mkdir(exist_ok=False)
    (output/"audit.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({"status":"passed", "full_ranks":[x["rank"] for x in full_rows],
                      "known_rotation_ranks":[x["rank"] for x in linear_rows],
                      "max_linear_truth_residual":max(x["truth_max_abs_residual"] for x in linear_rows)}))


if __name__ == "__main__":
    main()
