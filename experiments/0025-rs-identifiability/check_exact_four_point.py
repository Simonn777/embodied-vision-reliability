#!/usr/bin/env python3
"""One newly declared exact-rational four-point witness; stdlib only.

Prepared after the initial numerical qualification. It does not reproduce,
replace, or change that initial experiment. No random construction, search,
parameter fitting, numerical rank threshold, dataset or model is used. If the
fixed determinant is zero, retain the failure and do not try another fixture.
"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time


SCHEMA = "evr0025.exact-rational-four-point.v1"
ORIGINAL_NUMERIC_SOURCE_SHA256 = "0b058405fbc151b8765dafdb4e4a81f8c7e48f515da3d217b2df33b9a9565683"
ORIGINAL_NUMERIC_RESULT_SHA256 = "befcf2f368ad964324bee7f72c2179c5487212c9899c1cd39d276fceccf2f7c5"
DECLARATION = {
    "points": [["-1.8", "-1.2", "3.2"], ["1.5", "-1.1", "4.5"],
               ["-1.3", ".9", "2.8"], ["1.9", "1.2", "5.2"]],
    "cayley_vector": [".1", "-.065", ".055"],
    "cayley_convention": "R=(I+[c]x)(I-[c]x)^(-1)",
    "translation": [".42", "-.16", ".25"],
    "a1": "1", "b1": ".18", "a2": "1.17", "b2": "-.09",
    "K": [["620", "0", "639.5"], ["0", "600", "479.5"], ["0", "0", "1"]],
    "width": 1280, "height": 960, "readout_seconds": ".04",
    "true_twist1": ["0"]*6, "true_twist2": ["0"]*6,
    "unknown_order": ["t_x", "t_y", "t_z", "v1_x", "v1_y", "v1_z",
                      "v2_x", "v2_y", "v2_z", "b1", "a2", "b2"],
    "equation": "[-I, s1*R, -s2*I, -R*r1, d2*r2, r2] beta = d1*R*r1",
    "observation_contract": "Known reference R and zero angular rates; row times are projected pixel rows. Linear row velocities remain unknown columns, although their generating truth is zero.",
    "construction_history": "New exact-rational extension proposed after the saved initial numerical result, using a fixed rational Cayley rotation instead of its exponential rotation. The first four original points and other stated constants are retained.",
}


def identity(n):
    return [[F(int(i == j)) for j in range(n)] for i in range(n)]


def transpose(A):
    return [list(row) for row in zip(*A)]


def matmul(A, B):
    if len(A[0]) != len(B):
        raise ValueError("matrix shape mismatch")
    return [[sum((a*b for a, b in zip(row, col)), F(0))
             for col in zip(*B)] for row in A]


def matvec(A, x):
    return [sum((a*b for a, b in zip(row, x)), F(0)) for row in A]


def add(A, B, sign=1):
    return [[a+sign*b for a,b in zip(ra,rb)] for ra,rb in zip(A,B)]


def inverse(A):
    n = len(A)
    if any(len(row) != n for row in A):
        raise ValueError("inverse requires square matrix")
    B = [list(row)+unit for row,unit in zip(A, identity(n))]
    for k in range(n):
        pivot = next((i for i in range(k,n) if B[i][k] != 0), None)
        if pivot is None:
            raise ValueError("singular fixed matrix")
        B[k], B[pivot] = B[pivot], B[k]
        q = B[k][k]
        B[k] = [v/q for v in B[k]]
        for i in range(n):
            if i != k:
                q = B[i][k]
                B[i] = [a-q*b for a,b in zip(B[i],B[k])]
    if [row[:n] for row in B] != identity(n):
        raise AssertionError("exact inverse reduction failed")
    return [row[n:] for row in B]


def determinant_and_rank(A):
    n = len(A)
    if any(len(row) != n for row in A):
        raise ValueError("determinant requires square matrix")
    B = [list(row) for row in A]
    pivot_row, parity, pivots = 0, 1, []
    for column in range(n):
        pivot = next((i for i in range(pivot_row,n) if B[i][column] != 0), None)
        if pivot is None:
            continue
        if pivot != pivot_row:
            B[pivot_row], B[pivot] = B[pivot], B[pivot_row]
            parity = -parity
        value = B[pivot_row][column]
        pivots.append(value)
        for i in range(pivot_row+1,n):
            q = B[i][column]/value
            B[i] = [a-q*b for a,b in zip(B[i],B[pivot_row])]
        pivot_row += 1
    determinant = F(parity)
    if pivot_row < n:
        determinant = F(0)
    else:
        for value in pivots:
            determinant *= value
    return determinant, pivot_row, pivots


def rational(value):
    if isinstance(value,F):
        return str(value.numerator)+"/"+str(value.denominator)
    if isinstance(value,dict):
        return {key:rational(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):
        return [rational(item) for item in value]
    return value


def write(path, value):
    path.write_text(json.dumps(rational(value),indent=2,sort_keys=True,allow_nan=False)+"\n")


def build_fixed_system():
    d = DECLARATION
    points = [[F(v) for v in row] for row in d["points"]]
    cx,cy,cz = map(F,d["cayley_vector"])
    C = [[F(0),-cz,cy],[cz,F(0),-cx],[-cy,cx,F(0)]]
    I = identity(3)
    R = matmul(add(I,C),inverse(add(I,C,sign=-1)))
    rotation_determinant,rotation_rank,_ = determinant_and_rank(R)
    if matmul(transpose(R),R) != I or rotation_determinant != 1 or rotation_rank != 3:
        raise AssertionError("fixed Cayley matrix is not exact SO(3)")
    t = list(map(F,d["translation"]))
    K = [[F(v) for v in row] for row in d["K"]]
    b1,a2,b2 = F(d["b1"]),F(d["a2"]),F(d["b2"])
    readout,width,height = F(d["readout_seconds"]),d["width"],d["height"]
    if F(d["a1"]) != 1 or a2 <= 0 or readout <= 0:
        raise AssertionError("invalid predeclared scale/readout")
    observations,A,y = [],[],[]
    for index,X in enumerate(points):
        X2 = [a+b for a,b in zip(matvec(R,X),t)]
        z1,z2 = X[2],X2[2]
        if z1 <= 0 or z2 <= 0:
            raise ValueError("fixed manufactured point has nonpositive axial depth")
        r1,r2 = [v/z1 for v in X],[v/z2 for v in X2]
        p1,p2 = matvec(K,r1),matvec(K,r2)
        for pixel in (p1,p2):
            if pixel[2] != 1 or not (0 <= pixel[0] <= width-1 and 0 <= pixel[1] <= height-1):
                raise ValueError("fixed manufactured point outside sensor")
        s1 = readout*(p1[1]-F(height-1,2))/F(height-1)
        s2 = readout*(p2[1]-F(height-1,2))/F(height-1)
        if not (-readout/2 <= s1 <= readout/2 and -readout/2 <= s2 <= readout/2):
            raise AssertionError("projected row time outside readout")
        d1,d2 = z1-b1,(z2-b2)/a2
        Rr1 = matvec(R,r1)
        for k in range(3):
            A.append([-v for v in I[k]]+[s1*v for v in R[k]]+[-s2*v for v in I[k]]
                     +[-Rr1[k],d2*r2[k],r2[k]])
            y.append(d1*Rr1[k])
        observations.append(dict(index=index,world_point=X,second_middle_point=X2,
            ray1=r1,ray2=r2,pixel1=p1[:2],pixel2=p2[:2],time1=s1,time2=s2,
            corrected_depth1=z1,corrected_depth2=z2,predicted_depth1=d1,predicted_depth2=d2))
    beta = t+[F(0)]*6+[b1,a2,b2]
    residual = [a-b for a,b in zip(matvec(A,beta),y)]
    if any(residual):
        raise AssertionError("fixed exact physical truth does not satisfy linear equations")
    return dict(R=R,rotation_determinant=rotation_determinant,
        rotation_orthogonality_exact=True,observations=observations,matrix=A,rhs=y,
        truth_solution=beta,truth_residual=residual,corrected_depths_positive=True,
        all_pixels_inside_sensor=True,unknown_count=12,equation_count=12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter()
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    receipt = dict(schema=SCHEMA,status="started",source_sha256=source_hash,
        python=sys.version,started_utc=datetime.now(timezone.utc).isoformat(),
        exact_arithmetic="fractions.Fraction; all geometric inputs constructed from decimal strings",
        declaration=DECLARATION,original_numeric_source_sha256=ORIGINAL_NUMERIC_SOURCE_SHA256,
        original_numeric_result_sha256=ORIGINAL_NUMERIC_RESULT_SHA256,
        original_artifacts_read_or_modified=False,real_data_or_model_access=False,
        fixture_search=False)
    write(args.output_dir/"receipt.json",receipt)
    result_path = args.output_dir/"mathematics.json"
    try:
        result = dict(schema=SCHEMA,source_sha256=source_hash,**build_fixed_system())
        determinant,rank,pivots = determinant_and_rank(result["matrix"])
        result.update(determinant=determinant,rank=rank,elimination_pivots=pivots,
            determinant_nonzero=determinant != 0,
            interpretation="A fixed physical existence witness for the complete gauge-fixed 12-parameter linear system only. Not a real-data result, noise guarantee, global genericity theorem, or proof that translation direction alone needs four points.")
        # Preserve the complete fixed matrix and exact determinant even on a zero result.
        write(result_path,result)
        if determinant == 0 or rank != 12:
            raise ValueError("fixed exact four-point determinant is zero; no fixture search permitted")
        solution = matvec(inverse(result["matrix"]),result["rhs"])
        if solution != result["truth_solution"]:
            raise AssertionError("exact solved vector differs from declared physical truth")
        result.update(solved_solution=solution,solution_equals_truth_exactly=True)
        write(result_path,result)
        receipt.update(status="completed",rank=rank,determinant=determinant,
                       exact_unique_full_parameter_solution=True)
    except BaseException as exc:
        receipt.update(status="failed",error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        if result_path.exists():
            receipt["result_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
        receipt["wall_seconds"] = time.perf_counter()-started
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write(args.output_dir/"receipt.json",receipt)
    print(json.dumps(rational({k:receipt[k] for k in ("status","rank","determinant","wall_seconds")})))


if __name__ == "__main__":
    main()
