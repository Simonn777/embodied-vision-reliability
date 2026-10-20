"""Exact rational verification of the single predeclared analytical witness."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path


def tr(a):
    return [list(v) for v in zip(*a)]


def mm(a, b):
    return [[sum(x*y for x, y in zip(row, col)) for col in tr(b)] for row in a]


def add(a, b):
    return [[x+y for x, y in zip(u, v)] for u, v in zip(a, b)]


def scale(a, s):
    return [[s*x for x in row] for row in a]


def gram(a, b=None):
    return mm(tr(a), a if b is None else b)


def vec(a):
    return [a[0][0], a[0][1], a[1][1]]


def det(a):
    return sum(a[0][j]*(a[1][(j+1)%3]*a[2][(j+2)%3]
        - a[1][(j+2)%3]*a[2][(j+1)%3]) for j in range(3))


def solve(a, b):
    z=[list(row)+[v] for row, v in zip(a, b)]
    for k in range(len(a)):
        pivot=next(i for i in range(k,len(a)) if z[i][k])
        z[k],z[pivot]=z[pivot],z[k]
        s=z[k][k];z[k]=[v/s for v in z[k]]
        for i in range(len(a)):
            if i!=k:
                s=z[i][k];z[i]=[x-s*y for x,y in zip(z[i],z[k])]
    return [row[-1] for row in z]


def convolution(a, b):
    c=[F(0)]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):c[i+j]+=x*y
    return c


def main():
    z=F(0);o=F(1)
    R=scale([[2,-2,1],[2,1,-2],[1,2,2]], F(1,3))
    B=[[o,z],[z,o],[z,z]];r1=[[z],[z],[o]];r2=[[F(1,2)],[z],[o]]
    A=[[F(1,2),-o],[z,F(1,3)]];g1=[[o,z]];g2=[[F(2),F(8)]]
    U0=add(B,mm(r1,g1));U1=B
    V1=mm(B,A);V0=mm(add(B,mm(r2,g2)),A)
    t=[[F(1,6)],[F(2,3)],[F(1,3)]]
    assert gram(R)==[[o,z,z],[z,o,z],[z,z,o]] and det(R)==o
    assert add(mm(R,r1),t)==r2 and mm(R,U0)==V0
    M=tr([vec(gram(V0)),vec(add(gram(V0,V1),gram(V1,V0))),vec(gram(V1))])
    assert M==[[F(2),o,F(1,4)],[z,F(-4,3),F(-1,2)],[o,F(14,9),F(10,9)]]
    assert det(M)==F(-85,54)
    Q=tr([vec(gram(U0)),vec(add(gram(U0,U1),gram(U1,U0))),vec(gram(U1))])
    Y=tr([solve(M,c) for c in tr(Q)])
    expected=[[o,F(94,85),F(47,85)],[z,F(-54,85),F(-27,85)],[z,F(144,85),F(72,85)]]
    assert Y==expected
    poly=[a-b for a,b in zip(convolution(Y[0],Y[2]),convolution(Y[1],Y[1]))]
    factored=convolution([z,F(2),o],[F(6120),F(5310),F(2655)])
    assert poly==[c/F(7225) for c in factored]
    jac=tr([vec(add(gram(U0,U1),gram(U1,U0))),
            vec(scale(gram(V0),F(-2))),
            vec(scale(add(gram(V0,V1),gram(V1,V0)),F(-1)))])
    assert jac==[[F(2),F(-4),F(-1)],[z,z,F(4,3)],[F(2),F(-2),F(-14,9)]]
    assert det(jac)==F(-16,3)
    print(json.dumps({'status':'passed','arithmetic':'fractions.Fraction exact rational',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'M_determinant':str(det(M)),'calibration_jacobian_determinant':str(det(jac)),
        'quartic_ascending_coefficients':[str(v) for v in poly],
        'positive_depth_witness':{'b':0,'a':1,'c':0},
        'random_search':False,'benchmark_access':False,'accuracy_claim':False},indent=2))


if __name__=='__main__':
    main()
