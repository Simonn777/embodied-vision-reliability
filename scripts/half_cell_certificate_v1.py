#!/usr/bin/env python3
"""One fixed saved-R-only mathematical compatibility proposal; stdlib only.

Input: {"R_hex": row-major 9 float32 bytes as hex, "byte_order": "little"|"big"}.
No dataset/candidate imports, optimization, search, projection, camera replacement,
or threshold change. Exact SO(3) plus exact half rounding-cell membership is the
only positive certificate. Failure is INCONCLUSIVE, never infeasibility evidence.
Actual-data input/output retain the source dataset terms; not automatically MIT.
"""
import argparse
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
import sys


def power2(exponent):
    return F(2**exponent) if exponent >= 0 else F(1, 2**(-exponent))


def positive_half_value(bits):
    """Decode finite nonnegative binary16 integer encoding by rational arithmetic."""
    if not 0 <= bits <= 0x7bff:
        raise ValueError('finite positive half code required')
    exponent, fraction = bits >> 10, bits & 1023
    if exponent == 0:
        return fraction * power2(-24)
    return (1024 + fraction) * power2(exponent - 25)


def positive_half_bits(value):
    """Exact half-grid index; no half conversion or nearest-value rounding."""
    if value < 0 or value > 65504:
        return None
    if value == 0:
        return 0
    if value < power2(-14):
        units = value / power2(-24)
        return units.numerator if units.denominator == 1 else None
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < power2(exponent):
        exponent -= 1
    units = value / power2(exponent - 10)
    if units.denominator != 1:
        return None
    bits = ((exponent + 15) << 10) + units.numerator - 1024
    assert positive_half_value(bits) == value
    return bits


def half_cell(value, negative_zero=False):
    """Exact nearest-even cell. Exact real zero canonically rounds to positive zero.

    Thus a negative-zero target requires a strictly negative witness coefficient;
    failing this conservative convention is merely INCONCLUSIVE.
    """
    if negative_zero and value != 0:
        raise ValueError('negative-zero flag only valid for zero')
    bits = positive_half_bits(abs(value))
    if bits is None:
        return None
    if value == 0:
        if negative_zero:
            return {'lower': -power2(-25), 'upper': F(0),
                    'lower_closed': True, 'upper_closed': False,
                    'half_bits': '8000'}
        return {'lower': F(0), 'upper': power2(-25),
                'lower_closed': True, 'upper_closed': True,
                'half_bits': '0000'}
    middle = positive_half_value(bits)
    previous = positive_half_value(bits - 1)
    # The virtual next binade value yields the finite-overflow midpoint 65520.
    following = positive_half_value(bits + 1) if bits < 0x7bff else F(65536)
    lower, upper = (previous + middle) / 2, (middle + following) / 2
    closed = bits % 2 == 0
    if value < 0:
        lower, upper = -upper, -lower
    return {'lower': lower, 'upper': upper, 'lower_closed': closed,
            'upper_closed': closed, 'half_bits': f'{bits | (0x8000 if value < 0 else 0):04x}'}


def inside(cell, value):
    if cell is None:
        return False
    lower_ok = value > cell['lower'] or (cell['lower_closed'] and value == cell['lower'])
    upper_ok = value < cell['upper'] or (cell['upper_closed'] and value == cell['upper'])
    return lower_ok and upper_ok


def transpose(matrix):
    return [list(column) for column in zip(*matrix)]


def multiply(left, right):
    return [[sum((a*b for a,b in zip(row,column)), F(0))
             for column in transpose(right)] for row in left]


def determinant(matrix):
    total = F(0)
    for perm in itertools.permutations(range(3)):
        inversions = sum(perm[i] > perm[j] for i in range(3) for j in range(i+1,3))
        term = F(1)
        for row,column in enumerate(perm):
            term *= matrix[row][column]
        total += (-1 if inversions % 2 else 1) * term
    return total


def fixed_proposal(rotation):
    """Largest standard trace/diagonal branch; w,x,y,z tie order; exactly one q."""
    r = rotation
    a = [1+r[0][0]+r[1][1]+r[2][2],
         1+r[0][0]-r[1][1]-r[2][2],
         1-r[0][0]+r[1][1]-r[2][2],
         1-r[0][0]-r[1][1]+r[2][2]]
    assert sum(a) == 4
    branch = max(range(4), key=lambda index: a[index])
    assert a[branch] >= 1
    wx, wy, wz = r[2][1]-r[1][2], r[0][2]-r[2][0], r[1][0]-r[0][1]
    xy, xz, yz = r[0][1]+r[1][0], r[0][2]+r[2][0], r[1][2]+r[2][1]
    candidates = [(a[0],wx,wy,wz), (wx,a[1],xy,xz),
                  (wy,xy,a[2],yz), (wz,xz,yz,a[3])]
    q = list(candidates[branch])
    w,x,y,z = q
    norm2 = sum((v*v for v in q),F(0))
    assert norm2 > 0
    numerator = [[w*w+x*x-y*y-z*z, 2*(x*y-w*z), 2*(x*z+w*y)],
                 [2*(x*y+w*z), w*w-x*x+y*y-z*z, 2*(y*z-w*x)],
                 [2*(x*z-w*y), 2*(y*z+w*x), w*w-x*x-y*y+z*z]]
    proposal = [[entry/norm2 for entry in row] for row in numerator]
    return {'branch': 'wxyz'[branch], 'branch_quantities': a,
            'homogeneous_quaternion': q, 'quaternion_squared_norm': norm2,
            'hypothetical_rotation': proposal}


def decode(source):
    if type(source) is not dict or set(source) not in ({'R_hex'}, {'R_hex','byte_order'}):
        raise ValueError('use exactly R_hex and optional byte_order')
    order = source.get('byte_order','little')
    if order not in ('little','big') or not isinstance(source['R_hex'],str):
        raise ValueError('invalid byte order or hex type')
    raw = bytes.fromhex(source['R_hex'])
    if len(raw) != 36:
        raise ValueError('nine original float32 values required')
    values = struct.unpack(('<' if order == 'little' else '>')+'9f',raw)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('finite original float32 values required')
    rational = [F.from_float(v) for v in values]
    rotation = [rational[i*3:(i+1)*3] for i in range(3)]
    negative_zero = [int.from_bytes(raw[4*i:4*i+4],order) == 0x80000000 for i in range(9)]
    return raw, order, rotation, negative_zero


def serial(value):
    if isinstance(value,F):
        return {'numerator':str(value.numerator),'denominator':str(value.denominator)}
    if isinstance(value,dict):
        return {key:serial(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):
        return [serial(item) for item in value]
    return value


def certify(source):
    raw, order, rotation, negative_zero = decode(source)
    proposal = fixed_proposal(rotation)
    q = proposal['hypothetical_rotation']
    identity = [[F(int(i==j)) for j in range(3)] for i in range(3)]
    rr,rt,det = multiply(q,transpose(q)),multiply(transpose(q),q),determinant(q)
    exact_so3 = rr == identity and rt == identity and det == 1
    assert exact_so3
    entries = []
    for i in range(3):
        for j in range(3):
            index = 3*i+j
            cell = half_cell(rotation[i][j],negative_zero[index])
            entries.append({'index':[i,j], 'original_f32_hex':raw[4*index:4*index+4].hex(),
                            'stored_value':rotation[i][j], 'negative_zero':negative_zero[index],
                            'rounding_cell':cell, 'hypothetical_entry':q[i][j],
                            'cell_membership':inside(cell,q[i][j])})
    success = all(entry['cell_membership'] for entry in entries)
    return serial({
        'status':'COMPATIBILITY_CERTIFIED' if success else 'INCONCLUSIVE',
        'method':'one deterministic largest trace/diagonal branch; w,x,y,z tie order; rational homogeneous quaternion; no sqrt, fitting, or search',
        'proposals_attempted':1,
        'input':{'R_hex':raw.hex(),'byte_order':order,'sha256':hashlib.sha256(raw).hexdigest()},
        'proposal':proposal,
        'exact_SO3':{'RRt':rr,'RtR':rt,'determinant':det,'verified':exact_so3},
        'cells':entries, 'cells_satisfied':sum(e['cell_membership'] for e in entries),
        'zero_convention':'Exact rational zero rounds to +0; a -0 target requires a negative witness entry. No assertion about upstream signed-zero arithmetic.',
        'interpretation':('An exact proper rotation whose nearest-even half rounding reproduces these nine stored coefficients exists under the stated zero convention. This is mathematical compatibility, not actual generation provenance.' if success else 'This single fixed proposal did not certify compatibility. The hypothesis is unresolved; no infeasibility or causal conclusion follows.'),
        'forbidden_uses':['camera replacement','projection','data qualification','threshold alteration','pre-quantization provenance claim'],
    })


def unique_pairs(pairs):
    result={}
    for key,value in pairs:
        if key in result:
            raise ValueError('duplicate input key')
        result[key]=value
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    args=parser.parse_args()
    if args.input.stat().st_size > 4096:
        raise ValueError('saved-R input exceeds 4 KiB')
    source=json.loads(args.input.read_text(),object_pairs_hook=unique_pairs)
    print(json.dumps(certify(source),indent=2,allow_nan=False))

if __name__=='__main__':
    main()
