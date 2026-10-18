#!/usr/bin/env python3
"""Independent exact verifier of a published saved-R / claimed-Q certificate.

Never imports the proposal generator or regenerates its quaternion. It uses exact
rational nearest-neighbor half rounding and pivoted Gaussian determinant. Inputs
are saved JSON files only. No repository, dataset, archive, model, or network use.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import struct


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def half_value(code):
    magnitude=code & 0x7fff
    if magnitude >= 0x7c00:
        raise ValueError('finite half required')
    exponent= magnitude >> 10
    significand= (magnitude & 1023) + (1024 if exponent else 0)
    shift= exponent-25 if exponent else -24
    value=F(significand * 2**shift) if shift >= 0 else F(significand,2**(-shift))
    return -value if code & 0x8000 else value


def nearest_half_code(value):
    """Exact binary search of finite positive codes, then nearest/even decision.

    Exact mathematical zero -> +0; a negative nonzero value may round to -0.
    0x7c00/0xfc00 denote overflow, not an admissible finite coefficient.
    """
    if value == 0:
        return 0
    sign=0x8000 if value < 0 else 0
    value=abs(value)
    if value > half_value(0x7bff):
        return sign | (0x7bff if value < 65520 else 0x7c00)
    left,right=0,0x7bff
    while left < right:
        middle=(left+right+1)//2
        if half_value(middle) <= value:
            left=middle
        else:
            right=middle-1
    lower=left
    if half_value(lower) == value:
        return sign | lower
    upper=lower+1
    twice=2*value
    midpoint_twice=half_value(lower)+half_value(upper)
    chosen=lower if twice < midpoint_twice else upper if twice > midpoint_twice else (lower if lower % 2 == 0 else upper)
    return sign | chosen


def rational(record):
    if type(record) is not dict or 'numerator' not in record or 'denominator' not in record:
        raise ValueError('rational record required')
    n,d=record['numerator'],record['denominator']
    if not isinstance(n,str) or not isinstance(d,str) or len(n)>2000 or len(d)>2000:
        raise ValueError('bounded integer-string rational required')
    if not n.lstrip('-').isdigit() or not d.isdigit() or int(d) <= 0:
        raise ValueError('positive rational denominator required')
    return F(int(n),int(d))


def matrix(record):
    if not isinstance(record,list) or len(record)!=3 or any(not isinstance(row,list) or len(row)!=3 for row in record):
        raise ValueError('3x3 rational matrix required')
    return [[rational(x) for x in row] for row in record]


def transposed(m):
    return list(map(list,zip(*m)))


def product(a,b):
    return [[sum((a[i][k]*b[k][j] for k in range(3)),F(0)) for j in range(3)] for i in range(3)]


def determinant_elimination(matrix):
    work=[row[:] for row in matrix]
    determinant=F(1)
    for column in range(3):
        pivot=next((row for row in range(column,3) if work[row][column] != 0),None)
        if pivot is None:
            return F(0)
        if pivot != column:
            work[column],work[pivot]=work[pivot],work[column]
            determinant=-determinant
        divisor=work[column][column]
        determinant*=divisor
        for row in range(column+1,3):
            multiplier=work[row][column]/divisor
            for j in range(column,3):
                work[row][j]-=multiplier*work[column][j]
    return determinant


def original_values(source):
    if type(source) is not dict or set(source) not in ({'R_hex'},{'R_hex','byte_order'}):
        raise ValueError('original input requires R_hex and optional byte_order only')
    order=source.get('byte_order','little')
    if order not in ('little','big') or not isinstance(source['R_hex'],str):
        raise ValueError('invalid input schema')
    raw=bytes.fromhex(source['R_hex'])
    if len(raw)!=36:
        raise ValueError('exactly 36 original f32 bytes required')
    floats=struct.unpack(('<' if order=='little' else '>')+'9f',raw)
    if not all(math.isfinite(value) for value in floats):
        raise ValueError('nonfinite original value')
    values=[F.from_float(value) for value in floats]
    codes=[]
    for index,value in enumerate(values):
        is_minus_zero=int.from_bytes(raw[4*index:4*index+4],order)==0x80000000
        code=0x8000 if is_minus_zero else nearest_half_code(value)
        codes.append(code if code & 0x7fff < 0x7c00 and half_value(code)==value else None)
    return raw,order,values,codes


def expected_cell(code):
    """Independently validate reported interval metadata; decisions use rounding."""
    if code is None:
        return None
    magnitude=code & 0x7fff
    negative=bool(code & 0x8000)
    if magnitude == 0:
        return (F(-1,2**25),F(0),True,False) if negative else (F(0),F(1,2**25),True,True)
    target=half_value(magnitude)
    low=(half_value(magnitude-1)+target)/2
    high=(target+(half_value(magnitude+1) if magnitude<0x7bff else F(65536)))/2
    if negative:
        low,high=-high,-low
    even=magnitude%2==0
    return low,high,even,even


def check(condition,message):
    if not condition:
        raise ValueError(message)


def verify(source,claimed):
    raw,order,stored,targets=original_values(source)
    q=matrix(claimed['proposal']['hypothetical_rotation'])
    rr,rt=product(q,transposed(q)),product(transposed(q),q)
    det=determinant_elimination(q)
    identity=[[F(int(i==j)) for j in range(3)] for i in range(3)]
    so3=rr==identity and rt==identity and det==1
    check(so3,'claimed proposal is not exact SO(3)')
    check(claimed['proposals_attempted']==1 and type(claimed['proposals_attempted']) is int,'proposal count is not exactly one')
    check(claimed['input']=={'R_hex':raw.hex(),'byte_order':order,'sha256':sha(raw)},'claimed input does not bind original bytes')
    check(matrix(claimed['exact_SO3']['RRt'])==rr,'reported RRt mismatch')
    check(matrix(claimed['exact_SO3']['RtR'])==rt,'reported RtR mismatch')
    check(rational(claimed['exact_SO3']['determinant'])==det,'reported determinant mismatch')
    check(claimed['exact_SO3']['verified'] is True,'SO3 verification flag mismatch')
    check(len(claimed['cells'])==9,'exactly nine cell records required')
    rows=[]
    for index,(value,target,entry) in enumerate(zip(stored,targets,claimed['cells'])):
        i,j=divmod(index,3)
        expected_code=nearest_half_code(q[i][j])
        membership=target is not None and expected_code==target
        check(entry['index']==[i,j],'cell indices/order mismatch')
        check(entry['original_f32_hex']==raw[4*index:4*index+4].hex(),'cell source bytes mismatch')
        check(rational(entry['stored_value'])==value,'cell stored rational mismatch')
        check(rational(entry['hypothetical_entry'])==q[i][j],'cell witness entry mismatch')
        check(entry['negative_zero']==(value==0 and target==0x8000),'source signed zero mismatch')
        check(type(entry['cell_membership']) is bool and entry['cell_membership']==membership,'cell membership differs from exact nearest-even result')
        interval=expected_cell(target)
        metadata=entry['rounding_cell']
        if interval is None:
            check(metadata is None,'non-half input must have no reported cell')
        else:
            low,high,lc,uc=interval
            check(rational(metadata['lower'])==low and rational(metadata['upper'])==high,'reported cell endpoints mismatch')
            check(metadata['lower_closed'] is lc and metadata['upper_closed'] is uc,'reported tie ownership mismatch')
            check(metadata['half_bits']==f'{target:04x}','reported half code mismatch')
        rows.append({'index':[i,j],'target_half_bits':None if target is None else f'{target:04x}',
                     'exact_rounded_witness_half_bits':f'{expected_code:04x}','membership':membership})
    count=sum(item['membership'] for item in rows)
    status='COMPATIBILITY_CERTIFIED' if count==9 else 'INCONCLUSIVE'
    check(type(claimed['cells_satisfied']) is int and claimed['cells_satisfied']==count,'claimed satisfied-cell count mismatch')
    check(claimed['status']==status,'claimed certificate status mismatch')
    return {'verification':'passed','independently_expected_status':status,'cells_satisfied':count,
            'proposals_attempted':1,'original_f32_values':9,'exact_SO3':True,
            'gram_entries_checked':18,'determinant':str(det),'cell_checks':rows,
            'arithmetic':'Fraction exact nearest-neighbor/even half-code rounding; determinant via pivoted Gaussian elimination',
            'boundary':'Checks the supplied exact witness and saved-byte binding. Does not regenerate quaternion proposal, prove execution provenance, or evaluate geometry. Positive certificate proves compatibility only; failed proposal is inconclusive.'}


def unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key]=value
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-input',required=True,type=Path)
    parser.add_argument('--claimed-output',required=True,type=Path)
    args=parser.parse_args()
    originals=[]
    for path in [args.original_input,args.claimed_output]:
        if path.stat().st_size>1024**2:
            raise ValueError('saved JSON exceeds 1 MiB')
        originals.append(path.read_bytes())
    report=verify(*(json.loads(raw,object_pairs_hook=unique_object) for raw in originals))
    report['original_input_sha256']=sha(originals[0])
    report['claimed_output_sha256']=sha(originals[1])
    report['verifier_source_sha256']=sha(Path(__file__).read_bytes())
    print(json.dumps(report,indent=2,allow_nan=False))

if __name__=='__main__':
    main()
