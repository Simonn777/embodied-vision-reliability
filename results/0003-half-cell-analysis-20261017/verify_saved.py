"""Saved witness only: exact Gaussian SO(3) and independently decoded cell endpoints."""
from fractions import Fraction as F
from pathlib import Path
import hashlib,json,struct
B=Path('results/0003-half-cell-analysis-20261017');h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
o=json.loads((B/'output.json').read_text());source=json.loads((B/'input.json').read_text());pre=json.loads((B/'pre-execution.json').read_text());execution=json.loads((B/'execution.json').read_text())
assert o['input']['R_hex']==source['R_hex'] and source['byte_order']=='little'
raw=bytes.fromhex(source['R_hex']);assert len(raw)==36 and hashlib.sha256(raw).hexdigest()==o['input']['sha256']==execution['original_R36_bytes_sha256']
assert h(B/'input.json')==pre['input_json_sha256']==execution['input_sha256']
assert h(B/'output.json')==execution['stdout_sha256'] and h(B/'stderr.log')==execution['stderr_sha256']
assert h(B/'pre-execution.json')==execution['pre_execution_receipt_sha256'] and pre['verified_before_invocation_utc']<execution['started_utc']
assert execution['observed_exit_code']==execution['raw_wait_status']==0
for p,d in execution['source_sha256'].items():assert h(p)==d,p
assert pre['actual_execution_commit']==execution['execution_commit']=='565c1aa4d07f548c9201eebf18204e79b3f72473'
assert pre['prepublished_method_commit']==execution['prepublished_method_commit']=='c1bcc597db980cd3aa14c0747af0183da266e6af'
def fraction(d):return F(int(d['numerator']),int(d['denominator']))
Q=[[fraction(v)for v in row]for row in o['proposal']['hypothetical_rotation']]
def gram(rows):return [[sum((a*b for a,b in zip(x,y)),F())for y in rows]for x in rows]
identity=[[F(int(i==j))for j in range(3)]for i in range(3)]
assert gram(Q)==gram(list(zip(*Q)))==identity
A=[row[:]for row in Q];det=F(1)
for k in range(3):
 pivot=next(i for i in range(k,3)if A[i][k])
 if pivot!=k:A[k],A[pivot]=A[pivot],A[k];det=-det
 lead=A[k][k];det*=lead
 for i in range(k+1,3):
  factor=A[i][k]/lead
  for j in range(k,3):A[i][j]-=factor*A[k][j]
assert det==1 and fraction(o['exact_SO3']['determinant'])==1 and o['exact_SO3']['verified']
for key in ['RRt','RtR']:assert [[fraction(v)for v in row]for row in o['exact_SO3'][key]]==identity
def magnitude(code):
 # Direct integer encoding formula, independent of production half-cell functions.
 exponent=code//1024;mantissa=code%1024
 if exponent==0:return F(mantissa,2**24)
 return F(1024+mantissa)*F(2)**(exponent-25)
failed=[]
for index,row in enumerate(o['cells']):
 i,j=divmod(index,3);assert row['index']==[i,j]
 b=raw[4*index:4*index+4];value=struct.unpack('<f',b)[0];bits=struct.unpack('<H',struct.pack('<e',value))[0]
 assert struct.pack('<f',struct.unpack('<e',struct.pack('<H',bits))[0])==b
 assert row['original_f32_hex']==b.hex() and fraction(row['stored_value'])==F.from_float(value)
 sign=bits>>15;code=bits&32767;assert code<=31743
 if code==0:
  lo,hi=(-F(1,2**25),F(0))if sign else(F(0),F(1,2**25));closed_lo=True;closed_hi=not sign
 else:
  prev=magnitude(code-1);at=magnitude(code);nxt=magnitude(code+1)if code<31743 else F(65536)
  lo,hi=(prev+at)/2,(at+nxt)/2
  if sign:lo,hi=-hi,-lo
  closed_lo=closed_hi=(code%2==0)
 cell=row['rounding_cell'];assert cell['half_bits']==f'{bits:04x}'
 assert fraction(cell['lower'])==lo and fraction(cell['upper'])==hi
 assert cell['lower_closed']==closed_lo and cell['upper_closed']==closed_hi
 q=Q[i][j];assert fraction(row['hypothetical_entry'])==q
 passed=(q>lo or closed_lo and q==lo)and(q<hi or closed_hi and q==hi)
 assert row['cell_membership']==passed
 if not passed:failed.append([i,j])
assert len(o['cells'])==9 and o['proposals_attempted']==1 and o['cells_satisfied']==9-len(failed)
assert o['status']==('COMPATIBILITY_CERTIFIED'if not failed else'INCONCLUSIVE')
print(json.dumps({'status':'saved_output_verified','checker_outcome':o['status'],'exact_SO3_verified_independently':True,'Gaussian_determinant':str(det),'cells_checked':9,'cells_satisfied':9-len(failed),'failed_indices':failed,'exact_tolerance':0,'new_proposals_or_search':0,'new_cache_or_geometry_access':0,'interpretation':'verifies this saved proposal and cell accounting only; a failed proposal is not proof no compatible predecessor exists'},indent=2))
