"""Independent exact Gaussian arithmetic and dyadic half-grid checks; saved JSON only."""
from fractions import Fraction as F
from pathlib import Path
import hashlib,json,math,struct,subprocess
B=Path('results/0003-rotation-stageB-20261017');h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=json.loads((B/'audit/diagnostic.json').read_text());e=json.loads((B/'audit/camera-excerpt.json').read_text());s=a['statistics'];count=0
matrices={}
for name,n in [('intrinsics',3),('extrinsics',4)]:
 c=e['camera_access'][name];assert c['status']=='completed' and c['counts_complete']
 assert c['header']['shape']==[849,n,n] and c['header']['order']=='C' and c['header']['dtype']=='<f4'
 matrix=[];payload=b''
 for i,row in enumerate(c['values']):
  values=[]
  for j,v in enumerate(row):
   raw=bytes.fromhex(v['stored_bytes_hex']);x=struct.unpack('<f',raw)[0]
   assert len(raw)==4 and x.hex()==v['float_hex'] and str(F.from_float(x))==v['exact_rational'] and v['finite'] and math.isfinite(x)
   assert v['byte_offset_in_payload']==4*(i*n+j);values.append(F.from_float(x));payload+=raw;count+=1
  matrix.append(values)
 assert hashlib.sha256(payload).hexdigest()==c['accounting']['span_sha256']
 assert c['accounting']['numeric_span_bytes']==n*n*4 and c['accounting']['discarded_bytes']==0
 matrices[name]=matrix
assert count==25
R=[r[:3]for r in matrices['extrinsics'][:3]]
def gram(rows):return [[sum((x*y for x,y in zip(row,col)),F())-int(i==j) for j,col in enumerate(rows)]for i,row in enumerate(rows)]
rr=gram(R);rt=gram(list(zip(*R)))
# Gaussian elimination determinant is independent of production permutation expansion.
q=[r[:]for r in R];det=F(1)
for col in range(3):
 pivot=next((i for i in range(col,3)if q[i][col]),None)
 if pivot is None:det=F();break
 if pivot!=col:q[col],q[pivot]=q[pivot],q[col];det=-det
 lead=q[col][col];det*=lead
 for i in range(col+1,3):
  factor=q[i][col]/lead
  for j in range(col,3):q[i][j]-=factor*q[col][j]
def decode(v):return F(int(v['numerator']),int(v['denominator']))
assert [[decode(x)for x in r]for r in s['exact_fraction']['RRt_minus_I']]==rr
assert [[decode(x)for x in r]for r in s['exact_fraction']['RtR_minus_I']]==rt
assert decode(s['exact_fraction']['det_R'])==det
tau=F.from_float(s['original_tau']);assert tau==F(1,2**20)
assert not s['original_rotation_gate_pass'] and not s['exact_fraction']['rotation_gate_pass']
assert all(v>tau for v in [max(map(abs,sum(rr,[]))),max(map(abs,sum(rt,[]))),abs(det-1)])
u=F(1,2**24);assert max(map(abs,sum(rr+rt,[])))>6*u+3*u*u or abs(det-1)>9*u+27*u*u+27*u**3
assert s['single_final_f32_rounding']['excluded']
def half_grid(x):
 # Integer significand lattice, no half conversion/rounding instruction.
 x=abs(x)
 if x==0:return True
 if x>65504:return False
 if x<F(1,2**14):return (x*2**24).denominator==1
 exponent=-14
 while exponent<15 and x>=F(2)**(exponent+1):exponent+=1
 mantissa=x/(F(2)**(exponent-10))
 return mantissa.denominator==1 and 1024<=mantissa<=2047
actual={}
for name,g in s['binary16_exact_representability']['groups'].items():
 matrix=matrices['intrinsics' if name=='K9' else 'extrinsics'];flags=[]
 for row in g['values']:
  i,j=row['index'];x=matrix[i][j];flag=half_grid(x);flags.append(flag)
  assert row['exactly_binary16']==flag
  assert row['original_f32_le_hex']==e['camera_access']['intrinsics' if name=='K9'else 'extrinsics']['values'][i][j]['stored_bytes_hex']
 assert len(flags)==g['total'] and sum(flags)==g['exactly_representable_count'] and all(flags)==g['all_exactly_representable']
 actual[name]={'total':len(flags),'exact_count':sum(flags)}
assert a['status']=='completed_diagnostic_original_rotation_gate_failed' and a['counts_complete']
assert a['charged_bytes']==a['returned_bytes_including_carried']+a['reserves_including_carried']==22910156
assert a['new_returned_bytes']==22+128+128+100==378 and a['new_reserves']==2*65536+348+22==131442
for phase in ['preflight','diagnostic']:
 r=json.loads((B/(phase+'-parent.json')).read_text());assert r['observed_exit_code']==r['raw_wait_status']==0
 for p,d in r['source_sha256'].items():assert h(p)==d,p
 for st,d in r['logs_sha256'].items():assert h(B/(phase+'.'+st+'.log'))==d
 assert r['affinity']==[0,1] and r['AS_bytes']==6*1024**3 and r['timeout_seconds']==900
 p=json.loads((B/'preflight.stdout.log').read_text());assert p['verified_utc']<json.loads((B/'diagnostic-parent.json').read_text())['started_utc']
print(json.dumps({'status':'passed','saved_values_checked':count,'new_cache_reads':0,'exact_Gram_entries_checked':18,'determinant_method':'exact Fraction Gaussian elimination','half_grid_method':'exact dyadic integer significand lattice','half_groups':actual,'absolute_tolerance_for_exact_checks':0,'counts_and_wait_source_log_bindings':'passed','scope':'independent saved-record verification; no projection, convention changes or cache accesses'},indent=2))
