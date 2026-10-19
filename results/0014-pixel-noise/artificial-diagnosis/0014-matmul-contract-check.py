"""Artificial-only explicit and non-BLAS contraction comparison."""
import json
import numpy as np
import warnings
rng=np.random.default_rng(20261020)
N=10000
A=rng.uniform(-1.,1.,(N,3)); B=rng.uniform(-1.,1.,(3,3))
J=rng.uniform(-2.,2.,(N,5)); r=rng.uniform(-1.,1.,N); w=1/(.25+r*r)
ref={'Nx3':sum(A[:,j,None]*B[None,j,:] for j in range(3)),
     'H':np.array([[np.sum(J[:,i]*w*J[:,j]) for j in range(5)] for i in range(5)]),
     'g':np.array([np.sum(J[:,i]*w*r) for i in range(5)])}
ops={'Nx3_matmul':lambda:A@B,'Nx3_einsum':lambda:np.einsum('ij,jk->ik',A,B,optimize=False),
     'H_matmul':lambda:J.T@(w[:,None]*J),'H_einsum':lambda:np.einsum('ni,n,nj->ij',J,w,J,optimize=False),
     'g_matmul':lambda:J.T@(w*r),'g_einsum':lambda:np.einsum('ni,n->i',J,w*r,optimize=False)}
rows=[]
for name,op in ops.items():
 row={'operation':name,'conservative_result_bound':3. if name.startswith('Nx3') else (16.*N if name.startswith('H') else 8.*N)}
 try:
  with np.errstate(over='raise',divide='raise',invalid='raise'): result=op()
  row['strict_status']='success'
 except FloatingPointError as exc:
  row['strict_status']=str(exc)
  with warnings.catch_warnings(record=True) as records:
   warnings.simplefilter('always')
   with np.errstate(over='warn',divide='warn',invalid='warn'): result=op()
   row['warnings']=[str(v.message) for v in records]
 with np.errstate(over='raise',divide='raise',invalid='raise'):
  row['finite']=bool(np.isfinite(result).all())
  row['max_abs_difference_from_explicit_reduction']=float(np.max(np.abs(result-ref[name.split('_')[0]])))
 rows.append(row)
print(json.dumps({'N':N,'checks':rows},indent=2,allow_nan=False))
