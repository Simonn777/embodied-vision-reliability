"""Synthetic-only camera/numerical/Git/termination fixtures, no real cache reads."""
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import zipfile
from fractions import Fraction
from evr.bounded_point_audit import ReadBudget
from evr.point_geometry_v2 import header,atomic_json
from evr.rotation_diagnostic_v1 import (frame0,continued_budget,read_cameras,diagnose,double_statistics,exact_statistics,rounding_bounds,TAU)


def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def identity(n):return[[float(i==j)for j in range(n)]for i in range(n)]
def synthetic_npy(n=3,T=4,order='C',endian='<'):
 shape=(T,n,n);text=repr({'descr':endian+'f4','fortran_order':order=='F','shape':shape}).encode()
 prefix=b'\x93NUMPY\x01\x00'+struct.pack('<H',len(text))+text
 import itertools
 axes=[0,1,2]if order=='C'else[2,1,0];raw=[]
 for v in itertools.product(*(range(shape[i])for i in axes)):
  ix=[0]*3
  for axis,value in zip(axes,v):ix[axis]=value
  raw.append(struct.pack(endian+'f',100*ix[0]+10*ix[1]+ix[2]))
 return prefix+b''.join(raw)


def load_runner():
 p=Path(__file__).resolve().parents[1]/'scripts/run_rotation_diagnostic_v1.py'
 spec=importlib.util.spec_from_file_location('rotation_runner',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class NumericalTests(unittest.TestCase):
 def test_proper_f32_rotation_and_original_formula(self):
  c=f32(math.sqrt(.5));rt=identity(4);rt[:2]=[[c,-c,0.,0.],[c,c,0.,0.]]
  s=diagnose(identity(3),rt)
  self.assertTrue(s['original_rotation_gate_pass']);self.assertEqual(s['original_tau'],TAU)
  self.assertFalse(s['single_final_f32_rounding']['excluded'])
  # Exact rational equality, not rounded decimal lookup.
  exact=2*Fraction.from_float(c)**2-1
  self.assertEqual(s['exact_fraction']['RRt_minus_I'][0][0],{'numerator':str(exact.numerator),'denominator':str(exact.denominator)})

 def test_scale_shear_reflection_and_norms(self):
  rt=identity(4);rt[0][0]=f32(1.0001)
  s=diagnose(identity(3),rt);self.assertFalse(s['original_rotation_gate_pass']);self.assertTrue(s['single_final_f32_rounding']['excluded'])
  rt=identity(4);rt[0][1]=f32(.0001)
  s=diagnose(identity(3),rt);self.assertFalse(s['original_rotation_gate_pass']);self.assertGreater(s['double']['row_offdiagonal_dots'][0][2],0)
  rt=identity(4);rt[0][0]=-1.
  s=diagnose(identity(3),rt);self.assertEqual(s['double']['det_R'],-1);self.assertFalse(s['original_gate_checks']['determinant'])

 def test_fraction_hand_computed_and_double_difference(self):
  R=[[2.,1.,0.],[0.,1.,0.],[0.,0.,1.]]
  rr,rt,det=exact_statistics(R)
  self.assertEqual(rr,[[4,1,0],[1,0,0],[0,0,0]])
  self.assertEqual(rt,[[3,2,0],[2,1,0],[0,0,0]])
  self.assertEqual(det,Fraction(2))
  self.assertEqual(double_statistics(R),(rr,rt,det))

 def test_transpose_and_orientation_preserving_signs(self):
  R=[[f32(1.001),f32(.002),0.],[0.,1.,0.],[0.,0.,1.]]
  rr,rt,d=exact_statistics(R);a,b,dt=exact_statistics(list(map(list,zip(*R))))
  self.assertEqual((a,b,dt),(rt,rr,d))
  sign=[-1,1,-1];conjugated=[[sign[i]*R[i][j]*sign[j]for j in range(3)]for i in range(3)]
  x,y,z=exact_statistics(conjugated)
  self.assertEqual(max(map(abs,sum(x,[]))),max(map(abs,sum(rr,[]))));self.assertEqual(z,d)
  # Counterexample to a claim about ALL sign flips: changing handedness can
  # turn reflection into identity, but this forbidden repair is fixture-only.
  reflection=identity(4);reflection[0][0]=-1.
  self.assertFalse(diagnose(identity(3),reflection)['original_rotation_gate_pass'])
  self.assertTrue(diagnose(identity(3),identity(4))['original_rotation_gate_pass'])

 def test_exact_rows_nonfinite_dtype_and_shape(self):
  for bad in [math.nan,math.inf,-math.inf]:
   k=identity(3);k[0][0]=bad
   with self.assertRaises(ValueError):diagnose(k,identity(4))
  with self.assertRaises(ValueError):diagnose([[1]],identity(4))
  k=identity(3);k[0][0]=.1
  with self.assertRaises(ValueError):diagnose(k,identity(4))
  rt=identity(4);rt[3][0]=f32(1e-6)
  self.assertFalse(diagnose(identity(3),rt)['original_gate_checks']['RT_exact_bottom_row'])
  k=identity(3);k[0][0]=0.
  self.assertFalse(diagnose(k,identity(4))['original_gate_checks']['positive_focal'])

 def test_narrow_rounding_bound_constants(self):
  b=rounding_bounds();u=Fraction(1,2**24)
  self.assertEqual(b['gram'],6*u+3*u*u);self.assertEqual(b['determinant'],9*u+27*u*u+27*u**3)
  self.assertLess(b['gram'],Fraction.from_float(TAU));self.assertLess(b['determinant'],Fraction.from_float(TAU))


class ExtractionTests(unittest.TestCase):
 def test_only_frame0_c_f_both_endian(self):
  for n in [3,4]:
   for order in ['C','F']:
    for endian in ['<','>']:
     raw=synthetic_npy(n,4,order,endian);b=ReadBudget();f=io.BytesIO(raw);h=header(f,b);before=b.returned_bytes
     entries,a=frame0(f,h,b)
     for i in range(n):
      for j in range(n):self.assertEqual(float.fromhex(entries[i][j]['float_hex']),10*i+j)
     expected=n*n*4 if order=='C'else((n*n-1)*4+1)*4
     self.assertEqual(a['numeric_span_bytes'],expected);self.assertEqual(b.returned_bytes-before,expected)
     self.assertEqual(a['retained_scalars'],n*n)

 def test_exact_and_short_budget_fail_before_payload(self):
  raw=synthetic_npy(3,4,'F');b=ReadBudget();f=io.BytesIO(raw);h=header(f,b);start=b.returned_bytes
  span=((9-1)*4+1)*4
  for allowance,ok in [(span,True),(span-1,False)]:
   b=ReadBudget(start+allowance);f=io.BytesIO(raw);h=header(f,b);position=f.tell()
   if ok:frame0(f,h,b);self.assertEqual(b.charged_bytes,start+span)
   else:
    with self.assertRaisesRegex(ValueError,'BEFORE'):frame0(f,h,b)
    self.assertEqual(f.tell(),position)

 def test_carry_exact_and_truncation(self):
  b=continued_budget();self.assertEqual(b.charged_bytes,22778336);self.assertEqual(b.returned_bytes,20614538)
  self.assertEqual(b.stream_reserve_bytes,2163798);self.assertEqual(b.central_directory_reserved_bytes,1110)
  raw=synthetic_npy();b=ReadBudget();f=io.BytesIO(raw[:-120]);h=header(f,b)
  with self.assertRaises(ValueError):frame0(f,h,b)

 def test_no_other_payload_opened(self):
  # Only camera payloads carry a standard NPY; forbidden trajectory payloads
  # are deliberately non-NPY so any attempt to inspect them fails.
  f=io.BytesIO()
  with zipfile.ZipFile(f,'w')as z:
   for name,n in [('intrinsics.npy',3),('extrinsics.npy',4)]:z.writestr(name,synthetic_npy(n,849))
   z.writestr('trajs_2d.npy',b'NEVER OPEN OR PARSE')
  f.seek(0);opened=[]
  class Counting(zipfile.ZipFile):
   def open(self,name,*args,**kw):opened.append(name.filename if hasattr(name,'filename')else name);return super().open(name,*args,**kw)
  from unittest.mock import patch
  result={}
  with patch('evr.rotation_diagnostic_v1.zipfile.ZipFile',Counting):read_cameras(f,ReadBudget(),result,lambda:None)
  self.assertEqual(opened,['intrinsics.npy','extrinsics.npy'])
  self.assertEqual(sum(v['accounting']['retained_scalars']for v in result['camera_access'].values()),25)


class BindingTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.m=load_runner()
  self.g('init','-q');(self.root/'input.txt').write_text('original')
  self.g('add','.');self.commit('input');self.input=self.g('rev-parse','HEAD').strip().decode()
  self.path='receipt.json';self.remote='refs/remotes/origin/review'
  self.r={'schema':1,'scope':'FAILURE_FOLLOWUP_FRAME0_CAMERA_ONLY','input_commit':self.input,'inputs_sha256':{'input.txt':hashlib.sha256(b'original').hexdigest()}}
  self.write(self.r);self.g('add','.');self.commit('receipt');self.g('update-ref',self.remote,self.g('rev-parse','HEAD').strip().decode())
 def tearDown(self):self.tmp.cleanup()
 def g(self,*args):return subprocess.check_output(['git','-C',str(self.root),*args],stderr=subprocess.DEVNULL)
 def commit(self,msg):self.g('-c','user.name=Synthetic Fixture','-c','user.email=fixture@example.invalid','commit','-qm',msg)
 def write(self,r):(self.root/self.path).write_text(json.dumps(r))
 def verify(self):return self.m.verify_current_receipt(self.root,self.path,{'input.txt'},self.remote)
 def test_actual_git_binding_pass(self):self.assertEqual(self.verify()['input_commit'],self.input)
 def test_uncommitted_receipt_or_input_fails(self):
  (self.root/'input.txt').write_text('changed')
  with self.assertRaises(ValueError):self.verify()
  (self.root/'input.txt').write_text('original');r=copy.deepcopy(self.r);r['input_commit']='a'*40;self.write(r)
  with self.assertRaises(ValueError):self.verify()
 def test_empty_missing_extra_or_old_commit_new_hash_fails(self):
  for mapping in [{},{'extra':'b'*64},{'input.txt':'b'*64}]:
   r=copy.deepcopy(self.r);r['inputs_sha256']=mapping;self.write(r)
   with self.assertRaises(ValueError):self.verify()
 def test_index_remote_self_reference(self):
  (self.root/'input.txt').write_text('indexed change');self.g('add','input.txt');(self.root/'input.txt').write_text('original')
  with self.assertRaises(ValueError):self.verify()
  self.g('add','input.txt')
  self.g('update-ref',self.remote,self.input)
  with self.assertRaises(ValueError):self.verify()
  self.g('update-ref',self.remote,self.g('rev-parse','HEAD').strip().decode())
  r=copy.deepcopy(self.r);r['input_commit']=self.g('rev-parse','HEAD').strip().decode();self.write(r)
  with self.assertRaisesRegex(ValueError,'self-reference'):self.verify()
 def test_published_old_input_commit_with_new_hash_fails(self):
  (self.root/'input.txt').write_text('changed');r=copy.deepcopy(self.r);r['inputs_sha256']['input.txt']=hashlib.sha256(b'changed').hexdigest();self.write(r)
  self.g('add','.');self.commit('new values incorrectly bound to old input')
  self.g('update-ref',self.remote,self.g('rev-parse','HEAD').strip().decode())
  with self.assertRaisesRegex(ValueError,'input blob'):self.verify()
 def test_no_production_receipt_means_no_cache_access(self):
  cwd=Path.cwd()
  try:
   os.chdir(self.root)
   with self.assertRaises(ValueError):self.m.verify_published()
  finally:os.chdir(cwd)


class AtomicTests(unittest.TestCase):
 def test_sigterm_saved_intent_not_zero_claim(self):
  import select,signal,sys
  code="from evr.point_geometry_v2 import atomic_json;import sys,signal;atomic_json(sys.argv[1],{'camera_access_state':'NUMERIC_CAMERA_ACCESS_STARTING','counts_complete':False,'interrupted_counter_semantics':'LOWER_BOUNDS_OR_UNKNOWN'});print('READY',flush=True);signal.pause()"
  with tempfile.TemporaryDirectory()as d:
   path=Path(d)/'record.json';p=subprocess.Popen([sys.executable,'-c',code,str(path)],env={**os.environ,'PYTHONPATH':'src'},stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
   try:
    self.assertTrue(select.select([p.stdout],[],[],10)[0]);self.assertEqual(p.stdout.readline().strip(),'READY')
    p.terminate();p.wait(timeout=10);self.assertEqual(p.returncode,-signal.SIGTERM)
    r=json.loads(path.read_text());self.assertFalse(r['counts_complete']);self.assertEqual(r['camera_access_state'],'NUMERIC_CAMERA_ACCESS_STARTING')
   finally:
    if p.poll()is None:p.kill();p.wait(timeout=10)
    p.stdout.close();p.stderr.close()

if __name__=='__main__':unittest.main()
