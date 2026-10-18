"""Synthetic-only sparse/annotation separation, bindings and durable interruption."""
import copy,hashlib,importlib.util,io,json,math,os,select,signal,struct,subprocess,sys,tempfile,unittest,warnings,zipfile
from pathlib import Path
from evr.bounded_point_audit import ReadBudget
from evr.annotation_consistency_v1 import (SCHEMA,SPANS,requests,sparse_values,continued_budget,read_inputs,gate,camera_bytes,check_points)


def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def ident(n):return [[float(i==j)for j in range(n)]for i in range(n)]
def fixture():
 K=ident(3);RT=ident(4);s=f32(1.0001);RT[0][0]=s
 data={'intrinsics':{},'extrinsics':{},'trajs_2d':{},'trajs_3d':{},'visibs':{},'valids':{}}
 for t in range(3):
  for key,m in [('intrinsics',K),('extrinsics',RT)]:
   for i,row in enumerate(m):
    for j,v in enumerate(row):data[key][(t,i,j)]=v
  for p in (0,1):
   for c,x in enumerate([0.,0.]):data['trajs_2d'][(t,p,c)]=x
   for c,x in enumerate([0.,0.,1.]):data['trajs_3d'][(t,p,c)]=x
   data['visibs'][(t,p)]=data['valids'][(t,p)]=1
 old={'tau':2**-20,'RRt_max':s*s-1,'RtR_max':s*s-1,'det_minus_one_abs':s-1,'proper_rotation_pass':False}
 return data,{'K':camera_bytes(K),'RT':camera_bytes(RT)},old


class SparseTests(unittest.TestCase):
 def test_production_spans_counts_and_carry_without_real_data(self):
  b=continued_budget();self.assertEqual((b.returned_bytes,b.stream_reserve_bytes,b.central_directory_reserved_bytes,b.charged_bytes),(20614916,2295240,1480,22910156))
  retained=total=0
  for key,(shape,dtype,order)in SCHEMA.items():
   size=1 if dtype=='|b1'else 4;h={'shape':shape,'dtype':dtype,'order':order,'itemsize':size,'format':'<B'if size==1 else'<f'}
   values,a=sparse_values(io.BytesIO(bytes(SPANS[key])),h,requests(key),b)
   self.assertEqual(a['numeric_span_bytes'],SPANS[key]);self.assertEqual(set(values),set(requests(key)))
   retained+=a['retained_bytes'];total+=a['retained_scalars']
  self.assertEqual((retained,total),(432,117));self.assertEqual(b.returned_bytes-20614916,470884)
 def test_asymmetric_C_F_big_little_and_future_flag_discard(self):
  import itertools
  for order in ['C','F']:
   for endian in ['<','>']:
    shape=[4,2,2];axes=[0,1,2]if order=='C'else[2,1,0];raw=b''
    for ordered in itertools.product(*(range(shape[x])for x in axes)):
     ix=[0]*3
     for a,v in zip(axes,ordered):ix[a]=v
     raw+=struct.pack(endian+'f',100*ix[0]+10*ix[1]+ix[2])
    h={'shape':shape,'order':order,'itemsize':4,'format':endian+'f'};indices=list(itertools.product(range(3),(0,1),range(2)))
    values,a=sparse_values(io.BytesIO(raw),h,indices,ReadBudget())
    for ix in indices:self.assertEqual(values[ix],100*ix[0]+10*ix[1]+ix[2])
  h={'shape':[4,2],'order':'F','itemsize':1,'format':'<B'};indices=list(itertools.product(range(3),(0,1)))
  for future in [0,2,255]:
   raw=bytes([1,1,1,future,1,1,1,future]);values,a=sparse_values(io.BytesIO(raw),h,indices,ReadBudget())
   self.assertEqual(a['numeric_span_bytes'],7);self.assertEqual(a['discarded_bytes'],1);self.assertEqual(list(values.values()),[1]*6)
 def test_exact_cap_one_short_invalid_requests_truncation(self):
  h={'shape':[4,2],'order':'F','itemsize':1,'format':'<B'};ix=list(__import__('itertools').product(range(3),(0,1)))
  sparse_values(io.BytesIO(bytes(7)),h,ix,ReadBudget(7));f=io.BytesIO(bytes(7))
  with self.assertRaisesRegex(ValueError,'BEFORE'):sparse_values(f,h,ix,ReadBudget(6))
  self.assertEqual(f.tell(),0)
  for bad in [[],[(0,0),(0,0)],[(4,0)],[(True,0)]]:
   with self.assertRaises(ValueError):sparse_values(io.BytesIO(bytes(8)),h,bad,ReadBudget())
  b=ReadBudget()
  with self.assertRaises(ValueError):sparse_values(io.BytesIO(bytes(6)),h,ix,b)
  self.assertEqual(b.returned_bytes,6)
 def test_six_member_envelope_before_any_payload(self):
  for extra in ['secret.npy','../escape.npy','trajs_2d.npy']:
   f=io.BytesIO()
   with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    with zipfile.ZipFile(f,'w')as z:
     for name in [k+'.npy'for k in SCHEMA]+[extra]:z.writestr(name,b'NO NPY HEADER OR DATA')
   f.seek(0)
   with self.assertRaises(ValueError):read_inputs(f,ReadBudget(),{},lambda:None)
 def test_bad_header_schema_final_member_state(self):
  f=io.BytesIO();descr=repr({'descr':'<f4','fortran_order':False,'shape':(4,2,2)}).encode();payload=b'\x93NUMPY\x01\x00'+struct.pack('<H',len(descr))+descr+bytes(64)
  with zipfile.ZipFile(f,'w')as z:
   for name in SCHEMA:z.writestr(name+'.npy',payload)
  f.seek(0);progress={}
  with self.assertRaises(ValueError):read_inputs(f,ReadBudget(),progress,lambda:None)
  self.assertEqual(progress['arrays']['trajs_2d']['state'],'FAILED');self.assertTrue(progress['arrays']['trajs_2d']['counts_complete'])


class AnnotationTests(unittest.TestCase):
 def test_nonrigid_annotation_agreement_is_not_rigid_qualification(self):
  d,b,old=fixture();r={};check_points(d,b,old,r)
  self.assertEqual(r['outcome'],'COMPLETED_DESCRIPTIVE_WITH_ORIGINAL_RIGID_FAILURE')
  self.assertEqual((r['planned'],r['attempted'],r['unattempted'],r['projection_consistent_count'],r['valid_visible_annotation_support_count'],r['rigid_qualified_support_count']),(6,6,0,6,6,0))
  self.assertEqual(r['max_error_original_px'],0);self.assertEqual(r['rms_error_original_px'],0)
  text=json.dumps(r)
  for forbidden in ['stored_bytes_hex','projected_uv','world_coordinate','intrinsics','extrinsics','valid_visible_GT_support']:self.assertNotIn(forbidden,text)
 def test_visibility_separate_validity_stops_no_replacement(self):
  d,b,old=fixture()
  for k in d['visibs']:d['visibs'][k]=2
  r={};check_points(d,b,old,r);self.assertEqual(r['projection_consistent_count'],6);self.assertEqual(r['valid_visible_annotation_support_count'],0)
  d,b,old=fixture();d['valids'][(0,0)]=0;r={}
  with self.assertRaises(ValueError):check_points(d,b,old,r)
  self.assertEqual((r['attempted'],r['unattempted'],r['finite_error_count']),(1,5,0));self.assertIsNone(r['max_error_original_px'])
 def test_halfpixel_boundary_and_failed_finite_summary(self):
  d,b,old=fixture()
  for t,p in d['valids']:d['trajs_2d'][(t,p,0)]=-.5
  r={};check_points(d,b,old,r);self.assertEqual(r['projection_consistent_count'],6)
  d['trajs_2d'][(1,0,0)]=-1.;r={}
  with self.assertRaisesRegex(ValueError,'0.5'):check_points(d,b,old,r)
  self.assertEqual((r['attempted'],r['unattempted'],r['finite_error_count']),(3,3,3));self.assertEqual(r['max_error_original_px'],1)
  self.assertAlmostEqual(r['rms_error_original_px'],math.sqrt(.5));self.assertEqual(r['checks'][2]['state'],'FAILED')
 def test_byte_identity_proper_pass_and_bad_inputs_fail(self):
  for which in ['byte','pass','depth','nonfinite','overflow','visibility','row']:
   d,b,old=fixture()
   if which=='byte':b['K']=b'wrong'
   if which=='pass':
    for t,i,j in d['extrinsics']:
     if i==0 and j==0:d['extrinsics'][(t,i,j)]=1.
    b['RT']=camera_bytes(ident(4))
   if which=='depth':d['trajs_3d'][(0,0,2)]=0.
   if which=='nonfinite':d['trajs_2d'][(0,0,0)]=math.inf
   if which=='overflow':d['trajs_3d'][(0,0,0)]=1e308;d['trajs_3d'][(0,0,2)]=1e-308
   if which=='visibility':d['visibs'][(0,0)]=3
   if which=='row':d['intrinsics'][(0,2,0)]=1.
   r={}
   with self.assertRaises((ValueError,OverflowError)):check_points(d,b,old,r)
   self.assertEqual((r['attempted'],r['unattempted']),(1,5));self.assertEqual(r['checks'][0]['state'],'FAILED')


def load_runner():
 p=Path(__file__).resolve().parents[1]/'scripts/audit_0003_annotation_consistency_v1.py';s=importlib.util.spec_from_file_location('new_runner',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


class BindingTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.m=load_runner();self.g('init','-q');(self.root/'input.txt').write_text('original');self.g('add','.');self.commit('input');self.input=self.g('rev-parse','HEAD').decode().strip()
  self.r={'schema':1,'scope':'DEVELOPMENT_DESCRIPTIVE_ANNOTATION_SELF_CONSISTENCY_ONLY','input_commit':self.input,'inputs_sha256':{'input.txt':hashlib.sha256(b'original').hexdigest()}};self.write(self.r);self.g('add','.');self.commit('receipt');self.remote='refs/remotes/origin/review';self.g('update-ref',self.remote,self.g('rev-parse','HEAD').decode().strip())
 def tearDown(self):self.tmp.cleanup()
 def g(self,*a):return subprocess.check_output(['git','-C',str(self.root),*a],stderr=subprocess.DEVNULL)
 def commit(self,msg):self.g('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm',msg)
 def write(self,r):(self.root/'receipt.json').write_text(json.dumps(r))
 def verify(self):return self.m.verify_receipt(self.root,'receipt.json',{'input.txt'},self.remote)
 def test_initializer_current_index_and_input_commit_mismatch(self):
  name='src/evr/__init__.py';p=self.root/name;p.parent.mkdir(parents=True);p.write_text('SYNTHETIC_INITIALIZER=1\n')
  self.g('add',name);self.commit('synthetic initializer input');commit=self.g('rev-parse','HEAD').decode().strip();original=p.read_bytes()
  receipt='initializer-receipt.json';r={'schema':1,'scope':self.r['scope'],'input_commit':commit,'inputs_sha256':{name:hashlib.sha256(original).hexdigest()}}
  target=self.root/receipt;target.write_text(json.dumps(r));self.g('add',receipt);self.commit('initializer receipt');self.g('update-ref',self.remote,self.g('rev-parse','HEAD').decode().strip())
  verify=lambda:self.m.verify_receipt(self.root,receipt,{name},self.remote)
  self.assertEqual(verify()['input_commit'],commit)
  p.write_text('SYNTHETIC_INITIALIZER=2\n')
  with self.assertRaisesRegex(ValueError,'current input digest'):verify()
  self.g('add',name);p.write_bytes(original)
  with self.assertRaisesRegex(ValueError,'input blob/index'):verify()
  self.g('add',name)
  p.write_text('SYNTHETIC_INITIALIZER=3\n');r['inputs_sha256'][name]=hashlib.sha256(p.read_bytes()).hexdigest();target.write_text(json.dumps(r));self.g('add',name,receipt);self.commit('new initializer incorrectly bound to old commit');self.g('update-ref',self.remote,self.g('rev-parse','HEAD').decode().strip())
  with self.assertRaisesRegex(ValueError,'input blob/index'):verify()
 def test_actual_initializer_origin_and_hash(self):
  from types import SimpleNamespace
  repo=Path(__file__).resolve().parents[1];name='src/evr/__init__.py';digest=hashlib.sha256((repo/name).read_bytes()).hexdigest()
  self.assertIn(name,self.m.MANDATORY);self.m.verify_module(repo,self.m.evr,name,digest)
  other=self.root/'alternative.py';other.write_text('not the package initializer')
  with self.assertRaisesRegex(ValueError,'actual imported'):self.m.verify_module(repo,SimpleNamespace(__file__=str(other)),name,digest)
  with self.assertRaisesRegex(ValueError,'actual imported'):self.m.verify_module(repo,self.m.evr,name,'0'*64)
 def test_complete_actual_git_binding(self):self.assertEqual(self.verify()['input_commit'],self.input)
 def test_empty_missing_extra_scope_and_changed_receipt(self):
  for mapping in [{},{'extra':'0'*64},{'input.txt':'0'*64}]:
   r=copy.deepcopy(self.r);r['inputs_sha256']=mapping;self.write(r)
   with self.assertRaises(ValueError):self.verify()
  for key,value in [('scope','OTHER'),('schema',True)]:
   r=copy.deepcopy(self.r);r[key]=value;self.write(r)
   with self.assertRaises(ValueError):self.verify()
  self.write(self.r);(self.root/'receipt.json').write_text((self.root/'receipt.json').read_text()+' ')
  with self.assertRaises(ValueError):self.verify()
 def test_old_commit_new_hash_unsafe_path_and_index_remote(self):
  (self.root/'input.txt').write_text('changed');r=copy.deepcopy(self.r);r['inputs_sha256']['input.txt']=hashlib.sha256(b'changed').hexdigest();self.write(r);self.g('add','.');self.commit('incorrect old binding');self.g('update-ref',self.remote,self.g('rev-parse','HEAD').decode().strip())
  with self.assertRaisesRegex(ValueError,'input blob'):self.verify()
  for p in ['/tmp/example','../escape','a//b','a/./b']:
   with self.assertRaises(ValueError):self.m.regular(self.root,p)
  (self.root/'input.txt').write_text('indexed');self.g('add','input.txt');(self.root/'input.txt').write_text('changed')
  with self.assertRaises(ValueError):self.verify()
 def test_remote_self_reference_and_absent_real_receipt(self):
  self.g('update-ref',self.remote,self.input)
  with self.assertRaises(ValueError):self.verify()
  self.g('update-ref',self.remote,self.g('rev-parse','HEAD').decode().strip());r=copy.deepcopy(self.r);r['input_commit']=self.g('rev-parse','HEAD').decode().strip();self.write(r)
  with self.assertRaisesRegex(ValueError,'self-reference'):self.verify()
  cwd=Path.cwd()
  try:
   os.chdir(self.root)
   with self.assertRaises(ValueError):self.m.verify_published()
  finally:os.chdir(cwd)


class InterruptionTests(unittest.TestCase):
 def test_uncaught_stop_never_claims_no_access(self):
  repo=Path(__file__).resolve().parents[1]
  with tempfile.TemporaryDirectory()as directory:
   p=subprocess.run([sys.executable,str(repo/'scripts/audit_0003_annotation_consistency_v1.py'),'--cache','missing','--output','out'],cwd=directory,env={**os.environ,'PYTHONPATH':str(repo/'src')},capture_output=True,text=True)
   self.assertEqual(p.returncode,1);r=json.loads(p.stdout)
   self.assertFalse(r['counts_complete']);self.assertEqual(r['access_status'],'UNKNOWN_USE_ATOMIC_CHECKPOINT');self.assertEqual(r['state'],'INCOMPLETE')
 def test_atomic_start_survives_termination(self):
  code="from evr.point_geometry_v2 import atomic_json;import signal,sys;atomic_json(sys.argv[1],{'state':'NUMERIC_ACCESS_STARTING','counts_complete':False,'interruption_semantics':'unknown or lower bound'});print('READY',flush=True);signal.pause()"
  with tempfile.TemporaryDirectory()as d:
   path=Path(d)/'checkpoint.json';p=subprocess.Popen([sys.executable,'-c',code,str(path)],env={**os.environ,'PYTHONPATH':'src'},stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
   try:
    self.assertTrue(select.select([p.stdout],[],[],10)[0]);self.assertEqual(p.stdout.readline().strip(),'READY');p.terminate();p.wait(timeout=10)
    self.assertEqual(p.returncode,-signal.SIGTERM);r=json.loads(path.read_text());self.assertFalse(r['counts_complete']);self.assertEqual(r['state'],'NUMERIC_ACCESS_STARTING')
   finally:
    if p.poll()is None:p.kill();p.wait()
    p.stdout.close();p.stderr.close()

if __name__=='__main__':unittest.main()
