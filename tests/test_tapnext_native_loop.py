"""Artificial inputs and a fake backend: no upstream model or real data."""
import ast,contextlib,hashlib,importlib.util,json,pathlib,sys,tempfile,time,types,zipfile,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]


def run_checks(OUT):
    src=(ROOT/'scripts/tapnext_native_adapter.py').read_text();tree=ast.parse(src)
    spec=importlib.util.spec_from_file_location('draft_adapter_review',ROOT/'scripts/tapnext_native_adapter.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    checks=[]
    def record(name,body,exception=None):
     try:
      detail=body()
     except Exception as e:
      if exception is None:raise
      checks.append({'name':name,'pass':isinstance(e,exception),'outcome':'rejected','error':str(e)});assert isinstance(e,exception)
     else:
      assert exception is None,(name,'unexpected acceptance')
      checks.append({'name':name,'pass':True,'outcome':'accepted','detail':detail})
    with tempfile.TemporaryDirectory(prefix='synthetic-',dir=OUT) as td:
     p=pathlib.Path(td)/'own.npz'
     video=np.zeros((250,256,256,3),np.uint8);video[:8,:,:,0]=11;video[:8,:,:,1]=71;video[:8,:,:,2]=233
     points=np.full((30,250,2),np.nan,dtype='>f8');points[:8,0]=np.arange(16).reshape(8,2)/32
     occ=np.ones((30,250),np.bool_);occ[:8,0]=False
     def write(pt=points,oc=occ,vi=video):np.savez(p,video=vi,points=pt,occluded=oc)
     write()
     def initial_only():
      fr,q=m.read_fixed_inputs(p,np);assert np.array_equal(fr,video[:8]);assert np.array_equal(q,points[:8,0].astype(np.float32)*256);return {'frames':list(fr.shape),'query_xy':q.tolist(),'future_gt_nan_ignored':True,'big_endian_float64_correct':True}
     record('asymmetric_RGB_big_endian_frame0_only',initial_only)
     q=points.copy();q[0,0,1]=np.nan;write(pt=q);record('nonfinite_initial_rejected',lambda:m.read_fixed_inputs(p,np),ValueError)
     write(pt=np.asfortranarray(points));record('Fortran_points_rejected',lambda:m.read_fixed_inputs(p,np),ValueError)
     oo=occ.copy();oo[0,0]=True;write(oc=oo);record('fixed_hidden_query_rejected',lambda:m.read_fixed_inputs(p,np),ValueError)
     np.savez_compressed(p,video=video,points=points,occluded=occ);record('compressed_members_rejected',lambda:m.read_fixed_inputs(p,np),ValueError)
     write();
     with zipfile.ZipFile(p,'a') as z:z.writestr('extra.npy',b'x')
     record('unknown_member_rejected',lambda:m.read_fixed_inputs(p,np),ValueError)
     del video,points,occ
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run_capability');tr=next(n for n in fn.body if isinstance(n,ast.Try))
    start=next(i for i,n in enumerate(tr.body)if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='captured')
    end=next(i for i,n in enumerate(tr.body)if isinstance(n,ast.Expr)and ast.unparse(n)=='hook.remove()')
    loop=compile(ast.Module(body=tr.body[start:end+1],type_ignores=[]),'VERBATIM_ADAPTER_LOOP','exec')
    class Tensor:
     def __init__(self,a):self.a=a;self.shape=a.shape;self.dtype=a.dtype;self.device=types.SimpleNamespace(type='cpu')
     def detach(self):return self
     def cpu(self):return self
     def numpy(self):return self.a
     def contiguous(self):return Tensor(np.ascontiguousarray(self.a))
    class FakeModel:
     def __init__(self,mode):self._model=self;self.mode=mode;self.calls=[];self.previous=None;self.cb=None;self.removed=False
     def register_forward_hook(self,cb):self.cb=cb;return types.SimpleNamespace(remove=lambda:setattr(self,'removed',True))
     def track_frame(self,frame_bgr,query_points_xy,state,autocast):
      t=len(self.calls);assert np.array_equal(frame_bgr[0,0],[233,71,11]);assert autocast is False
      assert state is self.previous
      if t==0:
       assert np.array_equal(query_points_xy,np.arange(16,dtype=np.float32).reshape(8,2));q=np.zeros((1,8,3),np.float32);q[0,:,1:]=query_points_xy[:,::-1];self.q=Tensor(q)
      else:assert query_points_xy is None
      self.calls.append({'frame':t,'state_identity_chained':True,'queries_only_frame0':t==0})
      if self.mode=='mutate_query'and t==1:self.q.a[0,0,2]+=1
      hidden=[types.SimpleNamespace(rg_lru_state=Tensor(np.broadcast_to(np.float32(0),(1032,768))),conv1d_state=Tensor(np.broadcast_to(np.float32(0),(1032,3,768))))for _ in range(12)]
      if self.mode=='nan_state':hidden[0].rg_lru_state=Tensor(np.broadcast_to(np.float32(np.nan),(1032,768)))
      if self.mode=='wrong_state_shape':hidden[0].rg_lru_state=Tensor(np.zeros((1,1),np.float32))
      new=types.SimpleNamespace(step=t+1,query_points=self.q,hidden_state=hidden)
      if self.mode=='wrong_step':new.step=t
      tr=np.arange(16,dtype=np.float32).reshape(1,1,8,2);co=np.zeros((1,1,8,512),np.float32);vi=np.arange(-4,4,dtype=np.float32).reshape(1,1,8,1)
      if self.mode=='nan_logits':co[0,0,0,0]=np.nan
      result=(Tensor(tr),Tensor(co),Tensor(vi),new);self.cb(self,None,result)
      if self.mode=='double_forward':self.cb(self,None,result)
      pos=tr[0,0,:,::-1].copy();visible=vi[0,0,:,0]>0
      if self.mode=='wrong_xy':pos=tr[0,0].copy()
      if self.mode=='wrong_visibility':visible=~visible
      self.previous=new
      return pos,visible,new
    loop_details={}
    def check_loop(mode):
     with tempfile.TemporaryDirectory(prefix='fake-loop-',dir=OUT)as td:
      model=FakeModel(mode);receipt={'forwards_attempted':0,'frames':[],'frames_completed':0};frames=np.zeros((8,256,256,3),np.uint8);frames[:]=[11,71,233]
      env={'np':np,'torch':types.SimpleNamespace(float32=np.dtype(np.float32),inference_mode=contextlib.nullcontext),'model':model,'frames':frames,'queries_xy':np.arange(16,dtype=np.float32).reshape(8,2),'receipt':receipt,'time':time,'hashlib':hashlib,'output':pathlib.Path(td),'sha':m.sha,'persist':lambda:None}
      try:exec(loop,env)
      finally:loop_details[mode]={'calls_observed':len(model.calls),'attempts_recorded':receipt['forwards_attempted'],'frames_completed':receipt['frames_completed']}
      assert len(model.calls)==8 and receipt['frames_completed']==8 and receipt['forwards_attempted']==8 and model.removed
      assert len(list(pathlib.Path(td).glob('frame-*.npz')))==8
      return loop_details[mode]
    record('verbatim_loop_exactly_8_fake_forwards_state_identity_BGR_xy',lambda:check_loop('normal'))
    for mode in ['mutate_query','nan_state','wrong_state_shape','wrong_step','nan_logits','double_forward','wrong_xy','wrong_visibility']:
     record('loop_rejects_'+mode,lambda mode=mode:check_loop(mode),ValueError)
    return checks, loop_details


class NativeDataAndLoopTests(unittest.TestCase):
    def test_fixed_input_boundary_and_native_loop_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            checks, details = run_checks(pathlib.Path(directory))
        self.assertEqual(len(checks), 15)
        self.assertTrue(all(item['pass'] for item in checks))
        self.assertEqual(details['normal']['frames_completed'], 8)
        self.assertEqual(details['mutate_query']['calls_observed'], 2)


if __name__ == '__main__':
    unittest.main()
