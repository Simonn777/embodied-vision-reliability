"""Artificial file/receipt fixtures only; never call a training entrypoint."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/run_antithetic_training.py'
spec=importlib.util.spec_from_file_location('training_supervision_test',SCRIPT)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Supervision(unittest.TestCase):
    def test_bound_module_defeats_runtime_shadow(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'scripts').mkdir();shadow=root/'runtime';shadow.mkdir()
            p=root/'scripts/toy.py';p.write_text('ORIGIN="frozen"\n')
            (shadow/'toy.py').write_text('raise RuntimeError("runtime shadow executed")\n')
            with mock.patch.object(m,'ROOT',root),mock.patch.object(m,'REQUIRED',('scripts/toy.py',)),mock.patch.object(sys,'path',[str(shadow)]+sys.path):
                bound=m.bound_module('toy',{'files':{'scripts/toy.py':m.digest(p)}})
                self.assertEqual(bound.ORIGIN,'frozen')
                with self.assertRaisesRegex(RuntimeError,'not frozen'):m.bound_module('toy',{'files':{'scripts/toy.py':'0'*64}})
            sys.modules.pop('toy',None)

    def test_cache_allowlist_symlink_escape_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'input';p.write_bytes(b'input')
            (root/'reference.npz').mkdir() # no broad directory reading
            h=m.digest(p);self.assertEqual(m.cached_file(root,'input',h),p)
            (root/'link').symlink_to(p)
            for name in ('link','../escape',str(p)):
                with self.assertRaises(RuntimeError):m.cached_file(root,name,h)
            with self.assertRaisesRegex(RuntimeError,'hash differs'):m.cached_file(root,'input','0'*64)

    def test_full_commit_and_balanced_order(self):
        with self.assertRaisesRegex(RuntimeError,'full canonical'):m.verify_freeze('main')
        cfg={'cases':[{'index':i,'name':str(i)} for i in range(8)]}
        rows=m.operation_order(cfg)
        self.assertEqual(len(rows),32)
        for slot in range(4):
            self.assertEqual({method:sum(rows[i*4+slot][1]==method for i in range(8)) for method in m.METHODS},{method:2 for method in m.METHODS})

    def test_cost_allocation_charges_shared_and_rejects_bad_times(self):
        r={'input_seconds':.1,'fit':{'shared_setup_seconds':.2,'training_wall_seconds':7},'endpoints':{
            'updates500':{'training':{'update_completion_elapsed_seconds':2.99,'training_elapsed_seconds':3},'weights_write_seconds':.2,'restore_infer_save_seconds':1,'algorithm_pipeline_seconds':4.5},
            'wall5s':{'training':{'update_completion_elapsed_seconds':5.01,'training_elapsed_seconds':5.02},'weights_write_seconds':.2,'restore_infer_save_seconds':1,'algorithm_pipeline_seconds':6.5}}}
        self.assertAlmostEqual(m.allocated_outer_cost({'wall_seconds':12},r,'updates500'),6.8)
        self.assertAlmostEqual(m.allocated_outer_cost({'wall_seconds':12},r,'wall5s'),8.82)
        for elapsed in (8,float('nan')):
            bad=copy.deepcopy(r);bad['endpoints']['wall5s']['training']['training_elapsed_seconds']=elapsed
            with self.assertRaises(RuntimeError):m.allocated_outer_cost({'wall_seconds':12},bad,'wall5s')
        with self.assertRaises(RuntimeError):m.allocated_outer_cost({'wall_seconds':2},r,'wall5s')

    def test_complete_saved_grid_and_tamper_gates_before_reference(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cache=root/'cache';cache.mkdir();output=root/'out';output.mkdir()
            (output/'cases').mkdir();(output/'control').mkdir();(root/'protocols').mkdir()
            (root/m.FREEZE).write_text('{}')
            frozen={'source_commit':'a'*40,'files':{x:'b'*64 for x in m.REQUIRED}}
            args=SimpleNamespace(cache=cache,output=output,runtime_root=root/'runtime',freeze_commit='c'*40)
            cases=[];executions=[]
            for i in range(8):
                name=f'd{i//2+1}-colored-{i%2}';path=cache/(name+'.npz')
                np.savez(path,noisy=np.zeros((128,128),np.float32),covariance=np.eye(49,dtype=np.float64))
                cases.append({'index':i,'name':name,'group':f'd{i//2+1}','seed':i%2,'input_relative':path.name,'input_sha256':m.digest(path)})
            cfg={'cases':cases};inputs=m.check_inputs(cache,cfg)
            weights={name:np.zeros(shape,np.float32) for name,shape in {'0.weight':(64,49),'0.bias':(64,),'2.weight':(32,64),'2.bias':(32,),'4.weight':(1,32),'4.bias':(1,)}.items()}
            with mock.patch.object(m,'ROOT',root):
                m.write(output/'study.started.json',{'freeze_commit':args.freeze_commit,'source_commit':frozen['source_commit'],'freeze_sha256':m.digest(root/m.FREEZE),'inputs':inputs,'order':[m.case_name(c,method) for c,method in m.operation_order(cfg)]})
                for c,method in m.operation_order(cfg):
                    name=m.case_name(c,method);folder=output/'cases'/name;folder.mkdir();prefix=output/'control'/name
                    fit={'status':'completed','failure':None,'parameter_count':5313,'initialization_seed':620000+100*(c['index']//2)+10*(c['index']%2)+1,'clean_reference_accessed':False,'inference_calls':0,'numeric_threads':4,'interop_threads':1,'architecture':[49,64,32,1],'batch_size':512,'alpha':.5,'requested_updates':500,'wall_target_seconds':5.,'allowed_wall_overrun_seconds':.05,'torch_version':'2.10.0+cpu','numpy_version':'2.0.2','deterministic_algorithms':True,'optimizer':{'name':'Adam','lr':.001,'betas':[.9,.999],'eps':1e-8,'weight_decay':0.,'amsgrad':False,'foreach':False,'initial_state_entries':0},'completed_updates':1000,'optimizer_steps_executed':1000,'drawn_batches':1000,'initial_weight_sha256':'initial','extra_setup':{'noisy_sha256':'noisy'},'shared_setup_seconds':.2,'training_wall_seconds':5.02,'endpoint_receipts':{},'patches_sha256':'p','covariance_sha256':'c','factor_sha256':'f'}
                    r={'status':'completed','identity':c,'method':method,'source_commit':frozen['source_commit'],'freeze_commit':args.freeze_commit,'freeze_sha256':m.digest(root/m.FREEZE),'source_hashes':{k:v for k,v in frozen['files'].items() if k.startswith('scripts/')},'fit':fit,'endpoints':{},'files':{},'input_seconds':.1}
                    for endpoint,step,elapsed in [('updates500',500,1.),('wall5s',1000,5.01)]:
                        path=folder/(endpoint+'-weights.npz');np.savez(path,**weights);wh=m.saved_state_hash(np,path)
                        pred=np.ones((122,122),np.float32);np.save(folder/(endpoint+'-prediction.npy'),pred)
                        tr={'valid':True,'step':step,'update_completion_elapsed_seconds':elapsed-.001,'training_elapsed_seconds':elapsed,'state_dict_sha256':wh,'stream_sha256':{k:'shared' for k in ('base_indices','gaussian_prefix','step_seed_tuple')}}
                        inf={'status':'completed','failure':None,'finite_outputs':True,'weight_unchanged':True,'weight_content_sha256':wh,'weight_before_sha256':wh,'weight_after_sha256':wh,'forward_calls':750,'forward_calls_attempted':750,'standard_normal_draws':36465800,'mc_samples':50,'chunk_size':1024,'optimizer_steps':0,'direct_forward_calls':0,'clipping':'none','interpolation':'none','accumulation_dtype':'float64','alpha':.5,'architecture':[49,64,32,1],'output_shape':[122,122],'prediction_sha256':m.array_hash(pred),'clean_reference_accessed':False,'case_index':c['index'],'numeric_threads':4,'interop_threads':1,'torch_version':'2.10.0+cpu','numpy_version':'2.0.2','inference_seed':int.from_bytes(hashlib.sha256(f"evr0008|{c['index']}|inference|0".encode()).digest()[:8],'little')&((1<<63)-1),'patches_sha256':'p','covariance_sha256':'c','factor_sha256':'f','noisy_sha256':'noisy'}
                        fit['endpoint_receipts'][endpoint]=tr
                        r['endpoints'][endpoint]={'training':tr,'inference':inf,'inference_source_sha256':frozen['files']['scripts/antithetic_training_inference.py'],'weights_write_seconds':.01,'restore_infer_save_seconds':.1,'algorithm_pipeline_seconds':.1+.2+elapsed+.01+.1}
                        for suffix in ('-weights.npz','-prediction.npy'):r['files'][endpoint+suffix]=m.digest(folder/(endpoint+suffix))
                    m.write(folder/'receipt.json',r)
                    m.write(prefix.with_suffix('.command.private.json'),{'argv':m.expected_command(args,c,method),'cwd':str(root)})
                    for suffix in ('.stdout.log','.stderr.log'):prefix.with_suffix(suffix).write_bytes(b'')
                    ex={'actual_exit':0,'raw_wait_status':0,'hard_timeout':False,'wall_seconds':15,'receipt_sha256':m.digest(folder/'receipt.json'),'stdout_sha256':m.digest(prefix.with_suffix('.stdout.log')),'stderr_sha256':m.digest(prefix.with_suffix('.stderr.log')),'command_file_sha256':m.digest(prefix.with_suffix('.command.private.json'))}
                    m.write(prefix.with_suffix('.execution.json'),ex);executions.append({'case':name,**ex})
                complete={'source_commit':frozen['source_commit'],'freeze_commit':args.freeze_commit,'inputs_after':inputs,'cases':executions};m.write(output/'study.completed.json',complete)
                self.assertEqual(len(m.validate_evidence(args,cfg,frozen,np)),64)
                # Rebind hashes to prove semantic gates, not just hash differences.
                c,method=m.operation_order(cfg)[0];folder=output/'cases'/m.case_name(c,method);prefix=output/'control'/m.case_name(c,method)
                original=m.read(folder/'receipt.json')
                def rebind(r):
                    (folder/'receipt.json').write_text(json.dumps(r));ex=m.read(prefix.with_suffix('.execution.json'));ex['receipt_sha256']=m.digest(folder/'receipt.json')
                    prefix.with_suffix('.execution.json').write_text(json.dumps(ex));complete['cases'][0]={'case':m.case_name(c,method),**ex};(output/'study.completed.json').write_text(json.dumps(complete))
                for field,value in [('forward_calls',749),('prediction_sha256','x'),('weight_after_sha256','x'),('clean_reference_accessed',True)]:
                    bad=copy.deepcopy(original);bad['endpoints']['wall5s']['inference'][field]=value;rebind(bad)
                    with self.assertRaises(RuntimeError):m.validate_evidence(args,cfg,frozen,np)
                bad=copy.deepcopy(original);bad['endpoints']['wall5s']['algorithm_pipeline_seconds']+=1;rebind(bad)
                with self.assertRaisesRegex(RuntimeError,'formula differs'):m.validate_evidence(args,cfg,frozen,np)
                bad=copy.deepcopy(original);bad['input_seconds']=-.1
                for row in bad['endpoints'].values():row['algorithm_pipeline_seconds']-=.2
                rebind(bad)
                with self.assertRaisesRegex(RuntimeError,'timing component'):m.validate_evidence(args,cfg,frozen,np)
                rebind(original)
                command=m.read(prefix.with_suffix('.command.private.json'));command['argv'][-1]='different';prefix.with_suffix('.command.private.json').write_text(json.dumps(command))
                ex=m.read(prefix.with_suffix('.execution.json'));ex['command_file_sha256']=m.digest(prefix.with_suffix('.command.private.json'));prefix.with_suffix('.execution.json').write_text(json.dumps(ex));complete['cases'][0]={'case':m.case_name(c,method),**ex};(output/'study.completed.json').write_text(json.dumps(complete))
                with self.assertRaisesRegex(RuntimeError,'argv/cwd'):m.validate_evidence(args,cfg,frozen,np)

if __name__=='__main__':unittest.main()
