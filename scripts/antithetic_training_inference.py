"""0008 locked-endpoint R2R MC50 inference; no optimizer or file I/O.

The wrapper must verify/load the frozen antithetic_training_core module first
and bind both source byte hashes. This module accepts no clean references.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import antithetic_training_core as core

SOURCE_IDENTIFIER = '0008-antithetic-training-inference-v1'
HELPER_IDENTIFIER = '0008-training-core-v1'


def inference_seed(case_index):
    case = core._integer(case_index,'case_index',0,8)
    return int.from_bytes(hashlib.sha256(f'evr0008|{case}|inference|0'.encode('ascii')).digest()[:8],'little') & ((1<<63)-1)


def _weights(state, widths=(49,64,32,1)):
    if not isinstance(state,dict) or set(state)!=set(core.PARAMETER_NAMES):
        raise ValueError('exact six named MLP state tensors required')
    result={}
    for layer,left,right in zip((0,2,4),widths[:-1],widths[1:]):
        for suffix,shape in (('weight',(right,left)),('bias',(right,))):
            name=str(layer)+'.'+suffix; a=np.asarray(state[name])
            if a.dtype!=np.dtype('float32') or a.shape!=shape or not np.isfinite(a).all():
                raise ValueError('invalid endpoint tensor: '+name)
            result[name]=np.array(a,copy=True,order='C')
    return result


def _mc_predict(torch,model,features,factor,*,seed,chunk_size,mc_samples,record):
    """Frozen0006 chunk-major then draw-major MC, without its extra direct pass."""
    generator=torch.Generator(device='cpu').manual_seed(seed)
    result=np.empty(features.shape[0],dtype=np.float32)
    with torch.inference_mode():
        for offset in range(0,features.shape[0],chunk_size):
            inputs=features[offset:offset+chunk_size]
            accumulated=torch.zeros(inputs.shape[0],dtype=torch.float64,device='cpu')
            for _ in range(mc_samples):
                epsilon=torch.randn(inputs.shape,generator=generator,dtype=torch.float32,device='cpu')
                record['standard_normal_draws']+=epsilon.numel()
                noise=epsilon @ factor.T
                record['forward_calls_attempted']+=1
                output=model(inputs+.5*noise)
                record['forward_calls']+=1
                if output.shape!=(inputs.shape[0],1) or output.dtype!=torch.float32 or output.device.type!='cpu':
                    raise RuntimeError('native pointwise output shape/dtype/device mismatch')
                if not bool(torch.isfinite(output).all()):
                    raise FloatingPointError('nonfinite MC prediction')
                accumulated+=output[:,0].to(torch.float64)
            output=(accumulated/mc_samples).to(torch.float32)
            if not bool(torch.isfinite(output).all()):
                raise FloatingPointError('nonfinite MC average')
            result[offset:offset+inputs.shape[0]]=output.numpy()
    return result


def _predict_patches(patches,covariance,state,*,case_index,widths=(49,64,32,1),
                     chunk_size=1024,mc_samples=50,output_shape=None,
                     clock=time.monotonic,overall_started=None,extra_setup=None):
    """Private artificial-fixture hook. Public settings cannot be overridden."""
    started=clock() if overall_started is None else overall_started
    case=core._integer(case_index,'case_index',0,8)
    chunk_size=core._integer(chunk_size,'chunk_size',1,1025)
    mc_samples=core._integer(mc_samples,'mc_samples',1,51)
    p=np.asarray(patches)
    if p.ndim!=2 or p.shape[0]==0 or p.shape[1]!=49 or p.dtype!=np.dtype('float32') or not np.isfinite(p).all():
        raise ValueError('finite nonempty float32 [N,49] patches required')
    if widths[0]!=49 or widths[-1]!=1:raise ValueError('49-input scalar-output MLP required')
    shape=(p.shape[0],) if output_shape is None else tuple(output_shape)
    if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in shape) or int(np.prod(shape))!=p.shape[0]:
        raise ValueError('invalid output shape')
    weights=_weights(state,widths)
    weight_hash=core.state_dict_sha256(weights)
    c,f,rounding=core.validate_covariance(covariance)
    torch=core._torch_runtime()
    model=core._new_model(torch,core.init_seed(case),widths)
    model.load_state_dict({k:torch.from_numpy(v) for k,v in weights.items()},strict=True)
    model.eval()
    before_hash=core._snapshot(model)[1]
    if before_hash!=weight_hash:raise RuntimeError('restored weights differ from supplied endpoint')
    features=torch.from_numpy(np.array(p,copy=True,order='C'));factor=torch.from_numpy(f)
    record={'schema':SOURCE_IDENTIFIER,'source_identifier':SOURCE_IDENTIFIER,'helper_identifier':HELPER_IDENTIFIER,
        'source_binding':'wrapper canonical freeze must attach actual inference/helper source byte hashes; module performs no file I/O',
        'case_index':case,'inference_seed':inference_seed(case),'seed_rule':"first8LE(SHA256(ASCII('evr0008|case|inference|0'))) & (2^63-1); unpadded decimal case",
        'architecture':list(widths),'parameter_count':sum(v.size for v in weights.values()),'parameter_names':list(core.PARAMETER_NAMES),
        'device':'cpu','dtype':'float32','accumulation_dtype':'float64','alpha':.5,'chunk_size':chunk_size,'mc_samples':mc_samples,
        'patch_count':p.shape[0],'output_shape':list(shape),'interpolation':'none','clipping':'none','direct_forward_calls':0,'optimizer_steps':0,
        'forward_calls':0,'forward_calls_attempted':0,'standard_normal_draws':0,
        'expected_forward_calls':((p.shape[0]+chunk_size-1)//chunk_size)*mc_samples,'expected_standard_normal_draws':p.shape[0]*49*mc_samples,
        'torch_version':str(torch.__version__),'numpy_version':np.__version__,'numeric_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'clean_reference_accessed':False,
        'noisy_sha256':None if extra_setup is None else extra_setup.get('noisy_sha256'),'patches_sha256':core.array_sha256(p),'covariance_sha256':core.array_sha256(c),'factor_sha256':core.array_sha256(f),
        'factor_max_absolute_covariance_error':rounding,'weight_content_sha256':weight_hash,'weight_before_sha256':before_hash,
        'extra_setup':extra_setup,'status':'completed','failure':None,'finite_outputs':False,'prediction_sha256':None}
    record['setup_seconds']=float(clock()-started)
    inference_started=clock();prediction=None
    try:
        prediction=_mc_predict(torch,model,features,factor,seed=record['inference_seed'],chunk_size=chunk_size,mc_samples=mc_samples,record=record).reshape(shape)
        record['finite_outputs']=bool(np.isfinite(prediction).all())
        record['prediction_sha256']=core.array_sha256(prediction)
        if record['forward_calls']!=record['expected_forward_calls'] or record['standard_normal_draws']!=record['expected_standard_normal_draws']:
            raise RuntimeError('incomplete inference call/draw grid')
    except (RuntimeError,FloatingPointError) as exc:
        record['status']='failed';record['failure']={'type':type(exc).__name__,'message':str(exc)};prediction=None
    record['inference_and_prediction_hash_seconds']=float(clock()-inference_started)
    try:after_hash=core._snapshot(model)[1]
    except FloatingPointError:after_hash=None
    record['weight_after_sha256']=after_hash;record['weight_unchanged']=after_hash==before_hash
    if not record['weight_unchanged']:
        record['status']='failed';record['failure']={'type':'WeightMutation','message':'model weights changed during inference'};prediction=None
    json.dumps(record,allow_nan=False)
    record['total_seconds']=float(clock()-started)
    return {'status':record['status'],'prediction':prediction,'receipt':record}


def predict_endpoint(noisy,covariance,state,*,case_index):
    """Fixed MC50/chunk1024 on122×122 valid centers; zero direct/inference extras.

    Returns status, float32 prediction[122,122] (None on failed inference), receipt.
    Invalid inputs fail before model execution. The caller enforces hard limits.
    """
    started=time.monotonic()
    patches=core.extract_patches(noisy)
    setup={'noisy_sha256':core.array_sha256(np.asarray(noisy)),'patch_extraction_seconds':float(time.monotonic()-started)}
    return _predict_patches(patches,covariance,state,case_index=case_index,output_shape=(122,122),overall_started=started,extra_setup=setup)
