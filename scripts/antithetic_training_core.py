"""Prospective 0008 training-only core; no file I/O, inference or clean input.

Array/state hashes, covariance validation and MLP conventions are copied from
frozen 0007/0006. This separate module does not import either research runner.
Only private fixture hooks relax dimensions/budgets. Public API fixes the study.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

METHODS = ('independent', 'random_antithetic', 'mean_antithetic', 'same_patch')
STREAMS = ('indices', 'augmentation', 'pairing')
PARAMETER_NAMES = ('0.weight', '0.bias', '2.weight', '2.bias', '4.weight', '4.bias')
THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
CENTER = 24


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) < high:
        raise ValueError(name + ' is outside its integer range')
    return int(value)


def array_sha256(value):
    a = np.ascontiguousarray(value)
    header = json.dumps({'shape': list(a.shape), 'dtype': a.dtype.str}, sort_keys=True, separators=(',', ':')).encode('ascii')
    return hashlib.sha256(header + b'\0' + a.tobytes()).hexdigest()


def state_dict_sha256(state):
    h = hashlib.sha256(b'evr-mlp-state-v1\0')
    for name, value in sorted(state.items()):
        a = np.asarray(value)
        if not isinstance(name, str) or a.dtype != np.dtype('float32'):
            raise ValueError('weight names and float32 arrays required')
        a = np.ascontiguousarray(a).astype('<f4', copy=False)
        header = json.dumps({'name': name, 'shape': list(a.shape), 'dtype': '<f4'}, sort_keys=True, separators=(',', ':')).encode('ascii')
        payload = a.tobytes()
        h.update(len(header).to_bytes(8, 'little')); h.update(header)
        h.update(len(payload).to_bytes(8, 'little')); h.update(payload)
    return h.hexdigest()


def validate_covariance(covariance):
    c = np.asarray(covariance)
    if c.dtype != np.dtype('float64') or c.shape != (49,49) or not np.isfinite(c).all() or c[CENTER,CENTER] <= 0:
        raise ValueError('finite float64 [49,49] covariance with positive center variance required')
    if np.max(np.abs(c-c.T)) > 64*np.finfo(np.float64).eps*float(np.max(np.abs(c))):
        raise ValueError('asymmetric covariance; no repair')
    try:
        f = np.linalg.cholesky(c).astype(np.float32)
    except np.linalg.LinAlgError as exc:
        raise ValueError('positive-definite covariance required; no jitter') from exc
    if not np.isfinite(f).all():
        raise ValueError('factor overflow')
    f64 = f.astype(np.float64)
    effective = np.sum(f64[:,None,:]*f64[None,:,:], axis=2)
    return np.array(c,copy=True,order='C'), f, float(np.max(np.abs(effective-c)))


def extract_patches(noisy):
    y = np.asarray(noisy)
    if y.dtype != np.dtype('float32') or y.shape != (128,128) or not np.isfinite(y).all():
        raise ValueError('finite noisy float32 [128,128] required')
    return np.array(np.lib.stride_tricks.sliding_window_view(y,(7,7)).reshape(-1,49), dtype=np.float32, order='C', copy=True)


def step_seeds(case_index, step):
    case = _integer(case_index,'case_index',0,8)
    number = _integer(step,'step',1,100001)
    return {stream: int.from_bytes(hashlib.sha256(f'evr0008|{case}|{stream}|{number}'.encode('ascii')).digest()[:8], 'little') & ((1<<63)-1) for stream in STREAMS}


def init_seed(case_index):
    case = _integer(case_index,'case_index',0,8)
    return 620000 + 100*(case//2) + 10*(case%2) + 1


def _torch_runtime():
    for name in THREAD_ENV:
        os.environ[name] = '4'
    import torch
    torch.set_num_threads(4)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    return torch


def _new_model(torch, seed, widths):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = torch.nn.Sequential(
            torch.nn.Linear(widths[0],widths[1],device='cpu',dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(widths[1],widths[2],device='cpu',dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(widths[2],widths[3],device='cpu',dtype=torch.float32))
    model.train()
    return model


def _snapshot(model):
    state = {name: value.detach().cpu().numpy().copy() for name,value in model.state_dict().items()}
    if tuple(state) != PARAMETER_NAMES or any(not np.isfinite(a).all() for a in state.values()):
        raise FloatingPointError('invalid checkpoint state')
    # Snapshot arrays cannot accidentally change when optimizer updates continue.
    for a in state.values(): a.flags.writeable = False
    return state, state_dict_sha256(state)


def _batch(torch, patches, factor, method, batch_size, seeds):
    indices = torch.randint(patches.shape[0], (batch_size,), generator=torch.Generator(device='cpu').manual_seed(seeds['indices']), dtype=torch.int64)
    rows = patches[indices]
    order = torch.arange(batch_size,dtype=torch.int64)
    used = indices
    if method == 'same_patch':
        rows = rows[:batch_size//2].repeat_interleave(2,dim=0)
        used = indices[:batch_size//2].repeat_interleave(2)
    elif method == 'random_antithetic':
        order = torch.randperm(batch_size,generator=torch.Generator(device='cpu').manual_seed(seeds['pairing']))
        rows = rows[order]; used = indices[order]
    elif method == 'mean_antithetic':
        means = np.mean(rows.numpy(),axis=1,dtype=np.float64)
        order = torch.from_numpy(np.argsort(means,kind='stable').astype(np.int64))
        rows = rows[order]; used = indices[order]
    count = batch_size if method == 'independent' else batch_size//2
    epsilon = torch.randn((count,49), generator=torch.Generator(device='cpu').manual_seed(seeds['augmentation']), dtype=torch.float32, device='cpu')
    noise = epsilon @ factor.T
    if method != 'independent': noise = torch.stack((noise,-noise),dim=1).reshape(batch_size,49)
    return rows+.5*noise, rows[:,CENTER]-2*noise[:,CENTER], indices, used, order, epsilon


def _fit_patches(patches, covariance, method, *, case_index, widths=(49,64,32,1),
                 update_limit=500, wall_target=5.0, wall_overrun=.05, batch_size=512,
                 max_steps=100000, max_case_seconds=90.0, clock=time.monotonic,
                 overall_started=None, extra_setup=None):
    """Private tiny-fixture interface; caller must never weaken public settings."""
    started = clock() if overall_started is None else overall_started
    case = _integer(case_index,'case_index',0,8)
    update_limit = _integer(update_limit,'update_limit',1,100001)
    batch_size = _integer(batch_size,'batch_size',2,513)
    max_steps = _integer(max_steps,'max_steps',1,100001)
    if method not in METHODS or batch_size%2 or widths[0]!=49 or widths[-1]!=1:
        raise ValueError('invalid method, even batch or architecture')
    if any(not np.isfinite(v) or v <= 0 for v in (wall_target,max_case_seconds)) or not np.isfinite(wall_overrun) or wall_overrun<0:
        raise ValueError('invalid timing bounds')
    p = np.asarray(patches)
    if p.ndim!=2 or p.shape[1]!=49 or p.shape[0]==0 or p.dtype!=np.dtype('float32') or not np.isfinite(p).all():
        raise ValueError('finite nonempty float32 [N,49] patches required')
    c,f,rounding = validate_covariance(covariance)
    torch = _torch_runtime()
    model = _new_model(torch,init_seed(case),widths)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,betas=(.9,.999),eps=1e-8,weight_decay=0.0,amsgrad=False,foreach=False)
    if optimizer.state: raise RuntimeError('optimizer is not fresh')
    initial, initial_hash = _snapshot(model)
    del initial
    pt = torch.from_numpy(np.array(p,copy=True,order='C')); ft = torch.from_numpy(f)
    digests = {name:hashlib.sha256() for name in ('base_indices','used_indices','pair_order','gaussian_prefix','step_seed_tuple')}
    endpoints = {}; losses = []; completed_steps = 0; optimizer_steps = 0; drawn_batches = 0; failure = None; last_loss = None
    receipt = {'schema':'0008-training-core-v1','method':method,'case_index':case,'initialization_seed':init_seed(case),
        'initial_weight_sha256':initial_hash,'architecture':list(widths),'parameter_count':sum(a.numel() for a in model.parameters()),
        'alpha':.5,'batch_size':batch_size,'sampling':'with-replacement-occurrences','dtype':'float32','device':'cpu',
        'optimizer':{'name':'Adam','lr':1e-3,'betas':[.9,.999],'eps':1e-8,'weight_decay':0.0,'amsgrad':False,'foreach':False,'initial_state_entries':0},
        'torch_version':str(torch.__version__),'numpy_version':np.__version__,'numeric_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'patches_sha256':array_sha256(p),'covariance_sha256':array_sha256(c),'factor_sha256':array_sha256(f),
        'factor_max_absolute_covariance_error':rounding,'requested_updates':update_limit,'wall_target_seconds':wall_target,'allowed_wall_overrun_seconds':wall_overrun,
        'max_steps':max_steps,'cooperative_case_seconds':max_case_seconds,'external_hard_timeout_required':True,
        'clean_reference_accessed':False,'inference_calls':0,'checkpoint_rule':'fixed updates and first completed update at/after wall target; no score selection',
        'seed_rule':"first8LE(SHA256(ASCII('evr0008|case|stream|step'))) & (2^63-1); decimal case/step, streams indices/augmentation/pairing",
        'audit_hash_rule':'concatenated per-update raw little-endian base/used/order int64, first B/2 standard-normal rows float32, three stream seeds uint64; all hashing inside training clock',
        'extra_setup':extra_setup}
    receipt['shared_setup_seconds'] = float(clock()-started)
    training_started = clock()
    final_state = None; final_hash = None; attempted_step = None

    def capture(step, trigger, snapshot=None):
        state, digest = _snapshot(model) if snapshot is None else snapshot
        checkpoint = {'state_dict_np':state,'receipt':{'step':step,'boundary_elapsed_seconds':trigger,'update_completion_elapsed_seconds':trigger,
            'state_dict_sha256':digest,'stream_sha256':{name:h.hexdigest() for name,h in digests.items()},'shared_snapshot':snapshot is not None}}
        checkpoint['receipt']['training_elapsed_seconds'] = float(clock()-training_started)
        return checkpoint

    try:
        for step in range(1,max_steps+1):
            attempted_step = step
            if clock()-started >= max_case_seconds: raise TimeoutError('cooperative case deadline')
            seeds = step_seeds(case,step)
            inputs,target,indices,used,order,epsilon = _batch(torch,pt,ft,method,batch_size,seeds)
            drawn_batches += 1
            # Same-sized prefix audit in every estimator; no free out-of-clock hashes.
            for key,value,dtype in (('base_indices',indices,'<i8'),('used_indices',used,'<i8'),('pair_order',order,'<i8'),('gaussian_prefix',epsilon[:batch_size//2],'<f4')):
                digests[key].update(value.numpy().astype(dtype,copy=False).tobytes())
            digests['step_seed_tuple'].update(b''.join(seeds[name].to_bytes(8,'little') for name in STREAMS))
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(inputs).squeeze(-1),target,reduction='mean')
            if not bool(torch.isfinite(loss)): raise FloatingPointError('nonfinite loss')
            loss.backward()
            if any(q.grad is None or not bool(torch.isfinite(q.grad).all()) for q in model.parameters()): raise FloatingPointError('nonfinite or missing gradient')
            optimizer.step()
            optimizer_steps += 1
            if any(not bool(torch.isfinite(q).all()) for q in model.parameters()): raise FloatingPointError('nonfinite weight')
            completed_steps = step
            last_loss = float(loss.detach())
            if step%50==0: losses.append({'step':step,'mean_batch_loss':last_loss})
            # Lock the optimizer-update boundary BEFORE any endpoint copy.
            # A 500-step copy crossing the time target cannot relabel that update.
            update_completion_elapsed = float(clock()-training_started)
            if step==update_limit:
                endpoints['updates500'] = capture(step,update_completion_elapsed)
                endpoints['updates500']['receipt']['valid'] = True
            if 'wall5s' not in endpoints and update_completion_elapsed >= wall_target:
                shared = endpoints.get('updates500') if step==update_limit else None
                snapshot = None if shared is None else (shared['state_dict_np'],shared['receipt']['state_dict_sha256'])
                endpoints['wall5s'] = capture(step,update_completion_elapsed,snapshot)
                if shared is not None: shared['receipt']['shared_snapshot'] = True
                wall = endpoints['wall5s']['receipt']
                wall['overrun_seconds'] = wall['training_elapsed_seconds']-wall_target
                wall['valid'] = wall['training_elapsed_seconds'] <= wall_target+wall_overrun
                if not wall['valid']: raise TimeoutError('matched-time checkpoint exceeded allowed overrun')
            if clock()-started >= max_case_seconds: raise TimeoutError('cooperative case deadline')
            if len(endpoints)==2: break
        else:
            raise TimeoutError('maximum update count reached before both endpoints')
    except (FloatingPointError,RuntimeError,TimeoutError) as exc:
        failure = {'type':type(exc).__name__,'message':str(exc),'attempted_step':attempted_step}
    # Last completed endpoint snapshot can be reused; it is detached and read-only.
    latest = next((v for v in endpoints.values() if v['receipt']['step']==completed_steps),None) if optimizer_steps==completed_steps else None
    if latest is not None:
        final_state = latest['state_dict_np']; final_hash = latest['receipt']['state_dict_sha256']
    else:
        try: final_state,final_hash = _snapshot(model)
        except FloatingPointError: pass
    training_wall = float(clock()-training_started)
    if clock()-started >= max_case_seconds and failure is None:
        failure = {'type':'TimeoutError','message':'cooperative case deadline during final snapshot','attempted_step':attempted_step}
    status = 'completed' if failure is None and len(endpoints)==2 else 'failed'
    receipt.update({'status':status,'failure':failure,'completed_updates':completed_steps,'attempted_step':attempted_step,
        'loss_every_50':losses,'last_loss':last_loss,'training_wall_seconds':training_wall,'final_weight_sha256':final_hash,
        'drawn_batches':drawn_batches,'optimizer_steps_executed':optimizer_steps,
        'stream_sha256':{name:h.hexdigest() for name,h in digests.items()},
        'gaussian_rows_per_update':batch_size if method=='independent' else batch_size//2,
        'gaussian_prefix_rows_hashed_per_update':batch_size//2,
        'endpoint_receipts':{key:value['receipt'] for key,value in endpoints.items()}})
    serialization_started = clock()
    json.dumps(receipt,allow_nan=False)
    receipt['receipt_serialization_seconds'] = float(clock()-serialization_started)
    receipt['total_seconds'] = float(clock()-started)
    if receipt['total_seconds'] >= max_case_seconds and status == 'completed':
        status = 'failed'
        receipt['status'] = status
        receipt['failure'] = {'type':'TimeoutError','message':'cooperative case deadline during receipt serialization','attempted_step':attempted_step}
    return {'status':status,'endpoints':endpoints,'final_state_dict_np':final_state,'receipt':receipt}


def fit_trajectory(noisy,covariance,method,*,case_index):
    """Production only: 500 updates + first >=5s update (<=5.05s), one trajectory.

    The caller provides only noisy/C arrays, serializes locked endpoint states,
    and enforces 120s hard case / 900s study limits. No inference occurs here.
    """
    started=time.monotonic()
    patches=extract_patches(noisy)
    setup={'noisy_sha256':array_sha256(np.asarray(noisy)),'patch_extraction_seconds':float(time.monotonic()-started)}
    return _fit_patches(patches,covariance,method,case_index=case_index,overall_started=started,extra_setup=setup)
