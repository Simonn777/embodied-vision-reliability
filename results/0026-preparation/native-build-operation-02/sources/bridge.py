"""No I/O at import; separate direct fixed-support PoseLib TC refinement bridge."""
import ctypes as ct
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

ABI_VERSION = 260001
SETTINGS = dict(loss_type='TRUNCATED_CAUCHY', scale_reproj=1/64, weight_sampson=1.,
                refine_shift=True, gradient_tol=1e-12, step_tol=1e-8, relative_cost_tol=1e-10,
                initial_lambda=1e-3, min_lambda=1e-10, max_lambda=1e10,
                lambda_update='NIELSEN', damping='LEVENBERG', uniform_weights=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''): digest.update(part)
    return digest.hexdigest()


class NativeResult(ct.Structure):
    _fields_ = [(k, ct.c_uint64) for k in ('native_called','completed','iterations','invalid_steps')] + [
        ('R',ct.c_double*9),('q',ct.c_double*4),('t',ct.c_double*3)] + [
        (k,ct.c_double) for k in ('scale','shift1','shift2','initial_cost','cost','lambda','nu',
                                 'step_norm','grad_norm','loss_scale')] + [
        ('output_positive_depth1',ct.c_uint64),('output_positive_depth2',ct.c_uint64)]


class RefinementError(RuntimeError):
    def __init__(self, message, *, status=None, raw=None):
        super().__init__(message); self.status, self.raw = status, raw

    def as_record(self):
        nonfinite=[]
        def convert(value,path):
            if isinstance(value,float) and not math.isfinite(value):
                nonfinite.append({'path':path,'value':repr(value)}); return None
            if isinstance(value,list): return [convert(v,path+'/'+str(i)) for i,v in enumerate(value)]
            if isinstance(value,dict): return {k:convert(v,path+'/'+k) for k,v in value.items()}
            return value
        raw=convert(self.raw,'raw')
        return {'error':str(self),'status':self.status,'raw':raw,'nonfinite_values':nonfinite}


def raw_result(result):
    value = {}
    for key, _ in result._fields_:
        item = getattr(result,key)
        value[key] = list(item) if isinstance(item,ct.Array) else item
    return value


def checked_array(value, shape, name):
    a=np.asarray(value,dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all(): raise ValueError(name+' shape/nonfinite')
    return np.ascontiguousarray(a)


class NativeRefiner:
    def __init__(self, library_path, build_receipt_path):
        self.path=Path(library_path).resolve()
        receipt=json.loads(Path(build_receipt_path).read_text())
        here=Path(__file__).resolve().parent
        lock=json.loads((here/'input-lock.json').read_text())
        source_hashes={name:sha256(here/name) for name in ('build.py','refiner.cc','bridge.py','input-lock.json','NOTICE.md')}
        if (receipt.get('schema')!='evr0026.native-build.v1' or receipt.get('status')!='completed'
                or receipt.get('exit_code')!=0 or receipt.get('abi_version')!=ABI_VERSION
                or receipt.get('library_sha256')!=sha256(self.path)
                or receipt.get('bridge_python_sha256')!=sha256(__file__)
                or receipt.get('inputs')!=lock or receipt.get('source_sha256')!=source_hashes):
            raise ValueError('Native build receipt/binary/wrapper binding mismatch')
        self.library_sha256=receipt['library_sha256']
        self.lib=ct.CDLL(str(self.path))
        self.lib.nr_abi_version.argtypes=[]; self.lib.nr_abi_version.restype=ct.c_uint64
        self.lib.nr_result_size.argtypes=[]; self.lib.nr_result_size.restype=ct.c_uint64
        if self.lib.nr_abi_version()!=ABI_VERSION or self.lib.nr_result_size()!=ct.sizeof(NativeResult):
            raise ValueError('Native ABI version/struct size mismatch')
        p=ct.POINTER(ct.c_double)
        self.lib.nr_refine.argtypes=[p,p,p,p,ct.c_uint64,p,p]+[ct.c_double]*5+[
            ct.c_uint64,ct.POINTER(NativeResult),ct.POINTER(ct.c_char),ct.c_uint64]
        self.lib.nr_refine.restype=ct.c_int

    def refine(self,x1,x2,d1,d2,R,t,a,b,c,focal1,focal2,max_iterations=100):
        total=time.perf_counter()
        x1=np.asarray(x1,dtype=np.float64)
        if x1.ndim!=2 or x1.shape[1]!=2 or not 4<=len(x1)<=1000000:
            raise ValueError('x1 must have shape Nx2 with 4..1000000 rows')
        n=len(x1)
        arrays=[checked_array(v,s,k) for v,s,k in zip((x1,x2,d1,d2,R,t),
            ((n,2),(n,2),(n,),(n,),(3,3),(3,)),('x1','x2','d1','d2','R','t'))]
        values=[float(v) for v in (a,b,c,focal1,focal2)]
        if not all(math.isfinite(v) for v in values): raise ValueError('Nonfinite scalar input')
        if type(max_iterations) is not int or not 0<=max_iterations<=100:
            raise ValueError('max_iterations must be integer 0..100')
        # Physical input validity is checked again in C before the upstream call.
        ptr=[v.ctypes.data_as(ct.POINTER(ct.c_double)) for v in arrays]
        result=NativeResult(); error=ct.create_string_buffer(2048)
        start=time.perf_counter()
        status=self.lib.nr_refine(*ptr[:4],n,*ptr[4:],*values,max_iterations,ct.byref(result),error,len(error))
        native_seconds=time.perf_counter()-start
        raw=raw_result(result)
        if status!=0 or result.native_called!=1 or result.completed!=1:
            raise RefinementError(error.value.decode('utf8',errors='replace'),status=status,raw=raw)
        finite=all(math.isfinite(v) for k,v in raw.items() if isinstance(v,float)) and all(
            math.isfinite(v) for k in ('R','q','t') for v in raw[k])
        if not finite: raise RefinementError('Nonfinite native output',status=3,raw=raw)
        Rout=np.asarray(raw['R']).reshape(3,3)
        rotation_ok=(np.max(np.abs(np.einsum('ki,kj->ij',Rout,Rout,optimize=False)-np.eye(3)))<=1e-8
                     and abs(np.linalg.det(Rout)-1)<=1e-8)
        physical=(raw['scale']>0 and raw['output_positive_depth1']==n
                  and raw['output_positive_depth2']==n and max(abs(v) for v in raw['t'])>0 and rotation_ok)
        derived_c=raw['scale']*raw['shift2']
        if not math.isfinite(derived_c): raise RefinementError('Derived c overflow',status=3,raw=raw)
        return {'schema':'evr0026.native-refinement.v1','geometry':{
            'R':Rout.tolist(),'q':raw['q'],'t':raw['t'],'scale':raw['scale'],
            'shift1':raw['shift1'],'shift2':raw['shift2'],'a':raw['scale'],'b':raw['shift1'],'c':derived_c},
            'bundle_stats':{k:raw[k] for k in ('iterations','initial_cost','cost','lambda','nu','invalid_steps','step_norm','grad_norm')},
            'settings':{**SETTINGS,'loss_scale':raw['loss_scale'],'max_iterations':max_iterations},
            'output_checks':{'physical_admissible':bool(physical),'proper_rotation':bool(rotation_ok),
                'positive_depth1':raw['output_positive_depth1'],'positive_depth2':raw['output_positive_depth2'],
                'count':n,'native_called':True,'completed':True},
            'timings':{'native_call_seconds':native_seconds,'outer_call_seconds':time.perf_counter()-total},
            'library_sha256':self.library_sha256}
