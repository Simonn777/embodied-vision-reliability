"""Manufactured ABI/refinement qualification only; never reads an image/cache/GT.

Explicit --library and --build-receipt are mandatory when executed. Native tests
are prepared in advance and must not be mistaken for already executed checks.
"""
import argparse
import copy
import ctypes as ct
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np
import bridge

BACKEND=None


def fixture():
    # Fully fixed, nonplanar camera-1 points; no RNG or scientific observations.
    j=np.arange(48,dtype=float)
    X=np.column_stack((.8*np.sin(.7*j),.6*np.cos(.4*j),3.+.055*j+.15*np.sin(.3*j)))
    angle=.13
    R=np.array([[math.cos(angle),0.,math.sin(angle)],[0.,1.,0.],[-math.sin(angle),0.,math.cos(angle)]])
    t=np.array([.35,-.04,.08]);Y=np.einsum('ij,nj->ni',R,X,optimize=False)+t
    a,b,c=1.3,4.2,-.16
    return dict(x1=X[:,:2]/X[:,2,None],x2=Y[:,:2]/Y[:,2,None],d1=X[:,2]-b,
                d2=(Y[:,2]-c)/a,R=R,t=t,a=a,b=b,c=c,focal1=500.,focal2=650.)


def independent_cost(data):
    # Scalar residual formulas, including native inactive-Z/count convention.
    R,t=data['R'],data['t'];a,b,c=data['a'],data['b'],data['c']
    tx=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0.]])
    E=np.einsum('ij,jk->ik',tx,R,optimize=False)
    loss_scale=.5*(1/data['focal1']+1/data['focal2']);s2=loss_scale**2
    total=0.;count=0
    for x,y,d1,d2 in zip(data['x1'],data['x2'],data['d1'],data['d2']):
        r1=np.array([*x,1.]);r2=np.array([*y,1.]);Ex=E.dot(r1);Ety=E.T.dot(r2)
        denominator=sum(float(v)**2 for v in (*Ex[:2],*Ety[:2]))
        terms=[float(r2.dot(Ex))**2/denominator]
        forward=R.dot((d1+b)*r1)+t
        backward=R.T.dot(a*(d2+c/a)*r2-t)
        for point,target in ((forward,y),(backward,x)):
            if point[2]>0:
                terms.append(sum((float(point[k]/point[2])-float(target[k]))**2 for k in (0,1))/64.)
        total+=sum(s2*math.log1p(min(z/s2,1.)) for z in terms);count+=len(terms)
    return total/max(1,count),count


class WrapperArtificial(unittest.TestCase):
    def test_shape_and_nonfinite_rejected_before_ctypes(self):
        obj=object.__new__(bridge.NativeRefiner);obj.lib=mock.Mock()
        for change in ({'x2':np.zeros((47,2))},{'d1':np.full(48,np.nan)},
                       {'R':np.zeros((9,))},{'max_iterations':True}):
            data=fixture();data.update(change)
            with self.assertRaises(ValueError):obj.refine(**data)
        obj.lib.nr_refine.assert_not_called()

    def test_binary_source_and_dependency_tamper_rejected_before_dlopen(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);library=folder/'fake.dylib';library.write_bytes(b'not-a-library')
            here=Path(bridge.__file__).resolve().parent
            good={
                'schema':'evr0026.native-build.v1','status':'completed','exit_code':0,
                'abi_version':bridge.ABI_VERSION,'library_sha256':bridge.sha256(library),
                'bridge_python_sha256':bridge.sha256(bridge.__file__),
                'inputs':json.loads((here/'input-lock.json').read_text()),
                'source_sha256':{k:bridge.sha256(here/k) for k in ('build.py','refiner.cc','bridge.py','input-lock.json','NOTICE.md')}}
            for kind in ('binary','source','dependency'):
                altered=copy.deepcopy(good)
                if kind=='binary':altered['library_sha256']='0'*64
                elif kind=='source':altered['source_sha256']['refiner.cc']='0'*64
                else:altered['inputs']['compiler_sha256']='0'*64
                receipt=folder/'receipt.json';receipt.write_text(json.dumps(altered))
                with mock.patch.object(bridge.ct,'CDLL') as load:
                    with self.assertRaises(ValueError):bridge.NativeRefiner(library,receipt)
                    load.assert_not_called()

    def test_nonfinite_failure_retains_explicit_serializable_evidence(self):
        e=bridge.RefinementError('manufactured',status=3,raw={'t':[1.,float('nan'),float('inf')]})
        record=e.as_record();json.dumps(record,allow_nan=False)
        self.assertEqual(record['raw']['t'],[1.,None,None]);self.assertEqual(len(record['nonfinite_values']),2)


class NativeManufactured(unittest.TestCase):
    def test_zero_iteration_preserves_nonidentity_pose_and_affine_mapping(self):
        data=fixture();result=BACKEND.refine(**data,max_iterations=0);g=result['geometry']
        np.testing.assert_allclose(g['R'],data['R'],rtol=0,atol=2e-15)
        np.testing.assert_array_equal(g['t'],data['t'])
        self.assertEqual(g['scale'],data['a']);self.assertEqual(g['shift1'],data['b'])
        self.assertEqual(g['shift2'],data['c']/data['a'])
        self.assertEqual(result['bundle_stats']['iterations'],0)
        self.assertTrue(result['output_checks']['physical_admissible'])
        self.assertTrue(np.any(data['d1']<0))  # Raw negative depth is legal after positive affine correction.

    def test_independent_initial_cost_and_focal_scaling(self):
        for inactive in (False,True):
            data=fixture();data['t']=data['t']+np.array([.001,-.0005,.0004])
            if inactive:data['R']=np.diag([1.,-1.,-1.])
            expected,count=independent_cost(data);result=BACKEND.refine(**data,max_iterations=0)
            self.assertAlmostEqual(result['bundle_stats']['initial_cost'],expected,delta=1e-16+1e-10*abs(expected))
            self.assertEqual(result['settings']['loss_scale'],.5*(1/500.+1/650.))
            self.assertEqual(count,48 if inactive else 144)

    def test_only_supplied_fixed_support_is_used(self):
        data=fixture();data['t']=data['t']+np.array([.001,-.0005,.0004])
        for key in ('x1','x2','d1','d2'):data[key]=data[key][::3].copy()
        result=BACKEND.refine(**data,max_iterations=0)
        self.assertEqual(result['output_checks']['count'],16)
        self.assertAlmostEqual(result['bundle_stats']['initial_cost'],independent_cost(data)[0],delta=1e-16)

    def test_refinement_returns_all_stats_and_does_not_mutate_inputs(self):
        data=fixture();data['t']=data['t']+np.array([.001,-.0005,.0004])
        original={k:v.copy() for k,v in data.items() if isinstance(v,np.ndarray)}
        result=BACKEND.refine(**data);stats=result['bundle_stats']
        self.assertEqual(set(stats),{'iterations','initial_cost','cost','lambda','nu','invalid_steps','step_norm','grad_norm'})
        self.assertTrue(all(math.isfinite(float(v)) for v in stats.values()))
        self.assertLessEqual(stats['cost'],stats['initial_cost']);self.assertLessEqual(stats['iterations'],100)
        self.assertTrue(result['output_checks']['proper_rotation'])
        self.assertGreater(max(abs(v) for v in result['geometry']['t']),0.)
        for k,v in original.items():np.testing.assert_array_equal(data[k],v)

    def test_native_input_rejection_precedes_refiner(self):
        for change in ({'a':0.},{'b':-100.},{'c':-100.},{'t':np.zeros(3)},
                       {'R':np.diag([-1.,1.,1.])},{'focal1':-1.},{'focal1':1e-300}):
            data=fixture();data.update(change)
            with self.assertRaises(bridge.RefinementError) as raised:BACKEND.refine(**data)
            self.assertEqual(raised.exception.status,1)
            self.assertEqual(raised.exception.raw['native_called'],0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',required=True);parser.add_argument('--build-receipt',required=True)
    args=parser.parse_args();BACKEND=bridge.NativeRefiner(args.library,args.build_receipt)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromTestCase(WrapperArtificial),
        unittest.defaultTestLoader.loadTestsFromTestCase(NativeManufactured)]))
    sys.exit(0 if result.wasSuccessful() else 1)
