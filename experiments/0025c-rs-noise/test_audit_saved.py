"""Small artificial audit-contract checks; never generate the fixed cohort."""
import copy
import json
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import audit_saved as a


def report():
    return dict(failures=[],max_absolute_difference=0.,max_tolerance_ratio=0.)


def null_record(d,t):
    tn=math.hypot(*t);out={'translation_norm':tn}
    for name,r in [('numerical',d['rank']),('weak',d['weak'])]:
        N=d['vt'][r:].T;P=d['scales'][:,None]*N
        M=None if tn==0 else (np.eye(3)-np.outer(t/tn,t/tn))@P[:3]/tn
        out[name]=dict(nullity=N.shape[1],basis_scaled=N,basis_physical=P,
                       translation_direction_map=M,translation_direction_frobenius=None if M is None else math.hypot(*M.ravel()))
    return out


class Checks(unittest.TestCase):
    def test_series_exact_axis_and_zero(self):
        q,j=a.rotation_integral([0,0,0],.03)
        np.testing.assert_array_equal(q,np.eye(3));np.testing.assert_array_equal(j,.03*np.eye(3))
        w=.7;s=.2;c=math.cos(w*s);sn=math.sin(w*s)
        q,j=a.rotation_integral([0,0,w],s)
        np.testing.assert_allclose(q,[[c,-sn,0],[sn,c,0],[0,0,1]],atol=2e-16,rtol=0)
        np.testing.assert_allclose(j,[[sn/w,-(1-c)/w,0],[(1-c)/w,sn/w,0],[0,0,s]],atol=2e-16,rtol=0)

    def test_directed_endpoints_and_failure_mass(self):
        self.assertEqual(a.angular([1,0,0],[1,0,0]),0)
        self.assertEqual(a.angular([-1,0,0],[1,0,0]),180)
        m=a.group_metrics([dict(error=0,status='valid'),dict(error=2.5,status='valid'),dict(error=180,status='unavailable')])
        self.assertEqual(m['attempted'],3);self.assertEqual(m['useful_count'],2)
        self.assertEqual(m['empirical_cdf_integral_auc5_percent'],50.)
        self.assertEqual(m['recall']['180'],2/3)

    def test_svd_qr_and_rank_deficiency(self):
        A=np.array([[1.,2],[2,1],[3,-1]]);x=np.array([2.,-3.]);d=a.decomposition(A,A@x)
        self.assertEqual(d['rank'],2);np.testing.assert_allclose(d['x'],x,atol=1e-14)
        np.testing.assert_allclose(d['alternate'],x,atol=1e-14)
        singular=a.decomposition(np.array([[1.,1],[2,2]]),np.array([3.,6.]))
        self.assertEqual(singular['rank'],1);self.assertIsNone(singular['alternate'])

    def test_null_projector_sign_and_zero_functional(self):
        d=a.decomposition(np.zeros((8,9)),None);saved=null_record(d,np.zeros(3))
        for k in ('numerical','weak'):
            saved[k]['basis_scaled']=-saved[k]['basis_scaled'];saved[k]['basis_physical']=-saved[k]['basis_physical']
        r=report();a.null_check(saved,d,np.zeros(3),'null',r);self.assertEqual(r['failures'],[])
        saved['weak']['translation_direction_map']=np.zeros((3,9))
        with self.assertRaises(ValueError):a.null_check(saved,d,np.zeros(3),'null',report())

    def test_zero_translation_is_coverage_not_audit_failure(self):
        data=dict(R=np.eye(3),w1=np.zeros(3),w2=np.zeros(3),s1=np.zeros(8),s2=np.zeros(8),
                  r1=np.tile([0.,0.,1.],(8,1)),r2=np.tile([0.,0.,1.],(8,1)))
        A,y,g=a.form(data,list(range(8)),'N8');d=a.decomposition(A,y)
        s=dict(A=A,y=None,indices=list(range(8)),pool_size=8,column_norms=d['norms'],column_scales=d['scales'],singular_values=d['s'],rank_rtol=d['rtol'],weak_rtol=1e-8,
               rank=d['rank'],weak_rank=d['weak'],right_vectors_shape=[9,9],full_matrices=True,Vt=d['vt'],solution=d['x'],task_null=null_record(d,d['x'][:3]),
               status='unavailable',reason='undefined_translation',parameters=None)
        row=dict(arm='N8',indices='0-1-2-3-4-5-6-7',solver=s,case='artificial',regime='P',status='unavailable',reason='undefined_translation',raw_angle_degrees=None,error_degrees=180.)
        r=report();out=a.solve_check(data,row,dict(t=[1,0,0]),'zero',r)
        self.assertEqual(out['error'],180);self.assertEqual(r['failures'],[])
        row['error_degrees']=0
        with self.assertRaises(ValueError):a.solve_check(data,row,dict(t=[1,0,0]),'zero',report())

    def test_all_triangulations_required_even_with_majority(self):
        c=np.tile([0.,0.,1.],(8,1));b=c.copy();b[:,0]=1/np.arange(2.,10.)
        b[-1]=c[-1];zeros=np.zeros((8,3,3));g=(c,b,zeros,zeros);x=np.r_[1.,0.,0.,np.zeros(6)]
        saved=dict(required_positive=5,selected_sign=None,valid=False,all_triangulations_valid=False,hypotheses=[])
        for sign in (1,-1):
            points=[]
            for i in range(8):
                M=np.column_stack((c[i],-b[i]));d=a.decomposition(M,-sign*x[:3]);z=d['x'];regular=d['rank']==2
                points.append(dict(rank=d['rank'],depth1=z[0],depth2=z[1],singular_values=d['s'],residual_norm=math.hypot(*(M@z+sign*x[:3])),positive=bool(regular and np.all(z>0)),status='valid' if regular else 'rank_deficient'))
            saved['hypotheses'].append(dict(sign=sign,points=points,positive_count=sum(p['positive'] for p in points),all_triangulations_valid=False))
        r=report();self.assertIsNone(a.sign_checks(saved,g,x,'sign',r));self.assertEqual(r['failures'],[])
        self.assertEqual(saved['hypotheses'][0]['positive_count'],7)
        saved['selected_sign']=1;saved['valid']=True
        with self.assertRaises(ValueError):a.sign_checks(saved,g,x,'sign',report())

    def test_A2_six_parameter_columns(self):
        data=dict(R=np.eye(3),w1=np.zeros(3),w2=np.zeros(3),s1=np.zeros(8),s2=np.zeros(8),
                  r1=np.tile([.1,.2,1.],(8,1)),r2=np.tile([.2,.1,1.],(8,1)),d1=np.arange(2.,10.),d2=np.arange(3.,11.))
        A,y,_=a.form(data,[0,7],'A2');B,z,_=a.form(data,[0,7],'A8')
        self.assertEqual(A.shape,(6,6));np.testing.assert_array_equal(A,B);np.testing.assert_array_equal(y,z)

    def test_optimistic_A2_strict_integer_boundary(self):
        def rows(n,c):return [dict(error=0.,status='valid')]*c+[dict(error=180.,status='unavailable')]*(n-c)
        groups={(case,'S',arm):rows(n,c) for case in ('G','N','C') for arm,n,c in [('D4',1120,150),('A2',448,15)]}
        self.assertFalse(any(x['pass_gate'] for x in a.headroom_decisions(groups)))
        groups[('N','S','D4')]=rows(1120,151)
        decisions=a.headroom_decisions(groups);self.assertEqual([x['pass_gate'] for x in decisions],[False,True,False])
        groups[('N','S','A2')]=rows(447,15)
        self.assertFalse(any(x['pass_gate'] for x in a.headroom_decisions(groups)))

    def test_failed_outer_blocks_before_case_read(self):
        with tempfile.TemporaryDirectory() as td:
            outer=Path(td)/'receipt.json';outer.write_text(json.dumps(dict(schema='evr0025c.supervision.v1',operation='noise',expected_commit='a'*40,source_sha256={},status='failed')))
            fake=types.SimpleNamespace(OWNED=(),PROTOCOL='p',LEGACY='l',verify_frozen=lambda *x:{})
            args=types.SimpleNamespace(expected_commit='a'*40,input_dir=str(Path(td)/'missing'),outer_receipt=str(outer))
            with mock.patch.dict('sys.modules',{'supervise':fake}):
                with self.assertRaisesRegex(ValueError,'clean outer'):a.audit(args,report())

    def test_fixed_geometry_tolerance_and_json_duplicates(self):
        r=report();a.compare([0.],[2e-10],'geometry',r,geometry=True);self.assertEqual(r['failures'],['geometry'])
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.json';p.write_text('{"x":1,"x":2}')
            with self.assertRaisesRegex(ValueError,'duplicate'):a.load(p)


if __name__=='__main__':unittest.main()
