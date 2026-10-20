"""Artificial aggregation/coverage tests; never generate the frozen cohort."""
import math
import unittest
import numpy as np
import run


def row(case='G',regime='P',arm='D4',status='valid',error=0.):
    return dict(case=case,regime=regime,arm=arm,status=status,error_degrees=error)


class AggregateTests(unittest.TestCase):
    def test_directed_sign_not_folded(self):
        self.assertEqual(run.angle(np.array([-1.,0,0]),np.array([1.,0,0])),180.)
        self.assertIsNone(run.angle(np.zeros(3),np.array([1.,0,0])))

    def test_unavailable_stays_in_denominator_and_not_recall(self):
        m=run.metrics([row(),row(status='unavailable',error=180.)])
        self.assertEqual(m['attempted'],2)
        self.assertEqual(m['available'],1)
        self.assertEqual(m['useful_yield'],.5)
        self.assertEqual(m['empirical_cdf_integral_auc5_percent'],50.)
        self.assertEqual(m['recall']['180'],.5)

    def test_exact_cdf_integral(self):
        m=run.metrics([row(error=0),row(error=2.5),row(error=5),row(error=180)])
        self.assertEqual(m['empirical_cdf_integral_auc5_percent'],37.5)

    def test_each_group_independently_required(self):
        rows=[]
        for c in ('G','N','C'):
            for r in ('P','S'):
                rows += [row(c,r,error=0.) for _ in range(144)]
                rows += [row(c,r,status='unavailable',error=180.) for _ in range(976)]
        rows += [row('G','S','A2',status='unavailable',error=180.) for _ in range(448)]
        self.assertTrue(run.summarize(rows,True)['utility_gate_pass'])
        rows[0]['error_degrees']=6.
        self.assertFalse(run.summarize(rows,True)['utility_gate_pass'])
        rows[0]['error_degrees']=0.
        self.assertFalse(run.summarize(rows,False)['utility_gate_pass'])
        self.assertFalse(run.summarize(rows[1:],True)['utility_gate_pass'])

    def test_critical_case_not_extra_gate(self):
        rows=[]
        for c in ('G','N','C'):
            for r in ('P','S'):
                rows += [row(c,r) for _ in range(1120)]
        rows += [row('E','S',status='unavailable',error=180.) for _ in range(1120)]
        rows += [row('G','S','A2',status='unavailable',error=180.) for _ in range(448)]
        self.assertTrue(run.summarize(rows,True)['utility_gate_pass'])

    def test_a2_optimistic_headroom_integer_boundary(self):
        rows=[]
        for c in ('G','N','C'):
            for r in ('P','S'):
                rows += [row(c,r) for _ in range(144)]
                rows += [row(c,r,status='unavailable',error=180.) for _ in range(976)]
            rows += [row(c,'S','A2') for _ in range(15)]
            rows += [row(c,'S','A2',status='unavailable',error=180.) for _ in range(433)]
        self.assertFalse(run.summarize(rows,True)['utility_gate_pass'])
        index=next(i for i,r in enumerate(rows) if r['case']=='G' and r['arm']=='A2')
        rows[index]['status']='unavailable';rows[index]['error_degrees']=180.
        result=run.summarize(rows,True)
        self.assertTrue(result['utility_gate_pass'])
        self.assertEqual(sum(x['pass_gate'] for x in result['optimistic_A2_headroom']),1)

    def test_json_no_nonfinite_tokens(self):
        v=run.safe({'x':np.array([1.,np.nan,np.inf]),'y':np.int64(2)})
        self.assertEqual(v,{'x':[1.,None,None],'y':2})

    def test_network_child_denied(self):
        for event in ('socket.connect','subprocess.Popen','os.system'):
            with self.assertRaises(RuntimeError):
                run.offline_guard(event,())
        self.assertIsNone(run.offline_guard('open',()))

if __name__=='__main__':
    unittest.main()
