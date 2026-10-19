"""Artificial graph/null-classification tests; no saved scientific outputs."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('review_failure_test',Path(__file__).with_name('review_failure.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def fixture():
    good={'q':[1.,0.,0.,0.],'t':[1.,0.,0.],'R':[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],'scale':1.,'shift1':0.,'shift2':0.}
    bad=copy.deepcopy(good);bad['q']=[None]*4
    cost={'numerator':0.,'count':1,'mean':0.}
    return {'events':[
        {'kind':'lo_begin','lo_id':0,'parent_kind':'proposal','parent_id':0,'state':bad},
        {'kind':'initial','lo_id':0,'stage':'proposal_lo','state':bad,'native_cost':cost,'finite':False},
        {'kind':'exit','lo_id':0,'stage':'proposal_lo','state':bad,'native_cost':cost,'finite':False},
        {'kind':'lo_begin','lo_id':1,'parent_kind':'proposal','parent_id':1,'state':good},
        {'kind':'initial','lo_id':1,'stage':'proposal_lo','state':good,'native_cost':cost,'finite':False,'nu':None},
        {'kind':'attempt','lo_id':1,'stage':'proposal_lo','state':bad,'native_cost':{'numerator':None,'count':1,'mean':None},'finite':False,'accepted':False},
        {'kind':'exit','lo_id':1,'stage':'proposal_lo','state':good,'native_cost':cost,'finite':True},
        {'kind':'pipeline','phase':'returned','source_kind':'lo','source_id':1,'state':good}]}


class FailureReviewTests(unittest.TestCase):
    def test_discarded_invalid_initial_still_breaks_complete_state_barrier(self):
        found=r.required_nonfinites(fixture())
        self.assertEqual(found['counts']['required_states'],4)
        self.assertEqual(found['counts']['nonfinite_required_states'],2)
        self.assertTrue(all(not x['on_final_ancestry'] for x in found['nonfinite_required_states']))
        self.assertEqual(found['counts']['auxiliary_nonfinite_only'],1)
        self.assertEqual(found['counts']['nonfinite_rejected_trials'],1)

    def test_graph_only_can_find_invalid_ancestor_without_evaluating_state(self):
        t=fixture();t['events'][3].update(parent_kind='lo',parent_id=0)
        self.assertEqual(r.graph_only(t),[1,0])
        self.assertTrue(all(x['on_final_ancestry'] for x in r.required_nonfinites(t)['nonfinite_required_states']))

    def test_cycle_is_rejected(self):
        t=fixture();t['events'][3].update(parent_kind='lo',parent_id=1)
        with self.assertRaises(ValueError):r.graph_only(t)

    def test_accepted_nonfinite_cost_is_required_not_rejected(self):
        t=fixture();t['events'][5]['accepted']=True
        found=r.required_nonfinites(t)
        self.assertEqual(found['counts']['nonfinite_required_states'],3)
        self.assertNotIn('nonfinite_rejected_trials',found['counts'])


if __name__=='__main__':unittest.main()
