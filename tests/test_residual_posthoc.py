"""Small standard-library fixtures for the post-hoc diagnostic, not model tests."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('residual_posthoc',Path(__file__).resolve().parents[1]/'scripts/analyze_0002_residuals.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def row(t,score=0.,tie=None,visible=True,error=0.,available=True):
 return {'archive_index':1,'point':0,'frame':t,'visible':visible,'coordinate_valid':True,'available':available,'error_px':error,'native':score,'augmented':score,'tie':tie or str(t)}

class ResidualPosthocTests(unittest.TestCase):
 def test_original_tie_controls_equal_scores(self):
  rows=[row(t,tie=tie)for t,tie in enumerate(['e','d','c','b','a'],1)]
  selected,order=m.matched_set(rows,'native')
  self.assertEqual([r['tie']for r in order],['a','b','c','d','e'])
  self.assertEqual(selected,{m.identity(r)for r in rows if r['tie']!='e'})
 def test_strict_four_pixel_failure(self):
  self.assertFalse(m.failed(row(1,error=4.)))
  self.assertTrue(m.failed(row(1,error=4.000001)))
 def test_frame_zero_and_occluded_not_opportunities(self):
  rows=[row(0),row(1,visible=False),row(2),row(3)]
  selected,_=m.matched_set(rows,'native')
  self.assertEqual(selected,{m.identity(rows[2]),m.identity(rows[3])})
 def test_available_target_is_not_lowered(self):
  with self.assertRaises(ValueError):m.matched_set([row(t,available=t<3)for t in range(1,6)],'native')
 def test_published_count_failure_gate(self):
  rows=[row(t,error=5. if t==1 else 0.)for t in range(1,6)];selected,_=m.matched_set(rows,'native')
  m.require_published(rows,selected,5,4,1)
  with self.assertRaises(ValueError):m.require_published(rows,selected,5,3,1)
 def test_operational_full_pair_boundary(self):
  rows=[row(1,tie='a'),row(2,tie='b'),row(3,tie='c')]
  self.assertEqual(m.operational_set(rows,'native',[0.,'b']),{m.identity(rows[0]),m.identity(rows[1])})
  self.assertEqual(m.operational_set(rows,'native',None),set())
 def test_history_keeps_rejected_raw_coordinates(self):
  # Frame2 may be rejected, invalid-logit or GT-occluded: none of these flags
  # is accepted as input to residuals. Dropping it would change d3 from0 to20.
  raw=[(0,0.,0.),(1,10.,0.),(2,30.,0.),(3,50.,0.)]
  got=m.residuals(raw);self.assertEqual(got[3]['residual_px'],0.);self.assertEqual(got[3]['history_frames'],[1,2])
  self.assertEqual(m.residuals([raw[i]for i in [0,1,3]])[3]['residual_px'],20.)
 def test_true_frame_gaps_and_missing_coordinate_skip(self):
  got=m.residuals([(0,0.,0.),(1,float('nan'),0.),(2,4.,0.),(5,10.,0.)])
  self.assertIsNone(got[1]['residual_px']);self.assertEqual(got[5]['residual_px'],0.);self.assertEqual(got[5]['history_frames'],[0,2])
 def test_initialization_and_penalty_cap(self):
  got=m.residuals([(0,0.,0.),(1,0.,0.),(2,100.,0.)]);self.assertTrue(got[0]['history_insufficient']);self.assertTrue(got[1]['history_insufficient']);self.assertFalse(got[2]['history_insufficient']);self.assertEqual(got[2]['penalty'],10.)
 def test_bad_frame_order_or_bool_refused(self):
  for points in [[(0,0.,0.),(0,1.,1.)],[(True,0.,0.)]]:
   with self.assertRaises(ValueError):m.residuals(points)
 def test_four_membership_cells(self):
  self.assertEqual([m.category(n,a)for n,a in [(True,True),(True,False),(False,True),(False,False)]],list(m.CATEGORIES))

if __name__=='__main__':unittest.main()
