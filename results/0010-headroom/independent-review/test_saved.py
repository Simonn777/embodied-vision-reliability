"""Handwritten artificial saved-probability fixtures; no production/scientific calls."""
import copy,unittest
import check_saved as c

def fixture():
 ids=['%064x'%i for i in range(128)]
 rows=[]
 for i,q in enumerate(ids):
  rows.append(dict(id=q,batch=i//64,donor_id=ids[i//64*64+(i+1)%64],statuses={a:'ok' for a in c.ARMS},probabilities={a:[1.]+[0.]*9 for a in c.ARMS}))
 events=[]
 for b in range(2):
  group=ids[64*b:64*b+64]
  for j in range(66):
   excluded=None if j<2 else group[j-2];support=[q for q in group if q!=excluded];alpha=4 if j==1 else 1
   events.append(dict(status='completed',batch=b,excluded_id=excluded,alpha=alpha,support_n=len(support),support_ids_sha256=c.ih(support),tag='full-alpha'+str(alpha) if j<2 else 'delete-'+str(j-2)))
 return dict(status='completed',arms=list(c.ARMS),rows=rows,fit_states=events,successful_native_solver_executions=264,inputs={'ids_sha256':c.ih(ids)},costs_seconds=dict(zero_shot=.1,fits_with_parity=.2,one_way_readouts=.3,predictor_wall_before_final_save=1.))
def put(r,i,arm,k):r['rows'][i]['probabilities'][arm]=[float(j==k) for j in range(10)]

class TestSaved(unittest.TestCase):
 def test_eight_net_and_both_batches_no_population_claim(self):
  r=fixture()
  for i in [0,1,2,3,4,5,64,65,66,67,68,69]:
   for a in c.ARMS:put(r,i,a,1)
  for i in [0,1,2,3,64,65,66,67]:put(r,i,'loo_alpha1',0)
  out=c.reconstruct(r,[0]*128);s=out['summary']
  self.assertEqual(s['decision'],'CONTINUE_TO_MECHANISM');self.assertEqual(s['correct']['loo_alpha1'],124)
  self.assertEqual(s['strongest_fixed_controls'],list(c.CONTROLS));self.assertEqual(s['paired']['native_alpha1'],dict(n=128,corrections=8,harms=0,net=8,disagreements=8))
  self.assertEqual([b['zero_shot']['net'] for b in s['batches']],[4,4]);self.assertFalse(s['singleton_assessed']);self.assertEqual(len(out['argmax_by_arm']),5)
 def test_no_go_low_headroom_and_singleton_harm(self):
  r=fixture()
  for i in range(8):
   for a in c.CONTROLS:put(r,i,a,1)
  s=c.reconstruct(r,[0]*128)['summary'];self.assertEqual(s['decision'],'NO_GO');self.assertFalse(s['gates']['both_batches_positive'])
  self.assertEqual(c.reconstruct(fixture(),[0]*128)['summary']['decision'],'LOW_HEADROOM')
  r=fixture();y=[0]*128;y[63]=1
  for a in c.CONTROLS:put(r,63,a,1)
  for i in [0,1,2,3,4,64,65,66,67,68]:
   for a in c.CONTROLS:put(r,i,a,1)
  s=c.reconstruct(r,y)['summary'];self.assertEqual(s['paired']['native_alpha1']['net'],9);self.assertEqual(s['occupancy_strata']['one']['native_alpha1']['harms'],1);self.assertFalse(s['gates']['no_singleton_net_harm']);self.assertEqual(s['decision'],'NO_GO')
 def test_incomplete_no_label_access_and_corruption(self):
  class Bomb:
   def __iter__(self):raise AssertionError('labels accessed')
  self.assertEqual(c.reconstruct({'status':'incomplete'},Bomb()),{'status':'INCOMPLETE_UNSCORED'})
  for mutation in ['donor','fit','probability','argmax','count']:
   r=fixture()
   if mutation=='donor':r['rows'][0]['donor_id']=r['rows'][2]['id']
   elif mutation=='fit':r['fit_states'][2]['excluded_id']=r['rows'][1]['id']
   elif mutation=='probability':r['rows'][0]['probabilities']['loo_alpha1']=[float('nan')]*10
   elif mutation=='argmax':r['rows'][0]['probabilities']['loo_alpha1']=[.5,.5]+[0.]*8
   else:r['successful_native_solver_executions']=262
   with self.subTest(mutation=mutation),self.assertRaises(ValueError):c.reconstruct(r,[0]*128)

if __name__=='__main__':unittest.main(verbosity=2)
