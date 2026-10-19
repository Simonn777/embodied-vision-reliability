"""Control-flow/score fixtures only: no Torch import, actual StatA fit or encoder."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import headroom as h


def one_hot(k):
    p = np.zeros(10); p[k] = 1.; return p


class FakeBackend:
    kind = 'artificial_fake_fit'
    def __init__(self, fail_at=None):
        self.fits, self.native_calls, self.heldout_calls, self.zero_calls = [], [], [], []
        self.fail_at = fail_at
    def fit(self, x, p, ids, alpha):
        serial = len(self.fits)
        self.fits.append((tuple(ids), alpha, serial))
        if serial == self.fail_at: raise h.Invalid('fake_refusal')
        return dict(status='ok', ids=tuple(ids), alpha=alpha, serial=serial,
                    reference_bitwise_equal=True, native_solver_executions=2)
    def zero_shot(self, x, p):
        self.zero_calls.append(len(x)); return np.tile(one_hot(0), (len(x), 1))
    def native(self, state, q):
        assert q in state['ids']; self.native_calls.append((state['serial'], q))
        return one_hot((int(q, 16) + state['alpha']) % 10)
    def heldout(self, state, query, q):
        assert q not in state['ids']; self.heldout_calls.append((state['serial'], q))
        return one_hot(int(q, 16) % 10)


class Clock:
    def __init__(self): self.t = 0.
    def __call__(self): self.t += .001; return self.t


def fixture():
    x = np.zeros((128,512), dtype=np.float32)
    x[:,1] = np.arange(128)/1024.; x[:,0] = np.sqrt(1-x[:,1]**2)
    p = np.zeros((1,512,10), dtype=np.float32)
    for c in range(10): p[0,c,c] = 1
    return x,p,tuple('%064x' % i for i in range(128))


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='fake-headroom-', dir=Path(__file__).parent)
        cls.x,cls.p,cls.ids = fixture(); cls.backend = FakeBackend()
        cls.path = Path(cls.tmp.name)/'completed.json'
        cls.saved = h.predict(cls.x,cls.p,cls.ids,cls.backend,cls.path,clock=Clock())
    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()
    def fresh(self, name): return Path(self.tmp.name)/name
    def scored_record(self, predicted):
        r = copy.deepcopy(self.saved)
        for i in range(128):
            for m,a in enumerate(h.ARMS): r['rows'][i]['probabilities'][a] = one_hot(int(predicted[i,m])).tolist()
        return r
    def test_exact_132_states_and_no_repeated_deleted_fit(self):
        self.assertEqual(self.saved['status'], 'completed')
        self.assertEqual(len(self.backend.fits), 132)
        self.assertEqual(self.saved['successful_native_solver_executions'],264)
        deleted = [f for f in self.backend.fits if len(f[0]) == 63]
        self.assertEqual(len(deleted),128)
        self.assertEqual(len({f[0] for f in deleted}),128)
        self.assertEqual(sum(f[1] == 4 for f in self.backend.fits),2)
        self.assertEqual(self.backend.zero_calls,[64,64])
    def test_every_deleted_state_supplies_correct_loo_and_other(self):
        for b in range(2):
            bids = self.ids[b*64:(b+1)*64]
            for j,q in enumerate(bids):
                serial = b*66+2+j
                self.assertIn((serial,q),self.backend.heldout_calls)
                self.assertIn((serial,bids[(j-1)%64]),self.backend.native_calls)
                self.assertEqual([x for x in self.backend.native_calls if x[0] == serial],[(serial,bids[(j-1)%64])])
                self.assertEqual([x for x in self.backend.heldout_calls if x[0] == serial],[(serial,q)])
                self.assertNotIn(q,self.backend.fits[serial][0])
        self.assertEqual(len(self.backend.native_calls),384)
        self.assertEqual(len(self.backend.heldout_calls),128)
    def test_full_saved_contract_and_source_input_hashes(self):
        disk = json.loads(self.path.read_text()); self.assertEqual(disk,self.saved)
        self.assertEqual(len(disk['rows']),128)
        self.assertEqual(disk['inputs']['features_sha256'],h.array_hash(self.x))
        self.assertEqual(disk['inputs']['prototypes_sha256'],h.array_hash(self.p))
        for r in disk['rows']:
            self.assertTrue(all(r['statuses'][a] == 'ok' for a in h.ARMS))
            self.assertTrue(all(len(r['probabilities'][a]) == 10 for a in h.ARMS))
        self.assertEqual(disk['rows'][63]['donor_id'],self.ids[0])
        self.assertEqual(disk['rows'][127]['donor_id'],self.ids[64])
    def test_cost_categories_are_separate(self):
        cost=self.saved['costs_seconds']
        self.assertGreater(cost['fits_with_parity'],0)
        self.assertGreater(cost['one_way_readouts'],0)
        self.assertGreater(cost['zero_shot'],0)
        self.assertGreaterEqual(cost['predictor_wall_before_final_save'],sum(cost[k] for k in ('fits_with_parity','one_way_readouts','zero_shot')))
    def test_refusal_retains_all_rows_and_never_becomes_partial_success(self):
        backend=FakeBackend(fail_at=2)
        r=h.predict(self.x,self.p,self.ids,backend,self.fresh('refused.json'),clock=Clock())
        self.assertEqual(r['status'],'incomplete'); self.assertEqual(len(r['rows']),128)
        self.assertEqual(len(backend.fits),3); self.assertFalse(backend.heldout_calls)
        self.assertEqual(r['rows'][0]['statuses']['loo_alpha1'],'refused')
        self.assertEqual(r['rows'][63]['statuses']['other_out_alpha1'],'refused')
        self.assertEqual(r['rows'][64]['statuses']['zero_shot'],'not_attempted')
        self.assertEqual(r['fit_states'][-1]['status'],'failed')
        class Bomb:
            def __array__(self,*args): raise AssertionError('labels must not be examined')
            def __iter__(self): raise AssertionError('label IDs must not be examined')
        with self.assertRaisesRegex(h.Invalid,'unscorable_incomplete'): h.score(r,Bomb(),Bomb())
    def test_bad_probability_refuses_without_fallback(self):
        backend=FakeBackend()
        backend.zero_shot=lambda x,p: np.full((64,10),.1)
        r=h.predict(self.x,self.p,self.ids,backend,self.fresh('tie.json'),clock=Clock())
        self.assertEqual(r['status'],'incomplete'); self.assertFalse(backend.fits)
        self.assertEqual(r['error'],'probability_tie')
        self.assertTrue(all(v is None for row in r['rows'] for v in row['probabilities'].values()))
    def test_no_overwrite(self):
        before=self.path.read_bytes(); backend=FakeBackend()
        with self.assertRaisesRegex(h.Invalid,'fresh_output'): h.predict(self.x,self.p,self.ids,backend,self.path)
        self.assertFalse(backend.fits); self.assertEqual(self.path.read_bytes(),before)
    def test_invalid_shape_ids_dtype_refuse_before_fake_fit(self):
        for x,p,ids in [(self.x[:64],self.p,self.ids[:64]),(self.x.astype('float64'),self.p,self.ids),(self.x,self.p,('label-leak',)*128)]:
            backend=FakeBackend()
            with self.assertRaises(h.Invalid): h.predict(x,p,ids,backend,self.fresh('invalid.json'))
            self.assertFalse(backend.fits)
    def test_paired_counts_and_fixed_gate_arithmetic(self):
        pred=np.zeros((128,5),dtype=int)
        # All controls wrong on 12 rows; LOO repairs four in each batch.
        bad=[0,1,2,3,4,5,64,65,66,67,68,69]
        pred[bad,:]=1
        pred[[0,1,2,3,64,65,66,67],3]=0
        out=h.score(self.scored_record(pred),np.zeros(128,dtype=int),self.ids)
        self.assertEqual(out['decision'],'CONTINUE_TO_MECHANISM')
        self.assertEqual(out['correct']['loo_alpha1'],124)
        self.assertEqual(out['correct']['zero_shot'],116)
        self.assertEqual(out['paired']['native_alpha1'],dict(n=128,corrections=8,harms=0,net=8,disagreements=8))
        self.assertEqual([b['native_alpha1']['net'] for b in out['batches']],[4,4])
        self.assertFalse(out['singleton_assessed'])
    def test_aggregate_gain_without_group_consistency_is_no_go(self):
        pred=np.zeros((128,5),dtype=int); pred[:8,:]=1; pred[:8,3]=0
        out=h.score(self.scored_record(pred),np.zeros(128,dtype=int),self.ids)
        self.assertTrue(out['gates']['net_eight_over_best'])
        self.assertFalse(out['gates']['both_batches_positive']); self.assertEqual(out['decision'],'NO_GO')
    def test_stronger_alpha_control_prevents_oracle_claim(self):
        pred=np.zeros((128,5),dtype=int); pred[:12,:]=1; pred[:8,3]=0; pred[:8,2]=0
        out=h.score(self.scored_record(pred),np.zeros(128,dtype=int),self.ids)
        self.assertEqual(out['strongest_fixed_controls'],['native_alpha4'])
        self.assertFalse(out['gates']['net_eight_over_best']); self.assertEqual(out['decision'],'LOW_HEADROOM')
    def test_singleton_harm_is_not_hidden_in_aggregate(self):
        labels=np.zeros(128,dtype=int); labels[63]=1
        pred=np.tile(labels[:,None],(1,5)); bad=[0,1,2,3,4,64,65,66,67,68]
        pred[bad,:]=1; pred[bad,3]=0; pred[63,3]=0
        out=h.score(self.scored_record(pred),labels,self.ids)
        self.assertEqual(out['paired']['native_alpha1']['net'],9)
        self.assertTrue(out['singleton_assessed']); self.assertFalse(out['gates']['no_singleton_net_harm'])
        self.assertEqual(out['decision'],'NO_GO')
        self.assertEqual(out['occupancy_strata']['one']['native_alpha1']['harms'],1)
    def test_label_order_and_partial_or_corrupt_records_rejected(self):
        labels=np.zeros(128,dtype=int)
        with self.assertRaisesRegex(h.Invalid,'label_id_order'): h.score(self.saved,labels,self.ids[::-1])
        for change in ('status','row','donor','fit','probability','cost'):
            r=copy.deepcopy(self.saved)
            if change=='status': r['status']='incomplete'
            if change=='row': r['rows'].pop()
            if change=='donor': r['rows'][0]['donor_id']=self.ids[2]
            if change=='fit': r['fit_states'][2]['excluded_id']=self.ids[1]
            if change=='cost': r['costs_seconds']['fits_with_parity']=float('nan')
            if change=='probability': r['rows'][0]['probabilities']['loo_alpha1']=[float('nan')]*10
            with self.subTest(change=change),self.assertRaises(h.Invalid): h.score(r,labels,self.ids)
    def test_save_scoring_is_separate_and_exclusive(self):
        out=self.fresh('score.json')
        summary=h.score_file(self.path,np.zeros(128,dtype=int),self.ids,out)
        self.assertEqual(json.loads(out.read_text()),summary)
        with self.assertRaises(FileExistsError): h.score_file(self.path,np.zeros(128,dtype=int),self.ids,out)
    def test_no_torch_loaded_by_fixture_path(self):
        self.assertFalse(any(k=='torch' or k.startswith('torch.') for k in sys.modules))


if __name__=='__main__': unittest.main(verbosity=2)
