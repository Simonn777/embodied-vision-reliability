"""Manufactured fold isolation and saved-count boundary checks; no real inputs."""
import io
import unittest
import numpy as np
import core
import run


class WorkerBoundaryTests(unittest.TestCase):
    def test_held_clip_is_excluded_before_targets_are_passed(self):
        data = {name: dict(x=np.full((2,27),i,np.float32), y=np.full(2,i/4,np.float32),
                          w=np.full(2,.5,np.float64)) for i,(name,_) in enumerate(run.COHORT)}
        held = run.COHORT[1][0]
        first = run.fold_samples(data,held,np)
        data[held] = dict(x=np.full((99,27),np.nan,np.float32), y=None,w=None)
        second = run.fold_samples(data,held,np)
        self.assertNotIn(held, first[0])
        self.assertEqual(first[0], second[0])
        for a,b in zip(first[1:],second[1:]):
            np.testing.assert_array_equal(a,b)
        self.assertAlmostEqual(float(first[3].sum()),1.0)

    def test_all_empty_training_clips_remain_valid_empty_fold(self):
        data = {name: dict(x=np.empty((0,27),np.float32),y=np.empty(0,np.float32),
                          w=np.empty(0,np.float64)) for name,_ in run.COHORT}
        sources,x,y,w=run.fold_samples(data,run.COHORT[0][0],np)
        self.assertEqual(len(sources),3)
        self.assertEqual(x.shape,(0,27))
        self.assertEqual(len(y),len(w))

    def test_saved_scalar_decode_and_exact_original_pixel_score(self):
        truth=np.array([[1,1],[0,255]],np.uint8)
        b=np.array([[1,0],[0,0]],np.uint8)
        u=np.ones((2,2),np.uint8)
        controls=dict(zip(run.FIXED,(b,u,b,u)))
        counts=core.label_counts(truth,controls,np.array([[0,0],[1,1]],np.int64))
        memory=io.BytesIO()
        np.savez(memory,**{p+'__'+k:np.asarray(v) for p in run.POLICIES for k,v in counts[p].items()})
        memory.seek(0)
        with np.load(memory,allow_pickle=False) as f:
            packed={k:f[k] for k in f.files}
        for p in run.POLICIES:
            restored=run.unpack_counts(packed,p)
            self.assertIs(type(restored['total_gt']),int)
            score=core.score_actions(restored,np.array([True,False]),np.array([True,False]))
            self.assertEqual(score['J'],1.0)


if __name__=='__main__':
    unittest.main()
