"""Manufactured driver coverage/leakage checks; no real images or pose backend."""
import unittest
from unittest import mock
from types import SimpleNamespace
import numpy as np
import study


def fixture(n=24):
    rng=np.random.default_rng(233)
    xyz=np.column_stack((rng.uniform(-.5,.5,n),rng.uniform(-.4,.4,n),rng.uniform(2.,4.,n)))
    names=['A0']+[p+str(k) for p in ('B','C') for k in range(1,9)]
    images={}; poses={}; by_camera={}
    median=float(np.median(xyz[:,2])) if n else 1.
    for index,name in enumerate(names):
        k=np.array([[400.,0.,320.+index/100.],[0.,400.,240.],[0.,0.,1.]])
        t=np.array([.015*index, .008*np.sin(index),0.])
        point=xyz+t
        xy=point[:,:2]/point[:,2,None]*400.+k[:2,2]
        images[name]=dict(xy=xy,K=k,width=640,height=480,depth=xyz[:,2]/median,
                          depth_valid=np.ones(n,dtype=bool))
        poses[name]=(np.eye(3),t);by_camera[k[0,2]]=name
    def edge(a,b):return dict(i=a,j=b,ids_i=np.arange(n),ids_j=np.arange(n),scores=np.ones(n))
    references=[edge('A0','C'+str(k)) for k in range(1,9)]
    for k in range(1,9):
        neighbors=sorted(range(1,9),key=lambda c:(abs(2*c-(2*k-1)),c))[:3]
        references += [edge('B'+str(k),'C'+str(c)) for c in neighbors]
    scene=dict(scene='manufactured',images=images,reference_edges=references,
               query_edges=[edge('A0','B'+str(k)) for k in range(1,9)])
    class Backend:
        def estimate_relative_pose(self,a,b,ka,kb,options):
            ia,ib=by_camera[ka['params'][2]],by_camera[kb['params'][2]]
            r,t=poses[ib];relative=t-poses[ia][1];relative=relative/np.linalg.norm(relative)
            return SimpleNamespace(R=r,t=relative),dict(inliers=np.ones(len(a),dtype=bool))
        def estimate_absolute_pose(self,xy,points,k,options):
            name=by_camera[k['params'][2]];r,t=poses[name]
            # Reference pair was normalized to unit length before triangulation.
            # Recover its common scale from this exact manufactured 3D support.
            ratios=np.linalg.norm(points,axis=1)/np.linalg.norm(xyz[:len(points)],axis=1)
            scale=float(np.median(ratios))
            return SimpleNamespace(pose=SimpleNamespace(R=r,t=t*scale)),dict(inliers=np.ones(len(xy),dtype=bool))
    return scene,Backend()


class StudyTests(unittest.TestCase):
    def test_missing_anchor_depth_does_not_disable_geometry_control(self):
        scene,backend=fixture()
        scene['images']['A0']['depth_valid'][:]=False
        result=study.run_scene(scene,backend,options=dict(max_accepted=2,max_evaluations=3))
        self.assertEqual(result['reference']['status'],'field_scale_unavailable')
        self.assertEqual(result['geometry_reference']['status'],'completed')
        self.assertTrue(any(row['arms']['geometry_multiview']['status']=='completed' for row in result['queries']))
        for row in result['queries']:
            self.assertEqual(set(row['arms']),set(study.ARMS))
            for name in study.ARMS:
                if name!='geometry_multiview':self.assertEqual(row['arms'][name]['status'],'fallback')

    def test_separate_pose_blocks_do_not_hide_missing_direction_at_large_scale(self):
        info=study.pose_information(np.delete(np.eye(6),3,axis=0),np.arange(6),np.array([0.,0.,1e12]))
        self.assertFalse(info['observable'])
        self.assertGreater(info['direction_null_ratio'],.1)

    def test_failed_support_preserves_all_queries_and_every_delete_population(self):
        scene,backend=fixture(0); events=[]
        result=study.run_scene(scene,backend,emit=lambda kind,value:events.append((kind,value)))
        self.assertEqual(len(result['queries']),8)
        self.assertEqual(len(result['deletions']),8)
        self.assertEqual(result['native_calls'],[])
        for k,fold in enumerate(result['deletions'],1):
            removed='C'+str(k)
            self.assertNotIn(removed,fold['image_ids'])
            self.assertTrue(all(removed not in (e['i'],e['j']) for e in fold['input_edges']))
        for row in result['queries']:
            self.assertEqual(set(row['arms']),set(study.ARMS))
            self.assertTrue(all(x['status']=='failed' and x['pose'] is None for x in row['arms'].values()))
        self.assertEqual(sum(k=='query' for k,_ in events),8)

    def test_native_technical_error_is_not_relabelled_scientific_failure(self):
        scene,backend=fixture()
        with mock.patch.object(backend,'estimate_relative_pose',side_effect=ValueError('bad API contract')):
            with self.assertRaisesRegex(ValueError,'bad API contract'):study.run_scene(scene,backend)

    def test_geometry_start_failure_preserves_survivor_and_all_failed_fallback(self):
        reference=dict(poses={'B1':(np.eye(3),np.array([2.,0.,0.]))})
        pair=(np.eye(3),np.array([0.,1.,0.]))
        fit=dict(poses={'B1':(np.eye(3),np.array([0.,2.,0.]))},cost=4.,invalid_count=0,
                 capped_count=0,observation_count=8,baseline_norm=1.,converged=False,
                 termination_reason='manufactured budget',accepted_steps=2,evaluations=3,seconds=.01)
        args=({}, {}, reference, {}, 'B1', pair, {})
        fail=study.geometry.InitializationFailure('manufactured invalid triangulation')
        with mock.patch.object(study.geometry,'fit_geometry',side_effect=[fail,fit]) as fitter, \
             mock.patch.object(study.geometry,'pose_observability',return_value={'observable':True}) as observer:
            pose,observable,detail=study.geometry_query(*args)
            self.assertEqual(fitter.call_count,2);observer.assert_called_once_with(fit,'B1')
            self.assertEqual(detail['selected'],1)
            self.assertEqual([a['status'] for a in detail['attempts']],['initialization_failure','completed'])
            np.testing.assert_array_equal(pose[1],[0.,2.,0.]);self.assertTrue(observable['observable'])
        with mock.patch.object(study.geometry,'fit_geometry',side_effect=[fail,fail]), \
             mock.patch.object(study.geometry,'pose_observability') as observer:
            pose,observable,detail=study.geometry_query(*args)
            self.assertIsNone(pose);self.assertIsNone(detail['selected']);observer.assert_not_called()
            self.assertEqual(len(detail['attempts']),2)
            native={'pose':pair};out=study.outcome(pose,observable,native,detail)
            self.assertEqual(out['status'],'fallback');self.assertIs(out['pose'],pair)
            self.assertEqual(study.outcome(pose,observable,None,detail)['status'],'failed')
        with mock.patch.object(study.geometry,'fit_geometry',side_effect=ValueError('broken contract')) as fitter:
            with self.assertRaisesRegex(ValueError,'broken contract'):study.geometry_query(*args)
            self.assertEqual(fitter.call_count,1)

    def test_manufactured_complete_graph_uses_only_own_query_and_all_folds(self):
        scene,backend=fixture(); events=[]
        result=study.run_scene(scene,backend,options=dict(max_accepted=2,max_evaluations=3),
                               emit=lambda kind,value:events.append((kind,value)))
        self.assertEqual(result['reference']['status'],'completed')
        self.assertTrue(all(f['status']=='completed' for f in result['deletions']))
        self.assertEqual(len(result['queries']),8)
        for row in result['queries']:
            query_edges=[e for e in row['query_graph']['retained_edges'] if e['i']=='A0' and e['j'].startswith('B')]
            self.assertEqual([e['j'] for e in query_edges],[row['query']])
            self.assertEqual(set(row['arms']),set(study.ARMS))
            for name in ('hessian','residual_scalar','scalar_deletion','directional'):
                self.assertEqual(row['arms'][name]['detail']['information_count'],dict(reference_prior=1,query_edge=1))
            self.assertEqual(len(row['deletion']['deltas']),8)
        self.assertFalse(result['ground_truth_access'])


if __name__=='__main__':unittest.main()
