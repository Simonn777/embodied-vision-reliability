"""Manufactured arrays and injected APIs only; no PoseLib/model/data imports."""
import copy
import types
import unittest
import numpy as np
import geometry_ba as g
from field_ba import rotation_exp, CAP_COST


def fixture(n=20):
    rng = np.random.default_rng(237)
    points = np.column_stack((rng.uniform(-.8, .8, (n, 2)), rng.uniform(3., 6., n)))
    poses = {"A0": (np.eye(3), np.zeros(3)),
             "C1": (rotation_exp(np.array([.02, -.04, .01])), np.array([.8, .02, .05])),
             "B1": (rotation_exp(np.array([-.03, .01, .02])), np.array([.3, .5, .03]))}
    images = {}; k = np.array([[400., 0, 255.5], [0, 410., 191.5], [0, 0, 1.]])
    for key, (r, t) in poses.items():
        q = (r @ points.T).T+t
        xy = q[:, :2]/q[:, 2, None]*np.array([400., 410.])+k[:2, 2]
        images[key] = dict(xy=xy, K=k.copy(), width=512, height=384, depth=q[:, 2]/4.)
    edges = [dict(i=i, j=j, ids_i=np.arange(n), ids_j=np.arange(n), scores=np.linspace(1, .5, n))
             for i, j in [("A0", "C1"), ("C1", "B1")]]
    return images, edges, poses, points


class Backend:
    def __init__(self, poses):
        self.poses = poses; self.calls = []
    def estimate_relative_pose(self, a, b, ca, cb, options):
        self.calls.append(("relative", a.copy(), b.copy(), ca, cb, copy.deepcopy(options)))
        r, t = self.poses["C1"]
        return types.SimpleNamespace(R=r.copy(), t=t.copy()), {"inliers": [True]*len(a)}
    def estimate_absolute_pose(self, xy, xyz, camera, options):
        self.calls.append(("absolute", xy.copy(), xyz.copy(), camera, copy.deepcopy(options)))
        r, t = self.poses["B1"]
        return types.SimpleNamespace(pose=types.SimpleNamespace(R=r.copy(), t=t.copy())), {"inliers": [True]*len(xy)}


class GeometryTests(unittest.TestCase):
    def test_conflicts_determinism_and_reference_freeze(self):
        images, _, _, _ = fixture()
        def e(i,j,a,b,s):
            return dict(i=i,j=j,ids_i=np.array(a),ids_j=np.array(b),scores=np.array(s))
        edges = [e('A0','C1',[0,1],[0,1],[.9,.8]), e('C1','B1',[0,1],[0,0],[.7,.6])]
        a = g.build_tracks(images, edges); b = g.build_tracks(images, list(reversed(edges)))
        self.assertEqual(a['tracks'], b['tracks']); self.assertEqual(a['conflict_count'], 1)
        self.assertTrue(all(len(dict(t['observations'])) == len(t['observations']) for t in a['tracks']))
        query = e('A0','B1',[1],[0],[999.])
        ext = g.build_tracks(images, [query], frozen_reference=a)
        self.assertEqual(ext['conflict_count'], 1); self.assertEqual(ext['tracks'], a['tracks'])
        self.assertEqual(ext['match_count'], a['match_count'])

    def test_deduplicated_observation_mass_and_bad_ids(self):
        images, edges, _, _ = fixture()
        graph = g.build_tracks(images, edges+edges)
        self.assertEqual(graph['observation_count'], 60)
        self.assertEqual(graph['duplicate_count'], 40)
        bad = copy.deepcopy(edges); bad[0]['ids_i'][0] = -1
        with self.assertRaises(ValueError): g.build_tracks(images,bad)

    def test_native_interface_and_reference_ls_scale(self):
        images, edges, poses, points = fixture(); native = Backend(poses)
        # Invalid depth does not filter geometric inputs or enter scale alignment.
        images['A0']['depth_valid'] = np.ones(len(points), dtype=bool)
        images['A0']['depth_valid'][0] = False; images['A0']['depth'][0] = np.nan
        init = g.initialize_reference(images, edges, native)
        self.assertEqual([c[0] for c in native.calls], ['relative','absolute'])
        self.assertEqual(native.calls[0][-1], {'max_error':2.,'ransac':{'seed':0,'min_iterations':100,'max_iterations':1000}})
        np.testing.assert_array_equal(native.calls[0][1], images['A0']['xy'])
        self.assertAlmostEqual(init['receipt']['global_scale'], .25, places=12)
        self.assertTrue(init['depth_scale_available'])
        self.assertEqual(init['receipt']['scale_anchor_points'],19)
        self.assertEqual(init['gauge_image'],'C1'); self.assertEqual(init['receipt']['native_calls'],2)
        for key,(r,t) in poses.items():
            np.testing.assert_allclose(init['poses'][key][0],r,atol=1e-12)
            np.testing.assert_allclose(init['poses'][key][1],t*.25,atol=1e-12)
        for tr in init['graph']['tracks']:
            np.testing.assert_allclose(init['points'][tr['id']], points[dict(tr['observations'])['A0']]*.25, atol=1e-12)

    def test_missing_anchor_depth_preserves_geometric_unit_gauge(self):
        images,edges,poses,points=fixture(); native=Backend(poses)
        images['A0']['depth'][:]=np.nan
        images['A0']['depth_valid']=np.zeros(len(points),dtype=bool)
        init=g.initialize_reference(images,edges,native)
        self.assertFalse(init['depth_scale_available'])
        self.assertEqual(init['scale_status'],'unit_baseline_insufficient_anchor_depth')
        self.assertEqual(init['receipt']['scale_anchor_points'],0)
        self.assertEqual([call[0] for call in native.calls],['relative','absolute'])
        scale=1/np.linalg.norm(poses['C1'][1])
        self.assertAlmostEqual(init['baseline_norm'],1.,places=14)
        for key,(r,t) in poses.items():
            np.testing.assert_allclose(init['poses'][key][0],r,atol=1e-12)
            np.testing.assert_allclose(init['poses'][key][1],t*scale,atol=1e-12)
        for track in init['graph']['tracks']:
            expected=points[dict(track['observations'])['A0']]*scale
            np.testing.assert_allclose(init['points'][track['id']],expected,atol=1e-12)
        fit=g.fit_geometry(images,init['graph'],init,options={'max_accepted':2,'max_evaluations':3})
        self.assertLess(fit['cost'],1e-18)
        self.assertTrue(g.pose_observability(fit,'B1')['observable'])
        # A falsely declared valid NaN is still an input error, never fallback.
        images['A0']['depth_valid'][0]=True
        with self.assertRaisesRegex(ValueError,'valid anchor depth'):
            g.initialize_reference(images,edges,Backend(poses))

    def test_query_pair_never_requires_depth(self):
        images,edges,poses,_ = fixture()
        for item in images.values(): item.pop('depth')
        out=g.estimate_pair(images,edges[0],Backend(poses))
        self.assertEqual(out['input_count'],20)
        bad=Backend(poses); bad.estimate_relative_pose=lambda *a: (types.SimpleNamespace(R=np.eye(3),t=np.ones(3)),{'inliers':[1]*20})
        with self.assertRaises(ValueError): g.estimate_pair(images,edges[0],bad)

    def test_failed_initialization_and_pure_rotation(self):
        images,edges,poses,points=fixture()
        tiny=copy.deepcopy(edges); tiny[0]={k:(v[:7] if isinstance(v,np.ndarray) else v) for k,v in tiny[0].items()}
        with self.assertRaises(g.InitializationFailure): g.initialize_reference(images,tiny,Backend(poses))
        rotation_only={k:(r,np.zeros(3)) for k,(r,t) in poses.items()}
        self.assertIsNone(g.triangulate_track(images,[('A0',0),('C1',0)],rotation_only))

    def _problem(self):
        images,edges,poses,points=fixture(12); graph=g.build_tracks(images,edges)
        mapping={t['id']:points[dict(t['observations'])['A0']] for t in graph['tracks']}
        return g.GeometryProblem(images,graph,poses,mapping,'C1'), images, graph, poses, mapping

    def test_complete_jacobian_nonzero_chart_and_exact_scale(self):
        problem,*_=self._problem(); x=np.random.default_rng(12).normal(0,.001,problem.dimension)
        r,j=problem.evaluate(x); jac=j.toarray(); numerical=np.zeros_like(jac)
        for k in range(len(x)):
            step=np.zeros_like(x);step[k]=1e-6
            numerical[:,k]=(problem.evaluate(x+step)[0]-problem.evaluate(x-step)[0])/(2e-6)
        np.testing.assert_allclose(jac,numerical,atol=2e-7,rtol=2e-6)
        poses,_,_=problem.state(x)
        self.assertAlmostEqual(np.linalg.norm(poses['C1'][1]), problem.baseline, places=14)
        np.testing.assert_array_equal(poses['A0'][0],np.eye(3))
        np.testing.assert_array_equal(poses['A0'][1],np.zeros(3))

    def test_sphere_exponential_reaches_opposite_hemisphere_and_derivative(self):
        problem,*_=self._problem(); x=np.zeros(problem.dimension)
        cols=problem.layout['C1'][3:]; x[cols]=[1.8,.3]
        poses,_,derivatives=problem.state(x)
        self.assertLess(np.dot(poses['C1'][1],problem.unit),0)
        self.assertAlmostEqual(np.linalg.norm(poses['C1'][1]),problem.baseline,places=14)
        numeric=[]
        for col in cols:
            step=np.zeros_like(x);step[col]=1e-6
            numeric.append((problem.state(x+step)[0]['C1'][1]-problem.state(x-step)[0]['C1'][1])/(2e-6))
        np.testing.assert_allclose(derivatives['C1'][1],np.column_stack(numeric),atol=2e-10,rtol=2e-8)

    def test_missing_and_negative_depth_keep_fixed_mass(self):
        problem,images,graph,poses,points=self._problem()
        first=graph['tracks'][0]['id'];points[first]=None
        second=graph['tracks'][1]['id'];points[second]=np.array([0.,0.,-1.])
        bad=g.GeometryProblem(images,graph,poses,points,'C1')
        r,j,stats=bad.evaluate(np.zeros(bad.dimension),True)
        self.assertEqual(len(r),2*graph['observation_count'])
        self.assertEqual(stats['invalid_count'],6)
        self.assertAlmostEqual(stats['cost'],6*CAP_COST,places=10)
        rows=[k for k,obs in enumerate(bad.observations) if obs[2] in (first,second)]
        for row in rows: self.assertEqual(j[2*row:2*row+2].nnz,0)

    def test_sparse_solve_decreases_cost_with_scale_gauge(self):
        problem,images,graph,poses,points=self._problem()
        perturbed={k:(r.copy(),t.copy()) for k,(r,t) in poses.items()}
        perturbed['B1']=(rotation_exp(np.array([.005,-.003,.002]))@poses['B1'][0],poses['B1'][1]+[.01,-.01,.005])
        initial=dict(poses=perturbed,gauge_image='C1',anchor='A0')
        tri={t['id']:g.triangulate_track(images,t['observations'],perturbed) for t in graph['tracks']}
        before=g.GeometryProblem(images,graph,perturbed,tri,'C1')
        cost=before.evaluate(np.zeros(before.dimension),True)[2]['cost']
        out=g.fit_geometry(images,graph,initial,options={'max_accepted':8,'max_evaluations':30})
        self.assertLess(out['cost'],cost*.05)
        self.assertLessEqual(out['evaluations'],30);self.assertLessEqual(out['accepted_steps'],8)
        self.assertAlmostEqual(np.linalg.norm(out['poses']['C1'][1]),np.linalg.norm(poses['C1'][1]),places=14)
        self.assertEqual(out['observation_count'],graph['observation_count'])

    def test_observability_generic_and_point_elimination(self):
        problem,images,graph,poses,points=self._problem()
        solution=dict(problem=problem,poses=poses,points=points)
        # Test-only dense reference: the production algorithm never builds
        # this all-points dense nuisance matrix.
        _,jac=problem.evaluate(np.zeros(problem.dimension),information=True)
        dense=jac.toarray(); n=problem.pose_dimension
        pose=dense[:,:n].copy(); pose[:,problem.layout['C1'][3:]]/=problem.baseline
        u,s,_=np.linalg.svd(dense[:,n:],full_matrices=False)
        u=u[:,s>g.RANK_RTOL*s[0]]
        coefficient=np.einsum('ri,rj->ij',u,pose,optimize=False)
        effective=pose-np.einsum('ri,ij->rj',u,coefficient,optimize=False)
        threshold=g.RANK_RTOL*np.linalg.svd(pose,compute_uv=False)[0]
        rank=int(np.count_nonzero(np.linalg.svd(effective,compute_uv=False)>threshold))
        for query in ('C1','B1'):
            out=g.pose_observability(solution,query)
            self.assertTrue(out['observable']);self.assertEqual(out['rank'],rank)
            self.assertEqual(out['rank'],11);self.assertEqual(out['null_dimension'],0)
            self.assertEqual(out['point_block_rank_counts'],[0,0,0,12])
        # A final-state recenter also works beyond the old sphere hemisphere.
        x=np.zeros(problem.dimension);x[problem.layout['C1'][3:]]=[1.8,.3]
        moved,mapping,_=problem.state(x)
        recalibrated=copy.deepcopy(images)
        for key,(r,t) in moved.items():
            for track in graph['tracks']:
                feature=dict(track['observations'])[key]; p=r@mapping[track['id']]+t
                recalibrated[key]['xy'][feature]=p[:2]/p[2]*np.diag(recalibrated[key]['K'])[:2]+recalibrated[key]['K'][:2,2]
        final=g.GeometryProblem(recalibrated,graph,moved,mapping,'C1')
        self.assertTrue(g.pose_observability(dict(problem=final,poses=moved,points=mapping),'C1')['observable'])

    def test_query_length_null_allowed_but_capped_query_unobservable(self):
        images,_,poses,points=fixture(40)
        edges=[]
        for key,ids in [('C1',np.arange(20)),('B1',np.arange(20,40))]:
            edges.append(dict(i='A0',j=key,ids_i=ids,ids_j=ids,scores=np.ones(20)))
        graph=g.build_tracks(images,edges)
        mapping={t['id']:points[dict(t['observations'])['A0']] for t in graph['tracks']}
        problem=g.GeometryProblem(images,graph,poses,mapping,'C1')
        out=g.pose_observability(dict(problem=problem,poses=poses,points=mapping),'B1')
        # The two point sets share A0 but not 3D tracks: B1's scale is free.
        self.assertEqual(out['rank'],10);self.assertEqual(out['null_dimension'],1)
        self.assertTrue(out['observable']);self.assertLess(out['pose_null_ratio'],1e-8)
        images['B1']['xy']+=np.array([100.,0.])
        capped=g.GeometryProblem(images,graph,poses,mapping,'C1')
        _,exact=capped.evaluate(np.zeros(capped.dimension))
        _,info=capped.evaluate(np.zeros(capped.dimension),information=True)
        rows=[i for i,obs in enumerate(capped.observations) if obs[0]=='B1']
        for row in rows:
            self.assertGreater(exact[2*row:2*row+2].nnz,0)
            self.assertEqual(info[2*row:2*row+2].nnz,0)
        bad=g.pose_observability(dict(problem=capped,poses=poses,points=mapping),'B1')
        self.assertFalse(bad['observable']);self.assertGreater(bad['pose_null_ratio'],.1)
        self.assertEqual(bad['informative_observation_count'],60)

    def test_collinear_support_and_zero_translation_are_not_observable(self):
        images,edges,poses,_=fixture(12)
        points=np.column_stack((np.linspace(-.6,.6,12),np.zeros(12),np.full(12,4.)))
        for key,(r,t) in poses.items():
            q=(r@points.T).T+t;k=images[key]['K']
            images[key]['xy']=q[:,:2]/q[:,2,None]*np.diag(k)[:2]+k[:2,2]
        graph=g.build_tracks(images,edges)
        mapping={t['id']:points[dict(t['observations'])['A0']] for t in graph['tracks']}
        problem=g.GeometryProblem(images,graph,poses,mapping,'C1')
        out=g.pose_observability(dict(problem=problem,poses=poses,points=mapping),'B1')
        self.assertFalse(out['observable']);self.assertLess(out['rank'],11)
        zero=dict(poses);zero['B1']=(poses['B1'][0],np.zeros(3))
        out=g.pose_observability(dict(problem=problem,poses=zero,points=mapping),'B1')
        self.assertFalse(out['observable']);self.assertIsNone(out['pose_null_ratio'])

    def test_extreme_baseline_cannot_hide_direction_null(self):
        problem,images,graph,poses,points=self._problem()
        # Same image geometry at pathological numeric length units. The fixed
        # units rank test now regards translation columns as numerically null;
        # its output must not be diluted by the much larger rotation block.
        enlarged={key:(r,t*1e12) for key,(r,t) in poses.items()}
        mapping={key:p*1e12 for key,p in points.items()}
        moved=g.GeometryProblem(images,graph,enlarged,mapping,'C1')
        out=g.pose_observability(dict(problem=moved,poses=enlarged,points=mapping),'B1')
        self.assertLess(out['combined_pose_null_ratio'],1e-8)
        self.assertGreater(out['direction_null_ratio'],.1)
        self.assertFalse(out['observable'])


if __name__ == '__main__': unittest.main()
