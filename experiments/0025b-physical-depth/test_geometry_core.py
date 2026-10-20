"""Manufactured-only checks, prepared before execution; no assets or models."""
import math
import unittest

import numpy as np

import geometry_core as g


def source_plane():
    v,u = np.indices((g.SOURCE_HEIGHT,g.SOURCE_WIDTH),dtype=float)
    x=(u-g.SOURCE_K[0,2])/g.SOURCE_K[0,0]
    y=(v-g.SOURCE_K[1,2])/g.SOURCE_K[1,1]
    return 1./(.03*x-.02*y+.25)


def row_plane(geometry):
    ray=geometry["source_ray"]
    return 1./(.03*ray[...,0]-.02*ray[...,1]+.25*ray[...,2])


class GeometryChecks(unittest.TestCase):
    def test_scalar_inverse_row_rotation_all_views(self):
        for view in g.VIEWS:
            geometry=g.make_view_geometry(view)
            self.assertTrue(geometry["full_rgb_four_neighbor_inbounds"])
            for u,v in ((0,0),(447,335),(147,84),(223,168)):
                angle=geometry["row_angle_radians"][v,u]
                c,s=math.cos(angle),math.sin(angle)
                if view.startswith("rs_x"):
                    Q=np.array([[1.,0.,0.],[0.,c,-s],[0.,s,c]])
                elif view.startswith("rs_y"):
                    Q=np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])
                else:
                    Q=np.eye(3)
                r=np.array([(u-223.5)/481.2,(v-167.5)/481.2,1.])
                source=Q.T@r
                np.testing.assert_allclose(source,geometry["source_ray"][v,u],rtol=0,atol=2e-16)
                pixel=g.SOURCE_K@(source/source[2])
                np.testing.assert_allclose(pixel[:2],geometry["source_xy"][v,u],rtol=0,atol=2e-13)
                self.assertAlmostEqual(geometry["row_seconds"][v,u],.04*(v/335.-.5))

    def test_exact_half_up_lattice_and_checker(self):
        lattice=g.fixed_lattice()
        self.assertEqual(lattice["count"],4096)
        self.assertEqual(np.count_nonzero(lattice["train"]),2048)
        self.assertEqual(np.count_nonzero(lattice["eval"]),2048)
        self.assertFalse(np.any(lattice["train"] & lattice["eval"]))
        for i in range(64):
            for j in range(64):
                k=64*i+j
                self.assertEqual(lattice["u"][k],math.floor(j*447/63+.5))
                self.assertEqual(lattice["v"][k],math.floor(i*335/63+.5))
        self.assertEqual(len(set(zip(lattice["u"],lattice["v"]))),4096)

    def test_inverse_depth_planar_truth_and_h_direction(self):
        for view in g.VIEWS:
            geometry=g.make_view_geometry(view)
            result=g.sample_truth(source_plane(),geometry,max_relative_depth_spread=.02)
            self.assertEqual(result["status"],"usable")
            lattice=result["lattice"]
            exact=row_plane(geometry)[lattice["v"],lattice["u"]]
            np.testing.assert_allclose(result["row_z"][result["support"]],exact[result["support"]],rtol=2e-15,atol=2e-15)
            np.testing.assert_allclose(result["source_mid_z"][result["support"]],
                result["row_z"][result["support"]]*result["h"][result["support"]],rtol=2e-15,atol=2e-15)

    def test_truth_gap_and_missing_values_do_not_use_models(self):
        geometry=g.make_view_geometry("gs")
        depth=source_plane()
        result=g.sample_truth(depth,geometry,max_relative_depth_spread=.02)
        k=np.flatnonzero(result["support"])[200]
        x,y=np.floor(result["source_xy"][k]).astype(int)
        damaged=depth.copy();damaged[y,x]*=2.
        failed=g.sample_truth(damaged,geometry,max_relative_depth_spread=.02)
        self.assertFalse(failed["support"][k])
        damaged[y,x]=np.nan
        failed=g.sample_truth(damaged,geometry,max_relative_depth_spread=.02)
        self.assertFalse(failed["support"][k])
        blank=g.sample_truth(np.zeros_like(depth),geometry,max_relative_depth_spread=.02)
        self.assertEqual(blank["status"],"unavailable")
        self.assertEqual(blank["reason"],"insufficient_fixed_support")

    def test_row_ls_recovers_affine_and_holdout_unchanged(self):
        geometry=g.make_view_geometry("rs_x_positive")
        result=g.sample_truth(source_plane(),geometry,max_relative_depth_spread=.02)
        # Negative raw depth is legal in the affine observation model.
        depth=(row_plane(geometry)-10.)/1.2
        designs=g.build_designs(depth,np.ones((g.HEIGHT,g.WIDTH))*4.,result)
        fit=g.fit_on_training(designs["rs_row"],result)
        self.assertEqual(fit["status"],"usable")
        np.testing.assert_allclose(fit["coefficients"],[1.2,10.],rtol=1e-11,atol=1e-11)
        evaluation=g.evaluate_fixed_holdout(fit,designs["rs_row"],result)
        self.assertEqual(evaluation["status"],"usable")
        self.assertEqual(evaluation["holdout_count"],result["counts"]["eval"])
        self.assertLess(np.max(evaluation["absolute_relative_error"]),1e-12)
        scale_only=g.fit_on_training(designs["rs_scale_only"],result)
        self.assertEqual(scale_only["status"],"unavailable")

    def test_ordinary_prediction_interpolation_and_four_designs(self):
        geometry=g.make_view_geometry("rs_y_negative")
        result=g.sample_truth(source_plane(),geometry,max_relative_depth_spread=.02)
        v,u=np.indices((g.HEIGHT,g.WIDTH),dtype=float)
        gs=-3.+.001*u+.002*v
        rs=(row_plane(geometry)-.15)/1.2
        designs=g.build_designs(rs,gs,result)
        support=result["support"]
        xy=result["canonical_gs_xy"]
        expected=-3.+.001*xy[:,0]+.002*xy[:,1]
        np.testing.assert_allclose(designs["gs_transport"][support,0],expected[support]/result["h"][support],rtol=2e-15,atol=2e-15)
        np.testing.assert_allclose(designs["rs_mid"][support,0],designs["rs_row"][support,0]/result["h"][support])
        self.assertEqual(set(designs),set(g.ARMS))

    def test_model_invalid_heldout_fails_without_deleting_support(self):
        geometry=g.make_view_geometry("rs_x_negative")
        result=g.sample_truth(source_plane(),geometry,max_relative_depth_spread=.02)
        before=result["support"].copy()
        depth=(row_plane(geometry)-.15)/1.2
        k=np.flatnonzero(result["support"] & result["lattice"]["eval"])[10]
        depth[result["lattice"]["v"][k],result["lattice"]["u"][k]]=np.nan
        designs=g.build_designs(depth,np.ones((g.HEIGHT,g.WIDTH))*4.,result)
        fit=g.fit_on_training(designs["rs_row"],result)
        self.assertEqual(fit["status"],"usable")
        evaluation=g.evaluate_fixed_holdout(fit,designs["rs_row"],result)
        self.assertEqual(evaluation["status"],"unavailable")
        self.assertEqual(evaluation["invalid_model_count"],1)
        self.assertEqual(evaluation["holdout_count"],result["counts"]["eval"])
        np.testing.assert_array_equal(result["support"],before)

    def test_training_only_rank_and_positive_scale_failures(self):
        result=g.sample_truth(source_plane(),g.make_view_geometry("gs"),max_relative_depth_spread=.02)
        constant=np.ones((4096,2))
        rank_failure=g.fit_on_training(constant,result)
        self.assertEqual(rank_failure["reason"],"rank_deficient")
        self.assertEqual(set(("rank","rank_threshold","singular_values","column_norms"))-set(rank_failure),set())
        # Positive model values but negative slope against varying positive truth.
        first=10.-np.nan_to_num(result["row_z"],nan=4.)
        negative=g.fit_on_training(np.column_stack((first,np.ones(4096))),result)
        self.assertEqual(negative["reason"],"nonpositive_scale")
        for key in ("rank","rank_threshold","singular_values","column_norms","coefficients"):
            self.assertIn(key,negative)
            self.assertTrue(np.all(np.isfinite(negative[key])))
        invalid=constant.copy()
        k=np.flatnonzero(result["support"] & result["lattice"]["train"])[0]
        invalid[k,0]=np.inf
        self.assertEqual(g.fit_on_training(invalid,result)["reason"],"invalid_training_model_depth")


if __name__ == "__main__":
    unittest.main()
