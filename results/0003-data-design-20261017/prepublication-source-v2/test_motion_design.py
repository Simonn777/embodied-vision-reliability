"""Hand-computed geometry/diagnostic fixtures and adversarial Git freeze cases."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from evr import motion_design as m

I=((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))
K=((100,0,50),(0,100,50),(0,0,1))
def xy(values):return[(v,0)for v in values]

class GeometryTests(unittest.TestCase):
    def test_genuine_motion_no_error(self):
        d=m.decompose([0,1,2],xy([0,1,4]),xy([0,1,4]));self.assertEqual(d['r_pred'],(2,0));self.assertEqual(d['r_gt_image'],(2,0));self.assertEqual(d['r_error'],(0,0))
    def test_constant_offset_wrong_and_zero_residual(self):
        d=m.decompose([0,1,2],xy([10,11,12]),xy([0,1,2]));self.assertEqual(d['d_pred'],0);self.assertEqual(12-2,10)
    def test_linear_drift_wrong_and_zero_residual(self):
        d=m.decompose([0,1,2],xy([0,3,6]),xy([0,0,0]));self.assertEqual(d['d_pred'],0);self.assertGreater(6,4)
    def test_quadratic_error_positive_control(self):
        d=m.decompose([0,1,2],xy([0,1,4]),xy([0,0,0]));self.assertEqual(d['r_error'],(2,0))
    def test_signed_cancellation(self):
        d=m.decompose([0,1,2],xy([0,1,2]),xy([0,1,8]));self.assertEqual(d['r_gt_image'],(6,0));self.assertEqual(d['r_error'],(-6,0));self.assertEqual(d['r_pred'],(0,0))
    def test_orthogonal_norms_do_not_add_and_strict_error(self):
        d=m.decompose([0,1,2],[(0,0),(0,0),(3,4)],[(0,0),(0,0),(3,0)]);self.assertEqual((d['d_gt_image'],d['d_error'],d['d_pred']),(3,4,5));self.assertFalse(4>4)
    def test_irregular_timestamps(self):self.assertEqual(m.innovation([0,2,5],xy([0,4,10])),(0,0))
    def test_invalid_gt_no_anchor_replacement(self):
        d=m.decompose([0,1,2],xy([0,1,2]),[(0,0),(float('nan'),0),(2,0)]);self.assertFalse(d['defined']);self.assertIn('retain primary denominator',d['reason'])
    def test_projection_translation_and_scaling(self):
        self.assertEqual(m.project((1,2,2),K,I),(100,150));self.assertEqual(m.resize_xy((960,540),960,540),(256,256))
        T=((1,0,0,1),(0,1,0,0),(0,0,1,0),(0,0,0,1));self.assertEqual(m.project((1,2,2),K,T),(150,150))
    def test_camera_motion_is_not_world_acceleration(self):
        world=[(0,0,2)]*3;T=((1,0,0,1),(0,1,0,0),(0,0,1,0),(0,0,0,1));p=[m.project(world[0],K,I),m.project(world[1],K,I),m.project(world[2],K,T)]
        self.assertEqual(m.world_acceleration([0,1,2],world),(0,0,0));self.assertEqual(m.innovation([0,1,2],p),(50,0))
    def test_bad_camera_and_times(self):
        with self.assertRaises(ValueError):m.project((0,0,-1),K,I)
        with self.assertRaises(ValueError):m.project((0,0,2),K,((2,0,0,0),*I[1:]))
        with self.assertRaises(ValueError):m.innovation([0,0,1],xy([0,1,2]))
    def test_history_keeps_rejected_invalid_logit_and_occluded(self):
        self.assertEqual(m.reference_penalties([0,1,2,3],xy([0,10,30,50])),[0,0,1.25,0])
        self.assertEqual(m.reference_penalties([0,1,3],xy([0,10,50]))[-1],2.5)
    def test_missing_raw_skips_with_real_gap(self):
        self.assertEqual(m.reference_penalties([0,1,2,5],[(0,0),(float('nan'),0),(4,0),(10,0)]),[0,None,0,0])
    def test_archive_path_links_devices_rejected(self):
        for path,kind in [('../x','file'),('/x','file'),('x\\y','file'),('a/./b','file'),('a','symlink'),('a','device')]:
            with self.assertRaises(ValueError):m.safe_member(path,kind)
        self.assertEqual(m.safe_member('family/anno.npz'),'family/anno.npz')
    def test_allocation_quarantines_unknown_or_audited_whole_family(self):
        groups=[{'family_id':str(i),'audited':i==0,'independence_verified':i!=1}for i in range(36)];a=m.allocate(groups);self.assertEqual(a,m.allocate(list(reversed(groups))))
        self.assertEqual(a['0'],a['1']);self.assertEqual(a['0'],'development');self.assertEqual(list(a.values()).count('calibration'),10);self.assertEqual(list(a.values()).count('evaluation'),20)
    def test_string_independence_never_allocates_heldout(self):
        self.assertEqual(m.allocate([{'family_id':'a','audited':False,'independence_verified':'unknown'}])['a'],'development')
    def test_readiness_fraction_over_one_rejected(self):
        e=dict.fromkeys(['whole_archive_verified','geometry_verified','family_provenance_verified','cohort_frozen','development_precision_plan_passed'],True);e.update(independent_evaluation_groups=20,calibration_groups=10,valid_gt_fraction=1.1)
        self.assertFalse(m.readiness(e)['ready'])
    def test_readiness_unknown_fails_closed(self):self.assertEqual(m.readiness({})['status'],'BLOCKED_NO_FINAL_COHORT')
    def test_relative_goal_and_inconclusive_gates(self):
        kw=dict(test_groups=20,error_groups=5,native_failures=20,defined_replicates=1900)
        self.assertTrue(m.improvement_gate(.1,.07,-.001,**kw)['continuation_signal'])
        self.assertFalse(m.improvement_gate(.1,.09,-.001,**kw)['continuation_signal'])
        self.assertEqual(m.improvement_gate(0,0,-.001,**kw)['status'],'INCONCLUSIVE')
        for name,value in [('test_groups',19),('error_groups',4),('native_failures',19),('defined_replicates',1899)]:
            self.assertEqual(m.improvement_gate(.1,.07,-.001,**(kw|{name:value}))['status'],'INCONCLUSIVE')

class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.git('init','-q');self.git('config','user.name','Fixture');self.git('config','user.email','fixture@example.invalid')
        for path in m.MANDATORY:
            p=self.root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture '+path+'\n')
        self.git('add','.');self.git('commit','-qm','inputs');self.input=self.git('rev-parse','HEAD').strip()
        self.r={'schema':1,'freeze_kind':'DESIGN_ONLY','readiness':'BLOCKED_NO_FINAL_COHORT','input_commit':self.input,'config_path':m.CONFIG,'protocol_path':m.PROTOCOL,'inputs_sha256':{p:hashlib.sha256((self.root/p).read_bytes()).hexdigest()for p in m.MANDATORY}}
        self.write();self.git('add','.');self.git('commit','-qm','receipt');self.publish()
    def git(self,*args):return subprocess.check_output(['git','-C',str(self.root),*args],stderr=subprocess.DEVNULL,text=True)
    def write(self): (self.root/m.RECEIPT).write_text(json.dumps(self.r))
    def publish(self):self.git('update-ref','refs/remotes/'+m.BRANCH,self.git('rev-parse','HEAD').strip())
    def commit(self):self.git('add','.');self.git('commit','-qm','changed');self.publish()
    def test_valid_canonical_commit_publication(self):self.assertEqual(m.verify_design(self.root)['files'],9)
    def test_empty_missing_and_extra_map_rejected(self):
        original=self.r['inputs_sha256'].copy()
        for mapping in [{},{k:v for k,v in original.items()if k!=m.CONFIG},original|{'../outside':'0'*64}]:
            self.r['inputs_sha256']=mapping;self.write()
            with self.assertRaises(ValueError):m.verify_design(self.root)
    def test_old_commit_with_new_hash_rejected(self):
        p=self.root/m.CONFIG;p.write_text('new bytes');self.r['inputs_sha256'][m.CONFIG]=hashlib.sha256(p.read_bytes()).hexdigest();self.write();self.commit()
        with self.assertRaises(ValueError):m.verify_design(self.root)
    def test_untracked_or_edited_receipt_rejected(self):
        self.r['input_commit']='0'*40;self.write()
        with self.assertRaises(ValueError):m.verify_design(self.root)
    def test_unpublished_receipt_rejected(self):
        (self.root/m.RECEIPT).write_text(json.dumps(self.r,indent=2));self.git('add','.');self.git('commit','-qm','unpublished')
        with self.assertRaises(ValueError):m.verify_design(self.root)
    def test_missing_public_ref_rejected(self):
        self.git('update-ref','-d','refs/remotes/'+m.BRANCH)
        with self.assertRaises(ValueError):m.verify_design(self.root)
    def test_alternate_paths_and_binding_rejected(self):
        with self.assertRaises(ValueError):m.verify_design(self.root,'../receipt.json')
        self.r['protocol_path']='other.md';self.write()
        with self.assertRaises(ValueError):m.verify_design(self.root)
    def test_changed_source_and_symlink_rejected(self):
        p=self.root/m.CONFIG;p.unlink();p.symlink_to(self.root/m.PROTOCOL)
        with self.assertRaises(ValueError):m.verify_design(self.root)
    def test_duplicate_keys_rejected(self):
        with self.assertRaises(ValueError):m.unique_json('{"x":1,"x":2}')
    def test_self_reference_rejected(self):
        self.r['input_commit']=self.git('rev-parse','HEAD').strip();self.write();self.commit()
        with self.assertRaises(ValueError):m.verify_design(self.root)
    def test_committed_missing_map_rejected(self):
        del self.r['inputs_sha256'][m.CONFIG];self.write();self.commit()
        with self.assertRaises(ValueError):m.verify_design(self.root)
    def test_staged_source_change_rejected(self):
        (self.root/m.CONFIG).write_text('staged mutation');self.git('add',m.CONFIG)
        with self.assertRaises(ValueError):m.verify_design(self.root)

if __name__=='__main__':unittest.main()
