import unittest
from evr.motion_phases import check_phase

STRUCTURE=dict(schema_identity_checked=True,development_geometry_quarantined=True,allocation_source_manifest_frozen=True,pipeline_binds_v1_and_addendum=True)
class MotionPhaseTests(unittest.TestCase):
    def test_frame0_GT_access_is_truthfully_recorded(self):
        r=check_phase('pre_cohort',frame0_query_GT_accessed=True);self.assertTrue(r['frame0_GT_access']);self.assertFalse(r['execution_authority'])
    def test_heldout_future_GT_before_freeze_rejected(self):
        for stage in ['pre_cohort','pre_prediction']:
            with self.assertRaises(ValueError):check_phase(stage,heldout_GT_beyond_frame0_accessed=True,**STRUCTURE)
    def test_pre_prediction_quality_remains_pending(self):
        self.assertEqual(check_phase('pre_prediction',**STRUCTURE)['GT_quality'],'PENDING_POST_PREDICTION')
    def test_prevalidated_heldout_quality_rejected(self):
        with self.assertRaises(ValueError):check_phase('pre_prediction',all_required_family_GT_gates_pass=True,**STRUCTURE)
    def test_metadata_and_bound_complete_manifest_required(self):
        for field in STRUCTURE:
            with self.assertRaises(ValueError):check_phase('pre_prediction',**(STRUCTURE|{field:False}))
    def test_all_predictions_must_precede_GT_adjudication(self):
        with self.assertRaises(ValueError):check_phase('post_prediction',heldout_GT_beyond_frame0_accessed=True,GT_adjudication_after_prediction_freeze=True,**STRUCTURE)
    def test_bad_chronology_rejected(self):
        with self.assertRaises(ValueError):check_phase('post_prediction',all_prediction_hashes_frozen=True,heldout_GT_beyond_frame0_accessed=True,**STRUCTURE)
    def test_failed_or_undefined_quality_retains_all_families(self):
        r=check_phase('post_prediction',all_prediction_hashes_frozen=True,GT_adjudication_after_prediction_freeze=True,all_required_family_GT_gates_pass=False,**STRUCTURE)
        self.assertEqual(r['status'],'INCONCLUSIVE');self.assertTrue(r['all_allocated_families_retained']);self.assertTrue(r['primary_definition_unchanged'])
    def test_passed_GT_gate_is_not_efficacy_or_authority(self):
        r=check_phase('post_prediction',all_prediction_hashes_frozen=True,heldout_GT_beyond_frame0_accessed=True,GT_adjudication_after_prediction_freeze=True,all_required_family_GT_gates_pass=True,**STRUCTURE)
        self.assertEqual(r['status'],'GT_GATE_PASSED_NOT_EFFICACY');self.assertFalse(r['execution_authority'])
    def test_typed_flags(self):
        with self.assertRaises(ValueError):check_phase('pre_cohort',frame0_query_GT_accessed='false')
if __name__=='__main__':unittest.main()
