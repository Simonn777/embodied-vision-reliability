import copy
import importlib.util
import math
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location('evaluate_0017_test', Path(__file__).with_name('evaluate_saved.py'))
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def fixture():
    ids, records, gt = [], [], {}
    for i in range(80):
        scene = e.SCENES[i // 16]
        ids.append({'pair_index': i, 'scene': scene, 'key': 'pair-' + str(i % 16)})
        gt[i] = {'R': I, 't': [1, 0, 0]}
        for j, arm in enumerate(e.ARMS):
            phi = math.radians(j + 1)
            records.append({'pair_index': i, 'scene': scene, 'key': ids[-1]['key'], 'arm': arm,
                'status': 'completed', 'estimate': {'R': I, 't': [math.cos(phi), math.sin(phi), 0]},
                'final_scores': {'pixel_msac': float(4 - j)},
                'depth_diagnostics': {'returned_mask': {'denominator': 4, 'native_branch_inactive_count': 0}},
                'timings': {'input_seconds': 0.1, 'call_seconds': j + 1.0, 'score_seconds': 0.2,
                            'charged_total_seconds': j + 1.3}})
    return ids, records, gt


def angle(record, degrees):
    a = math.radians(degrees)
    record['estimate']['t'] = [math.cos(a), math.sin(a), 0]


class EvaluationTests(unittest.TestCase):
    def test_full_denominators_scenes_and_cost(self):
        ids, records, gt = fixture()
        out = e.evaluate_records(e.prepare_records(records, ids), gt)
        self.assertEqual(out['summaries']['e_norm']['pair_count'], 80)
        self.assertTrue(all(x['e_norm']['pair_count'] == 16 for x in out['scene_summaries'].values()))
        self.assertTrue(all(x['e_norm']['pair_count'] == 64 for x in out['leave_one_scene_out_descriptive'].values()))
        self.assertAlmostEqual(out['external_selector_charged_seconds'], 896)
        self.assertTrue(all(r['choices']['pixel_msac_external']['arm'] == 'depth_affine' for r in out['rows']))
        self.assertFalse(out['decisions']['new_method_improvement_established'])

    def test_gt_does_not_change_deployed_choice(self):
        ids, records, gt = fixture()
        prepared = e.prepare_records(records, ids)
        for v in gt.values():
            v['t'] = [-1, 0, 0]
        out = e.evaluate_records(prepared, gt)
        self.assertTrue(all(r['choices']['pixel_msac_external']['arm'] == 'depth_affine' for r in out['rows']))
        self.assertGreater(out['rows'][0]['outcomes']['e_norm']['error']['translation_degrees'], 170)
        self.assertLess(out['rows'][0]['outcomes']['e_norm']['error']['translation_signfolded_degrees'], 2)

    def test_score_tie_uses_fixed_arm_order(self):
        ids, records, _ = fixture()
        for r in records:
            r['final_scores']['pixel_msac'] = 0
        self.assertTrue(all(r['selected_arm'] == 'e_norm' for r in e.prepare_records(records, ids)))

    def test_outer_wall_overhead_and_unknown_interrupted_components(self):
        ids, records, gt = fixture()
        records[0]['timings']['charged_total_seconds'] += 10
        records[1]['status'] = 'timeout'
        records[1]['estimate'] = None
        records[1]['timings']['call_seconds'] = None
        records[1]['timings']['charged_total_seconds'] = 90
        out = e.evaluate_records(e.prepare_records(records, ids), gt)
        self.assertAlmostEqual(out['external_selector_charged_seconds'], 896 + 10 + 90 - 2.3)

    def test_unavailable_outputs_stay_in_denominator(self):
        ids, records, gt = fixture()
        for r in records[:4]:
            r['status'] = 'timeout'
            r['estimate'] = None
            r['final_scores'] = None
        out = e.evaluate_records(e.prepare_records(records, ids), gt)
        self.assertEqual(out['rows'][0]['choices']['oracle_all']['penalized_max_degrees'], 180)
        self.assertEqual(out['summaries']['e_norm']['status_counts'], {'timeout': 1, 'completed': 79})
        self.assertEqual(out['summaries']['pixel_msac_external']['pair_count'], 80)
        self.assertGreater(out['summaries']['e_norm']['success_only_auc_fraction']['5'], out['summaries']['e_norm']['auc_fraction']['5'])

    def test_same_arm_two_scene_mechanism(self):
        ids, records, gt = fixture()
        for index in (0, 16):
            r = records[index * 4 + 2]
            angle(r, 10)
            r['depth_diagnostics']['returned_mask']['native_branch_inactive_count'] = 1
        out = e.evaluate_records(e.prepare_records(records, ids), gt)
        self.assertTrue(out['decisions']['omission_mechanism_trigger_by_arm']['depth_scale'])
        self.assertFalse(out['decisions']['omission_mechanism_trigger_by_arm']['depth_affine'])
        self.assertFalse(out['decisions']['manuscript_admitted'])

    def test_do_not_pool_different_depth_arms_for_gate(self):
        ids, records, gt = fixture()
        for index, armindex in ((0, 2), (16, 3)):
            r = records[index * 4 + armindex]
            angle(r, 10)
            r['depth_diagnostics']['returned_mask']['native_branch_inactive_count'] = 1
        out = e.evaluate_records(e.prepare_records(records, ids), gt)
        self.assertFalse(any(out['decisions']['omission_mechanism_trigger_by_arm'].values()))

    def test_all_match_omission_is_not_masked_omission(self):
        ids, records, gt = fixture()
        for r in records:
            if r['arm'] == 'depth_scale':
                angle(r, 20)
                r['depth_diagnostics']['all'] = {'native_branch_inactive_count': 100}
        out = e.evaluate_records(e.prepare_records(records, ids), gt)
        self.assertFalse(any(out['decisions']['omission_mechanism_trigger_by_arm'].values()))

    def test_projection_invalidity_does_not_replace_native_branch(self):
        ids, records, gt = fixture()
        for r in records:
            if r['arm'] == 'depth_scale':
                angle(r, 20)
                r['depth_diagnostics']['returned_mask']['inactive_transformed_direction_count'] = 2
        out = e.evaluate_records(e.prepare_records(records, ids), gt)
        self.assertFalse(out['decisions']['omission_mechanism_trigger_by_arm']['depth_scale'])

    def test_oracle_additional_information_gate(self):
        ids, records, gt = fixture()
        for r in records:
            angle(r, 10 if r['arm'].startswith('e_') else 1)
        out = e.evaluate_records(e.prepare_records(records, ids), gt)
        self.assertTrue(out['decisions']['selector_necessary_headroom_pass'])
        self.assertGreater(out['decisions']['additional_depth_oracle_headroom_pp'], 50)
        self.assertFalse(out['decisions']['new_method_improvement_established'])

    def test_bad_coverage_identity_and_timings_rejected(self):
        ids, records, _ = fixture()
        modifications = [lambda r: r.pop(), lambda r: r.__setitem__(1, copy.deepcopy(r[0])),
            lambda r: r[0].__setitem__('scene', 'unknown'),
            lambda r: r[0].__setitem__('status', 'binding_error'),
            lambda r: r[0]['timings'].__setitem__('charged_total_seconds', 0),
            lambda r: r[0]['timings'].__setitem__('call_seconds', float('nan'))]
        for mutate in modifications:
            with self.subTest(mutate=mutate):
                bad = copy.deepcopy(records)
                mutate(bad)
                with self.assertRaises(ValueError):
                    e.prepare_records(bad, ids)

    def test_depth_missing_mask_is_not_zero_omission(self):
        ids, records, gt = fixture()
        records[2]['depth_diagnostics'] = None
        with self.assertRaises(ValueError):
            e.evaluate_records(e.prepare_records(records, ids), gt)

    def test_bad_prediction_rotation_rejected_before_gt(self):
        ids, records, _ = fixture()
        records[0]['estimate']['R'] = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
        with self.assertRaises(ValueError):
            e.prepare_records(records, ids)

    def test_bad_gt_rotation_does_not_receive_sign_repair(self):
        ids, records, gt = fixture()
        gt[0]['R'] = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
        with self.assertRaises(ValueError):
            e.evaluate_records(e.prepare_records(records, ids), gt)


if __name__ == '__main__':
    unittest.main()
