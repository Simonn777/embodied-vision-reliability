"""Saved-output-only metadata repair of evaluate_diagnostic; predictions unchanged."""
import argparse
from pathlib import Path
import run_diagnostic as runner

metrics = runner.metrics
BASELINE_EVALUATION_SHA = 'f8fe46392810b78d485a8dd90ae1849eae6491eaba620d2c9454b6c0f15cb5a4'
MODES = ('full_original', 'subset_original', 'subset_refined', 'full_refined')

def decision(auc, rows):
    full_gain = auc['full_refined']['auc_percent']['5'] - auc['full_original']['auc_percent']['5']
    subset_gain = auc['subset_refined']['auc_percent']['5'] - auc['subset_original']['auc_percent']['5']
    full20 = auc['full_refined']['auc_percent']['20'] - auc['full_original']['auc_percent']['20']
    improvements = sum(r['errors']['full_original']['max_degrees'] -
                       r['errors']['full_refined']['max_degrees'] > 1e-9 for r in rows)
    enough = sum(r['admitted_count'] >= 8 for r in rows)
    gates = {'complete_full_cohort': len(rows) == 16,
             'at_least_12_usable_subsets': enough >= 12,
             'full_AUC5_gain_at_least_0_25_pp': full_gain >= 0.25,
             'full_AUC20_loss_at_most_0_5_pp': full20 >= -0.5,
             'at_least_10_full_pair_improvements': improvements >= 10,
             'subset_AUC5_gain_at_least_2_pp': subset_gain >= 2}
    return {'passed': all(gates.values()), 'criteria': gates, 'usable_subset_pairs': enough,
            'full_pair_improvements': improvements, 'full_AUC5_gain_pp': full_gain,
            'full_AUC20_gain_pp': full20, 'subset_AUC5_gain_pp': subset_gain,
            'scope': 'Sparse configuration expansion only; no novelty/manuscript/held-out admission'}

def evaluate(manifest, original_seal, photo_seal, pose_seal, baseline_evaluation, output):
    if Path(output).exists():
        raise FileExistsError('evaluation output already exists')
    seal, sh, geometric = runner.verify_phase(manifest, original_seal, pose_seal, 'geometric', photo_seal)
    photo, ph, photos = runner.verify_phase(manifest, original_seal, photo_seal, 'photometric')
    _, _, originals, _, _ = metrics.verify_seal(manifest, original_seal)
    # No label file read until both complete cohorts, source hashes and all arm poses are verified.
    truth, th = metrics.load_hashed_json(baseline_evaluation)
    runner.require(th == BASELINE_EVALUATION_SHA and truth.get('seal_sha256') == seal['baseline_seal_sha256']
                   and len(truth.get('rows', [])) == 16, 'saved baseline labels differ')
    errors, rows = {m: [] for m in MODES}, []
    for i, ((_, record), (_, photo_record), gt_record) in enumerate(zip(geometric, photos, truth['rows'])):
        runner.require(gt_record['pair_index'] == i and gt_record['key'] == record['key'], 'label key/order differs')
        gt = gt_record['ground_truth']
        estimates = {'full_original': {'status': 'completed', 'estimate': record['full_original_reference']['estimate']},
                     **record['arms']}
        values = {}
        for mode in MODES:
            arm = estimates[mode]
            if arm['status'] == 'unavailable':
                values[mode] = {'max_degrees': 180.0, 'rotation_degrees': None, 'translation_degrees': None,
                                'unavailable': True, 'reason': arm['reason']}
            else:
                pose = arm['estimate']
                values[mode], _ = metrics.pose_errors(pose['R'], pose['t'], gt['R_original'], gt['t_original'])
                values[mode]['unavailable'] = False
            errors[mode].append(values[mode]['max_degrees'])
        runner.require(abs(values['full_original']['max_degrees']-gt_record['errors']['max_degrees']) < 1e-9,
                       'baseline metric recomputation disagrees')
        rows.append({'pair_index': i, 'key': record['key'], 'errors': values,
            'selected_count': photo_record['selected_count'], 'initialized_count': photo_record['initialized_count'],
            'admitted_count': photo_record['admitted_count'],
            'target_coordinate_changed_count': photo_record['target_coordinate_changed_count'],
            'photo_timing_seconds': photo_record['timing_seconds'],
            'pose_timing_seconds': record['timing_seconds'],
            'native_call_seconds': {m: record['arms'][m].get('call_seconds', 0.0) for m in runner.geometry.ARM_ORDER},
            'full_original_native_call_seconds': originals[i][1]['timing_seconds']['estimator_call']})
    auc = {m: metrics._auc_summary(errors[m]) for m in MODES}
    result = {'schema': 'evr0015.evaluation.v1', 'source_sha256': seal['source_sha256'],
        'evaluation_source_sha256': runner.sha(__file__),
        'repair': 'Read baseline timing_seconds.estimator_call; original scorer failed on absent top-level call_seconds. No prediction/metric/gate change.',
        'photometric_seal_sha256': ph, 'geometric_seal_sha256': sh, 'baseline_evaluation_sha256': th,
        'coverage': {'selected': 16, 'completed_full': 16,
            'subset_unavailable_pairs': sum(r['admitted_count'] < 8 for r in rows)},
        'auc': auc, 'expansion_gate': decision(auc, rows), 'rows': rows,
        'metric_convention': 'Same0013 signed translation, projected rotation, max angle, trapezoid AUC',
        'unavailable_subset_penalty_degrees': 180.0,
        'photometric_loss_is_not_pose_accuracy': True,
        'scope': 'Classical image-information control; reused16 dependent scene0015 pairs; no population inference',
        'operation_wall_seconds_at_final_seal': seal['wall_seconds_since_operation_start']}
    runner.infrastructure.write_new(output, result)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'baseline-seal', 'photometric-seal', 'geometric-seal', 'baseline-evaluation', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.manifest, args.baseline_seal, args.photometric_seal, args.geometric_seal,
             args.baseline_evaluation, args.output)
