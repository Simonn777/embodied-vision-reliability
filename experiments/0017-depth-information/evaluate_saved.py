#!/usr/bin/env python3
"""Score sealed 0017 outputs; no estimator calls and no GT-dependent selection."""
import argparse
from collections import Counter
import importlib.util
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = ('e_norm', 'e_ts', 'depth_scale', 'depth_affine')
SCENES = ('florence_cathedral', 'lincoln_memorial', 'london_bridge',
          'milan_cathedral', 'sagrada_familia')
COUNT = 80


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


metrics = module('evr0017_metrics', HERE.parent / '0013-pose-headroom/evaluate_sealed.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonnegative(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            'expected finite nonnegative scalar')
    return float(value)


def prepare_records(records, identities):
    """Validate full Cartesian product and select using saved scores only."""
    require(len(identities) == COUNT and len(records) == COUNT * len(ARMS),
            'expected 80 pairs and 320 records')
    require([p['pair_index'] for p in identities] == list(range(COUNT)), 'pair order')
    require(Counter(p['scene'] for p in identities) == Counter({s: 16 for s in SCENES}),
            'expected exactly 16 pairs per scene')
    require(len({(p['scene'], p['key']) for p in identities}) == COUNT, 'duplicate pair')
    metrics.json_safe(records)
    by_index = {i: {} for i in range(COUNT)}
    for r in records:
        i, arm = r.get('pair_index'), r.get('arm')
        require(type(i) is int and i in by_index and arm in ARMS, 'invalid record identity')
        require(arm not in by_index[i], 'duplicate arm/pair')
        identity = identities[i]
        require(r.get('scene') == identity['scene'] and r.get('key') == identity['key'],
                'record/manifest identity mismatch')
        require(r.get('status') in ('completed', 'timeout', 'estimator_error'),
                'nonterminal or integrity-error record')
        timing = r.get('timings', {})
        for name in ('input_seconds', 'call_seconds', 'score_seconds', 'charged_total_seconds'):
            require(name in timing, 'missing timing field')
            if timing[name] is None:
                require(r['status'] != 'completed' and name != 'charged_total_seconds',
                        'completed/total timing cannot be unknown')
            else:
                nonnegative(timing[name])
        require(sum(timing[k] or 0 for k in ('input_seconds', 'call_seconds', 'score_seconds'))
                <= timing['charged_total_seconds'] + 1e-6,
                'charged time omits work')
        if r['status'] == 'completed':
            estimate = r.get('estimate', {})
            metrics.valid_rotation(estimate.get('R'))
            metrics.unit_translation(estimate.get('t'))
            nonnegative(r.get('final_scores', {}).get('pixel_msac'))
            if arm.startswith('depth_'):
                omission_support(r)
        else:
            require(not r.get('estimate'), 'failed output has an estimate')
        by_index[i][arm] = r
    prepared = []
    for i in range(COUNT):
        bank = by_index[i]
        require(set(bank) == set(ARMS), 'incomplete Cartesian product')
        eligible = [a for a in ARMS if bank[a]['status'] == 'completed']
        chosen = min(eligible, key=lambda a: (bank[a]['final_scores']['pixel_msac'], ARMS.index(a))) if eligible else None
        prepared.append({'identity': identities[i], 'records': bank, 'selected_arm': chosen,
                         'selector_charged_seconds': sum(bank[a]['timings']['charged_total_seconds'] for a in ARMS)})
    return prepared


def summarize(values):
    require(len(values) > 0, 'empty metric denominator')
    errors = [v['penalized_max_degrees'] for v in values]
    folded = [v['penalized_max_signfolded_degrees'] for v in values]
    successes = [v for v in values if v['status'] == 'completed']
    return {'pair_count': len(values), 'status_counts': dict(Counter(v['status'] for v in values)),
            'auc_fraction': dict(zip(('5', '10', '20'), metrics.dms_auc(errors))),
            'signfolded_auc_fraction': dict(zip(('5', '10', '20'), metrics.dms_auc(folded))),
            'count_strictly_below': {str(t): sum(e < t for e in errors) for t in (5, 10, 20)},
            'success_only_auc_fraction': dict(zip(('5', '10', '20'), metrics.dms_auc(
                [v['penalized_max_degrees'] for v in successes]))) if successes else None}


def choice_result(arm, outcomes):
    if arm is None:
        return {'status': 'unavailable', 'arm': None, 'error': None,
                'penalized_max_degrees': 180.0, 'penalized_max_signfolded_degrees': 180.0}
    return {**outcomes[arm], 'arm': arm}


def omission_support(record):
    """Use exact pre-final mask denominator, never outliers from all-match counts."""
    if record.get('status') != 'completed':
        return 0, 0
    diag = record.get('depth_diagnostics') or {}
    mask = diag.get('returned_mask') or {}
    denominator = mask.get('denominator')
    inactive = mask.get('native_branch_inactive_count')
    require(type(denominator) is int and denominator >= 0, 'invalid masked denominator')
    require(type(inactive) is int and 0 <= inactive <= 2 * denominator, 'invalid inactive count')
    return denominator, inactive


def evaluate_records(prepared, ground_truth):
    rows = []
    for item in prepared:
        identity, bank = item['identity'], item['records']
        gt = ground_truth[identity['pair_index']]
        Rgt, projection = metrics.project_gt_rotation(gt['R'])
        tgt, norm = metrics.unit_translation(gt['t'])
        gt_meta = {'R_original': gt['R'], 't_original': gt['t'], 'R_projected': Rgt,
                   't_unit': tgt, 'translation_original_norm': norm, **projection}
        outcomes = {}
        for arm in ARMS:
            record = bank[arm]
            if record['status'] == 'completed':
                est = record['estimate']
                error, _ = metrics.pose_errors(est['R'], est['t'], gt['R'], gt['t'])
                folded = min(error['translation_degrees'], 180 - error['translation_degrees'])
                error['translation_signfolded_degrees'] = folded
                error['max_signfolded_degrees'] = max(error['rotation_degrees'], folded)
                outcomes[arm] = {'status': 'completed', 'error': error,
                                 'penalized_max_degrees': error['max_degrees'],
                                 'penalized_max_signfolded_degrees': error['max_signfolded_degrees']}
            else:
                outcomes[arm] = {'status': record['status'], 'error': None,
                                 'penalized_max_degrees': 180.0, 'penalized_max_signfolded_degrees': 180.0}
        choices = {'pixel_msac_external': choice_result(item['selected_arm'], outcomes)}
        for name, arms in (('oracle_geometry', ARMS[:2]), ('oracle_depth', ARMS[2:]),
                           ('oracle_all', ARMS)):
            eligible = [a for a in arms if outcomes[a]['status'] == 'completed']
            chosen = min(eligible, key=lambda a: (outcomes[a]['penalized_max_degrees'], ARMS.index(a))) if eligible else None
            choices[name] = choice_result(chosen, outcomes)
        comparisons, mechanism = {}, {}
        for depth in ARMS[2:]:
            comparisons[depth] = {}
            d = outcomes[depth]['penalized_max_degrees']
            for geo in ARMS[:2]:
                g = outcomes[geo]['penalized_max_degrees']
                comparisons[depth][geo] = {'depth_minus_geometry_degrees': d - g,
                    'strict_win': d < g, 'strict_loss': d > g, 'exact_tie': d == g,
                    'rescue_below5': g >= 5 and d < 5, 'damage_above5': g < 5 and d >= 5,
                    'material_benefit': g - d >= 1 or (g >= 5 and d < 5),
                    'material_harm': d - g >= 1 or (g < 5 and d >= 5),
                    'both_completed': outcomes[depth]['status'] == outcomes[geo]['status'] == 'completed'}
            denominator, inactive = omission_support(bank[depth])
            gs = [outcomes[a]['penalized_max_degrees'] for a in ARMS[:2]]
            both_geometries_completed = all(outcomes[a]['status'] == 'completed' for a in ARMS[:2])
            joint_harm = all(d - g >= 1 for g in gs) or (d >= 5 and all(g < 5 for g in gs))
            mechanism[depth] = {'returned_mask_denominator': denominator,
                                'native_branch_inactive_count': inactive,
                                'joint_material_harm_vs_both': joint_harm,
                                'qualifies': outcomes[depth]['status'] == 'completed' and both_geometries_completed
                                    and joint_harm and denominator > 0 and inactive > 0}
        rows.append({'pair_index': identity['pair_index'], 'scene': identity['scene'], 'key': identity['key'],
                     'ground_truth': gt_meta, 'outcomes': outcomes, 'choices': choices,
                     'comparisons': comparisons, 'mechanism': mechanism,
                     'timings_by_arm': {a: bank[a]['timings'] for a in ARMS},
                     'selector_charged_seconds': item['selector_charged_seconds']})
    methods = list(ARMS) + ['pixel_msac_external', 'oracle_geometry', 'oracle_depth', 'oracle_all']
    def summary_for(subset):
        result = {}
        for method in methods:
            result[method] = summarize([r['outcomes'][method] if method in ARMS else r['choices'][method] for r in subset])
        return result
    summaries = summary_for(rows)
    scene_summaries = {s: summary_for([r for r in rows if r['scene'] == s]) for s in SCENES}
    loso = {s: summary_for([r for r in rows if r['scene'] != s]) for s in SCENES}
    auc5 = lambda method: summaries[method]['auc_fraction']['5']
    headroom = 100 * (auc5('oracle_all') - auc5('oracle_geometry'))
    mechanism_scenes = {a: [s for s in SCENES if any(r['scene'] == s and r['mechanism'][a]['qualifies'] for r in rows)] for a in ARMS[2:]}
    comparisons_summary = {a: {g: {name: sum(r['comparisons'][a][g][name] for r in rows)
        for name in ('strict_win','strict_loss','exact_tie','rescue_below5','damage_above5','material_benefit','material_harm','both_completed')}
        for g in ARMS[:2]} for a in ARMS[2:]}
    return metrics.json_safe({'rows': rows, 'summaries': summaries, 'scene_summaries': scene_summaries,
        'leave_one_scene_out_descriptive': loso, 'comparisons_summary': comparisons_summary,
        'decisions': {'additional_depth_oracle_headroom_pp': headroom,
            'selector_necessary_headroom_pass': headroom >= 2.0,
            'oracle_all_minus_best_single_pp': 100 * (auc5('oracle_all') - max(auc5(a) for a in ARMS)),
            'omission_mechanism_scenes_by_depth_arm': mechanism_scenes,
            'omission_mechanism_trigger_by_arm': {a: len(mechanism_scenes[a]) >= 2 for a in ARMS[2:]},
            'new_method_improvement_established': False, 'manuscript_admitted': False},
        'charged_seconds_by_arm': {a: sum(r['timings_by_arm'][a]['charged_total_seconds'] for r in rows) for a in ARMS},
        'external_selector_charged_seconds': sum(r['selector_charged_seconds'] for r in rows)})


def evaluate(data_dir, manifest_path, seal_path, output_path):
    import sys
    sys.path.insert(0, str(HERE))
    import common
    import run_diagnostic
    output_path = Path(output_path)
    require(not output_path.exists(), 'output already exists')
    manifest, seal, records = run_diagnostic.verify_seal(manifest_path, seal_path)
    identities = manifest['selected_pairs']
    prepared = prepare_records(records, identities)
    source_hashes = common.source_hashes()
    manifest_hash, seal_hash = common.sha256(manifest_path), common.sha256(seal_path)
    data_dir = Path(data_dir)
    for scene in manifest['scenes']:
        data = data_dir / scene['filename']
        require(data.stat().st_size == scene['bytes'] and common.sha256(data) == scene['sha256'],
                'private cache hash differs before label access')
    import h5py
    ground_truth = {}
    for scene in manifest['scenes']:
        with h5py.File(data_dir / scene['filename'], 'r') as handle:
            for identity in identities:
                if identity['scene'] == scene['scene']:
                    group = handle[identity['key']]
                    ground_truth[identity['pair_index']] = {'R': group['R'][:], 't': group['t'][:]}
    results = evaluate_records(prepared, ground_truth)
    require(common.source_hashes() == source_hashes, 'source changed during evaluation')
    require(common.sha256(manifest_path) == manifest_hash and common.sha256(seal_path) == seal_hash,
            'manifest/seal changed during evaluation')
    run_diagnostic.verify_seal(manifest_path, seal_path)
    for scene in manifest['scenes']:
        require(common.sha256(data_dir / scene['filename']) == scene['sha256'], 'cache changed during evaluation')
    output = {'schema': 'evr0017.evaluation.v1', 'status': 'completed', 'pair_count': COUNT,
        'manifest_sha256': manifest_hash, 'seal_sha256': seal_hash, 'source_sha256': source_hashes,
        'oracle_warning': 'GT-dependent diagnostic bounds only; none is a deployable method.',
        'scope': 'Five permanently development scenes; cached frontend cost/checkpoints unavailable. Known-method diagnostic, not a novel method or final test.',
        'failure_penalty': '180 degrees in all-pair metrics; successful-only metrics separate.',
        'mechanism_warning': 'Final residual-inactivity counts are association only; qualifying cases require separately frozen matched-initialization LO evidence.',
        **results}
    common.write_new(output_path, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir', 'manifest', 'seal', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    evaluate(args.data_dir, args.manifest, args.seal, args.output)


if __name__ == '__main__':
    main()
