"""Evaluator-only GT join after selection lock; no generation or scoring calls."""
import numpy as np

from .geometry_evaluation import evaluate_transfer
from .geometric_stage0 import candidate_pool, array_sha256

FAMILIES = ('uniform', 'clustered_outlier', 'compact_true', 'sparse_isolated')
CONTROLS = ('raw', 'msac', 'cell', 'excess', 'ratio')
METHODS = CONTROLS + ('tail',)


def decision_gate(counts, paired, *, complete, primary_net_gain=8, max_harms=2):
    primary = counts['clustered_outlier']
    # Aggregate control tie priority is the fixed CONTROLS order.
    best = max(CONTROLS, key=lambda name: primary[name])
    gain = primary['tail'] - primary[best]
    harms = {f: {m: paired[f][m]['baseline_success_tail_failure'] for m in CONTROLS}
             for f in ('uniform', 'compact_true', 'sparse_isolated')}
    harmless = all(value <= max_harms for family in harms.values() for value in family.values())
    eligible = bool(complete and gain >= primary_net_gain and harmless)
    return {'decision': 'ELIGIBLE_PENDING_EXTERNAL_REVIEW' if eligible else 'NO_GO',
            'coverage_complete': bool(complete), 'primary_strongest_control': best,
            'primary_net_gain': gain, 'primary_gain_pass': gain >= primary_net_gain,
            'paired_harms_15': harms, 'paired_harm_pass': harmless,
            'insufficient_primary_headroom': primary[best] > 56,
            'independent_resource_and_saved_result_review': 'not_attested_by_this_evaluator'}


def _validate_selection(value, order):
    if not isinstance(value,dict):
        raise ValueError('missing selection record')
    selected, ties = value.get('selected_id'), value.get('tied_ids')
    if (type(selected) is not int or not isinstance(ties,list) or not ties or
            any(type(v) is not int for v in ties) or len(ties) != len(set(ties)) or
            ties != [v for v in order if v in set(ties)] or selected != ties[0]):
        raise ValueError('invalid first-in-order selection/ties')


def evaluate_cases(records, labels, *, expected_ids):
    """Records/labels keyed by case ID; all expected IDs remain in denominators."""
    if set(records) != set(expected_ids) or set(labels) - set(expected_ids):
        raise ValueError('unexpected or missing case-record inventory')
    counts = {f: {m: 0 for m in METHODS} for f in FAMILIES}
    totals = {f: 0 for f in FAMILIES}
    paired = {f: {m: {'tail_success_baseline_failure': 0,
                     'baseline_success_tail_failure': 0, 'both_success': 0,
                     'both_failure': 0} for m in CONTROLS} for f in FAMILIES}
    rows, complete = [], True
    for case_id in expected_ids:
        rec = records[case_id]
        family, seed_text = case_id.rsplit('-',1)
        if rec.get('family') != family or rec.get('seed_index') != int(seed_text):
            raise ValueError('case family/seed does not match canonical ID')
        if family not in FAMILIES:
            raise ValueError('unknown recorded family')
        totals[family] += 1
        if case_id in labels:
            truth = labels[case_id]
            if truth.get('true_candidate_id') != 4 or not np.array_equal(
                    np.asarray(truth.get('H_star'),dtype=np.float64),np.eye(3)):
                raise ValueError('evaluator label differs from fixed identity GT')
        if rec.get('status') not in ('completed','tail_failed','incomplete','invalid','unattempted'):
            raise ValueError('unknown case status')
        if 'inputs' in rec:
            pool = np.asarray(rec['inputs']['candidates'],dtype=np.float64)
            if not np.array_equal(pool,candidate_pool()):
                raise ValueError('saved candidate pool differs from fixed source')
            order = rec['inputs']['candidate_order']
            if len(order) != 32 or any(type(v) is not int for v in order) or set(order) != set(range(32)):
                raise ValueError('invalid candidate order')
            for row in rec.get('hypotheses',[]):
                cid = row['canonical_id']
                if type(cid) is not int or not 0 <= cid < 32 or row['H_sha256'] != array_sha256(pool[cid]):
                    raise ValueError('per-H source binding differs')
        if rec.get('controls_complete') is True:
            if 'inputs' not in rec or [h['canonical_id'] for h in rec.get('hypotheses',[])] != rec['inputs']['candidate_order']:
                raise ValueError('completed controls lack the full ordered hypothesis record')
            for method in CONTROLS:
                _validate_selection(rec.get('selections',{}).get(method),rec['inputs']['candidate_order'])
        if rec['status'] == 'completed':
            _validate_selection(rec.get('selections',{}).get('tail'),rec['inputs']['candidate_order'])
        ok = (rec['status'] == 'completed' and rec.get('controls_complete') is True
              and len(rec.get('hypotheses',[])) == 32
              and rec.get('tail_attempted') == 32
              and all(h['tail']['status'] == 'completed' for h in rec['hypotheses']))
        complete = complete and ok
        results = {}
        for method in METHODS:
            selected = rec.get('selections',{}).get(method,{}).get('selected_id')
            error, offset = None, None
            if selected is not None:
                if rec.get('controls_complete') is not True:
                    raise ValueError('selection requires completed controls')
                if case_id not in labels:
                    raise ValueError('selected case lacks evaluator GT')
                if type(selected) is not int or not 0 <= selected < 32:
                    raise ValueError('invalid selected canonical ID')
                h = rec['inputs']['candidates'][selected]
                error = evaluate_transfer(h, labels[case_id]['H_star'])
                offset = [h[0][2], h[1][2]]
            success = error is not None and error <= 3.
            if method == 'tail' and not ok:
                if selected is not None:
                    raise ValueError('failed tail case cannot retain a selection')
                success = False
            counts[family][method] += int(success)
            results[method] = {'selected_id': selected, 'offset': offset,
                               'transfer_error': error, 'success': success}
        for control in CONTROLS:
            a, b = results['tail']['success'], results[control]['success']
            key = ('both_success' if a and b else 'tail_success_baseline_failure' if a else
                   'baseline_success_tail_failure' if b else 'both_failure')
            paired[family][control][key] += 1
        rows.append({'case_id': case_id, 'family': family, 'status': rec['status'],
                     'methods': results})
    complete = complete and totals == {f:64 for f in FAMILIES}
    return {'schema': 'evr0009.evaluation.v1', 'metric': 'mean_one_way_euclidean_81_grid_0_512',
            'success_pixels': 3., 'family_totals': totals, 'success_counts': counts,
            'paired': paired, 'cases': rows,
            'gates': decision_gate(counts, paired, complete=complete),
            'scope': 'Oracle-covered synthetic translation-ranking development only; no calibrated false-match probability or real-image generality.'}
