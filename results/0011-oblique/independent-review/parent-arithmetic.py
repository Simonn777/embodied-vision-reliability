"""Independent stdlib-only audit of saved 0011 errors and diagnostics.

No frozen scorer imports, RGB, raw GT, native arrays, model calls, or retuning.
--root is the extracted evidence directory containing scratch/0011-oblique-v1.
Only this new receipt is written; saved evidence is never modified.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import time

GROUPS = ('g-036', 'g-002')
FRAMES = tuple(range(0, 241, 16))
METHODS = ('original_native', 'reference_native', 'oblique_native', 'axis_control_native',
           'reference_independent', 'coordinate_average', 'entropy_select', 'product_joint',
           'mixture_joint', 'oblique_coupling', 'axis_coupling')
CANDIDATE = 'oblique_coupling'
THRESHOLDS = (1, 2, 4, 8, 16)
PREFIX = Path('scratch/0011-oblique-v1')
FLOAT_TOLERANCE = 1e-10


def file_record(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def finite(value, nonnegative=False):
    if type(value) not in (float, int) or not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError('invalid finite numeric field')
    return value


def average(values):
    return math.fsum(values) / len(values) if values else None


def identity(row):
    g, f, p = row['group_id'], row['frame_id'], row['point_id']
    if g not in GROUPS or type(f) is not int or f not in FRAMES or type(p) is not int or not 0 <= p < 30:
        raise ValueError('invalid group/frame/point identity')
    return g, f, p


def index_grid(rows):
    if len(rows) != 256:
        raise ValueError('expected 256 saved rows')
    result = {}
    for r in rows:
        key = identity(r)
        if key in result:
            raise ValueError('duplicate saved identity')
        result[key] = r
    ids = {g: sorted(p for gg, f, p in result if gg == g and f == 0) for g in GROUPS}
    if any(len(v) != 8 for v in ids.values()) or set(result) != {
            (g, f, p) for g in GROUPS for f in FRAMES for p in ids[g]}:
        raise ValueError('missing group/frame/point combination')
    return result, ids


def recompute(cases):
    indexed, ids = index_grid(cases)
    for r in cases:
        if type(r['occluded']) is not bool or set(r['errors']) != set(METHODS):
            raise ValueError('error/occlusion schema differs')
        for value in r['errors'].values():
            finite(value, True)
    groups, denominators, exact = {}, {}, {}
    for g in GROUPS:
        all_rows = [r for k, r in indexed.items() if k[0] == g and k[1] != 0]
        visible = [r for r in all_rows if not r['occluded']]
        hidden = [r for r in all_rows if r['occluded']]
        if not visible:
            raise ValueError('empty eligible group')
        denominators[g] = {'saved': 128, 'initial_query_excluded': 8, 'postquery': len(all_rows),
                           'visible_eligible': len(visible), 'occluded_postquery': len(hidden), 'point_ids': ids[g]}
        methods, paired, exact[g] = {}, {}, {}
        for m in METHODS:
            errors = [r['errors'][m] for r in visible]
            counts = [sum(e < threshold for e in errors) for threshold in THRESHOLDS]
            exact[g][m] = Fraction(sum(counts), len(errors)*len(THRESHOLDS))
            methods[m] = {'quality': float(exact[g][m]), 'hit_counts': counts,
                          'eligible': len(errors), 'coverage': 1.0,
                          'mean_epe': average(errors), 'median_epe': statistics.median(errors),
                          'all_point_epe': average([r['errors'][m] for r in all_rows]),
                          'occluded_epe': average([r['errors'][m] for r in hidden])}
            if m != CANDIDATE:
                paired[m] = {
                    'corrections_4px': sum(r['errors'][CANDIDATE] < 4 <= r['errors'][m] for r in visible),
                    'harms_4px': sum(r['errors'][m] < 4 <= r['errors'][CANDIDATE] for r in visible)}
        groups[g] = {'methods': methods, 'paired': paired}
    exact_macro = {m: sum((exact[g][m] for g in GROUPS), Fraction())/2 for m in METHODS}
    macro = {m: {'quality': float(exact_macro[m]), 'mean_epe': average([
        groups[g]['methods'][m]['mean_epe'] for g in GROUPS])} for m in METHODS}
    controls = [m for m in METHODS if m != CANDIDATE]
    best = max(controls, key=lambda m: exact_macro[m])  # First fixed-order tie.
    gain = exact_macro[CANDIDATE]-exact_macro[best]
    group_gains = {g: exact[g][CANDIDATE]-max(exact[g][m] for m in controls) for g in GROUPS}
    gates = {'macro_gain_at_least_0.05': gain >= Fraction(1, 20),
             'each_group_above_every_control': all(x > 0 for x in group_gains.values()),
             'macro_epe_not_worse_than_best_quality_control': macro[CANDIDATE]['mean_epe'] <= macro[best]['mean_epe']}
    return {'groups': groups, 'macro': macro, 'strongest_control': best,
            'macro_quality_gain': float(gain), 'macro_quality_gain_exact': str(gain),
            'group_quality_gains_over_strongest': {g: float(x) for g, x in group_gains.items()},
            'quality_gates': gates}, denominators


def descriptive_numerics(diagnostics, predictions):
    di, ids = index_grid(diagnostics)
    pi, pi_ids = index_grid(predictions['rows'])
    if ids != pi_ids or tuple(predictions['methods']) != METHODS:
        raise ValueError('derived identities/method order differ')
    solvers = ('oblique_coupling', 'axis_coupling')
    stats = {m: {'iterations': [], 'row_error': [], 'column_error': [], 'residual': [], 'mean': []} for m in solvers}
    axis, parity, ties = [], [], {}
    converged = True
    for key, d in di.items():
        p = pi[key]
        for row in (d, p):
            if row['observation_index'] != key[1]//16 or row['point_index'] != ids[key[0]].index(key[2]):
                raise ValueError('derived positional identity differs')
        if set(p['predictions']) != set(METHODS) or set(d['solvers']) != set(solvers):
            raise ValueError('derived methods/solvers incomplete')
        for xy in p['predictions'].values():
            if len(xy) != 2:
                raise ValueError('coordinate shape invalid')
            for v in xy: finite(v)
        parity.append(max(abs(a-b) for a,b in zip(p['predictions']['reference_native'], p['predictions']['reference_independent'])))
        if abs(parity[-1]-finite(d['reference_native_parity_max_abs_difference_original'], True)) > FLOAT_TOLERANCE:
            raise ValueError('saved parity diagnostic differs from saved coordinates')
        axis.append(finite(d['axis_joint_max_abs_difference'], True))
        for m in solvers:
            s = d['solvers'][m]
            if type(s['iterations']) is not int or not 1 <= s['iterations'] <= 1000:
                raise ValueError('invalid saved iteration count')
            converged &= s['converged'] is True
            stats[m]['iterations'].append(s['iterations'])
            for n in ('row_error', 'column_error', 'residual'):
                stats[m][n].append(finite(s[n], True))
            if s['residual'] != max(s['row_error'], s['column_error']):
                raise ValueError('residual identity differs')
            delta = d['global_means'][m]['reference_delta_xy']
            if len(delta) != 2:
                raise ValueError('mean delta shape invalid')
            maximum = max(abs(finite(v)) for v in delta)
            if abs(maximum-finite(d['global_means'][m]['reference_max_abs_difference'], True)) > FLOAT_TOLERANCE:
                raise ValueError('mean diagnostic identity differs')
            stats[m]['mean'].append(maximum)
        for m,j in d['joint_distributions'].items():
            n = j['exact_mode_tie_count']
            if type(n) is not int or n < 1:
                raise ValueError('invalid saved tie count')
            ties.setdefault(m, []).append(n)
    summary = {m: {'rows': len(s['iterations']), 'iterations_min': min(s['iterations']),
                    'iterations_max': max(s['iterations']), 'iterations_median': statistics.median(s['iterations']),
                    'iterations_mean': average(s['iterations']), 'row_error_max': max(s['row_error']),
                    'column_error_max': max(s['column_error']), 'residual_max': max(s['residual']),
                    'global_mean_max_difference_reference': max(s['mean'])} for m,s in stats.items()}
    gates = {'all_couplings_converged': converged and all(max(s['residual']) <= 1e-9 for s in stats.values()),
             'fixed_global_means': all(max(s['mean']) <= 5e-5 for s in stats.values()),
             'axis_joint_invariant': max(axis) <= 1e-7, 'reference_decoder_parity': max(parity) <= 1e-3}
    return gates, {'label': 'Post-hoc description of saved diagnostics; no parameter selection or rerun',
                   'scope': 'Joint residuals/means come from saved diagnostics, not regenerated joints; parity is recalculated from saved coordinates.',
                   'solvers': summary, 'axis_joint_max_difference': max(axis), 'reference_parity_max_original': max(parity),
                   'mode_ties': {m: {'records': len(v), 'maximum': max(v), 'records_with_multiple_modes': sum(n > 1 for n in v)} for m,v in ties.items()}}


def compare(expected, actual, path='', stats=None):
    if stats is None:
        stats = {'leaves_checked': 0, 'floating_leaves': 0, 'max_absolute_float_difference': 0.0, 'mismatches': []}
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            stats['mismatches'].append(path+': mapping keys differ'); return stats
        for k,v in expected.items(): compare(v, actual[k], path+'/'+k, stats)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            stats['mismatches'].append(path+': list size differs'); return stats
        for i,v in enumerate(expected): compare(v, actual[i], path+'/'+str(i), stats)
    else:
        stats['leaves_checked'] += 1
        if type(expected) is float:
            stats['floating_leaves'] += 1
            if type(actual) not in (float,int) or not math.isfinite(actual):
                stats['mismatches'].append(path+': invalid numeric'); return stats
            error = abs(expected-actual)
            stats['max_absolute_float_difference'] = max(stats['max_absolute_float_difference'], error)
            if error > FLOAT_TOLERANCE: stats['mismatches'].append(path+': numeric mismatch')
        elif type(expected) is not type(actual) or expected != actual:
            stats['mismatches'].append(path+': exact mismatch')
    return stats


def audit(root):
    start = time.monotonic(); root = Path(root); records = {}
    def load(relative):
        path = root/PREFIX/relative
        records[str(PREFIX/relative)] = file_record(path)
        return json.loads(path.read_text())
    evaluation_receipt = load('evaluation/receipt.json')
    seal = load('derived/seal.json')
    cases = load('evaluation/case-errors.json'); saved = load('evaluation/results.json')
    diagnostics = load('derived/diagnostics.json'); predictions = load('derived/predictions.json')
    for n in ('case-errors.json', 'results.json'):
        if records[str(PREFIX/'evaluation'/n)] != evaluation_receipt['files'][n]:
            raise ValueError('evaluation receipt hash mismatch')
    for n in ('diagnostics.json', 'predictions.json'):
        if records[str(PREFIX/'derived'/n)] != seal['files'][n]:
            raise ValueError('derived seal hash mismatch')
    if (records[str(PREFIX/'derived/seal.json')] != evaluation_receipt['seal']
            or saved['seal_sha256'] != evaluation_receipt['seal']['sha256']):
        raise ValueError('seal chain mismatch')
    recomputed, denominators = recompute(cases)
    # Align all saved-error identities with the previously sealed derived table.
    case_index,_ = index_grid(cases); pred_index,_ = index_grid(predictions['rows'])
    if set(case_index) != set(pred_index): raise ValueError('errors/derived identity mismatch')
    numerical, posthoc = descriptive_numerics(diagnostics, predictions)
    recomputed.update(numerical_gates=numerical, status='completed_evaluation',
                      decision='GO' if all(recomputed['quality_gates'].values()) and all(numerical.values()) else 'NO_GO', model_calls=0)
    compared = compare(recomputed, {k:saved[k] for k in recomputed})
    paired_total = {m:{k:sum(recomputed['groups'][g]['paired'][m][k] for g in GROUPS)
                         for k in ('corrections_4px','harms_4px')} for m in METHODS if m != CANDIDATE}
    before = dict(records)
    for name,record in before.items():
        if file_record(root/name) != record: raise ValueError('evidence changed during audit')
    return {'status':'passed' if not compared['mismatches'] else 'failed',
        'scope':'Independent standard-library recalculation from saved errors; not a second GT/coordinate truth validation.',
        'limits':['No raw GT, RGB, NPZ, model, frozen scorer or prior checker imported/read.',
                  'Saved solver diagnostics are described post hoc; raw-logit replay is a separate review.',
                  'Two simulated development groups do not establish population uncertainty, novelty or publication-level improvement.'],
        'source':{'relative_path':'results/0011-oblique/independent-review/parent-arithmetic.py',**file_record(Path(__file__))},
        'runtime':{'python':platform.python_version(),'implementation':platform.python_implementation(),'stdlib_only':True},
        'inputs':records,'inputs_unchanged_after_review':True,'comparison_absolute_tolerance':FLOAT_TOLERANCE,
        'arithmetic_comparison':compared,'denominators':denominators,
        'counts':{'saved_rows':len(cases),'methods':len(METHODS),'finite_saved_errors':len(cases)*len(METHODS),
                  'query_rows_excluded':16,'postquery_rows':240,'visible_eligible':sum(d['visible_eligible'] for d in denominators.values())},
        'definitions':{'hits':'error < threshold, strictly','thresholds':list(THRESHOLDS),
                       'macro':'equal weight for each of two groups, not pooled frame/point weighting',
                       'paired':'candidate <4 and comparator >=4 is correction; comparator <4 and candidate >=4 is harm',
                       'tie_control':'first in fixed METHODS order'},
        'recomputed':recomputed,'paired_totals_across_groups_descriptive':paired_total,
        'posthoc_saved_numerical_diagnostics':posthoc,'wall_seconds':time.monotonic()-start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    # Exclusive creation preserves previous parent audits and original files.
    with args.output.open('x') as out:
        report=audit(args.root)
        json.dump(report,out,indent=2,allow_nan=False);out.write('\n')
    print(json.dumps({'status':report['status'],'decision':report['recomputed']['decision'],
                      'comparison':report['arithmetic_comparison']}))
    return 0 if report['status']=='passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
