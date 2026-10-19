#!/usr/bin/env python3
"""Independent saved-only 0016 review. No experiment imports, HDF5, or estimators.

NumPy is used only for SVD of the saved reference rotation. Dot products,
matrix products, angles, selection, and DMS trapezoids are implemented here.
The input coordinates are intentionally unavailable: final residual scores are
bound/copied/used for selection, not numerically recomputed from correspondences.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

BACKENDS = ('old_norm', 'current_norm', 'current_ts')
SEEDS = (0, 1, 2, 3)
BUDGETS = (0.1, 0.25, 0.5, 1.0, 2.0)
SALT = 'evr0016:development:2026-10-19\0'
ANGLE_ATOL = 3e-6  # acos at an endpoint amplifies scalar reduction roundoff.
AUC_ATOL = 1e-7
SOURCE_PATHS = {
    'experiments/0016-search-headroom/' + name for name in
    ('common.py', 'allocate.py', 'predict_pair.py', 'run_diagnostic.py', 'evaluate_saved.py')
} | {
    'protocols/0016-search-headroom.md', 'configs/0016-libraries.json',
    'experiments/0013-pose-headroom/predict_pair.py',
    'experiments/0013-pose-headroom/run_baseline.py',
    'experiments/0013-pose-headroom/evaluate_sealed.py',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite_tree(value):
    if isinstance(value, float):
        require(math.isfinite(value), 'nonfinite JSON value')
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)
    elif isinstance(value, dict):
        for item in value.values():
            finite_tree(item)


def load(path):
    raw = Path(path).read_bytes()
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key ' + key)
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=unique)
    finite_tree(value)
    return value, hashlib.sha256(raw).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, label)
    return float(value)


def close(a, b, label, atol=1e-9, rtol=1e-12):
    require(type(a) in (int, float) and type(b) in (int, float)
            and math.isfinite(a) and math.isfinite(b)
            and math.isclose(a, b, abs_tol=atol, rel_tol=rtol),
            '%s differs: %r versus %r' % (label, a, b))


def compare(a, b, label, atol=1e-9):
    if isinstance(a, dict):
        require(isinstance(b, dict) and set(a) == set(b), label + ' keys differ')
        for key in a:
            compare(a[key], b[key], label + '.' + key, atol)
    elif isinstance(a, list):
        require(isinstance(b, list) and len(a) == len(b), label + ' length differs')
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, label + '[%d]' % i, atol)
    elif type(a) in (int, float) and type(b) in (int, float):
        close(a, b, label, atol)
    else:
        require(type(a) is type(b) and a == b, label + ' differs')


def matrix(value):
    require(isinstance(value, list) and len(value) == 3
            and all(isinstance(row, list) and len(row) == 3 for row in value), 'matrix shape')
    require(all(type(x) in (int, float) and math.isfinite(x) for row in value for x in row), 'matrix finite')
    return [[float(x) for x in row] for row in value]


def determinant(a):
    return math.fsum((a[0][0] * a[1][1] * a[2][2], a[0][1] * a[1][2] * a[2][0],
                      a[0][2] * a[1][0] * a[2][1], -a[0][2] * a[1][1] * a[2][0],
                      -a[0][1] * a[1][0] * a[2][2], -a[0][0] * a[1][2] * a[2][1]))


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def multiply(a, b):
    return [[dot(a[i], [b[k][j] for k in range(3)]) for j in range(3)] for i in range(3)]


def proper_rotation(value):
    r = matrix(value)
    require(all(abs(dot(r[i], r[j]) - float(i == j)) <= 1e-6
                for i in range(3) for j in range(3)), 'rotation not orthogonal')
    require(abs(determinant(r) - 1) <= 1e-6, 'rotation not proper')
    return r


def unit(value):
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    elif isinstance(value, list) and len(value) == 3 and all(isinstance(v, list) and len(v) == 1 for v in value):
        value = [v[0] for v in value]
    require(isinstance(value, list) and len(value) == 3
            and all(type(x) in (int, float) and math.isfinite(x) for x in value), 'translation shape/finite')
    scale = max(abs(x) for x in value)
    require(scale > 0, 'zero translation')
    scaled = [x / scale for x in value]
    length = math.sqrt(dot(scaled, scaled))
    norm = scale * length
    require(math.isfinite(norm), 'unrepresentable translation norm')
    return [x / length for x in scaled], norm


def angle(cosine):
    return math.degrees(math.acos(min(1., max(-1., cosine))))


def rotation_angle(a, b):
    # trace(A B^T) is the Frobenius inner product; no BLAS reduction.
    return angle((math.fsum(a[i][j] * b[i][j] for i in range(3) for j in range(3)) - 1) / 2)


def reference(saved):
    raw = matrix(saved['R_original'])
    U, singular, Vt = np.linalg.svd(np.asarray(raw, dtype=float))
    projected = multiply(U.tolist(), Vt.tolist())
    require(min(singular) > 0 and determinant(projected) > 0, 'invalid GT SVD projection')
    proper_rotation(projected)
    t, norm = unit(saved['t_original'])
    compare(projected, saved['R_projected'], 'GT projected rotation', 2e-12)
    compare(t, saved['t_unit'], 'GT unit translation', 2e-12)
    close(norm, saved['translation_original_norm'], 'GT translation norm')
    if 'singular_values' in saved:
        compare(singular.tolist(), saved['singular_values'], 'GT singular values', 2e-12)
    if 'projection_frobenius_change' in saved:
        change = math.hypot(*(projected[i][j] - raw[i][j] for i in range(3) for j in range(3)))
        close(change, saved['projection_frobenius_change'], 'GT projection change', 2e-12)
    if 'projected_determinant' in saved:
        close(determinant(projected), saved['projected_determinant'], 'GT determinant', 2e-12)
    return projected, t


def errors(estimate, gt):
    R = proper_rotation(estimate['R'])
    t, _ = unit(estimate['t'])
    r = rotation_angle(R, gt[0])
    d = angle(dot(t, gt[1]))
    folded = min(d, 180. - d)
    return {'rotation_degrees': r, 'translation_degrees': d, 'max_degrees': max(r, d),
            'translation_signfolded_degrees': folded, 'max_signfolded_degrees': max(r, folded)}


def dms_auc(values, threshold):
    require(values and threshold > 0 and all(math.isfinite(v) and v >= 0 for v in values), 'AUC input')
    n = len(values)
    x_previous = y_previous = 0.
    pieces = []
    for rank, value in enumerate(sorted(values), start=1):
        if value >= threshold:  # DMS uses left insertion: exact ties are excluded.
            break
        y = rank / n
        pieces.append((value - x_previous) * (y + y_previous) / 2)
        x_previous, y_previous = value, y
    pieces.append((threshold - x_previous) * y_previous)
    return math.fsum(pieces) / threshold


def expected_options(backend, seed):
    r = {'min_iterations': 1000, 'max_iterations': 100000, 'dyn_num_trials_mult': 3.,
         'success_prob': .9999, 'seed': seed, 'progressive_sampling': False,
         'max_prosac_iterations': 100000}
    if backend == 'old_norm':
        return {**r, 'max_epipolar_error': 1., 'real_focal_check': False}
    return {'max_error': 1., 'tangent_sampson': backend == 'current_ts', 'real_focal_check': False,
            'ransac': {**r, 'score_initial_model': False}, 'bundle': {}}


def choice(candidates, budget=None):
    bank = [c for c in candidates if c['status'] == 'completed'
            and (budget is None or c['charged_end_seconds'] <= budget)]
    return min(bank, key=lambda c: (c['final_scores']['pixel_msac'], c['seed'])) if bank else None


def selection_record(candidate, candidate_errors):
    if candidate is None:
        return {'status': 'unavailable', 'seed': None, 'error': None,
                'penalized_max_degrees': 180., 'penalized_max_signfolded_degrees': 180.}
    e = candidate_errors[candidate['seed']]
    return {'status': 'completed', 'seed': candidate['seed'], 'error': e,
            'pixel_msac': candidate['final_scores']['pixel_msac'],
            'charged_end_seconds': candidate['charged_end_seconds'],
            'penalized_max_degrees': e['max_degrees'],
            'penalized_max_signfolded_degrees': e['max_signfolded_degrees']}


def check_oracle(candidates, independent_errors, evaluation_candidates, recorded_seed):
    """Check exact saved arithmetic and expose scalar-roundoff argmin changes.

    This does not relax any deployed selector or angle tolerance. A changed
    scalar oracle identity is permitted only if the stored oracle is the exact
    argmin of stored errors AND its independent regret is within ANGLE_ATOL.
    """
    successful = [c for c in candidates if c['status'] == 'completed']
    if not successful:
        require(recorded_seed is None, 'unavailable oracle seed')
        return None, None, None
    saved_errors = {c['seed']: c['error']['max_degrees'] for c in evaluation_candidates
                    if c['status'] == 'completed'}
    exact = min(successful, key=lambda c: (saved_errors[c['seed']], c['seed']))
    require(exact['seed'] == recorded_seed, 'saved oracle is not its exact declared argmin')
    independent = min(successful, key=lambda c: (independent_errors[c['seed']]['max_degrees'], c['seed']))
    regret = independent_errors[exact['seed']]['max_degrees'] - independent_errors[independent['seed']]['max_degrees']
    require(0. <= regret <= ANGLE_ATOL, 'oracle independent error regret exceeds angle tolerance')
    discrepancy = None
    if exact['seed'] != independent['seed']:
        discrepancy = {'saved_seed': exact['seed'], 'independent_scalar_seed': independent['seed'],
                       'independent_error_regret_degrees': regret,
                       'saved_seed_independent_error_degrees': independent_errors[exact['seed']]['max_degrees'],
                       'scalar_best_error_degrees': independent_errors[independent['seed']]['max_degrees']}
    return exact, independent, discrepancy


def summarize(rows):
    require(len(rows) == 64, 'summary must retain all64')
    out = {'pair_count': 64, 'status_counts': dict(Counter(r['status'] for r in rows)),
           'selected_seed_counts': dict(Counter(str(r['seed']) for r in rows if r['seed'] is not None))}
    for name, field in (('directed_primary', 'penalized_max_degrees'),
                        ('signfolded_secondary', 'penalized_max_signfolded_degrees')):
        values = [r[field] for r in rows]
        out[name] = {'auc_fraction': {str(t): dms_auc(values, t) for t in (5, 10, 20)},
                     'errors_degrees': values,
                     'count_strictly_below': {str(t): sum(v < t for v in values) for t in (5, 10, 20)}}
    return out


def audit(repo, manifest_path, seal_path, evaluation_path):
    repo, manifest_path, seal_path, evaluation_path = map(Path, (repo, manifest_path, seal_path, evaluation_path))
    m, mh = load(manifest_path)
    seal, sh = load(seal_path)
    ev, eh = load(evaluation_path)
    old_path = repo / 'configs/0012-development-manifest.json'
    old, oh = load(old_path)
    keys = m['selected_pair_keys']
    require(len(keys) == len(set(keys)) == 64 and all(k.startswith('0015-') for k in keys), 'allocation size/scene')
    require(not set(keys).intersection(old['selected_pair_keys']), 'previously exposed pair selected')
    require(m['selection_salt'] == SALT and keys == sorted(keys, key=lambda k: hashlib.sha256((SALT + k).encode()).hexdigest()), 'allocation order')
    require(m['excluded_manifest_sha256'] == oh and m['held_out_scenes'] == ['0022'], 'allocation exclusions')
    require(m['development_scene_pairs'] == 600 and m['coordinate_convention'] == 'raw_pixels', 'allocation metadata')
    require(m['dataset_sha256'] == old['dataset_sha256'] and m['dataset_bytes'] == old['dataset_bytes'], 'dataset declarations')
    require(seal['status'] == ev['status'] == 'completed', 'incomplete result')
    require(seal['manifest_sha256'] == ev['manifest_sha256'] == mh and ev['seal_sha256'] == sh, 'top-level bindings')
    require(ev['pair_count'] == 64 and ev['backend_count'] == 3 and ev['budgets_seconds'] == list(BUDGETS), 'evaluation protocol dimensions')
    require(ev['dataset_sha256'] == m['dataset_sha256'] and ev['dataset_bytes'] == m['dataset_bytes'], 'evaluation dataset declaration')
    sources = seal['source_sha256']
    require(set(sources) == SOURCE_PATHS and sources == ev['source_sha256'], 'source set')
    for relative, declared in sources.items():
        require(digest(repo / relative) == declared, 'source hash: ' + relative)
    libs, _ = load(repo / 'configs/0016-libraries.json')
    require(len(seal['rows']) == len(ev['rows']) == 192, '192-worker coverage')
    e_rows = {(r['backend'], r['pair_index']): r for r in ev['rows']}
    require(len(e_rows) == 192, 'duplicate evaluation row')
    expected = {(b, i) for b in BACKENDS for i in range(64)}
    require(set(e_rows) == expected, 'evaluation Cartesian product')
    rows, seen, ground = [], set(), {}
    scalar_oracles = {b: [] for b in BACKENDS}
    oracle_roundoff_ties = []
    candidate_status = {b: Counter() for b in BACKENDS}
    timing = {b: {'input_seconds': [], 'native_seconds': [], 'score_seconds': [],
                  'charged_total_seconds': [], 'process_wall_seconds': [],
                  'sampled_peak_rss_kib': [], 'prefix_costs': {str(v): [] for v in BUDGETS}}
              for b in BACKENDS}
    max_angle_difference = 0.
    warnings = []
    for sr in seal['rows']:
        b, i, key = sr['backend'], sr['pair_index'], sr['key']
        require(type(i) is int and (b, i) in expected and (b, i) not in seen and keys[i] == key, 'worker identity')
        seen.add((b, i))
        name = sr['filename']
        require(Path(name).name == name, 'unsafe worker path')
        path = seal_path.parent / name
        require(not path.is_symlink(), 'worker symlink')
        record, rh = load(path)
        process_path = seal_path.parent / (name + '.process.json')
        require(not process_path.is_symlink(), 'process symlink')
        process, ph = load(process_path)
        require(rh == sr['sha256'] and ph == sr['process_sha256'], 'record/process hashes')
        require(process['exit_code'] == 0 and not process['termination'] and not process.get('watchdog_error'), 'worker process failed')
        require(record['status'] == 'completed' and (record['backend'], record['pair_index'], record['key']) == (b, i, key), 'record identity/status')
        require(record['source_sha256'] == sources and record['manifest_sha256'] == mh, 'record source/input binding')
        require(record['library'] == libs['old' if b == 'old_norm' else 'current'], 'record library pin')
        er = e_rows[(b, i)]
        require(er['key'] == key and er.get('calibration') == record.get('calibration'), 'evaluation identity/calibration copy')
        candidates = record['candidates']
        require(len(candidates) == len(er['candidates']) == 4, 'four candidate coverage')
        own_choices = {'seed0': candidates[0] if candidates[0]['status'] == 'completed' else None}
        own_choices.update({'pixel_msac_budget_' + str(v): choice(candidates, v) for v in BUDGETS})
        own_choices['pixel_msac_full4'] = choice(candidates)
        # Choices above do not depend on the saved reference pose.
        gt = reference(er['ground_truth'])
        raw_gt = {k: er['ground_truth'][k] for k in ('R_original', 't_original')}
        if key in ground:
            require(raw_gt == ground[key], 'GT differs across backends')
        ground[key] = raw_gt
        candidate_errors = {}
        charged = number(record['input_seconds'], 'input time')
        close(er['input_seconds'], charged, 'evaluation input copy')
        last_end = 0.
        native_times, score_times = [], []
        for seed, (c, ec) in enumerate(zip(candidates, er['candidates'])):
            require(type(c['seed']) is int and c['seed'] == seed and c['options'] == expected_options(b, seed), 'candidate options/seed')
            # Evaluation uses "error" for pose error and overwrites the exception
            # message on failed candidates; the latter remains in the hashed raw record.
            require({k: v for k, v in ec.items() if k != 'error'} ==
                    {k: v for k, v in c.items() if k != 'error'}, 'candidate copied payload differs')
            require(c['status'] in ('completed', 'estimator_error'), 'candidate terminal status')
            candidate_status[b][c['status']] += 1
            call, score = number(c['call_seconds'], 'native duration'), number(c['score_seconds'], 'score duration')
            native_times.append(call); score_times.append(score)
            charged += call + score
            close(charged, c['charged_end_seconds'], 'charged prefix')
            require(c['charged_end_seconds'] >= last_end, 'prefix time not monotone')
            last_end = c['charged_end_seconds']
            if c['status'] == 'completed':
                for field in ('pixel_msac', 'normalized_msac', 'tangent_msac'):
                    require(number(c['final_scores'][field], 'score') <= 10000, 'score above cap')
                for field in ('pixel_inliers', 'normalized_inliers', 'tangent_inliers', 'cheiral_count'):
                    v = c['final_scores'][field]
                    require(type(v) is int and 0 <= v <= 10000, 'score count')
                mask = c['native_inliers_packbits_big_hex']
                require(len(mask) == 2500 and len(bytes.fromhex(mask)) == 1250, 'saved inlier mask length')
                own = errors(c['estimate'], gt)
                candidate_errors[seed] = own
                compare(own, ec['error'], 'candidate pose errors', ANGLE_ATOL)
                max_angle_difference = max(max_angle_difference, *(abs(own[k] - ec['error'][k]) for k in own))
            else:
                require(not c.get('estimate') and ec['error'] is None and score == 0., 'failed candidate coverage')
        close(charged, record['charged_total_seconds'], 'record total')
        close(charged, er['charged_total_seconds'], 'evaluation total copy')
        successful = [c for c in candidates if c['status'] == 'completed']
        oracle, scalar_oracle, discrepancy = check_oracle(candidates, candidate_errors, er['candidates'],
            er['selections']['oracle_best4_nondeployable']['seed'])
        scalar_oracles[b].append((i, selection_record(scalar_oracle, candidate_errors)))
        if discrepancy:
            oracle_roundoff_ties.append({'backend': b, 'pair_index': i, 'key': key, **discrepancy})
        own_choices['oracle_best4_nondeployable'] = oracle
        require(set(er['selections']) == set(own_choices), 'selector names')
        selections = {}
        for label, selected in own_choices.items():
            own = selection_record(selected, candidate_errors)
            recorded = er['selections'][label]
            compare(own, {k: recorded[k] for k in own}, 'selection ' + label, ANGLE_ATOL)
            require(recorded.get('reason') is None if selected is not None else bool(recorded.get('reason')), 'selection reason')
            selections[label] = own
        spread = [v['max_degrees'] for v in candidate_errors.values()]
        own_spread = {'min_directed_max_degrees': min(spread) if spread else None,
                      'max_directed_max_degrees': max(spread) if spread else None,
                      'range_directed_max_degrees': max(spread) - min(spread) if spread else None}
        compare(own_spread, er['candidate_error_spread'], 'candidate error spread', ANGLE_ATOL * 2)
        dispersion = []
        for a, ca in enumerate(successful):
            Ra, ta = proper_rotation(ca['estimate']['R']), unit(ca['estimate']['t'])[0]
            for cb in successful[a + 1:]:
                Rb, tb = proper_rotation(cb['estimate']['R']), unit(cb['estimate']['t'])[0]
                dispersion.append({'seed_a': ca['seed'], 'seed_b': cb['seed'],
                                   'rotation_degrees': rotation_angle(Ra, Rb),
                                   'translation_directed_degrees': angle(dot(ta, tb))})
        own_dispersion = {'successful_candidate_count': len(successful), 'pairwise': dispersion,
                          'max_rotation_degrees': max((r['rotation_degrees'] for r in dispersion), default=None),
                          'max_translation_directed_degrees': max((r['translation_directed_degrees'] for r in dispersion), default=None)}
        compare(own_dispersion, er['prediction_only_dispersion'], 'prediction dispersion', ANGLE_ATOL)
        time = timing[b]
        for field, value in (('input_seconds', record['input_seconds']), ('native_seconds', math.fsum(native_times)),
                             ('score_seconds', math.fsum(score_times)), ('charged_total_seconds', charged),
                             ('process_wall_seconds', process['wall_seconds']), ('sampled_peak_rss_kib', process['sampled_peak_rss_kib'])):
            time[field].append(number(value, field))
        for budget in BUDGETS:
            ends = [c['charged_end_seconds'] for c in candidates if c['charged_end_seconds'] <= budget]
            time['prefix_costs'][str(budget)].append(max(ends) if ends else None)
        log = path.with_name(path.name + '.log')
        if log.exists():
            for line in log.read_text(errors='replace').splitlines():
                if any(word in line.lower() for word in ('warning', 'traceback', 'runtimeerror', 'floatingpointerror')):
                    warnings.append({'file': log.name, 'text': line[:500]})
        rows.append({'backend': b, 'pair_index': i, 'key': key, 'selections': selections})
    require(seen == expected and len(ground) == 64, 'incomplete independent review')
    require('oracle_best4_nondeployable' in ev.get('oracle_warning', '') and 'secondary' in ev['oracle_warning'], 'oracle disclosure missing')
    summaries = {}
    for b in BACKENDS:
        subset = sorted((r for r in rows if r['backend'] == b), key=lambda r: r['pair_index'])
        summaries[b] = {}
        for label in subset[0]['selections']:
            own = summarize([r['selections'][label] for r in subset])
            given = ev['summaries'][b][label]
            for field in ('pair_count', 'status_counts', 'selected_seed_counts'):
                require(own[field] == given[field], 'summary coverage/seed counts')
            for convention in ('directed_primary', 'signfolded_secondary'):
                compare(own[convention]['errors_degrees'], given[convention]['errors_degrees'], 'summary errors', ANGLE_ATOL)
                compare(own[convention]['auc_fraction'], given[convention]['auc_fraction'], 'summary AUC', AUC_ATOL)
                require(own[convention]['count_strictly_below'] == given[convention]['count_strictly_below'], 'threshold counts')
            summaries[b][label] = own
    costs = {}
    for b, t in timing.items():
        costs[b] = {k: {'sum': math.fsum(v), 'median': statistics.median(v), 'min': min(v), 'max': max(v)}
                    for k, v in t.items() if k != 'prefix_costs'}
        costs[b]['offline_completed_prefix_seconds'] = {
            budget: {'available_prefix_count': sum(v is not None for v in values),
                     'sum_completed_prefix_seconds': math.fsum(v for v in values if v is not None)}
            for budget, values in t['prefix_costs'].items()}
    scalar_oracle_summaries = {b: summarize([r for _, r in sorted(values)]) for b, values in scalar_oracles.items()}
    current = BACKENDS[1:]
    score_auc = {b: summaries[b]['pixel_msac_full4']['directed_primary']['auc_fraction']['5'] for b in current}
    oracle_auc = {b: scalar_oracle_summaries[b]['directed_primary']['auc_fraction']['5'] for b in current}
    gaps = {o: {s: 100 * (oracle_auc[o] - score_auc[s]) for s in current} for o in current}
    necessary = any(all(v >= 2. for v in gaps[b].values()) for b in current)
    return {'schema': 'evr0016.independent-saved-review.v1', 'status': 'passed',
            'audit_source_sha256': digest(__file__), 'manifest_sha256': mh, 'seal_sha256': sh,
            'evaluation_sha256': eh, 'source_sha256': sources,
            'counts': {'pairs': 64, 'workers': 192, 'candidates': 768,
                       'candidate_status_by_backend': {b: dict(v) for b, v in candidate_status.items()}},
            'max_candidate_angle_difference_degrees': max_angle_difference,
            'angle_comparison_tolerance_degrees': ANGLE_ATOL, 'auc_fraction_tolerance': AUC_ATOL,
            'summaries': summaries, 'costs': costs, 'supervisor_wall_seconds': seal.get('wall_seconds'),
            'scalar_oracle_summaries': scalar_oracle_summaries,
            'oracle_roundoff_seed_discrepancies': oracle_roundoff_ties,
            'oracle_check': 'Exact saved-error argmin verified; any independent scalar identity change is disclosed and has regret within the unchanged angle tolerance. Deployed selector identities must agree exactly.',
            'log_warning_lines': warnings,
            'ordinal_candidate_bank_necessary_gate': {'current_oracle_minus_current_full4_auc5_pp': gaps,
                'at_least_one_current_oracle_exceeds_both_current_selectors_by_2pp': necessary,
                'meaning': 'Necessary search opportunity only; not depth information, novelty, or method efficacy.'},
            'limitations': [
                'No HDF5, images, estimator, or experiment module was opened/imported/run.',
                'Pose errors and AUC were independently recomputed; residual scores were only bound and used, because correspondence arrays are absent.',
                'Saved reference rotations/translations were checked internally; original dataset reference bytes were not reread.',
                'Allocation order/exclusions were checked; the full root-name inventory was not reread.',
                'Prefix cost is the last terminal candidate completed within budget, not the chosen candidate completion time; full runs incurred all four calls.',
                'No completed candidate before a budget does not mean zero work or a realizable interruptible estimator.',
                'One scene and shared photographs do not support independent-scene confidence intervals.',
                'Oracle is nondeployable; this diagnostic is not a method contribution or the project stopping result.',
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'manifest', 'seal', 'evaluation', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'audit output already exists')
    result = audit(args.repo, args.manifest, args.seal, args.evaluation)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
