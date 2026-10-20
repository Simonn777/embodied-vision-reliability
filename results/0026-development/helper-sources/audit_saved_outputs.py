"""Independent saved-only 0026 audit; never imports its scientific pipeline.

No original arrays, HDF5, model, fitting objective or optimizer is read/run.
NumPy is imported only after complete evidence and selector barriers pass.
GT projection uses SVD U@Vt; angles use atan2, independently of saved acos.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ANGLE_ATOL = 1e-6
AUC_PP_ATOL = 1e-5
TIME_ATOL = 1e-9
LABEL_SHA = '66f3f693fa7473aba5371fe473c4f8c9df39a3dd711a32941f28f3e0380eb151'
MANIFEST_SHA = 'a034601021b4baa08a6cc04e91e951ce27dc061d0c0556cf7f1ac141d6a66e34'
TC_SEAL_SHA = 'fb2537ea13b52ac1cc3d2f26fb84053e7fe2da62dedef3b857d3dd8718c9b38e'
SOURCE = Path('experiments/0026-calibration-moments')
SLOTS = ('continued_tc', 'G_neutral', 'G_incumbent', 'G_lift',
         'F_neutral', 'F_incumbent', 'F_lift')
ORDER = ('incumbent',) + SLOTS
UNIONS = {
    'incumbent': ('incumbent',),
    'continued_tc': ('incumbent', 'continued_tc'),
    'G_local': ('incumbent', 'continued_tc', 'G_neutral', 'G_incumbent'),
    'F_local': ('incumbent', 'continued_tc', 'F_neutral', 'F_incumbent'),
    'G_lift': ('incumbent', 'continued_tc', 'G_lift'),
    'F_lift': ('incumbent', 'continued_tc', 'F_lift'),
    'F_all': ('incumbent', 'continued_tc', 'F_neutral', 'F_incumbent', 'F_lift'),
    'G_all': ('incumbent', 'continued_tc', 'G_neutral', 'G_incumbent', 'G_lift'),
}
HISTORICAL = ('historical_e_norm', 'historical_e_ts', 'historical_TC_scale')
CONTROLS = ('incumbent', 'continued_tc', 'G_local', 'F_local', 'F_lift', 'F_all') + HISTORICAL
METHODS = tuple(UNIONS) + HISTORICAL


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def bound(folder, name):
    p = Path(name)
    need(not p.is_absolute() and '..' not in p.parts and p.parts, 'unsafe relative path')
    candidate = Path(folder) / p
    need(candidate.is_file() and not candidate.is_symlink(), 'missing/symlink evidence: ' + str(p))
    need(candidate.resolve().is_relative_to(Path(folder).resolve()), 'evidence escaped folder')
    return candidate


def number(value):
    return type(value) in (float, int) and math.isfinite(value)


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': repr(value)}
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, set):
        return [json_safe(v) for v in sorted(value)]
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value


class Checks:
    def __init__(self):
        self.comparisons = []
        self.failures = []

    def equal(self, name, actual, expected):
        ok = actual == expected
        row = {'name': name, 'passed': ok}
        if not ok:
            row.update(actual=actual, expected=expected)
            self.failures.append(name)
        self.comparisons.append(row)

    def near(self, name, actual, expected, atol):
        valid = number(actual) and number(expected)
        difference = abs(actual - expected) if valid else None
        ok = valid and difference <= atol
        self.comparisons.append({'name': name, 'passed': ok, 'actual': actual,
                                 'expected': expected, 'absolute_difference': difference, 'atol': atol})
        if not ok:
            self.failures.append(name)


def verify_barrier(args, report):
    """Finish all bindings and coverage before any fixed label file access."""
    repo, folder = Path(args.repo), Path(args.predictions)
    need(re.fullmatch('[0-9a-f]{40}', args.expected_commit) is not None, 'full commit required')
    consumed = {}

    def take(path, expected=None):
        actual = digest(path)
        if expected is not None:
            need(actual == expected, 'evidence hash mismatch: ' + Path(path).name)
        consumed[str(Path(path).resolve())] = actual
        return read(path)

    outer = take(bound(folder, 'outer-receipt.json'))
    need(outer.get('status') == 'completed' and outer.get('exit_code') == 0
         and outer.get('error') is None and outer.get('completed_pairs') == 80,
         'prediction outer is not clean and complete')
    need(number(outer['wall_seconds']) and 0 <= outer['wall_seconds'] <= 1800,
         'prediction outer wall limit')
    seal = take(bound(folder, 'seal.json'), outer['seal_sha256'])
    need(seal['schema'] == 'evr0026.seal.v1' and seal['status'] == 'completed', 'prediction seal status')
    need(seal['source_commit'] == outer['source_commit'] == args.expected_commit, 'frozen commit differs')
    job = take(bound(folder, 'job.json'), seal['job_sha256'])
    need(job['commit'] == args.expected_commit and job['source_sha256'] == seal['source_sha256'], 'job/source mismatch')
    source_dir = repo / SOURCE
    inventory = list(source_dir.glob('*.py')) + [source_dir / 'protocol.md']
    inventory += list((source_dir / 'native_bridge').glob('*.py'))
    inventory += list((source_dir / 'native_bridge').glob('*.cpp'))
    inventory += list((source_dir / 'native_bridge').glob('*.cc'))
    inventory += [source_dir / 'native_bridge/input-lock.json']
    need({str(p.relative_to(source_dir)) for p in inventory} == set(seal['source_sha256']), 'source inventory mismatch')
    for name, expected in seal['source_sha256'].items():
        path = bound(source_dir, name)
        need(digest(path) == expected, 'current source hash mismatch: ' + name)
        rel = str(path.relative_to(repo))
        committed = subprocess.check_output(['git', 'show', args.expected_commit + ':' + rel], cwd=repo)
        need(hashlib.sha256(committed).hexdigest() == expected, 'source is not frozen at expected commit')
        consumed[str(path.resolve())] = expected
    prior = job['prior']
    need(prior['manifest'] == MANIFEST_SHA and prior['TC_seal'] == TC_SEAL_SHA, 'fixed prior binding mismatch')
    manifest = take(repo / 'configs/0017-development-manifest.json', MANIFEST_SHA)
    need(job['selected_pairs'] == manifest['selected_pairs'], 'selected cohort differs from fixed manifest')
    need([r['pair_index'] for r in job['selected_pairs']] == list(range(80)), 'job cohort incomplete')
    for name, expected in prior['legacy_sources'].items():
        path = bound(repo, name)
        need(digest(path) == expected, 'legacy source mismatch')
        consumed[str(path.resolve())] = expected
    need('experiments/0013-pose-headroom/evaluate_sealed.py' in prior['legacy_sources'], 'metric source binding missing')
    need([r['pair_index'] for r in seal['rows']] == list(range(80)), 'sealed cohort incomplete/reordered')
    records, processes = [], []
    for row in seal['rows']:
        i = row['pair_index']
        files = {}
        for key, hk in (('path', 'sha256'), ('process', 'process_sha256'), ('partial', 'partial_sha256'), ('log', 'log_sha256')):
            need(Path(row[key]).name == row[key], 'nonflat child evidence path')
            p = bound(folder, row[key])
            need(digest(p) == row[hk], 'child evidence hash mismatch')
            consumed[str(p.resolve())] = row[hk]
            files[key] = p
        rec, proc = read(files['path']), read(files['process'])
        need(proc['exit_code'] == 0 and proc['termination'] is None, 'unclean pair process')
        need(number(proc['wall_seconds']) and proc['wall_seconds'] >= 0, 'invalid process wall time')
        need(number(proc['sampled_peak_rss_kib']) and 0 <= proc['sampled_peak_rss_kib'] <= 8*1024*1024,
             'pair sampled RSS bound')
        need(rec['status'] == 'completed' and rec['pair_index'] == i and rec['identity'] == job['selected_pairs'][i], 'pair identity/status')
        need(rec['source_sha256'] == seal['source_sha256'] and rec['native_binding'] == job['native'], 'pair source/native binding')
        need(rec['prior_baseline_sha256'] == prior['baselines'][str(i)], 'incumbent SHA mismatch')
        records.append(rec)
        processes.append(proc)
    references = {}
    for experiment, expected, arms in (
            ('0017-depth', prior['original_seal'], ('e_norm', 'e_ts')),
            ('0019-author-loss', prior['TC_seal'], ('depth_scale', 'depth_affine'))):
        sf = repo / 'results' / experiment / 'predictions'
        hs = take(sf / 'seal.json', expected)
        need(hs['status'] == 'completed', 'historical seal incomplete')
        for arm in arms:
            selected = sorted((r for r in hs['rows'] if r['arm'] == arm), key=lambda r: r['pair_index'])
            need([r['pair_index'] for r in selected] == list(range(80)), 'historical reference coverage')
            for row in selected:
                i = row['pair_index']
                old = take(bound(sf, row['path']), row['sha256'])
                need(old['pair_index'] == i and old['arm'] == arm, 'historical record identity')
                need(all(old[k] == job['selected_pairs'][i][k] for k in ('scene', 'key')), 'historical scene/key')
                references[i, arm] = old
                if arm == 'depth_affine':
                    need(row['sha256'] == prior['baselines'][str(i)], 'fixed incumbent file hash')
                    for key in ('estimate', 'geometry'):
                        need(records[i]['incumbent'][key] == old[key], 'incumbent changed')
                    for key in ('input_arrays', 'calibration', 'input_diagnostics'):
                        need(records[i]['input_bindings'][key] == old[key], 'input binding changed')
                    need(records[i]['support']['mask_sha256'] == old['mask_sha256'], 'fixed support hash differs')
    if args.evaluation_receipt:
        rp = Path(args.evaluation_receipt)
        er = take(rp)
        need(er.get('exit_code') == 0 and er.get('status', 'completed') == 'completed', 'evaluation operation failed')
        need(number(er.get('wall_seconds')) and er['wall_seconds'] >= 0, 'evaluation operation timing invalid')
        if 'output_sha256' in er:
            need(digest(args.evaluation) == er['output_sha256'], 'evaluation output receipt mismatch')
        for name, expected in er.get('log_sha256', {}).items():
            p = bound(rp.parent, name)
            need(digest(p) == expected, 'evaluation log changed')
            consumed[str(p.resolve())] = expected
    evaluation = take(Path(args.evaluation))
    need(evaluation['schema'] == 'evr0026.evaluation.v1' and evaluation['status'] == 'completed', 'evaluation incomplete')
    need(evaluation['bindings'] == {'seal_sha256': digest(folder/'seal.json'),
         'outer_sha256': digest(folder/'outer-receipt.json'), 'label_sha256': LABEL_SHA}, 'evaluation evidence binding')
    need([r['pair_index'] for r in evaluation['rows']] == list(range(80)), 'evaluation cohort incomplete')
    report['barrier'] = {'prediction_pairs': 80, 'historical_records_checked': len(references),
                         'source_commit': args.expected_commit, 'source_file_count': len(inventory),
                         'prediction_wall_seconds': outer['wall_seconds'],
                         'evaluation_operation_receipt_supplied': bool(args.evaluation_receipt)}
    report['bindings'] = {'prediction_seal_sha256': digest(folder/'seal.json'),
                          'prediction_outer_sha256': digest(folder/'outer-receipt.json'),
                          'evaluation_sha256': digest(args.evaluation),
                          'evaluation_receipt_sha256': digest(args.evaluation_receipt) if args.evaluation_receipt else None,
                          'source_sha256': seal['source_sha256'], 'manifest_sha256': MANIFEST_SHA,
                          'TC_seal_sha256': TC_SEAL_SHA, 'original_seal_sha256': prior['original_seal'],
                          'label_sha256_expected': LABEL_SHA}
    return job, records, processes, references, evaluation, consumed


def selections_and_times(records, references, checks):
    winners, times = [], []
    for i, rec in enumerate(records):
        tag = 'pair/%d/' % i
        checks.equal(tag+'attempt_inventory', set(rec['attempts']), set(SLOTS))
        checks.equal(tag+'selector_inventory', set(rec['selectors']), set(UNIONS))
        checks.equal(tag+'tie_order', rec['tie_order'], list(ORDER))
        checks.equal(tag+'execution_order', rec['execution_order'], list(SLOTS[i % 7:] + SLOTS[:i % 7]))
        bank = {'incumbent': rec['incumbent'], **rec['attempts']}
        oldscore = references[i, 'depth_affine']['final_scores']
        checks.equal(tag+'incumbent_score_fields', set(bank['incumbent']['final_scores']), set(oldscore))
        for key, value in oldscore.items():
            checks.near(tag+'incumbent_score/'+key, bank['incumbent']['final_scores'][key], value, 1e-10)
        for key, candidate in bank.items():
            need(candidate['status'] in ('completed', 'failed', 'unavailable'), 'unexpected proposal status')
            if candidate['status'] == 'completed':
                score = candidate['final_scores']['pixel_msac']
                need(number(score) and 0 <= score <= rec['support']['length'], 'invalid common selector cost')
        native_calls = 0
        for slot, candidate in rec['attempts'].items():
            tt = candidate['timings']
            pieces = [tt[k] for k in ('seed_seconds', 'optimizer_seconds', 'recovery_seconds',
                      'native_seconds', 'score_seconds', 'depth_diagnostics_seconds')]
            need(all(number(v) and v >= 0 for v in pieces+[tt['total_seconds']]), 'invalid slot component time')
            need(math.fsum(pieces) <= tt['total_seconds']+TIME_ATOL, 'slot components exceed enclosing time')
            native = candidate.get('native')
            raw_failure = candidate.get('failure', {}).get('native_raw') or {}
            native_calls += native is not None or raw_failure.get('native_called') == 1
        checks.equal(tag+'native_attempt_count', rec['timings']['native_attempt_count'], native_calls)
        selections, charged = {}, {}
        for method, members in UNIONS.items():
            eligible = [key for key in ORDER if key in members and bank[key]['status'] == 'completed']
            need(eligible and 'incumbent' in eligible, 'missing valid incumbent fallback')
            winner = min(eligible, key=lambda key: (bank[key]['final_scores']['pixel_msac'], ORDER.index(key)))
            saved = rec['selectors'][method]
            prefix = tag+method+'/'
            checks.equal(prefix+'members', saved['members'], list(members))
            checks.equal(prefix+'available_members', saved['available_members'], [k for k in members if k in eligible])
            checks.equal(prefix+'selected', saved['selected_attempt'], winner)
            checks.equal(prefix+'estimate', saved['estimate'], bank[winner]['estimate'])
            checks.equal(prefix+'geometry', saved['geometry'], bank[winner]['geometry'])
            checks.equal(prefix+'score', saved['score'], bank[winner]['final_scores']['pixel_msac'])
            prereq = ['common']
            if any(k.startswith(('G_', 'F_')) for k in members): prereq.append('moments')
            if any(k.endswith('_lift') for k in members): prereq.append('relaxation')
            checks.equal(prefix+'prerequisites', saved['prerequisites'], prereq)
            values = [rec['preparations'][k]['seconds'] for k in prereq]
            slot_values = [rec['attempts'][k]['timings']['total_seconds'] for k in members if k != 'incumbent']
            need(all(number(v) and v >= 0 for v in values+slot_values), 'invalid charged timing component')
            prep, attempt = math.fsum(values), math.fsum(slot_values)
            for key, value in (('preparation_seconds', prep), ('attempt_seconds', attempt), ('charged_seconds', prep+attempt)):
                checks.near(prefix+key, saved[key], value, TIME_ATOL)
            selections[method], charged[method] = winner, prep+attempt
        winners.append(selections)
        times.append(charged)
    return winners, times


def pose_error(estimate, gt):
    import numpy as np
    R = np.asarray(estimate['R'], dtype=float)
    G = np.asarray(gt['R_original'], dtype=float)
    need(R.shape == G.shape == (3, 3) and np.isfinite(R).all() and np.isfinite(G).all(), 'invalid rotation array')
    need(np.max(np.abs(R@R.T-np.eye(3))) <= 1e-6 and abs(np.linalg.det(R)-1) <= 1e-6, 'improper prediction')
    U, singular, Vt = np.linalg.svd(G)
    G = U @ Vt
    need(np.isfinite(singular).all() and singular[-1] > 0 and np.linalg.det(G) > 0, 'invalid GT projection')
    D = R @ G.T
    skew = (D[2, 1]-D[1, 2], D[0, 2]-D[2, 0], D[1, 0]-D[0, 1])
    rotation = math.degrees(math.atan2(.5*math.hypot(*skew), .5*(float(np.trace(D))-1)))
    def unit(value):
        a = np.asarray(value, dtype=float).reshape(-1)
        need(a.shape == (3,) and np.isfinite(a).all(), 'invalid translation array')
        scale = max(abs(float(v)) for v in a)
        need(scale > 0, 'zero translation')
        a = a / scale
        return a / math.hypot(*a.tolist())
    p, q = unit(estimate['t']), unit(gt['t_original'])
    cross = np.cross(p, q)
    translation = math.degrees(math.atan2(math.hypot(*cross.tolist()), float(np.dot(p, q))))
    return {'rotation_degrees': rotation, 'translation_degrees': translation,
            'max_degrees': max(rotation, translation)}


def auc(values, threshold):
    need(len(values) > 0 and all(number(v) and 0 <= v <= 180 for v in values), 'invalid full-denominator errors')
    return 100*math.fsum((threshold-v)/threshold if v < threshold else 0. for v in values)/len(values)


def audit(args, report, checks):
    job, records, processes, refs, evaluation, consumed = verify_barrier(args, report)
    winners, times = selections_and_times(records, refs, checks)
    need(not checks.failures, 'pre-label selector/timing evidence failed')
    labelpath = Path(args.repo)/'results/0017-depth/evaluation.json'
    need(digest(labelpath) == LABEL_SHA, 'fixed saved label digest differs')
    labels = read(labelpath)['rows']
    report['labels_accessed_after_complete_barrier'] = True
    consumed[str(labelpath.resolve())] = LABEL_SHA
    need([r['pair_index'] for r in labels] == list(range(80)), 'label coverage differs')
    rows, statuses = [], {k: Counter() for k in SLOTS}
    reference_arms = dict(zip(HISTORICAL, ('e_norm', 'e_ts', 'depth_scale')))
    for i, rec in enumerate(records):
        identity, saved, label = job['selected_pairs'][i], evaluation['rows'][i], labels[i]
        need(all(label[k] == identity[k] for k in ('scene', 'key')), 'label identity differs')
        checks.equal('pair/%d/scene' % i, saved['scene'], identity['scene'])
        bank = {'incumbent': rec['incumbent'], **rec['attempts']}
        raw = {}
        for name, candidate in bank.items():
            errors = pose_error(candidate['estimate'], label['ground_truth']) if candidate['status'] == 'completed' else {'max_degrees': 180.}
            raw[name] = {'status': candidate['status'], **errors}
            old = saved['raw_proposals'][name]
            checks.equal('pair/%d/raw/%s/status' % (i, name), old['status'], candidate['status'])
            for key, value in errors.items():
                checks.near('pair/%d/raw/%s/%s' % (i, name, key), old[key], value, ANGLE_ATOL)
            if name in statuses: statuses[name][candidate['status']] += 1
        outcomes = {name: raw[winners[i][name]] for name in UNIONS}
        for method, arm in reference_arms.items():
            candidate = refs[i, arm]
            errors = pose_error(candidate['estimate'], label['ground_truth']) if candidate['status'] == 'completed' else {'max_degrees': 180.}
            outcomes[method] = {'status': candidate['status'], **errors}
        checks.equal('pair/%d/selected_attempts' % i, saved['selected_attempts'], winners[i])
        checks.equal('pair/%d/worker_timings' % i, saved['worker_timings'], rec['worker_timings'])
        wt = rec['worker_timings']
        need(all(number(wt[k]) and wt[k] >= 0 for k in ('binding_seconds', 'input_seconds', 'total_seconds')),
             'invalid worker times')
        need(wt['binding_seconds']+wt['input_seconds'] <= wt['total_seconds']+TIME_ATOL
             and rec['timings']['core_processing_seconds'] <= wt['total_seconds']+TIME_ATOL
             and wt['total_seconds'] <= processes[i]['wall_seconds']+TIME_ATOL,
             'worker timings exceed enclosing wall time')
        for method in METHODS:
            checks.equal('pair/%d/%s/status' % (i, method), saved['outcomes'][method]['status'], outcomes[method]['status'])
            for key, value in outcomes[method].items():
                if key != 'status': checks.near('pair/%d/%s/%s' % (i, method, key), saved['outcomes'][method][key], value, ANGLE_ATOL)
        for method in UNIONS:
            checks.near('pair/%d/%s/charged' % (i, method), saved['core_seconds'][method], times[i][method], TIME_ATOL)
        rows.append({'pair_index': i, 'scene': identity['scene'], 'outcomes': outcomes,
                     'raw_proposals': raw, 'selected_attempts': winners[i], 'charged_core_seconds': times[i]})
    def summary(subset):
        return {m: {'count': len(subset), 'auc_percent': {str(t): auc([r['outcomes'][m]['max_degrees'] for r in subset], t)
                 for t in (5, 10, 20)}} for m in METHODS}
    pooled = summary(rows)
    scenes = list(dict.fromkeys(r['scene'] for r in rows))
    need(len(scenes) == 5 and all(sum(r['scene'] == s for r in rows) == 16 for s in scenes), 'scene coverage differs')
    per_scene = {s: summary([r for r in rows if r['scene'] == s]) for s in scenes}
    for group, actual, expected in [('pooled', evaluation['pooled'], pooled)] + [
            ('scene/'+s, evaluation['scenes'][s], per_scene[s]) for s in scenes]:
        checks.equal(group+'/methods', set(actual), set(METHODS))
        for m in METHODS:
            checks.equal(group+'/'+m+'/count', actual[m]['count'], expected[m]['count'])
            for t in ('5', '10', '20'):
                checks.near(group+'/'+m+'/'+t, actual[m]['auc_percent'][t], expected[m]['auc_percent'][t], AUC_PP_ATOL)
    best = max(CONTROLS, key=lambda m: (pooled[m]['auc_percent']['5'], -CONTROLS.index(m)))
    gain = pooled['G_lift']['auc_percent']['5']-pooled[best]['auc_percent']['5']
    margins = {s: per_scene[s]['G_lift']['auc_percent']['5']-per_scene[s][best]['auc_percent']['5'] for s in scenes}
    loss20 = pooled[best]['auc_percent']['20']-pooled['G_lift']['auc_percent']['20']
    clauses = {'auc5_gain_at_least2pp': gain >= 2., 'positive_in_at_least3_scenes': sum(v > 0 for v in margins.values()) >= 3,
               'no_scene_loses_more_than1pp': min(margins.values()) >= -1., 'auc20_loss_at_most0_5pp': loss20 <= .5}
    decision = {'primary': 'G_lift', 'best_control': best, 'auc5_gain_pp': gain,
                'scene_gain_pp': margins, 'auc20_loss_pp': loss20, 'clauses': clauses,
                'development_gate_passed': all(clauses.values()), 'main_goal_complete': False, 'manuscript_admitted': False}
    for key in ('primary', 'best_control', 'clauses', 'development_gate_passed', 'main_goal_complete', 'manuscript_admitted'):
        checks.equal('decision/'+key, evaluation['decision'][key], decision[key])
    for key in ('auc5_gain_pp', 'auc20_loss_pp'):
        checks.near('decision/'+key, evaluation['decision'][key], decision[key], AUC_PP_ATOL)
    for s, value in margins.items():
        checks.near('decision/scene/'+s, evaluation['decision']['scene_gain_pp'][s], value, AUC_PP_ATOL)
    for slot in SLOTS:
        checks.equal('statuses/'+slot, evaluation['proposal_status_counts'][slot], dict(statuses[slot]))
    for method in UNIONS:
        counts = dict(Counter(w[method] for w in winners))
        checks.equal('selection_counts/'+method, evaluation['selection_counts'][method], counts)
        for key, value in (('sum_core_seconds', math.fsum(v[method] for v in times)),
                           ('median_core_seconds', statistics.median(v[method] for v in times))):
            checks.near('timing/'+method+'/'+key, evaluation['timing_summary'][method][key], value, TIME_ATOL)
    for path, expected in consumed.items():
        need(digest(path) == expected, 'evidence changed during audit: ' + Path(path).name)
    report.update(rows=rows, pooled=pooled, scenes=per_scene, decision=decision,
                  proposal_status_counts={k: dict(v) for k, v in statuses.items()},
                  files_rehashed=len(consumed), record_count=80, raw_proposal_count=640,
                  new_slot_count=560, method_count=len(METHODS), source_binding_passed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'predictions', 'evaluation', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--evaluation-receipt', type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    checks = Checks()
    report = {'schema': 'evr0026.independent-saved-audit.v1', 'status': 'failed', 'passed': False,
              'audit_source_sha256': digest(__file__), 'labels_accessed_after_complete_barrier': False,
              'tolerances': {'angle_degrees': ANGLE_ATOL, 'auc_percentage_points': AUC_PP_ATOL, 'timing_seconds': TIME_ATOL},
              'scope': 'Independent saved bindings, selectors, component sums and pose metrics; no scientific-array or objective-fit rerun.',
              'independent_team_replication': False, 'original_pipeline_rerun': False}
    error = None
    try:
        audit(args, report, checks)
        report.update(status='passed' if not checks.failures else 'failed', passed=not checks.failures)
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
        report['error'] = error
    report.update(comparisons=checks.comparisons, failures=checks.failures,
                  comparison_count=len(checks.comparisons), elapsed_seconds=time.perf_counter()-started)
    finite_diffs = [r['absolute_difference'] for r in checks.comparisons if r.get('absolute_difference') is not None]
    report['max_absolute_comparison_difference_mixed_units'] = max(finite_diffs, default=0.)
    report['max_absolute_differences'] = {name: max((r['absolute_difference'] for r in checks.comparisons
        if r.get('atol') == tolerance and r.get('absolute_difference') is not None), default=0.)
        for name, tolerance in (('angle_degrees', ANGLE_ATOL), ('auc_percentage_points', AUC_PP_ATOL), ('timing_seconds', TIME_ATOL))}
    with (args.output_dir/'audit.json').open('x') as stream:
        json.dump(json_safe(report), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'passed': report['passed'], 'comparison_count': len(checks.comparisons),
                      'failures': checks.failures, 'error': error}, allow_nan=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
