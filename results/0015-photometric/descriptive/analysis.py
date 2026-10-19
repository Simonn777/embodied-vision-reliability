"""Descriptive replay of sealed JSON only; no image/HDF5/estimator/scorer imports."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent


def load(path):
    raw = path.read_bytes()
    def nonfinite(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, parse_constant=nonfinite, object_pairs_hook=unique), hashlib.sha256(raw).hexdigest()


def counts(values):
    return dict(sorted(Counter(values).items()))


def describe(values):
    values = sorted(values)
    if any(not math.isfinite(x) for x in values):
        raise ValueError('nonfinite descriptive input')
    if not values:
        return {'count': 0}
    def quantile(p):
        position = (len(values)-1)*p
        low, high = math.floor(position), math.ceil(position)
        return values[low] + (position-low)*(values[high]-values[low])
    return {'count': len(values), 'sum': math.fsum(values),
        'mean': statistics.fmean(values), 'min': values[0], 'p10': quantile(.1),
        'p25': quantile(.25), 'median': statistics.median(values), 'p75': quantile(.75),
        'p90': quantile(.9), 'max': values[-1]}


def summarize_queries(queries):
    patches = [q['photometric'] for q in queries if q['photometric'] is not None]
    failed = [q['photometric'] for q in queries if q['status'] == 'fit_failed']
    admitted = [q for q in queries if q['admitted']]
    reasons = []
    combinations = []
    for patch in failed:
        labels = patch['reason'] if isinstance(patch['reason'], list) else [patch['reason']]
        reasons.extend(labels)
        combinations.append(' + '.join(sorted(labels)))
    trial_statuses = {a['status'] for p in patches for a in p['attempts']}
    initial = [q['photometric']['initial_cost'] for q in admitted]
    final = [q['photometric']['final_cost'] for q in admitted]
    relative = [(a-b)/a for a, b in zip(initial, final) if a > 0]
    if any(b > a for a, b in zip(initial, final)):
        raise ValueError('an admitted fit increased saved loss')
    return {'selected': len(queries), 'query_status_counts': counts(q['status'] for q in queries),
        'initializer_failure_reason_counts': counts(q['reason'] for q in queries
                                                   if q['status'] == 'initialization_failed'),
        'kernel_status_counts': counts(p['status'] for p in patches),
        'rejected_fit_reason_counts_overlapping': counts(reasons),
        'rejected_fit_reason_combinations_exclusive': counts(combinations),
        'termination_counts_all_kernel_queries': counts(str(p['termination']) for p in patches),
        'trial_status_event_counts_all_fits': counts(a['status'] for p in patches for a in p['attempts']),
        'rejected_fit_trial_status_event_counts': counts(a['status'] for p in failed for a in p['attempts']),
        'rejected_fits_with_trial_status_overlapping': {
            status: sum(any(a['status'] == status for a in p['attempts']) for p in failed)
            for status in sorted(trial_statuses)},
        'admitted': {'count': len(admitted),
            'target_coordinate_changed': sum(q['target_coordinate_changed'] for q in admitted),
            'initial_patch_loss': describe(initial), 'final_patch_loss': describe(final),
            'loss_reduction': describe([a-b for a, b in zip(initial, final)]),
            'relative_loss_reduction': describe(relative),
            'zero_initial_loss_count': sum(a == 0 for a in initial),
            'pooled_loss_reduction_fraction': ((math.fsum(initial)-math.fsum(final))/math.fsum(initial)
                                               if math.fsum(initial) > 0 else None),
            'applied_displacement_l2_pixels': describe([math.hypot(*q['delta_cache']) for q in admitted]),
            'accepted_steps': describe([q['photometric']['accepted_steps'] for q in admitted]),
            'attempt_count': describe([len(q['photometric']['attempts']) for q in admitted])}}


def analyze():
    seal, seal_sha = load(RESULTS/'photometric/seal.json')
    evaluation, evaluation_sha = load(RESULTS/'evaluation-repair.json')
    if (seal.get('complete') is not True or seal.get('eligible_for_next_phase') is not True
            or len(seal.get('rows', [])) != 16 or len(evaluation.get('rows', [])) != 16
            or evaluation.get('photometric_seal_sha256') != seal_sha):
        raise ValueError('complete saved cohort/evaluation binding differs')
    all_queries, pairs, admitted_rows, inputs = [], [], [], {}
    for i, (entry, score) in enumerate(zip(seal['rows'], evaluation['rows'])):
        name = entry['path']
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('unsafe saved photo filename')
        path = RESULTS/'photometric'/name
        if path.resolve() != path.absolute():
            raise ValueError('saved photo symlink refused')
        photo, digest = load(path)
        if (digest != entry['sha256'] or photo.get('status') != 'completed'
                or photo['pair_index'] != i or score['pair_index'] != i
                or entry['pair_index'] != i or photo['key'] != score['key']
                or photo['key'] != entry['key']):
            raise ValueError('saved photo hash/order/key/status differs')
        inputs['photometric/'+name] = digest
        queries = photo['queries']
        for key, count in [('selected_count', len(queries)),
                           ('initialized_count', sum(q['photometric'] is not None for q in queries)),
                           ('admitted_count', sum(q['admitted'] for q in queries)),
                           ('target_coordinate_changed_count', sum(q['target_coordinate_changed'] for q in queries))]:
            if photo[key] != count or score[key] != count:
                raise ValueError('saved count differs: '+key)
        if len(queries) > 256 or len(set(q['index'] for q in queries)) != len(queries):
            raise ValueError('saved query allocation is not unique and bounded')
        for query in queries:
            if not query['admitted']:
                if query['delta_cache'] != [0., 0.] or query['target_cache'] != query['original_target_cache']:
                    raise ValueError('failed query was not preserved')
            else:
                patch = query['photometric']
                admitted_rows.append({'pair_index': i, 'correspondence_index': query['index'],
                    'initial_patch_loss': patch['initial_cost'], 'final_patch_loss': patch['final_cost'],
                    'applied_displacement_l2_pixels': math.hypot(*query['delta_cache']),
                    'accepted_steps': patch['accepted_steps'], 'attempt_count': len(patch['attempts'])})
        all_queries.extend(queries)
        errors = score['errors']
        available = not errors['subset_original']['unavailable']
        if (available != (photo['admitted_count'] >= 8)
                or available == errors['subset_refined']['unavailable']):
            raise ValueError('saved subset availability differs')
        pair = {'pair_index': i, 'key': photo['key'], **summarize_queries(queries),
            'subset_available': available, 'saved_errors': errors,
            'subset_error_change_degrees': (errors['subset_refined']['max_degrees']-
                errors['subset_original']['max_degrees']) if available else None,
            'full_error_change_degrees': errors['full_refined']['max_degrees']-errors['full_original']['max_degrees'],
            'photo_timing_seconds': photo['timing_seconds']}
        if not available and any(errors[m]['max_degrees'] != 180. for m in ('subset_original', 'subset_refined')):
            raise ValueError('unavailable subset penalty changed')
        pairs.append(pair)
    available_changes = [p['subset_error_change_degrees'] for p in pairs if p['subset_available']]
    return {'schema': 'evr0015.saved-descriptive.v1',
        'analysis_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inputs_sha256': {'photometric/seal.json': seal_sha, 'evaluation-repair.json': evaluation_sha, **inputs},
        'scope': 'Post-outcome descriptive arithmetic of saved JSON only; no fitting/scoring rerun or revised gates',
        'uncertainty': '16 dependent pairs from one scene; query counts are descriptive, not independent evidence',
        'quantiles': 'linear interpolation at p*(n-1); median is standard sample median',
        'loss': 'saved 0.5*sum(r^2) over961 intensities; all eight fit parameters can reduce it',
        'changes': 'refined minus original max-pose-error degrees; negative is an improvement',
        'reason_counting': 'Reason labels and trial-presence counts overlap; reason combinations are exclusive. Trial rejection events are not final failure causes.',
        'aggregate': summarize_queries(all_queries), 'pairs': pairs,
        'available_subset_descriptive': {'count': len(available_changes),
            'improved_gt_1e_9_degrees': sum(x < -1e-9 for x in available_changes),
            'worsened_gt_1e_9_degrees': sum(x > 1e-9 for x in available_changes),
            'unchanged_within_1e_9_degrees': sum(abs(x) <= 1e-9 for x in available_changes),
            'error_changes_degrees': describe(available_changes)},
        'unavailable_pair_indices': [p['pair_index'] for p in pairs if not p['subset_available']],
        'admitted_queries': admitted_rows,
        'saved_auc_unmodified': evaluation['auc'], 'saved_expansion_gate_unmodified': evaluation['expansion_gate']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'summary.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('descriptive output already exists')
    result = analyze()
    with args.output.open('x') as stream:
        json.dump(result, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'selected': result['aggregate']['selected'],
        'statuses': result['aggregate']['query_status_counts'],
        'subset': result['available_subset_descriptive'],
        'admitted': result['aggregate']['admitted'],
        'rejected_fits_with_trial_status': result['aggregate']['rejected_fits_with_trial_status_overlapping']}))


if __name__ == '__main__':
    main()
