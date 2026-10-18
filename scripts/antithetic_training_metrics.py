"""Pure development scoring for 0008; no file I/O, trainer import or fitting.

Reuse the frozen 0006 pixel/block scorer. The wrapper must separately verify
saved arrays, receipts, timing attribution and inference identity. Scalar rows
alone cannot prove that a model was run or a cost was attributed correctly.
"""
import copy
import math
import numbers

from conditional_residual_metrics import score_pair, _validate_score

GROUPS = ('d1', 'd2', 'd3', 'd4')
SEEDS = (0, 1)
METHODS = ('independent', 'random_antithetic', 'mean_antithetic', 'same_patch')
CONTROLS = ('independent', 'random_antithetic', 'same_patch')
CANDIDATE = 'mean_antithetic'
ENDPOINTS = ('updates500', 'wall5s')


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(name + ' must be a finite positive number')
    if not math.isfinite(value) or value <= 0:
        raise ValueError(name + ' must be a finite positive number')
    return float(value)


def _total(values, name):
    try:
        value = math.fsum(values)
    except (OverflowError, ValueError) as exc:
        raise ValueError(name + ' accumulation is nonfinite') from exc
    if not math.isfinite(value):
        raise ValueError(name + ' accumulation is nonfinite')
    return value


def _score(score, shape):
    if not isinstance(score, dict):
        raise ValueError('score must be a dictionary')
    try:
        fd = score['first_difference']
        if not isinstance(fd, dict) or set(fd) != {'horizontal', 'vertical', 'combined'}:
            raise ValueError('three first-difference summaries required')
        if any(type(n) is not int for n in score['shape']):
            raise ValueError('score shape must contain integer dimensions')
        # bool is numerically equal to 0/1 in Python, but is not a metric value.
        for part in [score, *fd.values()]:
            if type(part['count']) is not int:
                raise ValueError('score counts must be integers')
            for field in ('sse', 'mse'):
                if isinstance(part[field], bool) or not isinstance(part[field], numbers.Real):
                    raise ValueError('score errors must be numeric, not booleans')
        if isinstance(score['data_range'], bool):
            raise ValueError('data_range must be numeric, not boolean')
        if score['psnr_db'] is not None and (isinstance(score['psnr_db'], bool) or
                not isinstance(score['psnr_db'], numbers.Real) or not math.isfinite(score['psnr_db'])):
            raise ValueError('finite PSNR or explicit null required')
        for block in score['blocks']:
            if any(type(block[k]) is not int for k in ('row', 'col', 'row_start', 'row_stop', 'col_start', 'col_stop', 'count')):
                raise ValueError('block geometry/count must be integer')
            if isinstance(block['sse'], bool) or not isinstance(block['sse'], numbers.Real):
                raise ValueError('block SSE must be numeric, not boolean')
        _validate_score(score, shape)
    except (KeyError, TypeError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError('malformed score summary') from exc


def aggregate_cases(rows, expected_shape=(122, 122)):
    """Validate the complete fixed 4x2x4x2 grid and score two endpoints.

    Required row fields: group, seed, method, endpoint, score, status='completed',
    inference_mc_samples=50, updates (positive int), training_seconds and
    pipeline_seconds. This is the allocated observed outer cost, retaining shared
    startup/check/shutdown overhead and subtracting only measured other-endpoint
    extra training and disk/inference work. Full actual cost stays in the receipt.

    updates500 requires exactly500 updates. wall5s requires 5<=training<=5.05.
    Missing/duplicate/invalid/nonfinite records raise ValueError. Any zero MSE in
    a candidate/best-control comparison makes its finite dB gain not evaluable;
    no infinity or invented gain is substituted, including when both are perfect.
    Zero first-difference error is compared directly without dividing by zero.
    """
    try:
        shape = tuple(expected_shape)
    except TypeError as exc:
        raise ValueError('expected_shape must contain two integer dimensions') from exc
    if len(shape) != 2 or any(type(n) is not int or n < 8 for n in shape):
        raise ValueError('expected_shape must contain two integer dimensions >=8')
    expected = {(g,s,m,e) for g in GROUPS for s in SEEDS for m in METHODS for e in ENDPOINTS}
    indexed, clean_hashes = {}, {}
    try:
        for row in rows:
            if not isinstance(row, dict) or type(row['seed']) is not int:
                raise ValueError('row dictionary and integer seed required')
            key = (row['group'],row['seed'],row['method'],row['endpoint'])
            if key not in expected or key in indexed:
                raise ValueError('unexpected or duplicate case/endpoint')
            if row['status'] != 'completed' or type(row['inference_mc_samples']) is not int or row['inference_mc_samples'] != 50:
                raise ValueError('completed MC50 endpoint required')
            updates = row['updates']
            if type(updates) is not int or updates <= 0 or (key[3]=='updates500' and updates != 500):
                raise ValueError('invalid endpoint optimizer-update count')
            training = _positive(row['training_seconds'], 'training_seconds')
            pipeline = _positive(row['pipeline_seconds'], 'pipeline_seconds')
            if pipeline < training:
                raise ValueError('pipeline_seconds cannot be less than endpoint training_seconds')
            if key[3]=='wall5s' and not 5.0 <= training <= 5.05:
                raise ValueError('wall5s snapshot outside frozen 5.00..5.05 seconds')
            _score(row['score'],shape)
            digest = row['score']['clean_sha256_f64le']
            if key[0] in clean_hashes and clean_hashes[key[0]] != digest:
                raise ValueError('clean substrate differs within one image group')
            clean_hashes[key[0]] = digest
            indexed[key] = {k:copy.deepcopy(row[k]) for k in ('group','seed','method','endpoint','score','status','inference_mc_samples','updates')}
            indexed[key].update(training_seconds=training,pipeline_seconds=pipeline)
    except (KeyError, TypeError) as exc:
        raise ValueError('malformed endpoint row') from exc
    if set(indexed) != expected:
        raise ValueError('complete64-endpoint grid required')

    endpoints = {}
    for endpoint in ENDPOINTS:
        pairs = []
        for group in GROUPS:
            for seed in SEEDS:
                cand = indexed[group,seed,CANDIDATE,endpoint]['score']
                chosen = min(CONTROLS,key=lambda m:indexed[group,seed,m,endpoint]['score']['mse'])
                base = indexed[group,seed,chosen,endpoint]['score']
                gain = None if cand['mse']==0 or base['mse']==0 else cand['psnr_db']-base['psnr_db']
                if gain is not None and not math.isfinite(gain):
                    raise ValueError('nonfinite finite-dB subtraction')
                pairs.append(dict(group=group,seed=seed,oracle_best_control=chosen,
                    candidate_psnr_db=cand['psnr_db'],control_psnr_db=base['psnr_db'],gain_db=gain,
                    candidate_first_difference_mse=cand['first_difference']['combined']['mse'],
                    control_first_difference_mse=base['first_difference']['combined']['mse']))
        finite = all(p['gain_db'] is not None for p in pairs)
        group_means = {}
        for g in GROUPS:
            values = [p['gain_db'] for p in pairs if p['group']==g]
            group_means[g] = _total(values,'group gains')/2 if all(v is not None for v in values) else None
        mean_gain = _total([p['gain_db'] for p in pairs],'endpoint gains')/8 if finite else None
        fd_c = _total([p['candidate_first_difference_mse'] for p in pairs],'candidate FD')/8
        fd_b = _total([p['control_first_difference_mse'] for p in pairs],'control FD')/8
        # The validated finite eight-case sum bounds each mean before scaling.
        fd_gate = fd_c <= 1.05*fd_b
        gates = dict(finite_paired_db_gains=finite,
            mean_gain_ge_0_5_db=mean_gain>=0.5 if finite else None,
            at_least3_group_means_gt0=sum(v>0 for v in group_means.values())>=3 if finite else None,
            each_group_mean_ge_minus0_2_db=all(v>=-0.2 for v in group_means.values()) if finite else None,
            mean_pooled_first_difference_le_1_05=fd_gate)
        endpoints[endpoint] = dict(pairs=pairs,mean_gain_db=mean_gain,group_mean_gain_db=group_means,
            mean_candidate_first_difference_mse=fd_c,mean_corresponding_control_first_difference_mse=fd_b,
            gates=gates,pipeline_seconds_by_method={m:_total([indexed[g,s,m,endpoint]['pipeline_seconds'] for g in GROUPS for s in SEEDS],'pipeline total') for m in METHODS})
    cost = endpoints['wall5s']['pipeline_seconds_by_method']
    ratio = cost[CANDIDATE]/cost['independent']
    if not math.isfinite(ratio):
        raise ValueError('nonfinite pipeline-cost ratio')
    quality_passed = all(v is True for e in endpoints.values() for v in e['gates'].values())
    cost_passed = ratio<=1.10
    evaluable = all(e['gates']['finite_paired_db_gains'] for e in endpoints.values())
    passed = quality_passed and cost_passed
    ordered = [indexed[g,s,m,e] for e in ENDPOINTS for g in GROUPS for s in SEEDS for m in METHODS]
    return dict(schema='0008-training-quality-metrics-v1',
        status=('not_evaluable' if not evaluable else 'continue_to_novelty_review_only' if passed else 'stop_this_candidate'),
        eligible_for_novelty_review=passed,development_only=True,paper_success=False,
        scope='Fixed reused development substrates; no independent-scene interval or paper claim. Endpoints share trajectories.',
        case_count=64,trajectory_count=32,groups=list(GROUPS),seeds=list(SEEDS),methods=list(METHODS),endpoints=endpoints,
        oracle_control_order=list(CONTROLS),oracle_comparator='per-case PSNR-best control at each endpoint; exact ties use fixed control order',
        wall5s_candidate_vs_independent_pipeline_ratio=ratio,
        gates=dict(complete64_grid=True,both_endpoint_quality_gates_pass=quality_passed,
            wall5s_pipeline_ratio_le_1_10=cost_passed,all_passed=passed),cases=ordered)
