#!/usr/bin/env python3
"""Parent arithmetic review from public block measurements; no image/model IO."""
import hashlib
import itertools
import json
import math
import pathlib
import sys


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def same(a, b):
    return math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-14)


path = pathlib.Path(sys.argv[1])
report = json.loads(path.read_text())
rows = report['cases']
groups = ['d1', 'd2', 'd3', 'd4']
conditions = ['white', 'colored']
seeds = [0, 1]
methods = ['blindspot', 'safe_blindspot', 'residual', 'r2r_alpha05']
expected = set(itertools.product(groups, conditions, seeds, methods))
indexed = {}
for row in rows:
    key = tuple(row[k] for k in ['group', 'condition', 'seed', 'method'])
    check(key in expected and key not in indexed, ('bad/duplicate case', key))
    score = row['score']
    check(score['shape'] == [122, 122] and score['count'] == 14884, key)
    check(score['clipped'] is False and score['data_range'] == 1, key)
    blocks = score['blocks']
    check(len(blocks) == 64, key)
    covered = set()
    for b in blocks:
        width, height = b['col_stop']-b['col_start'], b['row_stop']-b['row_start']
        check(b['count'] == width*height, key)
        pixels = set(itertools.product(range(b['row_start'], b['row_stop']), range(b['col_start'], b['col_stop'])))
        check(not covered.intersection(pixels), key)
        covered.update(pixels)
        check(math.isfinite(b['sse']) and b['sse'] >= 0, key)
    check(covered == set(itertools.product(range(122), range(122))), key)
    pixel_sse = math.fsum(b['sse'] for b in blocks)
    mse = pixel_sse / 14884
    check(same(pixel_sse, score['sse']) and same(mse, score['mse']), key)
    psnr = None if mse == 0 else 10*math.log10(1/mse)
    check((psnr is None and score['psnr_db'] is None) or (psnr is not None and same(psnr, score['psnr_db'])), key)
    fd = score['first_difference']
    for direction in ['horizontal', 'vertical']:
        check(fd[direction]['count'] == 14762, key)
        check(same(fd[direction]['mse'], fd[direction]['sse']/14762), key)
    grad_sse = fd['horizontal']['sse'] + fd['vertical']['sse']
    grad_mse = grad_sse / 29524
    check(fd['combined']['count'] == 29524 and same(grad_sse, fd['combined']['sse']) and same(grad_mse, fd['combined']['mse']), key)
    indexed[key] = {'psnr': psnr, 'mse': mse, 'grad_mse': grad_mse, 'hash': score['prediction_sha256_f32le']}
check(set(indexed) == expected and report['case_count'] == 64, 'grid')
paired = []
group_gain = {}
white_equal = True
white_safe_equal = True
white_no_harm = True
for g in groups:
    per_group = []
    for seed in seeds:
        baselines = ['blindspot', 'safe_blindspot', 'r2r_alpha05']
        best = min(baselines, key=lambda m: indexed[g, 'colored', seed, m]['mse'])
        c, b = indexed[g, 'colored', seed, 'residual'], indexed[g, 'colored', seed, best]
        check(c['psnr'] is not None and b['psnr'] is not None, 'zero-error case needs separate review')
        gain = c['psnr'] - b['psnr']
        paired.append({'group': g, 'seed': seed, 'best_baseline': best, 'gain_db': gain, 'candidate_gradient_mse': c['grad_mse'], 'baseline_gradient_mse': b['grad_mse']})
        per_group.append(gain)
        w = indexed[g, 'white', seed, 'blindspot']
        r = indexed[g, 'white', seed, 'residual']
        s = indexed[g, 'white', seed, 'safe_blindspot']
        white_equal &= w['hash'] == r['hash']
        white_safe_equal &= w['hash'] == s['hash']
        white_no_harm &= w['mse'] == r['mse']
    group_gain[g] = math.fsum(per_group)/2
mean_gain = math.fsum(p['gain_db'] for p in paired)/8
mean_cgrad = math.fsum(p['candidate_gradient_mse'] for p in paired)/8
mean_bgrad = math.fsum(p['baseline_gradient_mse'] for p in paired)/8
gates = {
    'colored_pair_mean_ge_0_3_db': mean_gain >= .3,
    'colored_at_least3_group_means_gt0': sum(v > 0 for v in group_gain.values()) >= 3,
    'colored_each_group_mean_ge_minus0_2_db': all(v >= -.2 for v in group_gain.values()),
    'colored_first_difference_le_1_05': mean_cgrad <= 1.05*mean_bgrad,
    'white_predictions_bitwise_equal': white_equal,
    'white_safe_predictions_bitwise_equal': white_safe_equal,
    'white_psnr_harm_zero': white_no_harm,
}
status = 'continue_mechanism_exploration' if all(gates.values()) else 'stop_this_candidate'
check(report['gates'] == gates, 'gate disagreement')
check(report['status'] == status, 'status disagreement')
check(same(mean_gain, report['mean_colored_gain_db']), 'mean gain')
check(all(same(group_gain[g], report['group_mean_colored_gain_db'][g]) for g in groups), 'group gain')
check(report['development_only'] is True and report['paper_success'] is False, 'claim bounds')
result = {
    'artifact_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'audit_script_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
    'independent_block_arithmetic_matches': True,
    'case_count': 64,
    'mean_gain_db': mean_gain,
    'group_mean_gains_db': group_gain,
    'gates': gates,
    'decision': status,
    'colored_pairs': paired,
    'limits': 'Recomputes public scalar/block arithmetic only; does not reconstruct predictions, re-infer, or establish untouched data efficacy.',
}
print(json.dumps(result, indent=2, allow_nan=False))
