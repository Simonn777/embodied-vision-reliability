"""0026 sealed-output evaluation only; no scientific input reader or optimizer."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import study_io as io

LABEL_SHA = '66f3f693fa7473aba5371fe473c4f8c9df39a3dd711a32941f28f3e0380eb151'
CONTROLS = ('incumbent', 'continued_tc', 'G_local', 'F_local', 'F_lift', 'F_all',
            'historical_e_norm', 'historical_e_ts', 'historical_TC_scale')
SELECTORS = {
    'incumbent': ('incumbent',),
    'continued_tc': ('incumbent', 'continued_tc'),
    'G_local': ('incumbent', 'continued_tc', 'G_neutral', 'G_incumbent'),
    'F_local': ('incumbent', 'continued_tc', 'F_neutral', 'F_incumbent'),
    'G_lift': ('incumbent', 'continued_tc', 'G_lift'),
    'F_lift': ('incumbent', 'continued_tc', 'F_lift'),
    'F_all': ('incumbent', 'continued_tc', 'F_neutral', 'F_incumbent', 'F_lift'),
    'G_all': ('incumbent', 'continued_tc', 'G_neutral', 'G_incumbent', 'G_lift'),
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def auc(errors, threshold):
    need(bool(errors) and threshold > 0, 'invalid metric denominator')
    need(all(math.isfinite(e) and 0 <= e <= 180 for e in errors), 'invalid directed error')
    return 100.*math.fsum(max(0., 1.-e/threshold) for e in errors)/len(errors)


def verify_predictions(folder):
    """All required completed evidence is checked before a label path is opened."""
    folder = Path(folder)
    receipt = json.loads((folder/'outer-receipt.json').read_text())
    need(receipt.get('status') == 'completed' and receipt.get('exit_code') == 0
         and receipt.get('error') is None and receipt.get('completed_pairs') == 80,
         'clean complete outer receipt required before labels')
    seal_path = folder/'seal.json'
    need(io.digest(seal_path) == receipt['seal_sha256'], 'outer seal binding mismatch')
    seal = json.loads(seal_path.read_text())
    need(seal.get('schema') == 'evr0026.seal.v1' and seal.get('status') == 'completed', 'invalid seal')
    need(seal.get('source_commit') == receipt['source_commit'], 'receipt commit mismatch')
    need(seal['source_sha256'] == io.source_hashes(), 'experiment source changed')
    need(io.digest(folder/'job.json') == seal['job_sha256'], 'job binding mismatch')
    job = json.loads((folder/'job.json').read_text())
    need(job['source_sha256'] == seal['source_sha256'] and job['commit'] == seal['source_commit'], 'job/source mismatch')
    rows = seal['rows']
    need([r['pair_index'] for r in rows] == list(range(80)), 'incomplete or reordered cohort')
    records = []
    for row in rows:
        index = row['pair_index']
        for key, hash_key in (('path', 'sha256'), ('process', 'process_sha256'), ('log', 'log_sha256'), ('partial', 'partial_sha256')):
            need(Path(row[key]).name == row[key], 'invalid sealed child path')
            need(io.digest(folder/row[key]) == row[hash_key], 'sealed child changed')
        record = json.loads((folder/row['path']).read_text())
        process = json.loads((folder/row['process']).read_text())
        need(process['exit_code'] == 0 and process['termination'] is None, 'unclean pair process')
        need(record['status'] == 'completed' and record['pair_index'] == index
             and record['identity'] == job['selected_pairs'][index]
             and record['source_sha256'] == seal['source_sha256'], 'pair binding mismatch')
        need(record['prior_baseline_sha256'] == job['prior']['baselines'][str(index)]
             and record['native_binding'] == job['native'], 'pair baseline/native mismatch')
        records.append(record)
    return job, records, {'seal_sha256': io.digest(seal_path),
                          'outer_sha256': io.digest(folder/'outer-receipt.json')}


def saved_reference(repo, index, arm, seal, expected_seal_hash):
    need(io.digest(seal) == expected_seal_hash, 'historical reference seal changed')
    body = json.loads(Path(seal).read_text())
    rows = [r for r in body['rows'] if r['pair_index'] == index and r['arm'] == arm]
    need(len(rows) == 1, 'historical reference coverage')
    row = rows[0]
    need(Path(row['path']).name == row['path'], 'historical reference path')
    path = Path(seal).parent/row['path']
    need(io.digest(path) == row['sha256'], 'historical reference changed')
    return json.loads(path.read_text())


def audit_selection(record):
    """Rebuild score-only selections independently, without calling core."""
    bank = {'incumbent': record['incumbent'], **record['attempts']}
    need(set(record['selectors']) == set(SELECTORS), 'selector inventory mismatch')
    for name, members in SELECTORS.items():
        saved = record['selectors'][name]
        need(saved['members'] == list(members), 'selector members mismatch')
        eligible = [key for key in members if bank[key]['status'] == 'completed']
        need(eligible, 'missing incumbent fallback')
        for key in eligible:
            score = bank[key]['final_scores']['pixel_msac']
            need(type(score) in (float, int) and math.isfinite(score) and score >= 0, 'invalid saved score')
        choice = min(eligible, key=lambda key: (bank[key]['final_scores']['pixel_msac'], members.index(key)))
        need(saved['selected_attempt'] == choice, 'label-free selection mismatch')
        need(saved['estimate'] == bank[choice]['estimate'], 'selected estimate mismatch')
        need(saved['score'] == bank[choice]['final_scores']['pixel_msac'], 'selected score mismatch')
        need(math.isfinite(saved['charged_seconds']) and saved['charged_seconds'] >= 0, 'invalid charged core time')
    return bank


def evaluate(args):
    repo = Path(args.repo)
    job, predictions, binding = verify_predictions(args.predictions)
    banks = [audit_selection(record) for record in predictions]
    for path, expected in job['prior']['legacy_sources'].items():
        need(io.digest(repo/path) == expected, 'legacy/metric source changed')
    references = {}
    for index, record in enumerate(predictions):
        identity = job['selected_pairs'][index]
        for name, arm, seal, expected in (
            ('historical_e_norm', 'e_norm', repo/'results/0017-depth/predictions/seal.json', job['prior']['original_seal']),
            ('historical_e_ts', 'e_ts', repo/'results/0017-depth/predictions/seal.json', job['prior']['original_seal']),
            ('historical_TC_scale', 'depth_scale', repo/'results/0019-author-loss/predictions/seal.json', job['prior']['TC_seal']),
            ('incumbent', 'depth_affine', repo/'results/0019-author-loss/predictions/seal.json', job['prior']['TC_seal'])):
            reference = saved_reference(repo, index, arm, seal, expected)
            need(reference['scene'] == identity['scene'] and reference['key'] == identity['key'], 'reference identity differs')
            references[index, name] = reference
        for key in ('estimate', 'geometry'):
            need(record['incumbent'][key] == references[index, 'incumbent'][key], 'incumbent reference differs')
        fresh = record['incumbent']['final_scores']
        old = references[index, 'incumbent']['final_scores']
        need(fresh.keys() == old.keys(), 'incumbent score fields differ')
        for key in fresh:
            if type(old[key]) is float:
                need(math.isfinite(fresh[key]) and abs(fresh[key]-old[key]) <= 1e-10, 'incumbent score differs')
            else:
                need(fresh[key] == old[key], 'incumbent count differs')
    # This is the first numeric label read. No native/scipy/core module imported.
    label_path = repo/'results/0017-depth/evaluation.json'
    need(io.digest(label_path) == LABEL_SHA, 'saved development label binding changed')
    labels = json.loads(label_path.read_text())['rows']
    need([r['pair_index'] for r in labels] == list(range(80)), 'saved label cohort differs')
    metrics = io.module('evr0026_metrics_only', repo/'experiments/0013-pose-headroom/evaluate_sealed.py')
    rows = []
    statuses = {name: Counter() for name in ('continued_tc', 'G_neutral', 'G_incumbent', 'G_lift',
                                           'F_neutral', 'F_incumbent', 'F_lift')}
    for index, record in enumerate(predictions):
        identity = job['selected_pairs'][index]
        label = labels[index]
        need(label['scene'] == identity['scene'] and label['key'] == identity['key'], 'label identity differs')
        gt = label['ground_truth']
        bank = banks[index]
        outcomes, raw = {}, {}
        for name, value in bank.items():
            if value['status'] == 'completed':
                error, _ = metrics.pose_errors(value['estimate']['R'], value['estimate']['t'], gt['R_original'], gt['t_original'])
                raw[name] = {'status': value['status'], **error}
            else:
                raw[name] = {'status': value['status'], 'max_degrees': 180.}
            if name in statuses:
                statuses[name][value['status']] += 1
        for name, selection in record['selectors'].items():
            outcomes[name] = raw[selection['selected_attempt']]
        for name in CONTROLS[-3:]:
            reference = references[index, name]
            if reference['status'] == 'completed':
                error, _ = metrics.pose_errors(reference['estimate']['R'], reference['estimate']['t'], gt['R_original'], gt['t_original'])
                outcomes[name] = {'status': 'completed', **error}
            else:
                outcomes[name] = {'status': reference['status'], 'max_degrees': 180.}
        rows.append({'pair_index': index, 'scene': identity['scene'], 'outcomes': outcomes, 'raw_proposals': raw,
                     'selected_attempts': {k: v['selected_attempt'] for k, v in record['selectors'].items()},
                     'core_seconds': {k: v['charged_seconds'] for k, v in record['selectors'].items()},
                     'worker_timings': record['worker_timings']})
    methods = tuple(SELECTORS)+CONTROLS[-3:]
    def summarize(subset):
        return {name: {'count': len(subset), 'auc_percent': {str(t): auc([r['outcomes'][name]['max_degrees'] for r in subset], t)
                                                          for t in (5, 10, 20)}} for name in methods}
    pooled = summarize(rows)
    scenes = list(dict.fromkeys(r['scene'] for r in rows))
    need(len(scenes) == 5 and all(sum(r['scene'] == s for r in rows) == 16 for s in scenes), 'scene cohort differs')
    scene_metrics = {s: summarize([r for r in rows if r['scene'] == s]) for s in scenes}
    best = max(CONTROLS, key=lambda name: (pooled[name]['auc_percent']['5'], -CONTROLS.index(name)))
    margin = pooled['G_lift']['auc_percent']['5']-pooled[best]['auc_percent']['5']
    scene_margins = {s: scene_metrics[s]['G_lift']['auc_percent']['5']-scene_metrics[s][best]['auc_percent']['5'] for s in scenes}
    loss20 = pooled[best]['auc_percent']['20']-pooled['G_lift']['auc_percent']['20']
    clauses = {'auc5_gain_at_least2pp': margin >= 2., 'positive_in_at_least3_scenes': sum(v > 0 for v in scene_margins.values()) >= 3,
               'no_scene_loses_more_than1pp': min(scene_margins.values()) >= -1., 'auc20_loss_at_most0_5pp': loss20 <= .5}
    result = {'schema': 'evr0026.evaluation.v1', 'status': 'completed', 'bindings': {**binding, 'label_sha256': LABEL_SHA},
              'metric': 'exact empirical directed AUC; percent; all80; invalid180deg',
              'rows': rows, 'pooled': pooled, 'scenes': scene_metrics, 'proposal_status_counts': statuses,
              'selection_counts': {m: dict(Counter(r['selected_attempts'][m] for r in rows)) for m in SELECTORS},
              'timing_summary': {m: {'sum_core_seconds': math.fsum(r['core_seconds'][m] for r in rows),
                                      'median_core_seconds': statistics.median(r['core_seconds'][m] for r in rows)} for m in SELECTORS},
              'decision': {'primary': 'G_lift', 'best_control': best, 'auc5_gain_pp': margin,
                           'scene_gain_pp': scene_margins, 'auc20_loss_pp': loss20, 'clauses': clauses,
                           'development_gate_passed': all(clauses.values()), 'main_goal_complete': False,
                           'manuscript_admitted': False},
              'limitations': ['Already exposed development scenes; no held-out claim.',
                              'Prototype estimator timing and historical base cost; no matched complete-pipeline time claim.',
                              'No rigorous floating-point global certificate or established novelty.']}
    need(verify_predictions(args.predictions)[2] == binding, 'prediction evidence changed during evaluation')
    need(io.digest(label_path) == LABEL_SHA, 'labels changed during evaluation')
    for path, expected in job['prior']['legacy_sources'].items():
        need(io.digest(repo/path) == expected, 'legacy/metric source changed during evaluation')
    io.write_new(args.output, result)
    print(json.dumps(result['decision'], indent=2))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'predictions', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    raise SystemExit(evaluate(parser.parse_args()))
