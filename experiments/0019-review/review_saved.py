#!/usr/bin/env python3
"""Independent saved-JSON audit; no scientific-module imports or cache access."""
import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('independent0017', HERE.parent/'0017-review/review_saved.py')
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)
ARMS = ('depth_scale', 'depth_affine')
ANGLE_ATOL = 1e-7
AUC_ATOL = 1e-10
FREEZE = '542ca5ecee4ef9fc3316de8e6eae0a9d96429406'
EXTRA_SOURCES = {'experiments/0019-author-loss/'+x for x in
                 ('common.py', 'predict_pair.py', 'run_diagnostic.py', 'evaluate_saved.py', 'run_operation.py')}
EXTRA_SOURCES.add('protocols/0019-author-loss.md')


def parity(actual, expected):
    """Independent strict discrete / relative-max float comparison and counts."""
    floats = []
    def visit(a, b):
        if type(b) is float:
            u.need(type(a) in (int, float) and math.isfinite(a) and math.isfinite(b), 'Parity finite/type')
            error = abs(a-b)/max(1., abs(a), abs(b))
            u.need(error <= 1e-12, 'Pre-final numeric parity')
            floats.append((struct.pack('!d', a) == struct.pack('!d', b), error))
        elif isinstance(b, dict):
            u.need(isinstance(a, dict) and a.keys() == b.keys(), 'Parity object')
            for k in b: visit(a[k], b[k])
        elif isinstance(b, list):
            u.need(isinstance(a, list) and len(a) == len(b), 'Parity list')
            for x, y in zip(a, b): visit(x, y)
        else:
            u.need(type(a) is type(b) and a == b, 'Parity discrete')
    visit(actual, expected)
    return {'available': True, 'passed': True, 'float_relative_max_tolerance': 1e-12,
            'discrete_and_mask_exact': True, 'final_geometry_constrained': False,
            'floating_components': len(floats), 'bitwise_equal_float_components': sum(v[0] for v in floats),
            'max_normalized_float_error': max((v[1] for v in floats), default=0.)}


def harm(before, after):
    return after-before >= 1. or before < 5. <= after


def arithmetic(identities, baseline, records, reference, evaluation):
    names = (*u.ARMS, *(a+'_truncated_cauchy' for a in ARMS))
    u.need(len(evaluation['rows']) == 80, 'Evaluation all80 coverage')
    rows, max_angle = [], 0.
    for i, (identity, gtrow, saved) in enumerate(zip(identities, reference['rows'], evaluation['rows'])):
        for item in (gtrow, saved):
            u.need(all(item[k] == identity[k] for k in ('pair_index', 'scene', 'key')), 'Evaluation identity')
        gt = u.reference(gtrow['ground_truth'])
        outcomes = {a: u.outcome(baseline[i, a], gt) for a in u.ARMS}
        outcomes.update({a+'_truncated_cauchy': u.outcome(records[i, a], gt) for a in ARMS})
        u.compare(saved['outcomes'], outcomes, 'Independent pose errors', ANGLE_ATOL)
        for name, out in outcomes.items():
            if out['error']:
                max_angle = max(max_angle, *(abs(v-saved['outcomes'][name]['error'][k]) for k, v in out['error'].items()))
        comparisons = {}
        for arm in ARMS:
            name = arm+'_truncated_cauchy'; after = outcomes[name]['penalized_max_degrees']
            for before_arm in (arm, 'e_norm'):
                before = outcomes[before_arm]['penalized_max_degrees']
                comparisons[name+'_vs_'+before_arm] = {'error_delta_degrees': after-before,
                    'material_harm': harm(before, after), 'material_benefit': harm(after, before),
                    'damaged_5deg': before < 5. <= after, 'rescued_5deg': after < 5. <= before}
        u.compare(saved['comparisons'], comparisons, 'Independent comparisons', 2*ANGLE_ATOL)
        times = {a+'_truncated_cauchy': records[i, a]['timings'] for a in ARMS}
        u.need(saved['new_timings_by_arm'] == times, 'Evaluation timing copy')
        rows.append({'pair_index': i, 'scene': identity['scene'], 'outcomes': outcomes, 'comparisons': comparisons})
    summaries = {a: u.summary([r['outcomes'][a] for r in rows]) for a in names}
    scenes = {s: {a: u.summary([r['outcomes'][a] for r in rows if r['scene'] == s]) for a in names} for s in u.SCENES}
    u.compare(evaluation['summaries'], summaries, 'All80 summary', AUC_ATOL)
    u.compare(evaluation['scene_summaries'], scenes, 'Scene16 summaries', AUC_ATOL)
    comparisons = {}
    for arm in ARMS:
        name = arm+'_truncated_cauchy'
        for before_arm in (arm, 'e_norm'):
            key = name+'_vs_'+before_arm
            comparisons[key] = {'auc_delta_percentage_points': {t: 100*(summaries[name]['auc_fraction'][t]-summaries[before_arm]['auc_fraction'][t]) for t in ('5','10','20')},
                **{k: sum(r['comparisons'][key][k] for r in rows) for k in ('material_harm','material_benefit','damaged_5deg','rescued_5deg')}}
    u.compare(evaluation['comparisons'], comparisons, 'Aggregate comparisons', 100*AUC_ATOL)
    costs = {a+'_truncated_cauchy': math.fsum(records[i, a]['timings']['charged_total_seconds'] for i in range(80)) for a in ARMS}
    u.compare(evaluation['new_charged_seconds_by_arm'], costs, 'Charged costs', 1e-10)
    u.need(evaluation['new_method_improvement_established'] is False and evaluation['manuscript_admitted'] is False, 'Known-control scope')
    return {'max_absolute_angle_discrepancy_degrees': max_angle, 'summaries': summaries,
            'scene_summaries': scenes, 'comparisons': comparisons, 'charged_seconds_by_arm': costs,
            'native_call_seconds_by_arm': {a: math.fsum(records[i, a]['timings']['call_seconds'] or 0 for i in range(80)) for a in ARMS}}


def audit(repo):
    repo = Path(repo); folder = repo/'results/0019-author-loss'; pred = folder/'predictions'
    manifest = repo/'configs/0017-development-manifest.json'
    bs = repo/'results/0017-depth/predictions/seal.json'; be = repo/'results/0017-depth/evaluation.json'
    m, oldseal, ref, baseline, oldbindings = u.verify_files(repo, manifest, bs, be)
    seal, evaluation = u.load(pred/'seal.json'), u.load(folder/'evaluation.json')
    sources = {p: u.digest(repo/p) for p in u.SOURCE_PATHS | EXTRA_SOURCES}
    u.need(seal['schema'] == 'evr0019.seal.v1' and seal['status'] == 'completed' and seal['error'] is None, 'New seal status')
    u.need(seal['source_sha256'] == sources and seal['manifest_sha256'] == u.digest(manifest), 'New seal source/manifest')
    u.need(seal['library'] == oldseal['library'] and seal['verified_assets'] == oldseal['verified_assets'], 'Library/assets changed')
    oldrows = {(r['pair_index'], r['arm']): r for r in oldseal['rows']}
    binding = {'seal_sha256': u.digest(bs), 'predictions': {f'{i}:{a}': oldrows[i,a]['sha256'] for i in range(80) for a in ARMS}}
    u.need(seal['baseline_binding'] == binding, 'Baseline binding')
    integrity = pred/'integrity.json'
    u.need(u.digest(integrity) == seal['integrity_sha256'], 'Integrity hash')
    u.need(u.load(integrity) == {'manifest_sha256': u.digest(manifest), 'source_sha256': sources,
            'library': oldseal['library'], 'baseline_binding': binding, 'verified_assets': oldseal['verified_assets']}, 'Integrity content')
    order = [(i,a) for i in range(80) for a in ARMS[i%2:]+ARMS[:i%2]]
    u.need([(r['pair_index'],r['arm']) for r in seal['rows']] == order, '160 rotated coverage')
    records, hashes, stats, nonempty_logs = {}, {}, [], []
    for row in seal['rows']:
        i, arm = row['pair_index'], row['arm']; identity = m['selected_pairs'][i]; before = baseline[i,arm]
        paths = {kind: u.safe(pred, row[key]) for kind,key in (('final','path'),('raw','raw_path'),('process','process_path'))}
        for kind,key in (('final','sha256'),('raw','raw_sha256'),('process','process_sha256')):
            u.need(u.digest(paths[kind]) == row[key], 'Sealed file hash'); hashes[paths[kind].name] = row[key]
        raw, final, process = (u.load(paths[k]) for k in ('raw','final','process'))
        log = u.safe(pred, process['log_path']); hashes[log.name] = u.digest(log)
        u.need(hashes[log.name] == process['log_sha256'], 'Log hash')
        if log.stat().st_size: nonempty_logs.append({'path':log.name,'bytes':log.stat().st_size})
        u.need(process['raw_worker_path'] == row['raw_path'] and process['raw_worker_sha256'] == row['raw_sha256'], 'Raw/process link')
        expected = {k: before[k] for k in ('pair_index','scene','key','filename','dataset_bytes','dataset_sha256_declared','dataset_sha256_verified','library','manifest_sha256','input_arrays','calibration','input_diagnostics')}
        expected.update(schema='evr0019.prediction.v1', arm=arm, source_sha256=sources,
                        baseline_seal_sha256=u.digest(bs), baseline_prediction_sha256=oldrows[i,arm]['sha256'])
        options = u.expected_options(arm); options['bundle'] = {'loss_type':'TRUNCATED_CAUCHY'}
        expected['options'] = options
        u.need(all(raw.get(k) == v for k,v in expected.items()), 'Worker bindings/settings/inputs')
        u.need(row['scene'] == identity['scene'] and row['key'] == identity['key'], 'Seal identity')
        code, term = process['exit_code'], process['termination']
        if raw['status'] == 'completed' and code == 0 and term is None:
            status = 'completed'
            for key in ('native_inliers_packbits_big_hex','mask_sha256','mask_length','kept_count'):
                u.need(raw[key] == before[key], 'Pre-final mask/count differs')
            p = parity(raw['returned_ransac_stats'], before['returned_ransac_stats']); stats.append(p)
            u.need(raw['prefinal_parity'] == p, 'Pre-final receipt differs')
            u.rotation(raw['estimate']['R']); u.unit(raw['estimate']['t'])
            u.need(all(raw['geometry'][k] == raw['estimate'][k] for k in ('R','t')), 'Raw geometry differs')
        elif raw['status'] == 'error' and raw.get('failure_kind') == raw.get('failed_phase') == 'estimator' and code == 1 and term is None:
            status = 'estimator_error'
        elif raw['status'] == 'started' and raw.get('phase') == 'estimator' and (term == 'pair_timeout' or term is None and code != 0):
            status = 'timeout' if term == 'pair_timeout' else 'estimator_error'
        else: raise ValueError('Ineligible worker/process state')
        if status != 'completed': p = {'available':False,'passed':None,'reason':status,'final_geometry_constrained':False}
        timings = dict(raw['timings']); timings['charged_total_seconds'] = process['wall_seconds']
        for k in ('binding_seconds','input_seconds','call_seconds','score_seconds','worker_total_seconds'):
            v = timings.get(k); timings.setdefault(k, None)
            u.need(v is not None or status != 'completed' and k not in ('binding_seconds','input_seconds'), 'Timing absent')
            if v is not None: u.need(type(v) in (int,float) and v >= 0, 'Timing invalid')
        measured = math.fsum(timings[k] or 0 for k in ('binding_seconds','input_seconds','call_seconds','score_seconds'))
        u.need(process['wall_seconds']+1e-6 >= measured, 'Uncharged measured work')
        if timings['worker_total_seconds'] is not None:
            u.need(measured <= timings['worker_total_seconds']+1e-6 <= process['wall_seconds']+2e-6, 'Timing nesting')
        u.need(final == {**raw,'status':status,'prefinal_parity':p,'timings':timings,'process_receipt':process} and row['status'] == status, 'Final normalization')
        u.need(process['wall_seconds'] <= 90 and process['sampled_peak_rss_kib'] <= 8*1024*1024 and process.get('watchdog_error') is None, 'Worker resource/watchdog')
        records[i,arm] = final
    counts = Counter(r['status'] for r in records.values())
    u.need(seal['all_completed_prefinal_parity_passed'] is True and seal['completed_calls'] == counts['completed'] and seal['native_failure_calls'] == 160-counts['completed'], 'Seal parity/coverage')
    u.need(seal['wall_seconds'] <= 1800, 'Cohort time limit')
    u.need(evaluation['schema'] == 'evr0019.evaluation.v1' and evaluation['status'] == 'completed' and evaluation['pair_count'] == 80, 'Evaluation eligibility')
    u.need(evaluation['source_sha256'] == sources and evaluation['manifest_sha256'] == u.digest(manifest) and evaluation['seal_sha256'] == u.digest(pred/'seal.json') and evaluation['reference_evaluation_sha256'] == u.digest(be), 'Evaluation bindings')
    arithmetic_result = arithmetic(m['selected_pairs'], baseline, records, ref, evaluation)
    outer = {}
    for mode in ('prediction','evaluation'):
        p = u.load(folder/(mode+'-process.json')); log = folder/(mode+'.log')
        u.need(p['status'] == 'completed' and p['exit_code'] == 0 and p['termination'] is None and p['remaining_process_groups_after_cleanup'] == [], 'Outer process failure')
        u.need(p['source_commit'] == FREEZE and p['helper_sha256'] == sources['experiments/0019-author-loss/run_operation.py'], 'Outer freeze')
        u.need(p['log_sha256'] == u.digest(log) and p['wall_seconds'] <= p['hard_timeout_seconds'] == 1800 and p['peak_group_rss_kib'] <= p['group_rss_limit_kib'] == 8388608, 'Outer log/resource')
        lines = [json.loads(line) for line in log.read_text().splitlines()]
        if mode == 'prediction': u.need([v['completed_pairs'] for v in lines] == list(range(1,81)) and lines[-1]['replays'] == 160, 'Progress log coverage')
        else: u.need(len(lines) == 1 and lines[0]['status'] == 'completed' and lines[0]['comparisons'] == evaluation['comparisons'], 'Evaluation log')
        outer[mode] = p
    for path, sha in sources.items(): u.need(u.digest(repo/path) == sha, 'Source changed during review')
    for name, sha in hashes.items(): u.need(u.digest(u.safe(pred,name)) == sha, 'Record changed during review')
    return {'schema':'evr0019.independent-review.v1','status':'passed','freeze_commit':FREEZE,
        'review_source_sha256':u.digest(__file__),'independent_helper_sha256':u.digest(HERE.parent/'0017-review/review_saved.py'),
        'angle_tolerance_degrees':ANGLE_ATOL,'auc_fraction_tolerance':AUC_ATOL,'pair_count':80,'new_record_count':160,
        'status_counts':dict(counts),'all_completed_prefinal_parity_passed':True,
        'prefinal_floating_components':sum(p['floating_components'] for p in stats),
        'prefinal_bitwise_equal_floating_components':sum(p['bitwise_equal_float_components'] for p in stats),
        'prefinal_max_normalized_float_error':max((p['max_normalized_float_error'] for p in stats),default=0.),
        'nonempty_worker_logs':nonempty_logs,'outer_processes':outer,
        'bindings':{'source_sha256':sources,'manifest_sha256':u.digest(manifest),'baseline_seal_sha256':u.digest(bs),
            'baseline_evaluation_sha256':u.digest(be),'seal_sha256':u.digest(pred/'seal.json'),
            'evaluation_sha256':u.digest(folder/'evaluation.json'),'prediction_file_sha256':hashes},
        'limitations':['Known author-loss control, not a novel method or manuscript admission.',
            'Five permanently development landmark scenes; dependent pairs, no held-out generalization claim.',
            'No HDF5, images, native binary, estimator, or production evaluation module accessed.',
            'Input arrays, residual diagnostics and native execution are hash-bound saved evidence, not scientifically rerun.',
            'Historical timing is not a matched-compute comparison; sampled RSS can miss peaks.'],**arithmetic_result}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--repo',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    with a.output.open('x+') as f:
        report={'status':'started'}; f.write(json.dumps(report)+'\n'); f.flush()
        try: report=audit(a.repo)
        except BaseException as exc:
            report={'status':'failed','error':type(exc).__name__+': '+str(exc),'review_source_sha256':u.digest(__file__)}
            f.seek(0); json.dump(report,f,indent=2,allow_nan=False); f.write('\n'); f.truncate(); f.flush(); raise
        f.seek(0); json.dump(report,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n'); f.truncate()
    print(json.dumps({'status':report['status'],'records':report['new_record_count'],'max_angle_discrepancy':report['max_absolute_angle_discrepancy_degrees']}))
