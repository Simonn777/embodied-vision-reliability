"""One-shot two-phase image-information pilot; no ground-truth access."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LEGACY = HERE.parent / '0013-pose-headroom'

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result

base = module('_evr0015_runner_base', LEGACY / 'predict_pair.py')
_saved = sys.modules.get('predict_pair')
sys.modules['predict_pair'] = base
try:
    infrastructure = module('_evr0015_runner_infrastructure', LEGACY / 'run_baseline.py')
finally:
    if _saved is None:
        sys.modules.pop('predict_pair', None)
    else:
        sys.modules['predict_pair'] = _saved
metrics = module('_evr0015_runner_metrics', LEGACY / 'evaluate_sealed.py')
geometry = module('_evr0015_runner_geometry', HERE / 'estimate_pair.py')
PROTOCOL = REPO / 'protocols/0015-photometric-information-development.md'
RECEIPT = REPO / 'results/0015-admission/acquisition.json'
NAMES = ('refine_patch.py', 'select_queries.py', 'predict_pair.py', 'estimate_pair.py',
         'run_diagnostic.py', 'evaluate_diagnostic.py', 'acquire_images.py')
PHASES = ('photometric', 'geometric')
TOTAL_TIMEOUT = 1800.0

def sha(path):
    return base.sha256(path)

def require(condition, message):
    if not condition:
        raise ValueError(message)

def bindings():
    paths = [HERE / name for name in NAMES] + [LEGACY / name for name in
        ('predict_pair.py', 'run_baseline.py', 'evaluate_sealed.py')]
    paths += [PROTOCOL, REPO / 'configs/0015/posebench-images-archive.json', RECEIPT]
    return {str(path.relative_to(REPO)): sha(path) for path in paths}

def phase_identity(phase, sources, mh, bh, dataset, previous=None):
    return {'schema': 'evr0015.'+phase+'-seal.v1', 'phase': phase,
            'source_sha256': sources, 'manifest_sha256': mh, 'baseline_seal_sha256': bh,
            'dataset_sha256_declared': dataset, 'photometric_seal_sha256': previous}

def validate_geometry(record, photo, baseline, old_row, index, photo_sha, mh, bh, sources):
    expected = {'schema': 'evr0015.geometric.v1', 'status': 'completed',
        'pair_index': index, 'key': old_row['key'], 'manifest_sha256': mh,
        'baseline_seal_sha256': bh, 'baseline_prediction_sha256': old_row['sha256'],
        'dataset_sha256_declared': photo['dataset_sha256_declared'],
        'mask_sha256': photo['mask_sha256'], 'kept_count': photo['kept_count'],
        'photometric_sha256': photo_sha, 'library': baseline['library'],
        'calibration': baseline['calibration'], 'settings': base.SETTINGS,
        'source_sha256': sources['experiments/0015-photometric/estimate_pair.py'],
        'baseline_worker_sha256': sources['experiments/0013-pose-headroom/predict_pair.py'],
        'photometric_source_bindings': geometry.source_bindings(base),
        'selected_count': photo['selected_count'], 'admitted_count': photo['admitted_count'],
        'target_coordinate_changed_count': photo['target_coordinate_changed_count'],
        'photometric_queries': photo['queries'], 'full_correspondence_count': 10000,
        'full_original_reference': {'baseline_prediction_sha256': old_row['sha256'],
            'estimate': baseline['estimate'], 'rerun': False},
        'arm_order': list(geometry.ARM_ORDER), 'failed_calls': 0}
    for key, value in expected.items():
        require(record.get(key) == value, 'geometric binding mismatch: '+key)
    ids = [q['index'] for q in photo['queries'] if q['admitted']]
    require(record.get('success_query_indices') == ids, 'successful subset changed')
    require(set(record.get('arms', {})) == set(geometry.ARM_ORDER), 'all three arms required')
    calls = 0
    for name in geometry.ARM_ORDER:
        arm = record['arms'][name]
        n = 10000 if name == 'full_refined' else len(ids)
        require(arm.get('input_count') == n, 'arm input count changed')
        if name != 'full_refined' and n < 8:
            require(arm == {'status': 'unavailable', 'reason': 'fewer_than_eight_successful_queries',
                           'input_count': n, 'native_calls': 0}, 'invalid unavailable subset')
            continue
        calls += 1
        require(arm.get('status') == 'completed' and arm.get('native_calls') == 1,
                'native call not completed exactly once')
        metrics.valid_rotation(arm['estimate']['R'])
        metrics.unit_translation(arm['estimate']['t'])
        require(isinstance(arm.get('call_seconds'), (int, float)) and arm['call_seconds'] >= 0,
                'native time missing')
        mask = arm.get('returned_ransac_stats', {}).get('inliers')
        require(isinstance(mask, list) and len(mask) == n and all(type(x) is bool for x in mask),
                'native mask coverage differs')
        order = list(range(10000)) if name == 'full_refined' else ids
        require(arm.get('returned_inlier_indices_original') == [i for i, v in zip(order, mask) if v],
                'native inlier map differs')
    require(record.get('native_calls_started') == calls and record.get('native_calls_completed') == calls,
            'native call accounting differs')
    require(record.get('timing_seconds', {}).get('worker_elapsed', -1) >= 0, 'worker timing missing')
    base.json_safe(record)

def safe_child(parent, name):
    require(isinstance(name, str) and Path(name).name == name and name not in ('.', '..'),
            'seal filename unsafe')
    path = Path(parent) / name
    require(path.resolve() == path.absolute(), 'seal child symlink not permitted')
    return path

def verify_phase(manifest, original_seal, seal_path, phase, photo_seal=None):
    """Validate the complete cohort before advancing a phase or loading labels."""
    metadata, baseline, originals, mh, bh = metrics.verify_seal(manifest, original_seal)
    sources = bindings()
    photos, photo_hash = None, None
    if phase == 'geometric':
        require(photo_seal is not None, 'geometric seal requires photometric barrier')
        _, photo_hash, photos = verify_phase(manifest, original_seal, photo_seal, 'photometric')
    seal, digest = metrics.load_hashed_json(seal_path)
    expected = phase_identity(phase, sources, mh, bh, metadata['dataset_sha256'], photo_hash)
    require(all(seal.get(k) == v for k, v in expected.items()), 'phase seal binding differs')
    require(seal.get('complete') is True and seal.get('eligible_for_next_phase') is True
            and seal.get('error') is None, 'incomplete phase cannot advance')
    require(len(seal.get('rows', [])) == 16, 'all16 phase records required')
    records, seen = [], set()
    for index, (row, (old_row, old)) in enumerate(zip(seal['rows'], originals)):
        require(row.get('pair_index') == index and row.get('key') == old_row['key']
                and row.get('status') == 'completed', 'phase order/coverage differs')
        path = safe_child(Path(seal_path).resolve().parent, row.get('path'))
        require(path.name not in seen, 'duplicate phase record')
        seen.add(path.name)
        record, record_sha = metrics.load_hashed_json(path)
        require(record_sha == row.get('sha256'), 'phase record hash mismatch')
        process_path = safe_child(path.parent, row.get('process_path'))
        process, process_hash = metrics.load_hashed_json(process_path)
        require(process_hash == row.get('process_sha256') and process.get('exit_code') == 0
                and process.get('termination') is None and process.get('watchdog_error') is None
                and process.get('raw_worker_sha256') == record_sha, 'worker process failed or changed')
        if phase == 'photometric':
            _, _, verified, _, _ = geometry.verify_inputs(base, metrics, manifest, original_seal,
                                                         path, record_sha, index)
            require(verified == record, 'photo verifier disagreement')
        else:
            photo_row, photo_record = photos[index]
            validate_geometry(record, photo_record, old, old_row, index, photo_row['sha256'], mh, bh, sources)
        records.append((row, record))
    return seal, digest, records

def run(data, manifest, original_seal, images_root, image_receipt, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    sources = bindings()
    infrastructure.write_new(output/'started.json', {
        'schema': 'evr0015.operation.v1', 'utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': sources, 'pair_timeout_seconds': 90, 'total_timeout_seconds': TOTAL_TIMEOUT,
        'rss_limit_kib': infrastructure.RSS_LIMIT_KIB, 'thread_environment_per_worker': 1,
        'command': 'python experiments/0015-photometric/run_diagnostic.py --data <cache> --manifest configs/0012-development-manifest.json --baseline-seal results/0013-baseline-calibration-compatibility/predictions/seal.json --images-root <private-images> --image-receipt results/0015-admission/acquisition.json --output-dir results/0015-photometric'})
    mh, bh, metadata, error, phase, photo_hash, photos = None, None, {}, None, None, None, None
    try:
        metadata, baseline, originals, mh, bh = metrics.verify_seal(manifest, original_seal)
        require(Path(data).stat().st_size == metadata['dataset_bytes'] and sha(data) == metadata['dataset_sha256'],
                'actual dataset differs from frozen input')
        require(sha(image_receipt) == sha(RECEIPT), 'image receipt differs from pinned acquisition')
        library = infrastructure.expected_library()
        infrastructure.write_new(output/'integrity.json', {'dataset_sha256_verified': metadata['dataset_sha256'],
            'manifest_sha256': mh, 'baseline_seal_sha256': bh, 'library': library,
            'image_acquisition_receipt_sha256': sha(image_receipt), 'numeric_arrays_decoded': False})
        for phase in PHASES:
            folder = output/phase
            folder.mkdir()
            rows, phase_error = [], None
            try:
                for index, (old_row, old) in enumerate(originals):
                    remaining = TOTAL_TIMEOUT - (time.perf_counter()-started)
                    if remaining <= 0:
                        raise RuntimeError('total operation timeout')
                    stem = 'pair-%02d' % index
                    target = folder/(stem+'.json')
                    command = [sys.executable, str(HERE/('predict_pair.py' if phase=='photometric' else 'estimate_pair.py')),
                        '--data', str(Path(data).resolve()), '--manifest', str(Path(manifest).resolve()),
                        '--baseline-seal', str(Path(original_seal).resolve()), '--pair-index', str(index),
                        '--output', str(target.resolve())]
                    if phase == 'photometric':
                        command += ['--images-root', str(Path(images_root).resolve()),
                                    '--image-receipt', str(Path(image_receipt).resolve())]
                    else:
                        command += ['--photometric-record', str((output/'photometric'/photos[index][0]['path']).resolve()),
                                    '--photometric-sha256', photos[index][0]['sha256']]
                    process = infrastructure.run_child(command, folder/(stem+'.log'), remaining)
                    process['raw_worker_sha256'] = sha(target) if target.exists() else None
                    process['raw_worker_path'] = target.name
                    procpath = folder/(stem+'.process.json')
                    infrastructure.write_new(procpath, process)
                    require(process['exit_code'] == 0 and process['termination'] is None,
                            phase+' subprocess failed at pair '+str(index))
                    record, digest = metrics.load_hashed_json(target)
                    if phase == 'photometric':
                        geometry.verify_inputs(base, metrics, manifest, original_seal, target, digest, index)
                    else:
                        validate_geometry(record, photos[index][1], old, old_row, index,
                                          photos[index][0]['sha256'], mh, bh, sources)
                    rows.append({'pair_index': index, 'key': old_row['key'], 'path': target.name,
                        'sha256': digest, 'status': 'completed', 'process_path': procpath.name,
                        'process_sha256': sha(procpath)})
                    print(json.dumps({'phase': phase, 'pair_index': index, 'status': 'completed',
                        'selected': record.get('selected_count'), 'admitted': record.get('admitted_count'),
                        'wall_seconds': process['wall_seconds']}), flush=True)
                require(bindings() == sources and sha(manifest) == mh and sha(original_seal) == bh,
                        'frozen source or input changed during operation')
            except BaseException as exc:
                phase_error = {'type': type(exc).__name__, 'message': str(exc)}
                for private in (data, manifest, original_seal, images_root, image_receipt, output_dir):
                    phase_error['message'] = phase_error['message'].replace(str(Path(private).resolve()), '<local-path>')
            complete = len(rows) == 16 and phase_error is None
            infrastructure.write_new(folder/'seal.json', {
                **phase_identity(phase, sources, mh, bh, metadata['dataset_sha256'], photo_hash),
                'complete': complete, 'eligible_for_next_phase': complete, 'rows': rows,
                'error': phase_error, 'wall_seconds_since_operation_start': time.perf_counter()-started})
            if not complete:
                raise RuntimeError('phase failed; see preserved '+phase+' seal')
            if phase == 'photometric':
                _, photo_hash, photos = verify_phase(manifest, original_seal, folder/'seal.json', phase)
            else:
                verify_phase(manifest, original_seal, folder/'seal.json', phase, output/'photometric/seal.json')
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        for private in (data, manifest, original_seal, images_root, image_receipt, output_dir):
            error['message'] = error['message'].replace(str(Path(private).resolve()), '<local-path>')
    complete = phase == 'geometric' and error is None
    infrastructure.write_new(output/'operation.json', {'schema': 'evr0015.operation-result.v1',
        'complete': complete, 'eligible_for_evaluation': complete, 'error': error,
        'last_phase': phase, 'wall_seconds': time.perf_counter()-started,
        'source_sha256': sources, 'labels_accessed': False})
    print(json.dumps({'complete': complete, 'error': error}), flush=True)
    return 0 if complete else 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'manifest', 'baseline-seal', 'images-root', 'image-receipt', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.data, args.manifest, args.baseline_seal, args.images_root,
                         args.image_receipt, args.output_dir))
