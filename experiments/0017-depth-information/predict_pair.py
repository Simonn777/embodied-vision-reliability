"""One fixed known-method call; reads no reference pose or scientific labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import common as c
np = c.np
SCHEMA = 'evr0017.prediction.v1'


def input_hashes(pair):
    result = {}
    for name in ('x1', 'x2', 'depth1', 'depth2'):
        array = np.ascontiguousarray(pair[name])
        result[name] = {'sha256': hashlib.sha256(array.tobytes()).hexdigest(),
                        'dtype': str(array.dtype), 'shape': list(array.shape)}
    return result


def invoke(backend, pair, arm, options=None):
    opt = c.options(arm) if options is None else options
    cal = pair['calibration']
    args = (pair['x1'], pair['x2'])
    if arm.startswith('depth_'):
        args += (pair['depth1'], pair['depth2'])
        method = backend.estimate_monodepth_relative_pose
    else:
        method = backend.estimate_relative_pose
    return method(*args, cal['camera1'], cal['camera2'], opt, initial_pose=None)


def unpack_result(result, info, pair, arm):
    native_pose = result.pose if arm.startswith('depth_') else result
    estimate = {'R': np.asarray(native_pose.R), 't': np.asarray(native_pose.t)}
    c.validate_pose(estimate)
    geometry = None
    if arm.startswith('depth_'):
        geometry = {**estimate, **{name: float(getattr(result, name)) for name in ('scale', 'shift1', 'shift2')}}
        c.json_safe(geometry)
    stats = dict(info)
    mask = c.validate_mask(stats.pop('inliers'), len(pair['x1']))
    g = c.prepare_geometry(pair['x1'], pair['x2'], pair['calibration'])
    output = {'estimate': c.json_safe(estimate), 'geometry': c.json_safe(geometry),
              'final_scores': c.final_scores(estimate, g),
              'returned_ransac_stats': c.json_safe(stats),
              'native_inliers_packbits_big_hex': np.packbits(mask, bitorder='big').tobytes().hex(),
              'mask_sha256': hashlib.sha256(mask.astype(np.bool_).tobytes()).hexdigest(),
              'mask_length': len(mask), 'kept_count': int(mask.sum()),
              'returned_stats_semantics': 'RANSAC stats and epipolar mask precede final refinement; mask is not final active directional depth support',
              'depth_diagnostics': None}
    if geometry is not None:
        output['depth_diagnostics'] = c.depth_diagnostics(geometry, g, pair['depth1'], pair['depth2'], mask)
    return c.json_safe(output)


def run(data, manifest_path, output, arm, index):
    start_total = time.perf_counter()
    with Path(output).open('x+') as stream:
        record = {'schema': SCHEMA, 'status': 'started', 'phase': 'binding',
                  'arm': arm, 'pair_index': index,
                  'timings': {name: None for name in ('input_seconds','call_seconds','score_seconds','charged_total_seconds')}}
        def save():
            c.write_record(stream, record)
        save()
        failure_kind = 'binding'
        try:
            manifest = c.load_manifest(manifest_path)
            if arm not in c.ARMS or type(index) is not int or not 0 <= index < 80:
                raise ValueError('Invalid work item')
            row = manifest['selected_pairs'][index]
            record.update({key: row[key] for key in ('scene', 'key', 'filename')})
            hashes = c.source_hashes()
            manifest_hash = c.sha256(manifest_path)
            record.update(source_sha256=hashes, manifest_sha256=manifest_hash,
                          dataset_sha256_declared=row['dataset_sha256'], dataset_bytes=row['dataset_bytes'],
                          options=c.options(arm))
            import poselib
            record['library'] = c.verify_library(poselib)
            if Path(data).name != row['filename'] or Path(data).stat().st_size != row['dataset_bytes']:
                raise ValueError('Dataset path/size binding mismatch')
            if c.sha256(data) != row['dataset_sha256']:
                raise ValueError('Dataset hash binding mismatch')
            record['dataset_sha256_verified'] = row['dataset_sha256']
            record['phase'] = 'inputs'; failure_kind = 'input'; save()
            started = time.perf_counter()
            import h5py
            with h5py.File(data, 'r') as handle:
                pair = c.read_pair(handle, row['key'])
            record.update(calibration=c.json_safe(pair['calibration']),
                          input_diagnostics=c.json_safe(pair['input_diagnostics']), input_arrays=input_hashes(pair))
            record['timings']['input_seconds'] = time.perf_counter()-started
            record['phase'] = 'estimator'; failure_kind = 'estimator'; save()
            started = time.perf_counter()
            try:
                result, info = invoke(poselib, pair, arm)
            except Exception:
                record['timings']['call_seconds'] = time.perf_counter()-started
                raise
            record['timings']['call_seconds'] = time.perf_counter()-started
            record['phase'] = 'output'; failure_kind = 'output'; save()
            started = time.perf_counter()
            record.update(unpack_result(result, info, pair, arm))
            record['timings']['score_seconds'] = time.perf_counter()-started
            failure_kind = 'binding'; record['phase'] = 'postcheck'; save()
            if c.source_hashes() != hashes or c.sha256(manifest_path) != manifest_hash:
                raise ValueError('Source or manifest changed during worker')
            if c.verify_library(poselib) != record['library']:
                raise ValueError('Library changed during worker')
            record.update(status='completed', phase='complete')
            record['timings']['charged_total_seconds'] = time.perf_counter()-start_total
            save()
            return 0
        except BaseException as exc:
            message = type(exc).__name__+': '+str(exc)
            for private in (str(Path(data).resolve()), str(Path(output).resolve()), str(c.ROOT.resolve())):
                message = message.replace(private, '<local-path>')
            record.update(status='error', failed_phase=record['phase'], phase='error', error=message,
                          failure_kind=failure_kind if isinstance(exc, Exception) else 'interruption')
            record['timings']['charged_total_seconds'] = time.perf_counter()-start_total
            save()
            return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'manifest', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--arm', choices=c.ARMS, required=True)
    parser.add_argument('--pair-index', type=int, required=True)
    args = parser.parse_args()
    return run(args.data, args.manifest, args.output, args.arm, args.pair_index)


if __name__ == '__main__':
    sys.exit(main())
