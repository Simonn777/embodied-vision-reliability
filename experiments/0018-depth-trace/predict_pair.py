"""One log-only depth replay; no reference pose is read or supplied."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import common as c

SCHEMA = 'evr0018.prediction.v1'


def invoke(backend, pair, arm):
    cal = pair['calibration']
    return backend.estimate_monodepth_relative_pose_trace(
        pair['x1'], pair['x2'], pair['depth1'], pair['depth2'],
        cal['camera1'], cal['camera2'], c.options(arm), initial_pose=None,
        trace_capacity=c.TRACE_CAPACITY)


def identity(row, arm, manifest_hash, sources, library, baseline_binding, trace_byte_budget=c.TRACE_BYTES_LIMIT):
    return {'schema': SCHEMA, **{k: row[k] for k in ('pair_index', 'scene', 'key', 'filename')},
            'arm': arm, 'manifest_sha256': manifest_hash, 'source_sha256': sources,
            'library': library, 'dataset_sha256_declared': row['dataset_sha256'],
            'dataset_sha256_verified': row['dataset_sha256'], 'dataset_bytes': row['dataset_bytes'],
            'options': c.options(arm), 'baseline_seal_sha256': baseline_binding['seal_sha256'],
            'baseline_prediction_sha256': baseline_binding['predictions'][f'{row["pair_index"]}:{arm}'],
            'trace_capacity': c.TRACE_CAPACITY, 'trace_byte_budget': trace_byte_budget}


def run(data, manifest_path, baseline_seal, output, arm, index, trace_byte_budget=c.TRACE_BYTES_LIMIT):
    total_start = time.perf_counter()
    with Path(output).open('x+') as stream:
        record = {'schema': SCHEMA, 'status': 'started', 'phase': 'binding',
                  'pair_index': index, 'arm': arm,
                  'timings': {k: None for k in ('binding_seconds', 'input_seconds', 'call_seconds',
                                               'score_seconds', 'trace_save_seconds', 'worker_total_seconds')}}
        save = lambda: c.write_record(stream, record)
        save()
        failure_kind = 'binding'
        try:
            if arm not in c.ARMS or type(index) is not int or not 0 <= index < 80:
                raise ValueError('Invalid replay work item')
            if type(trace_byte_budget) is not int or not 0 < trace_byte_budget <= c.TRACE_BYTES_LIMIT:
                raise ValueError('Invalid remaining trace byte budget')
            manifest, baseline, binding = c.baseline_bundle(manifest_path, baseline_seal)
            row = manifest['selected_pairs'][index]
            source_hashes, mh = c.source_hashes(), c.sha256(manifest_path)
            import poselib
            library = c.verify_library(poselib)
            if Path(data).name != row['filename'] or Path(data).stat().st_size != row['dataset_bytes']:
                raise ValueError('Dataset path or size mismatch')
            if c.sha256(data) != row['dataset_sha256']:
                raise ValueError('Dataset hash mismatch')
            record.update(identity(row, arm, mh, source_hashes, library, binding, trace_byte_budget))
            record['timing_semantics'] = {
                'call_seconds': 'Instrumented native call including observer and native-to-Python trace serialization; observation overhead is not separately isolated',
                'trace_save_seconds': 'Python trace validation/JSON serialization/file write and hash',
                'worker_total_seconds': 'Whole worker; supervisor separately charges outer process wall time'}
            record['timings']['binding_seconds'] = time.perf_counter()-total_start
            failure_kind = 'input'; record['phase'] = 'inputs'; save()
            started = time.perf_counter()
            import h5py
            with h5py.File(data, 'r') as handle:
                pair = c.read_pair(handle, row['key'])
            record.update(input_arrays=c.input_hashes(pair), calibration=c.json_safe(pair['calibration']),
                          input_diagnostics=c.json_safe(pair['input_diagnostics']))
            for key in ('input_arrays', 'calibration', 'input_diagnostics', 'options'):
                if record[key] != baseline[index, arm][key]:
                    raise ValueError('Original input parity differs: '+key)
            record['timings']['input_seconds'] = time.perf_counter()-started
            failure_kind = 'estimator'; record['phase'] = 'estimator'; save()
            started = time.perf_counter()
            try:
                geometry, info, trace = invoke(poselib, pair, arm)
            finally:
                record['timings']['call_seconds'] = time.perf_counter()-started
            failure_kind = 'output'; record['phase'] = 'output'; save()
            started = time.perf_counter()
            record.update(c.base_worker.unpack_result(geometry, info, pair, arm))
            record['timings']['score_seconds'] = time.perf_counter()-started
            failure_kind = 'trace'; record['phase'] = 'trace'; save()
            started = time.perf_counter()
            trace_binding = {key: record[key] for key in (
                'pair_index', 'scene', 'key', 'arm', 'source_sha256', 'manifest_sha256',
                'library', 'baseline_seal_sha256', 'baseline_prediction_sha256',
                'input_arrays', 'calibration', 'dataset_sha256_verified')}
            envelope = {'schema': 'evr0018.trace.v1', 'bindings': trace_binding, 'native': c.json_safe(trace)}
            encoded = json.dumps(envelope, sort_keys=True, allow_nan=False).encode('utf8')
            record['trace_required_bytes'] = len(encoded)+1
            if len(encoded)+1 > trace_byte_budget:
                raise ValueError('Trace exceeds fixed byte cap; replay ineligible')
            trace_path = Path(output).with_suffix('.trace.json')
            with trace_path.open('xb') as trace_stream:
                trace_stream.write(encoded+b'\n')
                trace_stream.flush()
                os.fsync(trace_stream.fileno())
            record.update(trace_path=trace_path.name, trace_sha256=c.sha256(trace_path),
                          trace_bytes=trace_path.stat().st_size)
            # Preserve even an overflowed native trace before rejecting its eligibility.
            record['trace_event_count'] = c.validate_trace(trace)
            record['timings']['trace_save_seconds'] = time.perf_counter()-started
            failure_kind = 'parity'; record['phase'] = 'parity'; save()
            record['parity'] = c.check_parity(record, baseline[index, arm])
            failure_kind = 'binding'; record['phase'] = 'postcheck'; save()
            if c.source_hashes() != source_hashes or c.sha256(manifest_path) != mh:
                raise ValueError('Sources/manifest changed during replay')
            if c.sha256(baseline_seal) != binding['seal_sha256'] or c.verify_library(poselib) != library:
                raise ValueError('Baseline seal or binary changed during replay')
            record.update(status='completed', phase='complete')
            record['timings']['worker_total_seconds'] = time.perf_counter()-total_start
            save()
            return 0
        except BaseException as exc:
            message = type(exc).__name__+': '+str(exc)
            for private in (str(Path(data).resolve()), str(Path(output).resolve()), str(c.ROOT.resolve())):
                message = message.replace(private, '<local-path>')
            record.update(status='error', failed_phase=record['phase'], phase='error', error=message,
                          failure_kind=failure_kind if isinstance(exc, Exception) else 'interruption')
            record['timings']['worker_total_seconds'] = time.perf_counter()-total_start
            save()
            return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'manifest', 'baseline-seal', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--arm', required=True, choices=c.ARMS)
    parser.add_argument('--pair-index', required=True, type=int)
    parser.add_argument('--trace-byte-budget', type=int, default=c.TRACE_BYTES_LIMIT)
    args = parser.parse_args()
    return run(args.data, args.manifest, args.baseline_seal, args.output, args.arm, args.pair_index, args.trace_byte_budget)


if __name__ == '__main__':
    sys.exit(main())
