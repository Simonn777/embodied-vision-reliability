"""One committed 0026 development pair; input allowlist excludes reference poses."""
import argparse
import json
from pathlib import Path
import sys
import time
import study_io as io


def run(args):
    start = time.perf_counter()
    partial = Path(args.output).with_suffix('.partial.json')
    io.checkpoint(partial, {'status': 'started', 'phase': 'binding', 'pair_index': args.index})
    job = json.loads(Path(args.job).read_text())
    sources = io.source_hashes()
    if sources != job['source_sha256']:
        raise ValueError('worker sources changed')
    repo = Path(args.repo)
    baseline_path = repo/'results/0019-author-loss/predictions'/f'pair-{args.index:02d}-depth_affine.json'
    if io.digest(baseline_path) != job['prior']['baselines'][str(args.index)]:
        raise ValueError('worker historical baseline changed')
    baseline = json.loads(baseline_path.read_text())
    c, _ = io.legacy(repo)
    if c.source_hashes() != job['prior']['legacy_sources']:
        raise ValueError('legacy source changed')
    row = job['selected_pairs'][args.index]
    if row['pair_index'] != args.index:
        raise ValueError('worker pair identity mismatch')
    data = Path(args.data_dir)/row['filename']
    if data.stat().st_size != row['dataset_bytes'] or io.digest(data) != row['dataset_sha256']:
        raise ValueError('worker input file changed')
    import numpy as np
    import scipy
    import h5py
    import scipy.optimize._lbfgsb as lbfgsb
    if (np.__version__, scipy.__version__, h5py.__version__) != ('2.0.2', '1.13.1', '3.14.0'):
        raise ValueError('unqualified scientific runtime')
    if io.digest(args.native_library) != job['native']['library_sha256'] or io.digest(args.native_receipt) != job['native']['receipt_sha256']:
        raise ValueError('native qualification binding changed')
    from native_bridge.bridge import NativeRefiner
    from core import process_pair
    native = NativeRefiner(args.native_library, args.native_receipt)
    binding_seconds = time.perf_counter()-start
    t0 = time.perf_counter()
    with h5py.File(data, 'r') as handle:
        pair = c.read_pair(handle, row['key'])
    actual = dict(input_arrays=c.input_hashes(pair), calibration=c.json_safe(pair['calibration']),
                  input_diagnostics=c.json_safe(pair['input_diagnostics']))
    for key, value in actual.items():
        if value != baseline[key]:
            raise ValueError('converted input differs from historical baseline: '+key)
    input_seconds = time.perf_counter()-t0
    def progress(snapshot):
        io.checkpoint(partial, snapshot)
    result = process_pair(pair, baseline, native, c.base, args.index, progress=progress)
    result.update(identity=row, source_sha256=sources, prior_baseline_sha256=job['prior']['baselines'][str(args.index)],
                  input_bindings=actual, native_binding=job['native'],
                  runtime={'numpy': np.__version__, 'scipy': scipy.__version__, 'h5py': h5py.__version__,
                           'module_files': {name: {'basename': Path(mod.__file__).name, 'sha256': io.digest(mod.__file__)}
                                            for name, mod in (('numpy', np), ('scipy', scipy), ('h5py', h5py), ('lbfgsb', lbfgsb))}},
                  worker_timings={'binding_seconds': binding_seconds, 'input_seconds': input_seconds,
                                  'total_seconds': time.perf_counter()-start})
    if sources != io.source_hashes() or io.digest(data) != row['dataset_sha256']:
        raise ValueError('worker source/input changed during execution')
    io.write_new(args.output, result)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('job', 'repo', 'data-dir', 'native-library', 'native-receipt', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--index', required=True, type=int, choices=range(80))
    sys.exit(run(parser.parse_args()))
