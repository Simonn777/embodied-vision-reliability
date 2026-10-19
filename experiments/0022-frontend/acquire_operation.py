"""Acquire only the two prospectively fixed 0022 frontend asset batches."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import acquire

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ('configs/0022-source-model-assets.json', 'configs/0022-wheel-assets.json')


def run(args):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if head != args.expected_commit:
        raise ValueError('Unexpected asset-source commit')
    bound = [*MANIFESTS, 'protocols/0022-traceable-frontend-assets.md',
             'experiments/0022-frontend/acquire.py', 'experiments/0022-frontend/acquire_operation.py']
    bound_bytes = {}
    for relative in bound:
        frozen = subprocess.check_output(['git', 'show', head + ':' + relative], cwd=ROOT)
        actual = (ROOT/relative).read_bytes()
        if actual != frozen:
            raise ValueError('Asset source binding differs from frozen commit: ' + relative)
        bound_bytes[relative] = actual
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError('Asset operation destination must be fresh')
    out.mkdir(parents=True)
    record = {'schema': 'evr0022.asset-operation.v1', 'source_commit': head,
              'status': 'started', 'asset_or_dependency_only': True,
              'benchmark_payload_access': False,
              'bindings': {p: hashlib.sha256(bound_bytes[p]).hexdigest() for p in bound},
              'batches': []}
    receipt = out/'operation.json'
    began = time.perf_counter()
    def save():
        receipt.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    save()
    try:
        for manifest, dirname in zip(MANIFESTS, ('source-model-assets', 'wheel-assets')):
            result = acquire.acquire_all(json.loads(bound_bytes[manifest]), out/dirname)
            record['batches'].append({'manifest': manifest, 'directory': dirname,
                                      'status': result['status'], 'actual_body_bytes': result['actual_body_bytes'],
                                      'ledger_sha256': hashlib.sha256((out/dirname/'ledger.json').read_bytes()).hexdigest()})
            save()
        record['status'] = 'completed'
    except BaseException as exc:
        record.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        record['wall_seconds'] = time.perf_counter()-began
        save()
        print(json.dumps({k:record[k] for k in ('status', 'wall_seconds', 'batches')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
