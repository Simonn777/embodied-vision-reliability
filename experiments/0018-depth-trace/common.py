"""Frozen-input trace replay bindings; no labels or native calls at import."""
import contextlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LIBRARIES = ROOT / 'configs/0018-libraries.json'
PROTOCOL = ROOT / 'protocols/0018-depth-lo-trace.md'
MANIFEST = ROOT / 'configs/0017-development-manifest.json'
ARMS = ('depth_scale', 'depth_affine')
PARITY_RTOL = 1e-12
TRACE_CAPACITY = 50000
TRACE_BYTES_LIMIT = 64*1024*1024
COHORT_TRACE_BYTES_LIMIT = 512*1024*1024


@contextlib.contextmanager
def aliases(mapping):
    absent = object()
    previous = {name: sys.modules.get(name, absent) for name in mapping}
    sys.modules.update(mapping)
    try:
        yield
    finally:
        for name, old in previous.items():
            if old is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def _load(name, filename, dependencies=None):
    path = ROOT / 'experiments/0017-depth-information' / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with aliases(dependencies or {}):
        spec.loader.exec_module(module)
    return module


base = _load('evr0018_base_common', 'common.py')
base_worker = _load('evr0018_base_worker', 'predict_pair.py', {'common': base})
base_runner = _load('evr0018_base_runner', 'run_diagnostic.py',
                    {'common': base, 'predict_pair': base_worker})
np = base.np
sha256, json_safe = base.sha256, base.json_safe
write_new, write_record = base.write_new, base.write_record
load_manifest, read_pair = base.load_manifest, base.read_pair
validate_pose = base.validate_pose
bound_file = base_runner.bound_file
input_hashes = base_worker.input_hashes


def options(arm):
    if arm not in ARMS:
        raise ValueError('Only the two frozen depth arms are admitted')
    return base.options(arm)


def source_hashes():
    result = base.source_hashes()
    paths = [HERE/name for name in ('common.py', 'predict_pair.py', 'run_diagnostic.py',
                                   'diagnose_trace.py', 'instrument_source.py', 'build_instrumented.py',
                                   'trace0018.h.in', 'trace_pybind.cc.in',
                                   'qualify_native.py', 'qualify_observer.cpp', 'run_operation.py')]
    paths += [PROTOCOL, LIBRARIES]
    result.update({str(path.relative_to(ROOT)): sha256(path) for path in paths})
    return result


def library_config():
    config = json.loads(LIBRARIES.read_text())
    original = json.loads(base.LIBRARIES.read_text())['current']
    if config.get('baseline') != original:
        raise ValueError('Trace config changed original library identity')
    if config['instrumented']['version'] != original['version']:
        raise ValueError('Instrumented version differs')
    if config['instrumented']['binary_sha256'] == original['binary_sha256']:
        raise ValueError('Trace binary must be separately identified')
    return config


def verify_library(backend):
    return base.verify_library(backend, library_config()['instrumented'])


def baseline_bundle(manifest_path, seal_path):
    """Verify all 320 sealed original records before selecting the 160 depth records."""
    manifest, seal, records = base_runner.verify_seal(manifest_path, seal_path)
    if len(records) != 320 or any(r['status'] != 'completed' for r in records):
        raise ValueError('Original cohort is not the complete successful baseline')
    by_item = {(r['pair_index'], r['arm']): r for r in records if r['arm'] in ARMS}
    if len(by_item) != 160:
        raise ValueError('Original depth cohort is incomplete')
    row_map = {(r['pair_index'], r['arm']): r for r in seal['rows']}
    binding = {'seal_sha256': sha256(seal_path),
               'predictions': {f'{i}:{arm}': row_map[i, arm]['sha256'] for i, arm in by_item}}
    return manifest, by_item, binding


def parity(actual, expected, path='value'):
    """Exact discrete values; predeclared relative-max bound for each float."""
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError('Parity mismatch: '+path)
    elif type(expected) is int:
        if type(actual) is not int or actual != expected:
            raise ValueError('Discrete parity mismatch: '+path)
    elif type(expected) is float:
        if type(actual) not in (float, int) or not math.isfinite(actual) or not math.isfinite(expected):
            raise ValueError('Nonfinite/type parity mismatch: '+path)
        if abs(actual-expected) > PARITY_RTOL*max(1., abs(actual), abs(expected)):
            raise ValueError('Floating parity mismatch: '+path)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError('Array parity mismatch: '+path)
        for index, (a, b) in enumerate(zip(actual, expected)):
            parity(a, b, path+'['+str(index)+']')
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError('Object parity mismatch: '+path)
        for key in expected:
            parity(actual[key], expected[key], path+'.'+key)
    else:
        raise ValueError('Unsupported parity value: '+path)


def check_parity(record, baseline):
    exact = ('input_arrays', 'calibration', 'input_diagnostics', 'options',
             'native_inliers_packbits_big_hex', 'mask_sha256', 'mask_length', 'kept_count')
    for key in exact:
        if record.get(key) != baseline.get(key):
            raise ValueError('Exact baseline parity mismatch: '+key)
    for key in ('estimate', 'geometry', 'returned_ransac_stats'):
        parity(record[key], baseline[key], key)
    pairs = []
    def collect(actual, expected):
        if type(expected) is float:
            pairs.append((actual, expected))
        elif isinstance(expected, list):
            for a, b in zip(actual, expected): collect(a, b)
        elif isinstance(expected, dict):
            for key in expected: collect(actual[key], expected[key])
    for key in ('estimate', 'geometry', 'returned_ransac_stats'):
        collect(record[key], baseline[key])
    bitwise = sum(struct.pack('!d', a) == struct.pack('!d', b) for a, b in pairs)
    return {'passed': True, 'float_relative_max_tolerance': PARITY_RTOL,
            'discrete_and_mask_exact': True, 'raw_translation_gauge_preserved': True,
            'floating_components': len(pairs), 'bitwise_equal_float_components': bitwise,
            'all_float_components_bitwise_equal': bitwise == len(pairs),
            'max_normalized_float_error': max((abs(a-b)/max(1., abs(a), abs(b)) for a, b in pairs), default=0.)}


def validate_trace(trace):
    """Basic native coverage checks; the separate diagnosis validates LO semantics."""
    json_safe(trace)
    if (not isinstance(trace, dict) or trace.get('schema') != 'evr0018.native-trace.v1'
        or trace.get('complete') is not True or type(trace.get('dropped_events')) is not int or trace['dropped_events'] != 0):
        raise ValueError('Trace incomplete or overflowed')
    if trace.get('capacity') != TRACE_CAPACITY:
        raise ValueError('Trace capacity differs')
    events = trace.get('events')
    if not isinstance(events, list) or not events or len(events) > TRACE_CAPACITY:
        raise ValueError('Invalid trace event coverage')
    if type(trace.get('event_count')) is not int or trace['event_count'] != len(events):
        raise ValueError('Native event count differs')
    kinds = {'lo_begin', 'initial', 'attempt', 'exit', 'lo_end', 'proposal', 'incumbent', 'pipeline'}
    pipeline = lambda event, phase: event.get('kind') == 'pipeline' and event.get('phase') == phase
    if not pipeline(events[0], 'begin') or not pipeline(events[-1], 'returned'):
        raise ValueError('Missing native pipeline boundary')
    digest = trace.get('proposal_trace_hash_hex')
    if not isinstance(digest, str) or len(digest) != 16 or any(ch not in '0123456789abcdef' for ch in digest):
        raise ValueError('Invalid noncryptographic native proposal checksum')
    for key in ('sampled_draws', 'scored_proposals'):
        if type(trace.get(key)) is not int or trace[key] < 0:
            raise ValueError('Invalid native proposal count')
    lo_states = {}
    for event in events:
        if (not isinstance(event, dict) or event.get('kind') not in kinds
            or type(event.get('lo_id')) is not int or type(event.get('finite')) is not bool):
            raise ValueError('Malformed native trace event')
        kind, lo_id = event['kind'], event['lo_id']
        if kind in ('proposal', 'incumbent', 'pipeline'):
            if lo_id != -1:
                raise ValueError('Non-LO event has an LO id')
            continue
        stage = event.get('stage')
        if lo_id < 0 or stage not in ('proposal_lo', 'terminal_lo', 'final_refinement'):
            raise ValueError('Invalid LO stage or identity')
        if kind == 'lo_begin':
            if lo_id in lo_states:
                raise ValueError('Repeated LO identity')
            lo_states[lo_id] = [stage, 'lo_begin']
        else:
            if lo_id not in lo_states or lo_states[lo_id][0] != stage:
                raise ValueError('LO lacks matching begin/stage')
            prior = lo_states[lo_id][1]
            allowed = {'initial': ('lo_begin',), 'attempt': ('initial', 'attempt'),
                       'exit': ('initial', 'attempt'), 'lo_end': ('exit',)}
            if prior not in allowed[kind]:
                raise ValueError('Incomplete or reordered LO events')
            lo_states[lo_id][1] = kind
    if any(state[1] != 'lo_end' for state in lo_states.values()):
        raise ValueError('Unclosed native LO trace')
    return len(events)
