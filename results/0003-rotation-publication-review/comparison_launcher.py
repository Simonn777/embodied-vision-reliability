#!/usr/bin/env python3
"""Read only published Git blobs; invoke the prewritten independent matrix checker.

No candidate imports, cache/archive access, model, or projection are performed.
Writes only a /tmp review bundle. The excerpt and derived data retain its data_terms.
"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import time

EXPECTED_CHECKER_SHA256 = '230d214a14fa25fc56961298ff424cd0450b5e4e1364f88da33d56beaa3bb3c3'
CHECKER = Path('/tmp/0003-rotation-independent-math.py')
PREFIX = Path('/tmp/0003-rotation-native')

def sha(data):
    return hashlib.sha256(data).hexdigest()

def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')

def rational(value):
    return F(int(value['numerator']), int(value['denominator']))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--ref', required=True)
    args = parser.parse_args()
    def blob(path):
        return subprocess.check_output(['git', '-C', args.repo, 'show', f'{args.ref}:{path}'])
    paths = {
        'excerpt': 'results/0003-rotation-stageB-20261017/audit/camera-excerpt.json',
        'diagnostic': 'results/0003-rotation-stageB-20261017/audit/diagnostic.json',
    }
    raw = {key: blob(path) for key, path in paths.items()}
    excerpt = json.loads(raw['excerpt'])
    cloud = json.loads(raw['diagnostic'])
    assert sha(CHECKER.read_bytes()) == EXPECTED_CHECKER_SHA256
    assert excerpt['scalar_count'] == 25 and excerpt['frame'] == 0
    assert excerpt['unchanged_numeric_values'] is True
    assert cloud['camera_access'] == excerpt['camera_access']
    payloads = {}
    values = {}
    scalar_checks = 0
    for name, size, key in [('intrinsics', 3, 'K'), ('extrinsics', 4, 'RT')]:
        access = excerpt['camera_access'][name]
        assert access['status'] == 'completed' and access['counts_complete'] is True
        assert access['header']['shape'] == [849, size, size]
        assert access['header']['dtype'] == '<f4' and access['header']['order'] == 'C'
        assert access['header']['itemsize'] == 4
        assert len(access['values']) == size and all(len(row) == size for row in access['values'])
        payload = b''
        numbers = []
        for index, entry in enumerate(item for row in access['values'] for item in row):
            stored = bytes.fromhex(entry['stored_bytes_hex'])
            assert len(stored) == 4 and entry['byte_offset_in_payload'] == 4 * index
            number = struct.unpack('<f', stored)[0]
            assert entry['finite'] is True and math.isfinite(number)
            assert number.hex() == entry['float_hex']
            assert F.from_float(number) == F(entry['exact_rational'])
            payload += stored
            numbers.append(number)
            scalar_checks += 1
        accounting = access['accounting']
        assert len(payload) == accounting['retained_bytes'] == accounting['numeric_span_bytes'] == access['numeric_bytes_returned']
        assert accounting['retained_scalars'] == size**2 and accounting['discarded_bytes'] == 0
        assert sha(payload) == accounting['span_sha256']
        payloads[key] = payload
        values[key] = [numbers[i*size:(i+1)*size] for i in range(size)]
    assert scalar_checks == 25
    source_input = {'K_hex': payloads['K'].hex(), 'RT_hex': payloads['RT'].hex(), 'byte_order': 'little'}
    input_path = Path(str(PREFIX) + '-input.json')
    stdout_path = Path(str(PREFIX) + '-stdout.json')
    stderr_path = Path(str(PREFIX) + '-stderr.log')
    receipt_path = Path(str(PREFIX) + '-execution.json')
    result_path = Path(str(PREFIX) + '-review.json')
    dump(input_path, source_input)
    dump(Path(str(PREFIX) + '-data-terms.json'), excerpt['data_terms'])
    for key in raw:
        Path(str(PREFIX) + '-' + key + '.json').write_bytes(raw[key])
    command = [sys.executable, '-I', str(CHECKER), '--input', str(input_path)]
    started = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    process = subprocess.run(command, capture_output=True)
    elapsed = time.perf_counter() - start
    ended = datetime.now(timezone.utc).isoformat()
    stdout_path.write_bytes(process.stdout)
    stderr_path.write_bytes(process.stderr)
    receipt = {
        'reviewed_candidate': args.ref,
        'input_commit': cloud['input_commit'],
        'execution_commit': cloud['execution_commit'],
        'receipt_sha256': cloud['receipt_sha256'],
        'command': command,
        'started_utc': started, 'ended_utc': ended, 'elapsed_seconds': elapsed,
        'observed_child_returncode': process.returncode,
        'checker_sha256': sha(CHECKER.read_bytes()),
        'launcher_sha256': sha(Path(__file__).read_bytes()),
        'input_sha256': sha(input_path.read_bytes()),
        'stdout_sha256': sha(process.stdout), 'stderr_sha256': sha(process.stderr),
        'published_inputs': {key: {'path': paths[key], 'sha256': sha(raw[key])} for key in raw},
        'data_terms': excerpt['data_terms'],
        'boundary': 'Parent observed subprocess.run returncode and elapsed time for only the prewritten reviewer. This receipt has no separate observation of its own outer process exit. No candidate executable, archive, cache, model, or additional GT data accessed.',
    }
    dump(receipt_path, receipt)
    assert process.returncode == 0 and process.stderr == b''
    independent = json.loads(process.stdout)
    checked = cloud['statistics']
    own = independent['statistics']
    exact_count = 0
    maximum_double_exact_difference = F(0)
    for name, diffname in [('RRt_minus_I', 'RRt'), ('RtR_minus_I', 'RtR')]:
        assert len(checked['exact_fraction'][name]) == 3
        for row in range(3):
            for column in range(3):
                exact_value = rational(own[name][row][column])
                assert exact_value == rational(checked['exact_fraction'][name][row][column])
                difference = F.from_float(checked['double'][name][row][column]) - exact_value
                assert difference == rational(checked['double_minus_exact_fraction'][diffname][row][column])
                maximum_double_exact_difference = max(maximum_double_exact_difference, abs(difference))
                exact_count += 1
    det = rational(own['det'])
    assert det == rational(checked['exact_fraction']['det_R'])
    detdiff = F.from_float(checked['double']['det_R']) - det
    assert detdiff == rational(checked['double_minus_exact_fraction']['determinant'])
    maximum_double_exact_difference = max(maximum_double_exact_difference, abs(detdiff))
    exact_count += 1
    tau = rational(independent['original_f32_tolerance'])
    assert tau == F.from_float(checked['original_tau'])
    metric_names = [('RRt', 'RRt_max', 'maxabs_RRt'), ('RtR', 'RtR_max', 'maxabs_RtR'), ('determinant', 'det_minus_one_abs', 'abs_det_minus_one')]
    gate_defects = {}
    for cloud_name, own_name, double_name in metric_names:
        defect = rational(own[own_name])
        v = checked['violations'][cloud_name]
        assert F.from_float(checked['double'][double_name]) == defect
        assert F.from_float(v['value']) == defect
        assert F.from_float(v['threshold']) == tau
        assert F.from_float(v['excess']) == defect-tau
        assert F.from_float(v['ratio']) == defect/tau
        assert v['passes'] == (defect <= tau)
        assert checked['original_gate_checks'][cloud_name] == (defect <= tau)
        gate_defects[cloud_name] = {'exact_fraction': str(defect), 'decimal': float(defect),
                                   'defect_over_tau': float(defect/tau), 'passes': defect <= tau}
    for own_name, cloud_name in [('RRt_minus_I','RRt'),('RtR_minus_I','RtR')]:
        matrix = [[rational(x) for x in row] for row in own[own_name]]
        maximum = max(abs(x) for row in matrix for x in row)
        first = next([i,j] for i,row in enumerate(matrix) for j,x in enumerate(row) if abs(x) == maximum)
        assert checked['double']['argmax_' + cloud_name] == first
    for gram, normkey, dotkey in [('RRt', 'row_squared_norms', 'row_offdiagonal_dots'), ('RtR', 'column_squared_norms', 'column_offdiagonal_dots')]:
        for i in range(3):
            assert F.from_float(checked['double'][normkey][i]) == rational(own[gram][i][i])
        assert len(checked['double'][dotkey]) == 3
        for i,j,value in checked['double'][dotkey]:
            assert F.from_float(value) == rational(own[gram][i][j])
    assert checked['original_gate_checks'] == {'finite':True, 'K_exact_bottom_row':True, 'RT_exact_bottom_row':True, 'positive_focal':True, 'RRt':False, 'RtR':False, 'determinant':False}
    assert not checked['original_gate_pass'] and not checked['original_rotation_gate_pass']
    assert not checked['exact_fraction']['rotation_gate_pass']
    assert independent['proper_rotation_gate_fails'] and all(independent['transpose_invariant'].values())
    own_bounds = independent['storage_rounding_test']
    their_bounds = checked['single_final_f32_rounding']
    for first, second in [('unit_roundoff','u'),('gram_bound','gram'),('det_bound','determinant')]:
        assert rational(own_bounds[first]) == rational(their_bounds['bounds'][second])
    assert their_bounds['excluded'] == own_bounds['single_final_storage_rounding_hypothesis_excluded'] == True
    expected_items = {(entry['matrix'],entry['row'],entry['column']):entry for entry in independent['half_grid_fingerprint']['entries']}
    groups = checked['binary16_exact_representability']['groups']
    group_selectors = {
        'K9': [('K', i, j) for i in range(3) for j in range(3)],
        'R9': [('RT', i, j) for i in range(3) for j in range(3)],
        'RT16': [('RT', i, j) for i in range(4) for j in range(4)],
        'translation3': [('RT', i, 3) for i in range(3)],
        'homogeneous4': [('RT', 3, j) for j in range(4)],
    }
    half_summaries = {}
    for name, keys in group_selectors.items():
        group = groups[name]
        assert group['total'] == len(keys) == len(group['values'])
        expected_count = sum(expected_items[key]['representable'] for key in keys)
        assert group['exactly_representable_count'] == expected_count
        assert group['all_exactly_representable'] == (expected_count == len(keys))
        for key, entry in zip(keys, group['values']):
            own_entry = expected_items[key]
            assert entry['index'] == list(key[1:])
            assert entry['original_f32_le_hex'] == own_entry['input_float32_hex']
            assert entry['exactly_binary16'] == own_entry['representable']
        half_summaries[name] = {'representable':expected_count, 'total':len(keys)}
    assert cloud['status'] == 'completed_diagnostic_original_rotation_gate_failed'
    assert cloud['counts_complete'] is True
    summary = {
        'status':'APPROVE_with_explicit_scope', 'reviewed_candidate':args.ref,
        'input_commit':cloud['input_commit'], 'execution_commit':cloud['execution_commit'],
        'receipt_sha256':cloud['receipt_sha256'],
        'source_checker_sha256':sha(CHECKER.read_bytes()),
        'comparison_launcher_sha256':sha(Path(__file__).read_bytes()),
        'published_blob_sha256':{paths[key]:sha(raw[key]) for key in raw},
        'review_input_sha256':sha(input_path.read_bytes()), 'review_output_sha256':sha(process.stdout),
        'review_execution_receipt_sha256':sha(receipt_path.read_bytes()),
        'saved_scalar_byte_hex_rational_offset_checks':scalar_checks,
        'payload_bytes':sum(len(v) for v in payloads.values()),
        'payload_sha256':{key:sha(data) for key,data in payloads.items()},
        'exact_gram_entries_checked':18, 'exact_determinant_checked':1,
        'exact_determinant':str(det), 'max_absolute_published_double_minus_exact':str(maximum_double_exact_difference),
        'gate_defects':gate_defects,
        'half_grid_counts':half_summaries,
        'half_grid_unique_flags_checked':len(expected_items),
        'half_grid_nonrepresentable':[[key[0],key[1],key[2]] for key,item in expected_items.items() if not item['representable']],
        'transpose_invariance_pass':all(independent['transpose_invariant'].values()),
        'single_final_f32_rounding_excluded':own_bounds['single_final_storage_rounding_hypothesis_excluded'],
        'all_three_exceed_single_final_f32_bounds':all(own_bounds['strict_exceeds_bound'].values()),
        'original_gate_retained_failed':True,
        'stale_camera_access_state':{'value':cloud['camera_access_state'], 'documented':'Top-level stale starting label; completed overall/member statuses and complete counts are separately present.'},
        'data_terms':excerpt['data_terms'],
        'limits':[
            'Independent exact math over only the 25 published saved float32 values; no archive/cache/original annotation reread and no model/projection.',
            'Exact determinant uses pivoted elimination; half flags use rational grid membership, not the cloud half converter.',
            'Source/output execution receipt observes checker child exit, not a further outer supervisor exit.',
            'Numerical trigger confirmed. Actual generation pipeline and root cause remain unknown; half-grid membership is compatibility evidence only.',
            'No geometric validity, valid-visible GT support, cohort independence, or improvement conclusion follows.',
        ],
    }
    dump(result_path, summary)
    print(json.dumps({k:summary[k] for k in ['status','reviewed_candidate','exact_determinant','max_absolute_published_double_minus_exact','gate_defects','half_grid_counts','half_grid_nonrepresentable']}, indent=2))
    print('review_path=' + str(result_path))
    print('execution_path=' + str(receipt_path))

if __name__ == '__main__':
    main()
