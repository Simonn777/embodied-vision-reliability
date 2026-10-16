#!/usr/bin/env python3
"""Post-hoc, read-only analysis of experiment 0001's frozen prediction rows.

Standard library only. No renderer, tracker, fitting, or experiment code is run.
The existing acceptance flags are the outcomes; saved thresholds only validate them.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import random
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'results/0001-pilot'
CONDITIONS = ('clean', 'occlusion', 'appearance', 'abrupt_motion')
METHODS = ('static', 'temporal', 'random', 'accept_all')
TEST_IDS = tuple(range(120, 160))
TOLERANCE = .025
BOOTSTRAP_SEED = 20261018
BOOTSTRAP_REPEATS = 1000
FIELDS = ('sequence_id partition condition frame gt_x gt_y detected pred_x pred_y '
          'observed_area anisotropy initial_observed_area score_static score_temporal '
          'innovation_px error_norm tie_key accepted_static accepted_temporal '
          'accepted_random accepted_accept_all').split()
NULLABLE = ('pred_x', 'pred_y', 'anisotropy', 'score_static', 'score_temporal',
            'innovation_px', 'error_norm')
# All strata are post-hoc analyses. The appearance split at 10 is an analytical
# memory-stage split, not a boundary prespecified by the rendering protocol.
PHASES = {
    'clean': (('nominal', 1, 31, 'protocol condition'),),
    'occlusion': (('before', 1, 7, 'protocol intervention boundary'),
                  ('partial_first', 8, 13, 'protocol intervention boundary'),
                  ('full', 14, 17, 'protocol intervention boundary'),
                  ('partial_second', 18, 23, 'protocol intervention boundary'),
                  ('recovery', 24, 31, 'protocol intervention boundary')),
    'appearance': (('before', 1, 7, 'protocol intervention boundary'),
                   ('switch_transient', 8, 9, 'post-hoc memory stage'),
                   ('stable_wrong_identity', 10, 31, 'post-hoc memory stage')),
    'abrupt_motion': (('before', 1, 15, 'protocol intervention boundary'),
                      ('turn', 16, 16, 'protocol intervention boundary'),
                      ('after', 17, 31, 'protocol intervention boundary')),
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result

    def bad_constant(value):
        raise ValueError('Nonfinite JSON value: ' + value)

    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad_constant)


def partition(sequence):
    if not 0 <= sequence < 160:
        raise ValueError('Unexpected sequence ID')
    return 'fit' if sequence < 80 else 'dev' if sequence < 100 else 'cal' if sequence < 120 else 'test'


def parse_row(raw, thresholds):
    """Validate a frozen row without recalibrating or generating predictions."""
    if set(raw) != set(FIELDS) or any(value is None for value in raw.values()):
        raise ValueError('Malformed CSV row')
    row = dict(raw)
    for name in ('sequence_id', 'frame', 'observed_area', 'initial_observed_area'):
        try:
            row[name] = int(raw[name])
        except ValueError as exc:
            raise ValueError('Invalid integer: ' + name) from exc
        if str(row[name]) != raw[name]:
            raise ValueError('Noncanonical integer: ' + name)
    if row['partition'] != partition(row['sequence_id']):
        raise ValueError('Unexpected sequence partition')
    if row['condition'] not in CONDITIONS or not 1 <= row['frame'] <= 31:
        raise ValueError('Unexpected condition/frame')
    for name in ('detected',) + tuple('accepted_' + method for method in METHODS):
        if raw[name] not in ('True', 'False'):
            raise ValueError('Invalid boolean: ' + name)
        row[name] = raw[name] == 'True'
    for name in ('gt_x', 'gt_y') + NULLABLE:
        if raw[name] == '' and name in NULLABLE and not row['detected']:
            row[name] = None
        else:
            try:
                row[name] = float(raw[name])
            except ValueError as exc:
                raise ValueError('Invalid numeric field: ' + name) from exc
            if not math.isfinite(row[name]):
                raise ValueError('Nonfinite numeric field: ' + name)
    if not all(0 <= row[name] <= 95 for name in ('gt_x', 'gt_y')):
        raise ValueError('Ground-truth coordinate outside image')
    if not 1 <= row['initial_observed_area'] <= 96 * 96:
        raise ValueError('Invalid initial area')
    if row['detected']:
        if not (1 <= row['observed_area'] <= 96 * 96 and row['anisotropy'] >= 1):
            raise ValueError('Invalid observed component')
        if not all(0 <= row[name] <= 95 for name in ('pred_x', 'pred_y')):
            raise ValueError('Prediction coordinate outside image')
        if not (0 <= row['score_static'] <= row['score_temporal'] and row['innovation_px'] >= 0):
            raise ValueError('Invalid score/innovation')
        error = math.hypot(row['pred_x'] - row['gt_x'], row['pred_y'] - row['gt_y']) / math.hypot(96, 96)
        if not math.isclose(row['error_norm'], error, rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError('Error disagrees with saved coordinates')
    elif row['observed_area'] != 0 or any(row[name] is not None for name in NULLABLE):
        raise ValueError('Missing prediction must have empty measurements and zero area')
    key = sha256(f"{row['sequence_id']:03d}|{row['frame']}".encode())
    if row['tie_key'] != key:
        raise ValueError('Tie key disagrees with sequence/frame')
    for method in METHODS:
        expected = row['detected']
        if expected and method != 'accept_all':
            threshold = thresholds[method]
            score = 0. if method == 'random' else row['score_' + method]
            expected = threshold['score'] is not None and (score, key) <= (threshold['score'], threshold['tie_key'])
        if row['accepted_' + method] != expected:
            raise ValueError('Acceptance flag disagrees with frozen policy: ' + method)
    return row


def validate_test_grid(rows):
    expected = {(sid, condition, frame) for sid in TEST_IDS for condition in CONDITIONS for frame in range(1, 32)}
    actual = [(r['sequence_id'], r['condition'], r['frame']) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected or any(r['partition'] != 'test' for r in rows):
        raise ValueError('Incomplete, duplicate, or unexpected test grid')


def load_inputs(directory=INPUT):
    """Hash the exact bytes parsed; original provenance remains an input, not edited."""
    names = ('provenance.json', 'config.json', 'summary.json', 'predictions.csv')
    blobs = {name: (directory / name).read_bytes() for name in names}
    hashes = {name: sha256(data) for name, data in blobs.items()}
    provenance = strict_json(blobs['provenance.json'])
    expected = dict(provenance['artifact_sha256'], **{'config.json': provenance['config_sha256']})
    for name in names[1:]:
        if hashes[name] != expected[name]:
            raise ValueError('Input SHA-256 mismatch: ' + name)
    config, summary = (strict_json(blobs[name]) for name in ('config.json', 'summary.json'))
    checks = {'smoke': False, 'image_size': [96, 96], 'frames_per_sequence': 32,
              'conditions': list(CONDITIONS), 'failure_tolerance_image_diagonal': TOLERANCE}
    if any(config.get(key) != value for key, value in checks.items()):
        raise ValueError('Configuration does not describe the full frozen pilot')
    manifest = [{'sequence_id': sid, 'partition': partition(sid)} for sid in range(160)]
    if config['manifest'] != manifest:
        raise ValueError('Unexpected sequence manifest')
    protocol_hash = sha256((ROOT / 'protocols/0001-synthetic-localization.md').read_bytes())
    if config['protocol_sha256'] != protocol_hash or provenance['protocol_sha256'] != protocol_hash:
        raise ValueError('Frozen protocol SHA-256 mismatch')
    if summary.get('smoke') is not False or summary.get('experiment') != '0001-synthetic-localization':
        raise ValueError('Unexpected summary identity')
    thresholds = summary['thresholds']
    for method in METHODS[:-1]:
        threshold = thresholds[method]
        score, key = threshold['score'], threshold['tie_key']
        if score is None:
            if key is not None:
                raise ValueError('Undefined threshold has a tie key')
        elif (isinstance(score, bool) or not isinstance(score, (int, float)) or
              not math.isfinite(score) or score < 0 or not isinstance(key, str) or
              len(key) != 64 or any(c not in '0123456789abcdef' for c in key)):
            raise ValueError('Invalid frozen threshold')
    reader = csv.DictReader(io.StringIO(blobs['predictions.csv'].decode('utf-8'), newline=''))
    if reader.fieldnames != FIELDS:
        raise ValueError('Unexpected CSV header')
    seen, test = set(), []
    for raw in reader:
        row = parse_row(raw, thresholds)
        identity = (row['sequence_id'], row['condition'], row['frame'])
        if identity in seen:
            raise ValueError('Duplicate prediction row')
        seen.add(identity)
        if row['partition'] == 'test':
            test.append(row)
    if len(seen) != 160 * 4 * 31:
        raise ValueError('Incomplete original prediction grid')
    validate_test_grid(test)
    test.sort(key=lambda r: (r['sequence_id'], CONDITIONS.index(r['condition']), r['frame']))
    # The independent count calculation must agree with the saved pooled results.
    for condition in CONDITIONS:
        computed = aggregate([row for row in test if row['condition'] == condition])
        for method in METHODS:
            original = summary['test'][condition]['methods'][method]
            values = {key: computed[key] for key in ('frames', 'detected', 'missing', 'all_frame_failure_rate')}
            values.update(computed['methods'][method])
            for key in ('frames', 'detected', 'missing', 'accepted', 'coverage', 'selected_failure_rate', 'all_frame_failure_rate'):
                old, new = original[key], values[key]
                if not (old == new or (old is not None and new is not None and math.isclose(old, new, abs_tol=1e-14))):
                    raise ValueError('Saved summary disagrees with rows: ' + condition + '/' + method + '/' + key)
    return test, {'input_sha256': hashes, 'protocol_sha256': protocol_hash,
                  'original_source_commit': provenance['git_revision'],
                  'original_protocol_commit': provenance['protocol_commit'],
                  'frozen_thresholds': thresholds}


def aggregate(rows):
    """Pooled denominators include missing rows; conditional risk may be undefined."""
    count = len(rows)
    if not count:
        raise ValueError('Cannot aggregate an empty stratum')
    detected = sum(row['detected'] for row in rows)
    failures = sum(row['detected'] and row['error_norm'] > TOLERANCE for row in rows)
    result = {'frames': count, 'detected': detected, 'missing': count - detected,
              'valid_failure_count': failures, 'all_frame_failure_rate': (count - detected + failures) / count,
              'zero_innovation_valid_count': sum(row['detected'] and row['innovation_px'] == 0 for row in rows),
              'equal_static_temporal_valid_count': sum(row['detected'] and row['score_static'] == row['score_temporal'] for row in rows),
              'methods': {}}
    for method in METHODS:
        accepted = [row for row in rows if row['accepted_' + method]]
        n = len(accepted)
        fail = sum(row['error_norm'] > TOLERANCE for row in accepted)
        result['methods'][method] = {'accepted': n, 'accepted_failure_count': fail,
                                     'coverage': n / count, 'selected_failure_rate': fail / n if n else None}
    static, temporal = (result['methods'][method] for method in METHODS[:2])
    result['temporal_minus_static_coverage'] = temporal['coverage'] - static['coverage']
    risks = (temporal['selected_failure_rate'], static['selected_failure_rate'])
    result['temporal_minus_static_selected_failure_rate'] = None if None in risks else risks[0] - risks[1]
    return result


def sequence_draws(ids, repeats=BOOTSTRAP_REPEATS, seed=BOOTSTRAP_SEED):
    ids = sorted(ids)
    if not ids or len(set(ids)) != len(ids) or repeats < 1:
        raise ValueError('Need unique sequences and positive repeats')
    rng = random.Random(seed)
    return [tuple(rng.choice(ids) for _ in ids) for _ in range(repeats)]


def interval(values):
    defined = sorted(value for value in values if value is not None)
    required = math.ceil(.95 * len(values))
    def quantile(p):
        index = p * (len(defined) - 1)
        lo, hi = math.floor(index), math.ceil(index)
        return defined[lo] + (defined[hi] - defined[lo]) * (index - lo)
    valid = len(defined) >= required and bool(defined)
    return {'low': quantile(.025) if valid else None, 'high': quantile(.975) if valid else None,
            'defined_resamples': len(defined), 'undefined_resamples': len(values) - len(defined),
            'required_defined_resamples': required}


def bootstrap(totals, draws):
    """Resample whole per-sequence counts jointly; never average sequence risks."""
    samples = {method: {'coverage': [], 'selected_failure_rate': []} for method in METHODS}
    differences = {'temporal_minus_static_coverage': [], 'temporal_minus_static_selected_failure_rate': []}
    for draw in draws:
        selected = [totals[sid] for sid in draw]
        n = sum(item['frames'] for item in selected)
        replicate = {}
        for method in METHODS:
            accepted = sum(item['methods'][method]['accepted'] for item in selected)
            failures = sum(item['methods'][method]['accepted_failure_count'] for item in selected)
            replicate[method] = (accepted / n, failures / accepted if accepted else None)
            for key, value in zip(('coverage', 'selected_failure_rate'), replicate[method]):
                samples[method][key].append(value)
        static, temporal = replicate['static'], replicate['temporal']
        differences['temporal_minus_static_coverage'].append(temporal[0] - static[0])
        differences['temporal_minus_static_selected_failure_rate'].append(
            None if static[1] is None or temporal[1] is None else temporal[1] - static[1])
    result = {method: {key: interval(values) for key, values in metrics.items()} for method, metrics in samples.items()}
    result.update({key: interval(values) for key, values in differences.items()})
    return result


def flatten(result, **labels):
    common = {key: value for key, value in result.items() if key != 'methods'}
    return [dict(labels, **common, method=method, **metrics) for method, metrics in result['methods'].items()]


def analyze(rows):
    validate_test_grid(rows)
    draws = sequence_draws(TEST_IDS)
    output = {'analysis': '0001-post-hoc-failures', 'post_hoc': True,
              'new_experiment': False, 'thresholds_refitted': False,
              'claim': 'Descriptive failure analysis of a synthetic diagnostic; no new method or efficacy claim.',
              'failure_tolerance_image_diagonal': TOLERANCE,
              'bootstrap': {'repeats': BOOTSTRAP_REPEATS, 'seed': BOOTSTRAP_SEED, 'generator': 'random.Random.choice',
                            'unit': 'base sequence', 'sequence_count': 40, 'paired_across_methods_conditions_phases': True,
                            'interval': '95% percentile, linear interpolation (type 7)',
                            'minimum_defined_fraction': .95,
                            'draws_sha256': sha256(json.dumps(draws, separators=(',', ':')).encode()),
                            'scope': 'Conditional on frozen calibration and this renderer; unadjusted descriptive intervals, not simultaneous inference.'},
              'mechanism_counts': 'Exact numeric equality on valid saved rows; warm-up zeros are included. Missing rows are not zero innovations.',
              'conditions': {}}
    phase_table, per_sequence, per_frame, trace = [], [], [], []
    for condition in CONDITIONS:
        group = [row for row in rows if row['condition'] == condition]
        definitions = [('all', 1, 31, 'full condition')] + list(PHASES[condition])
        results = {}
        for name, first, last, origin in definitions:
            subset = [row for row in group if first <= row['frame'] <= last]
            result = aggregate(subset)
            totals = {sid: aggregate([row for row in subset if row['sequence_id'] == sid]) for sid in TEST_IDS}
            labels = {'condition': condition, 'phase': name, 'frame_first': first, 'frame_last': last, 'boundary_origin': origin}
            phase_table.extend(flatten(result, **labels))
            for sid, total in totals.items():
                per_sequence.extend(flatten(total, sequence_id=sid, **labels))
            result['bootstrap'] = bootstrap(totals, draws)
            result.update(frame_first=first, frame_last=last, boundary_origin=origin)
            results[name] = result
        output['conditions'][condition] = results
        for frame in range(1, 32):
            phase = next(name for name, first, last, _ in PHASES[condition] if first <= frame <= last)
            frame_rows = [row for row in group if row['frame'] == frame]
            per_frame.extend(flatten(aggregate(frame_rows), condition=condition, frame=frame, phase=phase))
            example = next(row for row in frame_rows if row['sequence_id'] == 120)
            trace.append(dict(example, phase=phase, valid_failure=(example['error_norm'] > TOLERANCE) if example['detected'] else None))
    return output, {'phase_summary.csv': phase_table, 'per_sequence_phase.csv': per_sequence,
                    'per_frame.csv': per_frame, 'sequence_120_trace.csv': trace}


def json_text(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'


def csv_text(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def readable_tables(table):
    lines = ['# Post-hoc failure tables for experiment 0001', '',
             'AI-assisted analysis of saved predictions only. No new experiment, new method, threshold refit, or efficacy claim.', '',
             'Coverage uses every frame, including missing predictions; selected failure rate is undefined when nothing is accepted.',
             'Appearance frames 8–9 and 10–31 are post-hoc memory stages. Other boundaries follow protocol interventions.',
             'The JSON includes descriptive paired sequence-bootstrap intervals; they are unadjusted for multiple strata and conditional on frozen calibration.', '',
             '| Condition / phase | Method | Frames | Missing | Accepted | Accepted failures | Coverage | Selected failure | Zero residual / valid | Equal scores / valid |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in table:
        risk = row['selected_failure_rate']
        cells = (f"{row['condition']} / {row['phase']} ({row['frame_first']}–{row['frame_last']})",
                 row['method'], row['frames'], row['missing'], row['accepted'], row['accepted_failure_count'],
                 f"{row['coverage']:.4f}", 'undefined' if risk is None else f'{risk:.4f}',
                 f"{row['zero_innovation_valid_count']}/{row['detected']}",
                 f"{row['equal_static_temporal_valid_count']}/{row['detected']}")
        lines.append('| ' + ' | '.join(str(value) for value in cells) + ' |')
    return '\n'.join(lines) + '\n'


def git_value(*args):
    result = subprocess.run(['git', '-C', str(ROOT), *args], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def run(output):
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    output = output.resolve()
    if output.exists():
        raise ValueError('Refusing to overwrite an existing output path')
    if INPUT == output or INPUT in output.parents:
        raise ValueError('Output cannot be inside the frozen input directory')
    # Repository-relative destinations keep public provenance free of private paths.
    try:
        relative = output.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError('Output must be a new directory inside the repository') from exc
    rows, inputs = load_inputs()
    result, tables = analyze(rows)
    result['frozen_thresholds'] = inputs['frozen_thresholds']
    contents = {'analysis.json': json_text(result), 'tables.md': readable_tables(tables['phase_summary.csv'])}
    contents.update({name: csv_text(data) for name, data in tables.items()})
    sources = ('scripts/analyze_0001_failures.py', 'tests/test_failure_analysis.py')
    provenance = {key: value for key, value in inputs.items() if key != 'frozen_thresholds'}
    provenance.update(status='completed', analysis='0001-post-hoc-failures', post_hoc=True,
                      command='python scripts/analyze_0001_failures.py --output ' + shlex.quote(relative),
                      input_directory='results/0001-pilot', python=platform.python_version(),
                      platform=platform.platform(), dependencies='Python standard library only',
                      git_revision=git_value('rev-parse', 'HEAD'),
                      analysis_source_dirty=bool(git_value('status', '--porcelain', '--', *sources)),
                      source_sha256={name: sha256((ROOT / name).read_bytes()) for name in sources},
                      started_at_utc=started_utc,
                      artifact_sha256={name: sha256(data.encode()) for name, data in contents.items()})
    output.mkdir(parents=True, exist_ok=False)
    for name, data in contents.items():
        (output / name).write_text(data, encoding='utf-8')
    provenance['elapsed_seconds'] = time.monotonic() - started
    (output / 'provenance.json').write_text(json_text(provenance), encoding='utf-8')
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='New repository-relative output directory; existing paths are refused')
    args = parser.parse_args()
    try:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        provenance = run(output)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, 'Analysis failed: ' + str(exc) + '\n')
    print(json_text({'status': provenance['status'], 'elapsed_seconds': provenance['elapsed_seconds'],
                     'test_rows': 4960, 'new_experiment': False, 'post_hoc': True}).rstrip())


if __name__ == '__main__':
    main()
