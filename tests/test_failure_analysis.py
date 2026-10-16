"""Independent fixtures for analysis of frozen outcomes, not new experiments."""
import copy
import csv
import importlib.util
import json
import math
from pathlib import Path
import random

import pytest

SPEC = importlib.util.spec_from_file_location('failure_analysis', Path(__file__).resolve().parents[1] / 'scripts/analyze_0001_failures.py')
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


@pytest.fixture(scope='module')
def frozen():
    return analysis.load_inputs()


def record(*, error=.01, detected=True, accept=True):
    return {'detected': detected, 'error_norm': error if detected else None,
            'innovation_px': 0. if detected else None,
            'score_static': 0. if detected else None,
            'score_temporal': 0. if detected else None,
            **{'accepted_' + method: detected and accept for method in analysis.METHODS}}


def test_missing_denominator_strict_failure_boundary_and_all_rejected():
    rows = [record(error=0.), record(error=.025), record(error=.1), record(detected=False)]
    result = analysis.aggregate(rows)
    assert result['frames'] == 4 and result['missing'] == 1
    assert result['valid_failure_count'] == 1 and result['all_frame_failure_rate'] == .5
    assert result['zero_innovation_valid_count'] == result['equal_static_temporal_valid_count'] == 3
    assert result['methods']['static'] == {'accepted': 3, 'accepted_failure_count': 1,
                                          'coverage': .75, 'selected_failure_rate': 1 / 3}
    rejected = analysis.aggregate([record(accept=False), record(detected=False)])
    assert rejected['methods']['temporal']['coverage'] == 0
    assert rejected['methods']['temporal']['selected_failure_rate'] is None
    assert rejected['temporal_minus_static_selected_failure_rate'] is None


def test_cluster_draws_are_reproducible_and_keep_whole_sequences():
    draws = analysis.sequence_draws([2, 1], repeats=100, seed=9)
    rng = random.Random(9)
    oracle = [tuple(rng.choice([1, 2]) for _ in range(2)) for _ in range(100)]
    assert draws == oracle == analysis.sequence_draws([1, 2], repeats=100, seed=9)
    assert any(draw == (2, 2) for draw in draws)
    with pytest.raises(ValueError):
        analysis.sequence_draws([1, 1])


def test_bootstrap_pools_counts_and_pairs_methods_using_known_draws():
    # Three successful frames in group 1, one failing frame in group 2.
    # The original pooled failure rate is 1/4, not a mean of group rates (1/2).
    first = [record(error=0.) for _ in range(3)]
    second = [record(error=.1)]
    # Temporal rejects the one failing prediction. Both methods share draws.
    second[0]['accepted_temporal'] = False
    totals = {1: analysis.aggregate(first), 2: analysis.aggregate(second)}
    draws = [(1, 1), (1, 2), (2, 1), (2, 2)] * 250
    result = analysis.bootstrap(totals, draws)
    assert result['static']['selected_failure_rate']['low'] == 0.
    assert result['static']['selected_failure_rate']['high'] == 1.
    assert result['temporal']['selected_failure_rate']['defined_resamples'] == 750
    assert result['temporal']['selected_failure_rate']['low'] is None
    assert result['temporal_minus_static_selected_failure_rate']['undefined_resamples'] == 250
    assert result['temporal_minus_static_coverage']['low'] == -1.
    assert result['temporal_minus_static_coverage']['high'] == 0.
    pooled = analysis.bootstrap(totals, [(1, 2)] * 1000)
    assert pooled['static']['selected_failure_rate']['low'] == .25
    assert pooled['temporal_minus_static_selected_failure_rate']['low'] == -.25
    assert pooled['temporal_minus_static_coverage']['low'] == -.25
    identical = analysis.bootstrap({1: analysis.aggregate(first), 2: analysis.aggregate([record(error=.1)])}, draws)
    assert identical['temporal_minus_static_selected_failure_rate']['high'] == 0.
    assert identical['temporal_minus_static_coverage']['low'] == 0.


def test_percentile_interpolation_and_undefined_cutoff():
    value = analysis.interval([0., 10.])
    assert value['low'] == .25 and value['high'] == 9.75
    assert analysis.interval([0.] * 949 + [None] * 51)['high'] is None
    assert analysis.interval([0.] * 950 + [None] * 50)['high'] == 0.
    assert analysis.interval([None] * 1000)['defined_resamples'] == 0


def test_frozen_inputs_reconcile_and_grid_is_exact(frozen):
    rows, metadata = frozen
    assert len(rows) == 4960
    assert metadata['original_source_commit'].startswith('bc9921f')
    assert len(metadata['input_sha256']) == 4
    analysis.validate_test_grid(rows)
    with pytest.raises(ValueError, match='grid'):
        analysis.validate_test_grid(rows[:-1])
    with pytest.raises(ValueError, match='grid'):
        analysis.validate_test_grid(rows[:-1] + [rows[0]])
    wrong = copy.deepcopy(rows)
    wrong[0]['frame'] = 0
    with pytest.raises(ValueError, match='grid'):
        analysis.validate_test_grid(wrong)


@pytest.fixture(scope='module')
def raw_row():
    with (analysis.INPUT / 'predictions.csv').open(newline='') as stream:
        return next(csv.DictReader(stream))


@pytest.mark.parametrize('key,value', [('detected', '1'), ('score_static', 'nan'),
                                      ('score_temporal', 'inf'), ('frame', '0'),
                                      ('observed_area', '-1'), ('partition', 'test'),
                                      ('pred_x', '96'), ('error_norm', '-0.1'),
                                      ('accepted_temporal', 'False'), ('tie_key', '0' * 64)])
def test_malformed_measurements_and_frozen_flags_fail(raw_row, frozen, key, value):
    row = dict(raw_row, **{key: value})
    with pytest.raises(ValueError):
        analysis.parse_row(row, frozen[1]['frozen_thresholds'])


def test_missing_cannot_be_accepted_or_encoded_as_zero(raw_row, frozen):
    row = dict(raw_row, detected='False', observed_area='0')
    row.update({key: '' for key in analysis.NULLABLE})
    row.update({'accepted_' + method: 'False' for method in analysis.METHODS})
    parsed = analysis.parse_row(row, frozen[1]['frozen_thresholds'])
    assert parsed['error_norm'] is None
    row['error_norm'] = '0'
    with pytest.raises(ValueError, match='Missing prediction'):
        analysis.parse_row(row, frozen[1]['frozen_thresholds'])
    row['error_norm'] = ''
    row['accepted_static'] = 'True'
    with pytest.raises(ValueError, match='Acceptance flag'):
        analysis.parse_row(row, frozen[1]['frozen_thresholds'])


@pytest.mark.parametrize('filename', ['config.json', 'summary.json', 'predictions.csv'])
def test_hash_mismatch_is_rejected_before_parsing(tmp_path, filename):
    # Other files need not parse: the corrupted byte snapshot must fail first.
    for name in ('provenance.json', 'config.json', 'summary.json', 'predictions.csv'):
        data = (analysis.INPUT / name).read_bytes()
        (tmp_path / name).write_bytes(data + b' ' if name == filename else data)
    with pytest.raises(ValueError, match='SHA-256 mismatch: ' + filename):
        analysis.load_inputs(tmp_path)


def test_duplicate_json_keys_and_nonfinite_values_fail():
    for data in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
        with pytest.raises(ValueError):
            analysis.strict_json(data)


def test_all_phases_partition_frames_and_memory_stage_is_posthoc():
    for definitions in analysis.PHASES.values():
        assert [frame for _, first, last, _ in definitions for frame in range(first, last + 1)] == list(range(1, 32))
    assert all('post-hoc' in item[3] for item in analysis.PHASES['appearance'][1:])


def test_saved_failure_mechanism_and_exports(frozen):
    result, tables = analysis.analyze(frozen[0])
    assert result['post_hoc'] and not result['new_experiment'] and not result['thresholds_refitted']
    stable = result['conditions']['appearance']['stable_wrong_identity']
    assert stable['frames'] == stable['zero_innovation_valid_count'] == stable['equal_static_temporal_valid_count'] == 880
    assert stable['methods']['temporal']['accepted_failure_count'] == stable['methods']['temporal']['accepted'] == 880
    assert stable['methods']['static']['accepted'] == 682
    transient = result['conditions']['appearance']['switch_transient']
    assert transient['methods']['temporal']['accepted'] == 0
    assert transient['methods']['temporal']['selected_failure_rate'] is None
    assert transient['bootstrap']['temporal']['selected_failure_rate']['defined_resamples'] == 0
    turn = result['conditions']['abrupt_motion']['turn']
    assert turn['valid_failure_count'] == 0 and turn['methods']['static']['accepted'] == 35
    assert turn['methods']['temporal']['accepted'] == 0
    assert len(tables['sequence_120_trace.csv']) == 124
    assert len(tables['per_frame.csv']) == 4 * 31 * 4
    assert len(tables['per_sequence_phase.csv']) == 40 * (12 + 4) * 4
    assert 'undefined' in analysis.readable_tables(tables['phase_summary.csv'])
    # Independently sum sequence exports back to each pooled phase and method.
    for phase in tables['phase_summary.csv']:
        group = [row for row in tables['per_sequence_phase.csv']
                 if all(row[key] == phase[key] for key in ('condition', 'phase', 'method'))]
        for key in ('frames', 'missing', 'accepted', 'accepted_failure_count', 'zero_innovation_valid_count'):
            assert sum(row[key] for row in group) == phase[key]


def test_two_constant_wrong_predictions_erase_motion_residual():
    # This is a unit fixture, not an experiment rerun or a change to the tracker.
    from evr.tracker import Detection, score_detection
    initial = Detection(20., 20., 100, 1.)
    wrong = Detection(70., 70., 100, 1.)
    transient = score_detection(wrong, initial, [(7, (20., 20.)), (8, (70., 70.))], 9)
    stable = score_detection(wrong, initial, [(8, (70., 70.)), (9, (70., 70.))], 10)
    assert transient['innovation'] > 0 and transient['temporal'] > transient['static']
    assert stable['innovation'] == 0 and stable['temporal'] == stable['static'] == 0
    assert math.hypot(wrong.x - 20., wrong.y - 20.) / math.hypot(96, 96) > analysis.TOLERANCE


def test_output_refuses_overwrites_and_frozen_directory(tmp_path):
    with pytest.raises(ValueError, match='overwrite'):
        analysis.run(tmp_path)
    with pytest.raises(ValueError, match='frozen input'):
        analysis.run(analysis.INPUT / 'must-not-be-created')
    with pytest.raises(ValueError, match='inside the repository'):
        analysis.run(tmp_path / 'outside-repository')
