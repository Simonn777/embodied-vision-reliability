import copy

import numpy as np
import pytest

from evr.metrics import (FAILURE_TOLERANCE, accepted, calibrate, cluster_resamples,
                         paired_bootstrap, risk_coverage_curve, sequence_totals, summarize)


def row(index, *, error=.01, score=0., detected=True, partition='cal', condition='clean', sequence=1):
    return {'sequence_id': sequence, 'partition': partition, 'condition': condition,
            'detected': detected, 'error_norm': error if detected else None,
            'score_static': score if detected else None, 'score_temporal': score if detected else None,
            'tie_key': f'{index:064x}'}


def test_calibration_ties_use_fixed_key_and_ignore_labels():
    rows = [row(index) for index in range(10)]
    threshold = calibrate(rows, 'static')
    assert threshold['calibration_accepted'] == 8
    assert threshold['realized_calibration_coverage'] == .8 and threshold['target_reachable']
    assert [accepted(value, 'static', threshold) for value in rows] == [True] * 8 + [False] * 2
    altered = copy.deepcopy(rows)
    for value in altered:
        value['error_norm'] = 100.
    assert calibrate(altered[::-1], 'static') == threshold
    for partition, condition in [('test', 'clean'), ('cal', 'occlusion')]:
        with pytest.raises(ValueError, match='clean calibration'):
            calibrate([row(0, partition=partition, condition=condition)], 'static')


def test_unreachable_coverage_retains_finite_threshold_and_missing_denominator():
    rows = [row(0, score=1.), row(1, detected=False), row(2, detected=False)]
    threshold = calibrate(rows, 'static')
    assert not threshold['target_reachable']
    assert threshold['realized_calibration_coverage'] == 1 / 3
    assert not accepted(row(3, score=2., partition='test'), 'static', threshold)
    assert not accepted(rows[1], 'accept_all')
    empty = calibrate([row(0, detected=False)], 'static')
    assert empty['score'] is None and not accepted(row(3), 'static', empty)


def test_known_metrics_count_missing_and_strict_failure_boundary():
    rows = [row(0, error=0.), row(1, error=FAILURE_TOLERANCE),
            row(2, error=.1), row(3, detected=False)]
    result = summarize(rows, 'accept_all')
    assert result['coverage'] == .75 and result['missing_rate'] == .25
    assert result['all_frame_failure_rate'] == .5
    assert result['selected_failure_rate'] == pytest.approx(1 / 3)
    assert result['valid_mean_error_norm'] == pytest.approx(.125 / 3)
    none = summarize(rows, 'static', {'score': None, 'tie_key': None})
    assert none['coverage'] == 0 and none['selected_mean_error_norm'] is None
    assert none['selected_failure_rate'] is None


def test_aurc_reachable_domain_and_oracle_are_correct_on_known_fixture():
    rows = [row(0, error=.1, score=0.), row(1, error=.01, score=1.), row(2, detected=False)]
    curve = risk_coverage_curve(rows, 'static')
    assert curve['coverage'] == [1 / 3, 2 / 3]
    assert curve['risk'] == [1., .5]
    assert curve['aurc_reachable'] == .5 and curve['max_coverage'] == 2 / 3
    assert risk_coverage_curve(rows, 'oracle')['aurc_reachable'] == pytest.approx(1 / 6)
    missing = risk_coverage_curve([row(0, detected=False)], 'static')
    assert missing['aurc_reachable'] is None and missing['max_coverage'] == 0


def test_cluster_resamples_are_whole_ids_with_replacement_and_repeatable():
    first = cluster_resamples([1, 2, 3], repeats=200, seed=7)
    np.testing.assert_array_equal(first, cluster_resamples([3, 1, 2], repeats=200, seed=7))
    assert first.shape == (200, 3) and set(first.ravel()) == {1, 2, 3}
    assert any(len(set(sample)) < 3 for sample in first)
    with pytest.raises(ValueError):
        cluster_resamples([1, 1])


def test_paired_bootstrap_pools_counts_and_preserves_method_condition_pairing():
    # One successful sequence has three frames, one failing sequence has one.
    # Pooled risk is not the average of the two sequence risks.
    rows = []
    for condition in ['clean', 'appearance']:
        rows.extend([row(index, error=.0, partition='test', condition=condition, sequence=1) for index in range(3)])
        rows.append(row(4, error=.1, partition='test', condition=condition, sequence=2))
    threshold = {'score': 0., 'tie_key': 'f' * 64}
    thresholds = {method: threshold for method in ['static', 'temporal', 'random']}
    result = paired_bootstrap(rows, thresholds, repeats=100, seed=4)
    assert result['sequence_count'] == 2 and result['paired_across_conditions_and_methods']
    assert result['intervals']['clean'] == result['intervals']['appearance']
    assert result['intervals']['clean']['temporal_minus_static_selected_failure']['low'] == 0.
    assert result['intervals']['clean']['temporal_minus_static_coverage']['high'] == 0.
    draws = cluster_resamples([1, 2], repeats=100, seed=4)
    expected = [sum(sid == 2 for sid in sample) / sum(3 if sid == 1 else 1 for sid in sample) for sample in draws]
    interval = result['intervals']['clean']['static']['selected_failure_rate']
    assert interval['low'] == np.quantile(expected, .025)
    assert interval['high'] == np.quantile(expected, .975)


def test_bootstrap_reports_undefined_risk_without_fabricating_zero():
    rows = [row(0, partition='test', sequence=1)]
    thresholds = {method: {'score': None, 'tie_key': None} for method in ['static', 'temporal', 'random']}
    result = paired_bootstrap(rows, thresholds, repeats=100)
    interval = result['intervals']['clean']['static']['selected_failure_rate']
    assert interval == {'low': None, 'high': None, 'defined_resamples': 0, 'required_defined_resamples': 95}
    assert result['intervals']['clean']['static']['coverage']['low'] == 0


def test_exported_sequence_totals_reconstruct_pooled_metrics():
    rows = [row(0, error=.01, partition='test', sequence=1),
            row(1, error=.1, partition='test', sequence=1),
            row(2, detected=False, partition='test', sequence=2)]
    thresholds = {method: {'score': 0., 'tie_key': 'f' * 64} for method in ['static', 'temporal', 'random']}
    totals = sequence_totals(rows, thresholds)
    assert len(totals) == 8
    for method in ['static', 'temporal', 'random', 'accept_all']:
        selected = [value for value in totals if value['method'] == method]
        result = summarize(rows, method, thresholds.get(method))
        total = sum(value['frames'] for value in selected)
        count = sum(value['accepted'] for value in selected)
        failures = sum(value['accepted_failure_count'] for value in selected)
        error_sum = sum(value['accepted_error_sum_norm'] for value in selected)
        assert count / total == result['coverage']
        assert failures / count == result['selected_failure_rate']
        assert error_sum / count == result['selected_mean_error_norm']
        missing = next(value for value in selected if value['sequence_id'] == 2)
        assert missing['missing'] == 1 and missing['accepted'] == 0
