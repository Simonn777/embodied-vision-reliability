import inspect
import math

import numpy as np
import pytest

from evr.scenes import CONDITIONS, make_scene, render_frame, split_manifest, trajectory
from evr.tracker import Detection, FixedTracker, detect, largest_component, score_detection
from evr.experiment import evaluate_sequence


def test_split_manifest_groups_whole_sequences_and_smoke_uses_different_ids():
    manifest = split_manifest()
    groups = {partition: {r['sequence_id'] for r in manifest if r['partition'] == partition}
              for partition in ('fit', 'dev', 'cal', 'test')}
    assert [len(group) for group in groups.values()] == [80, 20, 20, 40]
    assert sum(map(len, groups.values())) == len(set.union(*groups.values())) == 160
    assert set.union(*groups.values()).isdisjoint({row['sequence_id'] for row in split_manifest(True)})
    paired = [(row['sequence_id'], row['partition'], condition) for row in manifest for condition in CONDITIONS]
    for sequence in range(160):
        assert len({partition for sid, partition, _ in paired if sid == sequence}) == 1


def test_renderer_reproducibility_condition_pairing_and_real_pixel_effects():
    scene = make_scene(10001)
    clean = trajectory(scene, 'clean')
    for condition in CONDITIONS:
        points = trajectory(scene, condition)
        np.testing.assert_array_equal(points[:16], clean[:16])
        np.testing.assert_array_equal(render_frame(scene, points[0], 0, condition), render_frame(scene, clean[0], 0, 'clean'))
    frame = render_frame(scene, clean[8], 8, 'clean')
    np.testing.assert_array_equal(frame, render_frame(scene, clean[8], 8, 'clean'))
    assert frame.dtype == np.uint8 and frame.shape == (96, 96, 3)
    assert not np.array_equal(frame, render_frame(scene, clean[8], 8, 'occlusion'))
    assert not np.array_equal(trajectory(scene, 'abrupt_motion')[16:], clean[16:])


def test_components_use_four_connectivity_and_deterministic_row_major_ties():
    mask = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]], dtype=bool)
    component = largest_component(mask)
    assert {tuple(value) for value in component} == {(0, 0), (0, 1)}
    assert largest_component(np.zeros((3, 3))).shape == (0, 2)
    assert largest_component(np.eye(3, dtype=bool)).tolist() == [[0, 0]]


def test_single_pixel_covariance_and_missing_detection_are_explicit():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[3, 4] = [0, 255, 0]
    detection = detect(image, [0., 1., 0.])
    assert detection == Detection(4., 3., 1, 1.)
    assert detect(np.zeros_like(image), [0., 1., 0.]) is None
    scores = score_detection(None, detection, [], 1)
    assert math.isinf(scores['static']) and math.isinf(scores['temporal']) and scores['innovation'] is None


def track_prefix(images, initial):
    tracker = FixedTracker(images[0], initial)
    return [tracker.step(image, frame) for frame, image in enumerate(images[1:], 1)]


def test_future_images_do_not_change_prefix_predictions_or_scores():
    scene = make_scene(10002)
    points = trajectory(scene, 'clean', 6)
    images = [render_frame(scene, point, frame, 'clean') for frame, point in enumerate(points)]
    altered = [image.copy() for image in images]
    altered[4:] = [np.zeros_like(image) for image in altered[4:]]
    original = track_prefix(images, scene.initial)
    changed = track_prefix(altered, scene.initial)
    assert original[:3] == changed[:3] == track_prefix(images[:4], scene.initial)
    assert original[3:] != changed[3:]


def test_truth_has_no_path_to_deployable_scores_and_estimator_reads_pixels():
    scene = make_scene(10003)
    points = trajectory(scene, 'clean', 3)
    frames = [render_frame(scene, point, frame, 'clean') for frame, point in enumerate(points)]
    observed = track_prefix(frames, scene.initial)
    # Evaluating the identical predictions under two truths changes errors only.
    true_error = np.linalg.norm(np.array([observed[0][0].x, observed[0][0].y]) - points[1])
    fake_error = np.linalg.norm(np.array([observed[0][0].x, observed[0][0].y]) - [0., 0.])
    assert fake_error > true_error
    assert track_prefix(frames, scene.initial) == observed
    assert list(inspect.signature(FixedTracker.step).parameters) == ['self', 'image', 'frame']
    assert list(inspect.signature(score_detection).parameters) == ['detection', 'initial', 'history', 'frame']
    assert FixedTracker(frames[0], scene.initial).step(np.zeros_like(frames[1]), 1)[0] is None


def test_temporal_history_uses_elapsed_time_and_raw_rejected_predictions():
    reference = Detection(0., 0., 100, 1.)
    history = [(0, np.array([0., 0.])), (2, np.array([2., 0.]))]
    scores = score_detection(Detection(5., 0., 100, 1.), reference, history, 5)
    assert scores == {'static': 0., 'temporal': 0., 'innovation': 0.}
    assert score_detection(Detection(50., 0., 100, 1.), reference, history[:1], 1)['innovation'] == 0.
    with pytest.raises(ValueError, match='past'):
        score_detection(reference, reference, [(2, np.array([0., 0.]))], 2)
    scene = make_scene(10004)
    points = trajectory(scene, 'clean', 4)
    frames = [render_frame(scene, point, frame, 'clean') for frame, point in enumerate(points)]
    tracker = FixedTracker(frames[0], scene.initial)
    first, _ = tracker.step(frames[1], 1)
    # No acceptance decision is passed to tracker; even a rejected valid output
    # remains one of the raw observations used by the next score.
    assert tracker.history[-1][0] == 1
    np.testing.assert_array_equal(tracker.history[-1][1], [first.x, first.y])
    tracker.step(np.zeros_like(frames[2]), 2)
    assert tracker.history[-1][0] == 1
    tracker.step(frames[3], 3)
    assert [frame for frame, _ in tracker.history] == [1, 3]


def test_first_frame_excluded_and_full_occlusion_remains_missing_rows():
    rows = evaluate_sequence(10005, 'dev', 'occlusion')
    assert [row['frame'] for row in rows] == list(range(1, 32))
    missing = [row for row in rows if 14 <= row['frame'] <= 17]
    assert len(missing) == 4
    assert all(not row['detected'] and row['error_norm'] is None for row in missing)
    assert all(row['score_static'] is None and row['score_temporal'] is None for row in missing)


def test_validation_rejects_bad_initialization_and_repeated_timestamps():
    with pytest.raises(ValueError):
        FixedTracker(np.zeros((8, 8)), [4., 4.])
    with pytest.raises(ValueError):
        FixedTracker(np.zeros((8, 8, 3), dtype=np.uint8), [0., 0.])
    scene = make_scene(10006)
    image = render_frame(scene, scene.initial, 0, 'clean')
    tracker = FixedTracker(image, scene.initial)
    with pytest.raises(ValueError):
        tracker.step(image, 0)
    with pytest.raises(ValueError):
        tracker.step(image, True)
