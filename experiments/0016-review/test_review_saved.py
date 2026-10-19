"""Independent artificial fixtures; no experiment imports or scientific files."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('independent0016', HERE / 'review_saved.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
IDENTITY = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root):
    """Closed-form errors: one all-failed pair, one score/GT disagreement,62 perfect."""
    paths = review.SOURCE_PATHS
    for relative in paths:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('Artificial source only\n')
    library = {'old': {'version': '2.0.5', 'binary_sha256': 'a' * 64},
               'current': {'version': '3.0.0', 'binary_sha256': 'b' * 64}}
    write(root / 'configs/0016-libraries.json', library)
    source = {p: review.digest(root / p) for p in paths}
    old = {'selected_pair_keys': ['0015-old-%02d' % i for i in range(16)],
           'dataset_sha256': 'd' * 64, 'dataset_bytes': 1234}
    oh = write(root / 'configs/0012-development-manifest.json', old)
    keys = ['0015-artificial-%02d' % i for i in range(64)]
    keys.sort(key=lambda k: hashlib.sha256((review.SALT + k).encode()).hexdigest())
    manifest = {**old, 'selected_pair_keys': keys, 'selection_salt': review.SALT,
                'excluded_manifest_sha256': oh, 'held_out_scenes': ['0022'],
                'development_scene_pairs': 600, 'coordinate_convention': 'raw_pixels'}
    mp = root / 'configs/0016-development-manifest.json'
    mh = write(mp, manifest)
    sp = root / 'results/0016-search/predictions/seal.json'
    ep = root / 'results/0016-search/evaluation.json'
    sealrows, evalrows = [], []
    labels = ['seed0'] + ['pixel_msac_budget_' + str(t) for t in review.BUDGETS] + [
        'pixel_msac_full4', 'oracle_best4_nondeployable']
    error_template = {'rotation_degrees': 0., 'translation_degrees': 0., 'max_degrees': 0.,
                      'translation_signfolded_degrees': 0., 'max_signfolded_degrees': 0.}
    for backend in review.BACKENDS:
        for i, key in enumerate(keys):
            candidates = []
            for seed in range(4):
                call, score = [.07, .13, .23, .48][seed], .02
                t = ([1., 0., 0.] if i != 1 else
                     ([1., 0., 0.], [0., 1., 0.], [1., 0., 0.], [-1., 0., 0.])[seed])
                c = {'seed': seed, 'status': 'completed', 'options': review.expected_options(backend, seed),
                     'estimate': {'R': IDENTITY, 't': t}, 'call_seconds': call, 'score_seconds': score,
                     'charged_end_seconds': [.1, .25, .5, 1.][seed],
                     'final_scores': {'pixel_msac': [4., 3., 3., 2.][seed], 'normalized_msac': 2.,
                                      'tangent_msac': 2., 'pixel_inliers': 9999, 'normalized_inliers': 9999,
                                      'tangent_inliers': 9999, 'cheiral_count': 10000},
                     'native_inliers_packbits_big_hex': 'ff' * 1250}
                if i == 0:
                    c.update(status='estimator_error', call_seconds=call + score, score_seconds=0., error='Artificial failure')
                    for field in ('estimate', 'final_scores', 'native_inliers_packbits_big_hex'):
                        del c[field]
                candidates.append(c)
            record = {'status': 'completed', 'backend': backend, 'pair_index': i, 'key': key,
                      'manifest_sha256': mh, 'source_sha256': source,
                      'library': library['old' if backend == 'old_norm' else 'current'],
                      'input_seconds': .01, 'charged_total_seconds': 1., 'candidates': candidates,
                      'calibration': {'artificial': True}}
            name = backend + '-%02d.json' % i
            rh = write(sp.parent / name, record)
            process = {'exit_code': 0, 'termination': None, 'watchdog_error': None,
                       'wall_seconds': 1.2, 'sampled_peak_rss_kib': 1000}
            ph = write(sp.parent / (name + '.process.json'), process)
            sealrows.append({'filename': name, 'backend': backend, 'pair_index': i, 'key': key,
                             'sha256': rh, 'process_sha256': ph})
            ecs = copy.deepcopy(candidates)
            for seed, c in enumerate(ecs):
                e = copy.deepcopy(error_template)
                if i == 1 and seed in (1, 3):
                    d = 90. if seed == 1 else 180.
                    e.update(translation_degrees=d, max_degrees=d,
                             translation_signfolded_degrees=min(d, 180. - d),
                             max_signfolded_degrees=min(d, 180. - d))
                c['error'] = None if i == 0 else e
            selections = {}
            for label in labels:
                selected = 0 if label in ('seed0', 'pixel_msac_budget_0.1', 'oracle_best4_nondeployable') else (
                    1 if label in ('pixel_msac_budget_0.25', 'pixel_msac_budget_0.5') else 3)
                if i == 0:
                    selections[label] = {'status': 'unavailable', 'seed': None, 'error': None, 'reason': 'all failed',
                                         'penalized_max_degrees': 180., 'penalized_max_signfolded_degrees': 180.}
                else:
                    c = ecs[selected]
                    selections[label] = {'status': 'completed', 'seed': selected, 'error': c['error'], 'reason': None,
                        'pixel_msac': c['final_scores']['pixel_msac'], 'charged_end_seconds': c['charged_end_seconds'],
                        'penalized_max_degrees': c['error']['max_degrees'],
                        'penalized_max_signfolded_degrees': c['error']['max_signfolded_degrees']}
            pairwise = []
            if i != 0:
                ds = [90., 0., 180., 90., 90., 180.] if i == 1 else [0.] * 6
                for (a, b), d in zip([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], ds):
                    pairwise.append({'seed_a': a, 'seed_b': b, 'rotation_degrees': 0.,
                                     'translation_directed_degrees': d})
            gt = {'R_original': [[1.001, 0., 0.], [0., .999, 0.], [0., 0., 1.]],
                  't_original': [[2.], [0.], [0.]], 'R_projected': IDENTITY, 't_unit': [1., 0., 0.],
                  'translation_original_norm': 2., 'singular_values': [1.001, 1., .999],
                  'projection_frobenius_change': math.sqrt(2) * .001, 'projected_determinant': 1.}
            spread = {'min_directed_max_degrees': None if i == 0 else 0.,
                      'max_directed_max_degrees': None if i == 0 else (180. if i == 1 else 0.),
                      'range_directed_max_degrees': None if i == 0 else (180. if i == 1 else 0.)}
            evalrows.append({'backend': backend, 'pair_index': i, 'key': key, 'calibration': record['calibration'],
                'input_seconds': .01, 'charged_total_seconds': 1., 'candidates': ecs, 'ground_truth': gt,
                'selections': selections, 'candidate_error_spread': spread,
                'prediction_only_dispersion': {'successful_candidate_count': 0 if i == 0 else 4, 'pairwise': pairwise,
                    'max_rotation_degrees': None if i == 0 else 0.,
                    'max_translation_directed_degrees': None if i == 0 else (180. if i == 1 else 0.)}})
    sh = write(sp, {'status': 'completed', 'manifest_sha256': mh, 'source_sha256': source,
                    'rows': sealrows, 'wall_seconds': 235.})
    summaries = {}
    for backend in review.BACKENDS:
        subset = [r for r in evalrows if r['backend'] == backend]
        summaries[backend] = {}
        for label in labels:
            seed = subset[1]['selections'][label]['seed']
            summary = {'pair_count': 64, 'status_counts': {'completed': 63, 'unavailable': 1},
                       'selected_seed_counts': {str(seed): 63}}
            for conv, field in (('directed_primary', 'penalized_max_degrees'),
                                ('signfolded_secondary', 'penalized_max_signfolded_degrees')):
                values = [r['selections'][label][field] for r in subset]
                perfect = sum(v == 0. for v in values)
                summary[conv] = {'errors_degrees': values, 'auc_fraction': {str(t): perfect / 64 for t in (5, 10, 20)},
                                 'count_strictly_below': {str(t): perfect for t in (5, 10, 20)}}
            summaries[backend][label] = summary
    evaluation = {'status': 'completed', 'manifest_sha256': mh, 'seal_sha256': sh, 'source_sha256': source,
        'dataset_sha256': old['dataset_sha256'], 'dataset_bytes': old['dataset_bytes'],
        'pair_count': 64, 'backend_count': 3, 'budgets_seconds': list(review.BUDGETS),
        'rows': evalrows, 'summaries': summaries,
        'oracle_warning': 'oracle_best4_nondeployable is a secondary diagnostic only'}
    write(ep, evaluation)
    return mp, sp, ep


class Tests(unittest.TestCase):
    def test_scalar_angles_and_svd_projection(self):
        a = math.radians(30.)
        R = [[math.cos(a), -math.sin(a), 0.], [math.sin(a), math.cos(a), 0.], [0., 0., 1.]]
        e = review.errors({'R': R, 't': [-4., 0., 0.]}, (IDENTITY, [1., 0., 0.]))
        self.assertAlmostEqual(e['rotation_degrees'], 30., places=12)
        self.assertEqual(e['translation_degrees'], 180.)
        self.assertAlmostEqual(e['max_signfolded_degrees'], 30., places=12)
        with self.assertRaises(ValueError):
            review.proper_rotation([[1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
        with self.assertRaises(ValueError):
            review.unit([0., 0., 0.])

    def test_dms_auc_exact_threshold_ties(self):
        self.assertEqual(review.dms_auc([5., 5.], 5.), 0.)
        # Areas: (0+1/4)/2*1 + (1/4+1/2)/2*2 + (1/2)*2 = 1.875.
        self.assertAlmostEqual(review.dms_auc([1., 3., 5., 7.], 5.), .375)
        self.assertEqual(review.dms_auc([0., 0., 180., 180.], 5.), .5)

    def test_full192_768_artificial_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = fixture(root)
            result = review.audit(root, *paths)
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['counts']['candidates'], 768)
            self.assertEqual(result['counts']['candidate_status_by_backend']['current_ts'],
                             {'completed': 252, 'estimator_error': 4})
            self.assertAlmostEqual(result['costs']['old_norm']['charged_total_seconds']['sum'], 64.)
            self.assertAlmostEqual(result['costs']['old_norm']['offline_completed_prefix_seconds']['0.5']['sum_completed_prefix_seconds'], 32.)
            self.assertFalse(result['ordinal_candidate_bank_necessary_gate']['at_least_one_current_oracle_exceeds_both_current_selectors_by_2pp'])

    def test_saved_tampering_is_detected(self):
        for kind in ('angle', 'auc', 'selector', 'source', 'process', 'reference', 'copy', 'cost', 'oracle_label'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                mp, sp, ep = fixture(root)
                ev, _ = review.load(ep)
                if kind == 'angle': ev['rows'][1]['candidates'][0]['error']['max_degrees'] = 1.
                elif kind == 'auc': ev['summaries']['old_norm']['seed0']['directed_primary']['auc_fraction']['5'] = .1
                elif kind == 'selector': ev['rows'][1]['selections']['pixel_msac_full4']['seed'] = 0
                elif kind == 'source': (root / 'experiments/0016-search-headroom/common.py').write_text('Tampered')
                elif kind == 'process':
                    p = sp.parent / 'old_norm-00.json.process.json'
                    p.write_text(p.read_text() + ' ')
                elif kind == 'reference': ev['rows'][1]['ground_truth']['R_projected'][0][0] = .9
                elif kind == 'copy': ev['rows'][1]['candidates'][0]['final_scores']['pixel_msac'] = 1.
                elif kind == 'cost': ev['rows'][1]['charged_total_seconds'] = .9
                else: ev['oracle_warning'] = 'best method'
                write(ep, ev)
                with self.assertRaises(ValueError):
                    review.audit(root, mp, sp, ep)

    def test_score_tie_and_exact_budget_boundary(self):
        candidates = [{'seed': s, 'status': 'completed', 'charged_end_seconds': [.1, .25, .5, 1.][s],
                       'final_scores': {'pixel_msac': [4., 3., 3., 2.][s]}} for s in range(4)]
        self.assertIsNone(review.choice(candidates, math.nextafter(.1, 0.)))
        self.assertEqual(review.choice(candidates, .1)['seed'], 0)
        self.assertEqual(review.choice(candidates, .5)['seed'], 1)
        self.assertEqual(review.choice(candidates)['seed'], 3)

    def test_oracle_roundoff_identity_is_disclosed_but_wrong_argmin_is_rejected(self):
        candidates = [{'seed': s, 'status': 'completed'} for s in (0, 1)]
        own = {0: {'max_degrees': 1. + 1e-12}, 1: {'max_degrees': 1.}}
        saved = [{**c, 'error': {'max_degrees': 1.}} for c in candidates]
        recorded, scalar, discrepancy = review.check_oracle(candidates, own, saved, 0)
        self.assertEqual(recorded['seed'], 0)
        self.assertEqual(scalar['seed'], 1)
        self.assertGreater(discrepancy['independent_error_regret_degrees'], 0.)
        with self.assertRaises(ValueError):
            review.check_oracle(candidates, own, saved, 1)
        own[0]['max_degrees'] = 1. + 2 * review.ANGLE_ATOL
        with self.assertRaises(ValueError):
            review.check_oracle(candidates, own, saved, 0)

    def test_duplicate_and_nonfinite_json_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.json'
            for value in ('{"a":1,"a":2}', '{"a":NaN}'):
                path.write_text(value)
                with self.assertRaises(ValueError):
                    review.load(path)


if __name__ == '__main__':
    unittest.main()
