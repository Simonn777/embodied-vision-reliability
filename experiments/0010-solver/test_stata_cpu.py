"""Artificial fixtures only; no dataset, weights, or classification-quality trial."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import stata_cpu as core

torch.set_num_threads(4)
torch.set_num_interop_threads(1)


def fixture():
    x = torch.tensor([[1., .2, -.3, .4, .1], [-.2, 1., .3, -.1, .6],
                      [.1, -.3, 1., .7, -.2], [.8, .7, -.1, .2, -.4],
                      [-.4, .2, .8, -.5, .3], [.3, -.8, .2, .4, .9],
                      [.6, -.2, -.5, .8, .4]], dtype=torch.float32)
    x = x / x.norm(dim=1, keepdim=True)
    anchors = torch.stack([x[0], x[1], x[2]], dim=1)[None]
    return x, anchors, tuple('abcdefg')


def snapshot(state):
    return {key: value.clone() for key, value in state.items()
            if isinstance(value, torch.Tensor)}


class ParityFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.a, cls.ids = fixture()
        cls.state = core.fit(cls.x, cls.a, ids=cls.ids)

    def assertBoundary(self, code, function, *args, **kwargs):
        with self.assertRaises(core.BoundaryError) as context:
            function(*args, **kwargs)
        self.assertEqual(context.exception.status, 'boundary')
        self.assertEqual(context.exception.code, code)

    def test_hash_and_exact_four_substitutions(self):
        text = core.UPSTREAM_PATH.read_text()
        module = core.load_upstream()
        self.assertEqual(text.count('.cuda()'), 4)
        expected = text.replace('.cuda()', ".to('cpu')")
        self.assertEqual(module.device_source_sha256,
                         hashlib.sha256(expected.encode()).hexdigest())

    def test_source_corruption_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'not_upstream.py'
            path.write_text('raise AssertionError("must not execute")\n')
            self.assertBoundary('source_hash', core.load_upstream, path)

    def test_exact_native_reference_and_schedule(self):
        self.assertTrue(self.state['reference_bitwise_equal'])
        self.assertEqual(self.state['blocks'], 11)
        self.assertEqual(self.state['inner_updates'], 55)
        self.assertEqual(self.state['moment_updates'], 10)
        self.assertEqual(self.state['covariance_updates'], 10)
        self.assertEqual(self.state['native_solver_executions'], 2)
        module = core.load_upstream()
        expected = module.update_z(self.state['likelihoods'], self.state['y_hat'],
                                   self.state['z_before_final'], self.state['w'],
                                   1, 1, 3, self.state['covariance'], 1)
        self.assertTrue(torch.equal(expected, self.state['output']))

    def test_fixed_anchor_controls_share_reference_alpha(self):
        for alpha in (1, 4):
            with self.subTest(alpha=alpha):
                state = core.fit(self.x, self.a, ids=self.ids, alpha=alpha)
                reference = core.load_upstream()
                with torch.no_grad():
                    expected_y, expected = reference.StatA_solver(
                        self.x, torch.zeros(len(self.x)), self.a, alpha=alpha)
                self.assertEqual(state['alpha'], alpha)
                self.assertTrue(state['reference_bitwise_equal'])
                self.assertTrue(torch.equal(state['output'], expected))
                self.assertTrue(torch.equal(state['y_hat'], expected_y))
                for query_id in self.ids:
                    core.native_readout(state, query_id)

    def test_arbitrary_anchor_weights_rejected_before_execution(self):
        for alpha in (True, False, 0, 2, 3, 5, -1, 1.00001,
                      float('nan'), float('inf'), '1', None, torch.tensor(1)):
            with self.subTest(alpha=repr(alpha)):
                with patch.object(core, 'load_upstream', side_effect=AssertionError('must not execute')):
                    self.assertBoundary('alpha', core.fit, self.x, self.a,
                                        ids=self.ids, alpha=alpha)

    def test_present_readout_all_rows_argmax_and_probabilities(self):
        for row, query_id in enumerate(self.ids):
            output = core.native_readout(self.state, query_id)
            torch.testing.assert_close(output, self.state['output'][row],
                                       atol=2e-6, rtol=1e-5)
            self.assertEqual(output.argmax(), self.state['output'][row].argmax())

    def test_non_saturated_probabilities_and_signed_graph(self):
        anchors = torch.tensor([[1., .02, -.01, .03, .01],
                                [1., -.015, .01, -.02, .02]])
        anchors = (anchors / anchors.norm(dim=1, keepdim=True)).T[None]
        state = core.fit(self.x, anchors, ids=self.ids)
        self.assertTrue(bool(((state['output'] > .01) & (state['output'] < .99)).any()))
        self.assertTrue(bool((state['w']._values() < 0).any()))
        before = snapshot(state)
        for query_id in self.ids:
            core.native_readout(state, query_id)
        for key, saved in before.items():
            if saved.layout == torch.sparse_coo:
                self.assertTrue(torch.equal(saved.to_dense(), state[key].to_dense()))
            else:
                self.assertTrue(torch.equal(saved, state[key]))

    def test_source_text_scale_is_multiply_then_matmul(self):
        module = core.load_upstream()
        actual = module.get_zero_shot_logits(self.x, None, self.a)
        self.assertTrue(torch.equal(actual, ((100 * self.x) @ self.a).squeeze()))

    def test_labels_do_not_affect_fit(self):
        changed = core.fit(self.x, self.a, ids=self.ids,
                           query_labels=torch.tensor([99., -3., 5., 8., 0., 1., 2.]))
        for key in ['output', 'mu', 'covariance', 'z_before_final', 'y_hat']:
            self.assertTrue(torch.equal(changed[key], self.state[key]), key)

    def test_fit_does_not_mutate_inputs(self):
        before_x, before_a = self.x.clone(), self.a.clone()
        core.fit(self.x, self.a, ids=self.ids)
        self.assertTrue(torch.equal(before_x, self.x))
        self.assertTrue(torch.equal(before_a, self.a))

    def test_absent_readout_matches_full_one_step_with_frozen_support(self):
        state = core.fit(self.x[1:], self.a, ids=self.ids[1:])
        before = snapshot(state)
        actual = core.heldout_readout(state, self.x[0], query_id=self.ids[0])
        module = core.load_upstream()
        augmented = torch.cat([self.x[1:], self.x[:1]])
        w = module.build_affinity_matrix(augmented, 3)
        adapter = module.Gaussian(state['mu'], state['covariance'])
        likelihoods = adapter(augmented, no_exp=True)
        y_hat = torch.softmax(module.get_zero_shot_logits(augmented, None, self.a), dim=1)
        # Query dummy assignment cannot affect itself: diagonal edge is absent.
        z = torch.cat([state['z_before_final'], torch.tensor([[.2, .3, .5]])])
        expected = module.update_z(likelihoods, y_hat, z, w, 1, 1, 3,
                                   state['covariance'], 1)[-1]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
        z[-1] = torch.tensor([.8, .1, .1])
        again = module.update_z(likelihoods, y_hat, z, w, 1, 1, 3,
                                state['covariance'], 1)[-1]
        self.assertTrue(torch.equal(expected, again))
        for key, saved in before.items():
            current = state[key]
            if current.layout == torch.sparse_coo:
                self.assertTrue(torch.equal(saved.to_dense(), current.to_dense()), key)
            else:
                self.assertTrue(torch.equal(saved, current), key)

    def test_cyclic_other_deletion_support_and_retained_query(self):
        plans = core.cyclic_deletions(self.ids)
        self.assertEqual([p['other_excluded'] for p in plans], list(self.ids[1:] + self.ids[:1]))
        for p in plans:
            self.assertNotIn(p['query_id'], p['loo_support'])
            self.assertIn(p['query_id'], p['other_support'])
            self.assertNotIn(p['other_excluded'], p['other_support'])
            self.assertEqual(len(p['loo_support']), len(self.ids) - 1)
            self.assertEqual(len(p['other_support']), len(self.ids) - 1)
        p = plans[0]
        retained = [i for i, q in enumerate(self.ids) if q != p['other_excluded']]
        state = core.fit(self.x[retained], self.a, ids=p['other_support'])
        self.assertEqual(core.native_readout(state, p['query_id']).argmax(),
                         state['output'][0].argmax())

    def test_duplicate_feature_tie_is_boundary(self):
        x = self.x.clone(); x[1] = x[0]
        self.assertBoundary('graph_tie', core.fit, x, self.a, ids=self.ids)

    def test_exact_class_tie_is_boundary(self):
        same = self.a[:, :, :1].repeat(1, 1, 2)
        self.assertBoundary('class_tie', core.fit, self.x, same, ids=self.ids)

    def test_upper_support_cap_precedes_scientific_computation(self):
        self.assertBoundary('support_size', core.fit, torch.zeros(2500, 5),
                            self.a, ids=tuple(range(2500)))

    def test_actual_self_edge_is_boundary(self):
        module = core.load_upstream()
        bad = torch.sparse_coo_tensor(torch.tensor([[0], [0]]), torch.tensor([1.]),
                                     (len(self.x), len(self.x)))
        with patch.object(module, 'build_affinity_matrix', return_value=bad):
            self.assertBoundary('graph_self_edge', core._graph, module, self.x)

    def test_size_class_dimension_and_ids_boundaries(self):
        self.assertBoundary('support_size', core.fit, self.x[:4], self.a, ids=self.ids[:4])
        self.assertBoundary('prototype_shape', core.fit, self.x, self.a[:, :, :1], ids=self.ids)
        self.assertBoundary('sample_ids', core.fit, self.x, self.a, ids=('a',) * len(self.x))
        self.assertBoundary('deletion_ids', core.cyclic_deletions, self.ids[:5])

    def test_nonfinite_nonunit_and_dtype_boundaries(self):
        x = self.x.clone(); x[0, 0] = float('nan')
        self.assertBoundary('nonfinite_input', core.fit, x, self.a, ids=self.ids)
        self.assertBoundary('nonunit_input', core.fit, self.x * 2, self.a, ids=self.ids)
        self.assertBoundary('input_device_dtype', core.fit, self.x.double(), self.a, ids=self.ids)

    def test_zero_nonfinite_moments_and_snapshot_are_boundaries(self):
        bad = dict(self.state); bad['covariance'] = torch.zeros_like(bad['covariance'])
        self.assertBoundary('nonpositive_covariance', core.native_readout, bad, 'a')
        bad = dict(self.state); bad['mu'] = torch.zeros_like(bad['mu'])
        self.assertBoundary('zero_mean', core.native_readout, bad, 'a')
        bad = dict(self.state); bad['z_before_final'] = torch.full_like(bad['z_before_final'], float('nan'))
        self.assertBoundary('nonfinite_snapshot', core.native_readout, bad, 'a')

    def test_present_and_absent_identity_are_distinct(self):
        self.assertBoundary('native_query_id', core.native_readout, self.state, 'outside')
        self.assertBoundary('heldout_query_id', core.heldout_readout, self.state,
                            self.x[0], query_id='a')
        self.assertBoundary('graph_tie', core.heldout_readout, self.state,
                            self.x[0], query_id='outside')


if __name__ == '__main__':
    unittest.main(verbosity=2)
