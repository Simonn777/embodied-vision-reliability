"""Artificial-fixture CPU parity prototype. No data/encoder entry point.

StatA upstream is AGPL-3.0. This wrapper loads only its hash-verified solver;
no upstream source is embedded and no numerical boundary is silently repaired.
"""
import hashlib
from pathlib import Path
from types import ModuleType
import torch

UPSTREAM_SHA256 = '9fcd2eb8a22cabc2c7eb43823625cb04557476c637a24c8a12eb3115ede2b962'
UPSTREAM_PATH = Path('/tmp/evr-stata-source-admission/source/solvers/StatA.py')
KNN = 3


class BoundaryError(ValueError):
    def __init__(self, code, detail=''):
        self.status, self.code = 'boundary', code
        super().__init__(code + (': ' + detail if detail else ''))


def _require(condition, code):
    if not condition:
        raise BoundaryError(code)


def _finite(tensor, code):
    _require(bool(torch.isfinite(tensor).all()), code)


def _class_ties(probabilities):
    top = probabilities.topk(2, dim=-1).values
    _require(not bool((top[..., 0] == top[..., 1]).any()), 'class_tie')


def load_upstream(path=UPSTREAM_PATH):
    raw = Path(path).read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == UPSTREAM_SHA256, 'source_hash')
    source = raw.decode('utf-8')
    _require(source.count('.cuda()') == 4, 'cuda_substitution_count')
    cpu_source = source.replace('.cuda()', ".to('cpu')")
    module = ModuleType('_hash_verified_stata_cpu')
    module.__file__ = str(path)
    exec(compile(cpu_source, str(path), 'exec'), module.__dict__)
    module.device_source_sha256 = hashlib.sha256(cpu_source.encode()).hexdigest()
    return module


def _unit_rows(x):
    _require(isinstance(x, torch.Tensor), 'input_type')
    _require(x.device.type == 'cpu' and x.dtype == torch.float32, 'input_device_dtype')
    _finite(x, 'nonfinite_input')
    _require(bool(torch.allclose(x.norm(dim=-1), torch.ones(x.shape[:-1]),
                                 atol=2e-5, rtol=0)), 'nonunit_input')


def _graph(module, x):
    similarity = x @ x.T
    # A tie through the first excluded neighbor makes the authored topk ambiguous.
    top = similarity.topk(KNN + 2, dim=1).values
    _require(not bool((top[:, :-1] == top[:, 1:]).any()), 'graph_tie')
    w = module.build_affinity_matrix(x, KNN)
    _finite(w._values(), 'nonfinite_graph')
    _require(not bool((w._indices()[0] == w._indices()[1]).any()), 'graph_self_edge')
    return w


def _validate(features, prototypes, ids):
    _require(isinstance(features, torch.Tensor) and isinstance(prototypes, torch.Tensor), 'input_type')
    _require(features.ndim == 2 and 5 <= features.shape[0] < 2500, 'support_size')
    _require(prototypes.ndim == 3 and prototypes.shape[0] == 1 and
             prototypes.shape[1] == features.shape[1] and
             prototypes.shape[2] >= 2, 'prototype_shape')
    _unit_rows(features)
    _unit_rows(prototypes[0].T)
    _require(len(ids) == len(features) and len(set(ids)) == len(ids), 'sample_ids')


def _check_moments(mu, covariance):
    _finite(mu, 'nonfinite_mean')
    _finite(covariance, 'nonfinite_covariance')
    _require(bool((mu.norm(dim=-1) > 0).all()), 'zero_mean')
    _require(bool((covariance > 0).all()), 'nonpositive_covariance')


def fit(features, prototypes, *, ids, alpha=1, query_labels=None, source_path=UPSTREAM_PATH):
    """Two native executions: uninstrumented reference plus instrumented parity.

    ids follow the supplied, predeclared order. Labels are ignored by upstream;
    the optional argument exists only for the no-GT-effect artificial fixture.
    """
    _require(type(alpha) in (int, float) and alpha in (1, 4), 'alpha')
    alpha = int(alpha)
    ids = tuple(ids)
    _validate(features, prototypes, ids)
    x, anchors = features.detach().clone(), prototypes.detach().clone()
    labels = torch.zeros(len(x)) if query_labels is None else query_labels.clone()
    _require(labels.shape == (len(x),), 'label_shape')
    _finite(labels, 'nonfinite_labels')
    module = load_upstream(source_path)
    checked_w = _graph(module, x)
    state, adapters = {'blocks': 0, 'inner_updates': 0, 'moment_updates': 0,
                       'covariance_updates': 0}, []
    original_update, original_init = module.update_z, module.Gaussian.__init__
    original_mu, original_cov = module.update_mu, module.update_cov

    def record_init(adapter, *args, **kwargs):
        original_init(adapter, *args, **kwargs)
        adapters.append(adapter)

    def checked_mu(adapter, query, z, beta, initial):
        state['moment_updates'] += 1
        mass = z.sum(0)
        _require(bool((mass > 0).all()), 'zero_class_mass')
        resultant = (z.T @ query) / mass[:, None]
        _require(bool((resultant.norm(dim=1) > 0).all()), 'zero_resultant')
        result = original_mu(adapter, query, z, beta, initial)
        _finite(result, 'nonfinite_mean')
        return result

    def checked_cov(*args):
        state['covariance_updates'] += 1
        return original_cov(*args)

    def record_update(likelihoods, y_hat, z, w, ly, ll, k, sigma, max_iter=5):
        _check_moments(adapters[-1].mu, sigma)
        _class_ties(y_hat)
        _finite(likelihoods, 'nonfinite_likelihood')
        _require(max_iter == 5, 'inner_schedule')
        block = state['blocks']
        state['blocks'] += 1
        state['inner_updates'] += 5
        if block == 10:
            before = original_update(likelihoods, y_hat, z, w, ly, ll, k, sigma, 4)
            state.update(z_before_final=before.clone(), y_hat=y_hat.clone(),
                         w=w.clone(), likelihoods=likelihoods.clone())
            result = original_update(likelihoods, y_hat, before, w, ly, ll, k, sigma, 1)
        else:
            result = original_update(likelihoods, y_hat, z, w, ly, ll, k, sigma, 5)
        _finite(result, 'nonfinite_responsibility')
        _class_ties(result)
        _require(bool((result.sum(0) > 0).all()), 'zero_class_mass')
        return result

    module.Gaussian.__init__, module.update_z = record_init, record_update
    module.update_mu, module.update_cov = checked_mu, checked_cov
    with torch.no_grad():
        y_hat, output = module.StatA_solver(x, labels, anchors, alpha=alpha)
    _require(state['blocks'] == 11 and state['inner_updates'] == 55, 'outer_schedule')
    _require(state['moment_updates'] == state['covariance_updates'] == 10, 'moment_schedule')
    adapter = adapters[0]
    _check_moments(adapter.mu, adapter.cov)
    state.update(status='ok', ids=ids, alpha=alpha, features=x, prototypes=anchors,
                 mu=adapter.mu.clone(), covariance=adapter.cov.clone(),
                 output=output.clone(), source_sha256=UPSTREAM_SHA256,
                 device_source_sha256=module.device_source_sha256,
                 native_solver_executions=2)
    _require(torch.equal(state['w'].to_dense(), checked_w.to_dense()), 'graph_changed')
    reference = load_upstream(source_path)
    with torch.no_grad():
        ref_y, ref_output = reference.StatA_solver(x, labels, anchors, alpha=alpha)
    _require(torch.equal(y_hat, ref_y) and torch.equal(output, ref_output), 'native_parity')
    state['reference_bitwise_equal'] = True
    for query_id in ids:
        native_readout(state, query_id)
    return state


def _readout(state, query, w, row, support_rows, y_hat):
    _check_moments(state['mu'], state['covariance'])
    _finite(state['z_before_final'], 'nonfinite_snapshot')
    sigma = state['covariance']
    likelihood = -0.5 * (((query[None, :] - state['mu'][:, 0, :]) ** 2) * (1 / sigma)).sum(1)
    _finite(likelihood, 'nonfinite_likelihood')
    dense = w.to_dense()
    incoming = dense[support_rows, row] @ state['z_before_final']
    outgoing = dense[row, support_rows] @ state['z_before_final']
    intermediate = likelihood + (50 / (KNN * 2)) * (incoming + outgoing)
    intermediate -= 0.5 * sigma.log().sum(1)
    intermediate -= intermediate.max()
    result = y_hat * torch.exp(intermediate / 50)
    _require(bool(result.sum() > 0), 'zero_readout_mass')
    result = result / result.sum()
    _finite(result, 'nonfinite_readout')
    _class_ties(result)
    return result


def native_readout(state, query_id):
    """Use native W, but read frozen z-before-final without updating any support."""
    _require(state.get('status') == 'ok' and query_id in state['ids'], 'native_query_id')
    row = state['ids'].index(query_id)
    result = _readout(state, state['features'][row], state['w'], row,
                      torch.arange(len(state['ids'])), state['y_hat'][row])
    expected = state['output'][row]
    _require(torch.allclose(result, expected, atol=2e-6, rtol=1e-5) and
             result.argmax() == expected.argmax(), 'readout_parity')
    return result


def heldout_readout(state, query, *, query_id, source_path=UPSTREAM_PATH):
    """Only derive graph weights on support+query; never refit support moments/z."""
    _require(state.get('status') == 'ok' and query_id not in state['ids'], 'heldout_query_id')
    _require(query.shape == (state['features'].shape[1],), 'query_shape')
    _unit_rows(query[None, :])
    _require(len(state['ids']) + 1 < 2500, 'augmented_size')
    augmented = torch.cat([state['features'], query[None, :]], dim=0)
    module = load_upstream(source_path)
    w = _graph(module, augmented)
    logits = module.get_zero_shot_logits(query[None, :], None, state['prototypes'])
    y_hat = torch.softmax(logits, dim=0)
    return _readout(state, query, w, len(state['ids']),
                    torch.arange(len(state['ids'])), y_hat)


def cyclic_deletions(ids):
    ids = tuple(ids)
    _require(len(ids) >= 6 and len(set(ids)) == len(ids), 'deletion_ids')
    return [dict(query_id=q, loo_excluded=q, other_excluded=ids[(i + 1) % len(ids)],
                 loo_support=tuple(x for x in ids if x != q),
                 other_support=tuple(x for x in ids if x != ids[(i + 1) % len(ids)]))
            for i, q in enumerate(ids)]
