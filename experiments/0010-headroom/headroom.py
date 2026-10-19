"""Unfrozen headroom reference; no encoder, data acquisition or automatic fit."""
import hashlib
import json
import math
import re
import time
from pathlib import Path
import numpy as np

ARMS = ('zero_shot', 'native_alpha1', 'native_alpha4', 'loo_alpha1', 'other_out_alpha1')
CONTROLS = (0, 1, 2, 4)


class Invalid(ValueError):
    def __init__(self, message):
        self.code = message
        super().__init__(message)


def require(ok, message):
    if not ok:
        raise Invalid(message)


def array_hash(a):
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + b'\0' + a.tobytes(order='C')).hexdigest()


def ids_hash(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(',', ':')).encode()).hexdigest()


def probability(value):
    p = np.asarray(value, dtype=np.float64)
    require(p.shape == (10,) and np.isfinite(p).all() and (p >= 0).all(), 'invalid_probability')
    require(abs(float(p.sum()) - 1.) <= 2e-6, 'probability_mass')
    top = np.sort(p)[-2:]
    require(top[0] != top[1], 'probability_tie')
    return p.tolist()


class TorchStatA:
    """Explicit adapter only. Constructing this imports Torch but never fits."""
    kind = 'reviewed_stata_cpu'
    def __init__(self, stata_module, source_path):
        import torch
        self.torch, self.core, self.source_path = torch, stata_module, Path(source_path)
    def fit(self, x, p, ids, alpha):
        return self.core.fit(self.torch.from_numpy(x.copy()), self.torch.from_numpy(p.copy()), ids=ids, alpha=alpha, source_path=self.source_path)
    def zero_shot(self, x, p):
        t, m = self.torch, self.core.load_upstream(self.source_path)
        with t.no_grad():
            logits = m.get_zero_shot_logits(t.from_numpy(x.copy()), None, t.from_numpy(p.copy()))
            return t.softmax(logits, dim=1).numpy()
    def native(self, state, query_id):
        return self.core.native_readout(state, query_id).detach().numpy()
    def heldout(self, state, query, query_id):
        return self.core.heldout_readout(state, self.torch.from_numpy(query.copy()), query_id=query_id, source_path=self.source_path).detach().numpy()


def predict(features, prototypes, ids, backend, output_path, clock=time.perf_counter):
    """No labels accepted. Fail-fast; every row/arm remains explicitly represented.

    Save on each logical stage. External supervisor remains necessary for abrupt
    termination and resource limits; only a completed record can later be scored.
    """
    start = clock()
    x, p, ids = np.asarray(features), np.asarray(prototypes), tuple(ids)
    require(x.shape == (128, 512) and p.shape == (1, 512, 10), 'fixed_shapes')
    require(x.dtype == p.dtype == np.dtype('float32'), 'float32_required')
    require(np.isfinite(x).all() and np.isfinite(p).all(), 'finite_features')
    require(np.allclose(np.linalg.norm(x, axis=1), 1, atol=2e-5, rtol=0) and
            np.allclose(np.linalg.norm(p[0], axis=0), 1, atol=2e-5, rtol=0), 'unit_features')
    require(len(ids) == 128 and len(set(ids)) == 128 and
            all(isinstance(q, str) and re.fullmatch('[0-9a-f]{64}', q) for q in ids), 'opaque_ids')
    path = Path(output_path)
    require(not path.exists() and not path.is_symlink() and not path.with_suffix(path.suffix + '.pending').exists(), 'fresh_output_required')
    rows = [dict(id=q, batch=i // 64, donor_id=ids[(i // 64) * 64 + (i + 1) % 64], probabilities={a: None for a in ARMS},
                 statuses={a: 'not_attempted' for a in ARMS}, reasons={}) for i, q in enumerate(ids)]
    result = dict(schema=1, status='started', backend=backend.kind, arms=list(ARMS), rows=rows,
                  inputs=dict(features_sha256=array_hash(x), prototypes_sha256=array_hash(p), ids_sha256=ids_hash(ids)),
                  fit_states=[], successful_native_solver_executions=0,
                  costs_seconds=dict(zero_shot=0., fits_with_parity=0., one_way_readouts=0.))
    with path.open('x') as f:
        json.dump(result, f, allow_nan=False)
    def save():
        tmp = path.with_suffix(path.suffix + '.pending')
        with tmp.open('w') as f: json.dump(result, f, allow_nan=False)
        tmp.replace(path)
    def timed(category, function):
        began = clock()
        try: return function()
        finally:
            elapsed = clock() - began
            require(math.isfinite(elapsed) and elapsed >= 0, 'invalid_clock')
            result['costs_seconds'][category] += elapsed
    def put(index, arm, value):
        rows[index]['probabilities'][arm] = probability(value)
        rows[index]['statuses'][arm] = 'ok'
    def fitted(sub, sub_ids, alpha, tag, batch, excluded=None):
        event = dict(tag=tag, batch=batch, excluded_id=excluded, alpha=alpha, support_n=len(sub_ids),
                     support_ids_sha256=ids_hash(sub_ids), status='attempted')
        result['fit_states'].append(event); save()
        try:
            state = timed('fits_with_parity', lambda: backend.fit(sub, p, sub_ids, alpha))
            require(state.get('status') == 'ok' and state.get('reference_bitwise_equal') is True and state.get('native_solver_executions') == 2, 'fit_parity')
            event['status'] = 'completed'
            result['successful_native_solver_executions'] += 2
            return state
        except Exception:
            event['status'] = 'failed'; event['native_solver_execution_count_unknown'] = True
            raise
    affected = []
    try:
        for b in range(2):
            first = b * 64; bx, bids = x[first:first + 64], ids[first:first + 64]
            affected = [(first + j, ARMS[0]) for j in range(64)]
            zs = timed('zero_shot', lambda: backend.zero_shot(bx, p))
            require(np.asarray(zs).shape == (64, 10), 'zero_shot_shape')
            for j in range(64): put(first + j, ARMS[0], zs[j])
            save()
            for alpha, arm in ((1, ARMS[1]), (4, ARMS[2])):
                affected = [(first + j, arm) for j in range(64)]
                state = fitted(bx, bids, alpha, 'full-alpha' + str(alpha), b)
                for j, q in enumerate(bids):
                    put(first + j, arm, timed('one_way_readouts', lambda q=q: backend.native(state, q)))
                save()
            for j, excluded in enumerate(bids):
                previous = (j - 1) % 64
                affected = [(first + j, ARMS[3]), (first + previous, ARMS[4])]
                keep = [k for k in range(64) if k != j]
                state = fitted(bx[keep], tuple(bids[k] for k in keep), 1, 'delete-' + str(j), b, excluded)
                put(first + j, ARMS[3], timed('one_way_readouts', lambda: backend.heldout(state, bx[j], excluded)))
                put(first + previous, ARMS[4], timed('one_way_readouts', lambda: backend.native(state, bids[previous])))
                save()
        require(len(result['fit_states']) == 132 and result['successful_native_solver_executions'] == 264, 'fit_counts')
        require(all(r['statuses'][a] == 'ok' for r in rows for a in ARMS), 'incomplete_predictions')
        result['status'] = 'completed'
    except Exception as e:
        reason = getattr(e, 'code', type(e).__name__)
        for i, arm in affected:
            if rows[i]['statuses'][arm] != 'ok':
                rows[i]['statuses'][arm] = 'refused'; rows[i]['reasons'][arm] = reason
        result.update(status='incomplete', error=reason)
    finally:
        result['costs_seconds']['predictor_wall_before_final_save'] = clock() - start
        save()
    return result


def score(saved, labels, label_ids):
    """Only saved complete predictions enter label-dependent arithmetic."""
    require(saved.get('status') == 'completed' and saved.get('arms') == list(ARMS), 'unscorable_incomplete')
    cost = saved['costs_seconds']
    keys = ('zero_shot', 'fits_with_parity', 'one_way_readouts', 'predictor_wall_before_final_save')
    require(set(cost) == set(keys) and all(type(cost[k]) in (int, float) and math.isfinite(cost[k]) and cost[k] >= 0 for k in keys), 'cost_fields')
    require(cost[keys[-1]] >= sum(cost[k] for k in keys[:-1]), 'cost_accounting')
    rows = saved['rows']; require(len(rows) == 128, 'row_count')
    ids = tuple(r['id'] for r in rows)
    require(len(set(ids)) == 128 and all(re.fullmatch('[0-9a-f]{64}', q) for q in ids), 'row_ids')
    require(ids_hash(ids) == saved['inputs']['ids_sha256'], 'id_binding')
    require(tuple(label_ids) == ids, 'label_id_order')
    require(all(r['batch'] == i // 64 and r['donor_id'] == ids[(i // 64) * 64 + (i + 1) % 64] and all(r['statuses'][a] == 'ok' for a in ARMS) for i, r in enumerate(rows)), 'coverage')
    events = saved['fit_states']
    require(len(events) == 132 and saved['successful_native_solver_executions'] == 264 and
            all(e['status'] == 'completed' for e in events), 'fit_completion')
    for b in range(2):
        bids = ids[b*64:(b+1)*64]; e = events[b*66:(b+1)*66]
        for k, event in enumerate(e):
            deleted = None if k < 2 else bids[k-2]
            sub = bids if deleted is None else tuple(q for q in bids if q != deleted)
            require(event['batch'] == b and event['excluded_id'] == deleted and
                    event['alpha'] == (4 if k == 1 else 1) and event['support_n'] == len(sub) and
                    event['support_ids_sha256'] == ids_hash(sub), 'fit_lineage')
    probs = np.array([[probability(r['probabilities'][a]) for a in ARMS] for r in rows])
    y = np.asarray(labels)
    require(y.shape == (128,) and y.dtype.kind in 'iu' and (y >= 0).all() and (y < 10).all(), 'labels')
    predictions = probs.argmax(axis=2); correct = predictions == y[:, None]
    counts = correct.sum(axis=0); best = max(int(counts[c]) for c in CONTROLS)
    co_best = [c for c in CONTROLS if counts[c] == best]
    def pairs(mask, c):
        win = int((mask & correct[:, 3] & ~correct[:, c]).sum())
        harm = int((mask & ~correct[:, 3] & correct[:, c]).sum())
        return dict(n=int(mask.sum()), corrections=win, harms=harm, net=win-harm,
                    disagreements=int((mask & (predictions[:, 3] != predictions[:, c])).sum()))
    all_rows = np.ones(128, dtype=bool)
    paired = {ARMS[c]: pairs(all_rows, c) for c in CONTROLS}
    batches = [{ARMS[c]: pairs(np.arange(128)//64 == b, c) for c in CONTROLS} for b in range(2)]
    occupancy = np.concatenate([np.bincount(y[b*64:(b+1)*64], minlength=10)[y[b*64:(b+1)*64]] for b in range(2)])
    strata = {name: {ARMS[c]: pairs(mask, c) for c in CONTROLS} for name, mask in
              [('one', occupancy == 1), ('two', occupancy == 2), ('three_plus', occupancy >= 3)]}
    gates = dict(net_eight_over_best=int(counts[3])-best >= 8,
                 both_batches_positive=all(batches[b][ARMS[c]]['net'] > 0 for b in range(2) for c in co_best),
                 no_singleton_net_harm=all(strata['one'][ARMS[c]]['net'] >= 0 for c in (1, 2)))
    insensitive = 128-best < 8 or any(paired[ARMS[c]]['disagreements'] < 8 for c in co_best)
    decision = 'CONTINUE_TO_MECHANISM' if all(gates.values()) else ('LOW_HEADROOM' if insensitive else 'NO_GO')
    return dict(scope='development headroom only; no independent-scene confidence claim', decision=decision,
                n=128, correct={a:int(counts[k]) for k,a in enumerate(ARMS)},
                accuracy={a:float(counts[k]/128) for k,a in enumerate(ARMS)},
                strongest_fixed_controls=[ARMS[c] for c in co_best], paired=paired, batches=batches,
                occupancy_strata=strata, singleton_assessed=bool((occupancy == 1).any()), gates=gates,
                costs_seconds=saved['costs_seconds'])


def score_file(prediction_path, labels, label_ids, output_path):
    saved = json.loads(Path(prediction_path).read_text())
    summary = score(saved, labels, label_ids)
    with Path(output_path).open('x') as f: json.dump(summary, f, indent=2, allow_nan=False)
    return summary
