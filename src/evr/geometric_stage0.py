"""Unfrozen Stage0 translation-ranking implementation. No GT metric is imported."""
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
import signal
import threading
import time

import numpy as np

from .geometric_verification import project, matched_residuals, pairwise_radius_A, marginal_cell_counts
from .exact_assignment_tail import exact_permutation_tail, UnsupportedBoard

FAMILIES = ('uniform', 'clustered_outlier', 'compact_true', 'sparse_isolated')
STREAMS = ('global-shift', 'inlier-xy', 'inlier-noise', 'outlier-source',
           'outlier-target', 'outlier-pairing', 'wrong-noise', 'clone-base',
           'priority', 'candidate-order')
CONTROLS = ('raw', 'msac', 'cell', 'excess', 'ratio')
METHODS = CONTROLS + ('tail',)
TIMINGS = ('generation', 'deduplication', 'input_audit', 'spatial_counts',
           'projection', 'matched_residuals', 'graph_construction', 'graph_audit',
           'cheap_reductions', 'integer_tail', 'selection')
CONFIG = {
    'schema': 'evr0009.stage0.v1', 'families': list(FAMILIES), 'seeds_per_family': 64,
    'base_pairs': 128, 'true_pairs_first_half': 8, 'true_pairs_second_half': 16,
    'clone_base_count': 16, 'copies_per_clone': 2, 'tau': 3.0, 'dedup_radius': 1.0,
    'candidate_count': 32, 'case_seconds': 2.5, 'overall_seconds': 900,
    'operation_limit': 2000000, 'max_n': 512, 'small_side_limit': 12,
    'metric': 'mean_one_way_euclidean_81_grid_0_512', 'success_pixels': 3.0,
    'primary_net_gain': 8, 'max_paired_harms_per_family_control': 2,
    'coordinate_extent': 512.0, 'threads_max': 4, 'memory_bytes': 8589934592,
}


def array_sha256(value):
    a = np.asarray(value)
    if a.dtype.kind == 'f':
        a = np.asarray(a, dtype='<f8', order='C')
    elif a.dtype.kind in 'iu':
        a = np.asarray(a, dtype='<i8', order='C')
    elif a.dtype.kind == 'b':
        a = np.asarray(a, dtype=np.bool_, order='C')
    else:
        raise ValueError('unsupported array hash dtype')
    header = json.dumps({'shape': list(a.shape), 'dtype': a.dtype.str}, sort_keys=True,
                        separators=(',', ':')).encode('ascii')
    return hashlib.sha256(header + b'\n' + a.tobytes(order='C')).hexdigest()


def case_order():
    return [(FAMILIES[(j + s % 4) % 4], s) for s in range(64) for j in range(4)]


def stream_seeds(family, seed_index):
    if family not in FAMILIES or type(seed_index) is not int or not 0 <= seed_index < 64:
        raise ValueError('invalid production case ID')
    return {name: int.from_bytes(hashlib.sha256(
        f'evr0009|stage0|{family}|{seed_index}|{name}'.encode('ascii')).digest()[:8],
        'little') for name in STREAMS}


def candidate_pool():
    offsets = [(a, b) for a in (-8, 0, 8) for b in (-8, 0, 8)]
    offsets += [(128+a, 128+b) for a in (-8, 0, 8) for b in (-8, 0, 8)]
    offsets += [(-24,0),(24,0),(0,-24),(0,24),(-24,-24),(-24,24),(24,-24),
                (24,24),(32,0),(-32,0),(0,32),(0,-32),(32,32),(-32,-32)]
    h = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], len(offsets), axis=0)
    h[:, :2, 2] = offsets
    return h


def _disk(rng, count, radius):
    # All angles, then all radii: the array draw order is fixed, not interleaved.
    angle = 2 * np.pi * rng.random(count)
    radii = radius * np.sqrt(rng.random(count))
    return radii[:, None] * np.column_stack((np.cos(angle), np.sin(angle)))


def _noisy_target(x, shift, rng, offset=(0.0, 0.0)):
    result = np.empty_like(x)
    for i, point in enumerate(x):
        for _ in range(64):
            value = point + np.asarray(offset) + rng.normal(0, 1, size=2)
            if np.all((value + shift >= 0) & (value + shift <= 512)):
                result[i] = value
                break
        else:
            raise ValueError('generator exhausted 64 Gaussian attempts for one pair')
    return result


def construct_case(family, seed_index, rngs, *, base_count=128, true_count=None,
                   clone_count=16):
    """Generator accepts explicit streams; smaller dimensions are artificial-test hooks.

    Production caller fixes 128/8-or-16/16. Returns scoring inputs and separate
    evaluator-only labels. Labels never enter score_inputs or select_controls.
    """
    if family not in FAMILIES or set(rngs) != set(STREAMS):
        raise ValueError('unknown family or incomplete RNG streams')
    k = (8 if seed_index < 32 else 16) if true_count is None else true_count
    if not (0 < k < base_count and 0 <= clone_count <= base_count):
        raise ValueError('invalid generator dimensions')
    m = base_count - k
    shift = rngs['global-shift'].uniform(-8, 8, size=2)
    if family in ('uniform', 'clustered_outlier'):
        tx = rngs['inlier-xy'].uniform(32, 480, size=(k, 2))
    else:
        tx = np.array([256., 256.]) + _disk(rngs['inlier-xy'], k, 6)
    ty = _noisy_target(tx, shift, rngs['inlier-noise'])
    if family == 'clustered_outlier':
        centers = (np.array([(64,64),(160,64),(64,128),(160,128)], dtype=np.float64)
                   if seed_index % 2 == 0 else
                   np.array([(a,b) for a in (64,96,128,160) for b in (64,96,128)],
                            dtype=np.float64))
        radius = .4 if seed_index % 2 == 0 else 2.
        center_rows = centers[np.arange(m) % len(centers)]
        wx = center_rows + _disk(rngs['outlier-source'], m, radius)
        wy = center_rows + 128 + _disk(rngs['outlier-target'], m, radius)
        wy = wy[rngs['outlier-pairing'].permutation(m)]
    elif family == 'sparse_isolated':
        if m < 4:
            raise ValueError('sparse fixture needs four explicit wrong pairs')
        sparse = np.array([(64,64),(320,64),(64,320),(320,320)], dtype=np.float64)
        wx = np.concatenate((sparse, rngs['outlier-source'].uniform(32,480,size=(m-4,2))))
        wy = np.concatenate((_noisy_target(sparse, shift, rngs['wrong-noise'], (128,128)),
                             rngs['outlier-target'].uniform(32,480,size=(m-4,2))))
    else:
        wx = rngs['outlier-source'].uniform(32,480,size=(m,2))
        wy = rngs['outlier-target'].uniform(32,480,size=(m,2))
    x, y = np.concatenate((tx,wx)) + shift, np.concatenate((ty,wy)) + shift
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not (
            ((0 <= x) & (x <= 512)).all() and ((0 <= y) & (y <= 512)).all()):
        raise ValueError('generator produced invalid coordinates')
    chosen = rngs['clone-base'].choice(base_count, size=clone_count, replace=False)
    base_ids = np.concatenate((np.arange(base_count), np.repeat(chosen, 2)))
    raw_x, raw_y = x[base_ids], y[base_ids]
    priorities = rngs['priority'].random(len(base_ids))
    order = rngs['candidate-order'].permutation(32)
    inputs = {'raw_x': raw_x, 'raw_y': raw_y, 'priorities': priorities,
              'candidate_order': order, 'candidates': candidate_pool()}
    labels = {'H_star': np.eye(3).tolist(), 'true_candidate_id': 4,
              'base_ids': base_ids.tolist(), 'true_flags': (base_ids < k).tolist(),
              'clone_flags': (np.arange(len(base_ids)) >= base_count).tolist()}
    return inputs, labels


def deduplicate(x, y, priorities, radius=1.0):
    x, y, priorities = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(priorities, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or x.shape != y.shape or priorities.shape != (len(x),) or not len(x):
        raise ValueError('invalid dedup input shapes')
    if not all(np.isfinite(a).all() for a in (x,y,priorities)) or not np.isfinite(radius) or radius < 0:
        raise ValueError('invalid dedup values')
    order = np.lexsort((np.arange(len(x)), priorities))
    retained, decisions = [], []
    for raw_id in order.tolist():
        source = target = False
        if retained:
            dx, dy = x[retained] - x[raw_id], y[retained] - y[raw_id]
            source = bool(np.any(np.hypot(dx[:,0],dx[:,1]) <= radius))
            target = bool(np.any(np.hypot(dy[:,0],dy[:,1]) <= radius))
        keep = not (source or target)
        if keep:
            retained.append(raw_id)
        decisions.append({'raw_id': raw_id, 'accepted': keep,
                          'reason': ('both' if source and target else 'source' if source else
                                     'target' if target else None)})
    return retained, decisions


class CaseDeadline(TimeoutError):
    pass


@contextmanager
def deadline(seconds):
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, 'setitimer'):
        raise RuntimeError('case timer requires POSIX main thread')
    if signal.getitimer(signal.ITIMER_REAL) != (0., 0.):
        raise RuntimeError('refusing to replace an existing alarm')
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError('invalid case time cap')
    old = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise CaseDeadline('case wall deadline expired')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


@contextmanager
def timed(record, key):
    start = time.perf_counter()
    try:
        yield
    finally:
        record[key] += time.perf_counter() - start


def _components(a):
    remaining = set(np.flatnonzero(a.any(axis=1)).tolist())
    summaries = []
    while remaining:
        rows, cols, pending = set(), set(), [min(remaining)]
        while pending:
            i = pending.pop()
            if i in rows:
                continue
            rows.add(i)
            for j in np.flatnonzero(a[i]).tolist():
                if j not in cols:
                    cols.add(j)
                    pending.extend(np.flatnonzero(a[:,j]).tolist())
        remaining -= rows
        summaries.append({'rows': len(rows), 'columns': len(cols),
                          'edges': int(a[sorted(rows)].sum())})
    return summaries


def select_controls(records):
    selected = {}
    for method in CONTROLS:
        def key(row):
            if method == 'ratio':
                return Fraction(*row['ratio_key'])
            return row[method + '_key']
        best = max(key(row) for row in records)
        ties = [row['canonical_id'] for row in records if key(row) == best]
        selected[method] = {'selected_id': ties[0], 'tied_ids': ties}
    return selected


def score_inputs(inputs, result, *, tau=3., operation_limit=2000000,
                 tail_function=exact_permutation_tail):
    """Score inputs without GT. Mutable result preserves progress if timer interrupts."""
    t = result['timings']
    with timed(t, 'deduplication'):
        kept, decisions = deduplicate(inputs['raw_x'], inputs['raw_y'], inputs['priorities'])
        x, y = inputs['raw_x'][kept], inputs['raw_y'][kept]
    with timed(t, 'input_audit'):
        result['inputs'] = {k: np.asarray(v).tolist() for k,v in inputs.items()}
        result['input_hashes'] = {k: array_sha256(v) for k,v in inputs.items()}
        result['retained_ids'], result['dedup_decisions'] = kept, decisions
        result['retained_x'], result['retained_y'] = x.tolist(), y.tolist()
        result['retained_hashes'] = {'x': array_sha256(x), 'y': array_sha256(y)}
    with timed(t, 'spatial_counts'):
        cs, ct = marginal_cell_counts(x), marginal_cell_counts(y)
        weights = 1 / np.sqrt(cs.astype(np.float64) * ct.astype(np.float64))
        result['cell_counts'] = {'source': cs.tolist(), 'target': ct.tolist()}
    graphs = []
    for cid in inputs['candidate_order'].tolist():
        h = inputs['candidates'][cid]
        with timed(t, 'projection'):
            projected = project(h,x)
        with timed(t, 'matched_residuals'):
            r = matched_residuals(projected,y)
        with timed(t, 'graph_construction'):
            a = pairwise_radius_A(projected,y,tau)
        with timed(t, 'graph_audit'):
            graph_hash, comps = array_sha256(a), _components(a)
        with timed(t, 'cheap_reductions'):
            inside = r <= tau
            n, support, edges = len(x), int(inside.sum()), int(a.sum())
            if int(np.trace(a)) != support:
                raise ArithmeticError('trace A differs from matched support')
            ratio_key = [support*n,edges] if edges else [0,1]
            row = {'canonical_id': cid, 'n': n, 'N': support, 'S': edges,
                   'H_sha256': array_sha256(h), 'A_sha256': graph_hash,
                   'graph_components': comps,
                   'raw_key': support,
                   'msac_key': float(np.maximum(0,1-np.square(r[inside]/tau)).sum()),
                   'cell_key': float(weights[inside].sum()), 'excess_key': n*support-edges,
                   'excess': support-edges/n, 'mu': edges/n,
                   'ratio_key': ratio_key, 'ratio': float(Fraction(*ratio_key)),
                   'tail': {'status': 'not_attempted'}}
            result['hypotheses'].append(row)
            graphs.append(a)
    with timed(t, 'selection'):
        result['selections'].update(select_controls(result['hypotheses']))
        result['controls_complete'] = True
    for row, a in zip(result['hypotheses'], graphs):
        result['tail_attempted'] += 1
        row['tail'] = {'status': 'running'}
        try:
            with timed(t, 'integer_tail'):
                answer = tail_function(a.tolist(), row['N'], operation_limit=operation_limit)
                if not (1 <= answer.numerator <= answer.denominator):
                    raise ArithmeticError('observed assignment tail must lie in [1/n!,1]')
                row['tail'] = {'status': 'completed', 'numerator': str(answer.numerator),
                               'denominator': str(answer.denominator),
                               'operations': answer.operations, 'components': list(answer.components)}
        except UnsupportedBoard as exc:
            row['tail'] = {'status': 'unsupported', 'reason': exc.reason,
                           'operations': exc.operations}
            result['status'], result['reason'] = 'tail_failed', exc.reason
            return
    with timed(t, 'selection'):
        keys = [Fraction(int(r['tail']['numerator']),int(r['tail']['denominator']))
                for r in result['hypotheses']]
        best = min(keys)
        ties = [r['canonical_id'] for r,k in zip(result['hypotheses'],keys) if k == best]
        result['selections']['tail'] = {'selected_id': ties[0], 'tied_ids': ties}
        result['status'] = 'completed'


def execute_case(builder, *, seconds=2.5, tau=3., operation_limit=2000000,
                 tail_function=exact_permutation_tail):
    """Builder is executed inside the timer. Caller persists result after timer closes."""
    result = {'status': 'running', 'controls_complete': False, 'tail_attempted': 0,
              'hypotheses': [], 'selections': {'tail': {'selected_id': None, 'tied_ids': []}},
              'timings': {key: 0. for key in TIMINGS}}
    labels = None
    start = time.perf_counter()
    try:
        with deadline(seconds):
            with timed(result['timings'], 'generation'):
                inputs, labels = builder()
            score_inputs(inputs, result, tau=tau, operation_limit=operation_limit,
                         tail_function=tail_function)
    except CaseDeadline as exc:
        result['status'] = 'tail_failed' if result['controls_complete'] else 'incomplete'
        result['reason'] = str(exc)
        result['deadline_expired'] = True
        result['selections']['tail'] = {'selected_id': None, 'tied_ids': []}
        for row in result['hypotheses']:
            if row['tail']['status'] == 'running':
                row['tail'] = {'status': 'timeout'}
    except Exception as exc:
        result['status'] = 'invalid'
        result['reason'] = type(exc).__name__ + ': ' + str(exc)
        result['selections']['tail'] = {'selected_id': None, 'tied_ids': []}
    result['active_seconds'] = time.perf_counter() - start
    result['case_limit_seconds'] = seconds
    return result, labels
