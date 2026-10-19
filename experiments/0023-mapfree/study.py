"""Prospective small-cohort method driver. No I/O, labels, downloads or model loads.

Inputs are admitted by cache_inputs; backend is an explicitly injected, pinned
native estimator. Every geometric failure retains the query denominator. Other
exceptions are technical failures and must make the enclosing operation fail.
"""
import time
import numpy as np
import field_ba as field
import geometry_ba as geometry
import covariance
import conditional

ARMS = ('geometry_multiview', 'joint_field', 'hessian', 'scalar_deletion',
        'residual_scalar', 'directional')
SOLVER_OPTIONS = dict(max_accepted=30, max_evaluations=100)


def serial(value):
    if isinstance(value, np.ndarray):
        if not np.isfinite(value).all():
            raise ValueError('nonfinite result cannot be sealed')
        return value.tolist()
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError('nonfinite scalar result cannot be sealed')
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


class RecordedBackend:
    def __init__(self, backend):
        self.backend = backend
        self.calls = []
        self.scope = None

    def _call(self, kind, method, args):
        start = time.perf_counter()
        row = dict(scope=self.scope, kind=kind, input_count=len(args[0]),
                   options=serial(args[-1]), status='running')
        self.calls.append(row)
        try:
            result = method(*args)
            row['status'] = 'completed'
            return result
        except Exception as exc:
            row.update(status='failed', error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            row['seconds'] = time.perf_counter()-start

    def estimate_relative_pose(self, *args):
        return self._call('relative', self.backend.estimate_relative_pose, args)

    def estimate_absolute_pose(self, *args):
        return self._call('absolute', self.backend.estimate_absolute_pose, args)


def edge_summary(edge):
    return dict(i=edge['i'], j=edge['j'], ids_i=edge['ids_i'], ids_j=edge['ids_j'],
                match_count=len(edge['ids_i']))


def graph_summary(graph):
    return dict(tracks=graph['tracks'], retained_edges=[edge_summary(e) for e in graph['retained_edges']],
                conflict_count=graph['conflict_count'], duplicate_count=graph['duplicate_count'],
                observation_count=graph['observation_count'], match_count=graph['match_count'])


def field_summary(fit):
    return {k: fit[k] for k in ('state', 'objective', 'diagnostics', 'converged',
                              'termination_reason', 'accepted_steps', 'evaluations', 'seconds')}


def pose_information(j, pose_columns, translation):
    """Local identifiable rotation/direction; translation length may remain free."""
    cols = np.asarray(pose_columns, dtype=int)
    info = covariance.marginal_information(j, cols)
    norm = float(np.linalg.norm(translation))
    if not np.isfinite(norm) or norm <= 1e-6:
        return dict(observable=False, rank=info['rank'], reason='undefined translation direction')
    unit = translation / norm
    functional = np.zeros((6, 6)); functional[:3, :3] = np.eye(3)
    functional[3:, 3:] = (np.eye(3)-np.outer(unit, unit))/norm
    product = covariance.mm(functional, info['null'])
    ratios = [float(np.linalg.norm(product[s:s+3])/np.linalg.norm(functional[s:s+3])) for s in (0,3)]
    return dict(observable=max(ratios) <= 1e-8, rank=info['rank'], pose_null_ratio=max(ratios),
                rotation_null_ratio=ratios[0], direction_null_ratio=ratios[1])


def fallback(reason, native):
    return dict(status='fallback' if native is not None else 'failed', reason=reason,
                pose=None if native is None else native['pose'],
                fallback='native_E_reference_filtered_query' if native is not None else '180_degree_error')


def outcome(pose, observable, native, detail):
    result = dict(status='completed', pose=pose, detail=detail)
    if not observable['observable']:
        result = dict(fallback('locally unobservable pose direction', native), detail=detail)
    result['observability'] = observable
    return result


def fit_reference(images, raw_edges, backend, scope, options):
    backend.scope = scope
    began = time.perf_counter()
    graph = geometry.build_tracks(images, raw_edges)
    record = dict(scope=scope, image_ids=sorted(images), input_edges=[edge_summary(e) for e in raw_edges],
                  graph=graph_summary(graph))
    try:
        initial = geometry.initialize_reference(images, raw_edges, backend)
    except geometry.InitializationFailure as exc:
        record.update(status='initialization_failure', reason=str(exc), seconds=time.perf_counter()-began)
        return None, None, graph, record
    if not initial['depth_scale_available']:
        record.update(status='field_scale_unavailable', initialization=initial['receipt'],
                      reason='anchor depth cannot align the shared field gauge', seconds=time.perf_counter()-began)
        return None, initial, graph, record
    fit = field.fit_field(images, graph['retained_edges'], initial['poses'], options)
    record.update(status='completed', initialization=initial['receipt'], fit=field_summary(fit),
                  seconds=time.perf_counter()-began)
    return fit, initial, graph, record


def geometry_query(images, graph, reference_geometry, initial, query, pair_pose, options):
    began = time.perf_counter()
    if reference_geometry is None:
        return None, dict(observable=False, reason='geometry reference unavailable'), dict(attempts=[])
    starts = [reference_geometry['poses']]
    if pair_pose is not None:
        r, t = pair_pose; base = starts[0][query][1]
        if np.linalg.norm(t) > 1e-6 and np.linalg.norm(base) > 1e-6:
            second = dict(starts[0]); second[query] = (r, t*np.linalg.norm(base)/np.linalg.norm(t))
            starts.append(second)
    attempts = []; summaries = []
    for poses in starts:
        try:
            fitted = geometry.fit_geometry(images, graph, dict(initial, poses=poses), options)
        except geometry.InitializationFailure as exc:
            attempts.append(None); summaries.append(dict(status='initialization_failure',reason=str(exc)))
            continue
        attempts.append(fitted)
        summaries.append(dict(status='completed',**{k:fitted[k] for k in
            ('poses','cost','invalid_count','capped_count','observation_count','baseline_norm',
             'converged','termination_reason','accepted_steps','evaluations','seconds')}))
    valid = [k for k,a in enumerate(attempts) if a is not None]
    if not valid:
        return None, dict(observable=False,reason='no triangulated query tracks'), dict(
            attempts=summaries,selected=None,seconds=time.perf_counter()-began)
    selected = min(valid, key=lambda k:(attempts[k]['cost'], k)); fit = attempts[selected]
    observability = geometry.pose_observability(fit, query)
    detail = dict(selected=selected, seconds=time.perf_counter()-began,
                  attempts=summaries)
    return fit['poses'][query], observability, detail


def run_scene(scene, backend, options=None, emit=None):
    """All32 references, all8 deletions, eight independent one-query updates."""
    start = time.perf_counter(); options = dict(SOLVER_OPTIONS if options is None else options)
    images = scene['images']; references = scene['reference_edges']; queries = scene['query_edges']
    expected = {'A0'} | {'B'+str(i) for i in range(1,9)} | {'C'+str(i) for i in range(1,9)}
    if set(images) != expected or len(references) != 32 or len(queries) != 8:
        raise ValueError('frozen17-image32-reference8-query population required')
    if {e['j'] for e in queries} != {'B'+str(i) for i in range(1,9)} or any(e['i'] != 'A0' for e in queries):
        raise ValueError('query graph mismatch')
    backend = backend if isinstance(backend, RecordedBackend) else RecordedBackend(backend)
    def progress(kind, payload):
        if emit is not None: emit(kind, serial(payload))
    full, initial, full_graph, full_record = fit_reference(images, references, backend, scene['scene']+'/full', options)
    progress('reference', full_record)
    folds, fold_records = [], []
    for k in range(1,9):
        removed = 'C'+str(k)
        fold_images = {i:v for i,v in images.items() if i != removed}
        fold_edges = [e for e in references if removed not in (e['i'],e['j'])]
        fit, _, _, record = fit_reference(fold_images, fold_edges, backend, scene['scene']+'/delete_'+removed, options)
        folds.append(fit); fold_records.append(record); progress('reference', record)
    reference_geometry = None
    geometry_record = dict(status='initialization_failure',reason='full reference initialization unavailable')
    if initial is not None:
        try:
            reference_geometry = geometry.fit_geometry(images, full_graph, initial, options)
            geometry_record = dict(status='completed',**{k:reference_geometry[k] for k in
                ('poses','points','cost','invalid_count','capped_count','observation_count','baseline_norm',
                 'converged','termination_reason','accepted_steps','evaluations','seconds')})
        except geometry.InitializationFailure as exc:
            geometry_record = dict(status='initialization_failure',reason=str(exc))
    progress('geometry_reference',geometry_record)
    jac = field.jacobian_at(full).toarray() if full is not None else None
    rows = []
    for raw_query in sorted(queries,key=lambda e:int(e['j'][1:])):
        began = time.perf_counter(); query_id = raw_query['j']
        graph = geometry.build_tracks(images, [raw_query], frozen_reference=full_graph)
        retained = [e for e in graph['retained_edges'][len(full_graph['retained_edges']):]
                    if e['i'] == 'A0' and e['j'] == query_id]
        if len(retained) > 1: raise ValueError('one query edge required')
        edge = retained[0] if retained else dict(i='A0',j=query_id,ids_i=np.empty(0,dtype=int),
                                                 ids_j=np.empty(0,dtype=int),scores=np.empty(0))
        backend.scope = scene['scene']+'/query_'+query_id
        native = None; native_reason = None
        try:
            native = geometry.estimate_pair(images, edge, backend)
        except geometry.InitializationFailure as exc:
            native_reason = str(exc)
        row = dict(scene=scene['scene'],query=query_id,selected_query=edge_summary(raw_query),
                   retained_query=edge_summary(edge), query_graph=graph_summary(graph),
                   native_E=({**native,'status':'completed'} if native is not None else
                             dict(status='failed',reason=native_reason,pose=None,failure_error_degrees=180)),arms={})
        pair = None if native is None else native['pose']
        if initial is None:
            row['arms']['geometry_multiview'] = fallback('geometric reference initialization unavailable',native)
        else:
            pose, observable, detail = geometry_query(images, graph, reference_geometry, initial, query_id, pair, options)
            row['arms']['geometry_multiview'] = outcome(pose,observable,native,detail)
        if full is None:
            row['arms'].update({name:fallback('full field reference initialization or depth-scale unavailable',native)
                               for name in ARMS if name!='geometry_multiview'})
        else:
            if np.linalg.norm(full['poses'][query_id][1]) <= 1e-6:
                row['arms'].update({name:fallback('undefined reference translation direction',native)
                                    for name in ARMS if name!='geometry_multiview'})
                row['seconds'] = time.perf_counter()-began
                rows.append(row); progress('query',row)
                continue
            joint = conditional.fit_joint(full,edge,pair,options)
            chosen = joint['attempts'][joint['selected']]
            observable = pose_information(field.jacobian_at(chosen).toarray(),
                                           chosen['problem'].layout[query_id]['pose'], joint['pose'][1])
            detail = dict(selected=joint['selected'],seconds=joint['seconds'],
                          attempts=[field_summary(a) for a in joint['attempts']])
            row['arms']['joint_field'] = outcome(joint['pose'],observable,native,detail)
            keep = full['problem'].query_columns(query_id)
            information = covariance.marginal_information(jac,keep)
            residual_factor, alpha = covariance.residual_inflation_factor(information,full['block_errors'])
            factors = dict(hessian=information['precision_factor'], residual_scalar=residual_factor)
            row['reference_information'] = {k:information[k] for k in ('rank','dimension','basis','singular','null','rank_threshold')}
            row['residual_alpha'] = alpha
            if all(f is not None for f in folds):
                deltas = np.stack([field.state_delta(full,f)[keep] for f in folds])
                ai, bi = full['problem'].images['A0'], full['problem'].images[query_id]
                a, b = edge['ids_i'], edge['ids_j']
                shape = covariance.shape_derivative(ai['phi'][a],bi['phi'][b],full['theta']['A0'],full['theta'][query_id],
                                                     ai['depth_valid'][a],bi['depth_valid'][b])
                controls = covariance.covariance_controls(information,deltas,shape)
                factors.update(scalar_deletion=controls['precision_factors']['scalar_deletion'],
                               directional=controls['precision_factors']['directional'])
                row['deletion'] = dict(deltas=deltas,**{k:v for k,v in controls.items() if k!='precision_factors'})
            else:
                for name in ('scalar_deletion','directional'):
                    row['arms'][name] = fallback('not all eight deletion fits available',native)
            for name,factor in factors.items():
                result = conditional.fit_conditional(full,edge,factor,pair,options)
                detail = {k:result[k] for k in ('x','attempts','selected','seconds','information_count')}
                row['arms'][name] = outcome(result['pose'],result['observability'],native,detail)
            if set(row['arms']) != set(ARMS): raise ValueError('missing estimator outcome')
        row['seconds'] = time.perf_counter()-began
        rows.append(row); progress('query',row)
    result = dict(schema='evr0023.scene-prediction.v1',scene=scene['scene'],status='completed',
                  reference=full_record,deletions=fold_records,geometry_reference=geometry_record,queries=rows,
                  native_calls=[c for c in backend.calls if c['scope'].startswith(scene['scene']+'/')],
                  seconds=time.perf_counter()-start,solver_options=options,
                  ground_truth_access=False,convergence_policy='finite retained accepted iterates; flags preserved')
    return serial(result)
