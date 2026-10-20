"""Source-bound RGB cross-fitting screen. No third-party imports before offline guard."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

TRAIN = (("dog-agility", 25), ("planes-water", 38), ("dogs-scale", 83), ("dog-gooses", 86))
ARMS = ("native", "logit", "absolute", "prototype", "nn", "ordinary_relations", "bilateral",
        "signed_no_gram", "signed_gram", "displaced_gram", "rgb256")
OLD_KEYS = {"cache_receipt", "cache_outer", "cache_operation", "cache_worker_spec", "mapping", "data_index"}
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
DIMS = dict(logit=3, absolute=50, prototype=5, nn=5, ordinary_relations=17, bilateral=4,
            signed_no_gram=4, signed_gram=14, displaced_gram=14, rgb256=14)


def module(name, path):
    loader = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(loader)
    sys.modules[name] = value
    loader.loader.exec_module(value)
    return value


class Context:
    def __init__(self, spec_path):
        self.started = time.perf_counter()
        self.spec_path = Path(spec_path).resolve()
        preliminary = json.loads(self.spec_path.read_text())
        common_path = Path(preliminary['campaign_root']) / 'common.py'
        if hashlib.sha256(common_path.read_bytes()).hexdigest() != preliminary['source_sha256'].get(str(common_path)):
            raise ValueError('serialization source changed')
        self.c = module('rgb_common', common_path)
        self.spec, self.out, self.r, _ = self.c.start(self.spec_path, 'relative-rgb')
        self.r.update(schema='evr0034.relative-rgb-operation.v1', stage=self.spec['stage'],
                      offline_attempts=[], compatibility_probes=[], native_calls=0, image_encoder_calls=0,
                      native_model_loaded=False, dev_payload_access=False, final_test_access=False,
                      history_modified=False, outputs=[], frames=[], label_decodes=0, native_parities=0)
        self.fingerprints = dict(self.spec['source_sha256'])
        self.fingerprints[str(self.spec_path)] = self.c.sha(self.spec_path)
        self.here = Path(__file__).resolve().parent
        evaluation = Path(self.spec['evaluation_root'])
        required = {str(self.here / n) for n in ('run.py', 'features.py', 'models.py', 'metrics.py',
            'ADMISSION.md', 'SCOPE.md', 'FEATURES.md', 'MODEL.md', 'METRICS.md')}
        required |= {str(common_path), str(common_path.with_name('cache_baselines.py')), self.spec['guard_source']}
        required |= {str(evaluation / n) for n in ('evaluate_jf.py', 'evaluate_baseline.py', 'evaluate_rollout.py')}
        self.c.require(set(self.spec['source_sha256']) == required, 'exact source closure')
        for p, digest in self.fingerprints.items():
            self.c.require(self.c.sha(p) == digest, 'source/spec changed')
        self.c.require(sys.flags.no_site == 1 and sys.flags.dont_write_bytecode == 1
            and sys.version_info[:3] == (3, 12, 13)
            and Path(sys.executable).absolute() == Path(self.spec['qualified_python']).absolute()
            and all(os.environ.get(k) == '4' for k in THREADS), 'qualified runtime envelope')
        self.guard = module('rgb_guard', self.spec['guard_source'])
        self.guard.install_guard(self.r['offline_attempts'], self.r['compatibility_probes'])
        import site
        site.main()
        import torch
        import numpy as np
        from PIL import Image
        self.torch, self.np, self.Image = torch, np, Image
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        self.c.require(torch.__version__.split('+')[0] == '2.8.0' and np.__version__ == '2.2.6', 'qualified numeric runtime')
        sys.path.insert(0, str(evaluation))
        self.admission = module('rgb_admission', evaluation / 'evaluate_rollout.py')
        self.e = self.admission.e
        self.e.cv2.setNumThreads(0)
        self.features = module('rgb_features', self.here / 'features.py')
        self.models = module('rgb_models', self.here / 'models.py')
        self.metrics = module('rgb_metrics', self.here / 'metrics.py')
        self.r['runtime'] = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
            opencv=self.e.cv2.__version__, threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads())
        extra = set()
        if self.spec['stage'] in ('fit_predict', 'evaluate'):
            extra |= {f'prepared_{s}' for s in ('receipt', 'outer', 'operation', 'worker_spec')}
        if self.spec['stage'] == 'evaluate':
            extra |= {f'fitpred_{s}' for s in ('receipt', 'outer', 'operation', 'worker_spec')}
        self.c.require(set(self.spec['bindings']) == OLD_KEYS | extra, 'exact stage bindings')
        self.b = {k: self.check(d) for k, d in self.spec['bindings'].items()}
        self.cache, _ = self.admission.admit(self.b, 'cache', self.fingerprints, legacy=True)
        self.c.require(self.cache['schema'] == 'evr0030.development.baseline-cache.v1'
            and self.cache['completed_frames'] == 386 and self.cache['completed_sequences'] == 6
            and not self.cache['offline_attempts'] and self.cache['model_initial_hash'] == self.cache['model_final_hash'],
            'complete unchanged native cache')
        self.r['bindings'] = self.spec['bindings']
        self.r['baseline_history_cost'] = dict(cache=self.spec['bindings']['cache_receipt'],
            outer=self.spec['bindings']['cache_outer'], wall_seconds=json.loads(self.b['cache_outer'].read_text())['wall_seconds'],
            native_state_hash=self.cache['model_final_hash'], scope='inherited six-parent cache including prior teacher work; not free or matched timing')

    def checkpoint(self):
        self.r['wall_seconds'] = time.perf_counter() - self.started
        self.c.save(self.out / 'receipt.json', self.r)

    def bounded(self):
        self.c.require(time.perf_counter() - self.started < 1800, 'complete operation exceeds 1800 seconds')

    def check(self, d):
        p = Path(d['path']).resolve()
        self.c.require(p.is_file() and p.stat().st_size == d['bytes'] and self.c.sha(p) == d['sha256'], 'input changed: ' + str(p))
        self.c.require(str(p) not in self.fingerprints or self.fingerprints[str(p)] == d['sha256'], 'conflicting input identity')
        self.fingerprints[str(p)] = d['sha256']
        return p

    def relative(self, root, d):
        rel = Path(d['path']); path = (root / rel).resolve()
        self.c.require(not rel.is_absolute() and '..' not in rel.parts and path.is_relative_to(root.resolve()), 'unsafe artifact path')
        return dict(d, path=str(path))

    def output(self, path, kind, **ident):
        d = dict(path=str(path.relative_to(self.out)), bytes=path.stat().st_size, sha256=self.c.sha(path), kind=kind, **ident)
        self.r['outputs'].append(d)
        return d

    def admit_prior(self, prefix, stage):
        rec, worker = self.admission.admit(self.b, prefix, self.fingerprints)
        self.c.require(rec['schema'] == 'evr0034.relative-rgb-operation.v1' and rec['stage'] == stage
            and rec['source_sha256'] == self.spec['source_sha256'] and not rec['offline_attempts']
            and not rec['dev_payload_access'] and not rec['final_test_access'], 'wrong prior stage/source')
        expected = {k: self.spec['bindings'][k] for k in worker['bindings']}
        self.c.require(rec['bindings'] == worker['bindings'] == expected, 'prior input mismatch')
        root = self.b[prefix + '_receipt'].parent
        seal_path = root / 'seal.json'
        seal = json.loads(seal_path.read_text())
        self.fingerprints[str(seal_path)] = self.c.sha(seal_path)
        self.c.require(seal['complete'] and seal['receipt_sha256'] == self.c.sha(self.b[prefix + '_receipt'])
            and seal['outputs'] == rec['outputs'] and seal['stage'] == stage
            and seal['source_sha256'] == rec['source_sha256'], 'prior seal mismatch')
        self.c.require(len(rec['outputs']) == len({d['path'] for d in rec['outputs']}), 'duplicate prior outputs')
        for d in rec['outputs']:
            self.check(self.relative(root, d))
        return rec, root


def plans(ctx):
    c = ctx.c
    mapping, index = (json.loads(ctx.b[k].read_text()) for k in ('mapping', 'data_index'))
    bases = {r['sequence']: r for r in ctx.cache['rows']}
    sources = {r['sequence']: r for r in index['sequences']}
    labels = {(r['sequence'], r['frame']): r for r in mapping['mappings']}
    inventory = {d['path']: d for d in ctx.cache['outputs']}
    c.require(len(bases) == len(ctx.cache['rows']) == len(sources) == len(index['sequences']) == 6
        and set(bases) == set(sources) == {n for n, _ in TRAIN} | {'cat-girl', 'upside-down'}
        and mapping['status'] == 'completed' and mapping['member_count'] == len(labels) == 386
        and ctx.cache['data_index']['sha256'] == c.sha(ctx.b['data_index'])
        and len(inventory) == len(ctx.cache['outputs']), 'fixed metadata coverage')
    answer = []
    for name, n in TRAIN:
        base, source = bases[name], sources[name]
        c.require(base['status'] == 'completed' and base['frames'] == base['completed_frames'] == n
            and base['role'] == source['role'] == 'TRAIN' and base['parent_id'] == source['parent_id'] == name
            and [d['frame'] for d in base['predictions']] == list(range(n))
            and [d['frame'] for d in source['rgb']] == [d['frame'] for d in source['labels']] == list(range(n)), 'TRAIN frame identities')
        for frame in range(n):
            label, pred = labels[name, frame], base['predictions'][frame]
            c.require(label['role'] == 'TRAIN' and all(label[k] == source['labels'][frame][k] for k in ('path', 'bytes', 'sha256'))
                and pred == inventory[pred['path']] and pred['sequence'] == name, 'input metadata association')
            answer.append(dict(sequence=name, parent_id=name, role='TRAIN', frame=frame, frames=n,
                shape=base['shape'], target_id=base['target_id'], label=label, rgb=source['rgb'][frame],
                native=ctx.relative(ctx.b['cache_receipt'].parent, pred),
                tensor=ctx.relative(ctx.b['cache_receipt'].parent, inventory[f'{name}/frame-{frame:05d}.pt'])))
    c.require(len(answer) == 232, 'full TRAIN raw population')
    return answer


def empty_arrays(np):
    return dict(coords=np.zeros((0, 2), dtype=np.int64), c=np.zeros((0, 3), np.float32),
        absolute_prefix=np.zeros((0, 18), np.float32), proto=np.zeros((0, 5), np.float32), nn=np.zeros((0, 5), np.float32),
        ordinary=np.zeros((0, 17), np.float32), sym=np.zeros((0, 14), np.float32), e=np.zeros(0, np.float32),
        sym_coarse=np.zeros((0, 14), np.float32), e_coarse=np.zeros(0, np.float32), bilateral=np.zeros((0, 4), np.float32),
        sample_indices=np.zeros(0, np.int64))


def prepare(ctx):
    np, torch, c = ctx.np, ctx.torch, ctx.c
    selected = plans(ctx)
    ctx.r.update(rgb_decodes=0, tensor_loads=0, raw_frames=232, noninitial_frames=228, learned_metric_frames=224,
                 all_features_gt_free=True, coordinate_order='xy; row-major y then x', frame_count=0)
    for plan in selected:
        ctx.bounded()
        shape = tuple(plan['shape']); target = plan['target_id']; frame = plan['frame']; name = plan['sequence']
        c.require(len(shape) == 2 and min(shape) == 480 and type(target) is int and 0 < target < 255, 'shape/target')
        native_path, tensor_path = ctx.check(plan['native']), ctx.check(plan['tensor'])
        native = ctx.e._prediction(native_path, [target], shape)[0]
        item = torch.load(tensor_path, map_location='cpu', weights_only=True)
        ctx.r['tensor_loads'] += 1
        low, raw, presence = item['out']['pred_masks'], item['vision'][0], item['out']['object_score_logits']
        c.require(low.dtype == raw.dtype == presence.dtype == torch.float32
            and low.device.type == raw.device.type == presence.device.type == 'cpu'
            and tuple(low.shape) == (1, 1, 256, 256) and tuple(raw.shape) == (65536, 1, 32)
            and presence.numel() == 1 and all(bool(torch.isfinite(v).all()) for v in (low, raw, presence)), 'native tensor contract')
        score = float(presence.reshape(-1)[0])
        c.require(score > 0 or bool((low == -1024).all()), 'absence contract')
        logits = torch.nn.functional.interpolate(low, size=shape, mode='bilinear', align_corners=False)[0, 0].numpy().copy()
        c.require(np.array_equal(logits > 0, native), 'original native parity')
        ctx.r['native_parities'] += 1
        reason = 'initial' if frame == 0 else 'native_absent' if score <= 0 else 'native_empty' if not native.any() else 'native_full' if native.all() else None
        if reason:
            arrays, detail = empty_arrays(np), dict(noop_reason=reason, valid_queries=0)
        else:
            rgb_path = ctx.check(plan['rgb'])
            with ctx.Image.open(rgb_path) as im:
                rgb = np.asarray(im.convert('RGB'), dtype=np.float32) / np.float32(255)
            ctx.r['rgb_decodes'] += 1
            c.require(rgb.shape == (*shape, 3) and np.isfinite(rgb).all(), 'original RGB shape')
            rgb_tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0)
            coarse = torch.nn.functional.interpolate(torch.nn.functional.interpolate(rgb_tensor, size=(256, 256),
                mode='bilinear', align_corners=False, antialias=False), size=shape,
                mode='bilinear', align_corners=False, antialias=False)[0].permute(1, 2, 0).numpy().copy()
            arrays, detail = ctx.features.construct(rgb, logits, native, coarse)
            del rgb_tensor, rgb, coarse
        coords = arrays['coords']; q = len(coords)
        c.require(coords.dtype == np.int64 and coords.shape == (q, 2) and q != 1, 'coordinate contract')
        if q:
            xs, ys = coords.T
            c.require(((xs >= 0) & (xs < shape[1]) & (ys >= 0) & (ys < shape[0])).all()
                and np.all(np.diff(ys * shape[1] + xs) > 0), 'row-major unique in-image coordinates')
            feature = raw[:, 0].T.reshape(1, 32, 256, 256).contiguous()
            fine = torch.nn.functional.interpolate(feature, size=shape, mode='bilinear', align_corners=False)[0]
            fine_samples = fine[:, ys, xs].T.contiguous().numpy().copy()
            absolute = np.concatenate((arrays.pop('absolute_prefix'), fine_samples), axis=1)
            del feature, fine, fine_samples
        else:
            arrays.pop('absolute_prefix'); absolute = np.zeros((0, 50), np.float32)
        arrays['absolute'] = absolute
        displaced = arrays['sym'].copy()
        if q:
            displaced[:, 4:] = np.roll(displaced[:, 4:], q // 2, axis=0)
        arrays['displaced'] = displaced
        arrays.update(logits=logits, native=native)
        c.require(all(np.isfinite(v).all() for v in arrays.values()), 'finite preparation arrays')
        sample = arrays['sample_indices']
        c.require(sample.dtype == np.int64 and len(sample) <= 1024 and
            (len(sample) == 0 or (sample[0] >= 0 and sample[-1] < q and (np.diff(sample) > 0).all())), 'sample indices')
        path = ctx.out / f'{name}-frame-{frame:05d}.npz'
        np.savez(path, **arrays)
        descriptor = ctx.output(path, 'causal-features', sequence=name, frame=frame)
        ctx.r['frames'].append(dict(sequence=name, parent_id=name, role='TRAIN', frame=frame, frames=plan['frames'],
            shape=list(shape), target_id=target, native_presence=score, noop_reason=reason,
            feature=descriptor, native=plan['native'], rgb=plan['rgb'], label=plan['label'], tensor=plan['tensor'],
            query_count=q, sample_count=len(sample), detail=detail))
        ctx.r['frame_count'] += 1
        ctx.checkpoint()
        del item, low, raw, presence, arrays, logits, native, absolute, displaced
    c.require(ctx.r['native_parities'] == ctx.r['frame_count'] == 232 and ctx.r['label_decodes'] == 0, 'complete GT-free preparation')


def read_features(ctx, prepared_root, row):
    path = ctx.check(ctx.relative(prepared_root, row['feature']))
    with ctx.np.load(path, allow_pickle=False) as loaded:
        a = {k: loaded[k] for k in loaded.files}
    q = row['query_count']; np = ctx.np
    required = {'coords', 'c', 'absolute', 'proto', 'nn', 'ordinary', 'sym', 'e', 'sym_coarse', 'e_coarse',
                'bilateral', 'sample_indices', 'displaced', 'logits', 'native'}
    ctx.c.require(set(a) == required and a['coords'].shape == (q, 2)
        and a['native'].shape == a['logits'].shape == tuple(row['shape'])
        and a['native'].dtype == np.bool_ and a['logits'].dtype == np.float32
        and a['coords'].dtype == a['sample_indices'].dtype == np.int64
        and np.array_equal(a['logits'] > 0, a['native'])
        and len(a['sample_indices']) == row['sample_count'], 'prepared array contract')
    for name, dim in [('c', 3), ('absolute', 50), ('proto', 5), ('nn', 5), ('ordinary', 17),
                       ('sym', 14), ('sym_coarse', 14), ('bilateral', 4), ('displaced', 14)]:
        ctx.c.require(a[name].shape == (q, dim) and a[name].dtype == np.float32 and np.isfinite(a[name]).all(), 'feature shape/dtype')
    for name in ('e', 'e_coarse'):
        ctx.c.require(a[name].shape == (q,) and a[name].dtype == np.float32 and np.isfinite(a[name]).all()
            and (np.abs(a[name]) <= 1).all(), 'bounded evidence')
    return a


def arm_arrays(a):
    return dict(logit=a['c'], absolute=a['absolute'], prototype=a['proto'], nn=a['nn'], ordinary_relations=a['ordinary'],
        bilateral=a['bilateral'], signed_no_gram=a['sym'][:, :4], signed_gram=a['sym'],
        displaced_gram=a['displaced'], rgb256=a['sym_coarse'])


def fit_predict(ctx):
    np, c = ctx.np, ctx.c
    prepared, prepared_root = ctx.admit_prior('prepared', 'prepare')
    c.require(prepared['frame_count'] == prepared['native_parities'] == len(prepared['frames']) == 232
        and prepared['label_decodes'] == 0 and prepared['all_features_gt_free'], 'complete preparation')
    rows = prepared['frames']
    c.require([(r['sequence'], r['frame']) for r in rows] == [(n, f) for n, count in TRAIN for f in range(count)], 'prepared identities')
    data = []
    ctx.r.update(fit_failures=[], fits=[], folds=[], completed_fits=0, prediction_frames=0, arm_predictions=0,
        arms=list(ARMS), raw_frames=232, noninitial_frames=228, primary_interior_frames=224,
        labels_used_only_from_other_parents_per_fold=True, held_parent_scoring_performed=False)
    for row in rows:
        ctx.bounded()
        if row['frame'] == 0:
            continue
        a = read_features(ctx, prepared_root, row)
        ix = a['sample_indices']; xy = a['coords'][ix]
        xs, ys = xy.T if len(ix) else (np.zeros(0, np.int64), np.zeros(0, np.int64))
        if len(ix):
            truth = ctx.e._labels(ctx.check(row['label']))
            ctx.r['label_decodes'] += 1
            c.require(truth.shape == tuple(row['shape']), 'fit label original shape')
            target = (truth[ys, xs] == row['target_id']).astype(np.float64)
            void = int(np.count_nonzero(truth[ys, xs] == 255))
            del truth
        else:
            target, void = np.zeros(0, np.float64), 0
        data.append(dict(sequence=row['sequence'], frame=row['frame'], frames=row['frames'], count=len(ix),
            x={k: v[ix].astype(np.float64) for k, v in arm_arrays(a).items()},
            logits=a['logits'][ys, xs].astype(np.float64), target=target,
            e=a['e'][ix].astype(np.float64), e_coarse=a['e_coarse'][ix].astype(np.float64), void_samples=void))
        ctx.r['frames'].append(dict(sequence=row['sequence'], frame=row['frame'], sample_count=len(ix), void_samples_as_background=void))
        ctx.checkpoint()
        del a
    c.require(len(data) == 228, 'all noninitial fit identities retained')
    models = {}
    for held, _ in TRAIN:
        ctx.bounded()
        train = [d for d in data if d['sequence'] != held]
        c.require({d['sequence'] for d in train} == {n for n, _ in TRAIN} - {held}, 'cross-fit training parents')
        xraw = np.concatenate([d['x']['absolute'][:, 18:] for d in train])
        mean = xraw.mean(axis=0) if len(xraw) else np.zeros(32, np.float64)
        std = np.maximum(xraw.std(axis=0), 1e-6) if len(xraw) else np.ones(32, np.float64)
        norm = dict(mean=mean.tolist(), std=std.tolist(), source_sample_count=len(xraw),
            scope='unweighted population of fixed fitting samples from three other parents; no held-parent statistics', std_floor=1e-6)
        fold = dict(held_parent=held, training_parents=[n for n, _ in TRAIN if n != held],
            training_frames=len(train), sample_count=sum(d['count'] for d in train), normalization=norm,
            training_identities=[dict(sequence=d['sequence'], frame=d['frame'], samples=d['count']) for d in train])
        ctx.r['folds'].append(fold)
        L = np.concatenate([d['logits'] for d in train]); Y = np.concatenate([d['target'] for d in train])
        W = np.concatenate([np.full(d['count'], 1 / (3 * (d['frames'] - 1) * d['count']), np.float64)
                            if d['count'] else np.zeros(0, np.float64) for d in train])
        c.require(float(W.sum()) <= 1 + 1e-12 and np.isfinite(W).all(), 'parent/frame weights')
        models[held] = dict(normalization=norm, arms={})
        for arm in ARMS[1:]:
            ctx.bounded()
            X = np.concatenate([d['x'][arm] for d in train])
            if arm == 'absolute':
                X[:, 18:] = (X[:, 18:] - mean) / std
            evidence = np.concatenate([d['e_coarse' if arm == 'rgb256' else 'e'] for d in train]) if arm in ('signed_no_gram', 'signed_gram', 'displaced_gram', 'rgb256') else None
            before = time.perf_counter()
            try:
                model, trace = ctx.models.fit_model(arm, X, L, Y, W, evidence=evidence)
                state = 'completed'
            except ctx.models.FitError as exc:
                model = dict(schema='evr0034.native-fallback.v1', name=arm, failed=True, reason=str(exc))
                trace = exc.closure_trace
                state = 'failed'
                ctx.r['fit_failures'].append(dict(held_parent=held, arm=arm, error_type=type(exc).__name__, reason=str(exc),
                    partial_model=exc.partial_model, reason_code=exc.reason_code))
            elapsed = time.perf_counter() - before
            model_path = ctx.out / f'{held}-{arm}-terminal.json'; c.save(model_path, model)
            trace_path = ctx.out / f'{held}-{arm}-trace.json'; c.save(trace_path, trace)
            descriptor = ctx.output(model_path, 'terminal-readout', held_parent=held, arm=arm)
            trace_d = ctx.output(trace_path, 'complete-optimizer-closure-trace', held_parent=held, arm=arm)
            models[held]['arms'][arm] = model
            ctx.r['fits'].append(dict(held_parent=held, arm=arm, status=state, wall_seconds=elapsed,
                model=descriptor, trace=trace_d, training_frames=len(train), sample_count=len(Y), weight_sum=float(W.sum())))
            ctx.r['completed_fits'] += 1
            ctx.checkpoint()
        del X, L, Y, W, evidence, xraw
    c.require(ctx.r['completed_fits'] == 40 and len(models) == 4, 'all40 terminal fits attempted')
    del data
    terminals_path = ctx.out / 'terminal-models.json'; c.save(terminals_path, models)
    ctx.r['terminal_models'] = ctx.output(terminals_path, 'all-fold-terminal-models')
    ctx.r['terminal_models_sealed_before_prediction'] = True
    ctx.r['prediction_rows'] = []
    for row in rows:
        ctx.bounded()
        a = read_features(ctx, prepared_root, row)
        native = a['native']; coords = a['coords']; q = len(coords)
        xs, ys = coords.T if q else (np.zeros(0, np.int64), np.zeros(0, np.int64))
        masks = np.repeat(native[None], len(ARMS), axis=0)
        inputs = arm_arrays(a); fold = models[row['sequence']]
        normal = fold['normalization']; mean = np.asarray(normal['mean']); std = np.asarray(normal['std'])
        detail = {}
        for i, arm in enumerate(ARMS[1:], 1):
            model = fold['arms'][arm]
            if q and not model.get('failed', False):
                X = inputs[arm].astype(np.float64)
                if arm == 'absolute':
                    X[:, 18:] = (X[:, 18:] - mean) / std
                evidence = a['e_coarse' if arm == 'rgb256' else 'e'] if arm in ('signed_no_gram', 'signed_gram', 'displaced_gram', 'rgb256') else None
                delta = ctx.models.predict_delta(model, X, evidence=evidence)
                c.require(delta.shape == (q,) and np.isfinite(delta).all() and (np.abs(delta) <= 4).all(), 'finite bounded correction')
                masks[i, ys, xs] = a['logits'][ys, xs].astype(np.float64) + delta > 0
                maximum = float(np.max(np.abs(delta)))
            else:
                maximum = 0.0
            changed = int(np.count_nonzero(masks[i] != native))
            c.require(changed <= q and (q or changed == 0), 'correction outside sites')
            detail[arm] = dict(changed_pixels=changed, maximum_absolute_correction=maximum,
                learned_calls=bool(q and not model.get('failed', False)), fit_failed_native_fallback=bool(model.get('failed', False)))
        path = ctx.out / f'{row["sequence"]}-frame-{row["frame"]:05d}-predictions.npz'
        np.savez_compressed(path, masks=masks, arm_names=np.asarray(ARMS), object_ids=np.asarray([row['target_id']], np.int64))
        descriptor = ctx.output(path, 'all-arm-predictions', sequence=row['sequence'], frame=row['frame'])
        ctx.r['prediction_rows'].append(dict(sequence=row['sequence'], parent_id=row['parent_id'], frame=row['frame'], frames=row['frames'],
            shape=row['shape'], target_id=row['target_id'], query_count=q, sample_count=row['sample_count'],
            native_presence=row['native_presence'], noop_reason=row['noop_reason'], prediction=descriptor,
            label=row['label'], native=row['native'], feature=row['feature'], detail=detail))
        ctx.r['prediction_frames'] += 1; ctx.r['arm_predictions'] += len(ARMS)
        ctx.checkpoint()
        del a, masks, inputs, native
    c.require(ctx.r['prediction_frames'] == 232 and ctx.r['arm_predictions'] == 2552, 'complete cross-fitted outputs')
    c.require(json.loads(terminals_path.read_text()) == models, 'terminal models changed during prediction')
    ctx.r['terminal_models_unchanged'] = True


def prediction_arrays(ctx, root, row):
    np, c = ctx.np, ctx.c
    path = ctx.check(ctx.relative(root, row['prediction']))
    with np.load(path, allow_pickle=False) as loaded:
        c.require(set(loaded.files) == {'masks', 'arm_names', 'object_ids'}, 'exact prediction archive keys')
        masks, names, ids = loaded['masks'], loaded['arm_names'], loaded['object_ids']
    c.require(masks.dtype == np.bool_ and masks.shape == (11, *row['shape']) and names.tolist() == list(ARMS)
        and ids.dtype == np.int64 and ids.tolist() == [row['target_id']], 'all-arm prediction shape/order/target')
    return masks


def evaluate(ctx):
    np, c = ctx.np, ctx.c
    prepared, prepared_root = ctx.admit_prior('prepared', 'prepare')
    fitted, fit_root = ctx.admit_prior('fitpred', 'fit_predict')
    rows = fitted['prediction_rows']
    c.require(fitted['completed_fits'] == 40 and len(fitted['fits']) == 40
        and fitted['prediction_frames'] == len(rows) == 232 and fitted['arm_predictions'] == 2552
        and fitted['terminal_models_sealed_before_prediction'] and fitted['terminal_models_unchanged']
        and not fitted['held_parent_scoring_performed']
        and [(r['sequence'], r['frame']) for r in rows] == [(n, f) for n, count in TRAIN for f in range(count)], 'complete fixed cross-fit')
    prior_rows = {(r['sequence'], r['frame']): r for r in prepared['frames']}
    # Every sealed mask and no-op checked before this metric stage decodes labels.
    for row in rows:
        ctx.bounded()
        original = prior_rows[row['sequence'], row['frame']]
        for key in ('frames', 'shape', 'target_id', 'query_count', 'sample_count', 'label', 'native', 'feature'):
            c.require(row[key] == original[key], 'prediction/preparation identity')
        masks = prediction_arrays(ctx, fit_root, row)
        native = ctx.e._prediction(ctx.check(row['native']), [row['target_id']], tuple(row['shape']))[0]
        c.require(np.array_equal(masks[0], native), 'sealed native parity')
        a = read_features(ctx, prepared_root, original)
        allowed = np.zeros(tuple(row['shape']), bool)
        if row['query_count']:
            x, y = a['coords'].T; allowed[y, x] = True
        for i, arm in enumerate(ARMS[1:], 1):
            changed = masks[i] != native
            c.require(not (changed & ~allowed).any() and int(changed.sum()) == row['detail'][arm]['changed_pixels'], 'mask changed outside declared support')
            if row['detail'][arm]['fit_failed_native_fallback']:
                c.require(not changed.any(), 'failed fit not native fallback')
        ctx.r['native_parities'] += 1
        del a, masks
    ctx.r.update(predictions_sealed_before_metric_labels=True, verified_frames=232, verified_arm_predictions=2552,
                 fit_failures=fitted['fit_failures'])
    ctx.checkpoint()
    measurements = []
    for row in rows:
        ctx.bounded()
        masks = prediction_arrays(ctx, fit_root, row)
        truth = ctx.e._labels(ctx.check(row['label'])); ctx.r['label_decodes'] += 1
        c.require(truth.shape == tuple(row['shape']), 'metric label shape')
        target = truth == row['target_id']
        arms = {}
        for name, mask in zip(ARMS, masks):
            j, f = ctx.e.jf(mask, target); arms[name] = dict(J=j, F=f, JF=(j + f) / 2)
        measurements.append(dict(sequence=row['sequence'], frame=row['frame'], frames=row['frames'],
            target_id=row['target_id'], arms=arms, query_count=row['query_count'], sample_count=row['sample_count'],
            void_pixels_as_background=int(np.count_nonzero(truth == 255)), detail=row['detail']))
        ctx.checkpoint()
    report = ctx.metrics.build_report(measurements, fitted['fit_failures'])
    ref_path = ctx.check(ctx.spec['baseline_reference']); reference = json.loads(ref_path.read_text())['train']
    observed = report['secondary']
    for parent in observed['parents']:
        old = next(x for x in reference['parents'] if x['parent_id'] == parent['sequence'])
        c.require(all(math.isclose(parent['arms']['native'][k], old['native'][k], abs_tol=1e-12, rel_tol=0) for k in ('J', 'F', 'JF')),
                  'prior native secondary parent parity')
    c.require(all(math.isclose(observed['mean']['native'][k], reference['native'][k], abs_tol=1e-12, rel_tol=0) for k in ('J', 'F', 'JF')),
              'prior native secondary aggregate parity')
    report.update(native_secondary_reference=ctx.spec['baseline_reference'], native_secondary_parity=True,
        boundary_fraction=ctx.e.BOUNDARY_FRACTION, official_metric_reference_commit=ctx.e.REFERENCE,
        void_policy='255 background; binary original target ID', baseline_history_cost=ctx.r['baseline_history_cost'])
    path = ctx.out / 'evaluation.json'; c.save(path, report)
    ctx.r['evaluation'] = ctx.output(path, 'complete-cross-fit-metrics')
    ctx.r['continuation_passed'] = report['decision']['passed']
    ctx.r['frame_count'] = len(measurements)
    c.require(ctx.r['label_decodes'] == 232, 'all metric labels retained')


def run(spec_path):
    ctx = None
    try:
        ctx = Context.__new__(Context)
        ctx.__init__(spec_path)
        with ctx.guard.capture_warnings(ctx.r):
            {'prepare': prepare, 'fit_predict': fit_predict, 'evaluate': evaluate}[ctx.spec['stage']](ctx)
            ctx.bounded()
            for path, digest in ctx.fingerprints.items():
                ctx.c.require(ctx.c.sha(path) == digest, 'input/source changed during operation')
            ctx.c.require(not ctx.r['offline_attempts'], 'unexpected network attempt')
            ctx.r['status'] = 'completed'
        ctx.checkpoint()
        ctx.c.save(ctx.out / 'seal.json', dict(schema='evr0034.stage-seal.v1', complete=True, stage=ctx.spec['stage'],
            receipt_sha256=ctx.c.sha(ctx.out / 'receipt.json'), source_sha256=ctx.spec['source_sha256'],
            outputs=ctx.r['outputs']))
    except BaseException as exc:
        if ctx is not None and hasattr(ctx, 'r'):
            ctx.r.update(status='failed', error_type=type(exc).__name__, error=str(exc))
            ctx.checkpoint()
        raise


if __name__ == '__main__':
    run(sys.argv[1])
