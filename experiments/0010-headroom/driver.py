"""Prospective 0010 flow. Only explicit injected artificial fixtures have run."""
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np


class Invalid(ValueError): pass


def require(ok, message):
    if not ok: raise Invalid(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def record(path, value, *, exclusive=False):
    if exclusive:
        with Path(path).open('x') as f: json.dump(value, f, indent=2, allow_nan=False)
    else:
        tmp = Path(str(path) + '.pending')
        with tmp.open('w') as f: json.dump(value, f, indent=2, allow_nan=False)
        os.replace(tmp, path)


def selected_names(train, excluded):
    require(len(train) == len(set(train)) and len(excluded) == len(set(excluded)) == 8, 'name uniqueness')
    require(set(excluded) <= set(train), 'capability not in train')
    names = sorted(set(train) - set(excluded), key=lambda n: (hashlib.sha256(('0010-stata-headroom-v1|' + n).encode()).digest(), n))[:128]
    require(len(names) == 128, 'fewer than 128 new names')
    return names


def ids_for(names):
    return [hashlib.sha256(('0010-stata-headroom-id-v1|' + n).encode()).hexdigest() for n in names]


def run(output, marker, operations, headroom, clock=time.perf_counter):
    """No resumption/replacement. Runtime operations contain no label-fit interface.

    Operations are dependency-injected solely to permit artificial software tests;
    the future CLI constructs only the source-bound Native implementation.
    """
    out, marker = Path(output), Path(marker)
    require(not out.exists() and not out.is_symlink(), 'fresh output required')
    # Marker belongs to the fixed asset root: changing output cannot repeat a run.
    record(marker, {'schema': 1, 'output': str(out), 'status': 'started'}, exclusive=True)
    out.mkdir(parents=True, exist_ok=False)
    start = clock()
    receipt = dict(schema=1, status='started', stage='admission', image_calls_attempted=0,
                   image_calls_completed=0, images_completed=0, text_calls=0,
                   scoring_attempted=0, scoring_completed=0, members=[], costs_seconds={})
    def save():
        receipt['wall_before_receipt_save'] = clock() - start
        record(out/'receipt.json', receipt)
    def timed(label, function):
        began = clock()
        try: return function()
        finally:
            elapsed = clock() - began
            require(np.isfinite(elapsed) and elapsed >= 0, 'invalid clock')
            receipt['costs_seconds'][label] = receipt['costs_seconds'].get(label, 0.) + elapsed
    raw_rows = []
    try:
        save()
        spec = timed('admission', operations.admit)
        names, text = spec['names'], spec['text_normalized']
        require(len(names) == len(set(names)) == 128 and len(spec['members']) == 128, 'cohort coverage')
        require(text.dtype == np.float32 and text.shape == (10,512) and np.isfinite(text).all() and
                np.allclose(np.linalg.norm(text, axis=1), 1., atol=2e-5, rtol=0), 'reused text invalid')
        ids = ids_for(names)
        lock = dict(schema=1, names=names, ids=ids, batches=[i//64 for i in range(128)],
                    members=spec['members'], evidence=spec['evidence'])
        record(out/'selection-lock.json', lock, exclusive=True)
        receipt['selection_lock_sha256'] = digest(out/'selection-lock.json')
        receipt['stage'] = 'load_encoder'; save()
        timed('model_load_and_runtime', operations.load_encoder)
        seen = set(spec['capability_member_sha256'])
        require(len(seen) == 8, 'duplicate capability bytes')
        for index in range(0, 128, 4):
            receipt['stage'] = 'decode_batch_' + str(index//4); save()
            payloads = []
            for offset in range(4):
                raw = timed('member_reads', lambda j=index+offset: operations.read_member(j))
                sha = hashlib.sha256(raw).hexdigest()
                require(sha not in seen, 'duplicate image bytes')
                seen.add(sha)
                require(len(raw) == spec['members'][index+offset]['bytes'], 'member length')
                receipt['members'].append(dict(id=ids[index+offset], bytes=len(raw), sha256=sha))
                payloads.append(raw)
            tensors = timed('preprocess', lambda: operations.preprocess(payloads))
            receipt['stage'] = 'encode_batch_' + str(index//4)
            receipt['image_calls_attempted'] += 1; save()
            values = timed('image_encode', lambda: operations.encode(tensors))
            receipt['image_calls_completed'] += 1
            require(isinstance(values, np.ndarray) and values.dtype == np.float32 and values.shape == (4,512), 'image output contract')
            raw_rows.append(values.copy()); receipt['images_completed'] += 4
            def partial():
                with (out/'partial-features.npz.pending').open('wb') as f:
                    np.savez(f, image_raw=np.concatenate(raw_rows))
                os.replace(out/'partial-features.npz.pending', out/'partial-features.npz')
                receipt['partial_features_sha256'] = digest(out/'partial-features.npz')
            timed('partial_saves', partial)
            require(np.isfinite(values).all(), 'nonfinite image feature')
            norms = np.linalg.norm(values, axis=1)
            require(np.isfinite(norms).all() and (norms > 0).all(), 'zero/nonfinite image norm')
            save()
        timed('encoder_release', operations.release_encoder)
        receipt['runtime'] = operations.runtime_record
        require((receipt['image_calls_attempted'], receipt['image_calls_completed'], receipt['images_completed'], receipt['text_calls']) == (32,32,128,0), 'extraction coverage')
        raw = np.concatenate(raw_rows); normalized = raw/np.linalg.norm(raw, axis=1, keepdims=True)
        prototypes = text.T[None].copy()
        with (out/'features.npz').open('xb') as f:
            np.savez(f, image_raw=raw, image_normalized=normalized, text_normalized=text)
        receipt['features_sha256'] = digest(out/'features.npz')
        receipt['stage'] = 'predict'; save()
        backend = operations.backend()
        timed('prediction_pipeline', lambda: headroom.predict(normalized, prototypes, ids, backend, out/'predictions.json'))
        saved = json.loads((out/'predictions.json').read_text())
        require(saved['status'] == 'completed', 'incomplete predictions: no scoring')
        require(len(saved['rows']) == 128 and len(saved['fit_states']) == 132 and saved['successful_native_solver_executions'] == 264, 'fit coverage')
        require([r['id'] for r in saved['rows']] == ids and all(r['statuses'][a] == 'ok' for r in saved['rows'] for a in headroom.ARMS) and
                all(e['status'] == 'completed' for e in saved['fit_states']), 'saved arm coverage')
        require(saved['inputs'] == dict(features_sha256=headroom.array_hash(normalized), prototypes_sha256=headroom.array_hash(prototypes), ids_sha256=headroom.ids_hash(ids)), 'saved input binding')
        for row in saved['rows']:
            for arm in headroom.ARMS: headroom.probability(row['probabilities'][arm])
        seal = dict(predictions_sha256=digest(out/'predictions.json'), features_sha256=receipt['features_sha256'],
                    selection_lock_sha256=receipt['selection_lock_sha256'], all_arms=128*5)
        record(out/'prediction-seal.json', seal, exclusive=True)
        receipt['prediction_seal_sha256'] = digest(out/'prediction-seal.json')
        receipt['stage'] = 'score'; receipt['scoring_attempted'] = 1; save()
        record(out/'score-started.json', seal, exclusive=True)
        labels = operations.labels(names)  # The only class-label conversion, after seal.
        timed('scoring', lambda: headroom.score_file(out/'predictions.json', labels, ids, out/'scores.json'))
        require(digest(out/'predictions.json') == seal['predictions_sha256'], 'prediction changed during scoring')
        receipt['scoring_completed'] = 1; receipt['scores_sha256'] = digest(out/'scores.json')
        receipt.update(status='completed', stage='completed')
    except Exception as exc:
        receipt.update(status='incomplete', failure_type=type(exc).__name__,
                       failure=str(exc) if isinstance(exc, Invalid) else None)
    finally:
        try:
            operations.close()
        except Exception as exc:
            receipt.update(status='incomplete', close_failure_type=type(exc).__name__)
        save()
    return receipt
