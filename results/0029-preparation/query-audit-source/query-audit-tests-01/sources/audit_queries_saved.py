"""Independent saved-only arithmetic audit; no model/network calls.

Frozen seal_queries and run_evaluation.gt_contract are reused for provenance
ONLY. No production metric, selector, loss or gate is imported/called. Private
auditor source is recorded by SHA and must also be bound by its dispatcher.
Run only after explicit authorization, with the qualified CPython -S runtime.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

OWN = 'experiments/0029-memory-information/'
COHORT = (('dogs-scale', 83), ('upside-down', 65), ('dog-agility', 25),
          ('cat-girl', 89), ('planes-water', 38), ('dog-gooses', 86))
IDENTITIES = tuple((name, 4 + (2*j*(n-5)+7)//14) for name, n in COHORT for j in range(8))
CONTROLS = ('baseline', 'max_predicted_iou', 'max_unsigned_change', 'always_D1',
            'always_D3', 'random_deletion', 'random_on_signed', 'native_bank',
            'prompt_direct', 'prompt_signed', 'reset_direct', 'reset_signed')
METHODS = ('signed_deletion', *CONTROLS)
BANK = ('B', 'D1', 'D3')
NATIVE = ('M0', 'M1', 'M2')
ATOL = RTOL = 1e-9


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


class Checks:
    def __init__(self):
        self.count = 0
        self.errors = []

    def equal(self, actual, expected, path):
        self.count += 1
        if isinstance(expected, dict):
            if not isinstance(actual, dict) or set(actual) != set(expected):
                self.errors.append(path + ': mapping keys differ')
                return
            for key in expected:
                self.equal(actual[key], expected[key], path + '.' + str(key))
        elif isinstance(expected, (list, tuple)):
            if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
                self.errors.append(path + ': sequence length differs')
                return
            for i, value in enumerate(expected):
                self.equal(actual[i], value, path + '[' + str(i) + ']')
        elif type(expected) is float:
            if type(actual) not in (int, float) or not math.isfinite(actual) or not math.isfinite(expected) or abs(actual-expected) > ATOL + RTOL*abs(expected):
                self.errors.append(path + ': numeric mismatch')
        elif type(actual) is not type(expected) or actual != expected:
            self.errors.append(path + ': exact mismatch')

    def finish(self):
        require(not self.errors, 'independent audit mismatches: ' + '; '.join(self.errors[:8]))


def coverage(rows):
    require([(r['sequence'], r['frame']) for r in rows] == list(IDENTITIES),
            'fixed ordered 48-query coverage required before annotations')
    require(all(r.get('status') == 'completed' for r in rows), 'technical failure: no successful subset')
    require(all(type(r['actionable']) is bool for r in rows), 'actionability type')
    counts = {name: sum(int(r['actionable']) for r in rows if r['sequence'] == name) for name, _ in COHORT}
    videos = sum(int(n > 0) for n in counts.values())
    return dict(queries=48, videos=6, actionable_queries=sum(counts.values()),
                actionable_queries_by_video=counts, videos_with_evidence=videos, **{'pass': videos >= 4})


def mask(array):
    import numpy as np
    a = np.asarray(array)
    require(a.ndim == 2 and a.size > 0 and a.dtype.kind in 'bif' and bool(np.isfinite(a).all()),
            'invalid numeric prediction')
    return a if a.dtype.kind == 'b' else a > 0


def overlap(a, b, valid=None):
    """Integer counts; caller determines the empty-union convention."""
    import numpy as np
    require(a.shape == b.shape, 'mask shapes differ')
    if valid is not None:
        a, b = a & valid, b & valid
    return int(np.sum(a & b, dtype=np.int64)), int(np.sum(a | b, dtype=np.int64))


def foreground_j(prediction, labels, target):
    import numpy as np
    labels = np.asarray(labels)
    require(labels.dtype.kind in 'iu' and labels.ndim == 2 and bool((labels >= 0).all()), 'invalid annotation IDs')
    require(type(target) is int and target > 0 and target != 255, 'target identity')
    intersection, union = overlap(mask(prediction), labels == target, labels != 255)
    return intersection/union if union else 1.0


def bilinear_one(image, xy, original_shape):
    """Independent scalar interpolation with border-replicated pixel centres."""
    h, w = original_shape
    x, y = map(float, xy)
    require(math.isfinite(x) and math.isfinite(y) and 0 <= x < w and 0 <= y < h, 'witness outside image')
    lh, lw = image.shape
    x = min(lw-1., max(0., (x+.5)*lw/w-.5))
    y = min(lh-1., max(0., (y+.5)*lh/h-.5))
    ix, iy = math.floor(x), math.floor(y)
    dx, dy = x-ix, y-iy
    jx, jy = min(ix+1, lw-1), min(iy+1, lh-1)
    value = math.fsum(((1-dx)*(1-dy)*float(image[iy, ix]), dx*(1-dy)*float(image[iy, jx]),
                       (1-dx)*dy*float(image[jy, ix]), dx*dy*float(image[jy, jx])))
    require(math.isfinite(value), 'interpolation nonfinite')
    return value


def witness_loss(image, witness):
    parts = []
    for label, sign in (('positive', -1.), ('negative', 1.)):
        values = []
        for row in witness[label]:
            x = sign*bilinear_one(image, row['target_xy'], witness['current_shape'])
            values.append(max(0., x) + math.log1p(math.exp(-abs(x))))
        require(len(values) >= 4, 'insufficient class for balanced loss')
        parts.append(math.fsum(value/len(values) for value in values))
    return .5*parts[0]+.5*parts[1]


def choose(row, losses, changes, quality):
    """Uses only prediction/evidence values; never receives annotations/J."""
    text = '\0'.join(('evr0029:random-deletion:2026-10-20', row['sequence'], str(row['target_id']), str(row['frame'])))
    digest = hashlib.sha256(text.encode()).hexdigest()
    random = ('D1', 'D3')[int(digest, 16) % 2]
    result = {name: 'B' for name in METHODS}
    if row['actionable']:
        def positive_improvement(names):
            winner = names[0]
            for name in names[1:]:
                if losses[name] < losses[winner]:
                    winner = name
            return winner if losses['B']-losses[winner] > 1e-6 else 'B'
        def first_max(names, values):
            winner = names[0]
            for name in names[1:]:
                if values[name] > values[winner]:
                    winner = name
            return winner
        selected = positive_improvement(BANK)
        result.update(signed_deletion=selected, max_predicted_iou=first_max(BANK, quality),
            max_unsigned_change=first_max(BANK, changes), always_D1='D1', always_D3='D3',
            random_deletion=random, random_on_signed=random if selected != 'B' else 'B',
            native_bank=positive_improvement(('B', *NATIVE)), prompt_direct='P',
            prompt_signed=positive_improvement(('B', 'P')), reset_direct='R',
            reset_signed=positive_improvement(('B', 'R')))
    return dict(actionable=row['actionable'], choices=result,
                signed_action=result['signed_deletion'] != 'B', random=dict(branch=random, sha256=digest),
                bank_losses={n: losses[n] for n in BANK} if row['actionable'] else None)


def validate_witnesses(row):
    w = row['witnesses']
    require(w['target_id'] == row['target_id'] and w['actionable'] is row['actionable'], 'witness identity')
    h, width = w['current_shape']
    require(type(h) is int and type(width) is int and min(h, width) > 0, 'witness shape')
    require(row['actionable'] == (len(w['positive']) >= 4 and len(w['negative']) >= 4), 'witness counts')
    seen = set()
    for name, label in (('positive', 1), ('negative', 0)):
        require(len(w[name]) <= 32, 'witness cap')
        for point in w[name]:
            x, y = point['target_xy']
            require(math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < h, 'invalid continuous witness')
            ix, iy = math.floor(x+.5), math.floor(y+.5)
            require(0 <= ix < width and 0 <= iy < h and [ix, iy] == point['rounded_current_xy']
                    and point['label'] == label and (ix, iy) not in seen, 'witness lookup/collision')
            seen.add((ix, iy))


def numerical_preflight(rows, arrays, checks):
    """Validate all predictions/independent choices before any GT indexing."""
    import numpy as np
    cov = coverage(rows)
    require(set(arrays) == set(IDENTITIES), 'prediction array identities')
    selections, losses_by_row = {}, {}
    for row in rows:
        identity = row['sequence'], row['frame']
        validate_witnesses(row)
        values = arrays[identity]
        required = set(BANK+NATIVE) | ({'P', 'R'} if row['actionable'] else set())
        require(set(values) == required and set(row['candidates']) == required, 'complete candidate bank')
        binary = {name: mask(a) for name, a in values.items()}
        require(all(list(a.shape) == row['witnesses']['current_shape'] for a in binary.values()), 'candidate image shape')
        quality = {name: float(row['candidates'][name]['predicted_iou']) for name in values}
        require(all(math.isfinite(x) for x in quality.values()), 'candidate quality nonfinite')
        losses = {name: witness_loss(value, row['witnesses']) for name, value in values.items()} if row['actionable'] else {}
        changes = {}
        for name in BANK:
            intersection, union = overlap(binary[name], binary['B'])
            changes[name] = 1-intersection/union if union else 0.
            checks.equal(row['candidates'][name]['unsigned_change'], changes[name], 'unsigned.'+str(identity)+name)
        for name in values:
            checks.equal(row['candidates'][name]['loss'], losses.get(name), 'loss.'+str(identity)+name)
        choice = choose(row, losses, changes, quality)
        checks.equal(row['selection'], choice, 'selection.'+str(identity))
        require(type(row['costs']) is dict and set(row['costs']) == {'seconds', 'counts'}, 'cost fields')
        for key, x in row['costs']['seconds'].items():
            require(type(key) is str and type(x) in (int, float) and math.isfinite(x) and x >= 0, 'cost seconds')
        for key, x in row['costs']['counts'].items():
            require(type(key) is str and type(x) is int and x >= 0, 'cost counts')
        selections[identity], losses_by_row[identity] = choice, losses
    for name, _ in COHORT:
        require(len({r['target_id'] for r in rows if r['sequence'] == name}) == 1, 'changing video target')
    checks.finish()
    return cov, selections, losses_by_row


def precision(witness, labels, target):
    output = {}
    for name in ('positive', 'negative'):
        observed = [int(labels[p['rounded_current_xy'][1], p['rounded_current_xy'][0]]) for p in witness[name]]
        known = [x for x in observed if x != 255]
        correct = sum(int((x == target) == (name == 'positive')) for x in known)
        output[name] = dict(retained=len(observed), known=len(known), void_unknown=len(observed)-len(known),
                            correct=correct, incorrect=len(known)-correct, precision=correct/len(known) if known else None)
    return output


def aggregate(metric_rows, cov):
    """Independent equal-video aggregation and fixed necessary gates."""
    require(len(metric_rows) == 48 and [(r['sequence'], r['frame']) for r in metric_rows] == list(IDENTITIES), 'summary coverage')
    keys = tuple(metric_rows[0]['j'])
    videos = {}
    for name, _ in COHORT:
        rows = [r for r in metric_rows if r['sequence'] == name]
        require(len(rows) == 8, 'video denominator')
        means = {k: math.fsum(r['j'][k]/8 for r in rows) for k in keys}
        best = CONTROLS[0]
        for other in CONTROLS[1:]:
            if means[other] > means[best]:
                best = other
        videos[name] = dict(means=means, best_control=best, signed_margin=means['signed_deletion']-means[best])
    pooled = {k: math.fsum(videos[n]['means'][k]/6 for n, _ in COHORT) for k in keys}
    best = CONTROLS[0]
    for other in CONTROLS[1:]:
        if pooled[other] > pooled[best]:
            best = other
    margins = [videos[n]['signed_margin'] for n, _ in COHORT]
    candidate_gain = pooled['oracle_admissible']-pooled['baseline']
    signed_gain = pooled['signed_deletion']-pooled[best]
    gate = dict(coverage=cov['pass'], candidate_information=candidate_gain >= .02,
                pooled_selection=signed_gain >= .02, four_positive_videos=sum(x > 0 for x in margins) >= 4,
                no_video_below_minus_002=min(margins) >= -.02)
    gate['pass'] = all(gate.values())
    return dict(denominator=48, video_count=6, queries_per_video=8, coverage=cov, eligible=True,
        rows=metric_rows, pooled=pooled, videos=videos, best_pooled_control=best,
        candidate_gain=candidate_gain, signed_gain=signed_gain, positive_videos=sum(x > 0 for x in margins),
        worst_video_margin=min(margins), signed_actions=sum(r['selected_branches']['signed_deletion'] != 'B' for r in metric_rows), gate=gate)


def diagnostics(rows):
    output = dict(queries=len(rows), precision={}, native_bank={}, method_choices={}, loss_j_sign={}, costs={})
    for name in ('positive', 'negative'):
        counts = {k: sum(r['witness_precision'][name][k] for r in rows)
                  for k in ('retained', 'known', 'void_unknown', 'correct', 'incorrect')}
        counts['precision'] = counts['correct']/counts['known'] if counts['known'] else None
        output['precision'][name] = counts
    for mode in ('admissible', 'unrestricted'):
        output['native_bank'][mode+'_mean_gain'] = math.fsum(r['native_oracle'][mode+'_gain'] for r in rows)/len(rows)
    for method in METHODS:
        choices = [r['selected_branches'][method] for r in rows]
        output['method_choices'][method] = dict(branch_counts={c: choices.count(c) for c in sorted(set(choices))},
            selected_baseline=choices.count('B'), selected_mask_equals_baseline=sum(r['selected_mask_noop'][method] for r in rows))
    for branch in ('D1', 'D3', 'selected'):
        pairs = [(r['loss_j_sign'][branch]['loss_gain_sign'], r['loss_j_sign'][branch]['j_gain_sign'])
                 for r in rows if r['loss_j_sign'] is not None]
        output['loss_j_sign'][branch] = dict(reported_queries=len(pairs),
            sign_pair_counts={str(i)+','+str(j): pairs.count((i, j)) for i in (-1, 0, 1) for j in (-1, 0, 1)},
            strict_nonzero_concordant=sum(a == b and a != 0 for a, b in pairs),
            strict_nonzero_discordant=sum(a*b == -1 for a, b in pairs))
    harmed = [r for r in rows if r['signed_harm']['drop_gt_005']]
    total = math.fsum(r['signed_harm']['j_drop'] for r in harmed)
    worst = max(harmed, key=lambda r: r['signed_harm']['j_drop']) if harmed else None
    output['harmful_signed_actions'] = dict(threshold_strict_j_drop=.05, count=len(harmed), sum_drop=total,
        mean_drop_among_harmful=total/len(harmed) if harmed else None,
        worst=dict(sequence=worst['sequence'], frame=worst['frame'], j_drop=worst['signed_harm']['j_drop']) if worst else None)
    for unit in ('seconds', 'counts'):
        output['costs'][unit] = {}
        names = sorted({n for r in rows for n in r['costs'][unit]})
        for name in names:
            values = [r['costs'][unit][name] for r in rows if name in r['costs'][unit]]
            output['costs'][unit][name] = dict(sum=math.fsum(values) if unit == 'seconds' else sum(values), reported_queries=len(values))
    return output


def arithmetic(rows, arrays, annotations, checks):
    import numpy as np
    cov, choices, losses = numerical_preflight(rows, arrays, checks)
    require(set(annotations) == set(IDENTITIES), 'all 48 annotation identities required')
    metric_rows, details = [], []
    for row in rows:
        identity = row['sequence'], row['frame']
        binary = {name: mask(a) for name, a in arrays[identity].items()}
        candidate_j = {name: foreground_j(a, annotations[identity], row['target_id']) for name, a in binary.items()}
        selected = choices[identity]['choices']
        base = candidate_j['B']
        j = {method: candidate_j[selected[method]] for method in METHODS}
        j.update(ungated_D1=candidate_j['D1'], ungated_D3=candidate_j['D3'],
                 oracle_admissible=max(candidate_j[n] for n in BANK) if row['actionable'] else base,
                 oracle_unrestricted=max(candidate_j[n] for n in BANK))
        metric_rows.append(dict(sequence=row['sequence'], frame=row['frame'], status='completed',
                               actionable=row['actionable'], j=j, selected_branches=selected))
        gain = candidate_j[selected['signed_deletion']]-base
        relationships = None
        if row['actionable']:
            relationships = {}
            for key, branch in (('D1', 'D1'), ('D3', 'D3'), ('selected', selected['signed_deletion'])):
                dl = losses[identity]['B']-losses[identity][branch]
                dj = candidate_j[branch]-base
                relationships[key] = dict(branch=branch, loss_gain=dl, j_gain=dj,
                    loss_gain_sign=int(dl > 0)-int(dl < 0), j_gain_sign=int(dj > 0)-int(dj < 0))
        native_best = max(candidate_j[n] for n in ('B', *NATIVE))
        details.append(dict(sequence=row['sequence'], frame=row['frame'], target_id=row['target_id'],
            status='completed', actionable=row['actionable'], candidate_j=candidate_j,
            selected_branches=selected, witness_precision=precision(row['witnesses'], annotations[identity], row['target_id']),
            loss_j_sign=relationships, signed_harm=dict(action=selected['signed_deletion'] != 'B', j_delta=gain,
                j_drop=max(0., -gain), drop_gt_005=selected['signed_deletion'] != 'B' and -gain > .05),
            selected_mask_noop={method: bool(np.array_equal(binary[branch], binary['B'])) for method, branch in selected.items()},
            candidate_mask_noop={name: bool(np.array_equal(a, binary['B'])) for name, a in binary.items()},
            native_oracle=dict(unrestricted_j=native_best, unrestricted_gain=native_best-base,
                admissible_j=native_best if row['actionable'] else base,
                admissible_gain=native_best-base if row['actionable'] else 0.), costs=row['costs']))
    summary = aggregate(metric_rows, cov)
    return dict(eligible=True, denominator=48, coverage=cov, gt_accessed=True, summary=summary,
        gate=summary['gate'], rows=details, diagnostics=dict(pooled=diagnostics(details),
            videos={name: diagnostics([r for r in details if r['sequence'] == name]) for name, _ in COHORT}))


def load_arrays(context, checks):
    import numpy as np
    output = {}
    for row, directory in zip(context['rows'], context['query_prediction_dirs']):
        identity = row['sequence'], row['frame']
        output[identity], loaded = {}, {}
        for name, desc in row['candidates'].items():
            artifact = desc['output']
            path = directory/artifact['filename']
            require(path.stat().st_size == artifact['bytes'] and sha(path) == artifact['sha256'], 'opaque array bytes')
            if path not in loaded:
                with np.load(path, allow_pickle=False) as saved:
                    require(set(saved.files) == set(artifact['arrays']), 'array field inventory')
                    loaded[path] = {key: saved[key].copy() for key in saved.files}
                for key, a in loaded[path].items():
                    metadata = artifact['arrays'][key]
                    require(a.dtype.kind in 'bifu' and np.isfinite(a).all(), 'nonfinite/unsafe array')
                    require(list(a.shape) == metadata['shape'] and str(a.dtype) == metadata['dtype']
                            and hashlib.sha256(a.tobytes(order='C')).hexdigest() == metadata['raw_c_order_sha256'], 'numeric array binding')
            arrays = loaded[path]
            a = arrays[desc['array_key']]
            ch = desc['channel']
            require(a.ndim == 4 and a.shape[0] == 1 and type(ch) is int and 0 <= ch < a.shape[1], 'array channel')
            require(list(a.shape[2:]) == row['witnesses']['current_shape'], 'original resolution')
            output[identity][name] = a[0, ch].copy()
            quality = arrays['predicted_quality']
            require(quality.ndim == 2 and quality.shape[0] == 1, 'quality shape')
            selected = ch if name in NATIVE else desc['selected_native_index']
            require(type(selected) is int and 0 <= selected < quality.shape[1], 'quality channel')
            if name not in NATIVE:
                require(selected == int(np.argmax(quality[0])), 'native quality argmax changed')
            checks.equal(desc['predicted_iou'], float(quality[0, selected]), 'saved quality.'+str(identity)+name)
        selected = row['native_bank']['selected_index']
        require(selected in (0, 1, 2) and np.array_equal(output[identity]['B'], output[identity]['M'+str(selected)]), 'baseline/native bank parity')
    checks.finish()
    return output


def admission(args):
    root, proof = Path(args.repo_root).resolve(), Path(args.proof).resolve()
    seal = module('evr0029_independent_audit_seal', root/OWN/'seal_queries.py')
    admitted, context = seal.verify_queries(root, args.expected_commit, proof)
    h, old = context['h'], context['old_admission']
    evaluator = module('evr0029_independent_audit_gt_contract', root/OWN/'run_evaluation.py')
    own = h.frozen(root, args.expected_commit, [OWN+'run_evaluation.py', OWN+'evaluate_queries.py'])
    gt_root, gt_spec = Path(args.gt_root).resolve(), Path(args.gt_spec).resolve()
    evaluation_root, evaluation_spec = Path(args.evaluation_root).resolve(), Path(args.evaluation_spec).resolve()
    command = [str(context['installation']/'venv/bin/python'), '-S', str(root/OWN/'run_evaluation.py'),
        '--repo-root', str(root), '--expected-commit', args.expected_commit, '--proof', str(proof),
        '--gt-root', str(gt_root), '--gt-spec', str(gt_spec), '--output-dir', str(evaluation_root)]
    outer = old.outer_contract(h, root, args.expected_commit, evaluation_spec, evaluation_root/'receipt.json', command)
    masks, gt_binding = evaluator.gt_contract(h, old, root, args.expected_commit, proof, gt_root, gt_spec, admitted, context)
    record = read(evaluation_root/'receipt.json')
    require(record.get('schema') == 'evr0029.query-evaluation-operation.v1' and record.get('status') == 'completed'
        and record.get('eligible') is True and record.get('expected_commit') == args.expected_commit
        and record.get('prediction_admission') == admitted and record.get('gt_binding') == gt_binding
        and record.get('source_sha256') == {**context['sources'], **own}
        and record.get('offline_attempts') == [] and record.get('model_calls') == 0
        and record.get('query_masks_decoded') == 48 and record.get('prediction_numeric_preflight_complete') is True,
        'evaluation operation not completely admitted')
    files = dict(context['files'])
    for mapping in (outer['file_sha256'], gt_binding['file_sha256'], {str(root/n): v for n, v in own.items()}):
        for path, digest in mapping.items():
            require(path not in files or files[path] == digest, 'conflicting admission hash')
            files[path] = digest
    require(set(masks) == set(IDENTITIES), 'all 48 admitted GT files required')
    coverage(context['rows'])
    return seal, context, record, masks, files, outer


def run(args):
    root = Path(args.repo_root).resolve()
    out = Path(args.output_dir).absolute()
    require(not out.exists() and not out.is_symlink() and out.resolve() != root and root not in out.resolve().parents,
            'fresh private audit output required')
    out.mkdir(parents=True)
    started = time.perf_counter()
    self_hash = sha(__file__)
    record = dict(schema='evr0029.saved-query-audit.v1', status='started', eligible=False,
        expected_commit=args.expected_commit, auditor_sha256=self_hash, numerical_tolerance=dict(atol=ATOL, rtol=RTOL),
        query_masks_decoded=0, model_calls=0, offline_attempts=[], compatibility_probes=[])
    checks = Checks()
    def save():
        record.update(wall_seconds=time.perf_counter()-started, checks=checks.count, errors=checks.errors)
        temp = out/'receipt.pending'
        temp.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        os.replace(temp, out/'receipt.json')
    save()
    try:
        seals, context, saved, masks, files, outer = admission(args)
        h, old = context['h'], context['old_admission']
        record.update(admission_complete=True, prediction_admission=context['proof'], file_sha256=files,
                      evaluation_outer=outer, source_sha256=context['sources'])
        seals.require_runtime(context)
        q = module('evr0029_saved_audit_guard', root/'experiments/0022-frontend/qualify.py')
        q.install_guard(record['offline_attempts'], record['compatibility_probes'])
        sys.dont_write_bytecode = True
        import site
        site.main()
        import numpy as np
        from PIL import Image
        runtime = context['installation']/'venv/lib/python3.12/site-packages'
        require(Path(sys.prefix).resolve() == context['installation']/'venv'
                and Path(np.__file__).resolve().is_relative_to(runtime)
                and Path(Image.__file__).resolve().is_relative_to(runtime), 'qualified numerical runtime required')
        arrays = load_arrays(context, checks)
        numerical_preflight(context['rows'], arrays, checks)
        record['all_prediction_arithmetic_preflight_complete'] = True
        save()
        annotations = {}
        for identity in IDENTITIES:
            record['gt_access_attempted'] = True
            save()
            with Image.open(masks[identity]) as image:
                require(image.format == 'PNG' and image.mode in ('P', 'L', 'I', 'I;16'), 'integer annotation PNG required')
                annotations[identity] = np.asarray(image).copy()
            record['query_masks_decoded'] += 1
        result = arithmetic(context['rows'], arrays, annotations, checks)
        expected = saved['evaluation']
        checks.equal(expected.get('schema'), 'evr0029.query-evaluation.v1', 'evaluation schema')
        for key, value in result.items():
            checks.equal(expected.get(key), value, 'evaluation.'+key)
        costs = []
        for video in context['videos']:
            specification = read(video['spec'])
            observed = read(Path(specification['operation_output'])/'receipt.json')
            costs.append(dict(sequence=video['sequence'], wall_seconds=observed['wall_seconds'],
                peak_group_rss_kib=observed['peak_group_rss_kib'], calls=video['record']['native_video']['calls']))
        checks.equal(saved['complete_costs']['video_operations'], costs, 'complete video costs')
        checks.finish()
        seals.recheck(context)
        old.unchanged(h, files)
        require(sha(__file__) == self_hash and not record['offline_attempts'], 'audit source/offline postcheck')
        require(record['query_masks_decoded'] == 48, 'complete annotations')
        record.update(status='completed', eligible=True, independent_summary=result['summary'],
                      independent_diagnostics=result['diagnostics'], independent_gate=result['gate'],
                      scope={'arithmetic': 'Independent all48 logits/quality/loss/choice/J/precision/equal-video/gate arithmetic.',
                             'inherited': 'Frozen sealer and GT admission reused only for source, runtime, full6 outputs, receipts and inventories.',
                             'not_recomputed': 'No model, matching, witness construction, temporal-carrier traces or annotation acquisition repeated.',
                             'private_source': 'Auditor SHA is recorded and unchanged; dispatcher must bind private source separately.'})
    except BaseException as error:
        record.update(status='failed', eligible=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save()
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo-root', 'expected-commit', 'proof', 'gt-root', 'gt-spec', 'evaluation-root', 'evaluation-spec', 'output-dir'):
        parser.add_argument('--'+name, required=True)
    run(parser.parse_args())
