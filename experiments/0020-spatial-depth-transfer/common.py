"""Bound fixed-bank scoring helpers; no labels or native calls at import."""
import contextlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = ROOT/'protocols/0020-spatial-depth-transfer.md'
MANIFEST = ROOT/'configs/0017-development-manifest.json'
LIBRARIES = ROOT/'configs/0017-libraries.json'
COMPARISON_BINDINGS = ROOT/'configs/0020-comparison-bindings.json'
ARMS = ('e_norm', 'e_ts', 'depth_scale', 'depth_affine')
SELECTORS = ('pixel_msac', 'global_affine_ls', 'spatial_weight_only', 'random_point_transfer', 'spatial_transfer')
MAX_FITS_PER_PAIR = 40


@contextlib.contextmanager
def aliases(mapping):
    absent = object(); previous = {k: sys.modules.get(k, absent) for k in mapping}
    sys.modules.update(mapping)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is absent: sys.modules.pop(key, None)
            else: sys.modules[key] = value


def _load(name, filename, dependencies=None):
    path = ROOT/'experiments/0017-depth-information'/filename
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    with aliases(dependencies or {}): spec.loader.exec_module(value)
    return value


base = _load('evr0020_base_common', 'common.py')
base_worker = _load('evr0020_base_worker', 'predict_pair.py', {'common': base})
base_runner = _load('evr0020_base_runner', 'run_diagnostic.py', {'common': base, 'predict_pair': base_worker})
np = base.np
sha256, json_safe = base.sha256, base.json_safe
write_new, write_record = base.write_new, base.write_record
load_manifest, read_pair = base.load_manifest, base.read_pair
validate_pose = base.validate_pose
bound_file, input_hashes = base_runner.bound_file, base_worker.input_hashes


def source_hashes():
    sources = base.source_hashes()
    paths = [HERE/name for name in ('common.py', 'predict_pair.py', 'run_diagnostic.py',
                                    'score.py', 'evaluate_saved.py', 'run_operation.py')]
    paths += [ROOT/'experiments/0019-author-loss'/name for name in
              ('common.py','predict_pair.py','run_diagnostic.py','evaluate_saved.py','run_operation.py')]
    paths += [PROTOCOL, ROOT/'protocols/0019-author-loss.md', COMPARISON_BINDINGS]
    sources.update({str(path.relative_to(ROOT)): sha256(path) for path in paths})
    return sources


def library_config():
    return json.loads(LIBRARIES.read_text())


def baseline_bundle(manifest_path, seal_path):
    manifest, seal, records = base_runner.verify_seal(manifest_path, seal_path)
    if len(records) != 320 or any(r['status'] != 'completed' for r in records):
        raise ValueError('The entire original candidate bank must be completed')
    bank = {(r['pair_index'], r['arm']): r for r in records}
    if len(bank) != 320:
        raise ValueError('Duplicate or missing original candidate')
    candidates = {i: [bank[i, arm] for arm in ARMS] for i in range(80)}
    hashes = {(r['pair_index'], r['arm']): r['sha256'] for r in seal['rows']}
    binding = {'seal_sha256': sha256(seal_path),
               'predictions': {f'{i}:{arm}': hashes[i, arm] for i in range(80) for arm in ARMS}}
    return manifest, candidates, binding


def check_inputs(record, candidates):
    if len(candidates) != 4 or [r['arm'] for r in candidates] != list(ARMS):
        raise ValueError('Candidate order or coverage differs')
    for candidate in candidates:
        if candidate['status'] != 'completed': raise ValueError('Incomplete candidate')
        for key in ('input_arrays', 'calibration', 'input_diagnostics'):
            if record.get(key) != candidate.get(key):
                raise ValueError('Original candidate input differs: '+key)
        for key in ('pair_index', 'scene', 'key'):
            if record.get(key) != candidate.get(key):
                raise ValueError('Original candidate identity differs: '+key)
        if record['library'] != candidate['library']:
            raise ValueError('Original candidate library identity differs')


def fit_count(scores):
    """Reconstruct work counters from the required saved fit records."""
    rows=scores.get('candidates',[])
    if len(rows)!=4 or [r.get('arm') for r in rows]!=list(ARMS):
        raise ValueError('Candidate count/order differs')
    partition=scores.get('spatial_partition',{})
    state=partition.get('status');counts=partition.get('counts')
    if (state not in ('completed','unavailable') or not isinstance(counts,list) or len(counts)!=4
        or any(type(v) is not int or v<0 for v in counts) or sum(counts)!=scores['count']
        or (state=='completed')!=all(v>0 for v in counts)):
        raise ValueError('Invalid saved spatial partition availability')
    reconstructed={'candidate_count':4,'fit_calls':0,'svd_calls':0,
                   'fit_completed_count':0,'fit_unavailable_count':0}
    placeholder=[{'status':'unavailable','reason':'spatial_partition_unavailable'}]
    for row in rows:
        groups=row.get('fits',{})
        if set(groups)!=set(SELECTORS[1:]):raise ValueError('Missing or extra calibration control fits')
        for name in SELECTORS[1:]:
            records=groups[name]
            if state=='unavailable' and name!='global_affine_ls':
                if records!=placeholder:raise ValueError('Unavailable partition placeholder differs')
                continue
            folds=[None] if name in ('global_affine_ls','spatial_weight_only') else list(range(4))
            if not isinstance(records,list) or len(records)!=len(folds):
                raise ValueError('Incomplete calibration fit coverage')
            for record,fold in zip(records,folds):
                if (not isinstance(record,dict) or 'fold' not in record or record.get('fold')!=fold
                    or type(record.get('fold')) is not type(fold)
                    or not isinstance(record.get('fit'),dict) or not isinstance(record.get('validation'),dict)):
                    raise ValueError('Malformed or reordered calibration fit record')
                fit=record['fit'];status=fit.get('status');attempted=fit.get('svd_attempted')
                if (status not in ('completed','unavailable') or type(attempted) is not bool
                    or (status=='completed' and (not attempted or fit.get('reason') is not None))
                    or (status=='unavailable' and (not isinstance(fit.get('reason'),str) or not fit['reason']))):
                    raise ValueError('Invalid saved fit status/SVD outcome')
                reconstructed['fit_calls']+=1
                reconstructed['svd_calls']+=int(attempted)
                reconstructed['fit_completed_count']+=int(status=='completed')
                reconstructed['fit_unavailable_count']+=int(status=='unavailable')
    counters=scores.get('counters',{})
    if (set(counters)!=set(reconstructed) or any(type(v) is not int for v in counters.values())
        or counters!=reconstructed or reconstructed['fit_calls']>MAX_FITS_PER_PAIR):
        raise ValueError('Reported counters disagree with saved fit records')
    return reconstructed['fit_calls']


def validate_scores(scores,candidates,key):
    """Reconstruct saved score/counter identities without redoing geometry or fits."""
    json_safe(scores)
    n=candidates[0]['input_arrays']['x1']['shape'][0]
    if (scores.get('schema')!='evr0020.scores.v1' or scores.get('key')!=key
        or scores.get('count')!=n or scores.get('arms')!=list(ARMS)):
        raise ValueError('Score schema/input coverage differs')
    if set(scores.get('selectors',{}))!=set(SELECTORS):raise ValueError('Missing or extra selector')
    rows=scores.get('candidates',[])
    if len(rows)!=4 or [r.get('arm') for r in rows]!=list(ARMS):raise ValueError('Scored bank differs')
    fit_count(scores)
    for i,row in enumerate(rows):
        geometry=row.get('geometry',{});total=geometry.get('pixel_msac')
        if (type(geometry.get('count')) is not int or geometry['count']!=n
            or type(total) not in (int,float) or not np.isfinite(total) or not 0<=total<=n
            or abs(total-candidates[i]['final_scores']['pixel_msac'])>1e-10*max(1,n)):
            raise ValueError('Sealed geometric-score parity differs')
        g=total/n
        if type(row.get('geometric_mean')) not in (int,float) or row['geometric_mean']!=g:
            raise ValueError('Geometric mean differs from saved sum/count')
        depths=row.get('depth_scores',{});values=row.get('scores',{})
        if set(depths)!=set(SELECTORS[1:]) or set(values)!=set(SELECTORS):
            raise ValueError('Missing or extra candidate score component')
        if any(type(d) not in (int,float) or not np.isfinite(d) or not 0<=d<=1 for d in depths.values()):
            raise ValueError('Invalid capped depth score')
        reconstructed={'pixel_msac':g,**{name:g+depths[name] for name in SELECTORS[1:]}}
        if any(type(v) not in (int,float) for v in values.values()) or values!=reconstructed:
            raise ValueError('Hybrid score is not saved geometry plus depth')
        if scores['spatial_partition']['status']=='unavailable' and any(depths[name]!=1. for name in SELECTORS[2:]):
            raise ValueError('Unavailable spatial partition did not retain full depth cost')
    for name in SELECTORS:
        selection=scores['selectors'][name]; values=selection['scores']
        if not isinstance(values,list) or len(values)!=4:raise ValueError('Incomplete selector scores')
        cap=1. if name=='pixel_msac' else 2.
        if any(type(v) not in (int,float) or not np.isfinite(v) or not 0<=v<=cap for v in values):
            raise ValueError('Invalid bounded selector score')
        index=min(range(4),key=lambda i:(values[i],i))
        if (type(selection.get('selected_index')) is not int or selection['selected_index']!=index
            or selection.get('selected_arm')!=ARMS[index] or selection.get('score')!=values[index]):
            raise ValueError('Selector winner or exact tie order differs')
        if any(rows[i]['scores'][name]!=values[i] for i in range(4)):
            raise ValueError('Candidate and selector scores differ')
        if name=='pixel_msac' and any(rows[i]['geometric_mean']!=values[i] for i in range(4)):
            raise ValueError('Geometric selector differs')
    return True
