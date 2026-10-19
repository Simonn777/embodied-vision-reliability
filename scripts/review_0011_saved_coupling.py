"""Independent 0011 saved-output replay. NumPy/std lib only; no frozen imports.

Raw logits and native positions only, never images/GT/evaluation scores. Direct
positive-matrix row/column IPF deliberately differs from log-domain Sinkhorn.
CLI: --root ROOT --output NEW.json; --self-test runs only artificial fixtures.
A mismatch remains failure even if roundoff-sensitive modal ties explain it.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import tempfile
import unittest

import numpy as np

GROUPS = ('g-036', 'g-002')
VIEWS = ('original', 'reference', 'oblique', 'axis_control')
METHODS = tuple(v + '_native' for v in VIEWS) + (
    'reference_independent', 'coordinate_average', 'entropy_select',
    'product_joint', 'mixture_joint', 'oblique_coupling', 'axis_coupling')
CENTERS = np.arange(256, dtype=np.float64) + .5
FLOOR, TOL, COMPARE = 1e-12, 1e-9, 1e-5


def array(value, shape):
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind not in 'fiu':
        raise ValueError('invalid array shape/type')
    a = a.astype(np.float64)
    if not np.isfinite(a).all():
        raise ValueError('nonfinite array')
    return a


def marginals(logits):
    yx = array(logits, (512,)).reshape(2, 256) * .5
    with np.errstate(under='ignore'):
        p = np.exp(yx - yx.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    p = np.maximum(p, FLOOR)
    p /= p.sum(axis=1, keepdims=True)
    return p[1], p[0]


def rotation(view):
    if view in ('original', 'reference'):
        return np.eye(2)
    if view == 'axis_control':
        return np.array([[0., -1.], [1., 0.]])
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    return np.array([[c, -s], [s, c]])


def inverse_point(xy, view):
    # For column-vector T: R*s*(x-center)+center; row inverse uses R.
    scale = 1. if view == 'original' else .7
    return (array(xy, (2,)) - 128.) @ rotation(view) / scale + 128.


def likelihood(px, py, view):
    # Equal .7 shrink cancels exactly in T_view inv(T_reference).
    xx, yy = np.meshgrid(CENTERS - 128., CENTERS - 128.)
    r = rotation(view)
    ux, uy = r[0, 0]*xx + r[0, 1]*yy + 128., r[1, 0]*xx + r[1, 1]*yy + 128.
    fx = np.maximum(np.interp(ux, CENTERS, px, left=FLOOR, right=FLOOR), FLOOR)
    fy = np.maximum(np.interp(uy, CENTERS, py, left=FLOOR, right=FLOOR), FLOOR)
    return fx * fy


def ipf(kernel, px, py, *, tolerance=TOL, limit=1000):
    """Finite positive matrix IPF; expose last residual on nonconvergence."""
    q = array(kernel, (len(py), len(px))).copy()
    px, py = array(px, (q.shape[1],)), array(py, (q.shape[0],))
    if (np.any(q <= 0) or np.any(px <= 0) or np.any(py <= 0)
            or abs(px.sum()-1) > 1e-12 or abs(py.sum()-1) > 1e-12):
        raise ValueError('IPF requires positive kernel and normalized marginals')
    if type(limit) is not int or limit < 1 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('invalid IPF bound')
    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        for iteration in range(1, limit + 1):
            q *= (py / q.sum(axis=1))[:, None]
            q *= (px / q.sum(axis=0))[None, :]
            if not np.isfinite(q).all() or np.any(q <= 0):
                raise ValueError('IPF lost finite positive support')
            er = float(np.max(np.abs(q.sum(axis=1)-py)))
            ec = float(np.max(np.abs(q.sum(axis=0)-px)))
            if max(er, ec) <= tolerance:
                break
    ok = max(er, ec) <= tolerance
    return q if ok else None, {'iterations': iteration, 'row_error': er,
        'column_error': ec, 'residual': max(er, ec), 'converged': ok}


def decode(q, radius=20):
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or not np.isfinite(q).all() or np.any(q <= 0):
        raise ValueError('invalid joint')
    # C flattening of transpose is x-major, then y: exact lexicographic ties.
    index = int(np.argmax(q.T))
    x, y = divmod(index, q.shape[0])
    xs, ys = max(0, x-radius), max(0, y-radius)
    xe, ye = min(q.shape[1], x+radius+1), min(q.shape[0], y+radius+1)
    patch = q[ys:ye, xs:xe]
    mass = patch.sum()
    xy = np.array([patch.sum(axis=0) @ (np.arange(xs, xe)+.5),
                   patch.sum(axis=1) @ (np.arange(ys, ye)+.5)]) / mass
    peak = float(q[y, x])
    largest = np.partition(q.ravel(), -2)[-2:] if q.size > 1 else [peak, peak]
    return xy, {'mode_index_xy': [x, y], 'exact_mode_ties': int(np.count_nonzero(q == peak)),
                'relative_top_two_gap': float((peak-min(largest))/peak),
                'near_mode_count_relative_1e_12': int(np.count_nonzero(q >= peak*(1-1e-12)))}


def replay(logits, positions):
    if set(logits) != set(VIEWS) or set(positions) != set(VIEWS):
        raise ValueError('four views required')
    ps = {v: marginals(logits[v]) for v in VIEWS}
    px, py = ps['reference']
    q0 = py[:, None] * px[None, :]
    out = {v+'_native': inverse_point(positions[v], v) for v in VIEWS}
    ties, solvers, checks, failures = {}, {}, {}, []
    def read(name, q):
        xy, tie = decode(q)
        out[name] = inverse_point(xy, 'reference')
        ties[name] = tie
    read('reference_independent', q0)
    checks['reference_native_parity_max_difference_original'] = float(np.max(np.abs(out['reference_native']-out['reference_independent'])))
    if checks['reference_native_parity_max_difference_original'] > 1e-3:
        failures.append('reference native/independent parity gate failed')
    out['coordinate_average'] = .5*(out['reference_native']+out['oblique_native'])
    hs = {v: float(sum(-np.sum(p*np.log(p)) for p in ps[v])) for v in ('reference', 'oblique')}
    chosen = 'reference' if hs['reference'] <= hs['oblique'] else 'oblique'
    out['entropy_select'] = out[chosen+'_native'].copy()
    for v, name in (('oblique', 'oblique_coupling'), ('axis_control', 'axis_coupling')):
        g = likelihood(*ps[v], v)
        if v == 'oblique':
            product = q0*g
            read('product_joint', product/product.sum())
            read('mixture_joint', .5*q0 + .5*g/g.sum())
        q, info = ipf(g, px, py)
        solvers[name] = info
        if q is None:
            failures.append(name+': IPF did not converge')
            continue
        read(name, q)
        delta = np.array([(q.sum(axis=0)-q0.sum(axis=0)) @ CENTERS,
                          (q.sum(axis=1)-q0.sum(axis=1)) @ CENTERS])
        checks[name] = {'global_mean_delta_reference_xy': delta.tolist(),
                        'global_mean_max_reference': float(np.abs(delta).max()),
                        'global_mean_max_original': float(np.abs(delta).max()/.7)}
        if np.abs(delta).max() > 5e-5:
            failures.append(name+': mean gate failed')
        if v == 'axis_control':
            diff = float(np.max(np.abs(q-q0)))
            checks['axis_joint_max_difference'] = diff
            checks['axis_coordinate_max_difference'] = float(np.max(np.abs(out[name]-out['reference_independent'])))
            if diff > 1e-7:
                failures.append('axis joint gate failed')
    return {'methods': out, 'ties': ties, 'solvers': solvers, 'checks': checks,
            'entropy_selected_view': chosen, 'failures': failures}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def binding(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def verify(path, expected, read_hashes, root):
    if path.is_symlink() or binding(path) != expected:
        raise ValueError('saved-file hash binding failed: '+str(path.relative_to(root)))
    read_hashes[str(path.relative_to(root))] = expected


def row_key(row):
    g, t, p = row['group_id'], row['observation_index'], row['point_index']
    if g not in GROUPS or type(t) is not int or type(p) is not int or not 0 <= t < 16 or not 0 <= p < 8:
        raise ValueError('invalid row identity')
    if row['frame_id'] != 16*t or type(row['point_id']) is not int:
        raise ValueError('frame/point identity invalid')
    return g, t, p


def review(root):
    root = Path(root)
    started = time.monotonic()
    hashes = {}
    seal_path = root/'derived/seal.json'
    seal = read_json(seal_path)
    hashes['derived/seal.json'] = binding(seal_path)
    if seal['status'] != 'sealed' or seal['rows'] != 256 or seal.get('GT_opened') is not False:
        raise ValueError('missing complete pre-GT decode seal')
    for name in ('predictions.json', 'diagnostics.json'):
        verify(root/'derived'/name, seal['files'][name], hashes, root)
    prep_path = root/'prepared/receipt.json'
    verify(prep_path, seal['inputs']['preparation_receipt'], hashes, root)
    prep = read_json(prep_path)
    if prep['status'] != 'completed_preparation':
        raise ValueError('incomplete preparation')
    ids = {}
    for group in prep['groups']:
        g = group['group_id']
        if g in ids or g not in GROUPS:
            raise ValueError('wrong/duplicate preparation group')
        ids[g] = group['point_ids']
        if (len(ids[g]) != 8 or any(type(i) is not int or not 0 <= i < 30 for i in ids[g])
                or ids[g] != sorted(set(ids[g])) or group['frame_ids'] != list(range(0, 241, 16))):
            raise ValueError('preparation identity grid differs')
    if set(ids) != set(GROUPS):
        raise ValueError('missing preparation group')
    predicted = read_json(root/'derived/predictions.json')
    if tuple(predicted['methods']) != METHODS:
        raise ValueError('wrong declared methods')
    rows = predicted['rows']
    diag = read_json(root/'derived/diagnostics.json')
    if len(rows) != 256 or len(diag) != 256:
        raise ValueError('incomplete derived grid')
    indexed, diagnostics = {}, {}
    for values, target in ((rows, indexed), (diag, diagnostics)):
        for row in values:
            key = row_key(row)
            if key in target or row['point_id'] != ids[key[0]][key[2]]:
                raise ValueError('duplicate/mismatched derived identity')
            target[key] = row
    expected = {(g, t, p) for g in GROUPS for t in range(16) for p in range(8)}
    if set(indexed) != expected or set(diagnostics) != expected:
        raise ValueError('derived grid incomplete')
    report = {'status': 'failed', 'scope': 'saved logits/native coordinates only; no GT, images, model or quality scores',
              'algorithm': 'positive-domain matrix IPF, row then column; no frozen module imports',
              'coordinate_tolerance': COMPARE, 'marginal_tolerance': TOL, 'max_iterations': 1000,
              'expected_points': 256, 'processed_points': 0, 'compared_coordinates': 0,
              'max_absolute_difference_by_method': {m: 0. for m in METHODS}, 'failures': [], 'points': [],
              'inputs': hashes, 'source_sha256': sha(Path(__file__)),
              'runtime': {'python': platform.python_version(), 'numpy': np.__version__}}
    for g in GROUPS:
        folder = root/'predictions'/g
        verify(folder/'receipt.json', seal['inputs'][g]['receipt'], hashes, root)
        receipt = read_json(folder/'receipt.json')
        if (receipt['status'] != 'completed_predictions' or receipt['group_id'] != g
                or receipt['forwards_attempted'] != 64 or len(receipt['frames']) != 64
                or receipt['input_sha256'] != prep['files'][g+'.predictor.npz']['sha256']):
            raise ValueError('prediction receipt incomplete')
        frames, names = {}, set()
        for item in receipt['frames']:
            v, t, name = item['view'], item['observation_index'], item['output_file']
            if (v not in VIEWS or type(t) is not int or not 0 <= t < 16 or (v,t) in frames
                    or item['frame_index'] != 16*t or Path(name).name != name or name in names):
                raise ValueError('raw frame identity/name differs')
            expected_file = seal['inputs'][g]['frames'][name]
            if item['output_sha256'] != expected_file['sha256']:
                raise ValueError('raw receipt and seal hashes disagree')
            verify(folder/name, expected_file, hashes, root)
            frames[(v,t)] = name
            names.add(name)
        if len(names) != 64 or set(seal['inputs'][g]['frames']) != names:
            raise ValueError('raw frame seal incomplete')
        for t in range(16):
            raw = {}
            for v in VIEWS:
                with np.load(folder/frames[(v,t)], allow_pickle=False) as f:
                    if set(f.files) != {'coordinate_logits','positions_xy','visibility_logits'}:
                        raise ValueError('unexpected raw NPZ keys')
                    raw[v] = {n:f[n] for n in f.files}
                for n, shape in (('coordinate_logits',(8,512)),('positions_xy',(8,2)),('visibility_logits',(8,))):
                    if raw[v][n].dtype != np.float32:
                        raise ValueError('raw dtype differs')
                    array(raw[v][n], shape)
            for p in range(8):
                key = g,t,p
                saved, d = indexed[key], diagnostics[key]
                identity = {k:saved[k] for k in ('group_id','observation_index','frame_id','point_index','point_id')}
                result = dict(identity, differences={}, failures=[])
                report['processed_points'] += 1
                try:
                    if set(saved['predictions']) != set(METHODS):
                        raise ValueError('saved method set differs')
                    for v in VIEWS:
                        if saved['visibility_logits'][v] != float(raw[v]['visibility_logits'][p]):
                            raise ValueError('saved visibility differs from raw')
                    own = replay({v:raw[v]['coordinate_logits'][p] for v in VIEWS},
                                 {v:raw[v]['positions_xy'][p] for v in VIEWS})
                    result.update(ties=own['ties'], solvers=own['solvers'], checks=own['checks'],
                                  entropy_selected_view=own['entropy_selected_view'])
                    result['failures'].extend(own['failures'])
                    result['saved_exact_mode_ties'] = {m:x.get('exact_mode_tie_count')
                        for m,x in d['joint_distributions'].items()}
                    for m in METHODS:
                        saved_xy = array(saved['predictions'][m], (2,))
                        if m not in own['methods']:
                            result['failures'].append(m+': unavailable numerical reconstruction')
                            continue
                        diff = float(np.max(np.abs(saved_xy-own['methods'][m])))
                        result['differences'][m] = diff
                        report['compared_coordinates'] += 2
                        report['max_absolute_difference_by_method'][m] = max(report['max_absolute_difference_by_method'][m], diff)
                        if diff > COMPARE:
                            result['failures'].append(m+': coordinate mismatch')
                            result.setdefault('mismatches',{})[m] = {'saved_xy':saved_xy.tolist(),
                                'replayed_xy':own['methods'][m].tolist(), 'max_difference':diff,
                                'tie_diagnostic':own['ties'].get(m)}
                    # Do not trust booleans or iteration equality across routes.
                    if set(d['solvers']) != {'oblique_coupling','axis_coupling'}:
                        raise ValueError('saved solver set differs')
                    for m,s in d['solvers'].items():
                        if (s['converged'] is not True or type(s['iterations']) is not int or not 1 <= s['iterations'] <= 1000
                                or any(not np.isfinite(s[z]) or not 0 <= s[z] <= TOL for z in ('row_error','column_error','residual'))
                                or s['residual'] != max(s['row_error'],s['column_error'])):
                            result['failures'].append(m+': saved solver gate invalid')
                    for m in ('oblique_coupling','axis_coupling'):
                        value = d['global_means'][m]['reference_max_abs_difference']
                        if not np.isfinite(value) or not 0 <= value <= 5e-5:
                            result['failures'].append(m+': saved mean gate invalid')
                    value = d['axis_joint_max_abs_difference']
                    if not np.isfinite(value) or not 0 <= value <= 1e-7:
                        result['failures'].append('saved axis joint gate invalid')
                    if d['entropy_selected_view'] != own['entropy_selected_view']:
                        result['failures'].append('entropy selection differs')
                except (ValueError, KeyError, TypeError, FloatingPointError) as exc:
                    result['failures'].append(type(exc).__name__+': '+str(exc))
                if result['failures']:
                    report['failures'].append(dict(identity, reasons=result['failures']))
                report['points'].append(result)
    report['status'] = 'passed' if not report['failures'] and report['processed_points'] == 256 and report['compared_coordinates'] == 5632 else 'failed'
    report['wall_seconds'] = time.monotonic()-started
    return report


class ArtificialTests(unittest.TestCase):
    def test_known_three_by_three(self):
        q = np.array([[.12,.03,.05],[.04,.18,.08],[.04,.09,.37]])
        x, y = q.sum(0), q.sum(1)
        kernel = q * np.array([2.,3.,5.])[:,None] * np.array([7.,11.,13.])[None,:]
        actual, info = ipf(kernel,x,y,tolerance=1e-13)
        np.testing.assert_allclose(actual,q,rtol=0,atol=2e-13)
        self.assertTrue(info['converged'])

    def test_separable_and_nonuniform(self):
        px,py = np.array([.1,.3,.6]), np.array([.2,.5,.3])
        q,info=ipf(np.outer([3.,7.,2.],[9.,4.,1.]),px,py)
        np.testing.assert_allclose(q,np.outer(py,px),atol=1e-16)
        self.assertEqual(info['iterations'],1)

    def test_nonconvergence_is_explicit(self):
        q,info=ipf(np.array([[1.,.01],[.02,3.]]),[.3,.7],[.6,.4],limit=1,tolerance=1e-15)
        self.assertIsNone(q)
        self.assertFalse(info['converged'])

    def test_invalid_and_logits_axis(self):
        for bad in (np.zeros((2,2)),np.full((2,2),np.nan)):
            with self.assertRaises(ValueError): ipf(bad,[.5,.5],[.5,.5])
        l=np.zeros(512);l[17]=100;l[256+31]=100
        px,py=marginals(l)
        self.assertEqual(int(px.argmax()),31);self.assertEqual(int(py.argmax()),17)
        self.assertTrue(np.all(px>0));self.assertAlmostEqual(px.sum(),1)

    def test_lexicographic_tie_and_local_mean(self):
        q=np.ones((5,5));q[4,1]=9;q[0,3]=9
        xy,tie=decode(q,radius=0)
        np.testing.assert_array_equal(xy,[1.5,4.5]);self.assertEqual(tie['exact_mode_ties'],2)
        q=np.outer([.2,.3,.5],[.6,.1,.3])
        xy,_=decode(q,radius=20)
        np.testing.assert_allclose(xy,[1.2,1.8],atol=1e-15)

    def test_inverse_and_axis_geometry(self):
        point=np.array([81.,207.])
        for v in VIEWS:
            s=1. if v=='original' else .7
            forward=(point-128) @ rotation(v).T*s+128
            np.testing.assert_allclose(inverse_point(forward,v),point,atol=5e-14)
        px=np.arange(1.,257.);px/=px.sum();py=px[::-1]
        expected=px[::-1,None]*py[None,:]
        np.testing.assert_allclose(likelihood(px,py,'axis_control'),expected,atol=0,rtol=0)

    def test_full_256_bin_fixture(self):
        # Artificial Gaussian-shaped logits; no real saved outputs/model call.
        logits={};positions={}
        for i,v in enumerate(VIEWS):
            logits[v]=np.r_[-((CENTERS-(100+3*i))/25)**2,-((CENTERS-(131-2*i))/20)**2].astype(np.float32)
            px,py=marginals(logits[v]);positions[v]=decode(np.outer(py,px))[0]
        originals={v:logits[v].copy() for v in VIEWS}
        r=replay(logits,positions)
        self.assertEqual(set(r['methods']),set(METHODS));self.assertEqual(r['failures'],[])
        self.assertLess(r['checks']['axis_joint_max_difference'],1e-7)
        for m in ('oblique_coupling','axis_coupling'):
            self.assertLessEqual(r['solvers'][m]['residual'],TOL)
            self.assertLessEqual(r['checks'][m]['global_mean_max_reference'],5e-5)
        for v in VIEWS: np.testing.assert_array_equal(logits[v],originals[v])
        np.testing.assert_allclose(r['methods']['reference_independent'],r['methods']['reference_native'],atol=1e-12)

    def test_artificial_saved_complete_grid_and_persistent_failures(self):
        # File-layout/completeness fixture only; expected values generated by
        # this checker are NOT an independent numerical oracle for its math.
        with tempfile.TemporaryDirectory(prefix='0011-artificial-review-') as td:
            root=Path(td);(root/'prepared').mkdir();(root/'derived').mkdir()
            def write(path,obj): path.write_text(json.dumps(obj,allow_nan=False))
            logits={v:np.r_[-((CENTERS-110)/23)**2,-((CENTERS-140)/19)**2].astype(np.float32) for v in VIEWS}
            pos={v:decode(np.outer(marginals(logits[v])[1],marginals(logits[v])[0]))[0].astype(np.float32) for v in VIEWS}
            own=replay(logits,pos)
            self.assertFalse(own['failures'])
            prep={'status':'completed_preparation','groups':[],'files':{}}
            seal={'status':'sealed','rows':256,'GT_opened':False,'inputs':{},'files':{}}
            rows=[];diags=[]
            for g in GROUPS:
                folder=root/'predictions'/g;folder.mkdir(parents=True)
                prep['groups'].append({'group_id':g,'point_ids':list(range(8)),'frame_ids':list(range(0,241,16))})
                prep['files'][g+'.predictor.npz']={'sha256':'synthetic-unused-predictor'}
                receipt={'status':'completed_predictions','group_id':g,'forwards_attempted':64,
                         'input_sha256':'synthetic-unused-predictor','frames':[]}
                sealed={}
                for t in range(16):
                    for v in VIEWS:
                        name=f'{v}-{t}.npz'
                        np.savez(folder/name,coordinate_logits=np.repeat(logits[v][None,:],8,axis=0),
                                 positions_xy=np.repeat(pos[v][None,:],8,axis=0),visibility_logits=np.zeros(8,np.float32))
                        sealed[name]=binding(folder/name)
                        receipt['frames'].append({'view':v,'observation_index':t,'frame_index':16*t,
                                                  'output_file':name,'output_sha256':sealed[name]['sha256']})
                    for i in range(8):
                        identity={'group_id':g,'observation_index':t,'frame_id':16*t,'point_index':i,'point_id':i}
                        rows.append(dict(identity,predictions={m:own['methods'][m].tolist() for m in METHODS},
                                         visibility_logits={v:0. for v in VIEWS}))
                        diags.append(dict(identity,solvers=own['solvers'],
                            entropy_selected_view=own['entropy_selected_view'],
                            joint_distributions={m:{'exact_mode_tie_count':x['exact_mode_ties']} for m,x in own['ties'].items()},
                            global_means={m:{'reference_max_abs_difference':own['checks'][m]['global_mean_max_reference']}
                                          for m in ('oblique_coupling','axis_coupling')},
                            axis_joint_max_abs_difference=own['checks']['axis_joint_max_difference']))
                write(folder/'receipt.json',receipt)
                seal['inputs'][g]={'receipt':binding(folder/'receipt.json'),'frames':sealed}
            write(root/'prepared/receipt.json',prep)
            seal['inputs']['preparation_receipt']=binding(root/'prepared/receipt.json')
            write(root/'derived/predictions.json',{'methods':METHODS,'rows':rows})
            write(root/'derived/diagnostics.json',diags)
            for name in ('predictions.json','diagnostics.json'): seal['files'][name]=binding(root/'derived'/name)
            write(root/'derived/seal.json',seal)
            result=review(root)
            self.assertEqual(result['status'],'passed');self.assertEqual(result['processed_points'],256)
            self.assertEqual(result['compared_coordinates'],5632);self.assertEqual(len(result['inputs']),134)
            # Re-seal a coordinate mismatch: no numerical failure drops later rows.
            rows[0]['predictions']['original_native'][0]+=1
            write(root/'derived/predictions.json',{'methods':METHODS,'rows':rows})
            seal['files']['predictions.json']=binding(root/'derived/predictions.json');write(root/'derived/seal.json',seal)
            result=review(root)
            self.assertEqual(result['status'],'failed');self.assertEqual(result['processed_points'],256)
            self.assertEqual(len(result['failures']),1)
            self.assertEqual(result['max_absolute_difference_by_method']['original_native'],1.)
            # Missing and duplicated row with a coherent replacement seal fail grid validation.
            rows[-1]=rows[0]
            write(root/'derived/predictions.json',{'methods':METHODS,'rows':rows})
            seal['files']['predictions.json']=binding(root/'derived/predictions.json');write(root/'derived/seal.json',seal)
            with self.assertRaisesRegex(ValueError,'duplicate'): review(root)
            # Raw mutation is detected before numerical use.
            rows[-1]=dict(rows[-2],point_index=7,point_id=7)
            write(root/'derived/predictions.json',{'methods':METHODS,'rows':rows})
            seal['files']['predictions.json']=binding(root/'derived/predictions.json');write(root/'derived/seal.json',seal)
            with (root/'predictions/g-036/original-0.npz').open('ab') as f: f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'hash binding'): review(root)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        if args.root or args.output: parser.error('--self-test cannot read saved study files')
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ArtificialTests))
        return 0 if result.wasSuccessful() else 1
    if args.root is None or args.output is None: parser.error('--root and --output required')
    # Reserve result before reading; refusal prevents overwriting prior audit.
    with args.output.open('x') as out:
        try:
            report=review(args.root)
        except Exception as exc:
            report={'status':'failed','error':type(exc).__name__+': '+str(exc),'source_sha256':sha(Path(__file__)),
                    'scope':'input/schema or replay failure; no GT/model execution'}
        json.dump(report,out,indent=2,allow_nan=False);out.write('\n')
    print(json.dumps({'status':report['status'],'processed_points':report.get('processed_points',0),
                      'source_sha256':report['source_sha256']}))
    return 0 if report['status']=='passed' else 1


if __name__ == '__main__':
    sys.exit(main())
