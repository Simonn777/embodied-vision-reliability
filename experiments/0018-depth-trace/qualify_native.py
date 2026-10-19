"""Artificial-only API parity and independent trace-cost qualification.

Root executes this after building the separate instrumented package. Different
packages are loaded in separate subprocesses. No HDF5, image, label, or cached
scientific input is opened. All fixtures are regenerated from a fixed seed.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
CASES = ('clean', 'pixel_noise', 'affine_depth')
ARMS = ('depth_scale', 'depth_affine')


def safe(value):
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Qualification output contains unencoded nonfinite value')
    return value


def write(path, value, exclusive=False):
    with Path(path).open('x' if exclusive else 'w') as f:
        json.dump(safe(value), f, allow_nan=False, sort_keys=True, indent=2)
        f.write('\n')


def fixture(name):
    rng = np.random.default_rng(20261024)
    X = rng.uniform([-2., -1., 4.], [2., 1., 9.], (48, 3))
    angle = .09
    R = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
    t = np.array([.6, .03, .02])
    Y = np.einsum('ij,nj->ni', R, X, optimize=False) + t
    K = np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]])
    def project(points):
        h = np.einsum('ij,nj->ni', K, points, optimize=False)
        return h[:, :2]/h[:, 2, None]
    x1, x2 = project(X), project(Y)
    d1, d2 = X[:, 2].copy(), Y[:, 2].copy()
    if name == 'pixel_noise':
        x2 += rng.normal(0., .4, x2.shape)
    elif name == 'affine_depth':
        d1 = (d1-.4)/1.2
        d2 = (d2+.2)/.9
    elif name != 'clean':
        raise ValueError('Unknown artificial fixture')
    camera = {'model': 'PINHOLE', 'width': 640, 'height': 480, 'params': [800., 800., 320., 240.]}
    return {'x1': x1, 'x2': x2, 'depth1': d1, 'depth2': d2,
            'calibration': {'K1': K, 'K2': K, 'camera1': dict(camera), 'camera2': dict(camera)}}


def options(arm):
    return {'ransac': {'min_iterations': 4, 'max_iterations': 4, 'seed': 0,
                       'success_prob': .9999, 'dyn_num_trials_mult': 3.,
                       'progressive_sampling': False, 'score_initial_model': False},
            'bundle': {}, 'max_errors': [16., 2.], 'estimate_shift': arm == 'depth_affine', 'weight_sampson': 1.}


def unpack(result, info):
    geometry = {'R': np.asarray(result.pose.R), 't': np.asarray(result.pose.t),
                **{key: float(getattr(result, key)) for key in ('scale', 'shift1', 'shift2')}}
    return safe({'geometry': geometry, 'info': dict(info)})


def child(package, variant, output):
    receipt = {'schema': 'evr0018.artificial-native-child.v1', 'status': 'started',
               'scope': 'Generated 48-point fixtures only', 'variant': variant, 'calls': []}
    write(output, receipt, exclusive=True)
    started = time.perf_counter()
    try:
        sys.path.insert(0, str(Path(package).resolve()))
        import poselib
        origin = Path(poselib.__file__).resolve()
        if Path(package).resolve() not in origin.parents:
            raise ValueError('Unexpected package origin')
        receipt['package_file'] = str(origin)
        receipt['version'] = poselib.__version__
        for case in CASES:
            pair = fixture(case)
            cal = pair['calibration']
            args = (pair['x1'], pair['x2'], pair['depth1'], pair['depth2'], cal['camera1'], cal['camera2'])
            for arm in ARMS:
                opt = options(arm)
                item = {'case': case, 'arm': arm, 'options': opt, 'status': 'started',
                        'fixture_sha256': hashlib.sha256(json.dumps(safe(pair), sort_keys=True).encode()).hexdigest()}
                receipt['calls'].append(item)
                write(output, receipt)
                call_start = time.perf_counter()
                result, info = poselib.estimate_monodepth_relative_pose(*args, opt, initial_pose=None)
                item['disabled'] = unpack(result, info)
                if variant == 'instrumented':
                    result, info, trace = poselib.estimate_monodepth_relative_pose_trace(
                        *args, opt, initial_pose=None, trace_capacity=50000)
                    item['enabled'] = unpack(result, info)
                    item['trace'] = safe(trace)
                    result, info, capped = poselib.estimate_monodepth_relative_pose_trace(
                        *args, opt, initial_pose=None, trace_capacity=2)
                    item['overflow_output'] = unpack(result, info)
                    item['overflow_trace'] = safe(capped)
                item['native_seconds'] = time.perf_counter()-call_start
                item['status'] = 'completed'
                write(output, receipt)
        receipt['status'] = 'completed'
        receipt['seconds'] = time.perf_counter()-started
        write(output, receipt)
        return 0
    except BaseException as exc:
        receipt.update(status='error', error=type(exc).__name__+': '+str(exc), seconds=time.perf_counter()-started)
        write(output, receipt)
        raise


def parity(actual, expected, path='value'):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError('Parity key mismatch '+path)
        for key in expected:
            parity(actual[key], expected[key], path+'.'+key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError('Parity length mismatch '+path)
        for i, (a, b) in enumerate(zip(actual, expected)):
            parity(a, b, path+f'[{i}]')
    elif type(expected) is float:
        if type(actual) not in (int, float) or not math.isfinite(actual) or abs(actual-expected) > 1e-12*max(1., abs(actual), abs(expected)):
            raise AssertionError('Floating parity mismatch '+path)
    elif type(actual) is not type(expected) or actual != expected:
        raise AssertionError('Discrete parity mismatch '+path)


def qualify(baseline_package, instrumented_package, output):
    output = Path(output)
    receipt = {'schema': 'evr0018.artificial-native-qualification.v1', 'status': 'started',
               'scope': 'Generated fixtures only; no scientific data/native scientific calls', 'checks': []}
    write(output, receipt, exclusive=True)
    try:
        children = {}
        env = dict(os.environ)
        for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[key] = '1'
        for variant, package in (('baseline', baseline_package), ('instrumented', instrumented_package)):
            child_path = output.with_name(output.stem+'.'+variant+'.json')
            command = [sys.executable, str(Path(__file__).resolve()), '--child', '--variant', variant,
                       '--package', str(Path(package).resolve()), '--output', str(child_path.resolve())]
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
            receipt[variant+'_process'] = {'command': command, 'returncode': result.returncode,
                                           'stdout': result.stdout, 'stderr': result.stderr}
            write(output, receipt)
            if result.returncode:
                raise RuntimeError(variant+' artificial child failed')
            children[variant] = json.loads(child_path.read_text())
            if children[variant]['status'] != 'completed':
                raise RuntimeError('Incomplete artificial child')
        spec = importlib.util.spec_from_file_location('evr0018_qualification_diagnosis', HERE/'diagnose_trace.py')
        diagnosis = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = diagnosis
        spec.loader.exec_module(diagnosis)
        for base, traced in zip(children['baseline']['calls'], children['instrumented']['calls']):
            if (base['case'], base['arm'], base['fixture_sha256']) != (traced['case'], traced['arm'], traced['fixture_sha256']):
                raise AssertionError('Artificial identity mismatch')
            for mode in ('disabled', 'enabled', 'overflow_output'):
                parity(traced[mode], base['disabled'])
            if traced['enabled'] != traced['disabled'] or traced['overflow_output'] != traced['disabled']:
                raise AssertionError('Observer on/off artificial outputs must be exactly equal')
            trace, capped = traced['trace'], traced['overflow_trace']
            if not trace['complete'] or trace['dropped_events'] or trace['event_count'] != len(trace['events']):
                raise AssertionError('Incomplete artificial full trace')
            if capped['complete'] or capped['event_count'] != 2 or capped['dropped_events'] <= 0:
                raise AssertionError('Overflow contract violated')
            if (trace['proposal_trace_hash_hex'], trace['sampled_draws'], trace['scored_proposals']) != (
                    capped['proposal_trace_hash_hex'], capped['sampled_draws'], capped['scored_proposals']):
                raise AssertionError('Capacity altered proposal stream')
            pair = fixture(base['case'])
            mask = np.asarray(traced['enabled']['info']['inliers'], dtype=bool)
            if mask.shape != (len(pair['x1']),):
                raise AssertionError('Artificial native mask shape mismatch')
            report = diagnosis.diagnose_native_trace(diagnosis.prepare_inputs(pair), trace, base['arm'],
                                                      np.flatnonzero(mask).tolist())
            receipt['checks'].append({'case': base['case'], 'arm': base['arm'],
                                      'baseline_parity': True, 'on_off_bitwise_values': True,
                                      'overflow_parity': True, 'rolling_hash_parity': True,
                                      'cost_reconstruction': report})
            write(output, receipt)
        if len(receipt['checks']) != len(CASES)*len(ARMS):
            raise AssertionError('Artificial case coverage incomplete')
        receipt['status'] = 'passed'
        receipt['source_sha256'] = {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                                    for name in ('qualify_native.py', 'diagnose_trace.py')}
        write(output, receipt)
        print(json.dumps({'status': 'passed', 'artificial_cases': len(receipt['checks'])}))
        return 0
    except BaseException as exc:
        receipt.update(status='error', error=type(exc).__name__+': '+str(exc))
        write(output, receipt)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--baseline-package', type=Path)
    p.add_argument('--instrumented-package', type=Path)
    p.add_argument('--child', action='store_true')
    p.add_argument('--package', type=Path)
    p.add_argument('--variant', choices=('baseline', 'instrumented'))
    a = p.parse_args()
    if a.child:
        if a.package is None or a.variant is None:
            p.error('--child requires --package and --variant')
        sys.exit(child(a.package, a.variant, a.output))
    if a.baseline_package is None or a.instrumented_package is None:
        p.error('Qualification requires both package parent directories')
    sys.exit(qualify(a.baseline_package, a.instrumented_package, a.output))
