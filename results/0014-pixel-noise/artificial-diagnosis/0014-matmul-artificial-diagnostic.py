"""Artificial-only bounded Nx3 matmul diagnosis; no dataset or native estimator."""
import importlib.util
import json
from pathlib import Path
import traceback
import warnings
import numpy as np

repo = Path(__file__).resolve().parent / 'embodied-vision-reliability'
spec = importlib.util.spec_from_file_location('frozen_refine_0014', repo / 'experiments/0014-pixel-noise/refine.py')
refine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refine)


def explicit(A, B):
    return A[:, 0, None] * B[None, 0, :] + A[:, 1, None] * B[None, 1, :] + A[:, 2, None] * B[None, 2, :]


def probe(name, A, B):
    row = {'name': name, 'shapeA': list(A.shape), 'shapeB': list(B.shape),
           'max_abs_A': float(np.max(np.abs(A))), 'max_abs_B': float(np.max(np.abs(B)))}
    row['absolute_output_bound'] = 3.0 * row['max_abs_A'] * row['max_abs_B']
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise'):
            C = A @ B
        row['strict_status'] = 'success'
    except FloatingPointError as exc:
        row['strict_status'] = type(exc).__name__
        row['strict_message'] = str(exc)
        row['traceback'] = traceback.format_exc()
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        with np.errstate(over='warn', divide='warn', invalid='warn'):
            C = A @ B
        row['warn_messages'] = [str(w.message) for w in records]
    with np.errstate(over='raise', divide='raise', invalid='raise'):
        expected = explicit(A, B)
        row['explicit_finite'] = bool(np.isfinite(expected).all())
        row['matmul_finite'] = bool(np.isfinite(C).all())
        row['max_abs_difference'] = float(np.max(np.abs(C - expected)))
        row['max_abs_output'] = float(np.max(np.abs(C)))
    return row


results = []
for n in (120, 512, 1000, 4096, 10000):
    rng = np.random.default_rng(20261020)
    x = np.column_stack((rng.uniform(-.5, .5, (n, 2)), np.ones(n)))
    y = np.column_stack((rng.uniform(-.5, .5, (n, 2)), np.ones(n)))
    R = refine.so3_exp(np.array([.025, -.06, .015]))
    t = np.array([.7, -.1, .2]); t /= np.linalg.norm(t)
    E = refine.skew(t) @ R
    current = {'n': n, 'probes': [probe('g1=y@E', y, E), probe('g2=x@E.T', x, E.T)]}
    try:
        residual, jacobian = refine.residuals_and_jacobian(x, y, R, t, [620., 850.], [1100., 730.], 'N')
        current['kernel'] = {'status': 'success', 'residual_finite': bool(np.isfinite(residual).all()), 'jacobian_finite': bool(np.isfinite(jacobian).all())}
    except Exception as exc:
        current['kernel'] = {'status': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
    results.append(current)
report = {'numpy': np.__version__, 'coordinate_bounds': 'x,y each in [-.5,.5], homogeneous z=1; unit t, proper near-identity R', 'cases': results}
print(json.dumps(report, indent=2, allow_nan=False))
