"""Small fixed-weight affine-calibration relaxation, implementation prototype.

This module supplies numerical primal/dual diagnostics, NOT interval certificates,
a robust estimator, a pose guarantee, or an admitted scientific experiment.
The objective is [1,x]' H [1,x], x=(b,B,A,C,D), with B>=b*b and
[[A,C],[C,D]] positive semidefinite. Physical recovery uses (b,a,c).
"""
import math
import numpy as np


def validate_hessian_matrix(H):
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (6, 6) or not np.isfinite(H).all():
        raise ValueError('H must be a finite 6 by 6 matrix')
    if not np.allclose(H, H.T, atol=0., rtol=1e-13):
        raise ValueError('H must be symmetric')
    H = H * .5 + H.T * .5
    eigenvalues = np.linalg.eigvalsh(H)
    spectral_norm = float(np.linalg.norm(H, 2))
    if not np.isfinite(eigenvalues).all() or not math.isfinite(spectral_norm):
        raise ValueError('nonfinite H spectral arithmetic')
    # Admit roundoff-sized negatives as an explicit diagnostic, never project H.
    tolerance = 1e-12 * max(1., spectral_norm)
    if eigenvalues[0] < -tolerance:
        raise ValueError('H is materially indefinite')
    return H, eigenvalues


def objective(H, x):
    z = np.r_[1., x]
    with np.errstate(over='ignore', invalid='ignore'):
        value = float(z @ H @ z)
    if not math.isfinite(value):
        raise FloatingPointError('nonfinite computed objective')
    return value


def interior(x):
    b, B, A, C, D = map(float, x)
    slack = B-b*b
    determinant = A*D-C*C
    return bool(np.isfinite(x).all() and math.isfinite(slack)
                and math.isfinite(determinant) and slack > 0. and A > 0.
                and D > 0. and determinant > 0.)


def barrier_terms(H, x, mu):
    """Value, gradient and Hessian of f-mu log(B-b²)-mu log det(M)."""
    if not interior(x) or not math.isfinite(mu) or mu <= 0.:
        raise ValueError('strictly interior x and positive mu required')
    b, B, A, C, D = map(float, x)
    slack = B - b*b
    M = np.array([[A, C], [C, D]])
    inv = np.linalg.inv(M)
    determinant = A*D-C*C
    z = np.r_[1., x]
    value = float(z @ H @ z) - mu*(math.log(slack)+math.log(determinant))
    gradient = 2.*(H[1:] @ z)
    hessian = 2.*H[1:, 1:].copy()
    ds = np.array([-2.*b, 1., 0., 0., 0.])
    gradient -= mu*ds/slack
    hessian += mu*np.outer(ds, ds)/(slack*slack)
    hessian[0, 0] += 2.*mu/slack
    basis = (np.array([[1., 0.], [0., 0.]]),
             np.array([[0., 1.], [1., 0.]]),
             np.array([[0., 0.], [0., 1.]]))
    for j, E in enumerate(basis, 2):
        gradient[j] -= mu*np.trace(inv @ E)
        for k, F in enumerate(basis, 2):
            hessian[j, k] += mu*np.trace(inv @ E @ inv @ F)
    if not math.isfinite(value) or not np.isfinite(gradient).all() or not np.isfinite(hessian).all():
        raise FloatingPointError('nonfinite computed barrier value or derivatives')
    return value, gradient, hessian


def dual_diagnostic(H, multiplier, cone_dual):
    """Evaluate the convex relaxation's algebraic dual on supplied multipliers.

    The result is a floating-point lower-bound *diagnostic*. Neither this code
    nor eigvalsh establishes a rigorous rounding enclosure. Singular curvature
    is unavailable rather than silently using a pseudoinverse.
    """
    S = np.asarray(cone_dual, dtype=float)
    if not math.isfinite(multiplier) or multiplier < 0.:
        raise ValueError('negative or nonfinite scalar multiplier')
    if S.shape != (2, 2) or not np.isfinite(S).all() or not np.allclose(S, S.T):
        raise ValueError('cone dual must be finite symmetric 2 by 2')
    S = S*.5+S.T*.5
    if np.linalg.eigvalsh(S)[0] < 0.:
        raise ValueError('cone dual is not positive semidefinite numerically')
    Q = H[1:, 1:].copy()
    Q[0, 0] += multiplier
    linear = 2.*H[0, 1:] + np.array([0., -multiplier, -S[0, 0],
                                    -2.*S[0, 1], -S[1, 1]])
    eigenvalues = np.linalg.eigvalsh(Q)
    condition = math.inf if eigenvalues[0] <= 0. else float(eigenvalues[-1]/eigenvalues[0])
    if not math.isfinite(condition) or condition > 1e12:
        return dict(status='unavailable', reason='dual_curvature_not_well_conditioned_pd',
                    eigenvalues=eigenvalues, condition=condition)
    try:
        factor = np.linalg.cholesky(Q)
        solution = np.linalg.solve(factor.T, np.linalg.solve(factor, linear))
    except np.linalg.LinAlgError:
        return dict(status='unavailable', reason='dual_cholesky_failed',
                    eigenvalues=eigenvalues, condition=condition)
    value = float(H[0, 0] - .25*linear @ solution)
    residual = float(np.linalg.norm(Q @ solution-linear))
    if not math.isfinite(value) or not math.isfinite(residual):
        return dict(status='unavailable', reason='nonfinite_computed_dual',
                    eigenvalues=eigenvalues, condition=condition)
    return dict(status='numerical_only', value=value, condition=condition,
                stationarity_residual=residual,
                eigenvalues=eigenvalues)


def recover_physical(x, minimum_depth1, minimum_depth2):
    """A feasible upper candidate; rank slack is retained, not declared zero."""
    b, B, A, C, D = map(float, x)
    if not np.isfinite([*x, minimum_depth1, minimum_depth2]).all() or A <= 0.:
        return dict(status='unavailable', reason='nonpositive_scale_square_or_nonfinite')
    a = math.sqrt(A)
    c = C/a
    lifted = np.array([b, b*b, a*a, a*c, c*c])
    slack = B-b*b
    determinant = A*D-C*C
    corrected_minima = [b+float(minimum_depth1), a*float(minimum_depth2)+c]
    if not np.isfinite([a, c, *lifted, slack, determinant, *corrected_minima]).all():
        return dict(status='unavailable', reason='nonfinite_computed_recovery')
    if min(corrected_minima) <= 0.:
        return dict(status='unavailable', reason='nonpositive_corrected_depth')
    return dict(status='feasible', parameters=np.array([b, a, c]),
                lifted=lifted, parabola_slack=slack, determinant_slack=determinant)


def solve_relaxation(H, minimum_depth1, minimum_depth2, max_newton=40):
    """Bounded central path for artificial qualification; no fallback retries.

    A zero-centered strictly feasible initialization and fixed mu schedule are
    used. Any failure stays explicit. Internal backtracking is part of this one
    numerical solve. No constraints require positive reconstructed depths in
    the relaxation; physical recovery checks them on the returned candidate.
    """
    H, eigenvalues = validate_hessian_matrix(H)
    if not np.isfinite([minimum_depth1, minimum_depth2]).all():
        raise ValueError('finite support minima required')
    if not isinstance(max_newton, int) or max_newton < 1 or max_newton > 100:
        raise ValueError('max_newton must be between 1 and 100')
    scale = max(1., float(np.linalg.norm(H, 2)))
    scaled_H = H/scale
    x = np.array([0., 1., 1., 0., 1.])
    history=[]
    status='completed'
    for mu in (1., .1, .01, .001, .0001, .00001, .000001, .0000001, .00000001):
        converged=False
        for iteration in range(max_newton):
            value, gradient, hessian = barrier_terms(scaled_H, x, mu)
            try:
                direction = np.linalg.solve(hessian, -gradient)
            except np.linalg.LinAlgError:
                status='newton_linear_solve_failed';break
            decrement = -float(gradient @ direction)
            if not np.isfinite(direction).all() or not math.isfinite(decrement) or decrement < 0.:
                status='invalid_newton_direction';break
            if decrement <= 1e-12:
                converged=True;break
            step=1.
            accepted=False
            for backtrack in range(60):
                trial=x+step*direction
                if interior(trial):
                    trial_value=barrier_terms(scaled_H, trial, mu)[0]
                    if trial_value <= value - .01*step*decrement:
                        accepted=True;break
                step *= .5
            if not accepted:
                status='line_search_failed';break
            x=trial
        history.append(dict(mu=mu, iterations=iteration+1, converged=converged,
                            objective=objective(H, x)))
        if status != 'completed':break
        if not converged:
            status='newton_iteration_limit';break
    b,B,A,C,D=x
    multiplier=scale*mu/(B-b*b)
    cone_dual=scale*mu*np.linalg.inv(np.array([[A,C],[C,D]]))
    dual=dual_diagnostic(H, multiplier, cone_dual)
    physical=recover_physical(x,minimum_depth1,minimum_depth2)
    if physical['status']=='feasible':
        physical['objective']=objective(H,physical['lifted'])
        if dual['status']=='numerical_only':
            gap=physical['objective']-dual['value']
            if not math.isfinite(gap):
                raise FloatingPointError('nonfinite computed primal-dual gap')
            physical['numerical_gap']=gap
    return dict(status=status, x=x, objective=objective(H,x), H_eigenvalues=eigenvalues,
                H_scale=scale, history=history, dual=dual, physical=physical,
                rigorous_certificate=False, robust_estimator=False)
