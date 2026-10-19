# Artificial-only diagnosis of the 0014 arithmetic failure

No real coordinates, HDF5, labels, saved evaluations, or native estimator were
accessed for this diagnosis. The original frozen kernel was inspected at SHA256
`d0c3ac3c8d7bcaff6debdfa6bde583dea7bba6455f03c4d7337f8dc048afa789`.
Runtime inspection reports NumPy 2.0.2, Darwin aarch64, Apple Accelerate BLAS/LAPACK.

The bounded artificial fixture uses independent normalized coordinates in
[-0.5,0.5], homogeneous third coordinate 1, unit translation, and a proper
near-identity rotation. In the frozen kernel the traceback identifies line 140,
`g1, g2 = y @ E, x @ E.T`: its first product raises a divide-by-zero
FloatingPointError at N=1000,4096,10000; N=120 and512 succeed. Separate probes of
both products reproduce divide/overflow/invalid warnings. In warning mode the
returned products are entirely finite.

This is a demonstrably false arithmetic flag for these artificial contractions:
each input-point component has absolute value at most1, each E entry at most
0.9668345776, and each three-term output has absolute value at most2.9005037326.
There is no division in the mathematical product. Explicit component products
and additions under the same strict error state succeed, agreeing with returned
matmul values within2.220446049250313e-16. This does not establish validity of
the unread real pair or prove its exact failing operation from its generic log.

An additional N=10000 fixture with bounded random A/B/J/r also flags Nx3 matmul
and J.T@(W*J). Explicit reference bounds are3 and160000 respectively.
Unoptimized einsum succeeds for Nx3, H, and g contractions under strict checks;
maximum differences from explicit component/reduction references are4.44e-16,
5.09e-11, and3.84e-13 respectively. J.T@(W*r) itself did not flag in that probe.
These differences are ordinary summation-order differences, not bitwise parity.
The dimensionless Cauchy objective already uses elementwise log1p and sum.

Primary-source context: [NumPy issue28687](https://github.com/numpy/numpy/issues/28687)
reports size-dependent erroneous matmul flags on Apple hardware, including
NumPy2.0.2, and links the closed issue to
[PR29223](https://github.com/numpy/numpy/pull/29223). The observed pattern is
consistent with that report; no operating-system or dependency change was made.

After the parent's explicit authorization, only N-dependent kernel products
were replaced with `np.einsum(..., optimize=False)`: g1/g2, dg1/dg2, H/g, and final
gradient. Strict errstate, finite/zero-denominator checks, objective, manifold,
optimizer parameters, stopping criteria, and native calls are unchanged.
Updated kernel SHA256:
`f63a55f5b971f86549a184ca52bdd61f5871c36b54ab944ca68d2c768c5bc262`.
The test module now has12 passing cases, including10k-point N/P residual checks
against explicit component references, full optimization, monotone accepted
costs, and actual overflow rejection. No real retry was performed here.

Artifacts in this scratch directory:
`0014-matmul-artificial-diagnostic.py/.json` (original frozen-kernel reproduction),
`0014-matmul-contract-check.py/.json` (bounded contraction comparisons).
The first scratch-write command used an incorrect relative directory and failed
before creating or executing a diagnostic; its corrected invocation produced
the recorded reports. Re-running the first script now imports the repaired
working kernel, so its preserved original JSON is the original traceback record.
