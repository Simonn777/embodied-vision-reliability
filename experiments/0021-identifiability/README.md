# 0021 manufactured identifiability checks

This is a fixed mathematical check, not an accuracy experiment. It reads no scientific data, images, labels, or model outputs, and does not estimate or optimize a query pose. All constants below were written before execution. No random seeds, fixture search, or parameter sweep is used.

Coordinates are calibrated normalized camera coordinates with both K matrices equal to I. Features are `[d,u*d,v*d,1,u,v]`; theta multiplies these in that order. Reference prediction is `pi(R*(f*x)+t)` with `f=Phi*theta`, `x=(u,v,1)`. The analytic Jacobian columns are theta[6], left rotation[3], and translation[2 tangent coordinates or 3 free coordinates]. In particular rotation differentiates `R*(f*x)`, **not** `R*(f*x)+t`. Unit translation uses `normalize(t+B*delta)` and fixed orthonormal tangent basis B at the fixture.

Fixed constructions:

- Six reference pixels: `(-.65,-.45),(.5,-.4),(-.4,.55),(.6,.5),(.1,-.05),(-.15,.25)`.
- Generic depths: `2,3.1,4.6,2.7,5.3,3.8`; theta: `1.1,.12,-.08,.2,-.06,.09`.
- Rotation vector: `.13,-.19,.07`; translation: normalized `(.9,.2,.35)`.
- Plane fixture at the same reference locations but in a **separate manufactured scene**: depth `1/(.2*u-.15*v+.8)`, identity theta. Its analytic null is `(.8,.2,-.15,-1,0,0)`; the fixed finite perturbation is 0.1 times this vector.
- Eight distinct query pixels: `(-.55,.10),(.25,.65),(.72,-.15),(-.75,-.65),(.05,.45),(.35,-.55),(-.22,-.12),(.48,.12)`; their depths are `2.2,3.6,4.1,2.5,5,3.3,4.7,2.9`. No pixel has contradictory reference/query depth values. The plane-scene query support is deliberately off the reference plane.
- Pure-rotation fixture: generic support and rotation, zero free translation. No unit-translation parameterization is asserted at zero translation.

Predeclared checks and tolerances:

1. Independent scalar feature formula and central finite differences check the analytic projection/full Jacobian. Fixed step 1e-6, absolute tolerance 5e-8, relative tolerance 1e-6. **Only analytic Jacobians enter rank checks.**
2. SVD rank threshold is `1e-10 * largest singular value`. Also report ranks at the fixed 1e-12 and 1e-8 relative thresholds; generic expectations must agree at all three. Expected generic feature/pose/unit-joint/free-joint ranks: 6/5/11/11. This is a local condition, not global branch uniqueness.
3. The free-translation scale vector `(theta,0,t)` must have Jacobian residual <1e-12. Scaling theta and t by fixed 1.7 must preserve projections within 1e-12. Additive shift 0.15 must change projections >1e-4; its differential after projecting out pose columns must retain norm >1e-4.
4. Plane feature rank is 5 and joint unit-translation rank <=10. The known null, numerical analytic-J nullspace, and finite positive perturbation must preserve reference projections within 1e-12. Corrected depths and slopes on all fixed sample supports stay positive.
5. Project query depth changes orthogonally to `[f_query,ones]`. This quotient intentionally discards global scale **and shift**; shift is not a true reference geometric gauge. The known and analytic-J-derived reference nulls must remain visible with norm >0.01. The quotient must annihilate its own scale/shift basis within 1e-12.
6. Pure rotation has theta-block norm and finite depth-shift projection change <1e-12. Invalid/nonfinite inputs fail explicitly.

Any failed expectation is preserved in the fresh output directory. Do not change fixtures or thresholds to obtain a pass. A later source repair requires a distinct disclosed attempt. The predicted query-depth ambiguity is not an observed pose bias: a wrong depth shape can affect a query estimator in several ways, none of which is measured here. Likewise, generic full rank does not prove a shared real depth error, useful calibration, or a publishable contribution.

The runner writes all matrices, coefficients, singular values, null vectors, quotient residuals, failures, source hashes and actual child test-process exit/timing. It refuses an existing output directory. Run with the existing NumPy environment and one thread; no new dependency is needed.
