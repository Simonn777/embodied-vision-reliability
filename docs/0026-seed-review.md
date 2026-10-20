# Convex lift as an additional seed: prospective review

Source and mathematics only; no test invocation, data access or outcome-based choice. This note does not admit a real-data experiment.

The lift can be a defensible **additional deterministic initializer**. Feasibility alone does not justify replacing an incumbent pose. There is no evidence yet that loose solutions are frequent, but there is also no tightness theorem for noisy inputs. An exact, uniquely identified physical zero-loss solution can make the relaxation exact; a noisy optimum need not retain either rank-one constraint.

The current recovery `a=sqrt(A), c=C/a` replaces B by b² and D by C²/A. It is a feasible-candidate construction, not the optimum projection under H and not a proof that the relaxed solution corresponds to a physical calibration. Near A=0 it is sensitive even when every intermediate remains finite. A missing dual diagnostic does not logically invalidate a usable primal seed, and a small numerical dual gap does not prove pose quality.

## Structural failure that feasibility does not remove

With constant first-image depth d0, take b=-d0+epsilon, a=epsilon and c=0, with positive second-image raw depths. For every epsilon>0 both clouds have positive depths, but both shrink with epsilon. The squared pair-distance discrepancy scales as epsilon⁴ and the raw proper-Procrustes point loss as epsilon², regardless of whether the limiting shapes agree. Thus strict depth positivity and a low unnormalized 3D objective do not rule out collapse. Near-constant depths produce a conditioning concern, not an established generic impossibility. Adding a positive lower depth/scale bound would change the estimand and must be justified and applied to all methods prospectively.

The normalized Procrustes residual `F/(tr(CXX)+tr(CYY))`, when its denominator is nonzero, removes this common-scale incentive for comparison, but is a different optimization objective. The fixed-weight distance dual does not bound it. A common image-space score is preferable for selecting camera hypotheses.

## Minimal prospective acceptance and fallback

1. Treat the lift as one explicitly counted proposal. A simple initial policy is to require the declared solve status `completed`, finite recovered parameters/lift, a>0, and strictly positive corrected depths on the entire declared support. Any failure returns the unchanged incumbent and charges the attempted work. Freeze this policy before results; do not later admit only favorable iteration-limit cases.
2. Recover a proper SO(3) transform and apply the same existing cheirality, finite-projection and translation-direction validity rules used for every competing proposal. Evaluate actual geometry, not merely H or rank slacks. A planar cloud must not be rejected just because its centered 3D matrix lacks rank three. If a new numerical baseline-length rule is needed, declare a dimensionless threshold relative to centered cloud spread for all arms before testing; it would be a numerical availability rule, not a physical identifiability theorem.
3. Keep the original incumbent. Polish the new and control seeds with the **same downstream objective, support, stopping rules and resource allocation**. Select with one frozen, scale-invariant image-space criterion over fixed mass; do not use the relaxed objective or gap as the camera-quality selector. If a TC implementation includes additional depth factors or changing support, describe that exact objective rather than silently calling it this image-space selector.
4. Return the incumbent when no candidate survives or the common selector does not improve, with a fixed tie rule favoring the incumbent. Report the raw seed's availability and error separately from this fallback union. The union cannot be presented as a free improvement: charge moments, relaxation, recovery, selection and all added polishing.
5. Keep rank slacks, A, corrected-depth extrema, centered spread, alignment residuals and numerical dual diagnostics descriptive. Do not select a tightness threshold after seeing accuracy. If a tight-gap acceptance policy is desired, it is a separate predeclared variant whose rejected cases remain in the denominator.

## Fair controls and possible claim

Proper-Procrustes profiling from the same moments already has O(1) calibration evaluations. It needs the same proposal/polishing budget and selection rule. The unchanged TC baseline must remain visible; additionally compare a budget-matched incumbent/standard-start alternative against incumbent-plus-lift. Equal numbers of solver iterations are not necessarily equal time, so record total charged cost. A comparison against naïve pair enumeration alone establishes neither a strong speed claim nor a pose gain.

For an optimization-only claim, compare attained physical distance objective and numerical gap on the same fixed-weight problem. For a pose claim, compare the common downstream pipeline with all failures and costs retained. Neither a successful convex solve nor a feasible recovered seed is sufficient for the latter. The present mathematical review supports bounded artificial qualification of the initializer, not an expectation of real-data improvement or novelty.
