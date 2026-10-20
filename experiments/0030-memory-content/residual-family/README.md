# Pure residual-family preparation — not executed

`family.py` imports only `math`; callers pass Torch. It performs no model, native inference, file access, teacher optimization, student training or import-time work. `test_family.py` contains eight manufactured arithmetic/autograd checks, including independent NumPy least-squares comparisons. The first execution belongs to the parent; no tests or module imports have been performed during preparation.

For a single causal memory `[1,C,H,W]`, `prepare(memory, rho, torch=...)` computes detached float64 means and population variances, then stores the centered unit-RMS spatial basis in float32. Exact-zero variance disables both coefficients. The coefficients are `[C,2]` float32 energy coordinates `(a,b)` with residual `a*u+b` on active channels. `expand` never silently projects.

The measured two-column basis Gram `G_c` is retained, including measured means and finite-precision nonorthogonality. The common pre-BF16 energy constraint is

`sum_c q_c.T @ G_c @ q_c <= B**2`, with `B = sqrt(sum_c sigma_c**2) * rho`.

`project` uses `B*q/sqrt(max(energy,B**2))`, with the maximum **before** the square root and an explicit zero-radius branch. Its active-coordinate derivative at zero is the identity for positive radius. Inactive coordinates always have zero output/gradient. This is radial projection in the measured-energy metric, not Euclidean coefficient projection when `G != I`. Radius, shape, dtype and finite-value errors are explicit.

`least_squares_target` solves the measured per-channel 2×2 normal equations in float64 for a detached GT-memory difference, casts coefficients to float32, then applies the same energy constraint. In exact arithmetic this radial projection gives the constrained LS solution because both its quadratic objective and bound use `G`. The returned float32 coefficients/expanded residual introduce rounding; diagnostics expose actual pre-cast tensor energy, Gram energy, and their difference. The manufactured comparisons use `atol=2e-7, rtol=2e-6`; the expanded energy check permits relative `2e-6` rounding. There is no bit-exact LS or post-BF16 energy guarantee.

**No KD objective or teacher optimizer is selected here.** With this measured basis, spatial residual MSE is `sum_c Δq_c.T @ G_c @ Δq_c / C`; Euclidean coefficient KD instead uses `sum_c ||Δq_c||²` with its stated normalization. They are not exactly equal when `G != I`. At a zero-output student and inactive projection, Euclidean KD to a Euclidean one-step teacher gradient has the previously stated local gradient proportionality. Using spatial MSE with that teacher introduces `G`; reproducing the Euclidean direct gradient would instead require the corresponding inverse-metric teacher step on the active subspace. `G` being numerically close to identity is not an exact equivalence theorem. These choices remain for the later campaign.

Prospective caller responsibilities: choose `rho`, supervision, coefficient/teacher losses, update schedule and training budget before real targets; use the same family and bound in candidate and matched controls; preserve the native storage convention and report its separate quantization effect. No head or scientific execution is included.
