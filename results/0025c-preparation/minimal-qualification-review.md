# Minimal qualification before an image-based 0025c campaign

Prospective source/mathematics review only. No observations, images, saved scientific results, model calls or solver execution. This supplements `critical-feasibility.md` and does not allocate data or alter a frozen experiment.

**Root scope clarification, before execution:** the immediate Stage A is an all-inlier initializer-noise test only. No renderer, RANSAC or refiner is requested. Evaluate all 70 four-subsets of the fixed eight-point construction without selecting or averaging their poses, all 28 angular-only two-subsets, plus all-eight-point fits of the three controls. Uniform-subset useful-angle yields include failures in their denominators. Subsets share observations and noise, so they are not independent experimental units. The two-point arm and extra optimistic headroom gate were added prospectively after the algebraic control review; see `angular-two-point-control.md`. The common scorer/refiner discussion below applies only if a later robust-estimation stage is separately admitted. A common unconstrained linear refit now would erase the initializer differences being tested.

**Narrow recommendation:** establish useful translation-direction conditioning and a real sampling advantage on a small predeclared manufactured problem before building RGB-D mesh/ray-casting or inertial-data acquisition machinery. The existing 0025 nonplanar points, exact implicit row projection and declared critical configurations are sufficient starting code; their previous rank outcomes are not a new efficacy test.

## Initializers and controls, one observation/scoring contract

1. The candidate affine-depth RS initializer with four matches.
2. The **same** affine-depth RS initializer fitted to eight matches, overdetermined. This inexpensive control distinguishes a minimal-sample advantage from noisy interpolation of twelve unknowns.
3. The exact same-rotation, same-angular-motion depth-free RS homogeneous nine-vector control described in the companion memo; its eight-equation counting bound still needs rank qualification.
4. A same-rotation **angular-only RS affine-depth** control: retain the known `Q` and unnormalized dewarped rays, fixing only readout translation to zero. The six unknowns are `(t,b1,a2,b2)`. It reduces to GS affine alignment only when row angular motion is also zero. This stronger control identifies whether modeling readout translation is useful under the chosen motion.
5. The same angular-only affine initializer fitted to all 28 two-subsets. Its attainable two-point rank must not be hidden by using only an eight-point implementation; the complete eight-point positivity check remains unchanged.

Do not present the third control as an author implementation without one, or call the known-R branch full four-point pose. The 2017 model with both angular and linear readout motion is named **uniform**, not its zero-angular-velocity **linear** model.

For a later robust-estimation stage only, candidates need a common independently evaluated geometric score, consistent cheirality/sign resolution and the same final objective and optimization budget. Stage A has no such scorer/refiner. In that possible later stage, the depth-free initializer can enter a depth-aware local stage without an unfair handicap: from its arbitrary-gauge `u=(t,v1,v2)`, define `k_i=t-R B1_i v1+B2_i v2` and solve the four-unknown calibration bridge

\[
[-k_i,\;-R\bar r_{1i},\;d_{2i}\bar r_{2i},\;\bar r_{2i}]
[\lambda,b_1,a_2,b_2]^T=d_{1i}R\bar r_{1i}.
\]

This uses a shared estimated support, never an oracle inlier mask. Sign/gauge and positive-depth conditions must be explicit. A bridge failure counts against the method; it cannot silently remove the case. Sharing the depth input in scoring/refinement is necessary to attribute an advantage specifically to minimal initialization. Native complete methods can later be separate practical controls.

## The smallest informative manufactured cases

Use a fixed generic nonzero-readout-translation construction and retain the existing identity/near-identity, vertical-flow, and equal-row critical cases. They test different failure mechanisms and should not be collapsed into a single average success rate. An arbitrary set of handpicked full-rank matrices is insufficient: observed pixel y and row time must satisfy the declared exposure model.

Separate exact-input qualification from fixed input perturbations. In the noisy branch, recompute the entire design: noisy y changes the inferred exposure time as well as the ray; rate, middle-rotation and timestamp perturbations are separately declared. A learned-depth perturbation model tests that specified model only, not the network's empirical errors. No observation should be filtered by true error or by the eventual winning method. Use the same fixed match pool and sampling schedule, recording every sample rejection, score call and refinement call. The real learned-depth assumption will still require a later image test.

Do not build a large RANSAC framework for the first decision. First compare clean-sample perturbation sensitivity and task ambiguity. Only if the four-point branch remains usable should a small fixed contaminated-pool check test whether fewer-sample proposals actually offset their increased sensitivity. A fixed common proposal budget plus measured operation counts is enough for this admission; an end-to-end runtime claim is not.

## Measure the task, not merely matrix rank

For a parameter null vector with translation component `δt`, the observable task perturbation is

\[
\delta\hat t=(I-\hat t\hat t^T)\delta t/\|t\|.
\]

Report whether null directions reach this two-dimensional task. The existing vertical-flow degeneracy can affect nuisance variables while leaving translation direction unchanged; the identity-R four-point example can change the task. A condition number after column balancing must record the balancing and velocity/depth units. It is not itself a statistical uncertainty estimate.

With a noisy coefficient matrix, first-order parameter change is `A⁺(δb−δA β)`, not merely `A⁺δb`. Report actual directed translation error from the fully rebuilt noisy design, and preserve nonpositive depth, zero translation, singular-system and solver failures in the denominator. Folded translation angle would hide an incorrect cheirality/sign result. Fixed-R rotation error belongs to the shared input and must not be credited as an estimated rotation gain.

## Decisive stop conditions

- Four-point gains appear only with exact/oracle rotation, disappearing under the predeclared sensor perturbation: no measured-IMU campaign justified.
- Its lower clean-sample probability is overcome by much worse conditional pose sensitivity, or the eight-point affine fit wins at the same charged budget: no demonstrated four-point efficiency mechanism.
- In a later separately admitted robust stage, common refinement/scoring removes the difference: describe initialization parity, not a new accuracy result. This cannot be evaluated in Stage A.
- Only the angular-only RS control loses while the exact depth-free RS control is equally strong: evidence supports modeling readout translation, not the new depth-assisted initializer.

These are interpretation rules, not post-outcome parameter changes. Numeric admission thresholds and the finite fixture/noise schedule must be prospectively fixed by the root protocol before execution. No extra data, model, mesh renderer or acquisition code is needed for this decision.
