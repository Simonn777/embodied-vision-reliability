# 0027 prospective sensitivity falsifier

Status: proposed manufactured qualification, not executed or an experiment admission. This tests whether the shifted-depth tangent branch can yield a useful pose before a RANSAC or learned-derivative frontend is built. It does not establish novelty, real derivative accuracy, or a one-point sampling advantage. All choices below must be reviewed and source-frozen before generation. Maximum operation: 30 seconds, four CPU threads, 8 GiB, no network, new packages, images, models, benchmark arrays, or labels.

## 1. Fixed physical observations

Both cameras use zero-centered pixel coordinates and K=(fx=fy=500, cx=319.5, cy=239.5), 640×480. Let δ be the displacement from the source center in **normalized ray coordinates**. A surface supplies X=z1(δ)r1; Y=RX+t supplies the target pixel and axial z2. Differentiate this projection analytically for exact A and depth gradients; target gradient follows the chain rule through A. All solver derivatives are then expressed per **pixel**. No independently invented affine matrix or target gradient is allowed.

Two utility fixtures, fixed without numerical screening:

| case | source center | z1(δ) | R | t |
|---|---|---|---|---|
| G | (0,0) | 1+δx | [[2,−2,1],[2,1,−2],[1,2,2]]/3 | (1/6,2/3,1/3) |
| M | (.12,−.08) | 2.5+.8δx−.6δy+.1δx²+.15δxδy−.1δy² | Exp([.12,−.18,.08]) | (.25,−.05,.10) |

Each has two raw-depth calibrations, with d1=z1−b and d2=(z2−c)/a: **scale-only** (b,a,c)=(0,1.3,0), and **shifted** (.25,1.3,−.20). Generate a fixed 3×3 source pixel lattice with offsets {−16,0,16}², row-major. All nine physical correspondences and both depth observations are available to every method. This is explicitly a neighborhood experiment, not nine independent observations manufactured from one match.

There are four exact-derivative clean rows and four noiseless fitted-derivative rows. For the latter and every noisy row, retain the measured center/depth; fit A by ordinary least squares from source pixel differences to target pixel differences, with the center fixed. Fit each depth gradient from that image's pixel differences to depth differences, also with the center fixed. Use all nine samples (center zero row included); require machine-rank two. This separates the analytic solver from finite-neighborhood estimation bias.

Two additional exact descriptive cases use M's shifted calibration: **P**, with t_x,t_y chosen to make the transformed center's x,y zero and t_z=.10; and **F**, replacing z1 by constant 2.5. P exposes the invertible-M branch limitation. F has a calibration scale family: for k∈{.8,1,1.2}, b'=k(d1+b)−d1, a'=ka, c'=kc and t'=kt leave R and translation direction unchanged. Neither diagnostic is required to recover every calibration parameter, and neither enters the utility gate. Their failures must not be relabeled universal pose non-identifiability.

## 2. Small, shared measurement-noise cohort

Use NumPy PCG64 seed 20261019. Loop case G,M, calibration scale-only,shifted, then realization 0…15. At each iteration draw standard normal pixel noise of shape (2,9,2), followed by depth noise of shape (2,9). The same draws generate both regimes:

- L: pixel standard deviation .1 pixel; raw-depth standard deviation .001 times the true center axial depth divided by that image's true raw-depth scale (1 or 1.3).
- H: .5 pixel and .01 times that same depth unit.

Perturb both images' measured pixel coordinates and all raw depths, then recompute A and gradients by the common fitting rule. Do not perturb the derivatives again. This is a declared errors-in-variables stress distribution, **not** an empirically established MDE noise model. Save the draws, exact/noisy observations, fit matrices and resulting derivatives. Total: 128 noisy rows + 8 clean rows + 2 descriptive rows = **138**. No replacement of a difficult fixture or draw.

## 3. Methods with common information and selection

Use the convention in `affine-tangent-math.md`: U=U0+bU1, V=aV0+cV1, RU=V; calibrated rays have third coordinate one. The existing reviewed quartic solver supplies every real, physically admissible branch, with its original residual/rank checks and explicit rejection reasons. A singular elimination matrix is a branch-unavailable outcome, not a zero-error estimate.

**T — shifted tangent.** All quartic branches; no truth-selected root or post-hoc fallback.

**S — established scale-only 1AC+D.** The ECCV2020 SVD solution: SVD(V0U0ᵀ)=PΣQᵀ, D=diag(1,1,det(PQᵀ)), R=PDQᵀ, a=tr(ΣD)/||V0||F², t=a d2 r2−R d1 r1, b=c=0. Also retain the paper's ordered-frame alternative, using the same positive least-squares scale for its R. Select between these two by the common score, charging both. The axial-ray implementation is a coordinate translation of the published point/tangent geometry, not unchanged substitution into a unit-bearing/range formula.

**V — AC→three virtual points.** For each fixed ε∈{.25,1,4,16} pixels, use source center and center+εe1/εe2; target center and center+εAe1/εAe2; depths d1+εg1ᵀej and d2+εg2ᵀAej. Call the existing affine-depth three-point solver and retain **all** returned branches. The V arm selects across the full four-spacing union by the common score. Do not select ε using pose error. Report each spacing separately descriptively. These points share exactly the same noisy center, A and gradients as T; add no independent noise. Their finite-ε lifted point differences contain ε²gJ terms. Clean exact-derivative rows expose that truncation/cancellation tradeoff without claiming a new geometric constraint.

**D — original physical three-point control.** Use four fixed triangles, each consisting of the center and the original measured samples at (σx16,0),(0,σy16), σx,σy∈{−1,+1}. Retain every native branch from all four triangles and select by the same score. Thus a favorable triangle is not silently chosen after seeing truth; all four calls are charged. This control uses actual neighborhood measurements rather than fabricated independence.

For every arm, require finite proper R, nonzero t, a>0, and positive corrected depths on **all nine** observations. Preserve native branches rejected after the native solver's internal scale/shift refinement. On the same nine observations, compute forward/backward pixel reprojection from X=(d1+b)r1 and Y=(ad2+c)r2; a behind-camera/nonfinite projection contributes the full cap. The score is the average of 18 values min(||pixel residual||²/4,1). Select its minimum, then fixed generation order on exact ties. This selection is **in-sample**, not held-out validation. Retain score, all root counts/reasons, all calibration values and all poses. No local pose refinement, new robust estimator, truth-assisted root selection, or refitting to improve an unfavorable row.

## 4. Outcomes and prospective necessary gate

For each selected pose report rotation error and **directed** translation angle; task error is their maximum. Unavailable, invalid, or undefined-translation outcomes receive 180°, never exclusion. Report useful yield at ≤5°, full failure-inclusive errors and empirical AUC5 = 100 mean(max(0,1−error/5)). Separately report the best-root error as a labeled **oracle upper bound** to distinguish generation from selection failure; it cannot pass a gate. Report calibration residuals/condition diagnostics separately from pose errors.

Proposed deliberately small necessary screen:

1. T's four exact-derivative G/M clean selected poses must have task error ≤1e−4 degrees; P/F are excluded from this correctness requirement by declaration.
2. For **each** shifted G/M case, selected T must yield ≥12/16 useful poses in L and ≥8/16 in H. This rejects a branch that is already too fragile under these particular modest input perturbations; it is not a universal accuracy theorem.
3. On each scale-only case/regime, T's useful count may trail the S control by at most 2/16. On at least one shifted H case, T must exceed S by at least 2/16; otherwise this screen gives no reason to add offsets.
4. To justify a distinct initializer beyond the existing same-input conversion, T must not trail either V or D by more than 2/16 in either shifted H case, and must exceed max(V,D) by at least 2/16 in at least one. If it merely reproduces V, retain that equivalence and close this proposed accuracy mechanism; do not manufacture a speed claim from a handful of small native calls. A separately specified efficiency hypothesis would require a different matched-cost study.

These coarse gates are a **proposal for pre-run review**, not an established statistical significance rule. All 16 outcomes share each fixed geometry; no population confidence interval or real-data conclusion is justified. Failure closes this fixed sensitivity candidate without adjusting gradients, patch radius, ε set, fixture, or noise. Passing admits at most a bounded real-frontend question, not RANSAC efficiency, a novel solver priority claim, or a paper.

## 5. Existing callable baseline and receipts

Source-only verified at PoseLib commit `6a636b1d84273d5e3ba97676463476569611572a`:

- `pybind/bindings/solvers.cc:207–214,359–360` exposes `poselib.monodepth_pose_3pt(x1,x2,d1,d2)` → list of `MonoDepthTwoViewGeometry`. Provide exactly three calibrated **z=1** rays; this solver reads x/y directly and later multiplies the full ray. It does not normalize arbitrary homogeneous input.
- `PoseLib/solvers/relpose_monodepth_3pt.cc:134–184` returns `.pose.R/.pose.t`, `.scale`, `.shift1`, `.shift2`; convert a=scale,b=shift1,c=scale*shift2. The source internally refines its scale/offset candidates. Its header's “shift discarded” comment is stale relative to the implementation/binding.
- Local source root: `work/0016-preparation/inputs/PoseLib-6a636b1d84273d5e3ba97676463476569611572a`; runtime: `work/0016-preparation/runtime`. Binary SHA256 `3228d488ca11f966378b6b7657ee6ebc5716da474c7ae88fccff7ab433a6981d`, source solver SHA256 `6687da6a6f37f30f65aeb0223bc0853040ade8c476ac342bf9b35e3e4f7ccb04`. No new native bridge, dependency or compilation is needed. Binding callability is source-verified only; any later ABI qualification failure is retained, not hidden.

Save source/binary hashes, exact inputs/draws, every attempted branch, selected outputs, numerical warnings, complete timings, operation limits and exit status. Charge all four V calls, all four D calls, both S candidates and neighborhood fitting; no cached-frontend speed claim. A 30-second limit produces an incomplete qualification, not a reduced or selected cohort.

Primary overlap: [1AC+D, ECCV2020 §§2–3](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570613.pdf); [semicalibrated 1AC+monodepth, ECCV2024 §§2–3](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05645.pdf); [MADPose, CVPR2025 §4.1](https://arxiv.org/html/2501.05446v3); [RePoseD, ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Ding_RePoseD_Efficient_Relative_Pose_Estimation_With_Known_Depth_Information_ICCV_2025_paper.pdf). The distinction under test is offset-aware tangent estimation versus the existing differential/point constructions; none of differentiation, Gram invariants, quartic elimination or synthetic point conversion is claimed new here.
