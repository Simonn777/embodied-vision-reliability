# 0016: independent review of a fresh-development pose diagnostic

2026-10-19. Source/design review only. No coordinates, calibration values, depth, images, ground truth, estimator outputs, or new scientific runs were accessed for this review. The parent selected the design below before its execution. It is a bounded test of existing estimators and search stability, not a new method or evidence of a gain.

**Recommendation: proceed with this diagnostic once its implementation and artificial compatibility checks pass.** It answers a concrete question that the earlier sixteen easy, overlapping pairs cannot answer: does a fresh sample expose materially different outcomes from current official geometry or deterministic restarts? It does not require a new appearance/depth method first.

## Allocation and minimal comparisons

Select 64 keys from the 600-key scene0015 development population, excluding the sixteen keys in `configs/0012-development-manifest.json`. Before any numeric access, fix one namespace string, hash `namespace + NUL + exact_key` with SHA256, sort by `(digest, exact_key)`, and take the first64. Record the namespace, eligible-key digest, exclusion list, and resulting ordered manifest. Do not inspect inputs or labels to replace an inconvenient pair. Keep scene0022 untouched.

These are fresh *pair values*, not fresh independent scenes. Image identities may overlap both earlier pairs and other selected pairs; disclose the image-incidence graph from names. Do not claim 64 independent replications or a population confidence interval. The sample is useful for a descriptive development decision.

Use all supplied10000 original correspondences and authored cameras for each selected pair. No photometric replacements, image reads, confidence filters, supplied initial poses, or cross-backend pooled candidate method. Compare:

| Backend | Role |
| --- | --- |
| Official2.0.5 normalized Sampson | Historical native reference, original binary preserved |
| Pinned current3.0.0 source, normalized Sampson | Controls for upgrading version/defaults/toolchain |
| Same current3.0.0 source, Tangent Sampson | Isolates the current `tangent_sampson` option against the previous row |

Each backend gets exactly four candidate calls, seeds0,1,2,3 in that order: at most768 calls across64 pairs. Do not add a long single-seed arm after seeing results. All64 rows remain in every evaluation; invalid/timed-out estimates have explicit coverage status and the predeclared180-degree all-pair penalty. Broken input/source bindings are integrity failures, not scientific estimator failures. A global abort is an incomplete experiment, not permission to report only completed easy pairs.

## Exact API/default distinction

Current source pin: `6a636b1d84273d5e3ba97676463476569611572a`; locally inspected `VERSION.txt` says **3.0.0**. The parent acquired this source; this review did not acquire or build it. Its wrapper takes `estimate_relative_pose(x1,x2,cam1,cam2,opt,initial_pose=None)`. The old release takes separate RANSAC and bundle dictionaries. The wrappers are not interchangeable.

| Setting |2.0.5 call | Current3.0.0 call |
| --- | --- | --- |
|1px threshold | `ransac_options.max_epipolar_error` | `opt.max_error` |
| Seed, min/max iterations | flat RANSAC dictionary | `opt.ransac.seed`, `.min_iterations`, `.max_iterations` |
| Final bundle options | separate empty dictionary | `opt.bundle={}` |
| TS switch | absent | `opt.tangent_sampson=false/true` |
| Initial pose | none supplied | `initial_pose=None` |

Retain min1000/max100000, success probability0.9999, dynamic multiplier3, progressive samplingfalse and max-PROSAC100000 explicitly in their correct locations. Use official final defaults for each version: old Cauchy0.5px/gradient tolerance1e-10 versus current Cauchy1px/gradient tolerance1e-12. Current defaults additionally include relative-cost tolerance1e-10, max100 iterations, step tolerance1e-8, initial/min/max damping1e-3/1e-10/1e10, Nielsen updates and Levenberg damping. Therefore old-versus-current measures a combined version/default/toolchain change; it does not isolate TS. Current-normalized versus current-TS does isolate the option, including its official scoring and refinement behavior. [Pinned wrapper](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/pybind/bindings/estimators/relative_pose.cc), [parser](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/pybind/helpers.h), [defaults](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/PoseLib/types.h).

Unknown dictionary keys are not a reliable error mechanism: legacy top-level seed/threshold keys can be ignored by the new parser. Artificial compatibility checks should inspect the actual call dictionary and argument count, use the exposed `RansacOptions`/`BundleOptions` factories to verify parsed values/defaults, and check tiny fixed iteration counts on synthetic inputs. Do not require different poses from different seeds: legitimate convergence can produce identical poses. The factories share the parser helpers; there is no exposed `RelativePoseOptions` echo factory. [Factories](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/pybind/bindings/types.cc).

## Selector and timing make the comparison interpretable

The parent chose one independent, final-pose selector for every backend: raw-pixel F-Sampson truncated at1px. For each valid returned pose, form \(F=K_2^{-\mathsf T}[t]_\times RK_1^{-1}\) and

\[
S_P=\sum_i\min\left\{\frac{(\bar q_i^\mathsf TF\bar p_i)^2}
{(F\bar p_i)_x^2+(F\bar p_i)_y^2+(F^\mathsf T\bar q_i)_x^2+(F^\mathsf T\bar q_i)_y^2},1\right\}.
\]

Freeze degenerate/nonfinite handling and resolve exact score ties by earlier seed. Select only among that backend's available candidates. Preserve the native translation sign; this sign-invariant epipolar score cannot distinguish \(t\) from \(-t\), so it is not a replacement for the native cheirality decision. Record final cheirality diagnostics if implemented, without silently changing the selector.

Call this the **independent pixel-Sampson selector**, not the native TS objective. Current TS uses unit rays \(d\) and unprojection Jacobians \(M\):

\[
r_T^2=\frac{(d_2^\mathsf TEd_1)^2}
{\|M_2^\mathsf TEd_1\|^2+\|M_1^\mathsf TE^\mathsf Td_2\|^2}.
\]

Its pose score includes an official cheirality test. Ray normalization changes derivatives away from zero residual, so substituting raw F-Sampson is not an exact general identity. Native RANSAC score/mask are also produced before final refinement: never select final outputs using stale `model_score`. Rescoring final poses with native normalized/TS objectives is useful secondary diagnosis if implemented correctly. [Official scoring](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/PoseLib/robust/utils.cc), [final-refinement ordering](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/PoseLib/robust.cc).

Record each candidate's charged cumulative completion time: input preprocessing plus all preceding calls and selector computations, including failed work. Freeze the common budget grid **0.1,0.25,0.5,1,2 seconds**, and separately report full four-call results. At a budget, select only the completed prefix; no completed candidate means unavailable/penalized. Reuse the same saved calls for these predeclared prefix analyses—no reruns or timing-driven extra candidates.

This is an **offline equal-time-allowance analysis**, not a real-time implementation or proof of exactly equal spent computation. The native API has iteration caps, not a wall-clock callback; a call that finishes after a budget is excluded from that prefix even though it was actually executed. Record the full execution cost too. Equal four-call or iteration ceilings alone are not runtime matching because adaptive stopping and local-optimization costs vary. Serial execution, fixed cyclic backend order across pair indices, one native/BLAS thread,8GiB limit and30-minute global cap reduce avoidable measurement confounding. One bounded execution does not establish precise speed ratios.

## What to save and what each outcome means

Before reading the selected labels, seal all64 rows, all candidate poses/statuses/times, effective options, common final scores, selection indices at every budget, and source/binary bindings. This should permit all following analyses from saved outputs, without another estimator run:

- All-pair AUC5/10/20, directed translation and rotation errors, coverage and budget-prefix performance for all three backends; seed0 is the single-call reference.
- Within each backend, all six candidate-pair rotation/translation disagreements; final score spread; actual RANSAC iteration/refinement counts with their pre-final scope labeled.
- Per-pair changes from old-normalized to current-normalized and current-normalized to TS; all candidate errors shown after sealing. A secondary oracle best-candidate bound, if reported, must be named as unavailable to deployment and cannot count as progress.
- How often the fixed selector improves/worsens on seed0, and whether lower score accompanies lower or higher pose error. Large pose variation at almost equal score indicates objective ambiguity within the tested candidates; it is not direct evidence of correspondence truth.

| Saved outcome | Supported next decision |
| --- | --- |
| Current normalized improves old | Update the baseline; do not credit TS or invent a method |
| Current TS improves current normalized | Known TS pipeline matters on this sample; adopt it as a stronger control |
| Fixed multi-seed selector improves seed0 within a common budget | Tested search instability is actionable; quantify the ordinary restart baseline before new methods |
| An oracle candidate is good but the fixed selector is poor | Candidate selection/objective mismatch deserves attention; oracle gain is not deployable improvement |
| All tested candidates are close yet wrong | Restarting this search failed to expose a remedy; input bias is only one remaining explanation |
| No useful improvement or failure regime | Close this diagnostic without paper claims or threshold rescue |

Stable wrong outputs do **not** prove biased/noisy matches: shared local minima, calibration error, reference-pose error, residual-model mismatch and weak geometry remain alternatives. This bounded diagnostic can falsify the proposed simple remedies; it cannot identify a unique cause from seed stability alone. It also cannot establish generalization from one development scene.

## Separate mathematical assessment: ordinal depth under monotone calibration

This is not needed for the diagnostic and authorizes no depth acquisition or implementation. Suppose per-view predictions \(\hat d_{vi}\) relate to true positive depths through unknown strictly increasing functions \(f_v\). For a candidate pose, triangulate depths \(z_{vi}(R,t)\). With distinct finite prediction values, an increasing function interpolating all predicted-to-triangulated pairs exists **if and only if** their orders agree:

\[
\hat d_{vi}<\hat d_{vj}\quad\Rightarrow\quad z_{vi}(R,t)<z_{vj}(R,t).
\]

Necessity follows from monotonicity; sufficiency follows by increasing piecewise-linear interpolation through the sorted pairs. Inverse-depth orientation, ties and noisy comparisons require a fixed model, not a sign selected using ground truth. Check order separately in each camera: camera motion does not generally preserve within-view depth order across views.

This is weaker than affine depth and is robust to a larger class of calibration distortions. It can reject alternative pose basins whose triangulated order contradicts a reliable predictor. It also has an important limitation: if a finite set of strict order inequalities holds with positive margins and triangulation is continuous/nondegenerate, it holds in an open neighborhood of the pose. A pure rank-violation count is locally constant there. Consequently, order alone gives no local metric correction within that neighborhood; smooth margin losses add assumptions beyond an unknown monotone map. Positive translation scaling preserves all ranks, so absolute scale remains unobservable.

Wrong match identities, nearly equal depths, occlusion boundaries, local rank inversions and correlated scene hallucination can destroy the proposed signal. An arbitrary per-point rather than global monotone mapping removes the order constraint. A future artificial mechanism test would need competing plausible poses that differ in predicted ranks under known nonaffine monotone distortions, plus inverted/noisy-order countercontrols; fine-error gains within one rank cell would contradict a purely ordinal explanation. No such test ran here.

Ordinal invariance itself is established: Zoran et al., *Learning Ordinal Relationships for Mid-Level Vision*, ICCV2015, explicitly motivates monotonic invariance for depth estimation. This review read its institutional abstract/metadata only, not the full paper, and does not establish novelty of ordinal camera-pose selection. [Primary institutional record](https://dspace.mit.edu/entities/publication/64f07395-3d99-4fcd-9fbd-f673056b2671). The mathematical observation above is an elementary identifiability argument, not a new theorem or a reason to delay the bounded baseline diagnostic.
