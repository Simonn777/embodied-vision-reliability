# 0028 normal-offset task assessment

2026-10-20. Source-only strategic review; no array, GT, model, fit, download or experiment. Read the current public CURRENT_STATE/NEXT_ACTIONS, the normal observation/prior notes, and the already installed MoGe helper. **Verdict: a bounded known-correction diagnostic is scientifically useful; a distinct method contribution is not yet established.** Do not build a new solver or extensive infrastructure around scalar least squares alone.

The public state closes 0027 and all earlier fixed failed variants. It requires a different observable and substantial two-source held-out, matched-cost improvement before a manuscript. Direct learned normals are a different cached observable from the failed finite-difference depth/AC input: 51 original-view normal maps already exist. They are a separate head, **not an independent sensor or statistically independent prediction**. The current mask is geometry validity, not normal confidence.

## What the proposed correction actually does

For calibrated axial ray r(u), raw axial depth d(u), and a global additive correction b, the surface is X=(d+b)r. A normal n supplies, for each pixel coordinate k,

    nᵀ ∂kX = (d+b)nᵀ∂kr + (∂kd)nᵀr = 0.

Writing α=nᵀ∂kr and β=dα+(∂kd)nᵀr gives αb+β=0. With fixed support and fixed nonnegative weights, the scalar LS answer is −Σwαβ/Σwα² when the denominator is informative. Positivity and numerical degeneracy are separate admission issues. The sign replacement n→−n changes both α and β, so this consistency objective is sign-invariant; unlike direct oriented-normal alignment it need not choose a view-facing sign. That is a useful implementation property, not a new theorem or pose solver.

Under the restrictive model d=s z+c with true normals, the desired b=−c leaves s z, whose normals are unchanged by positive s. This gives a real, falsifiable hypothesis about **global offset mismatch**. It does not establish that MoGe's actual errors are predominantly of that form. A nearly frontoparallel surface has small nᵀ∂kr and little offset information. Local wrong normals, derivative errors, edges, camera-intrinsic mismatch, and spatially non-affine depth errors can all bias the one scalar. Many correlated pixels cannot be counted as independent measurements. The normal loss can decrease when geometric accuracy becomes worse.

## Closest existing operation and the remaining narrow hypothesis

The pinned [MoGe helper](https://github.com/microsoft/MoGe/blob/925b8ed835a7a9cdb7578ba15c658a0afc969030/moge/utils/geometry_torch.py), `refine_depth_with_normal` lines 206–236, already converts normals into perspective log-depth gradients and iteratively refines an entire depth map using depth-dependent local weights and a data anchor. It uses fixed defaults of 10 iterations, damping 1e−3 and kernel size 5. The proposed scalar correction restricts this broad class of normal-guided surface correction to one global degree of freedom; normal-depth consistency and its elementary scalar fit are known ingredients.

The only concrete remaining task hypothesis is:

> When the two MoGe heads disagree mainly through a globally coherent axial offset, fitting that offset once per image can supply a useful calibration prior while avoiding the local shape distortions introduced by an unconstrained normal-guided map update. Fixed-offset pose estimation then trades calibration flexibility for lower estimation variance.

This is a specific bias/variance question, not a claim of extra geometric identifiability. It might fail because the depth head is already sufficiently calibrated, normals are systematically biased, errors are spatially varying, or the free-affine robust pose estimator already handles the relevant offset better. A scalar shared across the whole image can also spread one bad region's bias to every correspondence. The previous failures motivate caution but supply no observations of this proposed normal correction and do not predetermine its outcome.

## A fatal attribution confound to avoid

If corrected depths d1+δ1 and d2+δ2 enter an otherwise unrestricted affine-depth pose model, its parameters can absorb those constants:

    d1+δ1+b′ = d1+b,                 b=b′+δ1,
    a(d2+δ2)+c′ = a d2+c,            c=c′+aδ2.

With equivalent admissible physical depths, the modeled pose family is unchanged. A gain there can arise from initialization, scoring, masks, termination or a changed prior; it cannot by itself be attributed to newly constrained pose geometry. This would risk returning to closed 0026 seed-generation territory. To test the stated variance hypothesis, the normal-derived offset must actually constrain subsequent calibration, with that restriction disclosed and the free-affine alternative retained as a strong comparator.

The two-normal/two-point pose direction also remains the known [Cui et al., ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Cui_Polarimetric_Relative_Pose_Estimation_ICCV_2019_paper.html) construction; appending a depth calibration does not make it a new direction solver. Same-normal rotation/translation and normal-plus-geometric refinement are controls, not new contributions.

## Minimum comparison and minimum useful positive result

Before implementation admission, fix the original matches, input masks, intrinsics convention, image support, handling of degenerate/negative corrected depths, solver budget and tie rules. Offset estimation must use only the corresponding image's cached predictions, without query pose/GT feedback or choosing patches by pose outcomes. Failed offsets retain the prescribed fallback and denominator; no successful subset becomes the primary score.

A compact decisive comparison needs these same-information controls:

1. Raw-depth scale-only pose and current strong free-affine TC pose, plus the geometric baseline. This separates offset removal from simply removing nuisance parameters or ignoring depth.
2. The existing full normal-guided depth helper followed by the same downstream estimator and scoring. Its published-source defaults and any required masking/coordinate adapter must be fixed before outcomes, not weakened to favor the scalar restriction.
3. The same normal observations used by the established normal-guided pose/refinement construction. If its pose information explains the result, scalar depth calibration is not established as the useful mechanism.
4. A free-affine continuation using corrected depths, explicitly labeled as a reparameterization/initialization control, if that route is used anywhere in the candidate. It is unnecessary to invent another optimizer to perform this attribution.

Disjoint fixed spatial support can test whether an offset fitted in one region predicts the other region's normal-depth consistency, but this is only a mechanism check. It is neither depth truth nor an accuracy gate; pixel observations are correlated and the global median/gauge must be disclosed. A positive consistency change alone does not justify a pose campaign or manuscript.

For continuing beyond a tiny cached development diagnostic, require **actual failure-inclusive pose benefit**, fixed prospectively: at least +2 AUC@5 percentage points over the strongest prespecified control, positive margins in at least 2/3 development places, no place worse by more than 1 point, and AUC@20 loss no more than 0.5 point. These reproduce the existing project's development scale of benefit rather than inventing a lower bar after failures. Report every comparator and total processing cost, including offset fitting, helper/refinement, unavailable cases and the shared original frontend. Merely improving raw scale-only depth while losing to free-affine TC or the normal helper is not enough. The already used places remain development forever; they cannot supply independent confirmation.

If the global restriction fails against the normal helper or gives no meaningful benefit over strong same-normal/free-affine controls, close this formulation rather than adding spatial weights, per-region offsets or confidence gates on the same labels. If it passes, the defensible next effort is independent-source validation and a mechanism ablation; novelty still requires a precise demonstrated advantage beyond a cheaper restricted implementation of an existing correction. There is no current basis for a new-solver or paper claim.

## Source boundaries

Read public `CURRENT_STATE.md` and `NEXT_ACTIONS.md` at their current 0027-closed state, the two private 0028 source notes, and local pinned MoGe source. No new literature search was needed. Local helper file SHA256: `2f8d5de7d671af16d25fe13c555fd73855d8447bc086bb23aa42e859b303fb05`. Normal observation contract SHA256: `765cc52af9fb4ac3cad94b6e85641959cf0c7a8ff134ed906f9b977603ff1103`. This assessment does not alter any closed experiment or itself authorize execution.


## Minimal support-conditioned design (source-only follow-up)

Historical design suggestion below: the final [0028 protocol](../experiments/0028-normal-information/protocol.md)
supersedes its saved reference-filtered E-mask suggestion. It recomputes E on
the fixed query-only matches, keeps that complete unmodified E baseline, and
uses the resulting common support. The source-only suggestion is retained to
make the change explicit; it is not the executed allocation.

For the parent's contemplated 24 A0→B queries, a new RANSAC is unnecessary. Reuse the exact saved native E inlier masks and their `retained_query` feature-ID mapping. Source `0023-mapfree/geometry_ba.py:155–174` returns pose, mask, input count and options; `study.py` saves these under `native_E`. The inputs had already passed the reference-track filter. Therefore this is explicitly a **reference-filtered, E-inlier-support-conditioned preliminary comparison**, not a new complete robust estimator, query-only system, or independent dataset result. The 27 unique anchor/query images suffice; do not recompute 51-image or 120-edge frontend work.

Construct the fitting support once from that mask and **original** valid sampling locations. Any required common depth/normal interpolation eligibility must be frozen before offset/helper outputs, with original and retained counts recorded. Apply the same support to every depth/normal arm. Correction-induced negative depth, nonfinite values or insufficient rank make an arm unavailable/fallback; they must not shrink its support until it looks good. Keep all 24 queries in evaluation. Also retain the original native E prediction on its original larger support as an unweakened benchmark; do not handicap geometry to create a benefit. Offset fitting uses per-image predictions alone and is shared across the eight queries containing an anchor.

Raw depth, scalar-corrected depth and full-helper depth can use the identical closed-form positive-scale proper similarity fit, without nonlinear calibration or sampling. For X=d1*r1 and Y=d2*r2 on the fixed support, center X,Y; proper SVD maximizes τ=Σ XcᵀRᵀYc. For the convention aY=RX+t, solve a=τ/Σ||Yc||², t=a*mean(Y)−R*mean(X). Require positive finite scale and sufficient geometric rank; never accept an improper reflection. This is ordinary Procrustes/Umeyama-type fitting, not a new solver. It avoids a second hand-tuned nonlinear optimizer and gives the three depth versions exactly the same estimator. Save both initial poses and any fixed common refinement output.

For the stronger free-affine comparator, the existing `0026-calibration-moments/native_bridge` calls original `refine_monodepth_relpose` on explicitly supplied calibrated coordinates/depths, without RANSAC or support refresh. Its current ABI is `NativeRefiner.refine(x1,x2,d1,d2,R,t,a,b,c,focal1,focal2,max_iterations=100)`. **It hardcodes `refine_shift=true`; it is not a scale-only refiner.** Reuse it for the free-affine TC control, preserving its native visibility behavior/diagnostics and full costs. A continued local refiner from a single arbitrarily poor seed is not the strongest full RePoseD estimator: label it honestly. A small fixed union of raw- and corrected-Procrustes starts, selected by one predeclared geometry score without labels, is a more credible local control and retains both failed calls. Do not silently call the full pose-estimation API with an initial pose, as its robust pipeline may change support and initialization.

Use the same fixed-support proper normal alignment and known-R translation with both signs/ordinary cheirality for the normal baseline. Its established normal-plus-Sampson refinement is the appropriate same-information continuation; a raw normal initializer alone is a weak control. Any shared pose refinement, its budget and selection score must be identical where applicable, and outcomes before/after refinement retained. If adding this strong baseline requires a new optimizer, first qualify that small known-residual implementation rather than inventing RANSAC infrastructure.

This can isolate whether global restriction has a task signal at low additional cached-input cost. It cannot establish outlier rejection, full-pipeline matched-time superiority or held-out performance. The threshold above is a prospective continuation criterion only; no computation is authorized by this design note.
