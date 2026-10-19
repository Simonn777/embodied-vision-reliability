# 0014: known pixel-noise control, not a new method

Source review: 2026-10-19. This prospective diagnostic asks whether a known
coordinate/noise-model difference or ordinary continued optimization explains
some remaining translation error. It does not integrate the failed 0012 scorer.
No new method, held-out improvement, or paper contribution is claimed.

## Primary prior and scope

Rydell, Torres and Larsson, **Revisiting Sampson Approximations for Geometric
Estimation Problems**, CVPR 2024, §2.1 Eq.21 gives the established covariance
form \(C^T(J\Sigma J^T)^{-1}C\). The relevant full-text section was read:
[official paper entry](https://openaccess.thecvf.com/content/CVPR2024/html/Rydell_Revisiting_Sampson_Approximations_for_Geometric_Estimation_Problems_CVPR_2024_paper.html),
[author manuscript v2](https://arxiv.org/html/2401.07114v2#S2.SS1).
Terekhov and Larsson's **Tangent Sampson Error**, ICCV 2023, already handles
central-camera projection derivatives and known measurement covariance;
supplement §D derives the scalar covariance denominator, and §E.1 describes
relative-pose optimization on \(S^2\times SO(3)\). Relevant main-paper and
supplement text was retrieved, rather than only the abstract:
[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Terekhov_Tangent_Sampson_Error_Fast_Approximate_Two-view_Reprojection_Error_for_Central_ICCV_2023_paper.pdf),
[supplement](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Terekhov_Tangent_Sampson_Error_ICCV_2023_supplemental.pdf).

Current PoseLib main documents a `tangent_sampson` option in a newer relative-pose
API. This diagnostic is pinned to the installed official **v2.0.5** release,
whose older wrapper is different. An improvement over that pinned objective
would not establish improvement over current main or a novel covariance method.
[Current README, inspected 2026-10-19](https://github.com/PoseLib/PoseLib#robust-estimation)

## Specialized PINHOLE formulas

Let \(x=K_1^{-1}[u_1,v_1,1]^T\), \(y=K_2^{-1}[u_2,v_2,1]^T\),
\(E=[t]_\times R\), \(p=Ex\), \(q=E^Ty\), and \(c=y^TEx\).
For each camera define \(f_j=(f_{xj}+f_{yj})/2\), and
\(a=(1/f_1+1/f_2)/2\). PoseLib's normalized-coordinate residual expressed
in comparable pseudo-pixel units is

\[
r_N=\frac{c}{a\sqrt{q_0^2+q_1^2+p_0^2+p_1^2}}.
\]

Unit isotropic noise in each original pixel coordinate induces
\(\Sigma=\operatorname{diag}(f_{x1}^{-2},f_{y1}^{-2},f_{x2}^{-2},f_{y2}^{-2})\)
in normalized coordinates. The known correction is

\[
r_P=\frac{c}{\sqrt{q_0^2/f_{x1}^2+q_1^2/f_{y1}^2+
                         p_0^2/f_{x2}^2+p_1^2/f_{y2}^2}}.
\]

This equals ordinary raw-pixel Sampson error with
\(F=K_2^{-T}EK_1^{-1}\). The coordinate/covariance equivalence is exact;
Sampson remains an approximation to geometric reprojection error. If all four
focal entries equal \(f\), then \(a=1/f\) and \(r_N=r_P\). Neither denominator
uses ground truth, confidence, or a learned noise estimate.

## Native continuation settings and comparability

In v2.0.5, `refine_relative_pose` unprojects raw pixels and multiplies the
user-specified loss scale by \(a\) exactly once. Its standalone default scale
is 1; the estimator wrapper instead sets 0.5 times the RANSAC threshold. Thus
the continuation explicitly passes the following settings:

```json
{"loss_type":"CAUCHY","loss_scale":0.5,"max_iterations":100,
 "gradient_tol":1e-10,"step_tol":1e-8,"initial_lambda":0.001,
 "min_lambda":1e-10,"max_lambda":10000000000.0,"verbose":false}
```

[v2.0.5 refinement wrapper](https://github.com/PoseLib/PoseLib/blob/v2.0.5/pybind/pyposelib.cc#L538-L558),
[estimator final refinement](https://github.com/PoseLib/PoseLib/blob/v2.0.5/PoseLib/robust.cc#L186-L203),
[option defaults](https://github.com/PoseLib/PoseLib/blob/v2.0.5/PoseLib/types.h#L59-L75).

Both new numerical controls use the same Cauchy objective, optimizer and fixed
inlier mask. The numerical kernel reports the dimensionless normalization
\(\sum_i\log(1+(r_i/s)^2)\), \(s=0.5\); canonical Cauchy multiplies this by
\(s^2=0.25\). Both are recorded separately. The native normalized objective is
\((0.5a)^2\) times the dimensionless \(N\) objective;
the minimizer agrees but gradient tolerances and damping are not automatically
equivalent across implementations. Preserve native statistics in their original
units and recompute both common objectives at every returned pose.
[v2.0.5 Cauchy implementation](https://github.com/PoseLib/PoseLib/blob/v2.0.5/PoseLib/robust/robust_loss.h#L104-L109)

All continuations start independently from the same sealed original pose and
the same saved pre-final inlier mask. Comparing \(P\) only against the original
pose would confound metric changes with extra optimization; compare it against
both native and same-optimizer \(N\) continuation. Preserve every pair and
failure. Any gain here remains a development diagnosis of an existing method.
The saved K matrices retain the existing camera-consistency check. Effective K
matrices for both numerical arms and objective calculations are constructed
from the exact authored camera parameters used by native PoseLib, because the
existing check tolerates small K-versus-camera differences. Both versions are
recorded. Recompute the dimensionless objective gradient
\(2J^T[r/(0.25+r^2)]\) at each pose; a solver's convergence flag alone does not
establish stationarity.
The interpretation check additionally records the undamped local Gauss-Newton
proxy: \(W_{ii}=1/(0.25+r_i^2)\), \(g=2J^TWr\), \(H=2J^TWJ=LL^T\),
\(D=\|L^{-1}g\|^2/[2\max(1,C)]\). A failed Cholesky or nonfinite computation
is explicitly invalid, without damping or a pseudoinverse fallback. This is a
local curvature diagnostic, not a global convergence certificate.
Cross-match correlated bias is not established by residual correlation alone:
pose error also induces coherent residuals, and fitting covariance to those
same residuals can absorb the signal being estimated.
