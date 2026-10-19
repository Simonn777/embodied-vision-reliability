# 0018: additional official control requirements

Source-only inspection2026-10-19, before0018 build or traces. No model, dataset
payload, estimator, installation or benchmark was run for this note. This note
does not change frozen0017 or the exact replay required by0018.

## Author-recommended final loss

The [official RePoseD README at7955c076](https://github.com/kocurvik/mdrp/blob/7955c0761a2fa7de30d18aa03dd888321bf2a0dc/README.md)
recommends `bundle_dict={'loss_type':'TRUNCATED_CAUCHY'}` for its final refinement.
It also reports an improved joint Sampson/reprojection local optimizer after the
ICCV camera-ready version. These are existing author controls, not new ideas.
The published0017 diagnostic used an empty bundle dictionary and thus native
CAUCHY defaults. Its four known-method arms are an information/diagnostic bank;
they do not exhaust the strongest official settings. Authors' separate reported
benchmark numbers cannot be compared directly to our80pair AUC table.

The [pinned current loss implementation](https://github.com/PoseLib/PoseLib/blob/6a636b1d84273d5e3ba97676463476569611572a/PoseLib/robust/robust_loss.h#L138)
caps TruncatedCauchy at threshold-squared times log(2) and returns zero derivative
weight beyond the threshold. CAUCHY and TRUNCATED_CAUCHY are therefore distinct
controls. An improvement only over the default cannot be presented as beating
the author-recommended estimator. A later prospectively frozen control must use
the current parser's option names and verify the effective settings; README
examples from an older API are not sufficient evidence of option acceptance.

## Contemporary implementation and novelty boundary

The RePoseD coauthor's [fastpose repository atc3bed2c0](https://github.com/kocurvik/fastpose/tree/c3bed2c03e108c65d40a60d40acbcbbe48e9628c)
(commit dated2026-09-02) implements monodepth-assisted estimation too. Its
[calibrated cost](https://github.com/kocurvik/fastpose/blob/c3bed2c03e108c65d40a60d40acbcbbe48e9628c/src/fastpose/refiners/monodepth.py#L483)
returns the sum of robust Sampson and valid directional reprojection terms;
nonpositive transformed-depth branches remain omitted. Its
[LM engine](https://github.com/kocurvik/fastpose/blob/c3bed2c03e108c65d40a60d40acbcbbe48e9628c/src/fastpose/refiners/lm.py)
uses a different damping schedule, with Numba fastmath on the caller. Thus it is
not a numerically interchangeable control for PoseLib's residual-count mean,
and changing a mean to a sum is already represented by public source code.
No fastpose performance claim has been independently checked here, and this
software repository was not asserted to be a peer-reviewed publication.

Together with the previously verified MADPose fixed-support/positivity design,
these controls rule out claiming ordinary loss replacement, fixed-denominator
cost, or positivity constraints alone as an established novel contribution.
The next method must specify a distinct mechanism and test it against these
known alternatives at matched complete-pipeline cost on untouched sources.
