# 0025c conditional RS pose: critical feasibility screen

Private, prospective, source-only. No scientific payload, array, pose/GT value, model output or new checkpoint was read; no estimator was called. The reported completion of 0025b operations is not treated here as a pose-accuracy result. This note does not allocate a dataset or admit an experiment.

**Judgment:** a small conditional-solver qualification is defensible; a two-source accuracy campaign is not yet ready. The plausible benefit is a lower-sample, depth-assisted initialization under measured rotation, not a new fact that translation and readout velocities become linear. The main risks are near-identity conditioning, gyro/time error being absorbed by depth shifts and velocities, and incomplete source-specific timing/calibration contracts.

## The comparator that prevents an artificial win

The 2017 [*Inertial-aided Rolling Shutter Relative Pose Estimation*](https://arxiv.org/html/1712.00184v1), full §§3 and 4.1–4.4 read, distinguishes three models. Its **linear** model sets angular velocity to zero; its **uniform** model keeps both angular and linear readout motion. Therefore “2017 linear RS” is not an adequate label for the nonzero-gyro control. The uniform model has an eleven-point unknown-rotation formulation and a nine-component conditional translation/velocity nullspace, §4.2 Eqs.16–17. Gravity supplies roll/pitch, not yaw. No author implementation was established by the bounded title/repository search; a new implementation must be labeled paper-based, not author-code reproduction. The 2019 [gyro five-point paper](https://arxiv.org/html/1904.06770v1), full §3.1 read, explicitly neglects readout translation and is a useful restricted ablation, not the sole strong RS baseline.

The following is a direct consequence of the frozen 0025 model, **not a claim about an existing implemented solver**. Define

\[
n_i=\bar r_{2i}\times(R\bar r_{1i}),\qquad
C_i=[n_i^T,\;-n_i^TRB_{1i},\;n_i^TB_{2i}].
\]

Eliminating both point depths gives

\[
C_i[t^T,v_1^T,v_2^T]^T=0.
\]

This uses exactly the same known rotations and exact angular integration as the proposed affine-depth system. The homogeneous nine-vector has a length gauge, so eight equations are the generic **counting lower bound**, not a demonstrated universally valid eight-point minimal solver. Its rank, task-relevant nullspace and finite-motion counterexamples require qualification. It is a stronger information-matched comparator than handing measured middle rotation only to the four-point candidate.

Minimum scientifically useful comparisons would therefore include:

1. The depth-free conditional RS nullspace above, with the same ray, gyro and exposure-time contract.
2. A same-rotation **angular-only RS affine-depth** control: retain the known row rotations and unnormalized dewarped rays, but set both readout translation velocities to zero. Its six unknowns are `(t,b1,a2,b2)`. This is not ordinary GS affine alignment: GS is the special case of zero row angular motion, and simply normalizing dewarped rays changes the axial-depth coefficients. Also report the current official free-rotation GS RePoseD scale/affine variants with `TRUNCATED_CAUCHY` as later practical controls. The existing local `0019-author-loss` source calls `estimate_monodepth_relative_pose(..., initial_pose=None)` with the official nested bundle option; it does **not** expose a fixed-R solver. Initializing a free-R optimizer is not equivalent to constraining R.
3. A common RS/depth scorer and refinement stage for both initializers, alongside complete native-method results. This separates minimal-sample benefit from different residuals, truncation, iteration budgets or final optimization.

The [2026 affine-correspondence RS method](https://arxiv.org/html/2606.26863v1) and its [author implementation](https://github.com/danini/rolling_shutter_made_practical) remain relevant strong current prior. Ordinary LightGlue points do not supply affine correspondences for free; acquiring those derivatives and their cost must be explicit. A first-order author RS model versus the exact exponential candidate also needs an exact-model control before assigning a gain to learned depth.

## What could make four points useful—and what could kill it

The four-point matrix has an exact full-rank witness, but the existing generic fixture has scaled condition number about 3815. At identity middle rotation with zero angular rates, four points have a rank ceiling of 11; the published counterexample can change translation direction. Short-interval robotic pairs often approach this limit. Narrow row coverage and nearly affine correspondences between source/target row times add risks. A witness is not a noise guarantee.

The potential robust-estimation benefit is smaller required support, provided four matches also carry sufficiently accurate, mutually compatible depth. The familiar inlier-probability powers for four versus eight samples illustrate potential only: learned-depth failures and spatially nearby matches are correlated. A common refinement may erase any initializer advantage. Conversely, a fixed-R model can move translation/velocity/shift to compensate for gyro error, producing a numerically excellent but wrong solution.

The root has narrowed the immediate test to **Stage A: all-inlier initializer-noise qualification**, using fixed physical points and all 70 four-subsets without best-subset selection, a renderer, RANSAC or refinement. See `minimal-qualification-review.md`. Only if that screen passes would a separately admitted contaminated-pool test compare conditional initializers with common scoring/refinement, fixed depth/gyro perturbations and all rejected/degenerate samples charged. No search for favorable fixture/noise settings after results. If benefit exists only with exact GT rotation or weaker controls, stop before a new dataset campaign. Neither stage has run for this review.

## Independent rotation is available in principle, not automatically as a valid R field

**TUM-RS/TUM-RSVI** is one grayscale source, not two RGB datasets. The [official page](https://cvg.cit.tum.de/data/datasets/rolling-shutter-dataset) and [directory README](https://cdn3.vision.in.tum.de/rolling/README.txt), previously read and rechecked from saved text, provide synchronized IMU, camera–IMU calibration, approximate manufacturer row delay 29.4737 µs, and first-row exposure-midpoint stamps. Calibrated IMU data correct axes, scale and constant bias. Optical mocap is a distinct reference instrument, but its time alignment uses differentiated orientation versus gyro, so calibration errors are not statistically independent. The recommended equidistant distortion model requires retaining original sensor-row coordinates when undistorting or resizing features. Data-license terms remain unverified; no conclusion about legality follows from that absence.

**GrandTour** has distinct ADIS16475-2 and Alphasense BMI085 gyro streams, each nominally 200 Hz, separate from the CPT7/HG4930 reference route. The [author paper](https://arxiv.org/html/2602.18164v1), §§3–4, and cached official `grandtour_release.yaml`, `message_parsing.py` and `update_imu_intrinsics.py` establish those input topics. The inspected calibration processor fills noise covariances but leaves ADIS/BMI085 measurement values unchanged; a mission-specific bias correction has not been established. The parser exports an `orien` field because ROS IMU messages contain it; this does not certify a measured usable orientation. Use integrated independent gyro, not `cpt7_ie_tc_odometry` or an unqualified IMU quaternion.

GrandTour HDR semantics are middle-row exposure midpoint. The correction code and intended processing order are documented, but the exact public-mission execution lineage and the descriptive 10 Hz versus 30 Hz discrepancy remain unresolved. Do not reapply the approximately 21.69 ms correction or derive row time from released frame cadence. ETH-1/2/3 remain excluded by the author's arrival-timestamp warning. Smaller missions are only size examples until their motion/reference support is qualified. ADIS/BMI085 and reference sensors share a rig and timing infrastructure; “distinct instrument” does not mean perfect independent truth.

Endpoint angular rates alone do not supply middle-frame R. Integrate calibrated gyro across the **whole pair interval**, in the correct camera frame. For world-to-camera poses, the physical body-gyro sign and the left-twist sign need an explicit derivation and manufactured test; copying an SO(3) exponential without that convention check is unsafe. Gravity alone leaves yaw unknown. Input rotation error is common to every fixed-R method, so report conditional translation performance and rotation error separately; do not describe it as recovering full pose from four points.

## Smallest calibration packet before final-data admission

One prospectively chosen development packet per source is enough to test the engineering contract, not scientific efficacy: camera intrinsics/distortion and dimensions; readout direction/line interval/stamp anchor; camera–IMU extrinsics with direction and units; frame timestamps; and input gyro samples bracketing the complete pair interval and both readouts. It must specify bias handling without using pose GT. Preserve original sensor y for every resized/undistorted feature. Check units, timestamp ordering and integration signs with manufactured rotations before consuming observations. If a stationary prefix is required for bias, declare its availability and use prospectively rather than select it from GT.

The available source evidence is enough to plan this packet, not to assume its values are correct. A rough analytic sensitivity illustrates the concern: at focal length 500 pixels, 0.1° orientation error is about 0.87 pixel; at angular speed 3 rad/s, 1 ms timing error is about 1.5 pixels. These are illustrative calculations, not measured dataset values or tuned tolerances. A future admission must fix acceptable calibration evidence before viewing final-pair outcomes.

Full cost must include two learned-depth forwards, features/matching, gyro integration, hypothesis validation and refinement. Reusing a frozen cache is useful for diagnosis but is not complete-pipeline speed evidence. No need for new weights is established; no extra model acquisition is proposed here.

## Read-depth and transfer accounting

- New primary full-text reads: the 2017 and 2019 arXiv HTML articles above; bounded exact-title/repository search did not verify author code. Secondary search hits were not used as evidence.
- Existing public 0025 derivation/admission and saved primary TUM/GrandTour text were read. GrandTour source object `7c7b7884e06a2082ac11f7b8a2ef3b7d25de5062` is a **Git tree**, not a verified commit. HF metadata revision is `daf91c5e2b6db564822a35cc58222d73f8e75bab`. No new claims of per-mission execution provenance are made.
- Newly metered direct HTTP response-body bytes: **0**. Web-tool retrieval bytes are provider-managed and unavailable, not counted as zero. No sensor/archive/image/depth/pose payload was requested or stored for this screen.

**Admission boundary:** useful conditional mathematical qualification is possible; no fair measured-rotation pose experiment or two-source final evaluation is admitted by this memo. The current concrete prerequisites are the equal-input exact RS control, noisy/near-degenerate qualification, and one complete non-GT timing/extrinsic/bias contract per development source.
