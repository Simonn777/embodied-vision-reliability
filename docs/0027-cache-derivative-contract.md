# 0027 derivative-input feasibility — metadata and producer source only

2026-10-19. No NPZ/HDF5 values, images, labels, models, fitted derivatives or
scientific computations were accessed. This is a resource/coordinate contract,
not method admission. No new network transfer occurred.

**Conclusion:** existing 0023 caches contain a traceable complete predicted
depth field from which a declared interpolation derivative can be computed.
They do **not** contain affine image correspondences. Existing 0017 caches
contain neither dense depth fields nor affine/dense correspondence derivatives;
neighbor fitting would be a newly defined estimator, not reconstruction of the
original model's derivative.

## 0023: available full fields; missing affine correspondence

Private directory: `work/0023-preparation/frontend-01/`.
Receipt SHA256:
`4bc0e5b7b47a5c1e0266e645115f3b71f92c099042ec83106ccd7b12148f8259`.
It records 51 completed image caches and 120 completed edge caches. All 51
depth arrays have metadata shape `[512,384]`; all 51 have a `normal` array.
No cache values were opened to make these statements.

Each `image-NNN.npz` is bound by compressed-file SHA256/size and per-array
dtype/shape/raw-C-order SHA256. Relevant fields are:

| Fields | Recorded producer semantics |
|---|---|
| `depth`, `mask` | full float32 axial-depth prediction and bool validity map |
| `points`, `normal`, `intrinsics` | full MoGe outputs, including model-predicted intrinsics |
| `K_original`, `K_resized`, original/resized image sizes | independently authored PINHOLE calibration and declared image resize |
| `xy`, `keypoints`, `descriptors`, `keypoint_scores`, `image_size` | sparse SuperPoint features at resized array pixel centers |
| `depth_sample_raw`, `depth_sample_normalized`, `depth_sample_valid`, `depth_positive_valid_median` | four-neighbor bilinear samples and once-per-image positive-valid median |

Each `edge-NNN.npz` retains raw LightGlue outputs and canonical
`match_ids` (N×2), `match_scores` (N), `xy0/xy1`, raw/normalized endpoint depth
samples and validity flags. These are sparse mutual matches, not a dense warp.
The receipt's array-name inventory has no affine/Jacobian/warp output.
SuperPoint descriptors and discrete matched positions do not determine a unique
continuous cross-image Jacobian.

Producer evidence is `experiments/0023-mapfree/frontend.py` at source SHA256
`5076452d2e69645745eb68675a40bb47473ec4a4a28db87cf6f9652229614bf7`,
recorded freeze `acf502b831ad9e3485ab56549217c8a603d56bf7`.
`Backend.depth` retains every tensor from `MoGeModel.infer`; image processing
stores those outputs before adding SP features and samples. `edge_arrays`
indexes the two immutable image caches. Thus the same image genuinely shares
one depth realization across its edges. No assumption based on a filename alone
is needed here.

The separately bound frontend qualification freeze is
`fa454585e67a8f0395ad5f67a03d5a28cc33a66a`; the MoGe2-S-normal checkpoint SHA is
`79a16621928c2bf0ed04659218c55c01075e950507f40bb3332fb4c873d3e1dc`.
The receipt records CPU float32, tokens1200, `apply_mask=False`,
`force_projection=True`, `use_fp16=False`, SP max2048/no extra resize, and
LightGlue flash/mp/adaptive stopping disabled. This contract supports reusing
the *saved prediction*, not a claim that learned derivatives are physically
accurate or that these scenes were absent from model training.

## Derivative convention that must be declared before use

An exact derivative of the same bilinear interpolant is possible from four
valid neighboring depth pixels. Inside a cell, with offsets `(alpha,beta)`,

`d_x=(1-beta)(d10-d00)+beta(d11-d01)` and
`d_y=(1-alpha)(d01-d00)+alpha(d11-d10)`.

It is a derivative of the cached interpolant, not of the underlying scene.
At integer pixel boundaries this interpolant is generally not differentiable;
many SP features can lie at such centers, so a future protocol must state a
one-sided cell convention or a separately defined derivative estimator. Masks,
image borders and depth discontinuities require explicit unavailable states.
Changing to a smoothed finite difference or local polynomial fit introduces a
new measurement estimator with its own support and bias. No usable coverage can
be inferred from the metadata alone.

Coordinates are zero-based resized pixel centers. Authored K is resized using
`fx'=sx*fx`, `cx'=(cx+.5)*sx-.5` (and y analog). If an affine image map is
estimated in resized pixels, its calibrated-coordinate Jacobian is
`A_cal=diag(1/fx2,1/fy2) A_pixel diag(fx1,fy1)`. A depth row-gradient transforms
as `grad_cal=grad_pixel diag(fx1,fy1)`. Whole-image median normalization divides
both depth and its derivative by that same frozen median. Do not use MoGe's
predicted `intrinsics` in one part and authored K in another unnoticed, or use
the separate learned `normal` head as if it were the analytic derivative of
`depth`.

An AC would still require an explicitly admitted producer, e.g. image-based
affine patch registration or local fitting to neighboring matches. Those are
known techniques, and their input/support cost must be counted. A “one-AC”
minimal solver does not imply one raw point measurement if its AC was estimated
from multiple neighboring matches. Estimating A and depth gradients jointly
from the same fitted geometry can also manufacture dependent constraints; it
would not establish an independent affine-depth measurement. Separate producer
paths reduce circularity, but shared images still prevent an automatic claim of
statistical independence.

## 0017: sparse opaque author-cache limit

`configs/0017-development-manifest.json` SHA256 is
`a034601021b4baa08a6cc04e91e951ce27dc061d0c0556cf7f1ac141d6a66e34`.
Its complete metadata covers 500 pair groups in five scene files, with 80
previously allocated pairs. Every group has the same field inventory:
`x1/x2` (2048×2), `depth1/depth2` (2048), `K1/K2`, camera model/params/width/
height, and the separately restricted R/t labels. No full depth/image/normal/
flow/affine-Jacobian field is present. This observation uses the saved manifest,
not HDF5 access.

The reviewed source consumer is PoseBench commit
`7b494256e89a13c47ab37b6917c2e0d3dae0c2e4`,
[`monodepth_relative_pose.py`](https://github.com/PoseLib/posebench/blob/7b494256e89a13c47ab37b6917c2e0d3dae0c2e4/src/posebench/monodepth_relative_pose.py),
with PoseLib axial-z consumer `6a636b1d84273d5e3ba97676463476569611572a`.
The earlier source audit in `docs/literature/0016-monodepth-consumer-contract.md`
did not establish a cache exporter, exact MoGe checkpoint, inference resize/crop,
or interpolation/filtering rules. This bounded gap remains; no new source search
was performed now. `roma_moge` labels do not supply the missing producer.

Local least-squares slopes are mathematically computable from sparse tuples if
the neighborhoods are sufficiently populated, but their density, rank,
occlusion contamination and bias are presently unknown. Calling those slopes
faithful RoMa/MoGe derivatives or independent differential observations would
overstate the available evidence. They cannot be obtained by merely unsealing
an existing saved affine field.

## Rights and scope

The original 0023 protocol keeps downloaded image/calibration/pose payloads and
numeric caches private; this note does not change that. The installed MoGe and
utils3d source LICENSE files state MIT; LightGlue's states Apache2.0. These code
licenses do not relicense the Map-free images or PoseBench's third-party-derived
caches. The existing PoseBench resource audit found no explicit separate hosted
cache redistribution license. The Map-free ZIP metadata lists a `LICENSE`
member, but its body was not among the inspected frontend members here, and this
review does not establish its data-license terms. That is an evidence limit,
not a legal prohibition or a new requirement to ask the user. Continue reporting
hashes/schema/provenance publicly rather than republishing these inputs under
the project's own license.

All three 0023 places and all five 0017 scenes are already development forever.
They can support a separately frozen mechanism/coverage diagnostic, not fresh
held-out efficacy. No new 0027 experiment or numeric access is admitted by this
resource screen.
