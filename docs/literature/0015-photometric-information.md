# 0015: classical photometric information and shared-pixel hypothesis

Source/metadata review, 2026-10-19. This is a prospective mechanism test, not
a new covariance estimator or an observed accuracy improvement. No image payload
or acquired numeric array was read for this source review.

## Use the image archive paired by the DMS authors

DMS commit **7d5b228de9b6fab460f43e466fb0ed8565ffe137** explicitly pairs its
DKM HDF5 file with
[PoseBench megadepth1500-images.zip](https://vision.maths.lth.se/viktor/posebench/relative/megadepth1500-images.zip):
[download script](https://github.com/jastermark/DMS/blob/7d5b228de9b6fab460f43e466fb0ed8565ffe137/data/download_megadepth.sh).
HEAD reports **623,957,998 bytes**. Its ZIP central directory has 809 entries.
The 23 images selected by the existing 16-pair manifest occupy **22,491,600
compressed / 22,559,656 uncompressed bytes**. All 23 have the same uncompressed
size and CRC32 as their ETH CVG counterparts. This is strong metadata evidence
of identical JPEG content, not a cryptographic payload-identity proof.

The [DMS loader](https://github.com/jastermark/DMS/blob/7d5b228de9b6fab460f43e466fb0ed8565ffe137/src/dms/dataloader.py)
reads images with OpenCV and returns stored coordinates unchanged unless
calibration is requested. The
[author demo](https://github.com/jastermark/DMS/blob/7d5b228de9b6fab460f43e466fb0ed8565ffe137/demo.ipynb)
loads this paired archive and overlays stored x1/x2 directly, without resizing.
This supports using the paired images in the stored raw-pixel frame.
The current PoseBench source, inspected at
**7b494256e89a13c47ab37b6917c2e0d3dae0c2e4**, distributes the same HDF5 URL but
does not expose its original matcher export script in its inspected source tree.

Do not import a different benchmark's scaling rule. Upstream DKM commit
**ef57565db52684e661052ba82eb361329c63af3d** has a
[MegaDepth benchmark](https://github.com/Parskatt/DKM/blob/ef57565db52684e661052ba82eb361329c63af3d/dkm/benchmarks/megadepth1500_benchmark.py)
that rescales coordinates and K to maximum dimension 1200 for its own evaluation.
This does not establish that the PoseBench cache uses that rule. DKM's normalized
grid-to-pixel formula is W(u+1)/2, so a later interpolator must state its pixel-center
convention rather than silently introduce an extra half-pixel shift. Image header
dimensions, authored K, coordinate bounds and the author's paired-image convention
can be checked before any pose outcome is inspected.

The prospective sampler convention is **array coordinate = cached coordinate
minus 0.5**, with refined coordinates converted back by adding 0.5 before
geometry; retain the original K without another transformation. This agrees
with upstream DKM's pixel-center construction and COLMAP's explicitly documented
upper-left pixel center at (0.5, 0.5):
[COLMAP database convention](https://github.com/colmap/colmap/blob/main/doc/database.rst),
[camera convention](https://github.com/colmap/colmap/blob/main/doc/cameras.rst).
The [MegaDepth source page](https://www.cs.cornell.edu/projects/megadepth/)
identifies COLMAP as the source of its SfM/MVS camera information. These are
independent consistency checks, not a recovered PoseBench export script.
Use no resize or pose-error-based offset selection. An input qualification
contradiction must be reported rather than resolved by trying offsets on labels.

Local metadata inventories are retained outside the repository:
work/0015-selected-images-metadata.json (ETH) and
work/0015-posebench-images-metadata.json (paired source), including local ZIP
offsets, compression, sizes and CRC32 for each member. This follow-up transferred
210,912 bytes to retain the ETH inventory and 87,869 bytes for the paired archive's
EOCD/central directory. Including the earlier 422,796 archive metadata bytes,
the cumulative archive metadata body total is **721,577 bytes**.
Source text and HTTP header traffic are not included in that total.

## Closest published fit and the actual adaptation required

Barath et al., *Making Affine Correspondences Work in Camera Geometry
Computation*, ECCV 2020, supplies the relevant established control:
[main §3.2/4.2](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560698.pdf),
[supplement §1](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560698-supp.pdf).
It estimates six geometric and two radiometric parameters symmetrically through
a latent middle patch, alternates geometry and latent intensities, uses bicubic
interpolation, and adapts the common square support. The supplement's precision
study uses window size M=7 times Lowe scale; the main paper initializes local
affinities from keypoint scales/orientations. It does not establish a universal
32-pixel window, iteration budget or identity-initialization default for our cache.

The inspected official repository at
**26f9d7f56703e11ac110fb33b0e779cd61fea2f9** exposes VLFeat affine extraction and
robust geometry. Its inspected
[feature helper](https://github.com/danini/affine-correspondences-for-camera-geometry/blob/26f9d7f56703e11ac110fb33b0e779cd61fea2f9/src/pyrobustac/include/feature_utils.h)
does not expose the paper's symmetric least-squares patch fitter. A small fixed
CPU fitter would therefore be an explicitly documented classical adaptation.

Large viewpoint change does not itself rule out a local affine approximation.
Depth boundaries, reflections, occlusions and insufficient texture do, and an
identity warp can be outside the convergence basin even when the center match
is good. A coordinate-neighborhood affine fit can initialize the local warp
without a new network; that initialization is known and contributes no new
image evidence. Freeze the support size, initialization, interpolation and
acceptance checks before inspecting pose outcomes. Record failed fits and use
the identical successful-fit set for every covariance arm. Do not replace
failed pairs with favorable ones.

## One testable mechanism, and what the perturbations identify

Hypothesis: after identical local photometric refinement, reuse of the same
physical pixels by overlapping patches creates appreciable cross-correspondence
error. Counting those shared directions repeatedly can reduce translation
precision; their covariance may retain useful, nonduplicate directions that
simple spatial thinning discards. This is a specific image-support hypothesis,
not an assertion that all dense matcher bias comes from sensor noise.

Let q_i(I) be a refined center displacement, with its source center fixed, and
B_i its derivative with respect to the two original image grids. For a specified
pixel-noise model V, the established propagation formula is
C_ij = B_i V B_j^T. Shared original pixel identities and interpolation weights,
not merely patch-center distances, determine the cross terms. Hold pose out of
this fit: estimating C from epipolar residuals can absorb the unknown pose error.

A direct check uses paired antithetic perturbations of the two whole images.
Every overlapping patch must see the same perturbation value at a shared pixel.
The response [q_i(I+epsilon)-q_i(I-epsilon)]/2 estimates local noise sensitivity.
Independent noise per extracted patch is a control with similar marginals but
without cross-patch sharing. Independent held-out perturbation draws test
covariance prediction, rather than evaluating on its fitting draws.
This identifies incremental response to the injected noise only. It does not
identify the native DKM error law, an unknown JPEG processing noise law, or a
signed correction to the original matches. Antithetic averaging does not remove
an equivariant common bias; warp consistency is already established
[prior](https://arxiv.org/abs/2104.03308).

Minimum decisive comparison after image/frame and fit qualification:

1. Original fixed correspondences and the strongest valid 0014 continuation.
2. The same photometrically refined centers with independent scalar uncertainty.
3. The same centers with full within-correspondence covariance but zero
   cross-correspondence blocks.
4. The same centers with image-derived cross-correspondence blocks, keeping the
   exact same diagonal blocks, initialization, robust loss and optimization budget.
5. Deterministic spatial thinning of those same fits with within-match covariance,
   reporting retained coverage and compute. Thinning cannot be omitted.

Plain full GLS is known; only a clear incremental gain of (4) over (3) and (5),
with validated shared-pixel response, would support this mechanism. Gains already
present in (2) are ordinary image-based refinement. If few patches can be fitted,
the predicted cross terms fail on held-out perturbations, correlation is negligible,
or thinning matches the accuracy, stop this mechanism rather than retune its
covariance on pose labels. All 16 pairs remain development; any encouraging
result still requires a frozen independent-scene test and strong DMS comparison.
The current evidence supports a bounded diagnostic, not a CVPR novelty claim.
