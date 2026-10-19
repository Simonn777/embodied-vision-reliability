# 0015: image-information source and metadata review

Reviewed 2026-10-19. **Transfer feasibility is established; no experiment,
new method, accuracy gain, or publication admission is established.** This note
records source/metadata inspection only. No model entries, image payloads,
acquired HDF5 numeric arrays, or new inference were read/run for this review.

## Official implementation and separable image input

Sun et al., *Learning Affine Correspondences by Integrating Geometric
Constraints*, CVPR 2025, pp.27038–27048:
[official paper](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_Learning_Affine_Correspondences_by_Integrating_Geometric_Constraints_CVPR_2025_paper.html),
[full-text v2](https://arxiv.org/html/2504.04834v2).
Relevant method and experiment sections were read, not only the abstract.
The official [DenseAffine repository](https://github.com/stilcrad/DenseAffine)
was inspected at commit **ddc3a112ad6b4b7e1618a0221d44750eb94eda63**,
dated 2026-07-27T03:15:52Z.

The paper fine-tunes the DKM refiner first and then trains patch orientation,
scale and residual affine shape. The released demo also runs its DKM checkpoint
to produce new matches. However, the patch modules can be separated at source
level: EstiNet.forward(patch) takes patches, while
LAFAffNetShapeEstimator.forward(laf, img) takes local affine frames and an
image. Existing coordinates could supply patch centers without rerunning DKM.
That would be an adapted known baseline, not a faithful full-pipeline reproduction.
[Estimator source](https://github.com/stilcrad/DenseAffine/blob/ddc3a112ad6b4b7e1618a0221d44750eb94eda63/Local_affine_estimation/S3Esti/abso_esti_net.py),
[shape source](https://github.com/stilcrad/DenseAffine/blob/ddc3a112ad6b4b7e1618a0221d44750eb94eda63/Network/essential_matrix_affine_without_kornia.py).

Compatibility issues must be resolved scientifically before interpreting outputs:

- The demo extracts 32×32 patches at rounded centers from cv2.imread(... )[:,:,0]
  and repeats them to nine channels. It is not a documented clean RGB32 API.
- The demo decodes 300 scale / 360 angle bins, while the imported estimator source
  defines 600 / 720. The checkpoint convention has not been inspected; this is an
  unresolved interface mismatch, not a verified checkpoint defect.
- Hard-coded CUDA calls need an explicit CPU adaptation. The requirements include
  CUDA/CuPy and conflicting Pillow pins; they are not a minimal Mac CPU recipe.
- The demo also applies HardNet descriptor filtering. Keeping this stage changes
  the correspondence set and would confound a fixed-correspondence comparison.

[Pinned demo](https://github.com/stilcrad/DenseAffine/blob/ddc3a112ad6b4b7e1618a0221d44750eb94eda63/demo/affine_feature_estimator.py),
[requirements](https://github.com/stilcrad/DenseAffine/blob/ddc3a112ad6b4b7e1618a0221d44750eb94eda63/requirements.txt).
CPU latency and peak memory remain unmeasured. Image-derived information is new
relative to stored coordinates; its errors are not therefore statistically independent.

## Asset metadata and transferred bytes

The official README links this
[weights archive](https://drive.google.com/file/d/1WX6Ug0ERmT9JC5OwMVWav20RBhkKOsvr/view).
HEAD and ZIP metadata give weights.zip **684,204,786 bytes**.
The official ETH CVG
[MegaDepth1500 archive](https://cvg-data.inf.ethz.ch/megadepth/megadepth1500.zip)
is **1,425,813,679 bytes**. The URL is also used by the
[official Glue Factory evaluator](https://github.com/cvg/glue-factory/blob/main/gluefactory/eval/megadepth1500.py).
Both servers returned byte-range support; range responses were checked for
HTTP 206 and exact Content-Range before reading their bodies.

| Relevant archive content | Compressed bytes | Uncompressed bytes |
| --- | ---: | ---: |
| Outdoor ori_scal.pth | 70,245,297 | 75,957,142 |
| Outdoor Aff_res_shape.pth | 329,315 | 361,642 |
| Optional HardNet++.pth | 4,979,287 | 5,345,606 |
| Outdoor DKM checkpoint, unnecessary for a patch-only adaptation | 269,065,661 | 289,797,889 |
| 23 images named by the existing 16-pair development manifest | 22,505,591 | 22,559,656 |

All 23 unique filenames were present in the ZIP central directory; the largest
uncompressed selected JPEG is 1,392,929 bytes. Two affine weights plus those images
sum to **93,080,203 compressed bytes**, before local ZIP headers and transport
overhead. The two complete archives sum to 2,110,018,465 bytes. Selective extraction
is technically possible and avoids the large original MegaDepth archives.
Filename presence does not yet establish resolution, undistortion, or pixel
coordinate identity with the cached matches.

Actual binary metadata bodies transferred during this admission review:
weights EOCD + central directory **972 bytes**; image EOCD + central directory
**210,912 bytes per pass**, read twice for inventory and manifest-name matching.
Total archive metadata bodies: **422,796 bytes**. No compressed entries were
read. HTTP headers and repository/paper HTML, JSON and source text were also read;
their byte counts were not separately metered here. These metadata counts are
not a claim about total session network traffic. No payload SHA256 has been
computed because payloads were not downloaded.

## Applicable rights and scope

The pinned repository's short
[LICENSE notice](https://github.com/stilcrad/DenseAffine/blob/ddc3a112ad6b4b7e1618a0221d44750eb94eda63/LICENSE)
states that models trained using noncommercial synthetic-source datasets cannot
be supplied under MIT, while it offers its MegaDepth models under MIT. It does
not establish a uniform MIT grant for every checkpoint or the whole code tree.
The [official MegaDepth page](https://www.cs.cornell.edu/projects/megadepth/)
describes CC BY 4.0 for its dataset outputs and notes the original photographs'
own licenses. We do not relabel the source photographs as MIT. This record is
not an extra approval requirement or a demand to obtain fresh permission for
every image: any later use must follow the actual applicable research-use and
redistribution terms. Public redistribution of downloaded image/model payloads
is not proposed.

## Novelty boundary

DenseAffine already supplies image-patch affine geometry, geometric training
constraints and affine relative-pose estimation. Barath et al., *Making Affine
Correspondences Work in Camera Geometry Computation*, ECCV 2020, already supplies
symmetric intensity refinement, affine-parameter covariance and propagation
inside robust estimation:
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560698.pdf),
[supplement](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560698-supp.pdf).
Generic patch refinement, two-affine solvers, per-match covariance weighting or
perturbation-based confidence are not admitted as a new contribution.

Off-diagonal error dependence between distinct overlapping correspondence
patches is a possible diagnostic target. It must be identified from shared
image evidence rather than fitted to pose residuals, which also contain the
unknown pose error. Synthetic image perturbations measure an estimator's
response to the injected noise; they do not by themselves identify the native
matcher error distribution or a signed bias correction. A separate,
prospectively controlled mechanism test would be needed before any such claim.
