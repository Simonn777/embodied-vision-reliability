# 0009 real-geometry data feasibility: metadata-only audit

Later update (same date): this is the **pre-directory** feasibility audit. The subsequent [cloud qualification archive](../results/0009-admission/qualification/README.md) contains runtime checks and ZIP directory metadata, but failed strict HTTP request/body accounting. The archive preserves the original receipts and unknown 302 body. No image or homography payload has been admitted. The scope statements below describe this earlier audit only.

2026-10-18. **No fully qualified ≤100 MB data route was established.** There is a promising, author-hosted HPatches range-read route, but its exact small-subset size remains unverified. Oxford supplies genuinely small official packages but the inspected pages did not specify a data license. HEB is scientifically better matched to robust estimation, but every listed scene file exceeds the limit and its data license was not established.

Scope: six search queries total; primary papers, author repositories, official dataset metadata/README, public read-only WebDAV listings, PyPI metadata and HTTP HEAD only. The 2,386-byte Oxford DataREADME was read as documentation. No dataset payload, ZIP central directory, image, feature, homography matrix, model or wheel was downloaded. No installation or cloud command was performed. No old experimental data was read. Sources below were checked live; these are metadata findings, not measurements of CPU runtime.

## 1. HPatches: best conditional route, not yet a verified bounded subset

The CVPR 2017 authors' [official repository](https://github.com/hpatches/hpatches-dataset) now links directly to **vbalnt/hpatches** on Hugging Face, rather than requiring an unofficial mirror. The author-hosted [dataset card](https://huggingface.co/datasets/vbalnt/hpatches/blob/main/README.md) explicitly declares `license: mit`; this is stronger dataset-specific evidence than merely treating the GitHub software LICENSE as a blanket image license. Preserve attribution and the authors' [per-sequence source references](https://github.com/hpatches/hpatches-dataset/blob/master/references.txt), which identify reused Oxford, AMOS, DTU and other sequences.

The [author API](https://huggingface.co/api/datasets/vbalnt/hpatches?blobs=true) reported:

| Item | Observed metadata |
|---|---|
| Repository revision | `594f211e80b75081ad84a396c93960c8653235f8` |
| Gated | `false` |
| Full-sequence ZIP | `hpatches-sequences-release.zip`, **1,279,545,572 bytes** |
| ZIP LFS SHA256 | `99cf7e1ca167896eb4ca2fe3d903beff6566243a3c0c8e07a13a2596648ad670` |
| Patch-only ZIP | 4,464,752,688 bytes; not the desired full-image route |
| HEAD on pinned full-sequence URL | HTTP 200; `Content-Length: 1279545572`; `Accept-Ranges: bytes` |

The API lists two ZIP archives, not individual scenes. Therefore **neither ordinary full download meets 100 MB**. A future explicitly authorized metadata stage could inspect the ZIP end/central directory using bounded ranges, select fixed entire sequence members, and verify that their compressed byte ranges plus overhead fit the cap. That operation was not performed here. An `Accept-Ranges` header does not prove a particular subset fits or that subsequent servers will honor every range. Reject a full-body fallback; require Content-Range/length checks, bounded extraction and member integrity checks. A subset cannot claim verification against the whole-archive SHA without downloading the whole archive; retain the immutable author revision, archive identity and independently recorded selected-member hashes/CRC evidence instead.

The official [split file](https://github.com/hpatches/hpatches-benchmark/blob/master/python/utils/splits.json) has 116 sequence IDs: splits a/b/c each use 76 train and 40 test; viewpoint has 59 and illumination 57. A sequence has one reference plus five targets and corresponding H. These are real captures with estimated GT, rather than our own artificial warp pairs. Keep every pair from a sequence together. A sequence ID is not a proof that all physical scenes/camera views are distinct: related day/night identifiers and reused source collections require grouping before a strict scene-heldout claim. Avoid moving any viewed/tuned sequence back into confirmation. Common public-benchmark use also prevents a claim of universal historical test cleanliness; handcrafted SIFT avoids learned-descriptor training overlap but not researcher-selection bias.

Scientific limitation: HEB's CVPR 2023 paper explicitly describes HPatches as relatively easy for robust estimators and notes the absence of one standard homography-evaluation correspondence/protocol choice. A bounded HPatches study could be a first real-data development check, not sufficient evidence of broad robust-estimation progress. [HEB paper, related datasets and §3–4](https://arxiv.org/html/2302.09997v1)

## 2. HEB: actual scene metadata does not fit the bound

I inspected the CVPR 2023 [author repository](https://github.com/danini/homography-benchmark), its current tree `7e1872128ea6d3bdb2f4792e941cfa70625200d7`, config, loader, and the linked [ETH public dataset share](https://polybox.ethz.ch/index.php/s/R5sPelZ8688It92). Read-only `PROPFIND Depth:1` listings of the share and its two folders returned these exact sizes; no HDF5 body was read:

| Folder / scene | Bytes |
|---|---:|
| training_and_validation / Alamo | 5,964,540,440 |
| training_and_validation / NYC_Library | 2,482,289,664 |
| test / Piazza_del_Popolo | **324,818,928** |
| test / Ellis_Island | 1,104,956,216 |
| test / Yorkminster | 1,523,300,712 |
| test / Madrid_Metropolis | 1,811,190,296 |
| test / Union_Square | 1,859,366,160 |
| test / Tower_of_London | 2,238,399,224 |
| test / Roman_Forum | 3,687,006,336 |
| test / Vienna_Cathedral | 12,081,338,720 |

No smaller shards or license document appeared in those three listings, and the 24-file GitHub tree contained no LICENSE. This is an **unresolved permission question**, not a claim that research use is prohibited. The official [1DSfM image page](https://www.cs.cornell.edu/projects/1dsfm/) supplies separate multi-hundred-MB/GB image collections, so it does not fix the bound. Remote HDF5 subset access could theoretically avoid full acquisition, but offsets, compression, range support and a lawful bounded subset have not been verified; do not label that an available route.

HEB's documented inputs include mutually nearest RootSIFT matches, coordinates, orientations, scales and ratio scores, with other-plane inliers removed. It directly addresses post-mutual-filtering outliers. However, its [current YAML](https://github.com/danini/homography-benchmark/blob/main/dataset_configuration.yaml) repeats Alamo/NYC_Library under `TEST_SCENES`, whereas the live test folder does not contain them and the paper describes scene-disjoint splits. This config/listing mismatch is **not evidence that the published data itself is contaminated**. A future protocol must pin actual scene membership rather than assume the YAML alone proves separation. Multiple planes/pairs from one landmark remain one scene cluster.

## 3. Oxford affine sequences: small official packages, license unresolved

The [official Oxford page](https://www.robots.ox.ac.uk/~vgg/research/affine/) says its per-sequence packages contain PPM images and H. Its linked [DataREADME](https://thor.robots.ox.ac.uk/affine/DataREADME) describes six images per sequence, real acquisition changes, and manually initialized then refined GT. The JPEG condition is an artificial photometric degradation, so it should not be presented as an independent captured-view geometry case.

HEAD requests followed official redirects to `thor.robots.ox.ac.uk/affine/`; all eight returned HTTP 200 and byte-range support:

| Archive | Bytes |
|---|---:|
| bikes.tar.gz | 7,366,360 |
| trees.tar.gz | 10,935,162 |
| graf.tar.gz | 8,414,417 |
| wall.tar.gz | 9,991,441 |
| bark.tar.gz | 5,941,281 |
| boat.tar.gz | 2,160,047 |
| leuven.tar.gz | 6,739,237 |
| ubc.tar.gz | 3,805,484 |
| **Total** | **55,353,429** |

This verifies a bounded transfer opportunity. It does **not** establish an explicit image/data license: neither inspected page nor DataREADME supplied one. Do not transfer the HPatches MIT declaration to separately downloaded Oxford archives without confirming that scope. The named sequences also overlap HPatches sources; they cannot become an independent confirmation set after tuning on the corresponding HPatches scenes. There are only eight conditions/sequences, with just two dedicated viewpoint-change sequences, limiting scene-disjoint inference even if permission is clarified.

## 4. Strong CPU baseline: one small new wheel is plausible, installation separate

Existing NumPy/SciPy can implement normalized DLT using [SVD](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html), mutual matching, fixed-pool count/MSAC scoring and the exact-null arithmetic. [cKDTree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html) provides an optional radius-search primitive. These primitives do not themselves supply an independently verified MAGSAC++ pipeline. A handwritten RANSAC or simple cell normalizer is a useful ablation, insufficient as the only strong baseline.

The official [OpenCV Python packaging repository](https://github.com/opencv/opencv-python) documents CPU-only wheels, SIFT inclusion, MIT packaging/Apache-2 OpenCV and bundled third-party licenses. [Version-specific PyPI metadata](https://pypi.org/pypi/opencv-python-headless/4.12.0.88/json) exposes this concrete option:

    opencv_python_headless-4.12.0.88-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
    size: 54,023,419 bytes
    sha256: 236c8df54a90f4d02076e6f9c1cc763d794542e886c576a6fee46ec8ff75a7a9
    Python >=3.9 dependency: numpy >=2, <2.3.0

This is conditional on the cloud's actual Linux x86_64/ABI/glibc identity and installed NumPy version. No upgrade or installation is authorized by this audit. If a later combined acquisition cap were 100 MB including software, the wheel leaves only 45,976,581 bytes before other overhead; all eight Oxford archives plus this wheel would exceed it. The dataset-only and total-acquisition caps must be stated separately.

OpenCV's [USAC documentation](https://docs.opencv.org/4.5.1/d1/df1/md__build_master-contrib_docs-lin64_opencv_doc_tutorials_calib3d_usac.html) and [4.12.0 source](https://github.com/opencv/opencv/blob/4.12.0/modules/calib3d/src/usac/ransac_solvers.cpp) support homography MAGSAC++, MSAC/LO and PROSAC controls. The source binds random state and parallel settings. Freeze actual API parameters, iteration/time budget, residual convention, input sorting and seed; prefer nonparallel solver execution for repeatability under the overall four-thread limit. Keep complete-pipeline MAGSAC++ distinct from a fixed-hypothesis-pool scoring ablation. CPU runtime and functional wheel compatibility are not measured here.

**Decision:** HPatches author-hosted range extraction is the strongest *conditional* next metadata route. It needs a separately bounded archive-directory/member-size check and a scene manifest before any dataset acquisition. HEB presently fails the simple ≤100 MB route; Oxford satisfies size but not the requested explicit-license verification. No qualified dataset subset or scientific run should be claimed from this audit.
