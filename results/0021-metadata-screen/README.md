# 0021 metadata feasibility screen

2026-10-19. **Topology: conditionally feasible. Hosted-cache provenance: unresolved. Scientific fitting: not admitted.**

Only the existing `configs/0017-development-manifest.json` and public author code/repository metadata were read. No HDF5 was opened, no numeric correspondence/depth/calibration/GT values were read, and no images, model weights, or dataset payloads were downloaded. The result describes candidate string tokens, not certified image identities or independent observations.

## Candidate graph

All 500 root keys match the strict syntax `([0-9]+_[0-9]+_o)_([0-9]+_[0-9]+_o)`. Nodes are scoped by scene; identical strings across scenes are not merged. Graphs are undirected, with no duplicate edges. The full graph has **430 scene-qualified token nodes, 500 edges, 16 connected components, and 5 triangles**. Its mean degree is 2.326. Nothing here establishes parallax, spatial support, pixel correspondences shared across edges, or statistical independence.

The table treats the original 16 selected pairs per scene as descriptive queries, and excludes **all 80 selected edges** from the reference pool. A supported query `(A,B)` has edges `(A,C)` and `(B,D)` with neither reference equal to the query. In all 53 supported cases, a choice with four distinct tokens is possible; reference edges can nevertheless share images with other queries and remain statistically dependent.

| Scene | Nodes | Edges | Degree min / max / mean | Components | Triangles | Queries with both endpoint references |
|---|---:|---:|---|---:|---:|---:|
| florence_cathedral | 89 | 100 | 1 / 7 / 2.247 | 4 | 0 | 13/16 |
| lincoln_memorial | 87 | 100 | 1 / 7 / 2.299 | 4 | 4 | 9/16 |
| london_bridge | 85 | 100 | 1 / 5 / 2.353 | 3 | 1 | 12/16 |
| milan_cathedral | 85 | 100 | 1 / 5 / 2.353 | 3 | 0 | 10/16 |
| sagrada_familia | 84 | 100 | 1 / 5 / 2.381 | 2 | 0 | 9/16 |
| Total | 430 | 500 | 1 / 7 / 2.326 | 16 | 5 | **53/80 (66.25%)** |

Other descriptive pools:

- Allow every edge except the particular query: 375/500 queries supported (75%); within the old 80-query cohort, 63/80 supported (78.75%).
- Use only the old 80 selected edges as potential references: 10/80 supported (12.5%).
- Exclude all 80 queries from references: only 1/80 queries has a common third token `C=D`; 19/80 have at least two reference neighbors at each endpoint, and 3/80 have at least three at each endpoint.

Thus the usual available construction is a four-token chain with separate reference-edge gauges, not a closed triangle. The already evaluated 80-pair bank alone does not support a broad calibration-transfer study. The other 420 pair keys are metadata-visible, but their scientific inputs have **not** been read or newly allocated here. All five scenes retain their existing development status; these graph counts do not create held-out scenes.

## What public author code actually establishes

**1. The token interpretation has source support, but the hosted exporter is missing.** In [RePoseD `eval.py`, lines 307–330](https://github.com/kocurvik/mdrp/blob/7955c0761a2fa7de30d18aa03dd888321bf2a0dc/eval.py#L307), the author removes `corr_`, splits a pair at `_o_`, and uses the resulting strings as `img_name_1/2` to look up per-image `K_<name>` and pairwise `corr_<name1>_<name2>`. Our strict parser agrees with that authored convention for all manifest keys. However, this is the original RePoseD consumer format, not the current pair-group PoseBench export. No inspected source establishes that the hosted keys were produced by losslessly removing `corr_`, or binds them to canonical image paths/content hashes. Treat them as source-corroborated candidate tokens, not certified identities.

**2. Author provenance ties the benchmark family to RePoseD, not every conversion detail.** [PoseBench PR #7](https://github.com/PoseLib/posebench/pull/7), created 2025-10-17 and merged 2025-12-08, says the monodepth benchmark contains 100 pairs from each of five PhotoTourism scenes, plus ScanNet, with RoMa/SP+LG and MoGe/UniDepth. Its changed files add the benchmark consumer and registration, not an exporter. The inspected PR comments are empty. The current [PoseBench loader](https://github.com/PoseLib/posebench/blob/7b494256e89a13c47ab37b6917c2e0d3dae0c2e4/src/posebench/monodepth_relative_pose.py) reads each pair's `x1/x2`, `depth1/depth2`, and cameras without interpreting the key. The [download script](https://github.com/PoseLib/posebench/blob/7b494256e89a13c47ab37b6917c2e0d3dae0c2e4/download_data.sh) identifies the exact hosted filenames. Neither describes their generation. The linked source ZIP was **not requested**.

**3. There is a per-image-depth preparation precedent, but it is not the hosted RoMa pipeline.** [RePoseD `prepare_pt_test.py`](https://github.com/kocurvik/mdrp/blob/7955c0761a2fa7de30d18aa03dd888321bf2a0dc/prepare_pt_test.py#L90) obtains each image's `.npy` depth map by an image-token glob, then samples it at matched coordinates. Lines 174–175 derive image names from the dataset's `images.txt`; lines 75–79 use those names for image files and feature storage. This supports the *intention* of reusable per-image maps in that script. However, it uses a LightGlue feature path and emits hyphen-separated pair keys plus separate depth datasets, unlike the hosted RoMa pair-group schema. Depth extraction is commented out, and `utils/monodepth.py` is empty at the inspected commit. No checkpoint, full-image depth hash, generation command, or RoMa-cache conversion was found in these sources. This cannot certify that all appearances of an image in the hosted cache sample the same prediction realization.

**4. Resize and pixel conventions remain unverified for the hosted cache.** The old preparation script passes an optional `resize` argument to feature extraction and floors feature coordinates when sampling `.npy` depth; its defaults are not evidence of the actual export invocation. The hosted consumer applies stored cameras to stored points, which supports their within-pair use, but does not establish a common cross-pair crop/resize coordinate system, half-pixel convention, depth-map sampling convention, or repeated-camera identity. The manifest stores shapes and dtypes only: its depths are per-correspondence vectors, not dense maps or provenance records. Neither identical vector length nor `depth_label=moge` proves shared predictions.

## Decision and exact missing evidence

The graph does **not** veto the mechanism: there are candidate reference edges across all five scenes. It also does not admit it. The missing links are (a) hosted pair-key → canonical image path/content, (b) repeated image → same depth prediction and preprocessing, and (c) stored pixel/camera/depth sampling → common per-image coordinate convention. An authored exporter with its invocation/version and input identities, or equivalent explicit author documentation, could close these gaps. A future numerical consistency audit might reveal contradictions but would not by itself prove that two opaque vectors came from the same full-image prediction.

No author contact, new data acquisition, pose estimation, depth calibration, or scientific allocation was performed. A separate identifiability review is still needed even if provenance is resolved, especially because independent reference edges have independent translation scales.

## Reproduction and receipts

- `experiments/0021-metadata-screen/screen_manifest.py`: standard-library-only parser/statistics; accepts only the authorized manifest filename, refuses overwrite, and binds manifest/script SHA-256.
- `graph-receipt.json`: exact graph counts, degree distributions, per-node degrees, and per-query availability. Seven artificial graph/parser assertions passed; no numerical scientific input was used.
- `source-receipt.json`: pinned author source URLs, commit/file hashes, byte counts for the final source-receipt fetch, and explicit unresolved provenance verdicts. Public text traffic is distinguished from dataset payloads; earlier browsing traffic is not presented as part of that byte counter.

```sh
$PYTHON \
  experiments/0021-metadata-screen/screen_manifest.py \
  --manifest configs/0017-development-manifest.json \
  --output results/0021-metadata-screen/graph-receipt-recheck.json
```

Search scope: current PoseBench tree and monodepth introduction/PR; RePoseD main and ICCV-evaluation trees; the named preparation and consumer code. This is a bounded search, not proof that no exporter exists elsewhere.
