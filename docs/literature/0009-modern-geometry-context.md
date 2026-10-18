# 0009 modern geometric-estimation context

2026-10-18. Limited primary-source follow-up, not a comprehensive 2026 survey.
Three queries followed an earlier two-query CVF search. Direct CVF HTML/PDF
opens returned HTTP403; indexed CVF text and author arXiv/README pages were
available. No paper implementation or published benchmark was reproduced.

- **Dense Match Summarization for Faster Two-view Estimation**, Jonathan
  Astermark, Anders Heyden and Viktor Larsson, CVPR2025; arXiv2506.02893v1,
  June3,2025. Read arXiv sections1–3,4.1–4.2 and part of4.3 (not the whole paper) and the
  [author README](https://github.com/jastermark/DMS), whose software is labeled
  MIT. The method clusters paired coordinates, selects representative matches,
  and summarizes refinement constraints with small matrices. The derivation
  treats a cluster as entirely inlier or entirely outlier and approximates
  Sampson denominators at representatives. It uses a common PoseLib-based
  estimation framework and reports CPU experiments. These are author findings,
  not ours. Relevance: reducing correspondence redundancy, richer group
  constraints and computational comparisons already have substantial prior art;
  simple grouping or deduplication is not a new contribution. Its proxy-cost
  approximation is different from an exact permutation tail, without establishing
  that the latter is novel or useful. [Versioned source](https://arxiv.org/html/2506.02893v1).
- **Less Biased Noise Scale Estimation for Threshold-Robust RANSAC**, Johan
  Edstedt, arXiv2503.13433v1, March17,2025; CVPR2025 **Image Matching Workshop**,
  not the main conference. Abstract/indexed primary text only; HTML was opened
  but detailed method sections were not read in this pass. SIMFIT++ addresses
  fitting-data reuse and threshold effects in noise estimation, and proposes
  combining scene-level estimates. The implication for this study is that a
  fixed pixel tolerance and noise model must remain explicit; beating a manually
  weak threshold is insufficient. [Versioned source](https://arxiv.org/html/2503.13433v1)
  · [official workshop record](https://openaccess.thecvf.com/content/CVPR2025W/IMW/html/Edstedt_Less_Biased_Noise_Scale_Estimation_for_Threshold-Robust_RANSAC_CVPRW_2025_paper.html).
- **Adapting Dense Matching for Homography Estimation with Grid-based
  Acceleration**, Kaining Zhang, Yuxin Deng, Jiayi Ma and Paolo Favaro,
  CVPR2025, pp6294–6303. Read indexed official abstract/introduction and
  [author README](https://github.com/KN-Zhang/GFNet), not full text. GFNet adapts
  learned dense matching through coarse grid flow. Its training/inference
  dependencies and data/checkpoints are not admitted to this CPU screen, and
  their license/size/runtime were not qualified here. Relevance: a later claim
  about complete homography estimation needs modern learned/matching context;
  outperforming count on toy translations cannot establish it.
  [Official paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Adapting_Dense_Matching_for_Homography_Estimation_with_Grid-based_Acceleration_CVPR_2025_paper.pdf).

No assets, source packages or checkpoints were acquired in this reading. The
small synthetic screen remains a falsification decision only. None of the above
is claimed to have been merged into a new method, distilled, or reproduced.

## Author-provided precomputed-match route (metadata only)

The DMS repository API identifies `master` as its default branch. Two failed
`main` raw-source opens returned404 before this was checked. The author
[download script](https://raw.githubusercontent.com/jastermark/DMS/master/data/download_megadepth.sh)
links DKM and RoMa MegaDepth1500 HDF5 files on the author's university server.
Two explicit no-redirect HTTPS HEAD requests each returned200, Content-Length
**332,175,160 B** and Accept-Ranges: bytes. Last-Modified was April15,2024 for
DKM and October10,2024 for RoMa; ETags were recorded in the observed headers.
No HDF5 body, image, match, GT or new experimental result was read. The software
MIT license is not being substituted for a dataset-specific license. Payload
schema, scene grouping, integrity, rights and actual CPU baseline remain unchecked.
These metadata could inform a later distinct proposal; they do not admit data,
change the current 0009 distribution or justify a method claim.
