# 0021: fresh ETH3D / IMC2021 source screen

2026-10-19. **ETH3D is a feasible second capture-source acquisition route within the remaining download budget.** This establishes access to public metadata and a bounded file plan, not successful payload acquisition, runtime qualification, shared-depth inputs, or a research result. No image, depth, pose, intrinsic, match or model payload was requested. No scientific cohort is allocated here.

## Author linkage, access and files

The [MBA author README at d367871dc5197c03fe9c0fd86f1aaa504f61093c](https://github.com/ShngJZ/Marginalized-Bundle-Adjustment/blob/d367871dc5197c03fe9c0fd86f1aaa504f61093c/README.md#dataset-download) directly identifies both Hugging Face repositories. Public API requests succeeded without authentication and report `private=false`, `gated=false`.

| Author-linked repository | Pinned dataset revision | Metadata result |
|---|---|---|
| [ETH3D](https://huggingface.co/datasets/shngjz/e1a6043209bf56124cfa8fbe263cbcad/tree/0c4c984affb3b55249c2140531e6a4ac59090ba2) | `0c4c984affb3b55249c2140531e6a4ac59090ba2` | Complete recursive tree: 1,404 files, 10,713,958,484 bytes. Thirteen scene directories contain images, intrinsics, poses and laser scans; no predicted monocular-depth or correspondence cache. |
| [IMC2021](https://huggingface.co/datasets/shngjz/975921f0c7d37a4a5287de3c1e54fb09/tree/4c1486d2c4905fda3ca33bd6b341f2c8e4acb12f) | `4c1486d2c4905fda3ca33bd6b341f2c8e4acb12f` | Nine landmark directories. One inspected metadata example, `phototourism_british_museum/5bag_000`, has 5 images, 5 intrinsic files and 5 pose files, totaling 2,155,395 bytes. Full size was not enumerated. |

The ETH3D full metadata inventory is `eth3d-file-inventory.json`; each LFS object includes publisher size/hash. These hashes are acquisition checks to perform later, not locally verified payload hashes. The HF trees have no dataset README/license card. IMC bags can repeat images and overlap PhotoTourism/MegaDepth origins. A new mirror or bag is not a new independent capture domain. ETH3D is a different benchmark acquisition source; upstream pretrained-model exposure is not established absent.

## License and read depth

The [official ETH3D homepage](https://eth3d.ethz.ch/) was opened through the web tool, with its complete 52-line text returned; its License section states **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International**. This is a direct official-page reading, not a license inferred from HF public access. The [official documentation](https://eth3d.ethz.ch/documentation) was read through primary-source search indexing: it describes original and pre-undistorted images, COLMAP intrinsics/extrinsics and laser-scan ground truth. Local Python/curl HTTPS requests to the official hosts failed before receiving a response body; certificate verification was never disabled. The HF author mirror worked through ordinary TLS. Nothing was done to bypass a denied resource.

MBA's README instructs users to follow the original dataset licenses. Its GitHub metadata has no repository license declaration, and its source tree has no LICENSE file; the license for reuse of MBA implementation code is therefore unresolved. Reading its source to establish a data contract does not authorize copying its implementation. IMC's image-specific attribution/license chain was not resolved in this screen; do not substitute ETH3D's license. Neither a blanket redistribution right nor a commercial-use right is claimed.

All complete third-party source/README/license/API snapshots were moved intact outside the repository to chat `work/0021-fresh-source-screen/private-snapshots`; they are not publication artifacts. `private-snapshot-map.json` records their unchanged byte counts and hashes. The original append-only HTTP receipt was not rewritten: its older `saved` values describe acquisition-time locations and must be resolved through this relocation map. Public artifacts contain links, pins, hashes, metadata summaries and factual analysis, not the full unlicensed implementation.

## What the public source actually produces

The pinned [ETH3D loader](https://github.com/ShngJZ/Marginalized-Bundle-Adjustment/blob/d367871dc5197c03fe9c0fd86f1aaa504f61093c/MargBA/datasets/eth3d.py) natural-sorts `images/*.JPG`, maps each filename to one image index, constructs unordered image pairs, reads the matching intrinsic/pose text files and treats stored poses as world-to-camera. It supplies zero placeholder depth. Its [image conversion](https://github.com/ShngJZ/Marginalized-Bundle-Adjustment/blob/d367871dc5197c03fe9c0fd86f1aaa504f61093c/MargBA/datasets/data_utils.py) only transposes and divides by 255; no resize occurs at that loader step. The original author-side export from ETH3D calibration/undistortion into this mirror is not supplied. Consequently the exact original-versus-undistorted image provenance and pixel-center convention still require a bounded, prospective validation before calibrated inference; the presence of a 3x3 intrinsic file alone does not prove them.

The [unary producer](https://github.com/ShngJZ/Marginalized-Bundle-Adjustment/blob/d367871dc5197c03fe9c0fd86f1aaa504f61093c/MargBA/monodepth_estimator/depth_utils.py) generates one predicted depth per image, writes it under `depth_pr`, then packs per-image outputs into HDF5. These generated files are not in the downloadable ETH3D tree. The [ETH3D preprocessing defaults](https://github.com/ShngJZ/Marginalized-Bundle-Adjustment/blob/d367871dc5197c03fe9c0fd86f1aaa504f61093c/experiments/eth3d/preprocess.py) are DUSt3R depth and RoMa correspondences; the producer explicitly uses CUDA. Its loader also reads GT poses together with images. Running this pipeline unchanged would violate our no-GT prediction boundary and local CPU resource design. There is no verified ready-made shared MoGe realization here.

## Minimal independent frontend plan, not an allocation

An illustrative complete-scene resource envelope avoids any image/pose-based choice:

| Scene | Images | Images + intrinsics + poses |
|---|---:|---:|
| pipes | 14 | 147,821,011 B |
| meadow | 15 | 241,040,650 B |
| office | 26 | 292,748,701 B |
| Total | 55 | **681,610,362 B** |

No laser scans are necessary for relative-pose evaluation. This envelope is an example, not an assignment of these scenes to development or final testing. A future protocol can select fewer filenames using metadata alone and retain all selected outcomes. Distinct image pairs within a scene are dependent; the scenes, not the pairs, are the principal independent units. `relief` and `relief_2` should not be counted as independent locations without a physical-overlap check.

Use the already screened small [MoGe2-S-normal](https://huggingface.co/Ruicheng/moge-2-vits-normal/tree/26b477f41595707c5db6770294c0d1721e8ed4ed) checkpoint: revision `26b477f41595707c5db6770294c0d1721e8ed4ed`, 140,550,416 B, publisher SHA256 `79a16621928c2bf0ed04659218c55c01075e950507f40bb3332fb4c873d3e1dc`. The pre-MoGe3 source pin `925b8ed835a7a9cdb7578ba15c658a0afc969030` and CPU float32/native-SDPA route are documented in `docs/literature/0018-small-heldout-plan.md`. Its backbone construction avoids an implicit second DINO checkpoint. This would be our separately pinned frontend, not MBA's default or an exact reproduction of its results. The checkpoint card declares MIT; training-image exclusion is not proven.

Official [LightGlue release metadata](https://github.com/cvg/LightGlue/releases/tag/v0.1_arxiv) gives 47,500,279 B for SuperPoint-LightGlue and 5,206,086 B for SuperPoint, totaling 52,706,365 B. LightGlue is Apache 2.0; the [separate SuperPoint license](https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/LICENSE), read in full, restricts use to noncommercial research and restricts redistribution. Do not publish those weights in our repository. A fixed SIFT matcher is a zero-weight input control, not evidence of matching a stronger learned frontend.

The three-scene envelope plus these weights is **874,867,143 B** before dependency installation, comfortably below 2.764 GB. Use an explicit minimal dependency lock, one image/model at a time, a fixed resize with the same camera transform, locally supplied weights, and blocked hidden downloads. First qualify the CPU path and peak memory under 8 GiB; file size alone does not establish time or memory feasibility. Cache a single depth realization per image under `(dataset revision, filename, image hash, preprocessing hash, model hash)` and sample that exact map for every incident edge. Keep raw pose files inaccessible until a complete prediction seal. This construction would resolve the repeated-image depth identity ambiguity by design.

## Transfer accounting and decision

`fresh-source-http.jsonl` records **4,299,624 response-body bytes** from direct public metadata/source requests; no scientific/model payload bytes were downloaded. This is not a wire/TLS-byte claim. Reused root-fetched MBA commit/tree responses were 37,515 B and must not be charged twice. One web search and one official-homepage open have unavailable network-byte counters; report them as unmetered metadata reads, not zero. The machine-readable `fresh-source-receipt.json` preserves that limitation. The initial large IMC metadata response is included in the direct-body total.

**Admission:** a bounded author-linked ETH3D file route and upstream license are established; the old inaccessible ETH3D SLAM endpoint no longer rules out a second acquisition source. **Still required before a scientific run:** payload hash validation, mirror camera/undistortion contract check, no-GT frontend implementation, CPU/memory qualification, and a prospective scene/edge split. No current method is thereby admitted or shown effective. IMC remains a small optional resource with unresolved independent-origin and attribution details, not the preferred second-domain claim.
