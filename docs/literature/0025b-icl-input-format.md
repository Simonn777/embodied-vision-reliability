# ICL-NUIM clean TUM PNG: source-only input contract

2026-10-19. No RGB/depth/pose payload or data archive was fetched or decoded; no frames were selected. This note concerns the official **TUM RGB-D Compatible PNGs** links, not the native `.depth` format, noisy variants, or third-party conversions.

**Decision:** the documented depth and coordinate semantics are sufficient to freeze a physical depth falsifier. A smaller official acquisition route is not yet established; do not pretend the advertised HTTP byte ranges provide indexed access to arbitrary members of a `.tar.gz`.

## Depth values

The [official ICL-NUIM dataset page](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html) explicitly states its TUM-compatible PNG data works with the TUM PNG input format, links that format, and separately requires ICL camera intrinsics. The linked [official TUM format specification](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats#color_images_and_depth_maps) specifies 16-bit monochrome PNG, scale factor 5000, zero for missing data. More decisively, its unprojection example under “Intrinsic Camera Calibration” sets `Z=depth_image[v,u]/factor`, then `X=(u-cx)*Z/fx`, `Y=(v-cy)*Z/fy`. Thus the documented contract is **axial z in meters = stored PNG value / 5000**. Do not apply an additional ray-length correction.

In contrast, the [official codes page](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html) states that native `.depth` files contain Euclidean camera-to-point distances. Its [compute3Dpositions.m](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/compute3Dpositions.m) explicitly loads the native values and divides by `sqrt(((u-u0)/fx)^2+((v-v0)/fy)^2+1)`. The author's [SceneNet VaFRIC.cpp](https://github.com/ankurhanda/SceneNetv1.0/blob/master/VaFRIC/VaFRIC.cpp), `getEuclidean2PlanarDepth`, uses the same conversion. The latter is corroborating author source, not a verified copy of the exact original PNG-export converter. No exact PNG-export implementation was located in this bounded source search; the official linked-format contract is the basis for the PNG interpretation.

## Pixel orientation and calibration

The official [ICRA2014 paper](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/icra2014.pdf), §II, Eq.1 (PDF p.2), gives zero-based pixel intrinsics

```
K_minus = [[481.20,   0, 319.50],
           [     0, -480, 239.50],
           [     0,    0,      1]]
```

It explains the negative fy by the POVRay Y-axis being flipped relative to the image Y-axis because of the left-handed POVRay convention. Frames are 640×480 and lens-distortion-free. The [author getcamK.m](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/getcamK.m) uses MATLAB's one-based centers `(W+1)/2,(H+1)/2`; `compute3Dpositions.m` uses `u=1:640,v=1:480`. Subtracting one from these centers gives the paper's `(319.5,239.5)`. No extra half-pixel shift belongs in zero-based pixel unprojection.

The negative fy is a **3D coordinate convention**, not an instruction to vertically flip the stored RGB/depth arrays. For this entirely custom, pose-free virtual-rotation experiment, define a pixel-down coordinate system by `S=diag(1,-1,1)` and `p_plus=S*p_minus`. Then `K_plus=K_minus*S` has positive fy, `K_plus*p_plus=K_minus*p_minus`, and z/ray length are unchanged. Keep the RGB/depth arrays unchanged, and define the newly manufactured rotations directly in this canonical coordinate system. S has determinant −1 and must never be passed off as an SO(3) camera rotation. If a physical rotation were ever converted between both conventions it would require conjugation `R_plus=S*R_minus*S`; transforming a camera/world pose additionally needs the world-frame convention. No such pose conversion is needed or admitted here.

The source has slightly unequal fx and |fy|. A canonical camera for MoGe's single-FOV interface should be produced by the prospectively defined RGB/depth remapping, with its exact new K, rather than silently replacing the original calibration by a square-pixel K. Any resize uses the actual pixel-center map consistently for RGB, depth sampling, and intrinsics.

## Acquisition findings without payload

The already saved parent `head-01.json` reports the official clean trajectory-2 archives:

| Scene | Official URL | Content-Length |
| --- | --- | ---: |
| Office | `http://www.doc.ic.ac.uk/~ahanda/traj2_frei_png.tar.gz` | 301,828,475 B |
| Living room | `http://www.doc.ic.ac.uk/~ahanda/living_room_traj2_frei_png.tar.gz` | 429,825,445 B |

Their combined full transfer is **731,653,920 B**. These are reused HEAD observations, not new downloads or an acquisition decision. Both advertise `Accept-Ranges: bytes`; no archive hash, member inventory, compressed-member boundaries, or small-prefix feasibility has been verified. The official page labels the `...traj2n...` variants noisy; they are not substitutes for the clean links.

The official dataset, codes, living-room, office-room, and older test-dataset pages link sequence archives, rendering assets, source scripts, and separately published trajectories. I found no official per-RGB/per-depth URL manifest, seek index, or paired tiny RGB-D sample in those links. I did not fetch the separately linked trajectories. Ordinary compressed byte ranges cannot by themselves establish access to a specific tar member; a streamed prefix might be possible only after a separately bounded acquisition contract admits its bytes and verifies layout. It could also require traversing most of an archive if RGB and depth members are grouped.

The author-owned [`ankurhanda/simkinect`](https://github.com/ankurhanda/simkinect) offers ten small clean depth PNG entries in its public Git tree, but **no paired RGB entries**. Tree-only inspection at commit `9f62b4a3b24706bb7c2c7c8050e3a9beff9d39d4` is saved; none of those PNGs was fetched. Therefore that sample does not solve this image-plus-depth falsifier. Method-author repositories containing mirrored ICL subsets also surfaced, but are not admitted as exact official clean assets, and no mirrored image/depth files were read.

The official dataset page and ICRA2014 paper state CC BY 3.0. Attribution to Handa, Whelan, McDonald, and Davison and the original dataset remains required; no new per-image permission requirement is inferred.

## Evidence retained and remaining qualifications

`source-format-01/receipt.json` binds seven official HTML/MATLAB source responses: **51,666 B**. `source-format-02/receipt.json` binds the simkinect tree metadata and author C++ source: **21,538 B**. Total direct counted source/metadata response bodies are **73,204 B**. Web-tool search/page/PDF response transfers were not independently byte-counted; this number must not be described as all network use. The pre-existing page/HEAD receipt is preserved unchanged.

Before any physical run, the frozen input protocol still needs a deterministic frame rule and acquired-file hashes, a strict paired filename/shape/PNG-type check, the declared canonical remap, GT-only validity rules, and explicit failure preservation. None requires reading source trajectory poses. The documentary axial/scale/handedness contract is sufficiently specific; the unresolved item is economical, provenance-preserving acquisition, not the meaning of TUM-compatible depth.
