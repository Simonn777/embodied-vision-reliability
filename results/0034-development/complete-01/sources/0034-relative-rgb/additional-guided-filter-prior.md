# Additional primary overlap: learned guided upsampling

Source-only reading on2026-10-20, with the0034 design fixed before scientific execution. [Wu et al., CVPR2018 official record](https://openaccess.thecvf.com/content_cvpr_2018/html/Wu_Fast_End-to-End_Trainable_CVPR_2018_paper.html) and [author v1, §§3.1–4.3](https://arxiv.org/html/1803.05619v1) were checked. The final conference PDF returned403; final-version equivalence was not audited. The v2 page is a later2019 extension and was not conflated with the2018 method.

The method computes local linear guidance coefficients at low resolution, upsamples them, and applies high-resolution guidance; it also learns a task-specific guidance transformation jointly with the upstream network. Therefore combining learned coarse predictions, original RGB and high-resolution supervision is established. The0034 pointwise signed-affinity screen is not a full reproduction of this stronger jointly trained system. A gain over the restricted local bilateral control would not establish superiority over all guided refiners. No claim of new upsampling or generic RGB refinement follows.

Web text only; no local paper/model/data payload was downloaded. Web response bytes were not metered. This note changes no frozen operator, optimization setting, gate or output.
