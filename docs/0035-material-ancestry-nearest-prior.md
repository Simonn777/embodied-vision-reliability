# Material ancestry, uncertain correspondences, and disconnected surfaces

Source-only review, 2026-10-20. Four primary works beyond the already reviewed SAM-PT, TubeletGraph, and FluxGraph. Web text only; no local PDF/model/data acquisition, scientific arrays, inference, or D35 gate changes. Web response bytes are unmetered. This is a bounded prior screen, not a novelty proof or a method admission.

**Conclusion:** I found direct prior art for preserving template identity through cuts and disconnected pieces, and for using correspondence reliability. I did not establish a defensible new mechanism from combining those operations. The remaining substantive question is whether available RGB observations disambiguate *material identity* when correspondences are ambiguous or absent. Merely deleting adjacency edges, changing overlap normalization, or adding tracker confidence does not answer it.

## Four closest primary reads

1. **Tsoli and Argyros, “Tracking Deformable Surfaces That Undergo Topological Changes Using an RGB-D Camera,” 3DV 2016.** [Author paper](https://users.ics.forth.gr/~argyros/mypapers/2016_10_3dv_tearing.pdf), [author project](https://projects.ics.forth.gr/cvrl/tearing/).

   Read author-indexed paper text for §§3.1–3.4, Eqs. (2)–(9), plus the author project abstract; direct full-PDF retrieval timed out. The method registers an initially known mesh to RGB-D geometry, constrains SIFT matches to previous and initial frames, and relaxes edge-length constraints using measured depth gaps. It then removes overstretched gap-crossing edges. This directly precedes “old material adjacency breaks while point identities persist.” Its extra observation is calibrated depth, and its initial geometric template is stronger than an initial RGB mask. It does not establish monocular segmentation of previously invisible interior material or a probabilistic solution to ambiguous point identity.

2. **Manogue et al., “Generalizing Shape-from-Template to Topological Changes,” STAG 2025.** [Author paper](https://arxiv.org/pdf/2511.03459), [publication/version metadata](https://arxiv.org/abs/2511.03459).

   Read §§3.2–3.5, Algorithm 1, and §§4.3–4.5. It takes a known 3D template and 2D correspondences, then optimizes a displacement field to reconstruct an approximately isometric surface despite tears and disconnected components. The formulation assumes an injective, unoccluded projection. Real examples use selected high-confidence RoMa matches. Crucially, §4.4 explicitly excludes a small torn piece that has no correspondences. Thus “disconnected material reconstruction from matches” is already implemented; absence of trackable evidence remains an actual coverage limitation, not just a missing graph operation. Its reported real-data reconstruction reference is constructed with manually separated pieces and per-piece SfT, not independently measured material-identity ground truth.

3. **Tsao and Tsao, “A topological solution to object segmentation and tracking,” PNAS 2022.** [Published DOI](https://doi.org/10.1073/pnas.2204248119), [author manuscript](https://arxiv.org/pdf/2107.02036).

   Read manuscript pp. 8–14: local affine correspondence, occluding-edge ownership, surface-component graph, synthetic results, and limitations. Distinct visible components receive persistent labels through a spacetime connectivity graph; the paper already discusses splits and joins in the image. These are demonstrated under deformation and occlusion, not evidence that arbitrary material fracture or mixing has been solved. The authors explicitly require accurate correspondences; their described synthetic demonstration uses textured surfaces without noise. This is close prior against claiming that correspondence-based surface connectivity itself is new.

4. **Doersch et al., “TAPIR: Tracking Any Point with Per-Frame Initialization and Temporal Refinement,” ICCV 2023.** [Official paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Doersch_TAPIR_Tracking_Any_Point_with_Per-Frame_Initialization_and_Temporal_Refinement_ICCV_2023_paper.pdf).

   Read CVF-indexed method text §§3.1–3.2 and uncertainty discussion/ablation §4.2; the direct PDF open failed. It initializes matches per frame, refines them temporally, and predicts both occlusion and localization uncertainty. Therefore “use confidence to reject unreliable trajectories” is an established control. These predictions describe a queried point's localization/visibility, not a posterior over which initial material owns a newly exposed region. The paper does not establish fracture-specific uncertainty calibration. A high-confidence tracker output cannot be assumed to certify ancestry.

## Exact distinction: material identity versus correspondence quality

For a surface element visible both initially and now, a *correct material correspondence* already transfers its initial identity. Any adjacency/velocity statistic computed from that same correct trajectory is identity-redundant. It can still regularize **uncertain** correspondence estimates, but the benefit would be a better association estimator—not an additional observation.

For a pixel \(y\), let \(\mathcal C(y)\) contain plausible initial correspondences, possibly including “not previously visible.” If every candidate lies in one initial object, the evidence supports that identity. If plausible correspondences cross object/background boundaries, a unique label is not identified. If only the unseen state remains, absence of a match is not negative evidence. Maintaining this ambiguity is ordinary uncertain label propagation; giving it a new name would not establish a contribution.

A decisive counterexample is two identical textured pieces exchanging positions while hidden: all visible frames, motion/adjacency statistics, and matcher confidence may be identical under two different ancestry assignments. Newly revealed interior material can likewise lack any initial-image surface counterpart. A topological graph cannot distinguish those worlds without another observation or an explicit learned prior. Conversely, a true fragment may separate along the viewing direction without a visible 2D adjacency rupture.

## Falsifiable missing-evidence question

The narrow unresolved question is: **on a complete, fixed candidate population, does RGB evidence available before the decision discriminate true descendants from background/other-object distractors beyond ordinary positive/negative correspondence propagation?**

If D35 later supplies enough support, the cheapest meaningful follow-up would hold its proposals and all initial objects fixed and compare the same correspondence observations, inference budget, and training supervision:

- ordinary propagated initial membership, with the tracker's existing confidence/occlusion handling;
- generic uncertainty-aware propagation that preserves multiple plausible parents and unknown mass;
- any proposed topology-conditioned association using those same inputs.

Keep untrackable and newly exposed surfaces in the denominator with a prespecified unknown behavior. Compare descendant recovery at matched foreign-object false positives, then the unchanged full-sequence J/Jtr. Do not evaluate only the confidently trackable subset. If topology supplies no improvement over the same-input uncertainty-aware control, it adds no demonstrated useful mechanism. If ordinary propagation accounts for the gain, report a known-control opportunity, not an ancestry-method innovation.

No new thresholds, experiments, or modifications to D35 are proposed here. A useful empirical opportunity is still possible, but this screen does not justify implementing a renamed cut graph or generic verifier.
