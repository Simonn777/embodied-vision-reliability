# Coordinate co-attention: bounded prior check

Checked October18 New York, before inspecting any0011 result. No protocol,
parameter, gate or allocation changed. This supplements the frozen prior note.

SimCC identifies Yin et al., *Attentive One-Dimensional Heatmap Regression for
Facial Landmark Detection and Tracking*. Reviewed introductions, complete§4,
§5.1 and selected ablations in [v1](https://arxiv.org/html/2004.02108v1)
(April5,2020) and [v7](https://arxiv.org/html/2004.02108v7) (August27,2020).
The arXiv history was checked; accepted venue and official code were not
independently verified here.

The method learns x/y feature affinities with row-softmax co-attention, mixes
intermediate features, then predicts supervised axis heatmaps and separate
maxima. A tracking extension mixes past features. Its output marginals may
change; the inspected equations do not constrain a joint to two fixed output
marginals or use an oblique test-time view. Thus broad x/y-dependence recovery is
already prior art. The narrow fixed-marginal/oblique-evidence development question
remains testable, but its novelty is unestablished. Standard transport is not a
new solver, and a positive tiny pilot cannot establish priority or paper readiness.

Two primary-domain searches were used: the named Yin/co-attention lead, then
pose/landmark/keypoint with oblique/copula/rotated-marginal terms. The latter
returned no result; this is weak retrieval evidence, never an absence proof.
No model, video, paid service or author implementation was run in this review.
