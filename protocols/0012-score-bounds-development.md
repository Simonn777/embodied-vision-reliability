# 0012: Aggregate bounds for dense geometric model selection

Date: 2026-10-19 America/New_York. Prospective numerical development study;
no improvement, novelty or manuscript claim. Historical failures stay closed.

Question: can bounds on aggregate nonconstant truncated Sampson cost reduce
individual residual evaluations enough to improve relative camera pose at matched
end-to-end time? Dense Match Summarization (DMS, CVPR2025) approximates the cluster
denominator/truncation. Space-Partitioning RANSAC (ECCV2022) already excludes
impossible inliers conservatively. The candidate difference is a moment-derived
bound on unresolved aggregate costs, not clustering or outlier culling alone.

## Fixed first diagnostic

- Independently check real-arithmetic inequalities and test mixed cells, near-zero
  denominators, model scaling and ties. Floating guards are not a formal certificate.
- Acquire only https://vision.maths.lth.se/viktor/posebench/relative/megadepth1500_dkm.h5
  (expected332,175,160 bytes; hard cap340,000,000 bytes). Record size/hash/source.
  No images, new model or cloud job. Inspect HDF5 names/shapes before any values.
- Allocate the lexicographically first complete scene permanently to development;
  select its first16 pair keys lexicographically for this numerical diagnostic.
  Determine scenes using names/metadata only, never quality or ground-truth poses.
  If identity cannot be established, stop before coordinate/label reads and revise
  explicitly. Do not use unselected scenes or read ground-truth R/t.
- Read selected coordinates and calibration only as needed for conventions.
  Generate64 eight-point rank-two hypotheses per pair, seed20261019; retain all
  valid hypotheses and failures. This is a numerical workload, not a strong pose
  baseline. Fixed leaf size32, starting partition128 cells, threshold1pixel;
  convert only if coordinates are normalized and intrinsics are supplied.
- Compare direct MSAC, the same partition with spatial-culling-only bounds,
  aggregate-moment bounds, and DMS-form approximate costs. Report exact argmin
  agreement, bound containment, residual/bound counts, timing including setup,
  and approximation-ranking differences. Record all16 pairs, no selected summary.
- Stop on bound violation/wrong winner. If no residual work is saved beyond
  culling, reject this mechanism. Actual speed requires same-language, comparable
  implementations; Python versus official C++ cannot establish efficiency.

## Outcome gate for a later frozen study

Before a manuscript: compare official DMS, strong PoseLib/SupeRANSAC and
SP-RANSAC using the same matches, samples, solver, refinement and device. Target
>=2 absolute pose AUC@5 points at matched end-to-end time on two independent
held-out sources, positive scene-clustered paired95% intervals, and no >0.5-point
AUC@20 loss. Report complete accuracy/time frontiers, failures and setup cost.
These are research gates, not CVPR acceptance criteria. A kernel speedup alone
does not meet the user's quality-improvement requirement. A benchmark scene used
for development is excluded from confirmation; the remainder cannot be called
full-standard-benchmark results. Novelty and independent replication are required.

## Resources

No paid services; <=4 CPU threads, about8GiB memory, <=30min per process.
Reuse the existing Python runtime read-only; if needed, h5py goes into this chat's
work directory with <=20MiB wheel budget and no dependency replacement. The
previously recorded remaining allowance is3,238,351,458 bytes; count new data and
package transfers. Publish only original code, small derived results and brief
attributed notes, never the upstream HDF5. Preserve the older checkout's staged
edits. An initial documentation command used the wrong working directory and
failed before repository edits; it was corrected before any data acquisition.
