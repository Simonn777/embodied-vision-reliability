# 0013: measured baseline pose headroom

**Measured, not a method improvement.** Official PoseLib2.0.5 completed all16 preallocated scene0015 pairs after published compatibility freeze `07f5350`. All10000 raw-pixel correspondences per pair were used with the fixed official settings. The first attempt's input failure remains [preserved separately](../0013-baseline/README.md).

| Metric | Result |
|---|---:|
| Successful estimates |16/16|
| Pose AUC@5 |81.169622%|
| Pose AUC@10 |90.584811%|
| Pose AUC@20 |95.292406%|
| Max-pose error below5 degrees |16/16|
| Translation dominates rotation error |15/16|
| Whole prediction operation |5.926280s|
| Sum of native estimator-call time |3.098395s|
| Median native estimator-call time |0.196478s|

Rotation errors range from0.036084 to1.014796 degrees; directed translation errors range from0.126850 to2.358317 degrees. This cohort does not demonstrate catastrophic matching or sign-flip failures. Its remaining error is mostly fine-grained translation precision. Sixteen overlapping pairs from one scene cannot characterize full-benchmark failure prevalence.

All predictions were sealed and independently hash-checked before the separate evaluator opened selected R/t. Both stages verified actual HDF5 content hashes, with no held-out numeric decoding. All estimates succeeded, so coverage-penalized and successful-only AUC coincide; no artificial failure penalties contribute here. Counts use strict thresholds and AUC follows the frozen DMS trapezoidal convention. Returned RANSAC masks/stats precede final refinement.

[Complete evaluation](evaluation.json), [per-pair table](per-pair.csv), [execution receipt](execution.json), [prediction seal](predictions/seal.json) and logs retain all results. Time describes this baseline only. No competing method, complete DMS reproduction, speed superiority or accuracy gain has been measured. No manuscript is admitted.

The next step is a distinct, prospectively specified diagnostic of translation precision, including controls for ordinary post-refinement and spatially correlated match errors. Do not select only the largest-error pairs or tune against held-out labels. The closed0012 speed family stays closed.
