# Outcome-focused research decision, 2026-10-19

The owner requires an innovative method with substantial reproducible improvement
before a CVPR-targeted manuscript. Current evidence does not meet that goal.
Direct inspection of saved results confirms: 0002 risk4/1768→7/1768; 0004 reset
46 repairs/120 harms; 0006 −0.570956dB; 0008 −0.000956dB fixed-update and −0.330065dB
fixed-time; 0010 61/128 versus65/128; 0011 62.9625% versus64.1878% with worse EPE.
These are saved-result readings, not new replications. Prior stop decisions stand.

The scientific problems are insufficient headroom, weak mechanism-to-error links,
small dependent cohorts and excessive emphasis on execution verification. For
example, 0002's baseline risk0.226244% cannot decrease by2 percentage points.
Negative average reset benefit also does not logically rule out every conditional
policy, although the historical experiment must not be rescued by tuning positives.
Per-case oracle-best baselines must be identified as oracles in future comparisons.

Current primary literature rules out generic multi-teacher verifier distillation
(Track-On-R), long-video redetection alone (TAPNext++), generic tracker memory
control (SAM3RL), and generic local-confidence mixture calibration (CoRe).
The next bounded hypothesis is [aggregate geometric cost bounds](../protocols/0012-score-bounds-development.md).
It is selected for an identifiable numerical approximation and accessible real
correspondence data, not because improvement is assured. It must distinguish
itself from both DMS and Space-Partitioning RANSAC. No paper has been started.
