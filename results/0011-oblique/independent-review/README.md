# Independent parent review

The parent fetched commit `0302bdab58fc38ca82669085e1ccb5874c54aac0`, verified the
four bundle blobs and all 101 archive members against their sizes/SHA256, then
recomputed the public error tables with a separate standard-library implementation.
[Source](parent-arithmetic.py) and [receipt](parent-arithmetic.json) preserve the
calculation: 341 compared leaves, including 157 floating values; no mismatches,
maximum difference 1.4211e-14. All 256 identities/2,816 method errors are present.
The two groups have 87 and 53 primary eligible rows, with complete method coverage.
Exact gain is -113/9222 and all three quality gates fail: **NO_GO**.

This parent calculation starts from saved errors, not raw GT. The cloud's
independent metrics checker separately recomputed errors against private GT,
and another checker independently replayed saved logits using positive-domain
IPF rather than the frozen log-domain solver. Their actual tool executions and
closed logs are linked through [review observations](parent-review-tool-observations.json)
and the text archive. Both source hashes match the reviewed public scripts.
The parent verified all 24 frozen files against current bytes, Git index,
source commit a288f65 and pre-execution main merge a0b0616.

[Scientific command observations](parent-tool-observations.json) directly record
all five tool exits. The two failed read-only summary commands are retained.
The second was transport-truncated and has no separately archived complete
terminal output; that gap remains open. The later publication turn was too large
for `read_thread`, so publication verification instead uses direct remote Git
objects and byte-exact archive retrieval. Neither limitation is rewritten as a
successful complete log retrieval.

Post-hoc description of saved diagnostics finds maximum oblique iterations 357,
residual 9.9878e-10, fixed-mean difference 3.6271e-6 pixels, axis-joint difference
8.8818e-16 and native/reference decoder difference 5.0980e-5 original pixels.
These are descriptive maxima under unchanged gates, not parameter selection.
The parent did not regenerate joints or validate the original 50-video selection;
the separate cloud replay covers joint reconstruction. Private cache preservation
is supported by the documented cloud checks, not a new local cache-byte audit.

No model, frozen scorer or data preparation was rerun. A correct reproduction
of failure is not task-quality improvement, statistical replication or novelty.
