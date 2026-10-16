# Protocol 0002-D implementation verification revision 1

2026-10-16. Preserve the original development plan and source freeze `e9ca032`.
Source `e11b70a` added a pinned-source digest gate and fresh-process minimal replay.
Its initial real-weight run is preliminary evidence, retained under its original
receipt; it predates the strengthened provenance/gates below and is not the final
validated run. No final-evaluation data was inferred.

Independent read-only review identified missing implementation gates. Publish
this revision and code before a serial development rerun: nonpassing numerical
checks must produce validation_failed/nonzero exit; validate the decode manifest
as whole video 0 and hash-check its video/points/occlusion arrays before inference;
record NPZ, manifest and acquisition-receipt SHA-256; verify actual import origins
before loading TAPIR and loaded file paths/hashes afterward against the fixed
upstream sources and acquisition receipt. Reject noninteger/Boolean arrivals,
non-uint8/non-256px input and invalid normalized coordinates; verify the exact
active-output mask. These repair implementation validation, not the research
hypothesis or original development subset.

Add the **new** late-arrival check: compare late_query_16[4:,1] to a separate
12-frame stream starting at original frame 4, same spatial query, local arrival 0.
This tests arrival image/coordinate initialization, beyond merely masking early
outputs. Stateful fake-model CI tests additionally check old-state continuation,
new-group initialization and zero-active bypass. Preserve original atol=1e-4,
rtol=0; report all differences, no widening after seeing results. Run one fresh
process four-frame replay after the strengthened main run, then compare it to the
first four-frame output. Earlier downloads, failures and preliminary run remain
in the evidence package and budget. No parallel model run is authorized.
