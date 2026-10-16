# 0002 preflight: 64-frame development and identity-only grouping

2026-10-16. Publish this plan/config/source before executing. Preserve prior
protocols, source freezes, results and parent review. No held-out tracking runs.

Only video 0's first 64 frames, first two point indices visible at frame 0, query
arrival 0, 256x256 RGB, true fixed official weights. No future-visibility filter.
Save complete xy/two raw logits and actual frame times as each frame finishes;
save each completed subrun immediately. Refuse reused output directories. Record
UTC start/end, commands including env/timeout, source/config/protocol/input hashes,
environment, process RSS and wall times. Two CPU affinity/intra/inter-op/numeric
threads, 6 GiB address-space cap and timeout 900. Reuse verified cached bodies and
patched torch 2.10.0+cpu; no new large downloads. All processes serial.

Main process: batch64, then single-query16; compare common prefix at original
atol 1e-4/rtol0 and compare the saved prior 16-frame batch on identical query IDs.
Separate new process: repeat64, compare all raw outputs to main batch64. Fail
gates with nonzero exit, retain partial CSV/receipts. Same-host replay is not
independent-data evidence. No error/score calculation on held-out videos.

Metadata process: restricted NumPy-only decode of the verified 50-video archive;
for every video/points/occluded array record full shape/dtype/SHA256 and ordinal.
Record full RGB and actual first64 RGB prefix SHA256, and every exact RGB frame
hash. Union videos sharing ANY exact frame, full hash or prefix64 hash transitively.
The entire group containing video 0 stays development/quarantined; only video 0
may receive development inference. Other videos are metadata-only, never used to
tune the model or score. Different hashes do not prove different scenes.

Official schema contains only video/points/occluded, no scene/episode IDs. Record
known shared triplet-4 assets/front-left camera; unknown episode/scene and near
duplicate relationships. Freeze, if other gates pass, an exact-duplicate-group-
disjoint exploratory video-held-out pilot, not scene-independent confirmation.
Checkpoint training/model-selection non-exposure is not established.

Outcome-independent allocation: group ID is sorted zero-padded archive indices;
sort by SHA256 UTF8 `0002|20261019|group_id` then ID. Assign whole non-dev groups
in order to calibration until >=10 videos, then evaluation until >=20, remainder
reserve. Report overshoot and actual counts. If either minimum is unavailable,
do not freeze an executable evaluation. No group crosses a partition.

After resource/audit checks, implement separate 0002 metrics and fixed-group
future inference runner, validate with fixtures/development only. Holdout inference
is explicitly deferred until a later session after published freeze and review.
Freeze inputs in one commit, then a separate receipt referencing that commit and
each input hash; never self-reference a receipt's own commit/hash. Preserve draft.
