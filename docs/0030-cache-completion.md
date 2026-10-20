# Cache completion: actual supervision scope

2026-10-20 clarification of the completed `cache-01` operation. The detailed counts and times remain in `cache-result-review-01.md`; no output, receipt, or original source has been changed or rerun.

The actual supervisor was `work/0026-supervise.py`, SHA256 `7c7798fe76756ace692fe9d9a83755d758690c62f0a6cc7f11af655dd5c60efe`, producing schema `evr0026.operation.v1`. It reused process monitor SHA256 `4b8b0efa4fd713a032967c9b34539eac128184ef6db4e8535e83d785c6bbebcd`. The saved outer receipt reports exit 0, `termination=null`, no remaining descendants, and no cleanup errors under that older supervisor's semantics.

This was **not** the later `supervise_exit_checked.py` implementation with SHA256 beginning `8f7bcbb8`. The old receipt does not contain its fresh post-exit snapshot fields or its stronger observation-order evidence. We do not infer or claim those checks from the old receipt. The parent accepts this clean cache under the actual older qualified semantics; future model operations use the newer exit-checked supervisor. The supervision difference is retained explicitly rather than hidden by a cache rerun.
