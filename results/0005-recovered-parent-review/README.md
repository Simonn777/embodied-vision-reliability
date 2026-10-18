# Independent review of the recovered CPU capability run

The completed result at `0436cb438666db9bd0645f131632fc2fafcdf025` passed an independent saved-artifact review. See [review](review.md), [machine-readable audit](audit.json), and [parent-observed cloud command statuses](observed-command-status.json). These are AI-assisted checks; they establish neither a new method nor an accuracy improvement.

All 64 output rows, 74 manifest-bound archived copies, 16 frozen inputs and 1,651 historical result files passed the stated checks. An independent float64 calculation from saved logits reconstructed the native head coordinates within 0.00001544924731433639 pixels. This is an arithmetic diagnostic, not ground-truth accuracy or a revised experiment threshold. The remote checkpoint/cache and unavailable state-tensor values were not independently rehashed by this local review.

The original archive remains byte-for-byte unchanged, including the two failed review-only commands and four as-executed scripts with trailing blank lines. Parent tool-status observations agree with the archived scientific process receipts, but are not an additional wait4 measurement or signed attestation.

## Reproduce the saved-output audit

Use a separate clean checkout at exactly `0436cb438666db9bd0645f131632fc2fafcdf025`. Keep the audit script outside that checkout, and pass its directory as the argument:

```sh
python3 /path/to/check_saved.py /path/to/clean-reviewed-checkout > audit-reproduced.json
```

Python with NumPy is sufficient. The script does not import repository scientific code, access raw datasets or weights, or perform inference. It deliberately rejects another HEAD or a dirty checkout. The review source and captured output are retained unchanged; the historical command in `review.md` records the original local execution path.
