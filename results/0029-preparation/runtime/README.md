# 0029 offline runtime checks

## Installation attempt 01: failed before model execution

Source `fe4ae08ccd708541a1e23f6018ed7925959558bb` passed [exact-source CI](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39378234232) before the supervised installation. All five child operations completed: fresh venv, local pinned wheels, two inspected offline source builds, and installation of those generated wheels. Final distribution enumeration then rejected `installed METADATA missing/ambiguous`. The outer result is **failed**, exit 1, 6.133821 seconds, 131,728 KiB sampled peak RSS; no remaining child or cleanup error.

Stdlib-only inspection identified the cause: setuptools 80.9.0's RECORD contains its own top-level METADATA and 16 nested vendored distribution METADATA files. The original suffix filter incorrectly counted all 17 as the owning distribution. The narrow prospective repair selects the single direct site-root metadata path; it retains the exact distribution closure and complete installed-file inventory. No package/model setting, fixture, quality threshold or scientific comparison changes.

The [attempt directory](installation-01/) preserves the failed inner/outer receipts, exact source CI, specification and all child logs. Public copies replace private path prefixes; their original and published hashes are recorded in `publication-manifest.json`. The original private destination remains unchanged. A new source freeze, exact CI and fresh installation destination are required before another attempt.

No model inference, benchmark image/mask access, accuracy result or research gain occurred in this failed installation.
