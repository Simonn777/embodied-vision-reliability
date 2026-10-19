# Cached first-probe guard (prospective)

`guard.py` is read-only toward the old capability worktree and cloud cache. It
writes only a fresh output directory. It never invokes the probe, loads a model,
deserializes weights, installs, downloads, deletes, resumes or repairs anything.
Only eight fake filesystem/process tests have run. No actual cached payload or
real `/proc` was inspected by this implementation task.

Future separately frozen invocation:

```
python guard.py --base ORIGINAL_CLOUD_BASE --repo ORIGINAL_CAPABILITY_WORKTREE --output FRESH_GUARD_OUTPUT
```

The old worktree must remain at `e9e7c760…`, clean, with the exact receipt
`93629d385…` / SHA256 `8dfec736…`, source `ececb253…`, and all 25 raw inputs
matching source, original main and index. No remote branch refresh is assumed.
`preservation-pins.json` maps the original 65 published closed evidence files to
their original cloud locations; it is itself hash-bound. Stage exits 0 and the
auxiliary before-probe exit 1 and logs must retain their identities. Original and
new published CI verification remains the parent's separate prerequisite; this
local guard makes no network CI claim.

All 12 final assets and the separately named download checkpoint are streamed and
verified. The 343,605,248-byte partial must have its fixed saved SHA256, a different
inode from both final paths, and match the final prefix exactly. Three named dot
files retain their original names/sizes and receive newly observed hashes; they
are **excluded from model inputs**, and their origin remains UNKNOWN. File
before/after stats, final stability assertions, cgroup before-state and available
CPU slots are retained. The guard does not purge caches or increase limits.

Every visible process cmdline is split on `bytes([0])`. Process/fd enumeration,
cmdline reads, each fd stat and each readlink have separate success / ENOENT /
permission / other counts. Matching inode descriptors additionally record fdinfo
access flags. Any related scientific process, matching holder (even read-only),
permission gap or other I/O gap blocks. ENOENT races are explicitly retained, not
silently counted as inspected absence. Empty visible process inventory blocks.
Even PASS is only a visible snapshot: hidden, prior or later writers and global
absence are unproven. The original probe marker, output and supervision path are
checked absent both before payload verification and before PASS.

`PASS_CACHED_GUARD_ONLY` is not a scientific result or launch action. An abrupt
kill may leave only the latest checkpoint; the external caller must preserve its
real exit/log evidence and never infer PASS from an incomplete file. Output is
fresh and cannot overlap preserved input directories. No old file is rewritten.

Run the artificial checks with `python -B -W error test_guard.py`. The successful
full-flow fixture mocks Git and uses tiny fake asset bytes; it does not establish
real cache identity, real Git preservation, cloud resource availability or probe
feasibility.
