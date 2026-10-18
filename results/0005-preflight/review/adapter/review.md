# Independent native adapter import-gate fix review

**Qualified approval for the repaired source-tree and Pillow-origin gates. Formal execution freeze is not approved by this scoped review.**

Snapshot SHA256: `a5da972f0e4904816e6d1b012041cc0b5687e41d02e8e361f1e69af803d7d5dd`. The repository source was copied verbatim for testing; repository and prior review artifacts were not edited. No upstream file, ML module, model, checkpoint, data, cloud runtime or network was used. Fixture pins were substituted only for five self-authored inert leaf files.

`python3 import_gate_checks.py` exited 0. **21 checks passed**:

- The exact five-leaf tree with namespace-only parents passes, and importing its inert toy leaf confirms all three parents on that branch resolve only to the intended namespace directories.
- Initializers placed separately in `tapnet`, `tapnext`, `tapnextpp`, and `votsp2026` reject before any marker file is written.
- An installed regular `tapnet` package later on `sys.path` (which normally overrides an earlier namespace), an alternate top-level `tapnet.py`, and a second namespace location all reject without executing them.
- Already loaded top-level or descendant TAPNet modules reject. Extra file, same-name parent module, extra empty directory, missing/changed leaf, and root/parent/leaf symlinks reject.
- The exact Pillow/extra-dependency origin guard, extracted verbatim and run with inert module records, accepts an overlay origin and rejects an external Pillow origin or a symlink escaping the overlay.

Although the new `PathFinder` check explicitly resolves only the top namespace, the exact tree inventory forbids regular-package initializers and alternative modules in every admitted parent. Under ordinary trusted Python import machinery, the accepted top namespace has exactly one search directory, so installed packages elsewhere cannot supply its descendants. The toy imports confirm this path; no new blocker for the requested parent-initializer/installed-package scenario was reproduced. This is not a proof against concurrent filesystem replacement or hostile custom import hooks installed by otherwise trusted dependencies.

**Version-gate update and evidence correction:** The initial live source read lacked the imported `__version__` guard, but the subsequently saved `a5da972f...` snapshot already contains the root's update. The first review note about that snapshot lacking the guard was stale; its original text is retained in `review-first-with-stale-version-note.md`. Current portable tests execute the exact guard AST and verify matching versions accept, each of five mismatches rejects, and missing Pillow version fails closed. Expected versions are recorded by the adapter after direct equality with actual modules, so the scoped version concern is closed.

The root then changed only the module and run_capability docstrings. AST comparison excluding both docstrings found no behavioral difference. All 19 portable unittest methods (including grouped adversarial subtests) passed again on the final source SHA256 `6b59e6da6be00540f2333dc78eb596a655dd04f7afef5630342a14e1af7a6d2c`. Test file SHA256: `eacf21333df0881f8b2fbeab3fe921b635d6516e3bb2bceb3e4a06b4dcdcc792`. The owned repository addition is only `tests/test_tapnext_native_adapter.py`; the prior 21-case standalone audit remains preserved. The first portable test attempt had one fixture-path error because macOS temporary paths resolve through a symlink; the fixture now resolves the base path as production does, without changing rejection expectations. That failed output is preserved.

The separate supervisor/canonical receipt/CLI exit-status gate remains outside this fix review. No real native execution, package compatibility, memory limit, scientific forward, accuracy result or eight-frame feasibility is established here. The previous inference-loop/NPZ fake checks were read for context and not rerun; current tests target the changed import gates only.

Reproducer and observed outputs: `import_gate_checks.py`, `results.json`, `command.stdout`, `command.stderr`. `candidate_snapshot.py` preserves the tested bytes. `manifest.json` records their hashes.
