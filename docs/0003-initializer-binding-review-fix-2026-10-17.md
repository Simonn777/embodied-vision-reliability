# Pre-freeze correction: bind the evr package initializer

AI-assisted review correction to candidate ec26b8fc1a4901d9e249480da6cc9f772885baca.
October 17, 2026 (America/New_York); exact test timestamps below are UTC October 18.
No method, allocation, resource budget, threshold, numerical result or data scope
changed. This is still an unfrozen candidate awaiting independent review.

Python executes `evr/__init__.py` when importing the package. The previous 44-file
mandatory set and 38 historical inputs did not bind that initializer. The new
runner explicitly imports `evr`, adds **src/evr/__init__.py** to mandatory inputs,
and verifies `evr.__file__` resolves to the canonical repository initializer and
its source hash matches the receipt, alongside the existing five submodules.
The same canonical receipt gate now binds its current/index/HEAD/input/published
Git blobs. The mandatory set is exactly **44 → 45**; the historical chain stays 38.
The initializer itself is unchanged, SHA256
631cc3c910f6e790b2b5661bf7d28e99a612c534e82619c54084d9b64f9c5abe.

Two new synthetic test methods cover initializer current/index/old-commit-new-hash
mismatches, canonical imported package origin and hash rejection. These use a
temporary Git repository and artificial source bytes, not a dataset or model.
The only previously published code modified is the not-yet-frozen new runner and
its tests. Formal helper/protocol/config, all older code and every attempt snapshot,
source audit and actual half-cell input/output/wait record remain byte-identical.
Earlier 44-input maps are historical; latest is `review-input-map-v3.json`.

Actual **attempt3: 17 tests passed**, command/source snapshots and real parent wait4
in [evidence](../results/0003-annotation-consistency-stageA-20261017/).
UTC 2026-10-18 02:22:18.734343–02:22:19.129160 (New York October 17 22:22:18–19).
Wall **0.394826 s**, peak RSS **16,128 KiB**, exit/raw wait **0/0**, two CPU slots,
five numeric thread variables 2, 6 GiB AS, external timeout 900 seconds. The parent
itself has no further observed supervisor exit receipt. The prior 14/15-case logs
are preserved as separate source-bound historical attempts; they do not cover the
new initializer check.

No NPZ/TAR/metadata/RGB, new GT/camera/geometry access, inference, saved-R rerun,
download/install/fees or permission/schedule changes. No canonical geometry receipt
or one-shot marker was created. The original rigid failure, half-cell INCONCLUSIVE
result and carried 22,910,156-byte ledger remain unchanged. Review this new input
SHA before authorizing its separate receipt; do not freeze or execute the prior
candidate by mistake. Final exact-head CI is verified in PR19.
