# 0026 publication verification

[PR57](https://github.com/Simonn777/embodied-vision-reliability/pull/57) merged at
2026-10-20T03:28:33Z. This verifies publication of a failed fixed comparison,
not research-goal completion.

- Source freeze: `b596542f9eee399219ac03fdbcd91b33e8417fd6`; [CI39363242000](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39363242000) succeeded before the single prediction operation.
- Complete result: `d68f88de6fd74a61447b12a1a35f93772542929a`; [CI39364275469](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39364275469) succeeded on that exact revision.
- Merged main: `0f3c9ec5a9c56d6d25ba967b17e1c26d63f4a27d`; [CI39364481803](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39364481803) completed successfully. The local isolated research checkout fast-forwarded to this merge and was clean; the owner's original checkout was not used for this publication.

The result package contains340 bound evidence copies:337 byte-identical and
three operation specifications with private path prefixes replaced. Its manifest
SHA256 is `703612fc59196aeabce35554029eb6cab8d43422afa98f411abf2eb33c583da5`.
A staged UTF-8/private-path/credential-pattern/binary scan passed350 files.
Frozen production source bytes were unchanged after source preparation.

The primary loses0.551569AUC@5percentage points to the strongest fixed control.
All80 pairs and560 new slots remain recorded. The independent saved-output audit
passes14,496 comparisons, but no held-out gain, novel-method success, complete-
pipeline time advantage or manuscript is established. [Result interpretation](0026-result-audit.md)
and [mechanism diagnosis](0026-mechanism-diagnosis.md) preserve this distinction.
