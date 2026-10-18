# Independent 0008 publication provenance review

**PASS — no publication blocker found** at `70bc57fd83634a8df66aecf072ea9b526e3af043`.

This is a read-only local provenance check. It does not rerun a metric, scorer, model, training trajectory or cloud operation. `audit.py` imports only the standard library and uses Git read commands. Actual audit execution is recorded in `execution.json`; its stdout/stderr and structured findings are retained.

The audit passed **9,885 checks** covering:

- All **265 public files**: 264 artifact hashes, 263 manifest entries excluding the two self-referential manifests, and 168 strict JSON parses. Manifest modes are 206 byte-exact copies, 52 redacted copies and five explicitly identified derivatives.
- All **52 redacted copies** restore in memory to their declared original SHA256 and size. The project/interpreter substitutions were independently recovered from the supplied original command tool records. Four publication source scripts required restoring only the interpreter assignment; their original literal placeholder destination strings remain untouched. No private reconstructed content was saved or exported.
- All **2,285 prior result/protocol Git blobs**, their working-tree bytes, **2,473 original tracked files** excluding the four authorized mutable files, **13 frozen inputs** and the original canonical freeze receipt. The dated note is append-only.
- All **18 parent-retrieved command records**, including their original command hashes, status/exit/duration and truncation flags. The nine complete stage child records independently extracted from original tool output equal the nine archived wait4 records field for field.
- All **32 unique trajectories**, their one-shot markers, source pins, recipe arguments, receipt hashes, **64 closed stdout/stderr files**, and **128 private prediction/weight file identity references**. This checks provenance and completeness, not the numeric arrays.
- Original compact-summary hash, byte length, valid JSON and terminal newline: no normalization was needed. Known original private project/interpreter paths and conversation IDs are absent from all public files; no binary payload or credential pattern was found within this scoped inspection.

The original publication link reviewer failed with exit 1 / raw status 256 on a baseline-only 0007 link in the date note. Its source, error and wait record are retained; the note's old bytes remain unchanged. The later reviewer passed. The README correctly distinguishes this from the scientific execution. Its saved 259-file verification snapshot precedes the final six publication artifacts; this independent audit checks the actual final 265 files.

Private preservation remains a qualified claim: the original cloud postcheck reports 496 non-reference 0006 cache files, 84 prior 0007 output files, 407 failed-cache files and 22 old markers preserved. Their private bytes were **not** accessed or rehashed here. Similarly, private output hashes establish linkage but do not provide their preimages. The independent replay's reported scientific correctness belongs to that separately observed saved-array replay, not this publication audit. Tool records do not constitute syscall-level proof of every internal access.

Reproduction requires a separate clean Git checkout at `70bc57fd83634a8df66aecf072ea9b526e3af043`, which contains exactly the original 265 cloud files. Run this parameterized script from outside that checkout and supply the evidence paths explicitly. Do not run it against the later parent-review commit or its dirty working tree:

```sh
python3 audit.py --repo REPOSITORY --output NEW_REVIEW_DIRECTORY \
  --observations ../tool-observations.json \
  --private-execution PRIVATE_EXECUTION_RETRIEVAL.json \
  --prior-inventory PRIOR_PUBLIC_INVENTORY.json
```

A local first-attempt auditor record (not part of this public bundle) preserves a schema-assumption error: the parent observation recipe uses `<PROJECT>`, whereas the archive uses `<PROJECT_ROOT>`. Recognizing both tokens resolved the failed audit checks without any repository or evidence changes. This was an audit implementation correction, not an artifact defect or scientific rerun.

`report.json`, `execution.json`, `stdout.log` and `stderr.log` preserve the original review bytes. The public `audit.py` is a packaging revision: private absolute locations become required CLI paths; an output-directory guard prevents writing into the inspected repository. The original exact-HEAD, clean-checkout and 265-file requirements are unchanged. Its original script SHA256 is `7ba74b8aebc9ca862363f47e57cd52274b202c31dc0e79201ea32a3cf3a9c214`; it is not claimed byte-identical to this parameterized script. See [packaging bindings](../packaging-bindings.json). No scientific algorithm or score is changed. The private execution retrieval is intentionally unavailable publicly, so a public clone alone cannot independently repeat the direct-tool comparison; the public metrics review needs no private inputs. No paper, novelty, performance-success or independent-scene claim follows from this PASS.

The public packaging revision has only been syntax/help checked, not executed as a new complete provenance audit. The preserved PASS and execution receipt belong to the original hash above. The prior inventory can be recreated from all `results/` and `protocols/` Git blobs at execution main `74f9330014aa8a13a8c55785992e115f045f2fa9`; its schema is `{"commit": "<full revision>", "files": {"<repository-relative path>": "<Git blob SHA1>"}}`, with 2,285 entries. The private execution retrieval cannot be reconstructed from public recipes.
