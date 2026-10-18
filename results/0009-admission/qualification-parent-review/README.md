# 0009 qualification archive — independent parent review

2026-10-18. **PASS with evidence limits for publication of the failure record. Data remains NOT ADMITTED.** This review made no HTTP request, import probe, model call, data decoding or geometry run, and left all 69 qualification files unchanged.

[The executed checker](check_qualification.py) completed **915 checks**, exit 0; see [result](check.json) and [actual local execution receipt](execution.json). It inspected the final 69-file / 895,428-byte archive at commit `7e38cb50019e22f2bed7cd431b82d5f46f9cd122`, whose artifact inventory SHA256 is `39aaf40e5d31897f3f38b96a42388bb61fcc806e58497f8f5170377907cc931a`.

## Integrity and direct observations

All 68 listed artifact digests match; the inventory excludes itself. The copy manifest covers the remaining 67 files: **29 byte-exact bindings, 36 redacted bindings and two explicitly derived interpretations**. All 36 redacted files reproduce the declared original byte lengths and SHA256 hashes after reversal. Four publication source recipes contain original placeholder literals, so reversal selects occurrences using length and exact digest constraints rather than replacing every token. All four yielded exactly one matching reconstruction.

The archive represents all 45 original runtime/directory text files and binds two excluded private binary ranges through the original 47-file preservation inventory. The actual two binary range bodies were not read locally. Reverse-hash consistency proves the relation between published bytes and recorded originals; it does not inspect the current cloud filesystem or independently establish that all 47 private files remain unchanged.

The safely redacted [runtime observations](runtime-observations.json) and [directory observations](directory-observations.json) are byte-exact copies of the parent's records. The checker verified their digests, redaction, exits, truncation flags and outputs against the corresponding original private retrievals, restricted to **four runtime and six directory commandExecution records**. The private retrievals and prefix values are not published. Saved capture receipts and final summaries agree with those actual tool outputs. NumPy/SciPy availability and both cv2 import failures remain qualified as import/attribute checks; they prove neither geometric functionality nor complete dependency integrity.

## Accounting failure is preserved

The source increments its request budget for `ranged_get` calls, while its `HTTPRedirectHandler` subclass delegates to the standard redirect machinery. One logical range request followed a 302, giving **three physical GETs for two metered range operations**, beyond the authorized maximum of two. The cloud's separately observed stdlib inspection found `fp.read()` before redirect continuation. The ledger meters only the two final 206 bodies.

Those recorded 206 bodies total **163,286 bytes**: 65,557 and 97,729, with contiguous nonoverlapping ranges, matching final Content-Range/Content-Length and ETag fields. **The 302 body length, complete response-body total, and strict 1 MiB total-body guarantee remain UNKNOWN/unverified.** Process exit 0 and directory parsing do not cure either failure. The sum 7,532,853,358 bytes is the historical ceiling record plus the measured 206 increment, not a verified replacement conservative total. The old ledger/result are unaltered, and the later correction and admission-status record preserve the stricter interpretation.

Recounting metadata gives 1,463 entries, 116 named sequence containers, 696 expected image names and 580 expected H names, with 70 extra names across 68 containers and no missing expected names. This review reads names, offsets and declared sizes only; no image/H values, local headers or compressed members. The whole-archive hash, payload CRCs, rights beyond supplied context, scene independence and usable geometry remain unverified. There is no complete network syscall trace.

## Publication repairs and review limits

The first publication verifier's exit 1 and traceback are retained: its generic path scanner matched the scanner's own detection literals. The revised checker exited 0. Its archived result covered **63 files at that time**; this parent checker covers the final **69**, including the later-added verification and repair records. All saved publication argv/source/log digests reconcile with their declared originals.

The wrong-relative-path refresh diagnostic is also retained. Its aggregate shell exit was 0, but the individual failed command's exit was not captured; that remains null. The repair is documented as publication-only. **These publication events were reviewed from archived code, receipts and logs, not independently replayed against actual publication tool returns.** The parent could not obtain the complete publication tool chain. This review does not add stronger remote-publication or current-cache attestation.

This parent's first checker also failed: naive full placeholder reversal changed literals already present in publication source. [attempt1](attempt1/execution.json) preserves that checker, exit/log hashes and the safely redacted traceback. The corrected checker uses bounded, uniquely digest-matching occurrence reversal. This was a local review fix; no qualification source or result changed.

For an authorized reviewer who has the original private command retrievals, repeat with a fresh output file:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error \
  results/0009-admission/qualification-parent-review/check_qualification.py \
  --repo /path/to/checkout \
  --private-runtime /path/to/private-runtime-retrieval.json \
  --private-directory /path/to/private-directory-retrieval.json \
  --output /tmp/0009-qualification-parent-review-new.json
```

A public clone alone can inspect the archive hashes and safe observations but cannot repeat the private-retrieval/reverse-prefix checks without those inputs. No staged files, commit, push, successful science or data admission is claimed by this review.
