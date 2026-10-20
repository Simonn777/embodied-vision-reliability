# 0035 acquisition feasibility and prospective adapter

2026-10-20. Source/cached-metadata review only. **No allocation, hold, HTTP, payload decoding, or scientific execution.** The four complete clips remain a proposal. The parent's subsequent 180,000,000-byte hold and phase limits are prospective preparation instructions, not an activated ledger entry.

## Preserved v1 and corrected v2

`build_manifest.py`, `proposed-manifest.json`, and `manifest-build-report.json` remain byte-identical at hashes `d42448b5382df8fe62c1703c80046701416bbd6462ecc743c917ee787552e925`, `fc0bbbd85cd66878426956dc1c382826e56dbffd7e92a240b13d6f09d5dc03f1`, and `296d32d0ca46903f96627ae0a233915e3f99b29f48fadf7343f86cefe2bdc364`.

The separately executed metadata-only `build_manifest_v2.py` reproduces v1 exactly before correction, then writes `proposed-manifest-v2.json`. V2 links the original three hashes and the saved source review. Cohort, all 892 descriptors, 446 complete frame stems, metadata bindings and byte totals are unchanged.

Correction: paper wording about ambiguous regions did not establish released-script behavior. At author commit `fe274574cb03c8a3ea83e121dd76e20b703781fd`, the [dataset loader](https://github.com/TRI-ML/VOST/blob/fe274574cb03c8a3ea83e121dd76e20b703781fd/evaluation/source/dataset.py#L80-L93) maps 255 to 0, and the [caller](https://github.com/TRI-ML/VOST/blob/fe274574cb03c8a3ea83e121dd76e20b703781fd/evaluation/source/evaluation.py#L50-L69) supplies `void=None`. Thus pinned-author primary J/Jtr treats those pixels as background. V2 records this source-derived prospective contract, not an executed metric or conclusion about author intent. The peer/parent inspected the pinned caller; this continuation read their saved review and the cached loader, with no new network. Earlier v1 and feasibility-note claims of ignored 255 are superseded on this point only.

## Minimal reusable pieces

Use immutable `experiments/0023-mapfree/range_inputs.py`, SHA `519a9eb08fab5a8ad8d9a9257927281093abac3354fd14cadd8e7599fa69ea50`:

- `Index`/`Limits` (lines 118/176), `NoRedirect` (193), deadline and `RangeReader` (227) are transport primitives usable with an explicitly constructed VOST index. Python integer ranges retain 64-bit offsets.
- The reader sends `Range`, `If-Match`, and identity encoding, requires exact 206/ETag/Content-Range/Content-Length, rejects duplicate identity headers and redirects, preserves partial-byte accounting and per-request timing, and never retries. HTTP error bodies are closed without reading. Each accepted request attempts at most expected-length plus one byte.
- `extra_fields` (39) rejects duplicate/truncated extras. Reuse this for bounded local ZIP metadata.

Use `experiments/0029-memory-information/acquire_davis.py`, SHA `4e2897ee0b1dfea25b3388805f019e155be53986d155f71c369101293919ee69`, as a **source pattern** for source binding, fresh outputs, phase receipts, exception preservation, postchecks, and the closed metric callback (lines 74, 206, 235). Do not import its execution policy as VOST admission.

Neither existing `load_plan`/`load_index` nor extractor can run on VOST unchanged. The old code requires version 20, flags 0 and exact local-header CRC/sizes. VOST selected central records require version 45, flags 8 and ZIP64 local offsets; local flag-8 CRC/sizes may be placeholders. The old 16 MiB metadata cap also rejects the 21,042,989-byte central directory; the 47,025,285-byte parsed inventory exceeds the 0023 32 MiB cap. The old DAVIS cohort, URL, 64 MiB output cap, phase counts and 60 MB reservation are inapplicable.

V2 delegates selected central parsing to the preserved v1 builder's `central_entry`. It verifies each selected record against the exact central bytes and resolves ZIP64 sentinel fields in order. A future adapter must additionally check regular-file attributes, timestamp/version/method identity, and bounded local-header structure; the builder's minimum-extent check is **not** full local-header qualification.

## Flag-8 extraction contract

Before any request, bind executing code, immutable helper/parser bytes, manifest, admission and cached metadata to an exact source commit and manifest hash. A new adapter can issue exactly three ranges per selected member: fixed 30-byte local header; bounded filename/extra fields; central-authoritative compressed payload.

Validate literal name, version 45, flags 8, DEFLATE method and timestamps against central metadata. Allow local CRC only zero or the central CRC. Local 32-bit sizes may be zero, the correct central size, or the ZIP64 sentinel; parse bounded ZIP64 size extras where required and reject contradictory values. Do not treat arbitrary nonzero values as placeholders. Reject encryption, unsupported methods/flags, links, multiple disks, malformed/duplicate extras, and a local/header/payload span beyond the recorded next physical header. No filename-driven automatic extraction.

Read exactly the central compressed length; bounded raw-DEFLATE decompression must produce the exact central uncompressed length, reach EOF with no unused/unconsumed stream bytes, and match central CRC. Hash both compressed and extracted bytes and the local header. A trailing data descriptor need not be fetched: central sizes/CRC define the extraction, while the next-header bound prevents overlaps. Member SHA is locally observed, not a publisher-provided digest.

## Prospective costs and unresolved admission

Successful response-body bounds below use a 4,096-byte local-extra cap. The reused reader conservatively charges one sentinel byte per request, so requested-upper accounting adds three bytes per member. Neither column is an actual transfer or reservation.

| Phase | Members / requests | Successful body bound B | Requested-upper bound B | Exact uncompressed output B |
|---|---:|---:|---:|---:|
| initial | 4 / 12 | 24,446 | 24,458 | 18,190 |
| RGB | 446 / 1,338 | 167,329,910 | 167,331,248 | 165,784,441 |
| metric | 442 / 1,326 | 4,597,141 | 4,598,467 | 3,626,472 |
| total | 892 / 2,676 | **171,951,497** | **171,954,173** | **169,429,103** |

Largest selected compressed/uncompressed member is 593,941/593,962 B; 8 MiB/member and 200 MiB output caps are sufficient without decoding. Proposed initial/RGB/metric caps of 50,000/170,000,000/5,000,000 B sum to 175,050,000 B, below the proposed 180,000,000 B hold. No HEAD is needed if every range binds the cached archive identity with `If-Match` and exact response metadata. The archive is 54,012,104,924 B, not a 32-bit ZIP; its cached ZIP64 end records and central hash are already bound.

The future adapter must use all four initial masks first, then all 446 RGB frames. Metric defaults closed until an explicit all-four clean prediction proof exists. All target IDs, resolution and object-count cost remain unknown. Prospective caps are 1,800 s/phase, 30 s/request, no retry, no redirect and no full-archive fallback; this document activates none of them. Source tests must cover flag-8 placeholders/ZIP64 extras, contradictory fields, physical overlap, bounded decompression, CRC failure, receipt preservation and the closed metric barrier before transport. Failed bytes/costs cannot be erased by a fresh output path.

The v2 manifest deliberately retains null allocation, error/retry/request policy fields: these require the later frozen admission. The extra sentinel accounting is informational and does not silently create an allowance. No ledger or repository file was changed.
