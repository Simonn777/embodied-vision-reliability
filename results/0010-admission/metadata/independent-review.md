# Independent review: 0010 retained metadata

**Decision: PASS for retained metadata identity and accounting, with two publication corrections.** This is not image capability, dataset admission, geographic independence, or a scientific result. The review performed 156 stdlib checks on existing local files, with no network, acquired-source imports, images, weights, model construction, scientific run, or repository edit.

## Confirmed

- TorchGeo EuroSAT pin `f9fa29ffb3980d9959c3d177dd2bc7c71ffeddcc`: all three text bodies match their fixed Git blob identities, sizes and subsequent SHA256 records. Train/validation/test have 16,200/5,400/5,400 unique names; their pairwise intersections are empty and union is exactly 27,000.
- Independent SHA256 ordering of `UTF8("0010-clip-capability-v1|" + basename)`, then basename, selects the same first eight train names: River_362, AnnualCrop_1607, Pasture_919, Residential_1561, Industrial_1563, AnnualCrop_1012, Highway_1291 and River_2042 (all `.jpg`). No class-based sampling or outcome selection appears in this rule. Selection file SHA256 is `75e31cc9742f6def800019dfd66009abb5294a9eafa7b8ef2d5130501617375c`.
- The metadata ledger and 33 event rows reconcile: six recorded HTTP requests, one 307 and one 200 per file; 522,430 final-body bytes plus 933 redirect-body bytes equals 523,363 bytes, below the fixed 1,048,576-byte cap. Every read's per-request and cumulative counts agree. Each completed final file has the claimed digest; no corresponding `.partial` remains. Manifest semantic hash agrees with the exclusive-start marker and ledger. Saved stdout/stderr match the execution receipt, which reports exit 0 and no timeout.
- The retained acquisition utility matches the pre-access plan's SHA256 `412c6cddba8fcd32241026611edb627c76690bf8dcc93195d6360a4ef24b2d75`. Static inspection confirms explicit redirect handling, bounded response reads, counted non-200 bodies, verified final-byte identity and no automatic retry loop.
- CLIP pin `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` and tree `8ff3473dab6037c117401572413d3eb9b30ca23d` agree with retained commit/tree responses. Five source/license bodies match both SHA256 and Git blob/size entries. Four source/tree GET records total 28,255 bytes; five cache-reuse records match preserved bodies. BPE remains tree metadata only: its body is absent and unverified. No model feasibility conclusion follows.

## Publication corrections

1. `command.json[0]` contains the user's private home-directory path. Do not publish this raw command. Publish a clearly identified redacted derivative and record original/public byte lengths and hashes; keep the private original unchanged. Original command: 457 bytes, SHA256 `bc2f400f8c9bf0c910ed42d54a83e06320aa62527505b9e70c3b83167b39d263`.
2. `plan-before-access.json.scope` says “no ... labels”. Filename metadata contains class names, including validation/test filenames. Preserve that original record and append a correction: **class-bearing filename metadata was read; no image pixels, image predictions, accuracy evaluation or outcome-based sample selection occurred in this metadata qualification**. Do not describe it as literally label-free.

## Evidence boundaries

This review recomputes file identities and accounting from retained evidence. It did not directly observe the original process exit or network traffic. Redirect-body content was not saved, so only its recorded counts can be reconciled. The preserved utility matches the planned SHA, but the command and execution receipt do not themselves attest the loaded source SHA at execution time. Future frozen invocations should verify the source/config digests immediately before launch and retain that check; this is not a reason to redownload these metadata.

Filename disjointness is not proof of physical-scene or geographic independence. ZIP membership, pixel shape/content, CPU model feasibility and predictive quality remain untested. Historical cumulative HTTP transfer remains unknown; these two bounded ledgers do not restore it. The 523,363-byte figure covers only spatial-list acquisition; CLIP source acquisition is separately 28,255 bytes and should not disappear from prospective bookkeeping.

Machine facts: `/tmp/evr-0010-metadata-independent-review.json`. AI-assisted independent review; no study outcome or novelty claim.
