# 0030 gradient probe run01 — saved-only review

2026-10-20. **PASS for the recorded manufactured API/resource qualification; no efficacy claim.** This review made 51 checks with zero failures using saved JSON/text, file hashes and `git show` of the frozen source. It did not import the runtime/probe, run a model, read image/mask/NPZ contents, or recompute a gradient. Numerical gradients, exact tensor parity and runtime-inventory checks remain the executed probe's recorded assertions, not an independent tensor calculation. Detail is in `saved-review-checks.json`.

## Binding and completion

- Source commit `ed00b1c153c1dda663bdfd44c8b27e24b1a17dc5`; reviewed probe SHA256 `5d8414398a715c3d8e5bd81f4c5fb1daa6490142a5b378f1a0c271daf0bed6dd`. All seven recorded source hashes match current bytes and the named Git commit, including supervisor/monitor. The private prepared probe is byte-identical. Original native qualification commit remains `37b54982bcc3b22b56a5817b6881f22489eba8d5`, correctly distinct from the new source commit. The recorded native provenance file map was rehashed without loading checkpoint/runtime binaries.
- Saved [CI 39398931791](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39398931791) names that exact commit, completed successfully at 09:13:38 UTC. Local filesystem evidence places saved CI metadata before the supervision directory's creation at 09:15:28.840637 UTC. This is saved-metadata chronology, not a new remote CI query or a cryptographic time attestation.
- The outer spec hash, exact redacted command and source/commit flags, inner-result hash, stdout/stderr hashes, and fixed output paths match. Qualified CPython runs with `-S -B -W error`; four thread environment values and the fixed 1,800-second / 8-GiB limits agree.
- Inner: `completed`, `eligible=true`, `manufactured_only=true`, `scientific_input_access=false`, **20.918290625 s**. Outer: `completed`, exit0, **21.274490334 s**, peak sampled summed RSS **2,335,824 KiB = 2.227615 GiB**. No termination, signals, cleanup errors or remaining descendants. Fresh post-exit and final samples are empty. All eleven inner stages completed.
- Warnings, forbidden attempts and compatibility-probe lists are empty. Stdout is exactly the inner completion summary; stderr contains only native frame-loading/propagation progress.

## What the saved numerical record establishes

Eight baseline outputs (frames0–7), eight image encoder calls, and four replays of frame7 are recorded. Every replay has seven spatial memories `[0,1,2,3,4,5,6]`, seven object pointers `[0,6,5,4,3,2,1]`, 28 pointer tokens and 28,700 total tokens. No extra encoded scientific image exists in this check.

The zero/BF16 replay is recorded exact. Grad-enabled selected logits have max absolute/relative difference0 and zero binary differences; raw decoder masks, quality and presence also match exactly on all three zero-update calls. This is stronger than the predeclared tolerance, not a reason to tighten a later test after seeing the result. All calls report native object present, so this fixture did **not** exercise the hard-absent branch. The executed objective is still the explicitly declared pre-presence raw-mask BCE plus presence BCE surrogate. BF16 backward remains PyTorch's surrogate derivative.

The spatial-memory leaf has 262,144 nonzero gradient entries, L2 `8.306096193742871e-05`. The 4,160 head parameters have nonzero gradients and the head hash changes after its single fixed SGD step. Coefficient-change L2 is `1.3131227902376019e-08`; resulting residual L2 is `1.518764669583561e-08`. **BF16 stored entries changed:0. Future raw/native logits, binary mask and surrogate loss changed:0.** Thus an optimizer update happened but no forward-effective memory correction was demonstrated. No learning-rate adjustment, second attempt or gain claim follows from this qualification.

Recorded before/after hashes of model state_dict, complete native state, and ordinary cache/history—including pointers—are equal. All recorded unchanged-source/runtime checks pass. This review independently compared those saved hash values and source bindings; it did not reopen tensor payloads or rehash the whole installed runtime.

## Bounded cost planning, not a guarantee

Measured isolated stages: leaf forward **0.639402083 s**, backward **1.803208375 s**, sum **2.442610458 s per one fixed future-context loss/gradient**. The head's first forward stage is1.909376 s and includes constructing its optimizer; it is not a clean steady-state forward benchmark. Native load/imports cost1.318511 s; source/native admission1.818951 s; final source/runtime checks2.908141 s. Their combined one-time overhead is approximately6.046 s. Baseline construction including eight image forwards/history cost7.308090 s.

For **one parent recording**, at most16 anchors ×6 teacher steps ×2 fixed future contexts =192 context forward/backward evaluations. At the observed leaf rate this is **468.98 s (7.82 min)** before baseline/cache construction, serialization, target constraints, optimizer overhead and scheduling variation. Four parents total approximately1,875.92 s of these evaluations alone; they must be four bounded complete operations, not one claim of fitting all four inside1,800 seconds.

A transparent provisional formula is `T_parent ≈ 468.98 + 6.05 + 0.914 N + I/O/teacher overhead` seconds, where N is that parent's full frame count. The0.914 factor is only the crude7.308/8 measurement, not a full-video measurement. As an illustration, N=100 gives about566 s before omitted overhead; even a twofold allowance would remain below1,800 s. This makes the proposed **per-parent** cap plausible, not established. Use the actual full parent count, retain all costs/failures and stop the complete operation at its unchanged cap.

This estimate assumes each future context is processed sequentially, its graph freed after backward, and the two context gradients accumulated for one declared teacher update. It assumes frozen-image caches and fixed baseline future histories, not differentiating a new recurrent rollout through intervening frames. Full rollout differentiation, simultaneous two-context graphs, more objects, all-sequence feature residency, a different teacher parameterization, or materially longer history needs a new resource estimate. The observed2.23-GiB peak applies to this one context; it does not justify multiplying graphs in memory. The real training schedule, teacher bound and BF16-effective update policy remain prospective development choices, with the same-context direct-loss control and complete costs retained.

## Saved-file SHA256

| File | SHA256 |
|---|---|
| `spec-01.json` | `1f176664d60a8c9c69f8694c7126d3d75580b7f091d9e43b81f632d89317261a` |
| `source-ci.json` | `63e6d140de80619518acc0558c8d0cc6e406e00dae185c6648d1c1de47b4e687` |
| `run-01/receipt.json` | `4d47aad1c9b2f522f87f7c2591e3047ba686c9d10fb6d07916156c0846e7fa19` |
| `supervision-01/receipt.json` | `345a19c451cdbcc333add35a40e690dd123965af1a7885020a37722785b926e1` |
| `supervision-01/stdout.log` | `66aedefcc9aeeb9c6fc646d76822461a892f459193849d4eefa137cb3a5851c7` |
| `supervision-01/stderr.log` | `e4e4d1ba507c325308e3b49e9e01b8edfd45bcaef7a37bb761906d59ae1988ac` |
