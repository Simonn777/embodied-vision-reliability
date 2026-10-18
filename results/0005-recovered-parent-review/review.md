# 0005 recovered native capability — independent saved-artifact review

**Disposition: no blocker found for archiving the completed eight-frame CPU capability run. This is not an efficacy or novelty result, and does not justify extending the old research topic.**

Reviewed clean repository HEAD `0436cb438666db9bd0645f131632fc2fafcdf025`, relative to execution base `f0787ab50f5dfc31f8b8a58bdcb165a12d2d83a6`. Only saved small outputs, JSON/logs, Git objects and previously cached source text were read. No cloud command, raw dataset, checkpoint, model import/forward or network call occurred. No repository files were changed.

## Independently reproduced

- All eight NPZs have valid ZIP CRC, exactly five expected arrays, exact shapes/dtypes, 33,088 finite float32 scalars and 64 boolean values. All 64 CSV rows exactly round-trip to saved positions, logits and booleans. Wrapper xy equals the raw model yx reversal exactly; visibility equals the saved logit `> 0` exactly.
- As a separate arithmetic diagnostic, a NumPy float64 implementation of the pinned native prediction-head formula (temperature 0.5, ±20-bin argmax window, normalized expectation, +0.5) reconstructs saved raw coordinates with maximum absolute difference **0.00001544924731433639 px**. This uses saved logits, not a model call; it neither changes a scientific threshold nor evaluates accuracy.
- Stored state assertions have steps 1–8, exactly 12 layers each, declared float32 CPU arrays of `[1032,768]` and `[1032,3,768]`, correct byte counts, finite flags and valid hashes. Query state shape `[1,8,3]` and its hash are stable; all 96 recorded recurrent-layer hashes differ. These checks concern saved assertions, not the absent tensor values.
- All **16 recovery frozen inputs** match working-tree, index, result HEAD, execution base and declared scientific-source Git bytes. The original seven-input freeze is unchanged. All **74 manifest-bound archived copies** match their declared size and SHA256; there are no unlisted payload files apart from the README and manifest itself.
- Independently hashed all **1,832 historical execution-base Git blobs** against the merged prior/incremental baseline. All **1,651 old result files** remain byte-identical locally. Exactly three pre-existing tracked files changed, as expected for status publication: `CURRENT_STATE.md`, `NEXT_ACTIONS.md`, `notes/2026-10-18.md`.
- Actual native external and supervisor child receipts both record exit 0/raw wait 0, consistent log hashes, nested timestamps, 4-core affinity `[0,1,2,3]`, 8 GiB AS, 1800 s timeout and peak RSS **3,727,996 KiB**. External time **55.082073132 s**, child **53.080107886 s**, adapter **50.420190969 s**. Forward sum **19.415824450 s**, frame-with-artifact sum **20.342805421 s**, phase sum **50.416812649 s**. The remaining adapter overhead is **0.003378320 s**; durations are correctly nested, not additive.
- Source/overlay manifests bind five upstream files and the same 378-file prior overlay; preparation receipts, import check, stage markers, frozen source/config hashes, checkpoint asset record and ledger snapshots cross-bind. The recorded first GET is 2,532,282,370 bytes, one attempt, HTTP 200 and no residual partial. Ledger arithmetic reproduces **7,499,066,782 bytes < 10 GiB**, retaining the historical ceiling and metadata allowance.
- The two review-only failures are preserved with their original exit 1/raw 256, code and matching logs; their corrected checkers exit 0. They are not concealed or represented as failed model forwards.

## Boundaries, already disclosed correctly

1. `results/0005-recovered-capability/README.md:79`–82 correctly says state tensors were not retained. Their numeric values, finite status and SHA preimages cannot be independently recomputed from this archive. The frozen runtime adapter checks actual tensors at `scripts/tapnext_native_adapter.py:275` onward; this review verifies its saved assertions and source binding.
2. The local parent audit cannot re-hash the remote checkpoint, 407-file failed cache, runtime or original cloud source paths. Their preserved-state claims are supported by cross-bound runtime receipts/manifests, not a fresh live-cloud observation by this reviewer. Actual published old-result bytes *were* independently verified.
3. `README.md:42`–43 correctly limits outer observation: the observer parent has no further durable parent wait4 receipt. The archived external timeout-command wait4 evidence is present and consistent.
4. `README.md:6`–7 and `native/run/receipt.json:2695` correctly restrict this result to short-prefix CPU capability. There is no measured tracking improvement, independent real-data transfer, long-clip validation or publishable-method claim.

## Reproduction

Run from a checkout of the reviewed HEAD, with Python + NumPy only:

```sh
python3 /tmp/evr-0005-recovered-independent/check_saved.py /Users/simonxie/Documents/Codex/2026-10-12/ban/work/embodied-vision-reliability
```

The independent script does not import or execute any repository audit or scientific script. Machine-readable output is `audit.json`; script SHA256 is recorded there. The exact reviewed checkout is required so later state edits do not silently enter the comparison. No all-suite tests were needed or run.
