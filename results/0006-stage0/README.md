# 0006 Stage 0: fixed numerical gate failed; one log discrepancy remains

The prospective **64-case development grid** was trained once and scored once on
October 18, 2026. Every fit completed 500 updates. The unchanged numerical rule
returns **`stop_this_candidate`**: mean colored gain against the predeclared
oracle-best baseline was **−0.570956 dB**, below +0.3 dB. All four image means were
negative and the pooled-gradient error ratio was **1.800684**, above 1.05. White
blindspot/safe/residual equivalence passed exactly. Stop this exact candidate and
configuration; do not tune its thresholds, seeds, images or update count.

There is also an **open provenance defect**: one saved stdout no longer matches
its execution receipt. It is retained, not repaired or counted as a passed check.
This is not an all-evidence-checks-pass result. No positive efficacy claim follows.

Execution main: `1edf58df657a3694a027f4b08b0cce76a9bfabfb`; source commit:
`ad3731de4f82f686da1a454bebeebcbfa4626dbb`; canonical freeze commit:
`7763fee2351bd4d58b30d1b5594c80b03ac7a0b5`, receipt SHA256
`1c71bbb85cbce43b054ace50502d949df2a599c2917e28053351e8896c3d9c96`.
All 13 required inputs matched current/index/HEAD/source Git bytes. Each frozen
entry also checked the canonical receipt. Publication follows execution and uses
a distinct commit. [Protocol](../../protocols/0006-conditional-residual-stage0.md).

## Scope and complete results

Four fixed PolyU mean JPEGs were used as natural-image **substrates for synthetic
Gaussian corruption**, not noiseless camera ground truth. This does not evaluate
real-camera-noise denoising, a full published R2R network, independent held-out
scenes, novelty or a paper contribution. All toy/plant/classroom/wall captures and
identified scene matches remain development forever. No final test is allocated.
[Source notices](../../docs/0006-data-license.txt) apply; no image is redistributed.

Each condition used two fixed seeds and four 49→64→32→1 pointwise MLP methods:
ordinary blindspot, covariance-safe blindspot, conditional residual, and R2R
alpha 0.5 with primary MC50 inference. All 16 image/condition/seed sets matched
initial-weight and minibatch-stream hashes, with 500 Adam updates / batch 512.
There was no early stopping, best-epoch selection, seed selection or retraining.
Fits received only noisy+C; reference arrays were used to generate corruption in
prepare and later by saved-array scorers, never by the fitting process.

| Mean PSNR across eight image/seed pairs (dB) | White | Colored |
| --- | ---: | ---: |
| Ordinary blindspot | 35.478321 | 20.321029 |
| Covariance-safe blindspot | 35.478321 | 32.033149 |
| Conditional residual | 35.478321 | 31.715140 |
| R2R alpha 0.5 / MC50 | 35.061745 | 31.746446 |

The primary comparator is the best baseline **per colored pair**, not whichever
single method has the highest average. It chose R2R four times and safe blindspot
four times, mean PSNR 32.286096 dB. It is an oracle diagnostic, not a deployable
selector. All eight paired comparisons and both seeds are in `control/summary.json`.

| Image substrate | Mean paired colored gain (dB) |
| --- | ---: |
| d1 toy | −0.548482 |
| d2 plant | −1.030441 |
| d3 classroom | −0.095342 |
| d4 wall | −0.609559 |

The mean-gain, at-least-three-positive-images, worst-image ≥−0.2 dB, and gradient
≤1.05 gates all failed. Candidate/baseline mean gradient MSE were respectively
0.00028894162045 / 0.00016046217291. Both white prediction-equality checks and
zero white PSNR harm passed. All 64 primary and 16 secondary direct-R2R scores,
including **5,120 block SSE/count records**, remain available; the direct-only
R2R output was never substituted for the primary MC50 comparator.

## Actual execution and independent checking

Cloud first ran `python -W error -m unittest discover ...`: **66 tests passed,
zero skips**, including all six Torch fixtures skipped in parent CI. Wall was
3.389159 seconds. Those small fixtures are software checks, not scientific fits.

| External command | Actual exit / raw wait | Wall seconds | Peak RSS KiB |
| --- | --- | ---: | ---: |
| Acquire four images | 0 / 0 | 17.021071 | 278,592 |
| Prepare once | 0 / 0 | 2.538986 | 306,564 |
| Group d1, 16 cases | 0 / 0 | 52.399633 | 300,160 |
| Group d2, 16 cases | 0 / 0 | 49.774837 | 300,384 |
| Group d3, 16 cases | 0 / 0 | 51.006417 | 300,428 |
| Group d4, 16 cases | 0 / 0 | 51.356767 | 300,280 |
| Frozen scorer, one invocation | 0 / 0 | 0.374904 | 30,304 |

Group execution UTC span: 19:20:13.017714–19:24:26.515945. Sum of group command
walls was **204.537654 seconds**; individual case walls sum to 195.159928 seconds,
and fit intervals sum to 43.987797 seconds. These are nested intervals, not
additive costs. Every case has its own original wait4 record and 120-second
supervisor; no fit hit 90 seconds, no group required time-budget continuation.
Commands used four CPUs/numeric threads, 8 GiB AS and external timeout 1800 seconds
with 5-second kill grace. Torch intra/inter-op counts were 4/1. Host: AMD EPYC
9V74, Linux x86_64, Python 3.12.14, cached Torch 2.10.0+cpu / NumPy 2.0.2 /
Pillow 13.0.0. No package install, GPU, API, model download or added fee occurred.

Saved-output integrity confirmed all predictions/weights/input hashes, complete
64-case identities, finite values, 500 updates, matched streams and eight exact
white-control sets. No fit was repeated. The frozen scorer ran on saved arrays.
A separate NumPy/stdlib implementation used Python float64 differences and
`math.fsum`, importing no trainer or Torch. It reproduced **5,615 numerical
comparisons**, all oracle choices and gates; max absolute difference
**2.842171e−14**, within the declared 1e−12 absolute/relative tolerance.

## Retained stdout discrepancy

The strict pre-score reviewer exited **1 / raw 256** on the stdout hash for
`d3-colored-0-blindspot`. Current stdout contains its 862-byte startup line but
lacks the 144-byte completion line. Hashing the current prefix plus the frozen
completion recipe **in memory** exactly matches the original recorded digest.
That is a diagnostic compatibility check, not restoration of original bytes or
proof of a filesystem cause. The original log/receipt were not rewritten.
The other 127 case stream hashes match. The case itself has actual exit 0 and
matching completed receipt, weights, prediction and input hashes.

A separately retained postfailure inventory verified the required scientific
artifacts and white controls while explicitly retaining `BLOCKED_LOG_PROVENANCE`;
its exit 0 means recording completed, not that the strict failed check passed.
Numerical scoring then proceeded once on these saved arrays; the numerical
failure and the unresolved evidence defect are reported separately. The
pre-score inventory's `not_started` field is a historical timestamped observation.
Independent parent review must assess the log gap. No new fit, repaired log,
relaxed scientific gate or modified frozen source was used to proceed.

## Archive, cost and next action

Exactly four fixed GETs acquired **68,858 bytes**. Historical ceiling
7,499,066,782 + 32 MiB metadata allowance + payload = **7,532,690,072 bytes**;
pre-GET planning separately allowed four sentinel bytes. No historical acquisition
is refunded by missing caches. No other image or source was acquired.

`scores.compact.json` retains the entire frozen scorer object, including all
block measurements, receipts, timings and secondary results. Compaction changes
whitespace only; `copy-manifest.json` records its original SHA/length and a
verified byte-exact reconstruction recipe. Original indented scores stay in
scratch. The manifest binds **380** small evidence files. Raw case receipts,
markers, logs, control/checker source and the failed strict review are included.
All image/reference/noisy/full-prediction arrays and weights stay in the separate
cloud cache. No reconstructable pixels or trained parameters are published.

All **1,741 previous result files**, old markers and 13 frozen inputs remain
unchanged; notes are appended. 0003 remains deferred and 0004 stopped. These
AI-assisted development measurements justify stopping this exact configuration,
not claiming a general theorem or camera-noise outcome. Next: independent review
of the evidence and log discrepancy, then a separately justified hypothesis or
new prospective design if authorized. No old-test tuning or manuscript follows.
