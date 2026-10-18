# Independent parent review of 0006

Reviewed cloud result commit `dabd5b98bb1bc0b8207a7bc491504dbfdc8c378c`.
The fixed numerical decision is **stop_this_candidate**. This review does not
convert the failed screen into a positive result or close the log defect.

The separate [provenance audit](provenance-review.md) verifies all 13 frozen
inputs, 380 original copy-manifest entries, 64 case bindings and 1,741 preserved
historical result files. Its portable local checker returned exit 0 in 0.747 s;
the reported open stdout defect remains explicit in `provenance.json`. That
successful audit invocation means the stated checks and exception were recorded,
not that the original strict grid check passed.

## Independent arithmetic and execution observations

The parent ran `audit_public_metrics.py` on `../scores.compact.json` using Python's
standard library, without importing the experiment's scorer or trainer. It
reconstructed each primary SSE from the 64 rectangular block measurements,
checked complete/nonoverlapping coverage, recomputed PSNR and gradient summaries,
and independently selected the strongest baseline in each of the eight colored
pairs. The complete 64-case grid and all seven numerical gates agree. Mean gain
is **−0.5709558643 dB**; all four image means are negative. White prediction
identities agree through the published array digests.

`arithmetic.json` binds the reviewed score bytes and independent checker source.
It verifies public arithmetic, **not a new reconstruction of the private pixels
or predictions**. The cloud's separate saved-array check remains the evidence
for its 5,615 numerical comparisons; the parent did not rerun model inference.

The parent also read eleven actual cloud command-execution records. Tests,
acquisition, preparation, four groups, scoring and the postfailure/metric
inventories returned 0; the strict grid reviewer returned **1**, as disclosed.
`tool-observations.json` retains those exits and envelope durations. Envelope
duration includes work outside inner timers and is not an extra training cost.

## Log discrepancy remains open

The `d3-colored-0-blindspot` stdout bytes fail the original recorded digest. An
in-memory candidate suffix matching that digest is a compatibility observation,
not recovery of the original log or proof of the cause. Neither rerunning the
fit nor rewriting the original receipt/log is warranted. Negative numerical
evidence can be reported with this defect prominently retained; this archive
must not be described as passing all provenance checks.

This review is AI-assisted. The screen uses four development image substrates
with synthetic corruption, not native camera noise or untouched evaluation
scenes. No manuscript or novel-method success follows.
