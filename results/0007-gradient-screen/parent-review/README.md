# Independent parent review of 0007

**Pass for publication of the local gradient-variance screen only.** Two independent
read-only audits reviewed cloud publication `5dc6839d272f48bb57ce4a2f02ccf7aa8233478c`.
Neither audit retrained a model, fetched raw gradients, rescored private arrays,
acquired assets or changed a frozen scientific input.

- [Public metric review](metric-review.md): 36,843 checks, including 2,055 numeric
  comparisons; zero failures, maximum difference 4.44e-16. All 4,096 timing records,
  64 variance summaries, 576 projected-mean threshold checks and gate arithmetic agree.
- [Provenance review](provenance-review.md): 31,182 assertions; the original 108-file
  archive, all eight frozen files and 2,164 prior result/protocol Git blobs agree.
  This audits the original cloud commit, separately from the parent additions here.
- Both checks are portable standard-library scripts. Their archived outputs record
  the original invocations; generate any future audit into a new path. No scientific
  operation is rerun by these commands.

```sh
python3 results/0007-gradient-screen/parent-review/metric_check.py \
  --results results/0007-gradient-screen --report /tmp/new-0007-metric-audit.json
python3 results/0007-gradient-screen/parent-review/provenance_check.py \
  --repo . \
  --prior-inventory results/0007-gradient-screen/parent-review/prior-public-inventory.json \
  --output /tmp/new-0007-provenance-audit.json
```

The 7/8 B best-control wins include one loss to random pairing. A same-patch
pairing has lower variance in all eight cases under a different fixed-half-batch
objective. B step-time ratio 1.08819 is close to its 1.10 limit and is one short
observed run, not established training efficiency. All these limits remain.

**Observation boundary:** the cloud saved-gradient checker reports a numerical
replay; these parent audits did not repeat that replay locally. Later cloud tool
returns could not be independently fetched due to an oversized turn. Their process
exits are supported by archived wait4/log/command chains. Parent directly observed
only the first four source/preflight calls; private-cache preservation is a bound
cloud assertion. These are not claims that every outer tool status was witnessed.

The final copy manifest's 79 byte-exact / 21 redacted entries follows an earlier
74 / 20 publication-review snapshot by six archival entries. Both records are
retained as written. New parent files and the derived figure are separate additions,
listed in `additions-sha256.json`; no old raw artifact is silently repaired.
The figure is generated solely from frozen score summaries with `plot_ratios.py`.

The recorded pass warrants one separately frozen training-quality screen. There
is still no PSNR improvement, faster optimization, independent-scene validation,
novelty proof or paper contribution. The old 0006 failure/log defect remains open.
