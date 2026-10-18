# Embodied Vision Reliability

**When should a vision system stop trusting its own position estimate?**

An ongoing research workspace on visual localization under occlusion and appearance changes. The first question is whether causal temporal consistency can help identify large errors that single-image confidence misses. We keep the position estimator fixed so that the comparison is about recognizing failures.

The program runs from October 2026 to October 2027. The first twelve weeks focus on selective prediction; later work depends on what these experiments show. This is AI-assisted research in progress, with code, protocols, raw measurements, and negative results kept together. [Contribution and evidence notes](docs/CONTRIBUTIONS.md).

## Start here

| Artifact | What it answers |
|---|---|
| [Current state](CURRENT_STATE.md) | What has actually been completed? |
| [Conversion evidence and executable geometry input](docs/0003-source-history-and-geometry-plan-2026-10-17.md) | What does the one saved-R compatibility result establish, and what complete prospective input awaits review? |
| [Cached conversion history and next candidate](docs/0003-source-history-stageA-2026-10-17.md) | How strong is the float16 explanation, what source links are missing, and what descriptive check is proposed? |
| [Frame 0 rotation failure: recorded diagnostic](docs/0003-rotation-diagnostic-stageB-2026-10-17.md) | Which exact stored camera defects trigger the unchanged gate, and what can precision evidence establish? |
| [Frozen Fortran audit: reader complete, geometry stopped](docs/0003-fortran-stageB-2026-10-17.md) | What did the one cache-only run access, and why did its first camera check stop? |
| [Motion versus drift: frozen design and data audit](docs/0003-review-and-readiness.md) | What can the next study distinguish, what is the improvement target, and why is its independent cohort still blocked? |
| [PointOdyssey cached sample audit](docs/0003-cached-resource-audit-2026-10-17.md) | What did the cloud actually download and inspect, and where did the bounded reader stop? |
| [Post-hoc motion-residual diagnosis](docs/analysis/0002-motion-residual-posthoc.md) | Which exact decision exchanges explain the negative pilot result? |
| [Frozen v2 evaluation](docs/0002-frozen-evaluation-v2-2026-10-17.md) | Did the extra motion residual meet the practical benefit criterion? No; all results and independent checks are retained. |
| [All frozen predictions: 30/30](docs/0002-prediction-collection-complete-2026-10-17.md) | Where are 10 calibration and 20 evaluation raw prediction sets, before scoring? |
| [Calibration collection: 10/10](docs/0002-calibration-complete-2026-10-17.md) | Where are all ten raw prediction sets, and what remains before evaluation? |
| [Calibration group: g-020](docs/0002-g020-calibration-inference-2026-10-17.md) | What ran next, with raw predictions and explicit child/parent exit evidence? |
| [Calibration group: g-039](docs/0002-g039-calibration-inference-2026-10-17.md) | What ran next, with complete predictions and supervised, cycle-safe verification records? |
| [Calibration group: g-040](docs/0002-g040-calibration-inference-2026-10-17.md) | What ran next, with raw predictions and parent-observed inference and verification exits? |
| [Calibration group: g-022](docs/0002-g022-calibration-inference-2026-10-17.md) | What ran next, with complete raw outputs, separate parent/child exits and independent provenance checks? |
| [Calibration group: g-013](docs/0002-g013-calibration-inference-2026-10-17.md) | What ran next, with independently observed wrapper exit and complete raw/provenance checks? |
| [First calibration group: g-033](docs/0002-g033-calibration-inference-2026-10-17.md) | What did the frozen model produce, and how was a dependency failure preserved and resolved? |
| [64-frame result and exploratory freeze](docs/0002-64frame-and-freeze.md) | What ran on cloud CPU, how are videos grouped, and what is frozen for later execution? |
| [Real-model CPU feasibility](docs/0002-cpu-feasibility.md) | Can the frozen Online BootsTAPIR model run within the cloud CPU budget? |
| [Research charter](docs/RESEARCH_CHARTER.md) | What is the question, and what would count as progress? |
| [Reading list](docs/LITERATURE.md) | Which primary papers inform the design, and how closely have they been read? |
| [Next actions](NEXT_ACTIONS.md) | What experiment or verification comes next? |
| [Cloud execution](docs/CLOUD_EXECUTION.md) | How is daily work validated and published, and is the hosted schedule active? |

## First result · October 16, 2026

A stable wrong track can still look reliable. In a deliberately constructed appearance-change case, both image-only and temporal scores continued accepting many wrong positions. Adding a motion residual did not resolve the identity switch. Under occlusion, it reduced accepted coverage without reducing the already-zero observed error among accepted predictions.

The diagnostic uses 40 held-out base sequences in four conditions, with a fixed color-based estimator. This is a reproducible failure analysis, not a learned-model or robot benchmark. [Read the result and limitations](notes/2026-10-16.md) · [Raw measurements](results/0001-pilot/summary.json) · [Protocol](protocols/0001-synthetic-localization.md).

The [follow-up failure analysis](docs/analysis/0001-failure-mechanism.md) explains the mechanism: two wrong positions fill the temporal history, making the next wrong position look consistent. After the transition, the temporal gate accepts all 880 wrong predictions in the saved test set. It also rejects correct predictions at a sudden turn and just after occlusion clears. The analysis preserves the original scores and thresholds and can be reproduced directly from the saved CSV.

![Synthetic diagnostic: risk–coverage curves and frozen operating points](results/0001-pilot/figure.png)

## Reproduce the first diagnostic

On macOS or Linux with Python 3.9 or later:

```bash
git clone https://github.com/Simonn777/embodied-vision-reliability.git
cd embodied-vision-reliability
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/run_experiment.py --output scratch/reproduce-0001 --protocol protocols/0001-synthetic-localization.md
```

The runner requires a committed, unchanged protocol, writes into a new output directory, and stops after 30 minutes. It records raw predictions, configuration, source hashes, environment, and measurements. Add `--smoke` and choose another output directory for a smaller run using separate sequence IDs. A repeated run checks reproducibility; it is not an independent research finding.

The [frozen protocol](protocols/0001-synthetic-localization.md) specifies the generator, predictor, scores, partitions, and metrics. No model weights or dataset downloads are needed for this diagnostic.

## Scope

**Current stage:** experiment 0003 has a frozen design and a verified PointOdyssey sample archive. The complete archive inventory contains 14,415 members and two named annotation parents. Their independence is unverified; the sample is reserved for development. The bounded annotation reader stopped on unsupported Fortran-order arrays before interpreting numeric ground truth, so no geometry or model improvement result is claimed. The [review and readiness guide](docs/0003-review-and-readiness.md) connects the protocol, its prospective addenda, actual evidence and remaining gates.

The following paragraphs retain the earlier experiment 0002 milestones; its complete prediction collection and negative frozen evaluation are reported at the end.

Online BootsTAPIR has completed the 64-frame development check and the first frozen calibration group, g-033. The latter produced 128 rows for two points on cloud CPU, with a 49.29-second inference subprocess and a 2.435 GiB peak RSS. A NumPy version mismatch stopped the first attempt before inference; its failure record is preserved alongside the exact-version environment reconstruction and completed run. [Read the measurements and raw outputs](docs/0002-g033-calibration-inference-2026-10-17.md).

The next group, g-013, has also completed once: 128 rows, 49.411 seconds for the inference subprocess, and 2.435 GiB peak RSS. A separate parent process recorded successful completion of the full wrapper. The run reused cached dependencies and data, with no downloads or frozen-input changes. [g-013 evidence](docs/0002-g013-calibration-inference-2026-10-17.md).

Group g-022 has now completed once: 128 rows, 57.829 seconds for the inference subprocess, and 2.435 GiB peak RSS. Both the inference child exit and the parent-observed wrapper exit were 0. No download, model replay or frozen-input change occurred. [g-022 evidence](docs/0002-g022-calibration-inference-2026-10-17.md).

Group g-040 completed once: 128 rows, 47.834 seconds for the inference child and 2,553,160 KiB peak RSS. Separate parent records cover wrapper and post-run verification exits. No downloads or frozen changes. [g-040 evidence](docs/0002-g040-calibration-inference-2026-10-17.md).

Group g-039 completed once: 128 rows, 48.157 seconds for the inference child and 2,553,996 KiB peak RSS. Parent records retain inference and post-run exits plus source/log bindings; no downloads or frozen changes. [g-039 evidence](docs/0002-g039-calibration-inference-2026-10-17.md).

Group g-020 completed once: 128 rows, 54.620 seconds for the inference child and 2,553,192 KiB peak RSS. Raw results and distinct child/wrapper/post-run records are retained; no downloads or frozen changes. [g-020 evidence](docs/0002-g020-calibration-inference-2026-10-17.md).

The final four calibration groups completed serially without replaying the previous six. Their receipts and the [complete calibration index](docs/0002-calibration-complete-2026-10-17.md) retain measurements and raw hashes. All 20 evaluation-group predictions have now been collected serially; see the [complete collection report](docs/0002-prediction-collection-complete-2026-10-17.md). The first frozen evaluation and calibration fit are now complete; the added residual did not meet the practical benefit criterion.

The allocation remains 1 development, 10 calibration, 20 evaluation and 19 reserve videos. Calibration prediction collection is **10/10 complete**; evaluation prediction collection is **20/20 complete**. The frozen comparison has now run once: augmented minus native matched risk is +0.169683 pp (95% CI -0.157781 to +0.712631 pp); practical signal false. See the [full result](docs/0002-frozen-evaluation-v2-2026-10-17.md). The [development report and frozen v2 protocol](docs/0002-64frame-and-freeze.md) describe the experiment and grouping. Shared objects and camera, unknown scene dependencies and unresolved checkpoint exposure limit interpretation.

The first experiment is a synthetic 2D localization diagnostic. Its purpose is to validate the evaluation machinery and expose failure cases. It does not measure robot control, prove generalization to real video, or establish a new method. Public-video evaluation and stronger frozen estimators are the next stages.

The initial implementation runs on CPU. Apple MPS is available on the development machine; CUDA is unavailable, and no NVIDIA GPU measurements are reported. No paid compute is configured.

## Research practice

Latest execution check: [October 18 cache blocker](docs/0003-cache-blocker-2026-10-18.md).
The 45-input frozen-source gate passes, but required data and historical markers
are unavailable in the current filesystem. The descriptive operation was not run;
no new geometry or model result is claimed.

- Keep full sequences in separate development, calibration, and evaluation partitions.
- Fix the protocol before inspecting the final comparison and preserve all conditions.
- Report error together with the fraction of predictions accepted; estimate uncertainty across sequences.
- Link claims to executable runs and distinguish hypotheses from measurements.

Original code and generated diagnostic assets: [MIT](LICENSE). Third-party papers, datasets, and model weights retain their own licenses.
