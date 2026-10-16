# Embodied Vision Reliability

**When should a vision system stop trusting its own position estimate?**

An ongoing research workspace on visual localization under occlusion and appearance changes. The first question is whether causal temporal consistency can help identify large errors that single-image confidence misses. We keep the position estimator fixed so that the comparison is about recognizing failures.

The program runs from October 2026 to October 2027. The first twelve weeks focus on selective prediction; later work depends on what these experiments show. This is AI-assisted research in progress, with code, protocols, raw measurements, and negative results kept together. [Contribution and evidence notes](docs/CONTRIBUTIONS.md).

## Start here

| Artifact | What it answers |
|---|---|
| [Current state](CURRENT_STATE.md) | What has actually been completed? |
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

Real Online BootsTAPIR weights run on video0's64-frame, two-query development clip in39.337seconds, with a776.1MiB process peak and exact fresh-process replay. All50 video identities and exact-frame/prefix groups were audited: development1/calibration10/evaluation20/reserve19. The [result and reproduction entry](docs/0002-64frame-and-freeze.md) links raw outputs and the complete prospective exploratory protocol (v2; the pre-merge provenance repair supersedes preserved v1 before evaluation). Shared objects/camera, unknown scene dependencies and upstream checkpoint exposure limit claims. No held-out model inference or reliability comparison has run.

The first experiment is a synthetic 2D localization diagnostic. Its purpose is to validate the evaluation machinery and expose failure cases. It does not measure robot control, prove generalization to real video, or establish a new method. Public-video evaluation and stronger frozen estimators are the next stages.

The initial implementation runs on CPU. Apple MPS is available on the development machine; CUDA is unavailable, and no NVIDIA GPU measurements are reported. No paid compute is configured.

## Research practice

- Keep full sequences in separate development, calibration, and evaluation partitions.
- Fix the protocol before inspecting the final comparison and preserve all conditions.
- Report error together with the fraction of predictions accepted; estimate uncertainty across sequences.
- Link claims to executable runs and distinguish hypotheses from measurements.

Original code and generated diagnostic assets: [MIT](LICENSE). Third-party papers, datasets, and model weights retain their own licenses.
