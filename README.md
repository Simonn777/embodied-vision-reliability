# Embodied Vision Reliability

**When should a vision system stop trusting its own position estimate?**

An ongoing research workspace on visual localization under occlusion and appearance changes. The first question is whether causal temporal consistency can help identify large errors that single-image confidence misses. We keep the position estimator fixed so that the comparison is about recognizing failures.

The program runs from October 2026 to October 2027. The first twelve weeks focus on selective prediction; later work depends on what these experiments show. This is AI-assisted research in progress, with code, protocols, raw measurements, and negative results kept together. [Contribution and evidence notes](docs/CONTRIBUTIONS.md).

## Start here

| Artifact | What it answers |
|---|---|
| [Current state](CURRENT_STATE.md) | What has actually been completed? |
| [Research charter](docs/RESEARCH_CHARTER.md) | What is the question, and what would count as progress? |
| [Reading list](docs/LITERATURE.md) | Which primary papers inform the design, and how closely have they been read? |
| [Next actions](NEXT_ACTIONS.md) | What experiment or verification comes next? |

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

The first experiment is a synthetic 2D localization diagnostic. Its purpose is to validate the evaluation machinery and expose failure cases. It does not measure robot control, prove generalization to real video, or establish a new method. Public-video evaluation and stronger frozen estimators are the next stages.

The initial implementation runs on CPU. Apple MPS is available on the development machine; CUDA is unavailable, and no NVIDIA GPU measurements are reported. No paid compute is configured.

## Research practice

- Keep full sequences in separate development, calibration, and evaluation partitions.
- Fix the protocol before inspecting the final comparison and preserve all conditions.
- Report error together with the fraction of predictions accepted; estimate uncertainty across sequences.
- Link claims to executable runs and distinguish hypotheses from measurements.

Original code and generated diagnostic assets: [MIT](LICENSE). Third-party papers, datasets, and model weights retain their own licenses.
