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

## Scope

The first experiment is a synthetic 2D localization diagnostic. Its purpose is to validate the evaluation machinery and expose failure cases. It does not measure robot control, prove generalization to real video, or establish a new method. Public-video evaluation and stronger frozen estimators are the next stages.

The initial implementation runs on CPU. Apple MPS is available on the development machine; NVIDIA CUDA has not been tested. No paid compute is configured.

## Research practice

- Keep full sequences in separate development, calibration, and evaluation partitions.
- Fix the protocol before inspecting the final comparison and preserve all conditions.
- Report error together with the fraction of predictions accepted; estimate uncertainty across sequences.
- Link claims to executable runs and distinguish hypotheses from measurements.

Original code and generated diagnostic assets: [MIT](LICENSE). Third-party papers, datasets, and model weights retain their own licenses.
