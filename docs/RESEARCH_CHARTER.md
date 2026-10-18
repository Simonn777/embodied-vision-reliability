# Research charter

**Program:** October 16, 2026–October 15, 2027.  
**Initial focus:** reliability of visual position estimates under partial observation.
**Owner:** Wumou (Simon) Xie. Research planning, implementation, literature screening, and execution are assisted by Codex; see [contributions](CONTRIBUTIONS.md).

## October 18 owner-directed revision

The owner explicitly authorized a completely new computer-vision topic, independent of old GitHub experiments, and requires substantial validated improvement before a manuscript. The initial localization question and milestones below remain the historical plan. They do not require continuing a rejected mechanism. A replacement topic must record its closest prior work, falsifiable mechanism, independent data and task-specific improvement gate before outcome inspection. Existing frozen results, test-set exclusions, resource limits and AI-contribution disclosures remain binding.

## Question for the first twelve weeks

For a fixed visual position estimator, can a causal temporal consistency signal identify large localization errors under held-out occlusion and appearance shifts better than confidence from the current image alone, at a comparable rate of accepted predictions?

This is a question, not a claimed contribution. Confidence, selective prediction, tracking, and active perception have extensive prior work. The [reading list](LITERATURE.md) records what has actually been checked. A search result or an abstract is not a full paper review.

An estimator that gives a plausible but incorrect position matters for embodied systems. We begin with perception: no robot controller, grasp-success result, safety certification, or real-world deployment is implied by a 2D experiment.

## What would count as progress

1. A reproducible evaluation pipeline with locked partitions, explicit failure cases, inexpensive baselines, and per-sequence uncertainty estimates.
2. Evidence about when temporal information helps, hurts, or adds nothing to single-image confidence.
3. A result that survives an independent video source and stronger visual estimators, or a clear account of why it fails to transfer.

A negative result is useful when the experiment rules out a concrete explanation. A larger GitHub contribution count is not a research outcome.

## Four phases, with decisions between them

| Dates | Main work | Decision and artifact |
|---|---|---|
| Oct 16, 2026–Jan 7, 2027 | Selective visual localization: synthetic diagnostic, public-video/data rights audit, fixed estimators, confidence baselines, episode-disjoint calibration | Reproduction package and twelve-week report. Continue a method only if its advantage survives controls; otherwise publish the failure analysis and revise the question. |
| Jan 8–Apr 8, 2027 | Cross-domain visual reliability: test frozen encoders or trackers, changes in scene/identity/camera, calibration transfer | Independent validation with no retuning on the final evaluation set. Clearly separate image perturbations from actual camera geometry. |
| Apr 9–Jul 8, 2027 | Observation decisions: abstain, wait, or request another view in a licensed lightweight simulator, if resources permit | Compare fixed observation budgets and inference costs. Use an open-loop perception study if a credible closed-loop setup is unavailable. |
| Jul 9–Oct 15, 2027 | Replication, ablations, failure analysis, and synthesis | Archived code, data manifests, complete results, and a technical report. A manuscript is conditional on evidence; conference acceptance is not a planned result. |

The later phases are provisional. Record changes in a dated decision note, including which observation or paper caused the change. Do not silently replace a failed hypothesis.

## Twelve-week milestones

- Weeks 1–2: read the closest primary papers; reproduce the diagnostic; test leakage, ties, missing observations, and causal scoring. Choose one licensed public video dataset and document its terms before acquisition.
- Weeks 3–4: implement one real-image baseline, lock scene/sequence partitions, and profile CPU/MPS cost. Verify annotations independently on a small sample.
- Weeks 5–6: compare single-frame, temporal, random, and oracle ranking controls. Fix the practical effect threshold and primary endpoint before the confirmatory run.
- Weeks 7–8: evaluate on held-out shifts; report accepted-prediction error and acceptance rate together, plus uncertainty at the sequence level.
- Weeks 9–10: test independent scenes/data and motion changes that can invalidate temporal smoothness. Include latency and failure analysis.
- Weeks 11–12: rerun from a clean environment, audit all claims against artifacts, and write the first report with a continue/change/stop decision.

## Daily work

Each session reads `CURRENT_STATE.md` and `NEXT_ACTIONS.md`, then completes one substantive research step. That may be a checked paper note, an experiment, a test or implementation needed for the protocol, a replication, or a reasoned hypothesis revision. Record evidence and the next action in `notes/YYYY-MM-DD.md`; append if a note already exists. Publish passing, reviewed changes with an informative commit message.

Search current ICML, CVPR, ICLR, CoRL, and relevant arXiv primary sources regularly. Select papers for their bearing on the question rather than adding unrelated papers to reach a quota. Every week synthesize what changed in the evidence. Every month inspect progress against the phase decision and available resources.

## Compute and publication defaults

Initial hardware is an Apple M5 with 24 GiB unified memory; CPU and Apple MPS are available, CUDA is not. The first diagnostic needs only CPU. Use up to four CPU threads, aim for under 8 GiB RAM, and cap an individual automated experiment at 30 minutes and an ordinary session at roughly 45 minutes. Split longer work into resumable stages. Keep downloaded datasets/checkpoints under 10 GiB in total and inspect size/license before acquisition.

No paid cloud jobs or paid API calls are configured. Do not provision them or claim NVIDIA GPU measurements. If local resources cannot support a valid next experiment, choose a smaller valid test or record the limitation instead of presenting an unrun result.

Publish only this project's original code, small generated artifacts, and brief source-attributed notes. Respect upstream licenses; do not rehost paper PDFs, private research data, unrelated repositories, credentials, or personally identifying datasets. Do not submit a paper, contact authors, or claim institutional affiliation on the owner's behalf.

Automatic work depends on the Codex scheduler, available account usage, the local execution host, network access, and GitHub authentication. A scheduled session is not a guarantee of a completed experiment. Missed work is logged honestly when execution resumes; dates and commits are never backfilled to look continuous. Research work stops after October 15, 2027 unless the user renews it.
