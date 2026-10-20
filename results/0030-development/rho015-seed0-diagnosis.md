# Complete first-configuration diagnosis

The rho=0.15 / seed=0 multistep teacher did **not** produce a DEV improvement. All six complete own-rollout arms are retained in the [completed comparison](rho015-seed0-complete/README.md), with the earlier teacher/KD operations preserved in the [initial bundle](rho015-seed0-initial/README.md). These are exposed, single-target development results on cat-girl and upside-down: 154 raw frames, 150 interior frames, equal-parent means.

| Arm | Mean J (%) | Mean F (%) | Mean J&F (%) | ΔJF pp vs native | cat-girl ΔJF pp | upside-down ΔJF pp |
|---|---:|---:|---:|---:|---:|---:|
| Native | 94.7652 | 89.5893 | 92.1772 | 0 | 0 | 0 |
| Multistep KD | 94.2442 | 89.1234 | 91.6838 | −0.4934 | −0.1744 | −0.8125 |
| One-step KD | 94.5860 | 89.3969 | 91.9915 | −0.1857 | −0.3086 | −0.0629 |
| Current-GT-memory KD | 94.8481 | 89.9342 | 92.3911 | +0.2139 | +0.0295 | +0.3983 |
| Cost-matched direct | 94.0891 | 88.4860 | 91.2875 | −0.8897 | −0.2873 | −1.4920 |
| Update-matched direct | 94.2013 | 88.6548 | 91.4281 | −0.7492 | −0.1032 | −1.3951 |

Multistep beats these two direct controls but loses to native, one-step and the GT-memory control. That does not establish the intended multistep advantage. The GT-memory arm learns from current TRAIN GT-derived memory targets and uses the same causal head without GT at deployment. Its small positive result is worth retaining, but is neither a substantial improvement nor held-out evidence.

## What the fixed TRAIN fit establishes

The completed diagnostic evaluates the final three KD heads once on all 48 saved TRAIN inputs/targets: 144 head forwards, no SAM2 calls, fitting updates or DEV access. `E` is measured-Gram squared residual error normalized by the anchor's energy bound squared; `E0` is the same error for zero output. The reported ratio divides the equal-parent mean E by the equal-parent mean E0; it is not the mean of individual ratios.

| Head | Mean E0 | E/E0 | Mean defined Gram cosine | Defined directions |
|---|---:|---:|---:|---:|
| Multistep | 0.996345 | 0.509740 | 0.686547 | 48/48 |
| One-step | 0.108796 | 0.637623 | 0.589434 | 47/48 |
| GT-memory | 0.067539 | 0.289997 | 0.508033 | 48/48 |

All three beat zero overall, while retaining substantial residual fit error. GT-memory is worse than zero on dog-gooses: E/E0=2.411844 and mean cosine=−0.148739. Different target energies make raw training-loss magnitudes unsuitable for ranking fit quality across arms. No constant-head optimum or generalization lower bound was measured.

The teacher's selected TRAIN surrogate improves about 5.67%, but selection includes the zero state and guarantees non-increase. Partial final-head fit and negative own-rollout accuracy leave several explanations open: imperfect target regression, parent generalization, repeated-write context shift, and mismatch between the low-resolution raw-mask/presence surrogate and native J/F. None is uniquely identified here.

The subsequently completed [saved-teacher replay](teacher-task-alignment-01/README.md) measures a small native TRAIN horizon-set benefit: J&F 95.209164%→95.536766%, +0.327602 pp (J +0.219612, F +0.435592). Parent J&F deltas are +0.265066/−0.003267/+0.818982/+0.229628 for dog-agility/planes-water/dogs-scale/dog-gooses. All 96 anchor-future events remain, covering 87 unique frames and including four final frames. These privileged fixed-context masks do not constitute a causal rollout, official benchmark score or upper bound. The small gain rules out a claim of no teacher task benefit in this measured set, but does not demonstrate the intended deployment gain or uniquely explain its failure. All-active-write training is not admitted by this diagnosis.

## Cost, provenance and limits

Teacher plus multistep regression used 1,168.509589 outer seconds. Cost-matched direct used 247 complete updates and 1,151.319815 seconds; update-matched direct used 288 updates and 1,339.648869 seconds. The two runs share all 247 prefix losses/anchor identities and saved checkpoint hashes at steps 48, 96, 144, 192 and 240. This is deterministic same-seed consistency, not independent-seed replication; unequal-step final checkpoints are not compared.

The new seven operations, including both full rollouts/evaluations and fixed-fit, total 2,720.611595 outer seconds. The earlier twelve total 1,630.789068 seconds. Cache and original failed-evaluation costs remain explicit references; overlapping inner timers are not added. Complete-pipeline comparisons must include the common cache and rollout/evaluation costs, not just head fitting.

The imported bundle contains 53 text files / 1,040,735 bytes, manifest SHA-256 `3732238f1c5434ce2a5e85f044c7a5ec25b4d7ff39ae7f304e6df9027bfb3e0c`. All 52 manifest entries and 49 original-to-public literal path transformations were checked; the repository copy is byte-identical and its prior-bundle references match. No model, checkpoint, image, mask or array payload was read or copied during integration.

The teacher-task-alignment replay is a separate later operation; its own linked bundle retains its costs and provenance and is not included in the seven-operation totals above. This single configuration does not establish novelty, a driving result, or the unchanged +3 official J&F gate on each of two independent held-out sources. It also does not prove that every learned memory correction must fail. Further authorized development must preserve these results and state its comparison and scope explicitly.
