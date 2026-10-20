# Exact channel-affine family: fixed development closure

The [seven-arm result](active-write-01/README.md) retains every completed rho=0.15 / seed=0 comparison. The final all-active direct control trains the same head on all consumed spatial writes, while retaining fixed native histories and pointers. It is not recurrent BPTT. The extra gradient-bearing contexts are part of the intervention, not a uniquely isolated explanation.

| Arm | DEV JF % | ΔJF pp vs native | cat-girl ΔJF pp | upside-down ΔJF pp |
|---|---:|---:|---:|---:|
| Native | 92.1772205 | 0 | 0 | 0 |
| Multistep KD | 91.6837887 | −0.4934318 | −0.1743853 | −0.8124782 |
| One-step KD | 91.9914708 | −0.1857496 | −0.3086098 | −0.0628895 |
| Current-GT-memory KD | 92.3911244 | +0.2139040 | +0.0295183 | +0.3982897 |
| Cost-matched direct | 91.2875480 | −0.8896724 | −0.2872975 | −1.4920474 |
| Update-matched direct | 91.4280666 | −0.7491538 | −0.1031775 | −1.3951301 |
| All-active direct | 92.4005376 | +0.2233172 | +0.4764950 | −0.0298606 |

The pre-fit rule required both a mean above current-GT-memory and an improvement over native on each DEV parent. All-active exceeds the former by +0.0094132 points but fails the latter. Its saved decision is `close_exact_channel_affine_family`. No seed-1 continuation or additional KD/rho rescue of this exact family follows. This preserves the small positive mean honestly; it neither turns the failed rule into success nor proves that all learned memory interventions fail.

All-active J is 94.7513294% (−0.0138531 pp), F 90.0497459% (+0.4604875 pp). Its 250-update fit used 1,152.3876841 outer seconds against the complete-fit budget 1,168.5095887; rollout and evaluation used 109.8032847 and 1.1027645 seconds. The three actual outer costs total 1,263.2937333 seconds. Complete per-step losses and checkpoint hashes are retained; only the last completed checkpoint was evaluated. Earlier twelve- and seven-operation bundles, shared cache, technical failures and the two privileged TRAIN replays retain separate costs.

The 40-file text bundle is 833,680 bytes, manifest `8e5bcc1074df45dacd135c1cd485801b6a516cc0fd45fbd0b17fba856fab2d8f`. Copy and manifest checks used text only. No model, checkpoint, mask or array was loaded during publication.

The [native output-error analysis](output-error-analysis-01/README.md) describes all 150 official interior DEV frames, with exact saved-native J/F agreement. Error mass within the unchanged GT-boundary radius is 77.9971%/97.8042%; neither parent has a complete target miss or a nonempty disjoint prediction. These evaluator-GT pixel sums do not identify a causal repair signal, prove identity correctness, or supply an achievable improvement bound. A different TRAIN paired-boundary screen is being prepared without reopening the closed family.

All six source videos remain permanently development; the reported scores cover one target on two exposed DEV videos, not the official all-object benchmark or held-out evidence. The substantial +3 official J&F-point gate on each of two independent held-out sources, uncertainty, strong controls, ablations and replication remains unmet. No efficacy, novelty, driving or manuscript claim follows.
