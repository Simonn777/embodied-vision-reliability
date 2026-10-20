# Native-feature supervised baseline: complete result and closure

The visual residual head improves the reported TRAIN fit subset but fails the fixed DEV continuation criteria. Close this exact architecture/training recipe without width, loss, epoch, radius or seed rescue. This is an ordinary supervised baseline and information ablation, not a novel method, official all-object benchmark or manuscript. The original goal remains unachieved.

## Complete accuracy

| Population | Native J&F (%) | Visual J&F (%) | Logit-only J&F (%) | Visual change (pp) | Logit-only change (pp) |
|---|---:|---:|---:|---:|---:|
| TRAIN fit-subset events | 95.2091636 | 95.8604743 | 95.2520425 | +0.6513107 | +0.0428790 |
| dog-agility | 94.1184338 | 94.4059074 | 94.1481663 | +0.2874736 | +0.0297325 |
| planes-water | 97.5378598 | 97.8168888 | 97.4337709 | +0.2790290 | -0.1040889 |
| dogs-scale | 93.6104010 | 94.6531532 | 93.7015047 | +1.0427522 | +0.0911037 |
| dog-gooses | 95.5699596 | 96.5659478 | 95.7247282 | +0.9959882 | +0.1547686 |
| DEV interiors | 92.1772205 | 91.8986949 | 91.9701540 | -0.2785256 | -0.2070664 |
| cat-girl | 92.2110594 | 90.6379920 | 92.0111152 | -1.5730673 | -0.1999442 |
| upside-down | 92.1433815 | 93.1593977 | 91.9291928 | +1.0160161 | -0.2141887 |

Visual DEV J/F change -0.5806481/+0.0235969 absolute points. Its J&F is -0.0714592 points relative to the learned logit-only ablation. Four of five conditions fail; only the mean-F-loss limit passes. The upside-down improvement does not cancel the requirement that both parents improve, and no successful-parent subset is reported as the result.

## Fixed experiment

Each arm has1,377 parameters: normalized32-channel native fine-scale image features plus sigmoid(native logits), Conv1×1(33→32), GELU, Conv3×3(32→1), and native logits+4*tanh(residual). The final convolution is zero initialized. The logit-only arm zeros the normalized image-feature channels, retaining identical architecture and initial state. Both use all228 noninitial TRAIN frames, same twelve shuffled epochs and Adam settings, original-resolution BCE+softDice, and only their terminal checkpoints. All228 frames are active: each arm completes2,736 updates;5,472 total. Normalization is TRAIN-only. No DEV payload enters fitting; no native model or extra encoder is loaded. Native history and pointers stay fixed.

Fitting weights228 frames equally, with24/37/82/85 frames by parent. The descriptive TRAIN accuracy table uses the older fixed96 horizon events/87 unique frames, equal24 events per parent, duplicates and four final-frame events retained; it is not a held-out test or full228-frame terminal evaluation. All154 raw DEV frames are retained, with first/last excluded from each parent mean (150 interiors) and the two parents weighted equally. Targets are unchanged original single-object IDs, not unions.

All228 training native-mask parities precede TRAIN label decoding. Prediction produces482 mask files for241 shared frames before current evaluation-label decoding. It passes241 native-zero parities plus482 zero-head parities; both DEV initial outputs remain native. Each trained arm has239 deployment head calls; zero-head checks incur additional charged verification arithmetic. Both terminal states remain unchanged, with zero SAM encoder/decoder calls and no network attempts. The published native DEV parent and mean J/F reproduce within1e−12.

## Costs and retained evidence

- fit: 115.000076459 outer seconds; peak group RSS2,428,400KiB; completed, exit0, no cleanup/limit failure.
- prediction: 17.848059458 outer seconds; peak group RSS377,712KiB; completed, exit0, no cleanup/limit failure.
- evaluation: 2.350311334 outer seconds; peak group RSS78,112KiB; completed, exit0, no cleanup/limit failure.

Seven manufactured head tests passed before fitting (1.450046000s process). The old native cache cost458.300764250s remains inherited separately and includes cache/GT-memory preparation. Cached features are not free or previously unseen information: the native decoder already uses them. This experiment does not measure a matched complete-pipeline speed advantage.

All5,472 training records, fixed data order, normalization, final checkpoint hashes, complete per-frame predictions/provenance, all96 TRAIN metric events, all154 DEV raw rows, every parent, sources/specs/logs/costs and the decision are preserved. Public text export retains all fields with literal workspace aliases. Model/data/cache/prediction arrays remain local, represented by hashes; a bare clone cannot run science without the declared qualified runtime and authorized inputs.

The fixed recipe is closed. This result does not prove why transfer failed, that no image information remains, or that every feature-based method must fail. Any materially distinct successor requires a falsifiable mechanism and strong controls; ordinary refinement/prototype conditioning have close priors. The unchanged two-independent-held-out-source +3 official J&F gate, recording-level uncertainty, component-loss limits, equal complete-pipeline-cost controls, ablations and independent replication still precede a manuscript.
