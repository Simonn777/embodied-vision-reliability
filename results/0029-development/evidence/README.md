# 0029 real-video evidence screen

The first necessary coverage condition **passes**:38 of48 fixed queries are actionable, and all6 development videos have at least one actionable query (required: at least4). This measures availability of transferred sparse evidence, not its correctness or segmentation improvement. No scientific SAM prediction or query annotation was accessed. The complete48-query population and original no-action rule remain fixed.

| Video | Complete acquired RGB frames | Actionable queries |
|---|---:|---:|
| dogs-scale | 83 | 3 / 8 |
| upside-down | 65 | 8 / 8 |
| dog-agility | 25 | 8 / 8 |
| cat-girl | 89 | 8 / 8 |
| planes-water | 38 | 8 / 8 |
| dog-gooses | 86 | 3 / 8 |

Exactly54 native SuperPoint extractions and48 LightGlue matches completed after source freeze `c09e240009869402e2b97a55af42617eea3c6405` and [exact CI39381860871](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39381860871). All386RGB files and6initial masks were checked before/after use. No additional matching confidence threshold, object substitution, query change or retry occurred. No warnings or offline-guard attempts were recorded.

| Operation | Supervised wall seconds | Peak sampled group RSS KiB | Response-body bytes |
|---|---:|---:|---:|
| RGB acquisition | 503.101386 | 47,376 | 54,201,147 |
| Initial-mask acquisition | 6.042253 | 47,328 | 27,524 |
| Full evidence computation and admission | 27.674457 | 1,207,040 | 0 |

All three outer operations completed with exit0, no resource termination, descendants or cleanup errors. The computational worker alone took21.485886s; use the full outer time for this stage's cost. The complete60,000,000-byte acquisition hold remains retained; no refund from measured54,228,671bytes. The48query-GT bodies remain unavailable.

The original six source-recording groups are now permanently development. Independent saved-output audit passed77,458checks with0failures in3.463088s,124,752KiB peak sampled RSS. It independently reconstructed all48 witness records using scalar neighborhood labels, collision handling and within-cell-rank ordering; all matched exactly. It read only the6initial masks, with no model or query GT. Five manufactured audit tests also passed on their first run. A coverage pass only permits completing/qualifying the frozen SAM comparison and all-six prediction-sealing implementation. The prespecified oracle/accuracy controls and two-source held-out manuscript gate remain unmet.

Receipts retain file/array hashes and all48 witness records; original RGB/masks and derived feature/match arrays remain private. Only absolute workspace paths were redacted; publication-manifest.json records both original and public hashes.
