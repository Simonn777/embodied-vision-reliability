# 0008: mean pairing failed the frozen training-quality screen

**Decision: `stop_this_candidate`.** All 32 trajectories and 64 predictions completed once. Both frozen quality gates failed. This exact mean-sort candidate is stopped; the broader research objective remains active. No manuscript, new method, novelty or quality-success claim follows. AI-assisted execution and reporting.

## Fixed results, including harms

The controls are independent, random antithetic and same-patch. The comparator is the per-case PSNR-best control (an oracle, not a deployable selector). All four images, both corruption seeds, methods and endpoints are retained below. Values are rendered from the unchanged [formal scores](study/scores.json), never recomputed by a model or scorer during publication.

### updates500

| Case | Independent PSNR | Random PSNR | Mean PSNR | Same-patch PSNR | Oracle control | Mean − oracle dB |
|---|---:|---:|---:|---:|---|---:|
| d1 seed 0 | 32.694147741 | 32.735706885 | 32.528643512 | 32.540188751 | random_antithetic | -0.207063373 |
| d1 seed 1 | 32.460313362 | 32.081318637 | 31.860305080 | 32.492469206 | same_patch | -0.632164126 |
| d2 seed 0 | 30.769219386 | 30.937307735 | 30.778461877 | 30.863058518 | random_antithetic | -0.158845858 |
| d2 seed 1 | 30.929965475 | 30.925336478 | 30.815224245 | 30.779932960 | independent | -0.114741230 |
| d3 seed 0 | 29.133947139 | 30.081824086 | 29.847602696 | 30.070119834 | random_antithetic | -0.234221390 |
| d3 seed 1 | 30.619898261 | 30.232727706 | 30.438517955 | 30.539533141 | independent | -0.181380307 |
| d4 seed 0 | 31.963512167 | 35.059921172 | 36.430520025 | 32.599280288 | random_antithetic | 1.370598853 |
| d4 seed 1 | 34.979647800 | 35.791049404 | 35.941222467 | 34.609840296 | random_antithetic | 0.150173062 |

### wall5s

| Case | Independent PSNR | Random PSNR | Mean PSNR | Same-patch PSNR | Oracle control | Mean − oracle dB |
|---|---:|---:|---:|---:|---|---:|
| d1 seed 0 | 32.424512482 | 32.165054757 | 32.333258373 | 32.425492797 | same_patch | -0.092234423 |
| d1 seed 1 | 32.688801460 | 31.918414279 | 32.485280340 | 32.205468687 | independent | -0.203521119 |
| d2 seed 0 | 29.951018413 | 29.893058776 | 29.785328065 | 29.812501515 | independent | -0.165690348 |
| d2 seed 1 | 29.554261724 | 30.089070649 | 30.121140641 | 29.956224319 | random_antithetic | 0.032069993 |
| d3 seed 0 | 29.855875919 | 30.040058889 | 30.159243175 | 29.723575914 | random_antithetic | 0.119184286 |
| d3 seed 1 | 30.038705032 | 30.873196866 | 30.722854278 | 30.874621477 | same_patch | -0.151767199 |
| d4 seed 0 | 37.222096830 | 36.437972236 | 36.607164790 | 36.680267937 | independent | -0.614932040 |
| d4 seed 1 | 37.836259973 | 37.559336297 | 36.272628521 | 37.563407535 | independent | -1.563631453 |

| Image token | 500-update mean gain dB | 5-second mean gain dB |
|---|---:|---:|
| d1 | -0.419613749 | -0.147877771 |
| d2 | -0.136793544 | -0.066810178 |
| d3 | -0.207800848 | -0.016291456 |
| d4 | 0.760385958 | -1.089281746 |

| Frozen gate | 500 updates | 5 seconds |
|---|---|---|
| `finite_paired_db_gains` | PASS | PASS |
| `mean_gain_ge_0_5_db` | FAIL | FAIL |
| `at_least3_group_means_gt0` | FAIL | FAIL |
| `each_group_mean_ge_minus0_2_db` | FAIL | FAIL |
| `mean_pooled_first_difference_le_1_05` | PASS | PASS |

Mean gain is **−0.0009555460164949459 dB** at 500 updates and **−0.3300652878587962 dB** at 5 seconds, below the unchanged +0.5 dB requirement. Positive image means are 1/4 and 0/4; the minimum image means are −0.4196137493 and −1.0892817462 dB. The ≥3-positive-images and every-image ≥−0.2 dB gates fail at both endpoints. Texture ratios are 1.025927862 and 1.047862584, both ≤1.05. The wall5s total mean/independent allocated outer-cost ratio is **0.9957494786484813**, passing ≤1.10. Passing texture/cost cannot rescue failed quality.

## Real execution and cost accounting

Execution main `74f9330014aa8a13a8c55785992e115f045f2fa9`; source `829688df750964d796417da569f0710453f62ca7`; canonical freeze `5680e6dae68ed102147ca89eeb115010cfe13687`. All 13 required inputs remain frozen. Publication uses this same saved execution, not the later result commit.

Pinned cloud software tests: **51 passed, zero skips** (7.049 seconds unittest; 7.754978 seconds outer), plus **16 independent artificial self-test groups**. CI Torch skips were not counted as runtime execution. The scientific outer wait4 returned exit 0 / raw status 0, **260.1861582160018 seconds**, peak RSS **304,904 KiB**. Actual training across shared trajectories used **98,994 optimizer updates**. All 64 MC50 predictions account for **48,000 forwards and 2,333,811,200 normal scalar draws**. No warmup or repeated trajectory occurred. All wall captures were between 5.000254871 and 5.003422856 seconds, within the original 5.05-second capture bound.

CPU Python 3.12.14, NumPy 2.0.2 and Torch 2.10.0+cpu on AMD EPYC 9V74 / Linux 6.18.55; four affinity slots [0,1,2,3], numeric threads 4, Torch interop 1, deterministic CPU and 8 GiB address space. Whole-study hard cap 900 seconds, case external 120 / inner training 90 seconds, audit/score/checker outer caps 120 seconds. All nine actual outer stage exit/raw wait records are 0.

The full dual-endpoint trajectory wall includes runtime/source checks, shared setup, training, both inference calls, I/O and shutdown. Endpoint **allocated outer costs** subtract only measured extra training after that endpoint and the other endpoint’s weight-write/restore/infer/save work. They are not independently timed standalone runs. Both endpoint costs are correlated allocations from one trajectory, and earlier snapshot overhead remains charged. Full case wall and all recorded components are preserved in the 32 receipts and [trajectory cost table](trajectory-costs.json).

| Allocated outer seconds, summed over eight cases | Independent | Random | Mean | Same-patch |
|---|---:|---:|---:|---:|
| updates500 | 27.496621860 | 27.684039420 | 28.141307961 | 27.435059347 |
| wall5s | 61.165986592 | 60.944992800 | 60.905999260 | 61.087942591 |

Audit passed after all 64 predictions locked and before the isolated scorer first opened four pinned clean references. Formal score ran once, outer exit 0, 0.324616 seconds, peak RSS 34,080 KiB. Independent saved-array replay used no production metric/Torch import or model: **45,770 checks / 5,287 numeric**, max absolute difference **7.105427357601002e-15**, fixed rtol 1e-10 / atol 1e-12, exit 0, 0.505234 seconds outer, peak RSS 32,444 KiB. It reproduces `stop_this_candidate`.

## Evidence, privacy and limitations

- [Copy/redaction manifest](copy-manifest.json): original project-relative source identity, original SHA/size, public SHA/size and exact transformation scope for every archived file. Command recipes never claim to be original private argv files.
- [Source/freeze bindings](source-bindings.json), [128 original output hashes](original-output-hashes.json), [before inventory](outer-control/pre-execution.json), [after check](outer-control/postcheck.json), [independent report](outer-control/independent-replay.json) and [nine outer stage records](outer-control/compact-execution.json). Full case receipts, started markers, closed stdout/stderr and wait records are in trajectories/ and trajectory-control/.
- The original compact summary **parses as valid JSON**, ends in a real newline (`0a`), and required no normalization. [Exact formatting review](summary-format-review.json) records its original hash; its public path substitutions are distinct from normalization. No claim is made that every unselected private artifact is valid JSON.
- All 2,249 prior public result files and 2,477 original tracked files were preserved except the four explicitly authorized README/state/note edits. Old protocols/source freezes remain unchanged. The 496 non-reference 0006 cache files, 84 original 0007 output files, all 407 failed-cache files and 22 old marker files remain byte-exact; four references were first checked/opened at the permitted scoring phase. 0006’s stdout gap remains **OPEN**, its candidate stays failed, and 0003 remains deferred.
- Raw images/references, noisy arrays, full predictions and trained weights remain private. Only non-reconstructable block SSE/counts, scalar metrics, logs, hashes and receipts are public. Original private absolute paths and command files stay private; `<PROJECT_ROOT>` / `<PYTHON>` are explicit placeholders.
- No scientific run, inference, formal scorer or independent replay was repeated for publication. New acquired asset bytes **0**; historical ceiling **7,532,690,072 bytes**, no added fees, package installation or schedule change.
- Four reused development image substrates, known synthetic covariance and a tiny MLP cannot establish independent-scene or real-camera efficacy, competitiveness against modern strong baselines, or population uncertainty. Generic antithetic novelty is already rejected. 0007’s variance pass did not establish a quality benefit.
- Future distinct hypotheses need a prospective rationale and untouched validation. Do not rescue this exact rule through hyperparameter, pairing-feature or budget tuning on these images or old tests. No manuscript is warranted.

Execution parents observe their timeout children via actual wait4; the parent itself has no recursively added outer observer. The parent task separately inspected the original tool returns. Publication-only validations do not replace original run evidence. Any publication-only exception is preserved separately.

Publication link review initially exited 1 on an unchanged historical 0007 link
in the append-only date note (`../../results/0007-gradient-screen/README.md`).
The old text is preserved; revised publication review distinguishes unresolved
baseline-only note links from newly added links. The new 0008 note link was
corrected before publication. Original failed reviewer source, stderr and wait4
record remain archived. This was a publication-only issue; no scientific output,
source, score or execution record changed.
