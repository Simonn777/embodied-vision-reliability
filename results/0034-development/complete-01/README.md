# 0034 fixed relative-RGB screen: negative; exact recipe closed

The fixed signed RGB/Gram gate fails five of eleven prospective development continuation clauses. **Native J&F is 95.4060107%; signed Gram is 95.3503264%, a loss of 0.0556843 percentage points.** All ten learned arms have lower pooled J&F than native. No new-method success, official benchmark result, driving result, held-out confirmation or manuscript is established.

## Complete primary result

Four fixed, permanently exposed TRAIN recordings use leave-one-parent-out fitting. Each of 40 terminal fits uses only the other three parents; the program loads all TRAIN labels once because each parent trains other folds. Predictions are sealed before this metric stage. The 224 interior frames exclude each first and last frame, with equal frame means within parent and four equal parent means. This computational cross-fit does not undo prior exposure or produce a final test.

| Arm | J (%) | F (%) | J&F (%) | J&F change vs native (pp) |
|---|---:|---:|---:|---:|
| native | 93.6297400 | 97.1822813 | 95.4060107 | +0.0000000 |
| logit | 93.5620152 | 97.2177976 | 95.3899064 | -0.0161042 |
| absolute | 93.5977111 | 97.1674138 | 95.3825625 | -0.0234482 |
| prototype | 93.4983001 | 97.1978877 | 95.3480939 | -0.0579168 |
| nn | 93.5452950 | 97.1769452 | 95.3611201 | -0.0448906 |
| ordinary_relations | 93.3985205 | 97.2051207 | 95.3018206 | -0.1041901 |
| bilateral | 93.4279069 | 97.1040115 | 95.2659592 | -0.1400515 |
| signed_no_gram | 93.6021661 | 97.1081585 | 95.3551623 | -0.0508484 |
| signed_gram | 93.6087984 | 97.0918545 | 95.3503264 | -0.0556843 |
| displaced_gram | 93.6005916 | 97.1086294 | 95.3546105 | -0.0514001 |
| rgb256 | 93.5944317 | 97.1279766 | 95.3612041 | -0.0448065 |

| Parent | Native J&F (%) | Signed Gram J&F (%) | Change (pp) |
|---|---:|---:|---:|
| dog-agility | 95.1135650 | 95.0855667 | -0.0279983 |
| planes-water | 97.5493528 | 97.5773818 | +0.0280290 |
| dogs-scale | 93.3846643 | 93.1499030 | -0.2347613 |
| dog-gooses | 95.5764606 | 95.5884541 | +0.0119935 |

Native is the strongest pooled control. Signed Gram beats ordinary relations on only one parent and the no-Gram gate on three; it remains below the capacity-matched displaced-Gram and RGB256 controls in pooled J&F. Neither adding Gram relations nor native-resolution RGB is supported as an advantage by this fixed comparison. These results do not prove that every RGB or relational refinement fails.

## Exact scope and controls

The same native-derived boundary band and reference geometry serve every arm. Native-scale RGB supplies class-relative prototype/nearest-reference differences, ten axial Gram relations, and a 7×7 bilateral control. The absolute control also receives the existing 32-channel SAM fine features. Ten small readouts use fixed 1,024 native-derived sample positions per noninitial frame, parent/frame-balanced weights, float64 L-BFGS and terminal states only. The signed gate cannot reverse the nearest-reference evidence; displaced relations and RGB→256→original are explicit controls. Initial signed correction is small and nonzero, not an exactly native initialization. See frozen ADMISSION/SCOPE and component documents for the full formulas and all thresholds.

Void 255 is treated as background under the inherited single-target metric policy. The candidate never updates SAM history or later support. This is a known-baseline/inductive-bias screen, not a full reproduction of learned guided upsampling, DenseCRF, FBA Matting, SegFix or SAMRefiner.

## Execution and completeness

All three scientific stages finish cleanly under four CPU threads, 8 GiB group-RSS and 1,800 seconds per complete operation. No model weights, source code, image or label payload was acquired. SAM is not loaded or called by these stages.

| Whole operation | Wall seconds | Peak group RSS (KiB) |
|---|---:|---:|
| prepare | 62.897802250 | 584,368 |
| fit_predict | 18.160135750 | 1,360,192 |
| evaluate | 10.060370875 | 483,568 |

The inherited native-cache cost of 458.300764250 seconds remains separate. These aggregate multi-arm stage costs do not establish a complete-pipeline speed improvement or a cost-matched final comparison. All 232 prepared frames pass native parity; 228 noninitial frames provide 3,780,352 eligible query positions and 233,472 training samples. The local 2,474,257,840-byte feature cache is generated output, not a download. Forty fits have no recorded failure; 232 prediction archives retain all 2,552 arm masks. The evaluation checks native parity and unchanged pixels outside eligible sites before reading metric labels.

All 80 individual terminal/trace JSON files and the aggregate model file are retained. 32/40 terminal gradients satisfy the configured tolerance; 8/40 reach the configured iteration limit. Actual iteration range is [8, 80] and closure-evaluation range [10, 92]. Finite completion is not a convergence guarantee. No fit is restarted or chosen by held-parent score.

Twenty-six manufactured tests passed before the run (1.835 seconds reported by unittest; 3.119596875 seconds whole process). These test geometry, weights, gradients, no-ops and metric accounting; they do not prove efficacy. The text publication checks exact original bytes and complete inventories, not an independent scientific rerun.

## Secondary population and decision

The older horizon population is descriptive only: all 96 events, 87 unique frames, and four final-frame events remain present. Native J&F is 95.2091636%; signed Gram is 95.1730261%, a -0.0361375-point change. Native values match the preceding saved TRAIN baseline to the fixed tolerance. This secondary population does not replace the primary decision.

| Predeclared clause | Passed |
|---|---|
| primary_JF_gain_at_least_1pp | false |
| margin_over_strongest_control_at_least_quarter_pp | false |
| beats_ordinary_relations_on_at_least_three_parents | false |
| beats_signed_no_gram_on_at_least_three_parents | true |
| no_parent_native_JF_loss_exceeds_half_pp | true |
| mean_native_J_loss_at_most_half_pp | true |
| mean_native_F_loss_at_most_half_pp | true |
| mean_JF_above_displaced_gram | false |
| mean_JF_above_rgb256_by_at_least_tenth_pp | false |
| no_fit_failures | true |
| full_coverage | true |

**Decision:** close the fixed signed RGB/Gram recipe without support/radius/gate/optimizer/threshold/seed rescue. All four parent results and all failed comparisons remain visible. No DEV continuation or paper follows. The six old recordings remain exposed development and the protected final recordings remain untouched.

The final gate stays +3 absolute official J&F points on each of two independent held-out sources, positive paired recording-level 95% intervals, neither J nor F down more than 0.5 point, strong same-information controls at a fixed common complete-pipeline cost, ablations and independent replication. It is not met.

The prior design and critique notes are source-only background. Suggestions of SciPy fitting, ignored void pixels or per-fold timeout in an earlier design note were superseded before data access by the frozen admission: qualified Torch fitting, void-as-background metric policy and one complete 40-fit operation. They are not the executed protocol.
