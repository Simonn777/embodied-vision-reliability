# Saved manufactured-result audit

The independent audit passed on its first invocation: exit 0, outer wall time **0.083567084 s**, empty stderr. Actual command, timestamps, hashes, and logs are in `../audit-execution-01/`. The auditor was prepared before this invocation; it did not import or execute the generator, regenerate observations, fit parameters, read real data, or call a model.

The generator source SHA-256 is `0b058405fbc151b8765dafdb4e4a81f8c7e48f515da3d217b2df33b9a9565683`; the saved mathematics SHA-256 is `befcf2f368ad964324bee7f72c2179c5487212c9899c1cd39d276fceccf2f7c5`. Source/result/receipt/stdout/stderr bindings agree. The original execution completed in 0.337930375 s externally, with 0.245473667 s recorded internally and no stderr. Auditor SHA-256: `442b26385b86ce75c3a878dd6c214ae9ec2fd0e2f1cfcebcf9d0774c8d653c42`.

Recomputed final saved Jacobian spectra, physical/scaled conversion, declared thresholds, and all saved Richardson rank counts agree. The six full-unknown ranks are **21, 21, 21, 9, 19, 15**, respectively: generic seven/eight points; zero intra-readout twists; zero readout duration; vertical flow; equal row times. Earlier Richardson matrices were not saved, so their rank counts were checked from their saved spectra rather than independently reconstructing their derivatives.

All eight known-R/angular-velocity matrices were independently rebuilt from saved rays, row times, and truth parameters. Matrix coefficients, truth vectors and right-hand sides agree; maximum reconstructed truth residual is **8.881784197001252e-16**.

| Known-R/omega fixture | Points | Rank / 12 | Scaled condition number |
|---|---:|---:|---:|
| Generic | 4 | 12 | 3814.55 |
| Generic | 5 | 12 | 1014.08 |
| Zero intra-readout twists | 4 | 12 | 7961.36 |
| Zero intra-readout twists | 5 | 12 | 1108.96 |
| Vertical flow | 4 | 11 | Rank deficient |
| Vertical flow | 5 | 11 | Rank deficient |
| Identity reference rotation | 4 | 11 | Rank deficient |
| Identity reference rotation | 5 | 12 | 2648.21 |

The identity-rotation four-point left-null projection verifies the analytic rank upper bound of 11. Its one-dimensional nullspace changes translation direction: the relative functional null norm is **0.0168989855**. By contrast, vertical-flow four/five-point nullspaces have translation-direction ratios **6.66e-16 / 2.17e-16**; this particular ambiguity concerns depth offsets and row velocities, with no observed translation-direction ambiguity. These ratios use the same saved-matrix rank threshold and the declared scaled parameter coordinates. They are continuous numerical diagnostics, not noise robustness guarantees.

`zero_motion` is an imprecise internal fixture name. Its correct report label is **zero intra-readout twists, nonzero inter-frame pose**: both cameras have zero velocity during their own readouts, but reference R and t differ between images; row times and readout duration remain nonzero. Infinitesimal changes in the unknown row velocities therefore still affect the residual. Full local rank at this zero-velocity truth is not contradictory. The zero-readout-duration control instead sets all row times to zero and removes velocity sensitivity, producing rank 9.

This is a saved numerical/algebraic qualification, not a scientific accuracy experiment. Finite-difference disagreement is not a certified error bound; neither full rank nor the observed condition numbers establish global uniqueness, practical robustness, the affine-depth assumption on rolling-shutter images, or novelty. No real-data experiment is admitted by this audit.
