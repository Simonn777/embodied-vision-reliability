# Conditional residual training core: independent mathematical review

Disposition: **no mathematical or baseline-implementation blocker found** in `scripts/conditional_residual_training.py`, SHA256 `93f68b4efe2f655590e5b8fc3ac7ff61006c8d031f9dd822c9d1f7f5d6ac5c41`. This is a source/NumPy-fixture review, not a scientific fit or a claim that tensor-runtime tests passed.

- Lines 61–77: full float64 49×49 covariance, finite positive variance and symmetry, full Cholesky; no diagonal replacement, jitter or repaired covariance. Factor is explicitly rounded to float32. The effective covariance error is measured and recorded at lines 92–104.
- Lines 80–105: raster 7×7 valid patches give 14884 rows. `k=C[:,24]/C[24,24]` is formed in float64 and cast once. The safe mask comes from exact float64 covariance nonzeros, independent of rounded gains; a gain underflow does not silently unmask a correlated feature.
- Lines 108–133: every residual row subtracts its own raw center. Feature24 is zero. Safe blindspot zeros exactly the supplied covariance mask. R2R receives unmodified 49D raw patches. There is no spatial transform or cross-row normalization after these features.
- Lines 136–147, 181–187: same explicit CPU float32 49→64→32→1 Linear/ReLU model with biases, same initialization under the caller's prescribed seed, and independent local generators for minibatch indices and augmentation. Adam updates/settings are common. Parameter count is 5313.
- Lines 202–216: R2R samples a full 49D standard-normal row and uses `noise=epsilon@L.T`, which has covariance `LL.T=C` before float32 rounding. The SAME row of noise enters `inputs+alpha*noise` and `noisy_center-noise_center/alpha`. The sign, reciprocal alpha and center coordinate are correct; there is no separate target draw or clean-target access.
- Lines 249–293: primary R2R output is the arithmetic mean of exactly 50 augmented forwards per valid patch in the public API, accumulated in float64 then converted to float32. Direct input is separately recorded and not used as primary. No Noisier2Noise correction is applied. Calls/draws and both inference times are recorded. The pointwise model needs only marginal patch draws, not a globally coherent auxiliary field.
- Lines 296–396: public settings cannot silently shorten the requested 500 steps or T50. `fit_one` accepts only noisy image, covariance, method and fixed settings/seeds. It reads no files and has no clean image, reference score, epoch-selection or metric callback. Failed fits do not return successful predictions. Clean/reference access outside this module belongs to the separate launcher/scorer review, not this audit.

## Executed artificial checks

Only NumPy was imported; Torch is not installed locally. Independently built the declared covariance using its analytic rational autocorrelation rather than the implementation's covariance generator. Confirmed the actual colored mask has 25 zeros/24 retained raw features; residual gains and same-center arithmetic agree exactly; three white-condition feature arrays are byte-identical. The factor's measured covariance error is `7.288262476173868e-10`, and the receipt value agrees exactly with an independent reduction.

The nine `NumpyInterfaceTests` passed under `-W error` in 0.012 s, with all five numeric-thread environment limits set to four. No Torch fit, data read, model download, network or scientific score occurred; repository files were not changed. `numpy-check.json` and `unit.stderr.log` retain the evidence.

## Runtime boundary and small nonblocking note

The tiny tensor fixtures (four artificial rows, two optimizer steps) must pass in the existing CPU Torch runtime before real experimental fits. Source inspection and identical feature/seed/index logic support white equality, but this local review did not execute whole-model white equality or R2R MC tensor behavior. Do not describe skipped tensor tests as passing.

The core still accepts `r2r_alpha1` as an unused helper/test method. The current protocol/config admits only `r2r_alpha05`; the parent launcher must remain restricted to that frozen method list. This unused internal branch does not change the admitted 64-case grid and is not a mathematical blocker.
