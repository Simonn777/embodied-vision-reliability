# 0006 — Can target-conditioned residual inputs denoise correlated noise?

Status: prospective development screen. No images have been read or model trained
for this experiment when this protocol is prepared. A separately committed
canonical source/config/protocol freeze and passing CI are required before data
acquisition or execution. This is AI-assisted original implementation and review.

## Question and boundaries

For additive, stationary Gaussian noise with **known** covariance, does removing
its correlation with each supervised target from that target's entire input
patch improve a small single-image denoiser over ordinary blind spots and
covariance-matched R2R at the same architecture/update count?

For center i and its 48 other 7×7 patch pixels, use
`z_j = y_j - C[j,i]/C[i,i] * y_i`. All 48 features use the same center. A shared
pointwise MLP reads that row only. Center feature 24 is a constant zero, giving
49 allocated inputs for architecture matching; it is not a raw-center bypass.
No spatial convolution follows this transform, no per-image normalization is
applied, and no covariance is estimated from the input or reference image.

Known jointly Gaussian noise makes the whole residual input vector independent
of the target noise for a fixed predictor in the ideal model. Fitted parameters theta(y) depend on
the noisy image: the final fitted predictor is not claimed independent of target
noise or to have an unbiased finite-sample risk. It does not preserve all clean signal,
guarantee performance after finite-image training, or justify an estimated-C,
non-Gaussian, signal-dependent or camera-noise claim. Common-mode noise can erase
all DC information. Scalar paired-view decorrelation is insufficient for a CNN;
the [mathematical audit](../docs/0006-mechanism-audit.md) retains that counterexample.
Gaussian conditioning itself and covariance-aware self-supervision are existing
ideas. The [three-method check](../docs/literature/0006-conditional-residual-prior-art.md)
found no identical representation in its limited scope; this is not a global
novelty proof. A positive screen would justify further investigation, not a paper.

## Data admission before outcomes

`configs/0006-development-images.json` pins the official PolyU repository commit,
metadata-only selection rule, four exact mean-JPEG paths/sizes/Git blob identities
and conservative scene-token exclusions. They are d1 toy, d2 plant, d3 classroom,
d4 wall. All crops/captures from those identified physical scenes are development
forever, including later-discovered matches. Filename tokens do not establish
scene independence. Other images have not been allocated to a final test.

One GET per missing selected file; no retry, no alternate source after failure.
Normal TLS, 131072-byte per-file cap, a pre-GET budget reservation for one
possible overlength sentinel byte per selected image, expected size and Git blob SHA1 checked
before JPEG decoding, with body SHA256 recorded. Failed transfers preserve the
received-byte count and partial hash when the process can write a failure receipt;
an uncatchable termination can leave only the marker/partial. Cache hits are rehashed, never
redownloaded. Preserve attempt markers, partials and failed logs. Total selected
payload is 68,858 bytes. Dataset use is noncommercial research with attribution;
[original notices](../docs/0006-data-license.txt) apply. No images/reference frames
or reconstructable prediction arrays are published in GitHub.

Require JPEG format and stored size 512×512; convert RGB (L is converted to RGB),
without EXIF rotation or rescaling. Convert decoded 8-bit RGB with fixed weights
0.299/0.587/0.114, divide by 255, take stored-coordinate rows/columns 192:320,
and cast to float32. These JPEG mean images are only **fixed natural-image
substrates for synthetic corruption**. They are not noiseless sensor ground truth
and this screen does not denoise the original real noisy capture.

## Corruption and controls

Use sigma=25/255, two fixed seeds and two conditions: white h=[[1]], and colored
h=g outer g, g=[1,2,1]/sqrt(6). Generate a fresh Gaussian field with an extra
kernel-size-minus-one halo, valid-convolve into 128×128, add to the reference and
cast to float32. Never clip the noisy image. This produces the stationary
finite-patch covariance computed from the same float64 kernel autocorrelation;
no circular/reflect/zero noise padding is allowed. The Gaussian theorem concerns
the ideal pre-rounding model; casting y to float32 adds rounding, so exact
Gaussianity/covariance are not claimed for stored floats. Compute k=C[j,i]/C[i,i]
in float64 then cast it once to float32. Condition fields share deterministic
random streams with different halo shapes; they are not independent conditions.
Record array/kernel/C hashes.

Each image/seed/condition runs four methods: ordinary blindspot; covariance-safe blindspot;
the target-conditioned residual; and R2R alpha=0.5. The safe control zeros all
features whose known C[j,center] is nonzero: only the center for white noise,
and the central 5×5 for the colored kernel, leaving 24 outer raw neighbors.
Its entire retained input is independent of center noise under the same Gaussian
assumption. This tests whether residualized near neighbors add value beyond an
already-valid context. No covariance threshold is learned or tuned. All allocate the same
49→64→32→1 ReLU MLP, initial weights, minibatch index stream, optimizer, updates
and patch coverage. Blindspot/residual center-input weights receive zero input,
so equal allocated parameters do not imply equal effective information capacity.
Valid 7×7 centers give 122×122 predictions; no boundary filling or cherry-picked
regions. No pretrained weights, augmentation of the clean signal or external
training data are used.

R2R uses an independent Gaussian **49-dimensional patch** draw eta with the
full known covariance: input p+alpha*eta, target y_i-eta_i/alpha. Training draws
are renewed per sampled patch. Primary inference is the arithmetic mean of
50 independent f(p+alpha*eta) draws, as motivated by R2R Eq.16; direct f(p) is
secondary only. These are matched-architecture adaptations, not replications of
the published large-network benchmark. Record all extra inference cost; the primary R2R comparison uses the published
AWGN alpha=0.5 convention, not a deliberately weakened direct-only baseline.
No linear Noisier2Noise correction is added. See the [baseline audit](../docs/literature/0006-r2r-implementation.md).

`configs/0006-stage0.json` fixes seeds and all settings: 500 Adam updates,
batch512 with replacement, lr0.001, default betas/epsilon, no weight decay,
no schedule/early stopping. Model selection never reads reference error or
chooses the best epoch. Training and augmentation streams are separate.
The pre-freeze review replaced an initially drafted second R2R alpha=1 control
with covariance-safe blindspot, before any image access or training; there are
still exactly four methods and 64 cases. Four
numeric threads, CPU only, float32 model and inputs, float64 covariance factor
calculation, no autocast/GPU. A fit incomplete at 90 seconds is a runtime failure,
not a smaller successful experiment. Each case has an external 120-second hard
limit; a resumable image group has a 1800-second outer hard limit. Completed
case outputs/markers are verified and reused; a started-but-incomplete case is
not automatically repeated. No partial-grid result is called a completed screen.

## Outcomes and fixed development continuation rule

An independent NumPy-only scorer reads saved local reference/prediction arrays.
It never imports the trainer. Compute float64 SSE/MSE and PSNR with range1,
without clipping predictions; zero SSE has an explicit exact-reconstruction
flag and no JSON infinity. Also record horizontal/vertical first-difference
error MSE (pooled over both directions) and an 8×8 partition's SSE/counts
using np.array_split on each 122-pixel axis. Save all training curves, seeds,
raw block measurements, finite checks, output/weight hashes and times. Local
prediction arrays remain available for an independent score replay; GitHub gets
small aggregated measurements, not the images.

For each of the eight colored image/seed pairs, compare residual PSNR against
the **largest PSNR among the three baseline methods**. Ties use the config baseline order (blindspot, safe_blindspot, r2r_alpha05).
This is a deliberately strong, predeclared oracle-best comparator, not a
deployable baseline selector.
Continue only if all of the following hold:

- mean paired colored gain >=0.3 dB;
- at least three of four image mean gains >0, and every image mean >=-0.2 dB;
- mean candidate first-difference error <=1.05 times that of the corresponding
  PSNR-best baseline, comparing the mean pooled-gradient MSE across the same eight pairs;
- white-noise residual, covariance-safe and ordinary-blindspot predictions are exactly equal
  under the same initialization/index stream (the operators reduce identically);
- all 64 cases complete, all analytic/software checks pass and independent
  saved-output scoring agrees. Report both seeds and all failures.

If the gate fails, retain the result and stop this exact candidate/configuration.
Do not cherry-pick an alpha, seed, image or condition or weaken thresholds to
manufacture a pass. A reasoned revision would be a separately labeled exploratory
study. Four development scene tokens do not support a population significance
claim, and patches/pixels are not independent experimental replicates.

## Evidence required before a manuscript

This screen is insufficient even if it passes. A subsequent protocol must freeze
untouched physical groups on at least two independently sourced image sets,
several correlation/noise regimes, covariance misspecification, strong modern
and classical correlated-noise baselines, fair total-cost comparisons, ablations
and independent reruns. Initial practical target for the known-covariance task:
>=0.5 dB average over the strongest matched-resource baseline, paired 95%
scene-cluster interval lower bound >0, positive gains on >=2/3 of held-out groups,
white-noise loss <=0.1 dB, and clear texture/failure accounting. At a matched
accuracy target, runtime should be <=1.25 times the strongest comparator; report
full quality/cost curves when architectures differ. Any change to those targets
must precede the corresponding untouched evaluation. Unknown-covariance/native
camera performance requires separate evidence; no synthetic result substitutes.
No manuscript drafting, submission or acceptance claim follows from Stage0.

## Provenance and resource bounds

Freeze current source/config/protocol/manifest/test bytes in a separate receipt
before data access. Verify the published commit and file identities before every
stage. Record environment, exact commands, wall time, peak RSS and actual process
exit. Bound address space at 8 GiB and cores/threads at four. Verify no other
research experiment is running. Reuse the already installed cloud CPU runtime;
no package installation, model download, paid API or paid hardware is authorized.

Retain the prior 7,499,066,782-byte acquisition ceiling, add a conservative
32 MiB allowance for this topic's primary-document/metadata retrieval (not a
fabricated measured network count), plus actual newly acquired data payload.
Successful-acquisition planned total is 7,532,690,072 bytes; the pre-GET
reservation also allows up to four sentinel bytes, still under 10 GiB. Do not touch old
frozen test data, alter failed results, resume missing PointOdyssey caches or
change schedules. Publish substantive real measurements only after review and CI.
