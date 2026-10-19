# 0022 traceable frontend qualification

This is asset and runtime qualification, not a scientific experiment or a new
method. No benchmark image, intrinsic, depth, match or pose payload is admitted
by this protocol. No scene or pair is allocated. The 0012 efficacy gate and all
closed studies remain unchanged. Model feasibility does not establish novelty.

The asset-only freeze `ce7b02ef130ca6f99a71fdb13f590a41e4293641` completed
acquisition after exact CI. The second implementation freeze admits isolated
installation and manufactured qualification after independent review and exact
CI. Installation must validate the successful acquisition outer/inner receipts;
qualification must validate the successful installation outer/inner receipts and
same second-freeze source identities. A failed outer operation cannot be
promoted by an inner success. All original acquisition evidence is retained.

## Fixed assets and bounds

Use only `configs/0022-source-model-assets.json` and
`configs/0022-wheel-assets.json`. MoGe source is pinned to
925b8ed835a7a9cdb7578ba15c658a0afc969030, utils3d to
3fab839f0be9931dac7c8488eb0e1600c236e183, and LightGlue to
eb42fee2d71449efb0aa5c10549752b5d75384d8. Preserve upstream files and licenses
privately. Source blobs are checked against their Git blob identities; source
SHA256 is additionally recorded. No editable installation of broad upstream
requirements or source-code substitution is authorized.

MoGe2-S-normal checkpoint revision 26b477f41595707c5db6770294c0d1721e8ed4ed is
140,550,416 bytes, publisher SHA256
79a16621928c2bf0ed04659218c55c01075e950507f40bb3332fb4c873d3e1dc.
Official SuperPoint/LightGlue release files total 52,706,365 bytes. Their release
metadata supplies no digest: only these two exact official release URLs use an
explicit first-download SHA256, with exact size validation. Do not describe that
as comparison to a publisher digest. Once acquired, every load must compare the
saved hash. Never publish weights or third-party source in this repository.

The source/model manifest has 105 files totaling 194,142,010 bytes. Its maximum
response-body budget is 210,000,000 bytes. The wheel batch has a separate
90,000,000-byte maximum; all wheels must have publisher SHA256/size pins. These
limits include redirect/error bodies and leave a 20,000,000-byte reserve for
recent and follow-up metadata (including unmetered web-tool reads). The prior
conservative remainder was 2,764,780,692 bytes; after the full 320,000,000-byte
reservation, at least 2,444,780,692 bytes remain reserved for later work. Actual
HTTP body receipts, not disk size, update the ledger. No full dataset downloads.

Use the reviewed 0022 acquisition helper. One batch per fresh output directory;
no silent retry, replacement, denied-access workaround, or ignored size/hash
failure. Failed partials and receipts remain. A distinct technical repair must
be disclosed before another bounded attempt. Allowed HTTPS hosts are explicit;
TLS verification stays enabled and signed redirect URLs are redacted in logs.

## Isolated runtime

Use existing Python 3.9.6/Torch 2.8.0/NumPy 2.0.2 read-only. Add only pinned
wheels into a new task-private `runtime` directory with `pip --no-index
--no-deps --target`; record the actual command, exit, package list and hashes.
Do not modify the shared environment or allow resolver downloads. Reconstruct
upstream package directories from the verified source layout. Preserve utils3d's
checked-in lazy export maps. Model constructors use local checkpoints only;
loading uses `weights_only=True`, CPU mapping and complete learned-key checks.
Do not invoke MoGe `init_weights` or fetch DINO weights. Enable the author
PyTorch native-SDPA adapter after strict weight checks. Set Torch intra-op to
four, inter-op to one and OpenCV to one thread. Inspected Python network/process
audit events are denied and recorded during import/model loading/inference;
this is not an OS sandbox for arbitrary native syscalls. No GPU/MPS or
paid/cloud execution.

## Fixed artificial qualification

The exact deterministic manufactured RGB construction is fixed in `qualify.py`:
seed 0 noise, fixed gradients and four fixed colored squares. Two 384 by 512 images differ by a 12-pixel horizontal shift with
zero fill. This is a numerical/API fixture, not real scene data and not evidence
of depth or pose quality. MoGe uses full author `infer` on CPU float32,
`num_tokens=1200`, `use_fp16=False`, `apply_mask=False`, and
`force_projection=True`. Record the raw depth and author validity mask separately;
do not replace full inference with a raw network forward pass. Each image gets
one prediction, stored with its image/model/source/config hashes.

Extract SuperPoint once per image, at most 2048 features, `resize=None`.
LightGlue uses the SuperPoint model, no flash attention or mixed precision, and
both adaptive stopping/pruning confidences disabled (-1). Retain the upstream
fixed matching threshold. Local loading preserves the official legacy key
renaming; report all missing/unexpected keys and distinguish documented,
deterministic buffers from learned parameters. Missing learned keys or ignored
model configuration is a failure, never silently randomized model content.

The qualification checks image/depth/mask dimensions, finite values on valid
pixels, feature/descriptor/match dimensions and index bounds, a nonempty valid
depth map and at least one matched pair. These are execution eligibility checks,
not accuracy targets. Keep all outputs, warnings and failures. No fixture or
threshold tuning after the result. Technical compatibility repairs require a
new labeled attempt preserving the original. No benchmark/GT data may be read.

Supervise the full process and descendants: at most four CPU threads, 8 GiB
summed RSS and 1800 seconds. Record wall time and sampled peak RSS, including
import, loading and both image inferences. A pass establishes only local resource
and interface feasibility. Full frontend costs remain chargeable in every later
scientific comparison; no cached-cost speed claim.

## Subsequent scientific admission

A separate prospective protocol must fix scenes/edges, camera/resize contracts,
match/depth validity handling, method, strong controls and failure-inclusive
metrics before any benchmark payload/inference. Use one immutable full depth
realization per image for all incident edges. Keep GT poses outside prediction
inputs until complete predictions are sealed. Reference edges must be charged
and supplied equally to geometry/multiview and same-field joint-BA controls.
No manuscript or efficacy claim follows from this qualification.

## Technical attempt 02: optional CPU-probe compatibility

Attempt 01 at `b1d337622e4d7bfc1adbc736fe8355196af759d0` installed successfully
but failed during imports before any model construction or inference. The exact
NumPy 2.0.2 `check_support_sve()` optional `lscpu` probe received our general
`QualificationError` instead of its caught missing-command `OSError`. The first
receipts and full private logs are retained in the attempt-01 record.

The next source freeze permits one fresh installation and one qualification with
the same assets, artificial images, numerical settings and acceptance checks.
The only runtime compatibility exception raises `FileNotFoundError` for the
verified NumPy optional probe on macOS without launching a subprocess, recording
it separately as a compatibility probe. All other process/network events remain
denied and recorded as violations. This does not authorize arbitrary subprocess
execution or any scientific payload. Keep the shared NumPy package unchanged.

## Technical attempt03: exact unused LightGlue release entries

Attempt02 at `6ec0980bc9c53ad70deb70667ef85d3d1c7cc27f` passed imports and
strict MoGe/SuperPoint loading, then failed before inference on exactly nine
unused release-checkpoint entries `log_assignment.{0..8}.r`. The pinned author
MatchAssignment never defines/reads these members and its loader ignores them
with strict=False. The retained local weights-only inspection verifies each is
float32 shape[1], finite and exactly one. No scientific input or inference ran.

After review and exact CI, attempt03 may freshly install the same assets and
repeat the identical artificial qualification. Require that exact complete
nine-key set, nine model layers with no r member, and the inspected dtype/shape/
value; record canonical tensor hashes, remove only those entries, then strictly
load and verify every active state value. All other unexpected/missing active
keys remain fatal. This is an explicit compatibility conversion, not weight
relaxation or a changed model. Keep both earlier attempts unchanged.
