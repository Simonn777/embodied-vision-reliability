# 0011 independent implementation review

Reviewed October18 New York / October19 UTC, before any new real-data decode,
prediction, fusion or score. **Approved for the bounded development pilot only,
conditional on publication of the canonical source freeze and exact main CI.**
No task improvement, model qualification or CVPR novelty is established here.

Separate Codex reviewers checked: (1) preparation/native prediction interfaces,
(2) probability/geometry/fusion, (3) saved-output scoring and protocol, and
(4) external supervision. This is AI-assisted independent code review, not a
claim of independent human reproduction. The owner did not personally author
or certify these implementations; see `CONTRIBUTIONS.md`.

Preparation uses the existing restricted NumPy unpickler and published array
hashes, selects queries only from frame0, and exports label-free predictor NPZ
separately. All50 examples necessarily enter preparation memory. A separate
process prevents keeping that container alongside the model. Small artificial
memory probes found streaming opcode inspection, but do not guarantee the real
archive's memory use; the hard8GiB limit and actual RSS remain necessary.

Native collection preserves official BGR/RGB preprocessing, xy→[0,y,x] queries,
raw yx→xy output mapping, original query state and twelve recurrent layers.
One read-only hook observes each single forward. Four streams reset only at
their starts; the output has64 forwards per group and no fusion/GT access.

Numerical fixtures check nonseparable pairing, separable-cost invariance,
fixed marginals/global means, true image/coordinate inversion, full native-head
order, native local-mean equivalence, and explicit nonconvergence. Transport
and local decoding are standard constructions, not novelty evidence. Exact
ties can cause coordinate instability despite tiny grid differences; tie
counts and axis-coordinate discrepancies are retained, never silently merged.

Independent review found and repaired these pre-freeze defects:

- Save both solvers' iterations/residuals and reconstruct numerical gates in
  the scorer, rather than trusting a `completed` flag.
- Reject negative/duplicate row indices, wrong GT ID dtypes, incomplete methods,
  reused raw filenames and symlink raw files.
- Compare the5pp gate using exact rational hit counts; binary subtraction
  otherwise rejected an artificial exactly5pp improvement.
- Require the supervisor's complete source list and40-hex commit before launch;
  verify prior phase freeze/execution bindings before continuation.
- Protect `wait4` reap/status assignment from interruption. Document immediate
  process-group kill at the deadline, matching implementation.

The initial parent full local suite returned7 failures,764 passes,7 skips in
52.61s. All7 failures were historical path-integrity fixtures under macOS's
symlinked `/var` temporary parent, not new scientific-code defects. A rerun of
both affected historical modules and all seven new modules with a canonical
workspace TMPDIR passed **149 tests,1 Linux/4-CPU skip in4.18s**. No historical
source/test was changed to obtain that result. Earlier new-module run had one
error-message regex mismatch after a schema fix; the diagnostic wording was
made consistent without changing the rejection or test threshold.

The final individual artificial suites have14 coupling,15 geometry,16 fusion,
18 preparation,13 native fake-backend,22 saved-evaluation and16 supervisor tests
(the last has1 Linux integration skip locally). No actual model/GT/real video
was used. Linux CI/cloud must exercise applicable resource tests before science.
Source/config hashes in the canonical freeze bind the reviewed versions.

Credential-free child environment, offline-library flags and audited code make
no network requests; there is no OS network-namespace isolation claim. Cache
immutability is enforced by reviewed code paths/hash checks, not a filesystem
sandbox. Seals bind saved files; actual chronological collection→seal→score
order still requires parent command receipts. Two simulated groups cannot
establish independent-scene generalization, statistical significance or paper
readiness, even if the development gate passes.
