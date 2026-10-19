# Native observer contract

This directory's generator is applied only to a fresh copy of PoseLib
`6a636b1d84273d5e3ba97676463476569611572a`. It verifies each touched original
file's SHA256 before preparing any mutation. The original accumulator is **not
edited**: its public fields are observed in the existing LM flow. The receipt
contains a unified patch and every original/result hash. The native templates
contain no build or dataset access commands.

`instrument_source.py --source FRESH_COPY --receipt NEW_RECEIPT` requires a copy
under `0018-preparation`; `--check-only` instead performs transformations in
memory without writing a source or receipt. A started receipt remains after an
interrupted mutation, and a retry cannot overwrite it silently. The root-owned
build script freezes compiler/dependency settings and builds a separate package.

The new Python API accepts camera dictionaries:

```python
geometry, info, trace = poselib.estimate_monodepth_relative_pose_trace(
    x1, x2, depth1, depth2, camera1, camera2, options,
    initial_pose=None, trace_capacity=50000)
```

The existing API remains available with the observer disabled. `geometry` and
`info` are returned from one invocation of the unchanged existing wrapper.
Capacity may be smaller for artificial overflow tests, never greater than
50,000. Overflow stops retention, not optimization, and marks the trace
incomplete. Python-runner serialized-byte limits are additional safeguards.

`trace.schema` is `evr0018.native-trace.v1`. It contains `complete`, `capacity`,
`event_count=len(events)`, `event_attempts`, `dropped_events`, and flat `events`.
`proposal_trace_hash_hex` is an FNV-1a64 diagnostic hash over all sampled indices
and scored proposals (raw quaternion/translation/depth calibration, already
computed score and inlier count); it is not a cryptographic provenance hash.
`sampled_draws` and `scored_proposals` accompany it. Only record-breaking proposals
are serialized individually. Source, binary, trace and receipt SHA256 bindings
remain the authoritative integrity checks.

Every event has `kind`, `lo_id`, and `finite`. Nonfinite floating observations
become explicit null values and set `finite=false`; no numeric value is replaced
by zero. `finite` covers all serialized floats, including auxiliary diagnostics.
A rejected nonfinite trial is retained. A finite state/cost with nonfinite
auxiliary diagnostics remains separately distinguishable. State objects contain
`q` in wxyz order, `R`, raw `t`, `scale`, `shift1`, `shift2`.

The five optimizer kinds `lo_begin`, `initial`, `attempt`, `exit`, `lo_end` have
nonnegative `lo_id` and a `stage` from `proposal_lo`, `terminal_lo`,
`final_refinement`. `lo_begin` supplies `support` original indices, all effective
BundleOptions and depth weights, and `parent_kind/parent_id`. `initial`,
`attempt`, `exit` contain `native_cost={numerator,count,mean}` and per-kind native
block counts. An `attempt` is emitted when a candidate state and cost were
actually computed; an early gradient/step termination has no candidate attempt
and is represented by `exit.reason` and existing norm/counter observations.
Accepted attempts have `accepted=true`. Exit cost/block snapshots are the last
accepted state's cost pass, even if the live accumulator contains a rejected
trial or Jacobian count. `live_count`, `jacobian_count` and `solve_count` preserve
that distinction. No accumulator reset or second cost evaluation obtains them.

The other kinds have `lo_id=-1`: `proposal` has monotonically increasing
`proposal_id`, sample iteration/index and existing scores; `incumbent` records
`source_kind/id` and `previous_source_kind/id`; `pipeline.phase` is `begin`,
`pre_final` or `returned`. The latter two give source identity and raw state.
`pre_final.final_refinement_enabled` and
`returned.final_refinement_performed` explicitly represent an omitted wrapper
stage. Final ancestry follows the returned source through each LO's parent;
temporarily adopted or discarded unrelated LOs are not final ancestors.

Artificial qualification, performed by the parent after review:

```text
python qualify_native.py --baseline-package ORIGINAL_PACKAGE_PARENT \
  --instrumented-package TRACE_PACKAGE_PARENT --output NEW_RECEIPT.json
```

This regenerates three fixed 48-point fixtures in separate package subprocesses.
It compares baseline/off/on/cap-2 outputs and cap-independent proposal hashes,
then calls `diagnose_native_trace` for all six full traces. Child receipts retain
traces and fixture hashes. It does not read HDF5, images or scientific labels.
`qualify_observer.cpp`, linked to the fresh PoseLib and existing Eigen, separately
checks manufactured accepted/rejected steps, early exits, count reuse,
TRUNCATED zero weights, overflow and nonfinite trials. It also calls the actual
refiner on artificial points to check directional visibility loss and gain.
These tests qualify observation and reconstruction; they do not establish
scientific accuracy or novel methodology.
