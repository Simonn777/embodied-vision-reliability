# D38 compressed/batched qualification failed

Source634f0561893b762e90aabb7fe1448c26075f256d. All22 manufactured unit checks pass in1.548070000s at246112KiB. The separate native qualification fails in20.395268125s at2557632KiB with exit1, no timeout, no remaining processes and no scientific payload reads. The failure, specifications and exact source snapshots are preserved here.

At invented frame8, the two alternative batched stacks and early-mean stack pass the frozen atol1e-4/rtol1e-5 tolerance with maximum absolute differences around1.9e-6. The reused selected stack is exact. The compressed flat stack fails: maximum absolute difference0.0002875328063964844. This is the first failed comparison, and the operation stops there.

Coverage is deliberately limited: the late-stack comparison, every seven-field SAM comparison, selected-slot/presence/mask comparisons, maximum-pointer check, subsequent frame9 and final plain-pass/state comparisons are not reached. The initialized `all_manufactured_low_masks_exact:false` field means pending, not a measured mask mismatch. The earlier unoptimized ten-frame parity pass is separate evidence and does not qualify this implementation.

The same attempt measures0.11142137495s for candidate writes and2.59308941604s for optimized readout including requested tensor audit copies. Their438-position extrapolation plus the earlier full native/scoring costs is1916.660337081s, still above1800s. This is not a measured scientific runtime or a proof that no faster implementation exists.

The algebraic common-token multiplicity identity remains valid in real arithmetic. The observed floating-point failure does not isolate additive-bias rounding, changed reduction/backend behavior or another detail as its sole cause. Do not relax the tolerance or use this unqualified optimization. No accuracy result or hypothesis rejection follows; the research objective is still unmet. Prepare any complete scientific executor against the separately qualified frozen original implementation, while keeping the current operation cap and no-launch decision explicit. No manuscript.

The manifest binds original and public text hashes, with literal private-prefix substitutions only. No model weights, tensor outputs or media are uploaded.
