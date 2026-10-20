# Saved native qualification audit: corrected premise, preserved first failure

The separate second audit passed **18,197 checks**, with zero failures. Its supervisor completed with exit 0 in **1.9372092499979772 s**, peak sampled group RSS **123,920 KiB**, no remaining descendants and no cleanup errors. This is a saved-artifact audit of the manufactured qualification, not a benchmark or segmentation-accuracy result.

It verified the frozen source/installation/acquisition/outer-log chain, all **17,602 installed inventory entries**, exact **26-distribution** closure, nine encoded manufactured inputs, all 24 saved NPZ files and their individual array hashes. Saved full/no-op, reversed deletion order, decoder selection, and subsequent frame-8 logits have exact bitwise parity. The native video bank has three masks, chooses index 2 by its native predicted IoU, and the saved LightGlue output contains **69 mutual matches**. The original qualification remains its original clean execution under commit `37b54982bcc3b22b56a5817b6881f22489eba8d5`.

## Why the first audit failed

`../native-result-audit-01` is unchanged and remains **FAILED: one failure in 18,162 checks**. Its extra assertion required the initial-only memory's pointer positional encoding to be bitwise equal to that pointer's encoding in the seven-pointer history. Spatial features, spatial positions and pointer features were equal; only the pointer temporal-position bytes differed.

The native source computes sine positional encoding and then a learned float32 linear projection. The conditioning frame's normalized distance is `7/8` in both histories. The full history projects seven rows `[7,1,2,3,4,5,6]/8`; the initial-only history projects one row `[7]/8`. Equality of their mathematical first-row input does not require bitwise equality of differently shaped floating-point matrix operations.

A separately authorized, single **isolated arithmetic diagnostic** in `../pointer-arithmetic-01` reproduced **both saved hashes exactly**, using the qualified Torch 2.8.0 CPU runtime at 4 intra-op/1 inter-op threads, the original byte-verified sine function, and only checkpoint tensors `obj_ptr_tpos_proj.weight` and `.bias`. No SAM2 model was constructed, and no image, predictor, or inference was run. The sine-encoding first rows were bitwise equal. The projected first rows differed by at most **9.5367431640625e-07**; 57 of 64 channels differed, repeated across four pointer tokens.

- Full-history position: `291e63ff793ffad08a0bc6c91bdb28088d40195c6315430411ae92cfc34bf050`
- Initial-only position: `979cde68454fcc367a63e930a49f38ad4345e421843a17bce063835c4f7ed8d6`

The diagnostic completed exit 0 in **1.4460324160172604 s**, peak sampled group RSS **368,496 KiB**, with no forbidden audit events or warnings. This localizes the observed discrepancy to the batch-shaped linear projection; it does not claim to identify a particular internal BLAS kernel.

The second audit replaces only the unsupported cross-batch position-equality premise with the **exact per-batch hash proof**. It still demands exact retained feature/spatial-position hashes, exact saved logits, exact reversed-order traces, and binds the first failed report and the separate diagnostic. **No tolerance was introduced or changed.** Native qualification was not repeated.

## Binding and scope

- First failed audit result: `436b84a089e7b5473c3cd8c4fbb875ca4d693693448a6a93c4948c736a13de73`
- Isolated arithmetic source: `6eed74cb1a3f214317f1c6384634152b0fb74a3af3a4dbe0979c26f16c48214a`
- Isolated arithmetic result: `eda51da93d5ead35c613ad8cf6bf515fda50247b2efcd2113bd4a95aa3278c8e`
- Second audit source: `8b20bbb878b59daedf4a55051e7f6df52d10b310fd2a88f2b0c02b92eb95402c`
- Second audit result: `a98981da2d3ca7daf59a3ccf954c35614a820557a918ad56041da84f0b900650`

The first and second audits use saved data and NumPy without importing models or Torch. The arithmetic diagnostic is separately disclosed above. Original in-memory attention tensors and complete state values were not exported: their recorded hashes and unchanged-state assertions remain source-bound execution evidence. Saved logits are independently compared. JPEGs are rehashed without decoding; CI is checked against the preserved successful response, not a fresh network query. This is an independent implementation review within the same agent team, not independent-team replication.
