# Runtime attempt04: annotation compatibility failure in postprocessing

Exact fa4a335 source passed CI39338599063 before fresh installation04 and
qualification04. Installation succeeded. Qualification failed after19.422827125s,
sampled peak2,053,168KiB, during the first MoGe full-infer postprocessing. All
strict model weights and execution settings (Torch4/1, OpenCV1 with parallelism
disabled) passed. The fixed artificial inputs were saved. The model forward and
focal/shift recovery ran, but no complete depth/features/matches were produced.

The original utils3d.torch.helpers references ParamSpec imported via typing.*,
which is absent on Python3.9. This is the recorded NameError, not an image-quality
failure. Attempt05 narrowly prebinds that annotation global from the already
pinned typing_extensions4.16.0, then executes the byte-verified original helper.
The provider's file hash is fixed. No global typing module or upstream file is
changed, and every model calculation/input remains fixed. A failed import removes
its partial sys.modules entry. No alternate data, thresholds or backend is used.

Original JSONs are unchanged copies. Public logs redact local path prefixes only;
private originals and fixed inputs remain hash-bound. This attempt remains failed;
partial model execution is not a complete inference pass or scientific result.
