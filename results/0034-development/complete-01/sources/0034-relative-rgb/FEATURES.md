# 0034 causal RGB feature construction

Source implementation only until the parent runs the manufactured qualification.
No scientific data, labels, models, fitting or executor is loaded by this module.
`ADMISSION.md` controls the all-fold operation budget and role/label policies.

## API

`features.construct(rgb, logits, native, coarse_rgb) -> (arrays, metadata)`

- `rgb`, `coarse_rgb`: NumPy float32 `[H,W,3]`, finite sRGB in `[0,1]`.
- `logits`: NumPy float32 `[H,W]`, finite, original-resolution admitted native logits.
- `native`: NumPy bool `[H,W]`, the admitted native mask after presence handling.
- All shapes must match. Source/mask/logit parity is the caller's admission
  responsibility; this function never modifies the provided native mask to
  impose a different presence rule.
- RGB256 generation is caller-owned: Torch bilinear original→256²→H,W,
  align-corners false, no antialiasing. No image resize happens here.
- Wrong shapes/dtypes, out-of-range RGB or nonfinite inputs fail the whole call
  before any empty-mask no-op. There is no dropped-invalid-input subset.

`coords` is int64 `[Q,2]` in **(x,y)** order, sorted by `(y,x)`. Every floating
output is finite float32, calculated using float64 accumulation. All floating
rows align with coords. The caller starts from native logits/mask and changes
only these coordinates. No-op sites remain absent from coords but present in
metadata and in the caller's complete-frame denominator.

| Array | Shape | Column order |
|---|---|---|
| `c` | Q×3 | 1, clip(logit/8,−1,1), 4P(1−P) |
| `absolute_prefix` | Q×18 | c; RGB center, left, right, up, down |
| `proto` | Q×5 | c; prototype squared distances F,B divided by 3 |
| `nn` | Q×5 | c; nearest RGB squared distances F,B divided by 3 |
| `ordinary` | Q×17 | c; prototype F,B; NN F,B; Gram ten |
| `sym` | Q×14 | 1, uncertainty, NN sum, absolute NN difference; Gram ten |
| `e` | Q | (NN_B−NN_F)/(NN_B+NN_F+1e−6) |
| `sym_coarse`, `e_coarse` | Q×14, Q | Same formulas using coarse RGB and **identical** support indices |
| `bilateral` | Q×4 | c; normalized RGB/spatial weighted native P minus center P |
| `sample_indices` | min(Q,1024) | int64 indices into coords, not flat image indices |

The root appends the 32 sampled native channels to `absolute_prefix` for 50
absolute-arm inputs. It selects `sym[:,:4]` for signed-no-Gram and performs the
prescribed displaced-Gram roll separately. This module never fits normalization
statistics or reads the native feature tensor.

## Geometry and support

OpenCV `distanceTransform(DIST_L2, DIST_MASK_PRECISE)` measures Euclidean
distance from every nonzero pixel center to the nearest zero pixel center.
Apply it to native and its complement, selecting the distance for each pixel's
class. Native band is distance≤8 on both sides. Empty or full native returns
empty arrays directly; image edges do not supply fictitious opposite pixels.

Eligible queries need all four axial neighbors in bounds. The reference grid
is x/y mod4=0. References must be outside the entire native band and within
Chebyshev distance32 of the query. Each native class is sorted by squared
spatial distance, then row-major image index; take the first32 and require≥4
in each class. No RGB score affects support membership/order. References are
pseudo-labeled by the native mask, so this separation does not guarantee their
true labels are correct.

All arms share support and query eligibility. Q<2 returns empty arrays with the
pre-rule eligible count retained. No point is clipped into the image. Safe
indexing placeholders in vectorized tables are masked out before selection.

For axial offsets left/right/up/down, define \(v_i=RGB(p+\delta_i)-RGB(p)\).
Gram uses \(v_i^\top v_j/3\), ordered
`00,01,02,03,11,12,13,22,23,33`. Multiplication/summation is explicit; no BLAS
matrix multiplication is needed. Pixel intensity is retained. A common color
translation preserves these differences up to float rounding. Swapping native
classes and negating nonzero native logits swaps matching distances and
evidence sign, leaving uncertainty and symmetric reliability features fixed.

The bilateral field uses all in-image offsets in a 7×7 window except center:
`exp(-spatial_squared/8 - RGB_squared/.02)`, normalized by their sum, applied
to native sigmoid probabilities. It does not restrict neighbors to references
or exclude pixels in the query band. Out-of-image offsets contribute nothing.

Uniform sample indices use exact integer round-half-up on
`k*(Q-1)/(m-1)`, k=0…m−1, m=min(Q,1024); endpoints are included. Q<2 is a
whole-frame no-op, so a nonempty construct result always has both endpoints.

## Accounting and bounded computation

Metadata schema `evr0034.features.v1` records the shape, native foreground
count, candidate band count, in-bounds stencil count, eligible pre-Q-rule
count, final Q, sample count and no-op reason. Failure counters distinguish
foreground/background support shortages, their union/intersection, stencil
failure and the Q<2 discard. Support summaries preserve uncapped counts,
capped counts for all checked rows, and capped counts for eligible rows;
empty summaries use null min/max/mean. All metadata is JSON-compatible.

Tiles have at most512 queries. For each of 16 grid phases the source preorders
at most289 spatial offsets. Temporary query/reference tables are O(512×289);
color gathers are at most512×32×3 per class. The bilateral table is at most
512×48. Permanent input/output/EDT storage is O(HW+Q), with no all-image
pairwise distance matrix. The caller owns the four-thread process limit;
the module does not change global thread settings.

Manufactured `test_features.py` covers independent brute-force nonstraight
EDT/support geometry and tie order, reference arithmetic, bilateral border
handling, translation/swap properties, empty/full/insufficient/single-query
no-ops, exact sampling, tile invariance and malformed/extreme inputs. These
tests are source-prepared and have not been run by the implementing agent.
