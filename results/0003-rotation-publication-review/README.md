# Independent publication review of the recorded rotation failure

Reviewed candidate: `bb5fe92fe38835266d660f63b159583151e64695`.
The cloud diagnostic was already finished; this bundle adds no cache access or
experiment. See the [review guide](../../docs/0003-rotation-independent-review.md).

`parent-review.json` audits all 19 published Stage B artifacts, frozen Git inputs,
run receipts, accounting and historical preservation. `numeric-review.json` audits
only the published frame-zero excerpt using a separately written stdlib checker.
`numeric-execution.json` records the **local macOS saved-record check**, not a cloud
experiment: 0.081646 seconds, child return code 0. Cloud execution measurements are
in the original Stage B evidence and remain unchanged.

The two Python files are byte-identical copies of the executed original review
sources. The recorded historical command uses temporary filenames; a portable,
saved-input-only check from the repository root is:

```sh
python3 -I results/0003-rotation-publication-review/independent_math.py --input results/0003-rotation-publication-review/numeric-input.json
```

No archive or model is needed. `comparison_launcher.py` preserves the original
launcher, including its temporary output paths; it reads published Git blobs and
expects the original checker path. The execution receipt observes the checker
child, not an additional outer supervisor. Artifact hashes in `bundle-manifest.json`
cover every other file here; the manifest deliberately excludes itself.

## Attribution and terms

PointOdyssey v1.2, Yang Zheng, Adam W. Harley, Bokui Shen, Gordon Wetzstein and
Leonidas J. Guibas. [Dataset source](https://huggingface.co/datasets/aharley/pointodyssey),
revision `4a2ec9492ca732dc3611adbb2a94248cc2bbe3eb`, selected member
`sample/r4_new_f/anno.npz`; the original NPZ and excerpt hashes remain in the
source records. The 25 original values are unchanged; representations and numeric
checks are derived from the published excerpt, with no extra samples.

The data and derived numeric outputs in this bundle retain
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), as detailed in
`data-terms.json`; they are **not covered by the repository MIT license**. Original
review software and prose use the repository terms. All review work was AI-assisted.
