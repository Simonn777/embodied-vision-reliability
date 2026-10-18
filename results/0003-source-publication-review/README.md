# Independent review of source history, saved-R analysis and geometry freeze

Approved input: `84dd6aa71027ab3d5c18e3d3fdef5a59ccdeb035`.
Separate receipt-only commit: `48ff18ff8e54be8a57bda5741a008b04582e3afc`.
Canonical receipt SHA256:
`fe7f949bb227ff0824fdee71cdabd29129ac4ec09fe39d3858466c0bc937ee44`.
The [review guide](../../docs/0003-source-history-independent-review.md) explains
scope, actual findings and limitations. All review work was AI-assisted.

`source-stageA-review.json` and `geometry-initial-review.json` preserve historical
reviews, including the initializer binding blocker. `geometry-final-review.json`
approves the corrected candidate and binds all 63 new result artifacts since the
baseline. `geometry-freeze-delta-review.json` separately confirms the receipt-only
commit and the actual local gate record. `initializer-probe-summary.json` preserves the bounded synthetic
reproduction without personal paths; it does not fabricate a raw execution receipt.

The half-output verifier checks already published JSON only. Its source, launcher,
original input, claimed output, independent output and actual local execution
receipt are byte-identical copies. The historical launcher uses temporary paths;
a portable saved-input-only check from the repository root is:

```sh
python3 -I results/0003-source-publication-review/half-output-verifier.py --original-input results/0003-source-publication-review/half-original-input.json --claimed-output results/0003-source-publication-review/half-claimed-output.json
```

The verifier confirmed **INCONCLUSIVE / 8 of 9 cells**, not a positive compatibility
certificate. It does not regenerate a proposal or read any dataset. The local
publication gate files record only `verify_published()`, after receipt publication;
no runner main, cache, dataset or projection was invoked. Historical commands and
elapsed times remain in the receipts. An observed child exit does not imply an
additional observed outer supervisor exit. Sources/output hashes and original
candidate identifiers are retained; later publication does not rewrite them.

## Data attribution and terms

PointOdyssey v1.2 by Yang Zheng, Adam W. Harley, Bokui Shen, Gordon Wetzstein and
Leonidas J. Guibas. [Source](https://huggingface.co/datasets/aharley/pointodyssey),
revision `4a2ec9492ca732dc3611adbb2a94248cc2bbe3eb`.
The already published original R bytes are unchanged; hypothetical matrix/cell
statistics are derived data, never a replacement camera or historical proof.
These data and derived numeric outputs/reviews retain
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), with further
source/excerpt hashes in `data-terms.json` and the review records. They are not
covered by repository MIT. Original review software and prose use repository terms.
No extra original dataset values were obtained or published for this review.

`bundle-manifest.json` covers all other bundle files and excludes only itself.
