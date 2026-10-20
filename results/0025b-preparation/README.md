# 0025b source preparation, not scientific results

The [protocol](../../protocols/0025b-rs-depth-semantics.md) defines an independent
rendered-depth qualification before any source payload or network inference.
This directory records source-only HEAD metadata and artificial checks. It
contains no learned-model outputs, GT diagnostic results or accuracy gain.

The integrated final-source artificial suite passed **56/56** tests on its first
run (0.664 s unittest / 0.834 s outer). Sources were unchanged during the run;
the receipt binds all 13 Python files and the protocol. Separate reviewed units
cover acquisition (10), provenance/supervision (7), geometry (8), prediction
(11), evaluation (13), and independently reconstructed arithmetic (7).

`head-https.json` preserves the two original official HTTPS HEAD responses:
200, no redirects, exact sizes/ETags, zero response-body bytes. A later single
acquisition is limited to 731,653,920 archive bytes. The first observed archive
SHA will be trust on first use, not a publisher digest. Unsupported archive
features or changed responses fail without automatic retries or substitution.

Artificial checks use invented arrays/PNGs, fake HTTP responses and fake model
backends. Each first run is retained; later checks follow documented source
review changes, not benchmark outcomes. Before data, review:

- made raw depth validation structural so affine correction can retain finite
  negative values, while nonfinite values stay recorded;
- retained finite rank/scale diagnostics in unavailable fits;
- completed source/outer-receipt and archive/header identity verification;
- recorded the full paired filename inventory and all encoded RGB hashes before
  RGB decoding;
- required all raw predictions and clean outer receipts before any GT sample,
  then all selected GT hashes/headers before the first numeric GT decode.

Original private receipts and source snapshots remain unchanged. Published
receipts redact local execution path prefixes, with original and public hashes
recorded in `publication-bindings.json`. Any such redaction is publication
formatting, not a scientific execution. Exact source CI is required before
the first acquisition; passing software checks does not satisfy the user’s
research objective or permit a manuscript.

The local pre-integration repository regression run had 779 passes, 8 Linux-only
skips and 7 failures in historical path-binding tests (53.54 s). macOS's default
temporary directory traverses the `/var` symlink, which those existing safety
checks reject. Without changing repository code, rerunning the two affected
test files with a resolved workspace temporary directory passed all 36 tests
(2.05 s pytest / 2.164 s outer). `local-regression-02` preserves this targeted
rerun and records the original terminal result. Linux CI remains required.
