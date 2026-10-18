# Qualification records: strict HTTP accounting failed; data not admitted

**Primary status: NOT ADMITTED.** Physical HTTP GET count was **3**, exceeding
the authorized maximum of 2. The 302 response body was not metered, so neither
the complete new response-body total nor the strict 1 MiB total-body bound was
verified. Parsing a directory and process exit 0 do not make this qualification
pass. These are AI-assisted runtime/directory records, not a scientific experiment.

## Saved runtime qualification

| Selected path variant | NumPy | SciPy | `scipy.spatial.cKDTree` | OpenCV |
|---|---|---|---|---|
| Existing base interpreter | 2.3.5 | 1.17.0 | exists | `ModuleNotFoundError`, exit 1 |
| Existing runtime-fixed prefix | 2.0.2 | 1.17.0 from base | exists | `ModuleNotFoundError`, exit 1 |

`cv2.USAC_MAGSAC`, `cv2.findHomography` and `cv2.SIFT_create` could not be assessed
because cv2 was absent. No package was installed. No image/matrix function, model,
dataset or geometric benchmark was called. Module entry-file paths and hashes
are retained in [the runtime summary](runtime/summary.json); these are not hashes
of the whole dependency closure. This availability is not geometric performance.

Python 3.12.14 / Linux 6.18.55 / AMD EPYC 9V74, four CPU affinity slots and numeric
threads, 8 GiB AS. Runtime-qualification outer exit/raw wait 0/0, wall 1.211469 s,
peak RSS 53,524 KiB; all six fresh import outcomes, including both cv2 failures,
remain separately recorded. The two qualification directories were not rerun.

## Saved HPatches directory evidence and failed accounting

The pinned author archive declares 1,279,545,572 bytes and whole-file SHA256
`99cf7e1ca167896eb4ca2fe3d903beff6566243a3c0c8e07a13a2596648ad670`.
**The whole file and its hash were not verified.** Only ZIP directory metadata
was requested. Normal TLS and the existing runtime route were used.

| Range operation | Requested inclusive byte range | Status | Metered 206 body |
|---|---|---:|---:|
| Tail | 1279480015–1279545571 | 206 | 65,557 B |
| Missing directory prefix | 1279382286–1279480014 | 206 | 97,729 B |

The two ranges are nonoverlapping and were reused to assemble 163,264 central
directory bytes plus the 22-byte EOCD. Both final responses matched the requested
Content-Range and Content-Length before their bodies were explicitly read.
The first operation followed one HTTPS 302; there were therefore **three physical
GET requests**, not two. Default urllib also read an unmeasured redirect body.
No retry or further request was made after discovery. Only final host
`us.aws.cdn.hf.co` is recorded; no signed URL or query token is published.

The old scientific-asset ceiling record **7,532,690,072 B** remains unchanged.
The verified 206-body increment is **163,286 B**. Their sum **7,532,853,358 B** is
only the old ceiling plus the measured increment, **not** a verified updated
total-response conservative ceiling. The 302 transport-body gap remains UNKNOWN;
complete new acquisition and all-body bounds are unverified. Original
[ledger](hpatches/ledger.json), [result](hpatches/result.json), metadata summary
and their historical labels are preserved, with the correction in
[redirect accounting review](hpatches/redirect-accounting-review.json) and
[current interpretation](admission-status.json). No old receipt was edited to pass.

[The complete directory](hpatches/directory.json) has **1,463 members**: 117 directory
entries and 1,346 regular files, grouped into 116 named sequence containers. It
lists 696 expected image names and 580 expected H names, with no missing expected
names, plus 70 unexpected files across 68 containers: `attribs.txt` ×64,
`readme.txt` ×4, `pt.txt` ×1 and `5.png` ×1. None was opened or repaired. Names,
sizes, offsets, CRC fields, flags and compression methods are metadata only.
Different names do not prove independent scenes. Local headers, CRC/payload
integrity, images and H values were never read or decoded. Offsets can inform a
future member-retrieval design; they do not establish data admission or valid geometry.
The parent supplied official-source/license context; this operation did not reread
license pages or independently establish licensing beyond that supplied context.

Directory-operation outer exit/raw wait was **0/0**, wall **11.560743 s**, peak RSS
**19,008 KiB**, four CPU slots/threads, 8 GiB AS and a 120-second external limit.
The original stderr has one regex `SyntaxWarning`, not an execution exception;
it is separate from the real HTTP count/body-accounting failure. Code and warning
bytes were not patched, and the operation was not retried.

## Preservation, privacy and publication scope

All 47 original files across the two directories remain byte-exact, including
two private `.bin` metadata ranges which are **not published**. All 45 original
text/source/command/log files are represented here. The complete directory JSON
is an exact copy, SHA256
`fcc34ad30d333e4b985c90cec3a528c6e4adbe4b88212d238aaacbe715108170`.
Images, H payloads and any other dataset binary are absent.

[Copy manifest](copy-manifest.json) gives original relative source identity,
SHA256/size, public SHA256/size, exact-copy versus redaction and transformation
scope. `<PROJECT>` and `<PYTHON_RUNTIME>` replace private absolute prefixes.
Changed source files are labeled recipes; a recipe is not the original bytes or
an instruction to repeat the operation. Original argv and source digests remain
bound in receipts. No dialogue metadata or signed URLs are included.
[Artifact hashes](artifact-hashes.json) excludes itself; the copy manifest excludes
both inventories to avoid a hash cycle. Publication-only review evidence is kept
under publication/. Any review failure and correction is preserved there.

This publication makes no data request, package import probe, science call,
installation, schedule change or added-fee operation. It modifies no repository
integration document, old freeze or result. Parent review and a separate justified
freeze would be required for any future experiment; none is authorized by these
records. No method-quality, novelty or paper claim follows.

Publication-only inspection initially exited 1 because its generic path scanner
matched the scanner's own `/workspace/` and `/opt/codex/` detection literals.
The failed source/log/wait record is retained. The revised checker tests the
actual private prefixes instead, while independently verifying every exact
redaction. This tooling correction did not rerun either qualification operation
or alter any original receipt, accounting failure or data-admission status.

A later publication refresh used a wrong relative script path. The aggregate
shell returned 0 because subsequent commands succeeded; the individual refresh
exit was not separately observed. Its diagnostic and scope are retained in
publication/refresh-command-error.json. Running the saved refresh with its
absolute path appended the completed verification evidence before remote
publication. No qualification operation or original output was repeated or changed.
