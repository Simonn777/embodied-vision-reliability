# 0010 preparation: completed assets, failed auxiliary guard

This is infrastructure evidence, **not a model capability or quality result**.
The original probe has never run. The broader research goal remains unmet.

After the [published source freeze and CI](../../docs/0010-capability-freeze-publication-verification.md),
the existing cloud CPU completed these stages. Times are enclosing captured
wait4 wall times; unittest and external tool timings remain separate records.

| Stage | Actual exit | Wall seconds |
|---|---:|---:|
| 54 artificial software tests, zero skips |0|0.205717|
| Four pinned wheels |0|2.086335|
| Offline installation in fresh overlay |0|0.336187|
| Nine imports, versions and origins |0|2.723101|
| Four fixed external assets and layout |0|37.304671|
| Additional non-frozen pre-probe guard |1|0.715672|

The [original guard](cloud/control/before_probe.py) required the download cache
to contain no `.partial` file. It found `ViT-B-32.pt.partial`, 343,605,248 bytes,
SHA256 `97127aac2ade9ec552c1828df711e2c40aeb5e50ddc08a44959734cb0d238e01`.
The failure/source/logs and partial remain unchanged. No probe marker or output
was created; zero image or text forward attempts occurred.

[Read-only diagnosis](cloud/partial-review/review.json) compared all partial
bytes to the complete file prefix: equal. Final and asset-layout checkpoints
share inode 1070329 and both match the complete 353,976,522-byte frozen SHA;
the partial has separate inode 1070339. Three stat observations agree. Root
cause remains **UNKNOWN**: the reported successful link/unlink path does not
explain the retained partial, and prefix equality does not prove its origin.

The fd review recorded 116 attempted entries and no matching holder, but its
source silently continued on `fd.stat()` errors. Its zero-error counter covers
other process-inspection paths, **not every fd stat**. This is an explicit gap;
the correction requires renewed inspection with success/race/permission/error
counts before first launch. Hidden platform processes and prior exited writers
remain outside a visible `/proc` snapshot. No repair or cleanup occurred.

The original inventory also retains three unexplained temporary copies:
`.EuroSAT.zip.91Kpzc` (24,117,248 bytes), `.ViT-B-32.pt.partial.hvJJcZ`
(197,132,288 bytes), and `.ViT-B-32.pt.partial.hvJJcZ.dBQxjR` (185,073,664 bytes).
They are outside the fixed asset layout and never model inputs. All are retained;
their existence alone establishes neither another HTTP transfer nor its absence.

The BPE has measured SHA256
`924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a`
and its pinned Git blob identity. ZIP/checkpoint and all other fixed asset
identities match. No image was decoded or model loaded.

Wheel bodies total 1,234,470 bytes; asset/redirect bodies 449,616,806 bytes.
With prior direct charges 5,235,758, the total is **456,087,034 bytes**, below
512 MiB. Scientific ZIP/checkpoint payload adds 448,257,089 bytes to the separate
historical asset bound. Historical all-HTTP/browser totals remain UNKNOWN.
No new paid service, API, GPU or purchase was used.

The [65-file copy manifest](cloud/copy-manifest.json) verifies 752,530 original
bytes exported from cloud. It preserves original logs/events and the exit 1.
The parent retrieved [eight captured stage returns and one direct inspection](parent-tool-observations.json),
plus the [later diagnosis's actual command return](diagnostic-parent-tool-observation.json).
The unrelated 447,974-byte pre-execution hash inventory was not exported; its
hash and this boundary remain. Historical 3,046-file/45-marker preservation is
a cloud postcheck claim, not an independent local rehash of private caches.
Images, weights, wheels and overlays stay in cloud and are not rehosted here.

The [prospective correction](../../protocols/0010-cached-probe-and-solver-admission.md)
retains the failed guard and replaces only its extra directory-emptiness
condition with stable complete-file and no-prior-probe checks. Its separate
published freeze precedes first inference. No original scientific code, sample,
threshold or failed historical result is changed.
