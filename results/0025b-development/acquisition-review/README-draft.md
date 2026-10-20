# 0025b acquisition: provenance only

The two official clean ICL-NUIM trajectory-2 archives were acquired once under
source/protocol commit `13d4b3259f8d12a1ba4d5d0de9ff56ec6527cf62`. The saved
[CI receipt](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39353648975)
reports successful completion before acquisition began. This review checked
that saved receipt; it did not make another network request.

| Archive | Response-body bytes | Structural members | Paired RGB/depth filenames |
| --- | ---: | ---: | ---: |
| `living_room_traj2_frei_png.tar.gz` | 429,825,445 | 1,766 | 881 |
| `traj2_frei_png.tar.gz` | 301,828,475 | 1,766 | 881 |
| Total | **731,653,920** | 3,532 | 1,762 |

Both responses used the fixed official HTTPS URLs, returned HTTP 200 without a
redirect, and matched the frozen ETag and Content-Length. Both operations are
recorded as completed; the frozen acquisition source permits one GET per archive
and no automatic retry. The inner operation took 57.336 seconds. Its supervisor
completed with exit 0 in 57.409 seconds, sampled peak group RSS 29,616 KiB, no
termination, remaining descendants or cleanup errors, and an empty stderr log.
All five declared thread limits were four. Sampling does not rule out brief
unobserved memory peaks.

Independent metadata review matched both log hashes, all eight outer source and
protocol hashes, and both complete index-file hashes and lengths. Each index has
881 numeric RGB filenames and 881 numeric depth filenames, with matching stems
0 through 880 and no unmatched filenames. This establishes the structural
population; it says nothing about image quality, depth values or model accuracy.

The recorded archive SHA256 values are first-observed hashes, not publisher
digests. The downloader and subsequent consumers verify archive bytes. This
independent review deliberately did **not** rehash or reopen the large archives
during prediction timing; it checked their receipt/index bindings and file sizes.

The full archives transport trajectory and association-file bytes. The index
records their names and sizes; this review neither extracted nor parsed those
contents, and did not read pixels, numeric GT, prediction arrays or model files.
Only the later complete prediction seal and successful outer receipt can admit
the numeric depth diagnostic.

The exact new download charge is **731,653,920 response-body bytes**. The prior
remaining reserve was approximate, so this record does not invent an exact
cumulative total or remaining balance. This is successful data acquisition and
provenance validation, not an accuracy gain, method result or paper admission.

`findings.json` records the checked metadata paths, hashes, counts and limits.
This plaintext is a private publication draft for later integration; no public
repository file was edited by the independent reviewer.
