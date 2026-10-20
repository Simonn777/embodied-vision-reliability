# 0030 development J/F source contract

Prepared 2026-10-20 initially as **source only, without evaluator/test execution or scientific array reads**. Subsequent authorized manufactured testing is recorded below. This helper does not grant annotation access or replace campaign admission/sealing. Its caller must admit the complete declared cohort before use. No new metric gate is introduced.

The official [DAVIS2017 evaluator](https://github.com/davisvideochallenge/davis2017-evaluation/blob/ac7c43fca936f9722837b7fbd337d284ba37004b/davis2017/evaluation.py#L102) uses `[1:-1]` for semi-supervised evaluation: **exclude first and last**. All frame scores, including endpoints, remain in our JSON; only the interior contributes to sequence means.

The [official metric implementation](https://github.com/davisvideochallenge/davis2017-evaluation/blob/ac7c43fca936f9722837b7fbd337d284ba37004b/davis2017/metrics.py) defines boundary radius `ceil(0.008*hypot(H,W))`, a Euclidean disk and its same-resolution boundary convention. `evaluate_jf.py` reimplements these operations using installed NumPy and OpenCV, replacing only the `skimage.disk` construction with its explicit integer-coordinate disk. J handles empty union before division to avoid a spurious warning. Both-empty boundary maps have F=1; one-empty has F=0. This includes the official full-frame-mask edge behavior.

The official [dataset reader](https://github.com/davisvideochallenge/davis2017-evaluation/blob/ac7c43fca936f9722837b7fbd337d284ba37004b/davis2017/davis.py#L81) maps label 255 to background; the semi-supervised evaluator passes no void mask. This implementation deliberately matches that path. It does **not** inherit old 0029's custom void-excluding metric.

`evaluate_sequence(spec)` takes:

```text
sequence, parent_id, shape_hw=[H,W], frame_ids=["00000",...],
object_mode="target", target_id=<initial selected ID>
  OR object_mode="all", object_ids=[explicit sorted IDs],
predictions=[{path,bytes,sha256}, ...],
annotations=[{path,bytes,sha256}, ...].
```

Each prediction NPZ has exactly `masks: bool[O,H,W]` and `object_ids: integer[O]`. Every frame is required, at original 480p dimensions, with no resizing, threshold choice, object matching, silent union or successful-subset evaluation. All encoded hashes are checked before the first label decode and after scoring. Multiple object masks must be disjoint; their label decision belongs upstream. `all` checks declared IDs against initial supervision and rejects later undeclared IDs. `target` consumes the explicit initial ID rather than assuming ID1 or selecting a target from later GT.

Results contain per-object, per-frame J/F/JF and endpoint inclusion flags. Sequence means equally weight its declared objects. `aggregate(reports)` equally weights sequences within original parent, then parents; it separately reports equal-object means so this development aggregation is not confused with official equal-object leaderboard aggregation. Values are fractions in [0,1], multiplied by 100 for points. Even `all` mode is labeled development, not an official benchmark result. No union mode is provided.

The pilot should retain old native target selection: **smallest positive initial ID excluding 255**. The old `choose_target`, `native_video` and `foreground_j` use `labels == target_id`; old 0029 was single-target fixed-query J, **not foreground union**. Existing attempt02 receipt metadata reports target ID1 in all six sequences. Receipt NPZ shape metadata gives cat-girl 480×911; the other five 480×854. Thus boundary radii are respectively 9 and 8 pixels. Total object counts are not established by those receipts; this source preparation did not inspect annotation values. Current pilot results must be called **object-subset development J&F**.

Runtime inventory lists NumPy 2.2.6, Pillow 11.3.0 and opencv-python 4.11.0.86. No SciPy/skimage installation is needed. The CLI sets OpenCV serial mode; it accepts `--manifest` (JSON with a `sequences` list) and exclusive `--output`. Seven manufactured tests cover endpoint handling, target semantics, empty/border masks, exact radius, hash-before-GT, overlap rejection and parent weighting.

Authorized first test execution: **7/7 PASS**, 0.066 s unittest time, 0.314406917 s supervised wall time, peak sampled RSS 47,952 KiB. Existing qualified CPython runtime, `-B -W error`, four-thread environment, 180 s bound. Exit 0, no termination, no signals, no remaining descendants or cleanup errors. Source snapshots, spec and original logs/receipt are in `tests-01/`; production/test source hashes remained `c9dd8f96…fcc40` / `ad15c112…d2af`. Only manufactured masks and temporary PNG/NPZ files were used; no native/model/real-data evaluation ran.

## Source-access accounting

Targeted local source searches found no cached official evaluator. Two official GitHub source pages were read using the web tool (response bodies not metered). Before the parent's subsequent page-only restriction arrived, five direct commit/source/license responses totaling **27,585 B** were fetched and retained privately under `official-source/`, with full URL/hash/byte receipt `http.json`; this was reported immediately. No further requests followed. These private snapshots are provenance references, not runtime imports or vendored public dependencies. The implementation and tests require no network.
