# 0005: dependency preparation failed; native inference not started

October 18, 2026. Execution main `e168ba830f9f0eff37192ee8745207ff7d3484ef`;
scientific source `a6cb51ee60bdee2650ce6068cb3a2c0255e84931`. The unchanged
[canonical freeze](../../protocols/0005-native-capability-freeze.json) SHA256 is
`3e5aad0b0123affb3991997607c04310bd7d08a38172b8ffbc2c0f29b01f36ef`.
Publication follows execution. Native CPU feasibility remains unknown.

| Actual operation | Exit / raw wait | Wall seconds | Peak RSS KiB |
| --- | --- | --- | --- |
| Seven-file verify-only gate | 0 / 0 | 0.087832 | 12,892 |
| Dependency preparation | 1 / 256 | 36.072080 | 43,916 |
| Installation child | 0 / 0 | 0.526895 | 43,916 |
| Synthetic software tests | 0 / 0 | 1.790227 | 54,888 |
| Saved-evidence review child | 0 / 0 | 0.211155 | 12,256 |

Dependency preparation ran from 18:01:30.592974 to 18:02:06.665219 UTC. Actual
commands, environments, CPU times, source hashes and logs are archived by phase.
The frozen preparer used CPUs 0–3 and an 8 GiB address-space limit; the external
command was bounded by `timeout --kill-after=5s 1800`. RSS is the wait4 high-water
measurement for the observed process/children, not whole-host memory. Parent
scripts have no additional durable outer wait4 observation.

All five pinned sources and three wheels were acquired once with HTTP 200, exact
lengths and SHA256 values. Installation succeeded. Before launching the import
child, the frozen source allowlist rejected
`tapnet/tapnext/tapnext_lru_modules.py.partial`. Later inventory also found empty
source partials for `pscan.py` and VOT-SP `utils.py`. Final source hashes match.
Receipts reported zero stored partial bytes; zero-sized entries are a separate
fact. Their creation/persistence mechanism was not investigated or repaired.
No deletion, rename, retry or gate bypass occurred. **Import check, checkpoint
acquisition and native inference were not started; zero forwards occurred.**
No dataset or GT was decoded. This is an infrastructure failure, not model evidence.

The preparer stopped before writing its approved overlay manifest. The complete
[postfailure inventory](review/postfailure-overlay-manifest.json) records all
**378 installed files** and is expressly **not** preparation/import approval.
Full installation logs, eight asset records, markers and ledgers are archived in
`preparation-dependencies/`. Assets, empty partials and overlay remain in cache.
No upstream source bodies, wheels, installed package contents or weights are
published here. Network-setting values and signed URLs are excluded.

New asset payload: **9,179,574 bytes**. Historical acquisition ceiling:
**4,823,387,101 bytes**, including previously acquired PointOdyssey data even
though currently unavailable. Historical ceiling plus new payload:
**4,832,566,675 bytes**. Current conservative ceiling, including 134,217,728 bytes
of metadata allowance and eight one-byte reservations: **4,966,784,411 bytes**.
Full planned ceiling: **7,499,066,782 bytes**. The 2,532,282,370-byte checkpoint
was not acquired. Git/connector traffic is separate. No additional paid service
was used.

[Saved-evidence review](review/failure-review.json) checks assets, ledger, actual
exits/logs and current/index/HEAD freeze hashes without ML imports or decoding.
All 1,769 pre-existing tracked files matched before state updates, including
**1,595 prior result files** and the three recorded old cache markers. All seven
frozen inputs remain unchanged. The targeted suite passed **44 synthetic tests**;
this is not real inference feasibility. Historical main CI is not this attempt's
runtime evidence. [Copy manifest](copy-manifest.json) binds 35 byte-exact original
metadata/code/log copies. Review source and actual wait receipts are included.

Next: independently review this failure and the acquisition/source-guard
interaction. Any fix or further attempt requires a reviewed prospective revision;
this record grants no retry authority. Do not start later stages from failed
preparation. 0004 remains stopped; 0003 remains deferred. This work is AI-assisted
and makes no accuracy, novelty, deployment improvement or paper claim.
