# Independent parent review of the 0009 public archive

**PASS for public-byte/provenance/reaggregation checks; scientific decision remains NO_GO.** This review performed 15,780 checks using only Python's standard library, Git objects and saved public files. It did not import EVR, invoke the production evaluator, reconstruct geometry scores, replay RNG or run tail dynamic programming.

Verified publication snapshot: `c4c89585d4272f081319ae55e9241900611a3c52`; execution baseline: `cd144b90d337a172a892b56d87cd794524605753`. All 56 original public files match their Git blobs; 55 manifest-bound files match sizes/SHA256. All 47 copy records match public bytes, including the one explicitly disclosed pre-execution-summary derivative. All **515** regular tar members match their individual sizes/SHA256 (**28,906,991 bytes**); the archive was read safely in memory without filesystem extraction. Its 514-file selection lock and supervisor binding also match.

All 15 frozen inputs plus the canonical freeze receipt remain exact across source/freeze/execution commits and current files. All 2,892 pre-existing tracked blobs remain unchanged in the publication commit; all 2,672 prior result/protocol files also match current working bytes. Parent-owned root documentation edits are outside this preservation comparison.

Saved selection IDs and their translation offsets independently reproduce all 256 outcomes, six-method counts, paired tables and every gate:

| Family /64 | raw | msac | cell | excess | ratio | tail |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 64 | 64 | 64 | 64 | 58 | 64 |
| clustered_outlier | 64 | 64 | 64 | 64 | 63 | 64 |
| compact_true | 64 | 64 | 64 | 64 | 51 | 59 |
| sparse_isolated | 63 | 63 | 12 | 63 | 4 | 56 |

Primary net gain is **0**, with the strongest control already 64/64; coverage fails and the 15-comparison paired-harm gate fails. Compact-true harms are 5/5/5/5/4 and sparse-isolated harms 7/7/0/7/0 in the table's five-control order; uniform harms are zero. There are 244 completed cases and 12 unsupported cases. The complete **12 unsupported IDs equal the 12 raw-success/tail-failure IDs**, checked individually against saved failure records and the descriptive lookup. Across 8,192 H slots, 7,985 are completed, 12 unsupported and 195 not attempted after those stops. The one completed-but-wrong tail case is `sparse_isolated-23`: all six methods selected canonical ID13, offset (128,128), error approximately 181.019335984 pixels. It is not a solver failure or a raw-success paired harm.

The original 20,013-byte parent tool-observation file is copied byte-exact as `tool-observations.json`. All eight actual tool records match archived execution JSON values, invocation arguments, stdout/stderr hashes and capture-source hash. Tool duration includes a different boundary from nested wait4 time and is not substituted for it:

| Phase | Actual tool exit | Tool duration ms | Archived wait4 wall seconds |
|---|---:|---:|---:|
| preflight | 0 | 24419 | 24.233076757 |
| frozen-verify | 0 | 234 | 0.141759581 |
| targeted-tests | 0 | 1866 | 1.819699297 |
| checker-self-test | 0 | 234 | 0.139105003 |
| scientific-collection | 0 | 19409 | 19.234951837 |
| evaluation | 0 | 1070 | 0.887383994 |
| independent-review | 0 | 9835 | 9.645058818 |
| postcheck | 0 | 12318 | 12.107300706 |

Frozen collection supervision separately records exit0/raw0, 19.198018780 seconds, 34,656 KiB peak RSS, four cores/threads, 8 GiB address-space limit and no timeout. The archived target-test report records 73 passes, zero skips and all seven Linux integration tests; these tests were not rerun locally. The original independent saved-result checker reports PASS/NO_GO, 1,549,368 checks and max numerical difference 2.842170943040401e-14; this parent review does not relabel those cloud checks as newly executed local checks.

No credential, signed-query or personal-home-path pattern was found in original text artifacts, archive members or the copied observations. Cloud project/interpreter paths and thread/run IDs are intentionally retained as provenance. This limited pattern scan is not a claim of an exhaustive privacy proof.

Boundaries: unavailable private source files and the 40,664 historical cloud-cache files were not rehashed here; their preservation relies on the archived cloud postcheck, not independent local observation. The observations file was supplied from the parent's actual tool retrieval and is matched here, not freshly retrieved. Publication commands were not all directly observable in the oversized publication turn. The failed shell push exit128 and Connector recovery route remain archived; remote `73d235f` verification is separately reported by the parent and is not claimed as a new network check by this reviewer. Historical HTTP accounting and the 0006 stdout gap remain unresolved.

Reproduce from repository root (read-only except the explicitly named report):

```sh
python3 results/0009-stage0/parent-review/check_public.py --report /tmp/0009-public-review.json
```

The final recorded local command exited **0**. During reviewer development, two earlier read-only attempts exited1 because this new checker initially assumed a timeout prefix for `frozen900` and a sorted raw-harm list; those assumptions were corrected against the saved schemas. No original artifact or experiment changed, and no scientific operation was rerun. `parent-command.json` records this review's command/exit boundary; `review.json` contains machine-readable checks and the final reviewer hash.
