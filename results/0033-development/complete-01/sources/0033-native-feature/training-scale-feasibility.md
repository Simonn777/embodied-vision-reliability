# Broader TRAIN data for a future supervised control

2026-10-20. **Metadata/source feasibility only; no allocation or scientific execution.** This note does not reopen the fixed 0033 recipe or propose “more training data” as a new method. No network requests, media/annotation decoding, model calls, training, acquisition specification, or budget hold were made. Candidate arithmetic used existing official-TRAIN inventory records; the protected final cohort was not selected or inspected for content, labels, or performance.

## What the four-parent result does and does not establish

The parent reports visual-head TRAIN J&F +0.6513 pp and DEV −0.2785 pp, with cat-girl −1.573 pp and upside-down +1.016 pp. The exact recipe is closed. Its four TRAIN recordings supply only 228 noninitial frames: 24 dog-agility, 37 planes-water, 82 dogs-scale and 85 dog-gooses. The two longest recordings supply **73.25% of the frame-weighted training objective**. Adjacent pixels and frames are not independent recording-level examples.

This is a valid small supervised baseline and an informative failure, but not a sufficiently broad benchmark to claim that native features cannot support output refinement or that this is the strongest achievable supervised control. TRAIN/DEV sign reversal is consistent with limited coverage or overfitting; it does **not identify the cause**. Objective/metric mismatch, resolution, target conditioning and the chosen head can also explain it. Broader training has no guaranteed benefit. The two exposed DEV recordings are also too few to establish broad generalization.

For a future **materially distinct method**, a larger fixed TRAIN cohort can make its learned controls more credible. All compared learned arms must receive that same supervision and declared training budget. Changing the cohort alone would remain control development, not an innovation claim. The existing four TRAIN and two DEV assignments remain unchanged; none becomes untouched again.

## Existing source and eligibility boundary

The cached official archive is [DAVIS-2017-trainval-480p.zip](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip), previously verified at **832,766,765 B**, ETag `"31a3032d-54c6d0af8b4dc"`. Its official lists contain 60 TRAIN and 30 validation sequences. These are prior metadata observations, not a fresh availability check. Cached terms specify CC BY-NC 4.0, retaining external-source conditions.

Use the already recorded conservative parent rule: canonicalize explicit author `SOURCES.md` YouTube IDs, group by original video ID, and exclude a TRAIN parent if it also appears in validation. Other author-owned or webpage-only entries lack a verified original-video identifier and are outside this conservative expansion. The old source audit leaves 22 eligible TRAIN parents; removing the six exposed recordings leaves **16 additional parents**, each with one clip. In particular, the TRAIN aliases `drift-turn` and `motocross-bumps` remain excluded because their original videos overlap protected validation footage.

An author-linked YouTube ID is a concrete original-recording grouping, not proof of independent actors, locations, or absence of reuploads. The protected 12 names and their source aliases must stay excluded in any future admission. No protected member payload, label value or outcome was opened for this note. Broader DAVIS TRAIN is also not an independent validation dataset or evidence of driving-domain performance.

All 16 candidate records have paired, consecutive JPEG/PNG stems `00000` through `N−1`. These are compressed-member inventory totals, not decoded sizes or observed object counts:

| Official TRAIN sequence | Frames | RGB compressed B | Masks compressed B | Combined B | Author parent ID |
|---|---:|---:|---:|---:|---|
| boxing-fisheye | 87 | 12,721,034 | 250,518 | 12,971,552 | pq4DN-V62Bk |
| classic-car | 63 | 8,044,596 | 220,381 | 8,264,977 | EuX_NCNHYK0 |
| crossing | 52 | 9,827,844 | 124,864 | 9,952,708 | YzpkoPGclas |
| disc-jockey | 76 | 5,471,406 | 477,573 | 5,948,979 | ZBOLTmudhEw |
| drone | 91 | 9,566,114 | 422,346 | 9,988,460 | KCpfQNTatMQ |
| kid-football | 68 | 7,660,896 | 226,700 | 7,887,596 | O0It_aMx9pg |
| lady-running | 65 | 7,339,873 | 170,780 | 7,510,653 | dOFv-Bl1IsQ |
| lindy-hop | 73 | 9,384,040 | 474,170 | 9,858,210 | w7wjhGr1mBw |
| miami-surf | 70 | 5,057,800 | 91,979 | 5,149,779 | h9-RZkEt3b4 |
| night-race | 46 | 3,724,846 | 77,253 | 3,802,099 | v5MESKmFtQs |
| schoolgirls | 80 | 18,786,811 | 363,786 | 19,150,597 | dGzO3E2GYMQ |
| scooter-board | 91 | 9,607,678 | 155,017 | 9,762,695 | xlnhNfnuMc8 |
| stunt | 71 | 8,702,454 | 122,583 | 8,825,037 | SRrqPqQL_Z8 |
| tuk-tuk | 59 | 8,303,939 | 392,595 | 8,696,534 | 4Fz4O1hGS8w |
| varanus-cage | 67 | 11,195,360 | 100,635 | 11,295,995 | oCZZE3i7cEI |
| walking | 72 | 8,621,996 | 282,720 | 8,904,716 | 0dq1ArzsyIY |
| **Total** | **1,131** | **144,016,687** | **3,953,900** | **147,970,587** | **16 original IDs** |

## Prospective choice and transfer feasibility

**Preferred coverage option:** use all 16 eligible additional parents, complete recordings and all paired frames. This removes any freedom to favor classes resembling current DEV errors. It yields 20 TRAIN parents and **1,343 noninitial TRAIN frames**. This is a resource option, not a selected/admitted acquisition.

If a smaller fixed study is needed, reuse the pre-existing 0029 hash rule rather than inventing a new salt after DEV outcomes:

`SHA256(b"evr0029:video-cohort:2026-10-20\0" + b"davis-dev\0" + parent_id.encode("utf-8"))`, ties by parent ID.

After removing the six already exposed parents, its first eight are **walking, classic-car, tuk-tuk, scooter-board, kid-football, varanus-cage, miami-surf, disc-jockey**. This prefix contains 566 new frames, giving 12 TRAIN parents and 786 noninitial TRAIN frames. A future study must fix either its full eligible pool or its prefix length before reading new pixels/labels or fitting; there is no outcome-dependent extension/replacement. The prefix is not a semantic “cat/human” choice.

For every requested JPEG/PNG, the reviewed transport reads a 30-byte local header, filename, at most 4,096 extra bytes, then the declared compressed body. Applying this bound to the exact candidate member names gives:

| Resource option | Compressed payload B | All local-header/name/extra upper B | Successful transfer-body upper B | Hypothetical balance afterward |
|---|---:|---:|---:|---:|
| Eight-parent original-hash prefix | 65,911,271 | 4,719,320 | **70,630,591** | 1,212,960,138 |
| All 16 eligible new parents | 147,970,587 | 9,428,867 | **157,399,454** | 1,126,191,275 |

Balances subtract only these illustrative bounds from the parent-supplied conservative **1,283,590,729 B**; they are **not ledger updates or holds**. Error bodies, any explicit retries and future metadata must also be capped if admitted. Even a prospective 160 MB cap for the all-16 option leaves 1,123,590,729 B before other future allocations. Existing conservative reservations are not refunded, and remaining validation/runtime needs still compete for that balance.

The existing `work/0029-preparation/acquire_davis.py` is hard-bound to the old six recordings/440 members, and `work/0030-memory-content/data-acquisition/acquire_masks.py` to the old masks. **Neither currently admits this new cohort.** Only their small reviewed transport pattern is reusable: exact 206/Content-Range and ETag; local/central header consistency; bounded extra fields and decompression; exact names, sizes, CRC and retained SHA256; no full-archive fallback. Any future caller needs its own reviewed manifest/source freeze. No such caller or specification was created here.

## Compute: measured anchors versus estimates

Measured existing receipts: the complete 0033 two-arm fit took **115.0001 s**, peak **2,428,400 KiB**; the existing six-video native cache took **458.3008 s** for 386 frames, peak **4,672,848 KiB**. The cache operation also included its declared extra target work, so its per-frame rate is only a rough planning anchor.

Linear extrapolation of those costs—not measurements of a new cohort—suggests:

| Option | Added native-cache estimate | Two-arm 12-epoch fit estimate |
|---|---:|---:|
| Eight new parents / 786 total noninitial TRAIN frames | ~672 s | ~396 s |
| Sixteen new parents / 1,343 total noninitial TRAIN frames | ~1,343 s | ~677 s |

These do not establish a 1,800 s complete-operation guarantee: loading, serialization, frame widths, active/no-op proportions, I/O, hashes and future method costs can differ. Fixed per-recording complete cache operations can bound peak memory, but must be declared before execution; no truncation after a timeout.

More importantly, the current fit retains all float32 finest features in memory. At `32×256×256×4` bytes per frame, the eight-parent option needs **6.140625 GiB for features alone**, and the all-16 option **10.4921875 GiB**. Logits, original-resolution targets, Python/Torch state and working tensors are additional. Therefore the existing hold-all implementation is **not qualified** for either expansion under 8 GiB; all-16 is impossible with that exact retention policy. Streaming/memory-mapped compact features or a bounded data cache is needed. This changes I/O and must be timed honestly; cached features consume disk, not acquisition bytes.

## Decision

**Broader supervised controls are affordable in transfer bytes and plausible in CPU work, with a concrete memory-management prerequisite.** The all-16 pool is the clearest prospective coverage option; the existing-hash eight-parent prefix is a smaller resource option. Neither is admitted here. A future distinct method should declare its scientific information advantage, same-input learned controls and complete resource plan first, then fix one training cohort and all comparison budgets. More training data cannot turn the closed 0033 result into a gain, establish novelty, or erase prior DEV exposure.

## Local metadata/source fingerprints

All paths below are relative to the current project root; no new HTTP bytes were transferred.

- `work/0029-screening/video-cohort-plan.json`: `f4962c6c8b3ca0806ee4bb751bc6bb6ad74de97771aed26ebd33106b078fc5d1`.
- `work/0029-screening/video-cohort-metadata/davis-members.json`: `32ad50e44e7c2c2fc325d81d42c56fb995492f3438268116fe1f101ccd19c5f0`.
- Same directory `davis-central.bin`: `97277af1eff6ad51f60bcad90139e82a7b46f3d0766eee2e8b83d62fa0728d9b`.
- `davis-train-txt.txt`: `d1276a73bdd6b37fd75266ae940de80347ce158fbbfa875151b61a3ba2249a8b`; `davis-SOURCES-md.txt`: `889de97e611e362e4d4fa677ebcf5c829cf36ef253aae4f0cd6dfdfa3fdcbec1`.
- `proposed-selection-rule.txt`: `7de45f7e4aa2d23024f14aa7727e4301a808cff5ea9a3083743c03d955941796`.
- `work/0029-preparation/acquire_davis.py`: `4e2897ee0b1dfea25b3388805f019e155be53986d155f71c369101293919ee69`.
- `work/0030-memory-content/data-acquisition/acquire_masks.py`: `bd99aefde39e30cc0ebf7dee221278bd63640082f300d51bf477e758f11b1769`.
- Fit cost receipt `work/0033-native-feature/fit-supervision-01/receipt.json`: `26191abf8956e0aaf4a2376320f41fee1ba407c1b14e89dd148c7aca75a8d4c1`.
- Cache cost receipt `work/0030-memory-content/campaign/cache-supervision-01/receipt.json`: `d7d9d59d664b55c2ca897f3c8c112133f54ff6ee894f5e17d98d85a13c81aceb`.
