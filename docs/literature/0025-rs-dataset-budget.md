# Smaller GrandTour HDR resource screen — no allocation

Official HF revision `daf91c5e2b6db564822a35cc58222d73f8e75bab`; only directory file sizes/hashes and the [official mission table](https://grand-tour.leggedrobotics.com/data/missions.json) were read. All 51 mission image directories were attempted, using the first already-cached directory and 50 new bounded requests. 48 returned directories contain 144 HDR front/left/right TAR entries. Three returned 404 and remain recorded: `2024-10-29-09-53-44`, `2024-11-04-16-52-38`, `2024-11-15-15-07-36`. No error was silently excluded or retried.

All 144 sizes/LFS hashes and official labels are retained in the private source directory as `annotated-candidates.json`; original HTTP responses and `all-hdr-candidates.json` are also preserved. The first lexicographic mission's front/left/right sizes are 1,051,596,800 / 1,101,557,760 / 1,129,512,960 B, so changing camera there cannot meet <700 MB.

**Important correction to the earlier feasibility example:** the official mission table explicitly warns ETH-1/ETH-2/ETH-3 HDR timestamps use GMSL2 arrival times instead of hardware timestamps. The first-lex ETH-1 example therefore fails the intended precise-RS-timing admission; its byte calculation remains valid but is not a suitable scientific selection. This was found from author metadata, not outcomes. Generic GrandTour timing source code cannot erase an explicit per-mission exception.

## All under-700 MB candidates

| Mission / official name | Camera | Image TAR B | TC ref listed | Conservative bundle B* | + TUM + 430 MB reserve B |
|---|---|---:|---|---:|---:|
| 2024-11-02-21-12-51 / ICE-1 | hdr_front | 288,716,800 | True | 313,692,873 | 1,958,689,353 |
| 2024-11-02-17-43-10 / SPX-3 | hdr_front | 343,480,320 | True | 367,227,640 | 2,012,224,120 |
| 2024-11-02-21-12-51 / ICE-1 | hdr_left | 469,268,480 | True | 494,244,553 | 2,139,241,033 |
| 2024-11-02-21-12-51 / ICE-1 | hdr_right | 486,522,880 | True | 511,498,953 | 2,156,495,433 |
| 2024-11-18-17-31-36 / ARC-7 | hdr_front | 505,835,520 | False | not established | not established |
| 2024-11-03-07-52-45 / SNOW-1 | hdr_front | 531,015,680 | True | 552,766,188 | 2,197,762,668 |
| 2024-11-02-17-43-10 / SPX-3 | hdr_left | 537,118,720 | True | 560,866,040 | 2,205,862,520 |
| 2024-11-02-17-43-10 / SPX-3 | hdr_right | 573,962,240 | True | 597,709,560 | 2,242,706,040 |
| 2024-11-03-07-52-45 / SNOW-1 | hdr_right | 606,709,760 | True | 628,460,268 | 2,273,456,748 |
| 2024-11-11-16-14-23 / HAUS-1 | hdr_front | 613,560,320 | False | not established | not established |
| 2024-11-18-17-13-09 / ARC-6 | hdr_front | 619,939,840 | False | not established | not established |
| 2024-11-03-08-17-23 / SNOW-3 | hdr_front | 659,496,960 | True | 683,571,962 | 2,328,568,442 |
| 2024-11-03-07-52-45 / SNOW-1 | hdr_left | 668,743,680 | True | 690,494,188 | 2,335,490,668 |
| 2024-12-09-09-41-46 / CON-2 | hdr_front | 681,840,640 | False | not established | not established |

*Conservative bundle includes one image TAR, its author mask TAR if listed, matching frame-data TAR, `cpt7_ie_tc_odometry`, ADIS IMU, TF and every metadata YAML in that mission. No masks or sensor metadata YAML/calibration values were opened. TUM seq2's verified 1,214,996,480 B is a budget example, not an allocation. The 430,000,000 B physical-falsifier reserve is deliberately rounded upward from the parent's approximately 429.8 MB option; exact acquisition would bind its actual HEAD size. It leaves ample extra reserve for the smaller bundles. `not established` means the named independent TC reference was absent from the inspected data directory, not that every possible reference is scientifically impossible.

## Scientific conditions cannot be selected by byte count

- **ICE-1**, the smallest (288,716,800 B image; Train): ice palace, GNSS label says signals penetrated ice. Author notes slippery motion, tiny supported steps followed by pushing with locked drives. TC-reference presence does not prove good reference quality or adequate image/row-motion support. Its unusual dynamics are a prospective condition, not a reason to cherry-pick successful portions.
- **SPX-3** (343,480,320 B front; Train): Jungfraujoch Sphinx, outdoor industrial night, people; limited-radius walk with rooftop total station. GNSS marked working. It has named TC reference and no explicit HDR-timestamp warning in the mission table, but darkness, motion/texture sufficiency, and reference quality have not been examined.
- **SNOW-1/SNOW-3** (Train): outdoor mountain/snow/sunlight with GNSS marked working. Metadata warns the total station may have unlevelled; SNOW-3 also lost total-station recording after 66 seconds and notes a possible crash. We have only established the separate IE-TC file exists; no claim it is unaffected or accurate is made.
- **HAUS-1/ARC-6/CON-2** have no named IE-TC file in the inspected HF data directories; ARC-7 is officially **Test** and also lacks it. The four are not substitutes for the verified reference contract merely because they are smaller. Do not use a mission labeled Test for development by default.

**Resource verdict:** the budget bottleneck is resolved conditionally. For example, a complete SPX-3 front bundle + the TUM whole-sequence example + 430 MB falsifier reserve fits well below 2.43 GB; ICE-1 is smaller still. This does not allocate either mission, approve scientific quality or resolve the earlier HF execution/10–30 Hz provenance gap. A future source-backed selection must be frozen before observing payloads/outcomes, with all failures retained.

## TUM-RS license final bounded check

The official linked [IROS 2019 paper full HTML, §V](https://arxiv.org/html/1911.01015v1#S5) states the dataset is publicly available and describes acquisition/calibration, but no explicit dataset license was found in this paper, the official page, top-level README or directory listings. The arXiv page's perpetual non-exclusive publication license concerns the paper and is not a dataset license. No author was contacted. Keep the data-terms status **unknown**, not prohibited and not implicitly inherited from TUM-VI.

## Exact metadata accounting

New direct body totals: receipt12 281,103 B; receipt13 68,464 B; receipt14 24,340 B; receipt15 339,705 B (includes an accounted-for 150,001-B capped mission-table response whose body was discarded); receipt16 HEAD 0 B; receipt17 277,792 B after HEAD established the exact length. This task adds **991,404 B**, bringing all-screen direct bodies to **2,635,272 B**. Provider-managed web traffic remains separately unknown. No sensor/archive payload, GT values or image values were requested by the direct client; no model or experiment ran. Underlying HTTP/source snapshots remain private; this summary is published with source attribution.
