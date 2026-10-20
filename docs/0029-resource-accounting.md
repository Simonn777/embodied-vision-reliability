# 0029 conservative acquisition planning balance

October 20, 2026. Local reconciliation of existing receipts and reservations; the pinned asset-only acquisition is complete, with no scientific data access. **Planning balance after metadata, asset-only and prospective DAVIS reservations: 1,289,868,314 bytes.** This keeps all identified prior holds. It is not an exact measurement of remaining network allowance: provider-managed web traffic is unmetered, and its full assignment to the earlier metadata reserve is unresolved. Leave additional headroom rather than spending to this number.

| Entry | Retained debit, bytes | Balance, bytes |
|---|---:|---:|
| [0017 ending ledger](../results/0017-preparation/transfer-ledger.json) | — | 2,771,072,148 |
| [0018 directory and source/API reservations](../results/0018-preparation/resource-screen/transfer-ledger.json) | 6,291,456 | 2,764,780,692 |
| [0022 full asset/wheel/metadata reservation](../protocols/0022-traceable-frontend-assets.md) | 320,000,000 | 2,444,780,692 |
| [0023 RGB/intrinsic cap](../results/0023-development/runtime-attempt-01/acquisition.json) | 20,971,520 | 2,423,809,172 |
| [0023 GT-text cap](../results/0023-development/completed-study/gt-acquisition.json) | 2,097,152 | 2,421,712,020 |
| [0025b completed archive acquisition](../results/0025b-development/acquisition/receipt.json) | 731,653,920 | 1,690,058,100 |
| [0025 source metadata](literature/0025-rs-dataset-budget.md), extra hold | 2,635,272 | 1,687,422,828 |
| [ICL format source metadata](literature/0025b-icl-input-format.md), extra hold | 73,204 | 1,687,349,624 |
| [0027 primary PDF receipt](../results/0027-preparation/primary-source-receipt.json), extra hold | 1,433,096 | 1,685,916,528 |
| 0029 initial video metadata bodies | 48,214 | 1,685,868,314 |
| 0029 traffic metadata, full allowance | 5,000,000 | 1,680,868,314 |
| 0029 video metadata, full allowance | 2,000,000 | 1,678,868,314 |
| 0029 multi-object/model-source metadata, full allowance | 1,000,000 | 1,677,868,314 |
| 0029 native-runtime dependency/source metadata, new full allowance | 3,000,000 | 1,674,868,314 |
| 0029 full VOST directory and split/parent metadata, full allowance | 25,000,000 | 1,649,868,314 |
| [0029 pinned asset-only acquisition](../protocols/0029-runtime-assets.md), full allowance | 300,000,000 | 1,349,868,314 |
| [0029 fixed DAVIS development phases](../protocols/0029-signed-memory-diagnostic.md), prospective full allowance | 60,000,000 | **1,289,868,314** |

The three extra holds total 4,141,572 bytes and may overlap the old 20 MB metadata reservation. They are deliberately retained pending a full assignment; this is **not a claim that these bodies were transferred twice**. No earlier reserve is released.

The [0022 operation](../results/0022-preparation/acquisition/operation.json) actually read 270,952,703 asset/dependency bytes. Actual 0023 RGB/intrinsic and GT-text bodies were 11,062,865 and 117,919 bytes. Using those two actual amounts explains the historical rounded balance: 2,444,780,692 − 11,062,865 − 117,919 = 2,433,599,908 bytes, approximately 2.43 GB. The table instead keeps their complete caps. The former hypothetical 430 MB physical-falsifier example was never an additional acquisition; the actual two-archive debit above supersedes that sizing example.

[0029 source-screen accounting](0029-task-screen.md) distinguishes exact body counts from full reservations. The 1 MB check read 200,553 direct source/metadata body bytes plus unmetered provider text; retain the full allowance. The completed 3 MB check read 1,319,651 direct source/metadata bytes across 58 requests. The completed 25 MB check read 22,806,230 bytes across 41 requests, including the 21,042,989-byte VOST central directory and bounded official split/parent text. Retain both full allowances. No image or mask values were read. The new 300 MB reservation admits only the pinned assets after review, source freeze and exact CI; expected payload is 257,103,668 bytes. Scientific cohort allocation and inference remain separate. Preserve every failure and actual transfer receipt; no reserve is refunded against a smaller measured subtotal.

The [completed0029 acquisition](../results/0029-preparation/acquisition/README.md) read257,104,722 bytes, including1,054redirect-body bytes beyond the257,103,668 asset payload. All83files verified. Retain the entire300MB debit and the1,349,868,314-byte conservative planning balance. No installation or benchmark payload at this receipt.

Before scientific payload access, the new60,000,000-byte hold covers386RGB+6initial masks+48query masks and bounded ZIP local headers. Only the first two phases are prepared; query GT is closed pending all-six prediction sealing. Retain every preceding allowance. Scientific assets are not rehosted.
