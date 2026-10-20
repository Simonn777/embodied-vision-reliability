# 0029 conservative acquisition planning balance

October 20, 2026. Local reconciliation of existing receipts and reservations; no scientific payload or model acquisition. **Planning balance after the latest metadata reservations: 1,649,868,314 bytes.** This keeps all identified prior holds. It is not an exact measurement of remaining network allowance: provider-managed web traffic is unmetered, and its full assignment to the earlier metadata reserve is unresolved. Leave additional headroom rather than spending to this number.

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
| 0029 full VOST directory and split/parent metadata, new full allowance | 25,000,000 | **1,649,868,314** |

The three extra holds total 4,141,572 bytes and may overlap the old 20 MB metadata reservation. They are deliberately retained pending a full assignment; this is **not a claim that these bodies were transferred twice**. No earlier reserve is released.

The [0022 operation](../results/0022-preparation/acquisition/operation.json) actually read 270,952,703 asset/dependency bytes. Actual 0023 RGB/intrinsic and GT-text bodies were 11,062,865 and 117,919 bytes. Using those two actual amounts explains the historical rounded balance: 2,444,780,692 − 11,062,865 − 117,919 = 2,433,599,908 bytes, approximately 2.43 GB. The table instead keeps their complete caps. The former hypothetical 430 MB physical-falsifier example was never an additional acquisition; the actual two-archive debit above supersedes that sizing example.

[0029 source-screen accounting](0029-task-screen.md) distinguishes exact body counts from full reservations. The 1 MB check read 200,553 direct source/metadata body bytes plus unmetered provider text; retain the full allowance. The later 3 MB check prepares native-runtime/source manifests only. The separate 25 MB allowance covers the known 21,042,989-byte VOST central directory plus bounded official split/parent-recording text; both checks are pending. No RGB, annotation-value, SAM2 checkpoint or dependency-wheel payload is allocated here. Before scientific acquisition, bind exact assets and prospective cohort, reserve their entire response-body limits, and retain failures and actual transfer receipts.
