# 0010 solver admission: narrow independent review

No blocking defect found in `run_admission.py` with the documented external source-freeze/once-only caller gate. All ten manifest source hashes match. Manifest SHA256: `9d0f764ce380c99d98ee66fc836bf4d429b816b2f50035ab4394f3c01cff5868`.

Static inspection confirms exactly T63(alpha1) plus one held-out readout, T641(alpha1), T644(alpha4), with no timing retry/warmup. Successful calls require two native executions and reference bitwise parity; failures retain attempt/time and mark uncertain native counts rather than invent a completion. The projection is `1.5*(128*T63+2*T641+2*T644)` against 360 s. Resource refusal returns nonzero, preserves times, and does not imply accuracy. Inputs are generated once and checked for mutation afterward.

Both public stages use the identical pinned supervisor at 120 s, four CPUs and 8 GiB. Functional execution records the exact one-line inter-op change from 1 to 4; timing sets and checks intra/inter-op 4/4. All original source bytes remain unchanged, portable-source transformations are counted and hashed, and upstream code identity is independently checked by the adapter. Parent must still enforce distinct fresh output paths, published expected manifest SHA and functional-before-timing ordering; this CLI does not make choosing a new output directory an authorization to repeat.

Executed **only** `python -W error -m unittest test_admission_fake.py -v` with the existing local NumPy interpreter: 10 artificial fake-backend tests passed, 0.033 s unittest time, actual exit 0. No discovery, Torch, StatA solver or real-shaped timing run occurred.

Actual logs:
- stdout: `/tmp/evr-0010-solver-review-fake.stdout.log`, 0 bytes, SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- stderr: `/tmp/evr-0010-solver-review-fake.stderr.log`, 895 bytes, SHA256 `a14df6dc588c28ed81809cb086426f58f9c909ee1adeeb57e7ccb0ba1e13b81d`.

AI-assisted software review only. No source edits, download, cloud command or scientific outcome inspection.
