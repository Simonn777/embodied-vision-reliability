# 0026 direct-refiner bridge qualification

This records one failed build, one prospectively approved compiler-only repair,
and the first execution of the eight fixed manufactured tests. No HDF5, scientific
cache, GT, image, model or real-pair estimator operation was used.

- `native-build-operation-01/receipt.json`: build exited1 in0.885315333s; sampled
  process-group peak20,608KiB. Its inner `native-build-01/stderr.log` reports three
  upstream `camera_models.h:95` unused-parameter errors under `-Wextra -Werror`.
  Source snapshots, logs and failed receipt remain intact; no binary/test/native
  call succeeded in this attempt.
- Parent approved classifying only dependency include roots as `-isystem`.
  The new wrapper retained `-Wall -Wextra -Werror`; no upstream source/archive,
  algorithm, test fixture, tolerance or native option changed.
- `native-build-operation-02/receipt.json`: exit0,1.143491875s, sampled peak16,688KiB,
  unchanged source hashes. The inner completed receipt is `native-build-02/receipt.json`.
- `native-tests-01/receipt.json`: exit0,0.7692315s outer, sampled peak35,856KiB,
  unchanged sources. stderr records **8 tests passed in0.026s** with Python
  `-W error`. Four native calls used zero iterations for mapping/scalar-cost checks,
  one used the default100 limit on manufactured correspondences; seven invalid
  seeds were rejected before native refinement. No rerun of tests was needed.

All operations set computational thread environment variables to1, used the
existing Python3.9.6 venv, and were watched with180s/8GiB limits. RSS is a sampled
process-group observation, not an exact maximum-memory measurement. Exact commands
and paths are in the receipts. The build explicitly used `--project-root`.

Library: `native-build-02/libevr0026_refiner.dylib`,858,360 bytes, SHA256
`00418b002b005e664e75198c45e4a318d576a2ad04d8d0cb7eb7788affc18621`.

| Artifact | SHA256 |
|---|---|
| refiner.cc | `a2ebb24202efa5c4b26dabdfe1ea96bbd04541a063dfc626ebc47d8a79c86657` |
| bridge.py | `5f3746e2ec7562c63dd6ae9c8180e28ca542510b0e3762e7297c5c335f6ee820` |
| build.py | `bea68b06b0c173ce3b454855defbbe678fd208d50ff5d17caff042f35184b4fd` |
| test_bridge.py | `2fee77efee97966526c72f7585b63d138cb076999e76cf94df4ffb828d372ce9` |
| input-lock.json | `80a0c7a2ba76025403bcc7d1abc4d3ace94c9dc141c479a1afe2180c026197ed` |
| failed inner build receipt | `f56a29c5d89c3f136b04049ea78162ce2bd12ad37971332e52e0581e860c0b11` |
| successful inner build receipt | `e36382ced70ba8bf065247c5ff6854c59b580f7b2338bd2322cea95853e50e82` |
| successful outer build receipt | `395681bc3d305481cfa31076566df5cc9e48289f2cbda897919a75420418532b` |
| first test receipt | `416cff66c122b2d9636983425540aaa7e82939558039308a0ae4b430557c7fe1` |

The Python loader checks the current wrapper/C++/builder/lock/notice source hashes,
exact input lock and library hash before dlopen. Passing this qualification shows
that the separate ABI reaches the intended native fixed-support TC calculation
and preserves its options/raw outputs on these fixtures. It is not pose-quality
evidence, a global certificate, or admission of the proposed80-pair diagnostic.
