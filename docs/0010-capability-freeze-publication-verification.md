# 0010 capability freeze publication

The parent verified publication before authorizing the one-shot cloud capability
operation. This records source publication and software checks, not model
feasibility or a scientific improvement.

- Source: `ececb253bfa00acffdb8d84c91d848c2802444a0`.
- Receipt-only child: `93629d385b286b9471167e0c15c49bf29120096e`.
- Receipt: `protocols/0010-clip-capability-freeze.json`, SHA256
  `8dfec7362d01fac80c46f9871fe4e8696c90d12ba07038c2635a6f8dd947b29f`.
- [PR 36](https://github.com/Simonn777/embodied-vision-reliability/pull/36)
  merged as `e9e7c7609e53e1bc1b3228096f9b36008cb78d6b`.
  Its tree has no file differences from the receipt child.
- [Exact PR CI](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39176989959)
  passed 649 tests, 31 disclosed historical Torch skips and 193 subtests
  in 23.82 seconds; the separate 54 artificial preparation tests passed in
  0.133 seconds; the smoke pipeline scored 1,240 frames in 2.95 seconds.
- [Exact main CI](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39177950945)
  passed the same counts in 25.13 seconds, 0.140 seconds and 3.03 seconds,
  respectively. The parent read actual completed/success status and logs.

The receipt binds 25 inputs, including the unchanged Linux supervisor and CI.
No separate source-only CI run is claimed: the source ancestor is covered by
the tested receipt-head tree. Cloud execution must still verify these bindings,
cache/operation state, successful preparation, and the fixed resource limits.
