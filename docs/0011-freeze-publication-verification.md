# 0011 freeze publication verified before data preparation

[PR40](https://github.com/Simonn777/embodied-vision-reliability/pull/40) was merged
as `a0b061660f78487f67fc4dedad6ad0c99aca4de7`. Its exact head was
`1951ae7bfd47f601c7187ee72ddd03ab905e7b56`; parent Git comparison finds no tree
difference between that head and the merge. Source commit
`a288f65bfb9d9e0d70dee72c0d83b020a0339458` precedes the canonical24-file receipt,
whose SHA256 is `f45cfe56f99cdb8c4cb2ad6376a3c0813693666e2798941b9627d535c7a61e46`.

The parent verified exact PR [CI39185145237](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39185145237)
and merge [CI39185292174](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39185292174):
both succeeded with763 tests,31 historical skips and272 subtests. Pytest elapsed
28.20s /28.54s. Auxiliary suites54/10/11/15/17 and1240 smoke frames also passed;
smoke elapsed3.68s /4.81s. CI installs its normal software-test dependencies;
the research cloud reused its already cached packages without installation.

In the original Work environment, turn
`01a151c2-4452-716e-816a-50f2121a1e73` tested all114 new artificial fixtures with
zero skips/errors/failures. The Linux micro-child actually exercised four-core,
8GiB limits. Outer wait4 observed exit0/raw wait0,3.257158551s,
61,432KiB peak RSS; NumPy2.0.2 came from the existing runtime and Torch was not
imported. Execution receipt SHA256
`f7b904a8645fd6b921122ff41d533c86acdae533e19c06afb0171e04b2b40ce4`.
Parent directly retrieved actual command records, including
`exec-5ea66f0b-047f-4baa-be38-7d1d786b6e22` and
`exec-7a74bbbf-d8a6-40b7-9aaa-1cb30a378ff1`, both exit0 with untruncated test
evidence. Two earlier source-display outputs were truncated; no success claim
relies on their unseen tails. Current/index/source-commit file identities,
clean checkout and absence of `scratch/0011-oblique-v1` were checked.

Only after the parent observed both CI successes and this software evidence did
it authorize the fixed preparation→g-036→g-002→derive→score sequence. A failed
phase must stop continuation without overwrite/retry. This verifies prospective
publication and software operation, not any scientific improvement.
