# Verified publication of the baseline and known-noise diagnostic

PR43 [Measure pose headroom and close negligible pixel-noise control](https://github.com/Simonn777/embodied-vision-reliability/pull/43)
merged on2026-10-19 at18:34:50UTC. Head79f060eb974bf39a8ac6c3fe04c38b53a98a8808
had a successful [exact PR check](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39303477626).
Merge73686b717db64d8807c945cad6c9fcfbd859b515 has a successful
[exact main check](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39303682287).
The branch was fetched and its published file tree compared with the PR head.

CI includes38 baseline tests (plus25 subtests),54 known-noise tests, native
artificial-bound checks and the existing regression/helper suites. These
artificial tests do not rerun scientific datasets or establish accuracy gains.
The baseline and known-control raw results, both initial failures, separately
frozen repairs and independent saved-output audits are now on main.

Publication is complete for0013/0014. Scientific success is not: pixel-Sampson
AUC@5 gain0.006510542percentage points is negligible and the strict interpretation
gate remains15/16. No paper is admitted and no held-out numeric data was read.
The active research goal continues with a distinct image-information question.
