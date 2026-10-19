# Fixed128-image headroom study: publication before pixel access

[PR38](https://github.com/Simonn777/embodied-vision-reliability/pull/38)
merged as `49c0c0b22f89e5d395130dec63bafad9377ba5c5` on October19 UTC
(October18 America/New_York). Parent Git comparison found no tree difference
from receipt head `de07509be1f314da2b8803a672b2d266b234f723`.

The source is `d14134c34e7fa097de5556aca162af962d920880`. Its direct child adds
only [the canonical receipt](../protocols/0010-real-data-headroom-freeze.json),
SHA256 `9ee3c49d547a8d645586c6792ad91ede49cf1ce5114fc82f49b1cf65ec21f0ff`.
It binds45 repository inputs and9 saved external input identities. The actual
plan SHA is `d8b1891dd99ea6b7e1680b830da478f81b9e37cedf11933f8c878511cfca21ce`.
No selected image or accuracy result informed this freeze.

The parent read both exact completed CI outputs before dispatch:

| Check | Receipt head | Merged main |
|---|---:|---:|
| Repository tests |649 passed,31 historical skips,193 subtests;24.75s|same counts;24.82s|
| Capability fake tests |54;0.151s|54;0.150s|
| Solver fake tests |10;0.017s|10;0.016s|
| Guard fake tests |11;0.168s|11;0.167s|
| Headroom predictor fake tests |15;1.584s|15;1.554s|
| Headroom driver fake tests |17;6.248s|17;6.285s|
| Synthetic smoke |1,240 frames;3.84s|1,240 frames;4.55s|

[Receipt-head CI](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39181113693)
and [main CI](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39181219870)
ran no real128-image study. This publication allows only the prescribed single
900-second development comparison, conditional on live cache/marker/resource
and full source/input verification. It proves neither an improvement nor novelty.
Historical capability and cached-admission worktrees/freezes remain separate.
