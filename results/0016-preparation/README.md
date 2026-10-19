# 0016 preparation: official current library and artificial qualification

Capability/provenance evidence only, not accuracy improvement. No fresh
scientific coordinates, image, confidence, depth or GT value was read here.

Official PoseLib commit6a636b1d84273d5e3ba97676463476569611572a was acquired from
its GitHub commit archive and built unchanged in an isolated local runtime.
VERSION.txt and the module report **3.0.0**; older PoseBench README examples
showing2.1.0 do not identify this build. Existing2.0.5 runtime and predictions
remain intact. [Runtime hashes](../../configs/0016-libraries.json).

Inputs are CMake3.31.6 macOS universal wheel, pybind11 3.0.1 wheel, Eigen3.4.0
official source and the pinned PoseLib archive. Exact URLs, lengths, SHA256s,
limits and times are in four acquisition receipts. Wheel hashes match official
PyPI metadata. No global package upgrade. Upstream licenses remain applicable;
no third-party archive or binary is redistributed here.

All four build stages exited0. Native compilation used2jobs, Release/O3,
no march-native, default WERROR and no benchmark/tests/stub generator. It took
73.377837seconds and sampled peak summed process RSS1,688,336KiB. The current
binary and old wheel can differ in compiler flags; old/current speed effects are
not isolated. The two current algorithms share exactly one binary.

`source/acquire.py` and `source/build.py` preserve actual preparation helpers,
originally executed from one private preparation directory. Reproduction copies
both there before use. They refuse existing start markers, cap downloads,
enforce20minute build time and8GiB process-tree RSS, and do not read datasets.
Full build logs remain private because they contain local paths. Public stage
receipts preserve actual exits/time/resources, not complete stdout.

All three backends passed qualify_native.py on48manufactured3D points, requesting
exactly2and4iterations. Each returned the requested count and a finite proper
pose. Six native artificial calls total; qualification records disclose runtime
hashes/defaults/times. This verifies option routing, not image quality.

The64-pair allocation read only HDF5 root names and opaque file integrity.
Manifest SHA256208247df135b9305f7a8bfd30bf77cbbc58f2cda3ec039fa0da7b3bb1dc34abe.
It precedes new numeric reads, excludes the prior16pairs, and remains one
landmark. See the [prospective protocol](../../protocols/0016-search-headroom.md).
Measured new build-input bodies total50,598,914bytes. Including2MiB conservative
metadata allowance (which covers the259,699-byte ScanNet directory read), the
remaining asset allowance is2,825,233,268bytes. No new dataset/model weights.

Corrected read-only lookup mistakes involved an obsolete protocol filename and
PoseLib's main-versus-master ref. One documentation patch used the parent path
instead of the checkout and was rejected without changes. Artificial reviews
caught finite-ray, binding and timing gaps before scientific execution. No
scientific prediction was repeated or result repaired in this preparation.

Final prefreeze check:56 artificial tests passed. Independent review covers all192 mocked workers, all12 backend/seed API paths, strict final-score formulas, seals, failure stops and cumulative-time accounting. No blocking issue was found; a failed final seal validation still refuses evaluation even if the already-written seal says completed.
