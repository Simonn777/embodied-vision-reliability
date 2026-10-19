# Runtime attempt03: thread-reporting compatibility failure

Source472deb8 passed exact CI39337303798 before execution. Installation passed
(exit0,1.819894667s,83,696KiB sampled peak). Qualification failed before any
fixture inference (exit1,20.239189334s,652,288KiB sampled peak). All active
MoGe, SuperPoint and LightGlue weights passed strict loading. The exact nine
unused release entries were verified/recorded. One optional NumPy probe was
suppressed without a child; ordinary guard violations were zero.

The failure was execution_thread_configuration: OpenCV with the GCD backend
reported10 after setNumThreads(1), while Torch reported requested4/1. Retained
API-only probes (no numerical/model workload) show setNumThreads(0) reports1.
OpenCV4.11.0 [parallel.cpp](https://github.com/opencv/opencv/blob/4.11.0/modules/core/src/parallel.cpp)
shows both0 and1 bypass the general parallel_for parallel path. GCD's reporting
function returns CPU count for1. Thus this failure does NOT establish that ten
workers executed; the earlier commentary suggesting ineffective execution
limiting is corrected. No inference had begun.

Attempt04 will explicitly disable OpenCV parallelism with0 and require reported1,
keeping Torch4/1 and every model, fixture and numerical setting unchanged. Earlier
attempts stay failed. JSON copies are byte-identical; path-only redacted public
logs and private originals have distinct recorded hashes. No scientific gain.
