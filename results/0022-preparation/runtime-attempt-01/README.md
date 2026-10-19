# 0022 runtime attempt 01: import compatibility failure

The exact second freeze `b1d337622e4d7bfc1adbc736fe8355196af759d0` passed
[CI 39334706191](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39334706191)
before either operation. The installer completed once, offline, in a fresh private
target: exit 0, 1.949170542 s outer wall time and 88,688 KiB sampled peak RSS.
All 14 wheels and 102 source files came from the already verified acquisition.
The existing shared environment was not modified.

The first manufactured qualifier then failed during **imports**, before model
construction, checkpoint loading or inference: exit 1, 10.201063042 s outer wall
time and 250,256 KiB sampled peak RSS. The Python audit guard rejected NumPy
2.0.2's optional `check_support_sve()` invocation of `subprocess.run('lscpu')`.
NumPy catches `OSError` for a missing command, but our guard raised the distinct
`QualificationError`. A read-only PATH lookup found no `lscpu` on this macOS host.
The guard prevented process creation. No benchmark data or poses were read.

This is an execution compatibility failure, not a model-quality outcome. A new
labeled attempt may narrowly preserve NumPy's missing-command fallback by raising
`FileNotFoundError` for the verified optional probe, while still creating no
process and retaining other guard failures. The original inputs, model settings
and acceptance checks must remain unchanged. A fresh isolated installation under
the new freeze maintains the existing same-commit receipt contract; no additional
asset download is needed. Both first-operation receipts remain immutable here.

Receipts are byte-identical copies. Public `.redacted.log` files replace only
local path prefixes; their hashes are distinct from the immutable private raw
logs bound by the operation receipts. `publication-bindings.json` records both
identities. The LibreSSL/urllib3 warning is retained; it was not this failure.
This record establishes no scientific accuracy gain or manuscript eligibility.
