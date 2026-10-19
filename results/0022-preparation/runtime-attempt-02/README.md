# 0022 runtime attempt02: strict legacy-key failure before inference

Freeze `6ec0980bc9c53ad70deb70667ef85d3d1c7cc27f` passed exact
[CI39336545262](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39336545262).
The fresh offline installation completed: exit0, 1.818263334s outer wall,
84,272KiB sampled peak. No asset download or shared-environment change occurred.

Qualification02 passed imports, recorded one suppressed optional NumPy CPU probe
and zero forbidden operations, and strictly loaded every active MoGe/SuperPoint
parameter. It then failed before inference at LightGlue's strict key check:
exactly `log_assignment.0.r` through `log_assignment.8.r` are unexpected; only
the previously documented deterministic confidence buffer is missing. Outer
exit1, 18.716249458s, 668,912KiB peak. No manufactured inputs or prediction arrays
were generated. The original receipts/logs remain immutable.

Pinned author LightGlue source `eb42fee2d71449efb0aa5c10549752b5d75384d8`
defines MatchAssignment with matchability/final_proj and never defines or reads
`r`. Its official release loader uses strict=False and ignores these same extra
entries. A separate weights-only inspection found all nine are finite float32
shape[1] constants equal to one; their canonical little-endian tensor SHA256 is
`e00e5eb9444182f352323374ef4e08ebcb784725fdd4fd612d7730540b3e0c8c`.
This inspection deserialized the existing local checkpoint but constructed no
model and performed no inference. The whole checkpoint hash remains bound.

The next explicit compatibility repair may remove **only** that exact verified
nine-entry set, record it, and then require strict loading and value equality for
all active weights. Every other unexpected key or missing active parameter stays
fatal. This restores the pinned author's behavior; it changes no model setting,
input, acceptance threshold or scientific outcome. The first two failures remain
failures. Public JSON receipts are byte-identical; path-redacted logs and their
private raw identities are distinguished in publication-bindings.json.
