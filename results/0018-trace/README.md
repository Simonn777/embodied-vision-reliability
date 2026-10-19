# 0018: complete replay, ineligible mechanism diagnosis

The 160-call replay completed once, and every final raw pose/scale/shift/statistic
and mask matches 0017. Independent review confirms all 4,640 floating components
are bitwise identical. The subsequent independent cost pass stopped after 88
reconstructions on a nonfinite initial optimizer state. **The frozen mechanism
study is INELIGIBLE, not a positive result and not evidence against the mechanism.**
No new method, accuracy gain, causal conclusion or manuscript is established.

The scientific freeze was `4770af453750fce77dd0c655266c66530cc0138a`;
[exact CI](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39324519428)
passed before either operation. Source, original binary, data and 0017 evidence
remain unchanged. No native call was repeated after the diagnostic failure.

| Operation | Actual exit | Wall seconds | Sampled peak group RSS KiB |
|---|---:|---:|---:|
| Prediction / 160 native calls | 0 | 61.360974125 | 94640 |
| Reconstruction / diagnosis | 1 | 20.742019125 | 63568 |

The prediction seal contains all 160 successful outcomes and 37,072,987 bytes of
native trace. The reconstruction receipt records `TraceMismatch: q has invalid
shape or nonfinite entries` at pair 44, depth_scale, after pairs 0–43/both arms.
The raw trace shows a nonfinite proposal becoming a temporary incumbent and
entering LO 0. Its initial/exit costs are nonfinite; this is not merely an
invalid rejected trial. Native execution later replaces it and returns a finite
pose. The predeclared complete-cohort barrier therefore correctly refused label
access. There is no reconstruction seal, label output or evaluation output.

The separate [saved-only review](independent-review/README.md) checks every trace
without coordinates or labels. It finds six nonfinite required states (initial
and exit) in three scale-arm LOs: pair 44/London, 61/Milan and 64/Sagrada. All
three are LO 0 and are outside the final returned ancestry. This graph fact does
not repair the original all-LO reconstruction requirement. The 88 saved completed
reconstructions are partial evidence and must not be scored as a selected cohort.

The complete prediction, partial reconstruction, actual outer receipts/logs and
independent review are retained. File paths are repository-relative; private
HDF5 arrays and compiled libraries are not uploaded. The next accuracy comparison
is a separately frozen author-recommended TRUNCATED_CAUCHY control on all 80
pairs. It does not change this result or establish a new method. The distinct
0017 depth-information opportunity remains unresolved; no paper is started.
