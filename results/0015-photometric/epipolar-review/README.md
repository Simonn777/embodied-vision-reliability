# Post-hoc epipolar compatibility of admitted corrections

This saved-output-only descriptive check uses all191 admitted points from all16
pairs. It holds the original source point and saved GT camera pose fixed, and
measures target-pixel perpendicular distance to that source point's GT epipolar
line. Native authored camera intrinsics and the saved projected GT rotation are
used. Along-line ambiguity and source-point error remain: this is **not true
correspondence error**, a population estimate or a new method selector.

Mean distance rises0.536246830px→0.596308882px; equal-pair mean rises
0.546650395px→0.607451459px. There are89 improvements/102 harms, and4/16 pairs
have lower mean distance. All points are retained; the original failed pose
gate remains unchanged. Independent normalized-ray/cross-product calculation
agrees within1.61e−12px.

[Source](analyze_saved.py) and [complete191-point result](analysis.json). The
script's artificial horizontal-line check ran before reading saved records;
no HDF5, images, model or pose solver was read or called. Inputs and script are
hashed in the result. This does not support enlarging or propagating the current
photometric updates without a separately justified information mechanism.
