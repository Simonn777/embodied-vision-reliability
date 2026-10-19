# Artificial arithmetic diagnosis

These saved reports were produced from bounded artificial arrays, without data,
labels, saved accuracy evaluations, or native estimators. Original local files
remain unchanged. `provenance.json` records original and public hashes; only
absolute workspace prefixes in tracebacks have been normalized for publication.

The original reproduction script assumes a sibling `embodied-vision-reliability`
checkout. To repeat the original-source probe, copy the two Python scripts to a
scratch directory, place a checkout of commit32f3e40 in its
`embodied-vision-reliability` child directory, and run with NumPy2.0.2 on the
reported Apple Accelerate runtime. The contraction-check script is standalone.
Results are platform dependent; passing on another backend does not refute this
record. Importing the repaired current kernel is a different test. The added
large artificial tests exercise that repaired implementation separately.

Independent saved-only review verified the first failed cohort, and another
bounded artificial reproduction agreed to3.33e-16 for Nx3 products. The reviewer
inspected repaired kernel f63a55f5b971f86549a184ca52bdd61f5871c36b54ab944ca68d2c768c5bc262
and worker2c4f1576 contractions and passed21 artificial kernel/worker tests.
The abbreviated real failure does not identify its exact failing operation;
these results support the technical repair without asserting real-call proof.
