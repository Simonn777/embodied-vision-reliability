# 0027 affine-tangent manufactured sensitivity screen

Original implementation and tests for the [prospective protocol](protocol.md); [admission and prior overlap](../../docs/0027-admission.md). This is not a real-image result or a manuscript.

`pipeline.py generate` requires the exact committed source and writes138 observation/truth records separately. `pipeline.py predict` consumes only observation records and generation metadata, binds the qualified native binary, and retains all attempted branches before sealing complete coverage. `evaluate.py` first verifies the complete seal/selection/source chain, then decodes truth. `audit_saved.py` independently checks saved scores, poses and the decision, with a clean external prediction receipt. The separately supervised dispatcher verifies actual helper/spec bytes and clean generation/prediction completion.

The native ABI fixture requires `EVR0027_NATIVE_RUNTIME`; generic CI skips that local-binary test explicitly. The other tests use exact witnesses or artificial records and never iterate the fixed noisy cohort. Run them with `python -W error -m unittest discover -s experiments/0027-affine-tangent -p 'test_*.py' -v`. The protocol, including native source limitations, governs any scientific invocation. No data/model download or paid service is needed for this screen.

The one-shot scientific commands use explicit `--repo`, `--expected-commit`, `--output`; prediction additionally needs `--generation` and `--native-runtime`. The source commit must pass CI before generation. Each operation is externally limited to30seconds, four CPU threads and8GiB group RAM, with process/log/output receipts. Frozen failures or partial cohorts are retained; an incomplete operation cannot be converted to a smaller successful dataset.
