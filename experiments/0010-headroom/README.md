# Prospective 0010 headroom integration

Resource and saved-input admission has passed; the real128-image experiment has not run. `plan.json` binds18 actual input identities and the separately verified capability parent execution; only its admission flag changed after review. `plan.template.json` stays unadmitted as a historical example. Verify the separate published protocol/source/plan freeze and exact CI before the single authorized launch.

## Source and files

`run.py`, `native.py`, `driver.py`, `headroom.py` and the referenced `stata_cpu.py` are unchanged reviewed bytes. Runtime sources are explicitly hash-checked. This directory contains the fixed 128-name `selection.json`, copied without changing the predeclared salt/order or excluded capability names. It contains no selected pixels or embeddings.

The sibling files supplied in this integration snapshot are exact references, not replacements for historical frozen components: original `../0010/{run_capability.py,config.json,selection.json}`, historical capability freeze, original `../0010-solver/{stata_cpu.py,manifest.json,upstream/...}`, the existing supervisor and the four pinned CLIP source files plus license. Their complete pre-existing bundles must remain available. In particular, the solver manifest refers to its original complete admission bundle, not just the reference subset copied into this snapshot.

The StatA source is AGPL-3.0; its exact upstream source, license and provenance remain under `../0010-solver/upstream/`. Do not silently apply a repository-root MIT license to it or its derivatives. CLIP's original MIT license is preserved alongside its source. No package, data, model or author-source download is needed to integrate these files.

## Filled plan and external admission boundary

The reviewed `plan.json` has18 filled input references. Relative source references resolve from the plan directory; actual saved-result references use the original cloud paths. The template retains its placeholders. Do not substitute `partial-features.npz`, import/preparation/guard receipts, old local Torch tests or fake-test receipts.

The new `external_bindings.capability_execution` field is intentionally **not consumed by current runtime code**. Independent saved-output review verified the actual capability parent zero exit, raw wait/log closure and byte identity; its reference and review are bound in the formal freeze. The current launcher also does not independently verify a canonical Git freeze; callers must obtain its exact approved plan SHA from the separately verified publication. A local `admitted: true` value alone is not admission.

Use the actual capability feature hashes, same assets/text/prototypes/prompts/class order, CPU Torch/NumPy versions and consistent runtime module paths. Functional and timing processes must share the same admitted environment; their receipts require four intra-op and four inter-op threads. Twenty actual functional tests and the single three-fit timing operation completed separately; this driver consumes those saved results without rerunning them. The 450-second encoder and 360-second solver projections remain admission estimates, not wall-time guarantees.

After the separate published freeze and exact CI are verified, the entry-point shape is:

```text
python run.py launch --plan plan.json --plan-sha256 APPROVED_SHA \
  --assetdir EXISTING_VERIFIED_ASSET_LAYOUT --output FRESH_OUTPUT \
  --supervision FRESH_SUPERVISION --python ADMITTED_INTERPRETER
```

No launch was executed in preparing this bundle. Persistent launch/child markers prevent changing output to repeat an attempt. The parent supervisor imposes 900 seconds, four CPU IDs and 8 GiB; all incomplete predictions remain invalid, with no survivor-only scoring. Preserve old partials/markers/evidence outside the scientific inputs.

## Portable fake checks

The only transformation is in `test_driver.py`: two `/tmp/...` fixture import expressions now resolve to `Path(__file__).resolve().parent` and its `test_headroom.py`. The latter file already uses a local `import headroom` and was copied unchanged. No test assertion, numerical rule or production code changed. Original test bytes and historical failure logs are retained in `../../results/0010-admission/headroom-software/history/`; its portable-review/test-only-conversion.json records original/new SHA and exact replacements.

Run the two files explicitly from any directory:

```text
python -B -W error /absolute/path/to/test_headroom.py
python -B -W error /absolute/path/to/test_driver.py
```

Actual portable checks used a fresh unrelated empty cwd and unset `PYTHONPATH`: **15 passed in 0.767 seconds; 17 passed in 3.093 seconds; both observed exit 0**. All copied source files remained unchanged after the tests. These checks use fake encoders/fits and assert that Torch was not imported. They do not establish Native numerical coverage, cloud isolation, real extraction timing or scientific improvement.

The original graph peer-set confound and lack of proven physical-scene independence remain. This is only a prospective headroom algorithm comparison; no accuracy, novelty or manuscript result follows from a passing software test.
