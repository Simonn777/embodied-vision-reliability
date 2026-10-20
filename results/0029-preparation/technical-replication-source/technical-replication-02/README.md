# Prospective all-six technical replication: admission-only private copies

These files prepare a new attempt and do not admit or relabel attempt01. All repository scientific files and original private sources remain unchanged. No scientific media, query GT, models or network were accessed in this work.

## Exact changes

- `run_video.py`: require new scientific commit and explicit `--qualification-commit ec94d0927fcd2779f702e161550d9a3e87303d6f`; verify unchanged scientific/qualification bytes against that old freeze; bind the new protocol and exit-checked supervisor; produce v2 video receipt/seal. Legacy native, qualification and evidence operations retain the original outer verifier.
- `seal_queries.py`: accept only proof v2 with six ordered, unique specs/output roots, one new scientific commit, one old qualification/evidence chain, and replication identifier `all-six-technical-replication-02`. Require each new video v2 receipt/seal and new clean outer before any GT admission. Expose `context["new_outer_contract"]`.
- `run_evaluation.py`: only the `gt_contract` outer-verifier call changes. Complete-video timing still reads the bound raw outer receipts. All other functions are AST-identical.
- `audit_queries_saved.py`: only the evaluation outer-verifier call in `admission` changes. Independent arithmetic and gates are AST-identical.
- `acquire_query_gt.py` and auditor arithmetic tests are byte-identical. The original range helper and its default GT prohibition are unchanged.

The new outer verifier requires the v1 exit-checked schema, exact command/spec/source/output/log bindings, exit0, no termination/signals/cleanup errors, fresh empty post-exit and final samples, and the fixed resource limits. It does not translate old failed receipts into success.

## Prospective proof

`schema = evr0029.prediction-proof.v2`; exact other keys are `expected_commit` (one new freeze), `evidence_commit` (c09e240009869402e2b97a55af42617eea3c6405), `qualification_commit` (ec94d0927fcd2779f702e161550d9a3e87303d6f), `replication` (all-six-technical-replication-02), and `videos`. Videos are six `{sequence, spec_path}` entries in fixed cohort order. Each spec command adds the same `--qualification-commit` flag. Old proof, video receipt and seal v1 are rejected.

The GT/evaluation CLIs remain unchanged and obtain the new supervisor contract through the admitted context. Scientific native worker, native model settings, diagnostic core, pure evaluator, frontend/witness logic, cohorts, masks, choices, gates and resource limits are excluded from changes.

## Prepared and checked

`original-sources/` and `original-source-bindings.json` preserve original bytes. `admission-only.diff` records all source/test changes; `source-bindings.json` contains final full hashes. `source-review.json` records AST comparisons and the source-comparison utility's initial function-name assertion typo, corrected without a production edit.

Parent-authorized fake-only tests: video6 + sealer11 + evaluation6 = 23 PASS, first attempt exit0; unittest1.611s, bounded outer1.799906125s, peak direct-child RSS51,937,280B, four threads. Sources unchanged. Receipt/logs/source snapshots: `../replication-admission-tests-01/`; receipt SHA256 `6b82a8be28b02bef9881b505bbe4d67fd7b74cf31d7a9227a98d22ee3d31ed51`. Six reused helpers were first compared byte-identical to repository copies. Unaffected auditor arithmetic tests were not repeated.

Independent source-only review found no blocker. This is preparation and artificial contract evidence, not scientific admission, completed replication or efficacy evidence.
