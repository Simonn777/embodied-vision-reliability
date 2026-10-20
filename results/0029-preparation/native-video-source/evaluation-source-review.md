# Pre-execution evaluation interface review

Scope: private source only, with no scientific arrays, masks, labels, model or
network access. Tests prepared separately are manufactured and have not been run
as part of this review.

One concrete interface bug was found and reported before execution:
`seal_queries.verify_queries` places `old.outer_contract`'s hash binding in
`context["videos"][i]["outer"]`. That binding has no `wall_seconds` or
`peak_group_rss_kib`. The first evaluator draft indexed those missing fields
after decoding all GT and computing metrics, which would raise `KeyError`.
Root's narrow repair `complete_video_costs` reads the raw outer receipt at the
already-bound video spec path, verifies its hash against `outer_sha256`, then
reads resources. The old helper and sealer interfaces remain unchanged.

The rest of the reviewed path is consistent:

- The exact six-video proof and 48 saved choices are verified before GT admission.
  `load_predictions` decodes and validates every archived array, including finite
  values, raw-array hashes, shape, dtype and channel. All 48 rows then pass the
  pure evaluator's choice/witness/cost checks before the first `Image.open`.
- GT acquisition wrapper/inner receipts use the same deterministic admission,
  source map and manifest; the evaluator checks exactly the fixed 48 member
  identities, saved paths, hashes, byte counts and 500,000-byte phase cap.
  The opaque acquisition receipt itself is bound to the clean outer operation.
- Saved scalar policy values are checked by the sealer; the evaluator does not
  independently recompute original sparse losses from the decoded logits.
  This is the declared frozen-worker provenance scope, not a new independent
  inference or loss audit.
- After the offline guard, `seals.recheck` calls hash/inventory/metadata-only
  helpers and `video_inputs`; none invokes Git or another subprocess. The Git
  source proof and outer contracts occur before the guard.
- Query costs are separate named columns; complete-video wall/RSS come from
  the verified raw outer receipts. Overlapping time columns are not summed
  into a claimed complete-pipeline runtime.

The five prepared fake regressions cover the actual outer-binding interface,
changed raw receipts, all-48 tiny numeric archives, a nonfinite last archive,
and a bad last selection blocking every GT decoder. They do not qualify real
data decoding, native inference or resource performance.
