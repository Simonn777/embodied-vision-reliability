# 0030 problem-card review

2026-10-20. Read-only review of `problem-card-draft.md` and `post0029-learning-feasibility.md`; no model import, code execution, scientific data access, or repository edit.

**Verdict:** proceed with the proposed small manufactured gradient/API/resource probe. No successful hand-coded selector, novelty proof, full BPTT implementation, or new opportunity assay is a prerequisite for that prototype. The card appropriately preserves 0029's closure and separates development from final evidence. The following are definitions to settle before real training, not reasons to block the probe.

## Specify the actual future objective

“A frozen downstream model” does not say whether its *state* is frozen. Choose and name one initial training scope:

- **Fixed-context differentiation:** replace only the selected stored tensor \(m_t\) by \(m_t+\delta\), hold every other cached memory/pointer constant, and recompute the selected future attention/decoder outputs. A sum over several future frames still differentiates only their direct dependence on that tensor. This is not recurrent BPTT.
- **Truncated recurrent unroll:** start from a declared detached context and regenerate intervening predictions and memory writes. Backpropagate through that specified horizon, recording any detach boundaries. This includes indirect temporal effects within the horizon; it is not full-sequence BPTT.

Teacher optimization and the direct same-head control must use the **same initial contexts, future frames, losses, loss normalization, regularizer, bound and gradient path** for the primary mechanistic comparison. The inexpensive fixed-context version is a legitimate first implementation. It cannot support a claim of superiority over a fully recurrent training algorithm that was not run. In either case, development selection must use the student's own complete causal rollout; the card already requires this.

Since the residual is written after the current mask is emitted, its segmentation objective starts at later frames that actually consume that memory. It cannot improve the already-emitted current mask in this intervention. Cached image features may be reused; cached memory-conditioned features cannot stand in for the changed rollout.

## Controls and equivalence

The candidate **is** ordinary regression of a multistep optimized delta. The feasibility note's proposed extra “ordinary same-teacher feature/delta KD” would therefore duplicate the candidate and should not be a required additional arm. Only a later new distillation loss would require vanilla same-teacher KD as its distinct control. Logit KD is a potentially informative different representation target, not a mandatory extra prerequisite now.

For an unprojected one-step target \(\delta^*=-\eta\nabla_\delta L(0)\), squared regression at a zero-output head has gradient \(2\eta J_\theta^\top\nabla_\delta L(0)\). This is proportional to the direct-loss gradient **at that point**, given the same computation and derivative convention. It does not establish identical subsequent optimization, equivalence after clipping/projection activates, or equivalence of multistep targets. The one-step arm can be a small diagnostic/ablation; it need not become an independent expensive campaign.

Native BF16 storage makes the actual forward map quantized. Autograd through casts supplies a surrogate derivative, not a proof of a smooth derivative of the discrete stored values. Keep that convention identical for teacher and direct control and retain the declared storage path. A useful API check verifies that a nonzero update can affect the intended future computation as well as producing finite gradients.

The current-GT-memory control is useful. Construct its target through the same frozen memory encoder and declared no-object/binarization policy. Do not feed current or future GT as an inference prompt to the deployed/student rollout. All controls receive the same *available* annotations; their objectives may legitimately use them differently. Charge teacher construction to the candidate's training budget, and give direct training a meaningful cost-matched update allowance. Update-matched and cost-matched results answer different questions; label both.

## Minimal probe and development boundary

Check the ordinary-tensor/grad-enabled path, a consumed memory leaf, zero-residual native parity, one complete head optimizer step, unchanged frozen weights and state carriers, and complete time/RSS. With a zero-initialized final layer, earlier head layers can have zero gradient on the first step. Require the intended memory gradient and at least the appropriate output-layer update, **not nonzero gradients in every parameter**. If a native absent-object gate removes the gradient, identify that branch and the declared training logits/loss rather than presenting a different surrogate as native behavior.

A small head does not make attention/decoder backward cheap. The probe is allowed to discover this; failure calls for a narrow documented implementation repair or a resource decision, not an accuracy conclusion. One future-frame probe qualifies only that graph, not a longer recurrent horizon.

The four/two parent split, finite configuration campaign, causal development rollout and second-seed comparison are sensible development rules. The two development parents remain exposed development, and the existing sparse query annotations remain insufficient for full-sequence J&F. No future label or target enters student deployment or protected test selection. Ordinary logged development iterations can proceed within the authorized campaign without serial new approval or publication stages.

The final two-independent-source **+3 absolute official J&F** gate, recording-level uncertainty, component-loss bound, strong controls and complete-cost accounting remain unchanged. This prototype may establish API feasibility or a useful training direction; it establishes neither novelty nor that final result.
