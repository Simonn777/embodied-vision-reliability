"""Conditional development adapter: all consumed noninitial spatial writes.

No work on import. The original campaign runtime, data, head, optimizer, loss
semantics and finalization are reused; only the direct-training write scope changes.
"""
import hashlib
import json
from pathlib import Path
import random
import sys
import time

BUDGET = 1168.509588667017
SAM2_BASE_SHA256 = "69a46b44e8625f509791352bd09beaafe589cd7d872721384e63d37f9fdc6e41"
train = None


def supported(runtime):
    model = runtime["model"]
    source = Path(model._prepare_memory_conditioned_features.__func__.__code__.co_filename)
    train.c.require(train.c.sha(source) == SAM2_BASE_SHA256, "pinned SAM2 memory selector")
    train.c.require(not model.training and model.num_maskmem == 7 and
        model.memory_temporal_stride_for_eval == 1 and model.max_cond_frames_in_attn == -1 and
        model.use_obj_ptrs_in_encoder and model.max_obj_ptrs_in_encoder == 16 and
        model.only_obj_ptrs_in_the_past_for_eval, "unsupported native memory convention")


def consumed(future):
    frame, history = future["frame"], future["history"]
    train.c.require(0 < frame < future["count"] and
        set(history["cond_frame_outputs"]) == {0} and
        set(history["non_cond_frame_outputs"]) == set(range(1, frame)), "full causal baseline history")
    # Native forward/eval stride1 keeps empty early slots; no backfill or cond0 duplication.
    return list(range(max(1, frame - 6), frame))


def active_loss(runtime, ex, future, head, family, head_module, cache, contexts, record):
    torch, model = runtime["torch"], runtime["model"]
    original = future["history"]
    history = dict(cond_frame_outputs=original["cond_frame_outputs"],
                   non_cond_frame_outputs=dict(original["non_cond_frame_outputs"]))
    selected, all_zero, changed_counts = consumed(future), True, []
    cached, _ = cache.history(ex["sequence"])
    for frame in selected:
        old = original["non_cond_frame_outputs"][frame]
        train.c.require(old is cached[frame]["out"], "unexpected nonbaseline context")
        key = ex["sequence"], frame
        if key not in contexts:
            context = family.prepare(old["maskmem_features"].float(), record["rho"], torch=torch)
            features = head_module.extract_input(context, old["object_score_logits"],
                cached[frame]["piou"], old["pred_masks"].sigmoid(), torch=torch)
            contexts[key] = context, features
            record["context_builds"] += 1
        context, features = contexts[key]
        # Recomputed for every future; no overlapping-write graph is reused after backward.
        q = train.head_coefficients(head, context, features, family, torch)
        all_zero = all_zero and float(q.detach().abs().max()) == 0
        base = old["maskmem_features"]
        changed = (base.float() + family.expand(context, q, torch=torch)).to(torch.bfloat16)
        history["non_cond_frame_outputs"][frame] = dict(old, maskmem_features=changed)
        changed_counts.append(int((changed.detach() != base).sum()))
        record["write_head_forward_calls"] += 1
        train.c.require(all(history["non_cond_frame_outputs"][frame][k] is v
                            for k, v in old.items() if k != "maskmem_features"), "carrier replaced")
    train.c.require(history["cond_frame_outputs"] is original["cond_frame_outputs"] and
        set(history["non_cond_frame_outputs"]) == set(original["non_cond_frame_outputs"]) and
        all(history["non_cond_frame_outputs"][i] is original["non_cond_frame_outputs"][i]
            for i in original["non_cond_frame_outputs"] if i not in selected), "history changed")
    capture = {}
    hook = model.sam_mask_decoder.register_forward_hook(lambda m, a, values: capture.update(values=values))
    try:
        result = model._track_step(future["frame"], False, future["vision"], future["pos"],
            future["sizes"], None, None, history, future["count"], False, None)
    finally:
        hook.remove()
    runtime["future_forward_calls"] = runtime.get("future_forward_calls", 0) + 1
    if all_zero:
        actual, reference = result[1][3], future["reference"]
        detail = dict(frame=future["frame"],
            max_abs=float((actual.detach().double()-reference.double()).abs().max()),
            binary_differences=int(((actual.detach()>0) != (reference>0)).sum()),
            grad_enabled=torch.is_grad_enabled(), atol=1e-4, rtol=1e-5, active_writes=selected)
        runtime.setdefault("zero_parity_details", []).append(detail)
        eligible = torch.allclose(actual, reference, atol=1e-4, rtol=1e-5) if torch.is_grad_enabled() else torch.equal(actual, reference)
        train.c.require(eligible, "all-zero active correction differs from native")
        runtime["zero_parity_checks"] = runtime.get("zero_parity_checks", 0) + 1
    # Same native-selected pre-presence mask and presence objective as train.future_loss.
    raw, quality, _, presence = capture["values"]
    index = int(quality[0].argmax())
    mask = torch.nn.functional.binary_cross_entropy_with_logits(raw[:, index:index+1], future["target"])
    visible = torch.nn.functional.binary_cross_entropy_with_logits(
        presence, torch.full_like(presence, float(future["presence"])))
    loss = mask + visible
    train.c.require(bool(torch.isfinite(loss)), "nonfinite all-active future loss")
    return loss, dict(frame=future["frame"], writes=selected, changed_bf16_entries=changed_counts)


def direct(spec, out, record, runtime, family, head_module, cache, operation_start):
    supported(runtime)
    torch = runtime["torch"]
    head = head_module.build_head(torch=torch)
    optimizer = torch.optim.Adam(head.parameters(), lr=.003)
    identifiers = [(seq, a) for seq in spec["train_sequences"] for a in cache.meta(seq)["anchors"]]
    train.c.require(len(identifiers) == 48 and len(set(identifiers)) == 48, "fixed 48 anchors")
    rng = random.Random(spec["seed"])
    record.update(checkpoints=[], rho=spec["rho"], write_scope="all_active_noninitial_spatial",
                  write_head_forward_calls=0, context_builds=0, sam2_selector_sha256=SAM2_BASE_SHA256)
    contexts = {}
    step, trained_seconds, slowest_step, stop = 0, 0., 15., False
    budget = spec["complete_cost_budget_seconds"]
    for epoch in range(100):
        order = identifiers.copy(); rng.shuffle(order)
        for sequence, anchor in order:
            if time.perf_counter()-operation_start + slowest_step + 5 >= budget:
                stop = True
                break
            began = time.perf_counter()
            ex = train.example(spec, cache, runtime, family, head_module, sequence, anchor)
            optimizer.zero_grad(set_to_none=True)
            loss_value, write_sets = 0., []
            for future in ex["futures"]:
                loss, detail = active_loss(runtime, ex, future, head, family, head_module,
                                          cache, contexts, record)
                loss = loss / 2
                loss_value += float(loss.detach()); write_sets.append(detail)
                loss.backward()
                del loss
            train.c.require(all(p.grad is None or bool(torch.isfinite(p.grad).all())
                                for p in head.parameters()), "head gradient nonfinite")
            optimizer.step(); step += 1
            elapsed = time.perf_counter()-began
            trained_seconds += elapsed; slowest_step = max(slowest_step, elapsed)
            row = dict(epoch=epoch+1, step=step, sequence=sequence, anchor=anchor, loss=loss_value,
                       update_seconds=elapsed, cumulative_training_seconds=trained_seconds,
                       active_contexts=write_sets)
            record["rows"].append(row); train.c.save(out/"receipt.json", record)
            print(json.dumps(row), flush=True)
        record["checkpoints"].append(train.save_checkpoint(
            out, "epoch-%02d" % (epoch+1), head, optimizer, step, torch))
        if stop:
            break
    train.c.require(step > 0, "no completed update")
    record.update(arm="direct_all_active", trained_seconds=trained_seconds,
                  complete_cost_budget_seconds=budget, comparison="cost_matched",
                  residual_scope="fixed baseline histories; not recurrent BPTT")


def main():
    global train
    spec = json.loads(Path(sys.argv[1]).read_text())
    directory = Path(spec["campaign_root"]).resolve()
    required = [Path(__file__).resolve(), directory/"train.py", directory/"common.py",
                Path(spec["family_root"])/"family.py", Path(spec["family_root"])/"head.py"]
    for path in required:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if spec["source_sha256"].get(str(path)) != digest:
            raise ValueError("adapter source missing/changed: " + str(path))
    if not (spec["mode"] == "direct" and spec["direct_comparison"] == "cost_matched" and
            spec["complete_cost_budget_seconds"] == BUDGET and spec["rho"] == .15 and spec["seed"] == 0 and
            spec["train_sequences"] == ["dog-agility", "planes-water", "dogs-scale", "dog-gooses"]):
        raise ValueError("conditional all-active diagnostic scope")
    sys.path.insert(0, str(directory))
    import train as campaign_train
    train = campaign_train
    original = train.direct
    try:
        train.direct = direct
        train.main()
    finally:
        train.direct = original


if __name__ == "__main__":
    main()
