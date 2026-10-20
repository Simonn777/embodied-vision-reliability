"""Bounded fixed-context memory teacher and same-head development controls."""
import json
from pathlib import Path
import random
import sys
import time
import common as c


def modules(spec):
    directory = Path(spec["family_root"])
    return c.module("campaign_family", directory / "family.py"), c.module("campaign_head", directory / "head.py")


class Cache:
    def __init__(self, root, expected, torch):
        self.root, self.torch = Path(root), torch
        c.require(c.sha(self.root / "receipt.json") == expected, "cache receipt changed")
        self.receipt = json.loads((self.root / "receipt.json").read_text())
        c.require(self.receipt["status"] == "completed", "cache incomplete")
        self.verified = set()
        self.files = {r["path"]: r for r in self.receipt["outputs"]}
        self.histories, self.shared = {}, {}

    def file(self, relative):
        row = self.files[relative]
        path = self.root / relative
        if relative not in self.verified:
            c.require(path.stat().st_size == row["bytes"] and c.sha(path) == row["sha256"], "cache file changed")
            self.verified.add(relative)
        return path

    def meta(self, sequence):
        return json.loads(self.file(sequence + "/meta.json").read_text())

    def tensor(self, relative):
        return self.torch.load(self.file(relative), map_location="cpu", weights_only=True)

    def history(self, sequence):
        if sequence not in self.histories:
            values = {}
            for frame in range(self.meta(sequence)["frames"]):
                item = self.tensor(sequence + "/frame-%05d.pt" % frame)
                values[frame] = dict(out=item["out"], piou=item["piou"])
            self.histories[sequence] = values
            self.shared[sequence] = self.tensor(sequence + "/shared.pt")
        return self.histories[sequence], self.shared[sequence]


def example(spec, cache, runtime, family, head_module, sequence, anchor):
    """Stream one anchor's image contexts; retain only compact memory history."""
    torch = runtime["torch"]
    meta = cache.meta(sequence)
    c.require(meta["role"] == "TRAIN" and anchor in meta["anchors"], "TRAIN anchor required")
    history, shared = cache.history(sequence)
    current = history[anchor]
    labels = cache.tensor(sequence + "/anchor-%05d.pt" % anchor)
    memory = current["out"]["maskmem_features"].float()
    context = family.prepare(memory, spec["rho"], torch=torch)
    features = head_module.extract_input(context, current["out"]["object_score_logits"],
        current["piou"], current["out"]["pred_masks"].sigmoid(), torch=torch)
    futures = []
    for horizon in (1, 3):
        frame = anchor + horizon
        item = cache.tensor(sequence + "/frame-%05d.pt" % frame)
        fixed = dict(cond_frame_outputs={0: history[0]["out"]},
            non_cond_frame_outputs={i: history[i]["out"] for i in range(1, frame)})
        futures.append(dict(frame=frame, vision=item["vision"], pos=shared["pos"],
            sizes=shared["sizes"], history=fixed, count=meta["frames"],
            target=labels["target_low"][horizon], presence=labels["presence"][horizon],
            reference=item["out"]["pred_masks"]))
    return dict(sequence=sequence, anchor=anchor, memory=memory, context=context,
        features=features, current=current, futures=futures, gt_memory=labels["gt_memory"])


def examples(spec, cache, runtime, family, head_module):
    for sequence in spec["train_sequences"]:
        for anchor in cache.meta(sequence)["anchors"]:
            yield example(spec, cache, runtime, family, head_module, sequence, anchor)


def future_loss(runtime, ex, coefficients, family, future):
    torch, model = runtime["torch"], runtime["model"]
    F = torch.nn.functional
    changed = (ex["memory"] + family.expand(ex["context"], coefficients, torch=torch)).to(torch.bfloat16)
    original = future["history"]
    history = dict(cond_frame_outputs=original["cond_frame_outputs"],
                   non_cond_frame_outputs=dict(original["non_cond_frame_outputs"]))
    history["non_cond_frame_outputs"][ex["anchor"]] = dict(
        history["non_cond_frame_outputs"][ex["anchor"]], maskmem_features=changed)
    capture = {}
    hook = model.sam_mask_decoder.register_forward_hook(lambda m, a, values: capture.update(values=values))
    try:
        result = model._track_step(future["frame"], False, future["vision"], future["pos"],
            future["sizes"], None, None, history, future["count"], False, None)
    finally:
        hook.remove()
    runtime["future_forward_calls"] = runtime.get("future_forward_calls", 0) + 1
    if float(coefficients.detach().abs().max()) == 0:
        actual, reference = result[1][3], future["reference"]
        difference = float((actual.detach().double() - reference.double()).abs().max())
        binary = int(((actual.detach() > 0) != (reference > 0)).sum())
        record = dict(frame=future["frame"], max_abs=difference, binary_differences=binary,
                      grad_enabled=torch.is_grad_enabled(), atol=1e-4, rtol=1e-5)
        runtime.setdefault("zero_parity_details", []).append(record)
        eligible = torch.allclose(actual, reference, atol=1e-4, rtol=1e-5) if torch.is_grad_enabled() else torch.equal(actual, reference)
        c.require(eligible, "zero correction differs from qualified native numerical convention")
        runtime["zero_parity_checks"] = runtime.get("zero_parity_checks", 0) + 1
    raw, quality, _, presence = capture["values"]
    index = int(quality[0].argmax())
    mask = F.binary_cross_entropy_with_logits(raw[:, index:index+1], future["target"])
    visible = F.binary_cross_entropy_with_logits(presence, torch.full_like(presence, float(future["presence"])))
    loss = mask + visible
    c.require(bool(torch.isfinite(loss)), "nonfinite future loss")
    return loss


def head_coefficients(head, context, features, family, torch):
    raw = head(features).reshape(64, 2)
    scale = (context["sigma"] * context["rho"]).float()
    return family.project(context, raw * scale[:, None], torch=torch)


def save_checkpoint(out, name, head, optimizer, step, torch):
    path = out / (name + ".pt")
    torch.save(dict(head=head.state_dict(), optimizer=optimizer.state_dict(), step=step), path)
    return dict(path=path.name, sha256=c.sha(path), bytes=path.stat().st_size, step=step)


def teacher(spec, out, record, runtime, family, head_module, cache):
    torch = runtime["torch"]
    targets = []
    for ex in examples(spec, cache, runtime, family, head_module):
        began = time.perf_counter()
        q = torch.zeros((64, 2), requires_grad=True)
        losses, best, one = [], None, None
        context = ex["context"]
        for step in range(6):
            value = 0.
            for future in ex["futures"]:
                with torch.set_grad_enabled(step < 5):
                    loss = future_loss(runtime, ex, q, family, future) / 2
                value += float(loss.detach())
                if step < 5:
                    loss.backward()
                del loss
            losses.append(value)
            if best is None or value < best[0]:
                best = value, q.detach().clone(), step
            if step <= 1 and (one is None or value < one[0]):
                one = value, q.detach().clone(), step
            if step < 5:
                c.require(q.grad is not None and bool(torch.isfinite(q.grad).all()), "teacher gradient")
                gradient = q.grad.double()
                active = context["active"]
                direction = torch.zeros_like(gradient)
                direction[active] = torch.linalg.solve(context["gram"][active], gradient[active, :, None])[:, :, 0]
                norm = torch.einsum("ci,cij,cj->", direction, context["gram"], direction).clamp_min(0).sqrt()
                update = direction * (context["radius"] / 3 / float(norm)) if float(norm) > 0 else direction
                q = family.project(context, (q.detach().double() - update).float(), torch=torch).detach().requires_grad_()
        gt = family.least_squares_target(context, ex["gt_memory"].float() - ex["memory"], torch=torch)["coefficients"]
        after = (ex["memory"] + family.expand(context, best[1], torch=torch)).to(torch.bfloat16)
        baseline = ex["memory"].to(torch.bfloat16)
        row = dict(sequence=ex["sequence"], anchor=ex["anchor"], losses=losses,
            best_step=best[2], one_step=one[2], baseline_loss=losses[0], selected_loss=best[0],
            changed_bf16_entries=int((after != baseline).sum()),
            post_bf16_rms=float((after.float()-baseline.float()).double().square().mean().sqrt()),
            wall_seconds=time.perf_counter()-began)
        record["rows"].append(row)
        targets.append(dict(sequence=ex["sequence"], anchor=ex["anchor"], context=context,
            features=ex["features"], multistep=best[1], one_step=one[1], gt_memory=gt))
        c.save(out / "receipt.json", record)
        print(json.dumps(row), flush=True)
    c.require(len(targets) == 48, "all four parents / 48 anchors required")
    path = out / "targets.pt"
    torch.save(targets, path)
    record["targets"] = dict(path=path.name, sha256=c.sha(path), bytes=path.stat().st_size)
    record["rho"] = spec["rho"]


def regress(spec, out, record, runtime, family, head_module):
    torch = runtime["torch"]
    path = Path(spec["targets"])
    c.require(c.sha(path) == spec["targets_sha256"], "teacher targets changed")
    data = torch.load(path, map_location="cpu", weights_only=True)
    c.require(len(data) == 48 and spec["arm"] in ("multistep", "one_step", "gt_memory"), "regression scope")
    c.require(all(x["context"]["rho"] == spec["rho"] and x["sequence"] in spec["train_sequences"] for x in data), "target family/split")
    head = head_module.build_head(torch=torch)
    optimizer = torch.optim.Adam(head.parameters(), lr=.003)
    record["checkpoints"] = []
    rng, step = random.Random(spec["seed"]), 0
    for epoch in range(6):
        order = list(range(len(data))); rng.shuffle(order)
        losses = []
        for index in order:
            item = data[index]
            optimizer.zero_grad(set_to_none=True)
            q = head_coefficients(head, item["context"], item["features"], family, torch)
            error = q - item[spec["arm"]]
            radius = item["context"]["radius"]
            loss = family.coefficient_energy(item["context"], error, torch=torch)
            loss = loss / radius**2 if radius else loss * 0
            c.require(bool(torch.isfinite(loss)), "regression loss nonfinite")
            loss.backward(); optimizer.step(); step += 1
            losses.append(float(loss.detach()))
        record["rows"].append(dict(epoch=epoch+1, steps=step, losses=losses, mean_loss=sum(losses)/len(losses)))
        record["checkpoints"].append(save_checkpoint(out, "epoch-%02d" % (epoch+1), head, optimizer, step, torch))
        c.save(out / "receipt.json", record)
    record["arm"], record["rho"] = spec["arm"], spec["rho"]


def direct(spec, out, record, runtime, family, head_module, cache, operation_start):
    torch = runtime["torch"]
    head = head_module.build_head(torch=torch)
    optimizer = torch.optim.Adam(head.parameters(), lr=.003)
    # Memory states/context construction is part of this arm's charged training.
    # Keep compact examples on disk order; reproducible shuffle uses identifiers.
    identifiers = [(seq, a) for seq in spec["train_sequences"] for a in cache.meta(seq)["anchors"]]
    rng = random.Random(spec["seed"])
    record["checkpoints"] = []
    step, trained_seconds, slowest_step = 0, 0., 15.
    budget = spec.get("complete_cost_budget_seconds")
    mode = spec["direct_comparison"]
    c.require(mode in ("update_matched", "cost_matched"), "direct comparison scope")
    c.require(mode == "update_matched" or 30 < budget <= 1750, "complete budget envelope")
    stop = False
    for epoch in range(100 if mode == "cost_matched" else 6):
        order = identifiers.copy(); rng.shuffle(order)
        for sequence, anchor in order:
            # Leave measured step and conservative final hash/receipt time. No
            # skipped or partial update is reported as trained. Final outer cost
            # still decides eligibility; this internal estimate is not proof.
            if mode == "cost_matched" and time.perf_counter()-operation_start + slowest_step + 5 >= budget:
                stop = True
                break
            began = time.perf_counter()
            ex = example(spec, cache, runtime, family, head_module, sequence, anchor)
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.
            for future in ex["futures"]:
                q = head_coefficients(head, ex["context"], ex["features"], family, torch)
                loss = future_loss(runtime, ex, q, family, future) / 2
                loss_value += float(loss.detach())
                loss.backward()
                del loss, q
            c.require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in head.parameters()), "head gradient nonfinite")
            optimizer.step(); step += 1
            elapsed = time.perf_counter() - began
            trained_seconds += elapsed
            slowest_step = max(slowest_step, elapsed)
            row = dict(epoch=epoch+1, step=step, sequence=sequence, anchor=anchor,
                       loss=loss_value, update_seconds=elapsed, cumulative_training_seconds=trained_seconds)
            record["rows"].append(row)
            c.save(out / "receipt.json", record)
            print(json.dumps(row), flush=True)
        record["checkpoints"].append(save_checkpoint(out, "epoch-%02d" % (epoch+1), head, optimizer, step, torch))
        if stop:
            break
    c.require(step > 0 and (mode != "update_matched" or step == 288), "incomplete direct comparison")
    record.update(arm="direct", rho=spec["rho"], trained_seconds=trained_seconds,
                  complete_cost_budget_seconds=budget, comparison=mode)


def main():
    spec, out, record, began = c.start(sys.argv[1], "training")
    runtime = None
    try:
        runtime = c.load_runtime(spec, record)
        family, head_module = modules(spec)
        record["head_configuration"] = head_module.configuration()
        if spec["mode"] == "regress":
            regress(spec, out, record, runtime, family, head_module)
        else:
            cache = Cache(spec["cache"], spec["cache_receipt_sha256"], runtime["torch"])
            c.require(spec["mode"] in ("teacher", "direct"), "mode")
            if spec["mode"] == "teacher":
                teacher(spec, out, record, runtime, family, head_module, cache)
            else:
                direct(spec, out, record, runtime, family, head_module, cache, began)
        c.finish(runtime, record)
        record["future_forward_calls"] = runtime.get("future_forward_calls", 0)
        record["zero_parity_checks"] = runtime.get("zero_parity_checks", 0)
        record["zero_parity_details"] = runtime.get("zero_parity_details", [])
        for filename, expected in spec["source_sha256"].items():
            c.require(c.sha(filename) == expected, "development source changed during operation")
        if spec["mode"] == "direct" and spec["direct_comparison"] == "cost_matched":
            c.require(time.perf_counter() - began <= spec["complete_cost_budget_seconds"], "complete inner direct cost exceeds budget")
            record["cost_comparison_eligibility"] = "pending full outer wall check, including startup and final receipts"
        record["status"] = "completed"
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record["wall_seconds"] = time.perf_counter() - began
        c.save(out / "receipt.json", record)


if __name__ == "__main__":
    main()
