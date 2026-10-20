"""Fixed two-arm supervised baseline. Import is stdlib-only; no native model."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

TRAIN = (("dog-agility", 25), ("planes-water", 38),
         ("dogs-scale", 83), ("dog-gooses", 86))
ARMS = ("visual", "logit_only")
BINDINGS = ("cache_receipt", "cache_outer", "cache_operation", "cache_worker_spec",
            "mapping", "data_index")
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def contract():
    return dict(train=[[name, n - 1] for name, n in TRAIN], train_frames=228,
        arms=list(ARMS), epochs=12, seed=0, batch_size=1, dtype="float32", device="cpu",
        optimizer="Adam", learning_rate=0.001, weight_decay=0.0001,
        betas=[0.9, 0.999], epsilon=1e-8, parameter_count=1377,
        loss="mean BCEWithLogits + 1-(2*sum(sigmoid(logits)*target)+1)/(sum(sigmoid(logits))+sum(target)+1)",
        resolution="original H,W; bilinear align_corners=False",
        normalization="all 228 TRAIN feature frames, population per-channel; float64 merged moments",
        std_floor=1e-6, residual="4*tanh", shuffle="random.Random(0), fresh list each epoch",
        no_op="native presence<=0 OR native original-resolution mask empty",
        no_op_loss="fixed native constant, no head forward/backward/optimizer step",
        aggregation="228-frame equal-weight online epoch loss; unequal parent counts",
        checkpoint="only terminal epoch12 per arm; no selection",
        maximum_whole_operation_seconds=1800)


def state_sha(state):
    """Logical state hash, independent of torch serialization container bytes."""
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        x = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([key, str(x.dtype), list(x.shape)],
                                 separators=(",", ":")).encode())
        digest.update(x.numpy().tobytes())
    return digest.hexdigest()


def loss_values(logits, target, torch):
    target = target.to(dtype=torch.float32)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    probability = logits.sigmoid()
    dice = 1 - (2 * (probability * target).sum() + 1) / (
        probability.sum() + target.sum() + 1)
    total = bce + dice
    if not bool(torch.isfinite(torch.stack((bce, dice, total))).all()):
        raise FloatingPointError("nonfinite BCE/Dice/total")
    return total, dict(bce=float(bce.detach()), dice=float(dice.detach()),
                       loss=float(total.detach()))


def run(spec_path):
    began = time.perf_counter()
    spec_path = Path(spec_path).resolve()
    preliminary = json.loads(spec_path.read_text())
    common_path = Path(preliminary["campaign_root"]).resolve() / "common.py"
    raw_sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if preliminary["source_sha256"].get(str(common_path)) != raw_sha(common_path):
        raise ValueError("serialization helper not source-bound")
    c = module("native_feature_common", common_path)
    spec, out, record, _ = c.start(spec_path, "native-feature-fit")
    record.update(schema="evr0033.native-feature-fit.v1", config=contract(),
        stage="admission", arms={}, outputs=[], parents=[], train_frames=[],
        model_calls=0, native_calls=0, image_encoder_calls=0, history_modified=False,
        dev_payload_access=False, initial_label_access=False, final_test_access=False,
        offline_attempts=[], compatibility_probes=[], tensor_loads=0, label_decodes=0,
        native_parity_checks=0, optimizer_steps=0, attempted_frames=0)
    fingerprints = dict(spec["source_sha256"])
    fingerprints[str(spec_path)] = record["spec_sha256"]
    log_path = out / "training-log.jsonl"

    def checkpoint():
        record["wall_seconds"] = time.perf_counter() - began
        c.save(out / "receipt.json", record)

    def bounded():
        c.require(time.perf_counter() - began < 1800, "whole fit exceeded 1800 seconds")

    def check(d):
        p = Path(d["path"]).resolve()
        c.require(p.is_file() and p.stat().st_size == d["bytes"]
                  and c.sha(p) == d["sha256"], "input changed: " + str(p))
        fingerprints[str(p)] = d["sha256"]
        return p

    def output(path, kind):
        d = dict(path=str(path.relative_to(out)), bytes=path.stat().st_size,
                 sha256=c.sha(path), kind=kind)
        record["outputs"].append(d)
        return d

    try:
        own = Path(__file__).resolve().parent
        evaluation = Path(spec["evaluation_root"]).resolve()
        guard_path = Path(spec["guard_source"]).resolve()
        required = {str(p) for p in (own / "train.py", own / "head.py", own / "SCOPE.md",
            own / "ADMISSION.md", common_path, common_path.with_name("cache_baselines.py"),
            evaluation / "evaluate_jf.py", evaluation / "evaluate_baseline.py",
            evaluation / "evaluate_rollout.py", guard_path)}
        c.require(set(spec["source_sha256"]) == required, "exact ten bound sources required")
        for path, digest in fingerprints.items():
            c.require(c.sha(path) == digest, "source/spec changed")
        c.require(set(spec["bindings"]) == set(BINDINGS) and spec.get("seed") == 0,
                  "six fixed metadata bindings and seed0 required")
        c.require(sys.flags.no_site == 1 and sys.flags.dont_write_bytecode == 1
                  and sys.version_info[:3] == (3, 12, 13)
                  and Path(sys.executable).absolute() == Path(spec["qualified_python"]).absolute()
                  and all(os.environ.get(k) == "4" for k in THREADS), "qualified runtime envelope")
        binding = {key: check(d) for key, d in spec["bindings"].items()}
        guard = module("native_feature_guard", guard_path)
        guard.install_guard(record["offline_attempts"], record["compatibility_probes"])
        import site
        site.main()
        import torch
        import numpy as np
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        c.require(torch.__version__.split("+")[0] == "2.8.0", "qualified torch2.8.0")
        sys.path.insert(0, str(evaluation))
        admission = module("native_feature_admission", evaluation / "evaluate_rollout.py")
        e = admission.e
        e.cv2.setNumThreads(0)
        head_module = module("native_feature_head", own / "head.py")
        record["runtime"] = dict(python=sys.version, torch=torch.__version__,
            numpy=np.__version__, threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(), native_model_loaded=False)
        with guard.capture_warnings(record):
            cache, _ = admission.admit(binding, "cache", fingerprints, legacy=True)
            mapping, index = (json.loads(binding[key].read_text()) for key in ("mapping", "data_index"))
            c.require(cache["schema"] == "evr0030.development.baseline-cache.v1"
                and cache["completed_frames"] == 386 and cache["completed_sequences"] == 6
                and not cache["offline_attempts"]
                and cache["model_initial_hash"] == cache["model_final_hash"]
                and mapping["status"] == "completed" and mapping["member_count"] == 386
                and cache["data_index"]["sha256"] == c.sha(binding["data_index"]),
                "complete unchanged native cache/mapping")
            bases = {r["sequence"]: r for r in cache["rows"]}
            inputs = {r["sequence"]: r for r in index["sequences"]}
            labels = {(r["sequence"], r["frame"]): r for r in mapping["mappings"]}
            files = {d["path"]: d for d in cache["outputs"]}
            c.require(len(bases) == len(cache["rows"]) == len(inputs) == len(index["sequences"]) == 6
                and set(bases) == set(inputs) == {n for n, _ in TRAIN} | {"cat-girl", "upside-down"}
                and len(labels) == len(mapping["mappings"]) == 386
                and len(files) == len(cache["outputs"]), "metadata duplicates or coverage")
            cache_root = binding["cache_receipt"].parent

            def cached(relative):
                p = (cache_root / relative).resolve()
                c.require(not Path(relative).is_absolute() and ".." not in Path(relative).parts
                          and p.is_relative_to(cache_root), "unsafe cache path")
                return check(dict(files[relative], path=str(p)))

            plans = []
            for name, n in TRAIN:
                base, source = bases[name], inputs[name]
                c.require(base["status"] == "completed" and base["frames"] == base["completed_frames"] == n
                    and base["role"] == source["role"] == "TRAIN"
                    and base["parent_id"] == source["parent_id"] == name
                    and [d["frame"] for d in base["predictions"]] == list(range(n))
                    and [d["frame"] for d in source["labels"]] == list(range(n))
                    and base["anchors"] == [1 + i * (n - 5) // 11 for i in range(12)],
                    "fixed TRAIN metadata")
                shape, target = tuple(base["shape"]), base["target_id"]
                c.require(len(shape) == 2 and min(shape) == 480 and
                    all(type(v) is int and v > 0 for v in shape)
                    and type(target) is int and 0 < target < 255, "native target/shape")
                record["parents"].append(dict(sequence=name, parent_id=name, role="TRAIN",
                    frames=n, fit_frames=n - 1, shape=list(shape), target_id=target))
                for frame in range(1, n):
                    label, pred = labels[name, frame], base["predictions"][frame]
                    c.require(label["role"] == "TRAIN"
                        and all(label[k] == source["labels"][frame][k] for k in ("path", "bytes", "sha256"))
                        and pred == files[pred["path"]] and pred["sequence"] == name,
                        "TRAIN label/prediction identity")
                    plans.append(dict(sequence=name, frame=frame, shape=shape, target_id=target,
                        tensor=cached(f"{name}/frame-{frame:05d}.pt"),
                        prediction=cached(pred["path"]), label=check(label)))
            c.require(len(plans) == 228, "all228 TRAIN frames required")
            record.update(bindings=spec["bindings"], source_cache_sha256=c.sha(binding["cache_receipt"]),
                cached_model_hash=cache["model_final_hash"], stage="cache_loading_and_native_parity",
                baseline_history_cost=dict(receipt=spec["bindings"]["cache_receipt"],
                    outer=spec["bindings"]["cache_outer"],
                    wall_seconds=json.loads(binding["cache_outer"].read_text())["wall_seconds"],
                    scope="inherited complete six-video native cache; not a free or matched pipeline"))
            checkpoint()

            data, count = [], 0
            mean = torch.zeros(32, dtype=torch.float64)
            m2 = torch.zeros_like(mean)
            for plan in plans:
                bounded()
                item = torch.load(plan["tensor"], map_location="cpu", weights_only=True)
                record["tensor_loads"] += 1
                raw, low, presence = item["vision"][0], item["out"]["pred_masks"], item["out"]["object_score_logits"]
                c.require(raw.device.type == low.device.type == presence.device.type == "cpu"
                    and raw.dtype == low.dtype == presence.dtype == torch.float32
                    and tuple(raw.shape) == (65536, 1, 32) and tuple(low.shape) == (1, 1, 256, 256)
                    and presence.numel() == 1 and all(bool(torch.isfinite(x).all()) for x in (raw, low, presence)),
                    "cached feature/logit/presence contract")
                feature = raw[:, 0].T.reshape(1, 32, 256, 256).contiguous().detach().clone()
                native = low.detach().clone()
                appearing = float(presence.reshape(-1)[0]) > 0
                c.require(appearing or bool((native == -1024).all()), "native hard absence changed")
                baseline = e._prediction(plan["prediction"], [plan["target_id"]], plan["shape"])[0]
                with torch.no_grad():
                    actual = head_module.original_logits(native, plan["shape"])[0, 0].numpy() > 0
                differences = int(np.count_nonzero(actual != baseline))
                c.require(differences == 0, "native zero-threshold parity failed")
                record["native_parity_checks"] += 1
                reason = "native_absent" if not appearing else "native_empty" if not baseline.any() else None
                row = dict(index=len(data), sequence=plan["sequence"], frame=plan["frame"],
                    target_id=plan["target_id"], shape=list(plan["shape"]), parity_differences=differences,
                    native_absent=not appearing, native_empty=not bool(baseline.any()), no_update_reason=reason)
                record["train_frames"].append(row)
                data.append(dict(feature=feature, native=native, shape=plan["shape"], metadata=row))
                values = feature.double().reshape(32, -1)
                var, local_mean = torch.var_mean(values, dim=1, correction=0)
                batch = values.shape[1]
                delta = local_mean - mean
                new_count = count + batch
                m2 = m2 + var * batch + delta.square() * (count * batch / new_count)
                mean = mean + delta * (batch / new_count)
                count = new_count
                del item, raw, low, presence, values, baseline, actual
                if len(data) % 24 == 0:
                    checkpoint()
            c.require(record["native_parity_checks"] == 228 and count == 228 * 65536
                and bool(torch.isfinite(mean).all()) and bool(torch.isfinite(m2).all())
                and bool((m2 >= 0).all()), "population moments/coverage")
            raw_std = (m2 / count).sqrt()
            mean32, std32 = mean.float(), raw_std.float().clamp_min(1e-6)
            normalization = dict(mean=mean32.tolist(), std=std32.tolist(), raw_std=raw_std.tolist(),
                source_frames=228, population_pixels=count, std_floor=1e-6,
                computation="float64 merged population moments; stored buffers float32",
                source_parents=[name for name, _ in TRAIN])
            c.save(out / "normalization.json", normalization)
            record["normalization"] = output(out / "normalization.json", "TRAIN-normalization")
            record.update(stage="TRAIN_labels", all_native_parities_before_labels=True)
            checkpoint()
            for item, plan in zip(data, plans):
                bounded()
                label = e._labels(plan["label"])
                record["label_decodes"] += 1
                c.require(tuple(label.shape) == plan["shape"], "original TRAIN label shape")
                item["target"] = torch.from_numpy((label == plan["target_id"]).copy())[None, None]
                item["metadata"]["void_pixels_as_background"] = int(np.count_nonzero(label == 255))
                if item["metadata"]["no_update_reason"]:
                    with torch.no_grad():
                        total, values = loss_values(head_module.original_logits(item["native"], item["shape"]),
                                                   item["target"], torch)
                    item["constant_loss"] = values
                    del total
                del label
            c.require(record["label_decodes"] == 228, "all228 TRAIN labels")

            rng, orders = random.Random(0), []
            for _ in range(12):
                order = list(range(228))
                rng.shuffle(order)
                orders.append(order)
            c.save(out / "orders.json", dict(seed=0, orders=orders,
                identities=[dict(sequence=p["sequence"], frame=p["frame"]) for p in plans]))
            record["orders"] = output(out / "orders.json", "shared-shuffles")
            torch.manual_seed(0)
            first = head_module.NativeFeatureHead(mean32, std32).float()
            initial = {k: v.detach().clone() for k, v in first.state_dict().items()}
            record["initial_state_sha256"] = state_sha(initial)
            del first
            active_count = sum(r["no_update_reason"] is None for r in record["train_frames"])
            record.update(active_frames=active_count, no_op_frames=228 - active_count, stage="fit")
            checkpoint()
            with log_path.open("x", buffering=1) as log:
                for arm in ARMS:
                    bounded()
                    arm_started = time.perf_counter()
                    torch.manual_seed(0)
                    head = head_module.NativeFeatureHead(mean32, std32, zero_features=arm == "logit_only").float()
                    head.load_state_dict(initial, strict=True)
                    c.require(sum(p.numel() for p in head.parameters()) == 1377
                        and state_sha(head.state_dict()) == record["initial_state_sha256"], "identical initial heads")
                    optimizer = torch.optim.Adam(head.parameters(), lr=0.001, weight_decay=0.0001,
                        betas=(0.9, 0.999), eps=1e-8, foreach=False, fused=False)
                    result = dict(status="started", epochs=0, optimizer_steps=0, attempted_frames=0,
                        zero_features=arm == "logit_only", epoch_summaries=[],
                        initial_state_sha256=record["initial_state_sha256"])
                    record["arms"][arm] = result
                    head.train()
                    for epoch, order in enumerate(orders, 1):
                        losses, active_losses = [], []
                        for position, index_ in enumerate(order):
                            bounded()
                            item = data[index_]
                            tick = time.perf_counter()
                            grad_norm, grad_nonzero = None, 0
                            reason = item["metadata"]["no_update_reason"]
                            if reason:
                                values = item["constant_loss"]
                            else:
                                optimizer.zero_grad(set_to_none=True)
                                low = head(item["feature"], item["native"])
                                logits = head_module.original_logits(low, item["shape"])
                                loss, values = loss_values(logits, item["target"], torch)
                                loss.backward()
                                c.require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                                              for p in head.parameters()), "nonfinite/missing head gradient")
                                grad_norm = math.sqrt(math.fsum(float(p.grad.double().square().sum())
                                                               for p in head.parameters()))
                                grad_nonzero = sum(int(torch.count_nonzero(p.grad)) for p in head.parameters())
                                c.require(math.isfinite(grad_norm), "nonfinite gradient norm")
                                optimizer.step()
                                c.require(all(bool(torch.isfinite(p).all()) for p in head.parameters())
                                    and all(not isinstance(v, torch.Tensor) or bool(torch.isfinite(v).all())
                                        for state in optimizer.state.values() for v in state.values()),
                                    "nonfinite optimizer/parameter state")
                                result["optimizer_steps"] += 1
                                record["optimizer_steps"] += 1
                                active_losses.append(values["loss"])
                                del low, logits, loss
                            losses.append(values["loss"])
                            result["attempted_frames"] += 1
                            record["attempted_frames"] += 1
                            event = dict(arm=arm, epoch=epoch, position=position, index=index_,
                                sequence=item["metadata"]["sequence"], frame=item["metadata"]["frame"],
                                no_update_reason=reason, optimizer_step=result["optimizer_steps"],
                                gradient_l2=grad_norm, gradient_nonzero_entries=grad_nonzero,
                                step_wall_seconds=time.perf_counter() - tick, **values)
                            log.write(json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n")
                            if record["attempted_frames"] % 24 == 0:
                                checkpoint()
                        result["epochs"] = epoch
                        result["epoch_summaries"].append(dict(epoch=epoch, frames=228,
                            active_frames=active_count, online_frame_mean=math.fsum(losses) / 228,
                            online_active_mean=math.fsum(active_losses) / len(active_losses) if active_losses else None))
                        checkpoint()
                    c.require(result["optimizer_steps"] == 12 * active_count
                        and result["attempted_frames"] == 2736 and result["epochs"] == 12, "complete fixed fit")
                    path = out / (arm + "-terminal.pt")
                    with path.open("xb") as stream:
                        torch.save(dict(schema="evr0033.native-feature-checkpoint.v1", arm=arm,
                            state_dict=head.state_dict(), normalization=dict(mean=mean32, std=std32),
                            config=contract(), epochs=12, optimizer_steps=result["optimizer_steps"],
                            initial_state_sha256=record["initial_state_sha256"]), stream)
                    result.update(status="completed", terminal_checkpoint=output(path, "terminal-head"),
                        final_state_sha256=state_sha(head.state_dict()),
                        wall_seconds=time.perf_counter() - arm_started)
                    del optimizer, head
                    checkpoint()
            record["training_log"] = output(log_path, "all5472-attempts")
            c.require(record["attempted_frames"] == 5472 and record["optimizer_steps"] == 24 * active_count
                and tuple(record["arms"]) == ARMS and all(r["status"] == "completed" for r in record["arms"].values()),
                "both complete arms required")
            record["retained_tensor_bytes"] = sum(x[k].numel() * x[k].element_size()
                for x in data for k in ("feature", "native", "target"))
            record["stage"] = "final_immutability_checks"
            checkpoint()
            for path, digest in fingerprints.items():
                c.require(c.sha(path) == digest, "source/input changed during fit: " + path)
            c.require(not record["offline_attempts"] and not any(n == "sam2" or n.startswith("sam2.")
                      for n in sys.modules), "offline/model-import violation")
            bounded()
            record.update(status="completed", stage="complete", input_sha256=fingerprints,
                          completed_train_frames=228, completed_arms=2)
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if log_path.exists() and "training_log" not in record:
            record["partial_training_log"] = dict(path=log_path.name, bytes=log_path.stat().st_size,
                                                 sha256=c.sha(log_path))
        checkpoint()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -B -S -W error train.py SPEC.json")
    run(sys.argv[1])
