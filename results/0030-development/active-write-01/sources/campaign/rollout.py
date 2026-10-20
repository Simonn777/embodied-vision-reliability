"""Full causal development rollouts with recomputed changed memory histories."""
import json
from pathlib import Path
import sys
import time
import common as c
import train


def run():
    spec, out, record, began = c.start(sys.argv[1], "own-rollout")
    runtime = None
    record.update(outputs=[], changed_writes=0, zero_parity_checks=0, frames=0)
    try:
        runtime = c.load_runtime(spec, record)
        torch, model, np = runtime["torch"], runtime["model"], runtime["np"]
        family, head_module = train.modules(spec)
        cache = train.Cache(spec["cache"], spec["cache_receipt_sha256"], torch)
        head = head_module.build_head(torch=torch).eval()
        if spec["arm"] != "native":
            checkpoint = spec["checkpoint"]
            c.require(c.sha(checkpoint["path"]) == checkpoint["sha256"], "checkpoint changed")
            weights = torch.load(checkpoint["path"], map_location="cpu", weights_only=True)
            head.load_state_dict(weights["head"], strict=True)
        else:
            c.require(spec.get("checkpoint") is None, "native has no learned checkpoint")
        head.requires_grad_(False)
        for sequence in spec["dev_sequences"]:
            meta = cache.meta(sequence)
            c.require(meta["role"] == "DEV", "own rollout restricted to DEV")
            shared = cache.tensor(sequence + "/shared.pt")
            initial = cache.tensor(sequence + "/frame-00000.pt")
            history = dict(cond_frame_outputs={0: initial["out"]}, non_cond_frame_outputs={})
            dest = out / sequence
            dest.mkdir()
            row = dict(sequence=sequence, parent_id=sequence, target_id=meta["target_id"],
                       shape=meta["shape"], frames=meta["frames"], predictions=[], writes=[])
            record["rows"].append(row)
            original = model._track_step
            quality = {}
            def observed(*args, **kwargs):
                result = original(*args, **kwargs)
                quality["value"] = float(result[1][2][0].max())
                return result
            model._track_step = observed
            try:
                with torch.inference_mode(), runtime["guard"].capture_warnings(record):
                    for frame in range(meta["frames"]):
                        item = initial if frame == 0 else cache.tensor(sequence + "/frame-%05d.pt" % frame)
                        if frame == 0:
                            current = initial["out"]
                        else:
                            current = model.track_step(frame_idx=frame, is_init_cond_frame=False,
                                current_vision_feats=item["vision"], current_vision_pos_embeds=shared["pos"],
                                feat_sizes=shared["sizes"], point_inputs=None, mask_inputs=None,
                                output_dict=history, num_frames=meta["frames"], track_in_reverse=False,
                                run_mem_encoder=True)
                            compact = {k: current[k] for k in ("maskmem_features", "maskmem_pos_enc",
                                "pred_masks", "obj_ptr", "object_score_logits")}
                            compact["maskmem_features"] = compact["maskmem_features"].to(torch.bfloat16)
                            current = compact
                            if spec["arm"] == "native":
                                for key in ("maskmem_features", "pred_masks", "obj_ptr", "object_score_logits"):
                                    c.require(torch.equal(current[key], item["out"][key]), "native own-rollout parity: " + key)
                                record["zero_parity_checks"] += 1
                            else:
                                base = current["maskmem_features"]
                                context = family.prepare(base, spec["rho"], torch=torch)
                                result = head_module.predict_coefficients(head, context,
                                    current["object_score_logits"], quality["value"], current["pred_masks"].sigmoid(),
                                    torch=torch, family=family)
                                delta = family.expand(context, result["coefficients"], torch=torch)
                                changed = (base.float() + delta).to(torch.bfloat16)
                                count = int((changed != base).sum())
                                current["maskmem_features"] = changed
                                record["changed_writes"] += int(count > 0)
                                row["writes"].append(dict(frame=frame, changed_bf16_entries=count,
                                    pre_bf16_rms=float(delta.double().square().mean().sqrt()),
                                    post_bf16_rms=float((changed.float()-base.float()).double().square().mean().sqrt())))
                            history["non_cond_frame_outputs"][frame] = current
                        logits = torch.nn.functional.interpolate(current["pred_masks"], size=meta["shape"],
                            mode="bilinear", align_corners=False)
                        c.require(bool(torch.isfinite(logits).all()), "nonfinite rollout logits")
                        path = dest / ("prediction-%05d.npz" % frame)
                        with path.open("xb") as stream:
                            np.savez_compressed(stream, masks=(logits[:, 0] > 0).cpu().numpy(),
                                                object_ids=np.array([meta["target_id"]], dtype=np.int64))
                        descriptor = dict(path=str(path.relative_to(out)), sha256=c.sha(path), bytes=path.stat().st_size,
                                          sequence=sequence, frame=frame)
                        record["outputs"].append(descriptor)
                        row["predictions"].append(descriptor)
                        record["frames"] += 1
                        c.save(out / "receipt.json", record)
            finally:
                model._track_step = original
            print(json.dumps(dict(sequence=sequence, frames=meta["frames"], status="completed")), flush=True)
        c.finish(runtime, record)
        c.require(record["frames"] == 154, "complete two-DEV coverage")
        for filename, expected in spec["source_sha256"].items():
            c.require(c.sha(filename) == expected, "source changed during rollout")
        record.update(status="completed", arm=spec["arm"], rho=spec["rho"])
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record["wall_seconds"] = time.perf_counter() - began
        c.save(out / "receipt.json", record)


if __name__ == "__main__":
    run()
