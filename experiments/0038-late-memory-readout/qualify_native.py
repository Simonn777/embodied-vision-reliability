"""Invented ten-frame native parity and full-context feasibility; no task data."""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys
import time


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def run(path):
    raw = Path(path).read_bytes()
    spec = json.loads(raw)
    common = module("d38_native_common", spec["common_path"])
    require = common.require
    out = Path(spec["output"])
    require(not out.exists(), "fresh output required")
    out.mkdir(parents=True)
    start = time.perf_counter()
    record = dict(schema="evr0038.native-qualification.v1", status="started",
        eligible=False, manufactured_only=True, scientific_payload_reads=0,
        spec_sha256=common.sha(path), source_sha256=spec["source_sha256"], warnings=[],
        native_frames=[], side_frames=[], native_passes=0)

    def save():
        record["wall_seconds"] = time.perf_counter() - start
        common.save(out / "receipt.json", record)

    save()
    try:
        runtime = common.load_runtime(spec, record)
        torch, np, Image = runtime["torch"], runtime["np"], runtime["Image"]
        model, nf = runtime["model"], runtime["nf"]
        adapter = module("d38_runtime_adapter", Path(__file__).with_name("runtime_adapter.py"))
        ops = module("d38_native_operators", Path(__file__).with_name("operators.py"))
        record["runtime_load_seconds"] = time.perf_counter() - start
        frames = out / "manufactured-inputs"
        frames.mkdir()
        y, x = np.indices((128, 192))
        initial = np.zeros((128, 192), dtype=bool)
        initial[38:81, 28:70] = True
        for frame in range(10):
            rgb = np.stack(((3*x + y + 2*frame) % 101,
                            (2*y + x//3) % 113, (x+3*y) % 97), -1).astype(np.uint8)
            rgb[38:81, 28+3*frame:70+3*frame] = [230, 50, 30]
            Image.fromarray(rgb).save(frames / ("%05d.jpg" % frame), quality=95, subsampling=0)
        native_logits = {}
        with torch.inference_mode(), runtime["guard"].capture_warnings(record):
            began = time.perf_counter()
            state = model.init_state(str(frames), offload_video_to_cpu=True,
                offload_state_to_cpu=True, async_loading_frames=False)
            prior = None
            with adapter.NativeObserver(model, torch) as observer:
                model.add_new_mask(state, frame_idx=0, obj_id=1, mask=initial)
                for frame, ids, logits in model.propagate_in_video(state):
                    require(ids == [1], "native object order")
                    capture = observer.pop(frame)
                    native_logits[frame] = logits.detach().cpu().clone()
                    record["native_frames"].append(dict(frame=frame,
                        logits_sha256=nf.tensor_hash(logits, torch),
                        capture_seconds=capture["capture_seconds"]))
                    if frame == 8:
                        require(prior["frame"] == 7, "causal previous capture")
                        history_hash = nf.state_hash(state, torch)
                        t = time.perf_counter()
                        writes, write_info = adapter.candidate_writes(model, torch, prior,
                            capture["history"]["non_cond_frame_outputs"][7])
                        predictions, read_info = adapter.readouts(model, torch, capture, writes, ops,
                            reuse_selected=False, identity_fastpath=False)
                        record["side_frames"].append(dict(frame=frame,
                            complete_seconds=time.perf_counter()-t,
                            candidate_writes=write_info, readouts=read_info))
                        require(set(predictions) == {"early_mean", "flat_histories", "late_mean",
                                                    "output_ensemble"}, "all four readouts")
                        require(all(tuple(v.shape) == (1, 1, 256, 256) and bool(torch.isfinite(v).all())
                                    for v in predictions.values()), "complete finite logits")
                        record["readout_hashes"] = {k: nf.tensor_hash(v, torch) for k,v in predictions.items()}
                        require(nf.state_hash(state, torch) == history_hash, "side readout mutated native state")
                        record["side_state_unchanged"] = True
                        # Manufacture the maximum pointer length in addition to the
                        # fully populated seven spatial memories. This is a shape/
                        # cost probe, not another clip or accuracy measurement.
                        contexts = [adapter.context_for_write(model, torch, capture, w) for w in writes]
                        padded = []
                        for context in contexts:
                            # context_for_write may return an accompanying audit dict.
                            if isinstance(context, tuple):
                                context = context[0]
                            c = adapter.clone_tree(context, torch)
                            n = c["num_obj_ptr_tokens"]
                            require(n == 32, "frame eight native pointer count")
                            for key in ("memory", "memory_pos"):
                                c[key] = torch.cat((c[key][:-n], c[key][-n:], c[key][-n:]), dim=0)
                            c["num_obj_ptr_tokens"] = 64
                            padded.append(c)
                        flat, flat_info = ops.flatten_contexts(padded, torch=torch, identity_fastpath=False)
                        require(flat_info["spatial_tokens"] == 3*7*4096 and
                                flat_info["pointer_tokens"] == 192, "maximum flat layout")
                        t = time.perf_counter()
                        max_out = model.memory_attention(**flat)
                        record["maximum_flat_shape_probe"] = dict(**flat_info,
                            stack_seconds=time.perf_counter()-t, output_shape=list(max_out.shape),
                            finite=bool(torch.isfinite(max_out).all()),
                            scope="one complete stack with invented repeated pointer tails; no complete-pipeline estimate")
                        require(record["maximum_flat_shape_probe"]["finite"], "finite maximum-shape output")
                        require(nf.state_hash(state, torch) == history_hash, "maximum-shape probe mutated state")
                        del writes, contexts, padded, flat, max_out, predictions
                    prior = capture
                    save()
                require(not observer.pending, "all native captures consumed")
            require(list(native_logits) == list(range(10)), "all ten native frames")
            first_hash = nf.state_hash(state, torch)
            record.update(observer_final_state_sha256=first_hash, native_passes=1,
                          observed_pass_seconds=time.perf_counter()-began)
            del state, prior, capture
            gc.collect()
            began = time.perf_counter()
            plain = model.init_state(str(frames), offload_video_to_cpu=True,
                offload_state_to_cpu=True, async_loading_frames=False)
            model.add_new_mask(plain, frame_idx=0, obj_id=1, mask=initial)
            parities = []
            for frame, ids, logits in model.propagate_in_video(plain):
                require(ids == [1] and torch.equal(logits, native_logits[frame]), "native trajectory differs")
                parities.append(dict(frame=frame, exact=True))
            require([r["frame"] for r in parities] == list(range(10)), "plain coverage")
            final_hash = nf.state_hash(plain, torch)
            require(first_hash == final_hash, "complete final native state differs")
            record.update(native_output_parities=parities, plain_final_state_sha256=final_hash,
                full_state_parity=True, native_passes=2, plain_pass_seconds=time.perf_counter()-began)
            del plain
        common.finish(runtime, record)
        require(all(common.sha(p) == h for p,h in spec["source_sha256"].items()), "bound source mutated")
        require(Path(path).read_bytes() == raw, "spec mutated")
        require(time.perf_counter()-start < 60, "complete qualification wall limit")
        record.update(status="completed", eligible=True,
            scientific_execution_admitted=False, scientific_result=False)
    except BaseException as exc:
        record.update(status="failed", eligible=False, error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        save()
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    result = run(parser.parse_args().spec)
    print(json.dumps({k: result[k] for k in ("status", "eligible", "wall_seconds")}), flush=True)
