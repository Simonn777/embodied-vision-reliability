"""Manufactured numerical equivalence of compressed/batched readouts; no task data."""
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
    record = dict(schema="evr0038.optimized-qualification.v1", status="started",
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
        optimized = module("d38_optimized", Path(__file__).with_name("optimized.py"))
        record["numeric_tolerance"] = dict(atol=1e-4, rtol=1e-5, selected_slot_exact=True, presence_sign_exact=True)
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
                        reference_stacks, reference_heads = [], []
                        old_stack = model.memory_attention.forward
                        old_head = model._forward_sam_heads
                        def stack_spy(*args, **kwargs):
                            value = old_stack(*args, **kwargs)
                            reference_stacks.append(value.detach().clone())
                            return value
                        def head_spy(*args, **kwargs):
                            value = old_head(*args, **kwargs)
                            reference_heads.append(adapter.clone_tree(value, torch))
                            return value
                        with adapter._replace_attribute(model.memory_attention, "forward", stack_spy):
                            with adapter._replace_attribute(model, "_forward_sam_heads", head_spy):
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
                        audit = {}
                        optimized_predictions, optimized_info = optimized.optimized_readouts(
                            model, torch, capture, writes, ops, adapter, audit_tensors=audit)
                        require(len(reference_stacks) == 5 and len(reference_heads) == 6,
                                "complete unbatched reference calls")
                        reference_stack_map = dict(zip(("slot0", "slot1", "slot2", "early", "flat"), reference_stacks))
                        reference_stack_map["late"] = ops.mean_stack_outputs(reference_stacks[:3], torch=torch)[0]
                        reference_head_map = dict(zip(("slot0", "slot1", "slot2", "early", "flat", "late"), reference_heads))
                        comparisons, selections = [], []
                        record["optimized"] = dict(metadata=optimized_info, comparisons=comparisons,
                            selections=selections, all_manufactured_low_masks_exact=False,
                            interpretation="prospective numeric tolerance on invented inputs only; no universal bitwise guarantee")
                        def compare(name, actual, expected):
                            require(actual.shape == expected.shape and actual.dtype == expected.dtype,
                                    name + " layout differs")
                            require(bool(torch.isfinite(actual).all()), name + " nonfinite")
                            maximum = float((actual-expected).abs().max().item())
                            okay = torch.allclose(actual, expected, atol=1e-4, rtol=1e-5)
                            comparisons.append(dict(name=name, max_abs_error=maximum, within_tolerance=bool(okay)))
                            require(okay, name + " exceeds prospective numeric tolerance")
                        for name, expected in reference_stack_map.items():
                            compare("stack_"+name, audit["stack_by_name"][name], expected)
                        for name, expected in reference_head_map.items():
                            actual = audit["sam_by_name"][name]
                            require(len(actual) == len(expected) == 7, "all SAM fields")
                            for field, (a,b) in enumerate(zip(actual,expected)):
                                compare("sam_"+name+"_"+str(field), a, b)
                            selection = dict(name=name,
                                actual_slot=torch.argmax(actual[2],dim=1).tolist(),
                                reference_slot=torch.argmax(expected[2],dim=1).tolist(),
                                slot_exact=bool(torch.equal(torch.argmax(actual[2],dim=1), torch.argmax(expected[2],dim=1))),
                                presence_sign_exact=bool(torch.equal(actual[6] > 0, expected[6] > 0)),
                                low_mask_exact=bool(torch.equal(actual[3] > 0, expected[3] > 0)))
                            selections.append(selection)
                            require(selection["slot_exact"], "native quality-selected slot changes")
                            require(selection["presence_sign_exact"], "presence sign changes")
                            require(selection["low_mask_exact"], "invented selected mask changes")
                        for name, expected in predictions.items():
                            compare("readout_"+name, optimized_predictions[name], expected)
                            require(torch.equal(optimized_predictions[name] > 0, expected > 0),
                                    "invented final low mask changes")
                        record["optimized"]["all_manufactured_low_masks_exact"] = True
                        save()
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
                                # Append only the common initial-pointer block;
                                # retain the native single variable t-1 pointer.
                                extra_common = c[key][-n:-n+4].repeat(8, 1, 1)
                                c[key] = torch.cat((c[key][:-n], c[key][-n:], extra_common), dim=0)
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
                            scope="one complete stack with invented repeated common pointer blocks; no complete-pipeline estimate")
                        require(record["maximum_flat_shape_probe"]["finite"], "finite maximum-shape output")
                        compressed, bias, compressed_info = optimized.compress_contexts(padded,
                            spatial_grid_tokens=4096, pointer_width=4, torch=torch, operators=ops)
                        t = time.perf_counter()
                        compressed_output = optimized.compressed_stack(model, torch, compressed, bias)
                        record["maximum_compressed_shape_probe"] = dict(**compressed_info,
                            stack_seconds=time.perf_counter()-t,
                            max_abs_error=float((compressed_output-max_out).abs().max().item()),
                            within_tolerance=bool(torch.allclose(compressed_output,max_out,atol=1e-4,rtol=1e-5)))
                        require(record["maximum_compressed_shape_probe"]["within_tolerance"],
                                "maximum pointer compressed stack numerical equivalence")
                        del audit, reference_stacks, reference_heads, reference_stack_map, reference_head_map
                        del optimized_predictions, compressed, bias, compressed_output
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
