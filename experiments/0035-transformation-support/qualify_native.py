"""Complete two-object manufactured native-capture parity; no scientific input."""
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
    spec = json.loads(Path(path).read_text())
    common = module("bank_qualification_common", spec["common_path"])
    require = common.require
    output = Path(spec["output"])
    require(not output.exists(), "fresh manufactured output required")
    output.mkdir(parents=True)
    began = time.perf_counter()
    record = dict(schema="evr0035.native-qualification.v1", status="started",
                  manufactured_only=True, scientific_input_access=False,
                  spec_sha256=common.sha(path), source_sha256=spec["source_sha256"],
                  eligible=False, warnings=[])

    def save():
        record["wall_seconds"] = time.perf_counter() - began
        common.save(output / "receipt.json", record)

    save()
    try:
        runtime = common.load_runtime(spec, record)
        np, torch, Image = runtime["np"], runtime["torch"], runtime["Image"]
        worker = module("qualified_bank_worker", spec["worker_path"])
        metrics = module("qualified_bank_metrics", spec["metrics_path"])
        frames = output / "manufactured-inputs"
        frames.mkdir()
        shape = (128, 192)
        y, x = np.indices(shape)
        initial = np.zeros(shape, dtype=np.uint8)
        initial[30:61, 20:51] = 1
        initial[75:102, 128:165] = 2
        initial_path = output / "manufactured-initial.png"
        Image.fromarray(initial).save(initial_path)

        def desc(p, frame):
            return dict(frame=frame, path=str(p), bytes=p.stat().st_size, sha256=common.sha(p))

        rgb = []
        for frame in range(7):
            image = np.stack(((x * 3 + y + frame * 2) % 101,
                              (y * 2 + x // 3) % 113, (x + y * 3) % 97), axis=-1).astype(np.uint8)
            image[30:61, 20 + 3*frame:51 + 3*frame] = [230, 50, 30]
            image[75-frame:102-frame, 128-2*frame:165-2*frame] = [30, 220, 230]
            p = frames / ("%05d.jpg" % frame)
            Image.fromarray(image).save(p, quality=95, subsampling=0)
            rgb.append(desc(p, frame))
        sequence = dict(sequence="manufactured_two_objects", role="TRAIN", rgb=rgb,
                        initial=desc(initial_path, 0))
        capture_root = output / "captured-operation"
        capture_root.mkdir()
        captured = worker.initialize_record(capture_root, common.sha(path), spec["source_sha256"],
                                            max_output_bytes=256 * 1024**2)
        captured["manufactured_only"] = True
        with torch.inference_mode(), runtime["guard"].capture_warnings(record):
            row = worker.execute_sequence(runtime, common, sequence, capture_root / "capture", captured,
                                          compose_controls=metrics.compose_controls)
            captured["status"] = "completed"
            worker.checkpoint(common, captured)
            common.save(output / "capture-receipt.json", captured)
            record["capture_sequence"] = row
            record["capture_receipt_sha256"] = common.sha(output / "capture-receipt.json")
            gc.collect()
            # The second pass deliberately uses no observer. Its native outputs
            # and full state establish that capture leaves both histories intact.
            model = runtime["model"]
            state = model.init_state(str(frames), offload_video_to_cpu=True,
                                     offload_state_to_cpu=True, async_loading_frames=False)
            for object_id in (1, 2):
                model.add_new_mask(state, frame_idx=0, obj_id=object_id, mask=initial == object_id)
            parities = []
            for frame, ids, logits in model.propagate_in_video(state):
                require(ids == [1, 2], "plain native object order")
                matches = [item for item in row["frame_records"] if item["frame"] == frame
                           and item["sequence"] == sequence["sequence"]]
                require(len(matches) == 1, "complete unique captured frame")
                item = matches[0]
                # Native worker exposes the archive path in each row; all data
                # here are generated in this qualification operation.
                archive_path = capture_root / item["output"]["path"]
                with np.load(archive_path, allow_pickle=False) as archive:
                    observed = archive["native_original_logits"]
                expected = logits[:, 0].detach().cpu().numpy()
                require(np.array_equal(observed, expected), "captured/plain native trajectory differs")
                parities.append(dict(frame=frame, objects=ids, exact=True,
                                     native_logits_sha256=runtime["nf"].tensor_hash(logits, torch)))
            require([r["frame"] for r in parities] == list(range(7)), "plain full coverage")
            final_hash = runtime["nf"].state_hash(state, torch)
            require(final_hash == row["final_native_state_sha256"], "capture mutated complete native state")
            record.update(native_output_parities=parities, plain_final_state_sha256=final_hash,
                          full_state_parity=True)
            del state
        common.finish(runtime, record)
        require(all(common.sha(p) == digest for p, digest in spec["source_sha256"].items()),
                "qualification source changed")
        require(time.perf_counter() - began < 1800, "qualification wall limit")
        record.update(status="completed", eligible=True)
    except BaseException as error:
        record.update(status="failed", eligible=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save()
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()
    result = run(args.spec)
    print(json.dumps({k: result[k] for k in ("status", "eligible", "wall_seconds")}), flush=True)
