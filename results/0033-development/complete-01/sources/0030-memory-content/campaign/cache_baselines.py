"""Development baseline/cache builder; no native imports or work on import.

CLI: --spec JSON. spec.data_index is {path,sha256}; the index has `sequences`,
each {sequence,parent_id,role,rgb:[{frame,path,sha256,bytes}],labels:[same]}.
One operation processes all six development parents, one native video at a time.
Only TRAIN anchor/current/+1/+3 labels and each initial label are decoded.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import common

ROLES = {"dog-agility": "TRAIN", "planes-water": "TRAIN", "dogs-scale": "TRAIN",
         "dog-gooses": "TRAIN", "cat-girl": "DEV", "upside-down": "DEV"}
COMPACT = {"maskmem_features", "maskmem_pos_enc", "pred_masks", "obj_ptr",
           "object_score_logits"}


def anchors_for(frames):
    common.require(type(frames) is int and frames >= 16, "at least 16 frames")
    anchors = [1 + i * (frames - 5) // 11 for i in range(12)]
    common.require(len(set(anchors)) == 12 and anchors[-1] + 3 < frames, "anchor bounds")
    return anchors


def check_file(item):
    path = Path(item["path"])
    common.require(path.is_file() and path.stat().st_size == item["bytes"]
                   and common.sha(path) == item["sha256"], "input bytes changed: " + str(path))
    return path


def inputs(spec):
    descriptor = spec["data_index"]
    path = Path(descriptor["path"])
    common.require(common.sha(path) == descriptor["sha256"], "data index changed")
    data = json.loads(path.read_text())
    sequences = data["sequences"]
    common.require(len(sequences) == 6 and
        {x["sequence"] for x in sequences} == set(ROLES), "exact six parents required")
    for seq in sequences:
        name, role = seq["sequence"], seq["role"]
        common.require(seq["parent_id"] == name and role == ROLES[name], "parent/split mismatch")
        frames = len(seq["rgb"])
        common.require([x["frame"] for x in seq["rgb"]] == list(range(frames))
                       and [x["frame"] for x in seq["labels"]] == list(range(frames)),
                       "complete ordered frame metadata")
        anchors = anchors_for(frames) if role == "TRAIN" else []
        labels_used = {0} | {a + h for a in anchors for h in (0, 1, 3)}
        for item in seq["rgb"]:
            check_file(item)
        for frame in sorted(labels_used):
            check_file(seq["labels"][frame])
    common.require(sum(len(s["rgb"]) for s in sequences) == 386, "complete 386-frame cohort")
    return sequences


def finite_tensors(value, torch):
    if isinstance(value, torch.Tensor):
        common.require(value.device.type == "cpu" and
                       bool(torch.isfinite(value).all().item()), "nonfinite/non-CPU cached tensor")
    elif isinstance(value, dict):
        for entry in value.values():
            finite_tensors(entry, torch)
    elif isinstance(value, (list, tuple)):
        for entry in value:
            finite_tensors(entry, torch)


def run(spec_path):
    spec, out, record, began = common.start(spec_path, "baseline-cache")
    runtime = None
    record.update(outputs=[], label_decodes=[], image_encoder_calls=0, track_step_calls=0,
                  gt_memory_calls=0, completed_frames=0, completed_sequences=0,
                  complete_operation="all six sequential native baseline videos")

    def checkpoint():
        record["wall_seconds"] = time.perf_counter() - began
        common.save(out / "receipt.json", record)

    def output(path, kind, **metadata):
        row = dict(path=str(path.relative_to(out)), sha256=common.sha(path),
                   bytes=path.stat().st_size, kind=kind, **metadata)
        record["outputs"].append(row)
        return row

    try:
        sequences = inputs(spec)
        record["data_index"] = spec["data_index"]
        checkpoint()
        runtime = common.load_runtime(spec, record)
        model, torch, np, Image = (runtime[k] for k in ("model", "torch", "np", "Image"))
        F = torch.nn.functional
        common.require(model.num_maskmem == 7 and model.mem_dim == 64 and
                       model.memory_temporal_stride_for_eval == 1 and model.fill_hole_area == 0
                       and model.binarize_mask_from_pts_for_mem_enc, "native cache configuration")

        def save_tensor(path, value, **metadata):
            finite_tensors(value, torch)
            pending = path.with_suffix(".pt.pending")
            with pending.open("xb") as stream:
                torch.save(value, stream)
            os.replace(pending, path)
            return output(path, "tensor-cache", **metadata)

        def read_label(seq, frame, shape=None):
            item = seq["labels"][frame]
            entry = dict(sequence=seq["sequence"], frame=frame, sha256=item["sha256"],
                         status="started")
            record["label_decodes"].append(entry)
            checkpoint()
            path = check_file(item)
            with Image.open(path) as image:
                common.require(image.format == "PNG" and image.mode in ("P", "L"), "indexed label PNG")
                array = np.array(image)
            common.require(array.ndim == 2 and array.dtype == np.uint8 and
                           (shape is None or tuple(array.shape) == shape), "label dimensions/type")
            entry["status"] = "completed"
            return array

        for seq in sequences:
            seq_started = time.perf_counter()
            name, role, count = seq["sequence"], seq["role"], len(seq["rgb"])
            dest = out / name
            dest.mkdir()
            row = dict(sequence=name, parent_id=name, role=role, frames=count,
                       status="started", completed_frames=0, predictions=[])
            record["rows"].append(row)
            checkpoint()
            initial = read_label(seq, 0)
            ids = np.unique(initial)
            ids = ids[(ids > 0) & (ids != 255)]
            common.require(ids.size > 0, "initial target absent")
            target_id, shape = int(ids.min()), tuple(initial.shape)
            anchors = anchors_for(count) if role == "TRAIN" else []
            metadata = dict(frames=count, target_id=target_id, shape=list(shape), role=role,
                            anchors=anchors, sequence=name, parent_id=name)
            common.save(dest / "meta.json", metadata)
            output(dest / "meta.json", "sequence-metadata", sequence=name)
            # Temporary links supply the unchanged native JPEG loader; no RGB copies.
            links = dest / "native-input-links"
            links.mkdir()
            for item in seq["rgb"]:
                (links / ("%05d.jpg" % item["frame"])).symlink_to(Path(item["path"]).resolve())
            quality = {}
            original_track = model._track_step
            hooks = []
            encoder_before, steps_before = record["image_encoder_calls"], record["track_step_calls"]

            def encoded(_module, _args, _answer):
                record["image_encoder_calls"] += 1

            def observed(*args, **kwargs):
                answer = original_track(*args, **kwargs)
                frame = int(args[0] if args else kwargs["frame_idx"])
                sam = answer[1]
                common.require(len(sam) == 7 and sam[2].ndim == 2 and sam[2].shape[0] == 1,
                               "native quality bank")
                finite_tensors(sam, torch)
                quality[frame] = float(sam[2][0].max().item())
                record["track_step_calls"] += 1
                return answer

            model._track_step = observed
            hooks.append(model.image_encoder.register_forward_hook(encoded))
            try:
                with torch.inference_mode(), runtime["guard"].capture_warnings(record):
                    state = model.init_state(str(links), offload_video_to_cpu=True,
                        offload_state_to_cpu=True, async_loading_frames=False)
                    common.require(state["num_frames"] == count and
                        (state["video_height"], state["video_width"]) == shape, "native frame dimensions")
                    for link in links.iterdir():
                        link.unlink()
                    links.rmdir()
                    model.add_new_mask(state, frame_idx=0, obj_id=target_id, mask=(initial == target_id))
                    shared = None
                    for frame, object_ids, logits in model.propagate_in_video(state):
                        common.require(frame == row["completed_frames"] and object_ids == [target_id],
                                       "baseline order/object identity")
                        common.require(tuple(logits.shape) == (1, 1, *shape), "native original mask size")
                        finite_tensors(logits, torch)
                        _, _, vision, pos, sizes = model._get_image_feature(state, frame, 1)
                        common.require(len(vision) == len(pos) == len(sizes) == 3, "three feature levels")
                        finite_tensors(vision, torch)
                        # Earlier-level position tensors are not consumed by _track_step.
                        if shared is None:
                            shared = dict(pos=[None, None, pos[-1].detach().clone()],
                                          sizes=[tuple(s) for s in sizes])
                            save_tensor(dest / "shared.pt", shared, sequence=name)
                        else:
                            common.require(shared["sizes"] == [tuple(s) for s in sizes] and
                                torch.equal(shared["pos"][-1], pos[-1]), "shared image position changed")
                        history = state["output_dict_per_obj"][0]
                        compact = history["cond_frame_outputs" if frame == 0 else "non_cond_frame_outputs"][frame]
                        common.require(set(compact) == COMPACT and frame in quality, "compact baseline output")
                        common.require(compact["maskmem_features"].dtype == torch.bfloat16 and
                            tuple(compact["maskmem_features"].shape) == (1, 64, 64, 64), "native memory storage")
                        save_tensor(dest / ("frame-%05d.pt" % frame),
                            dict(vision=vision, out=compact, piou=quality[frame]), sequence=name, frame=frame)
                        pred = dest / ("prediction-%05d.npz" % frame)
                        with pred.open("xb") as stream:
                            np.savez_compressed(stream, masks=(logits[:, 0] > 0).cpu().numpy(),
                                                object_ids=np.array([target_id], dtype=np.int64))
                        row["predictions"].append(output(pred, "prediction", sequence=name, frame=frame))
                        if frame in anchors:
                            current = read_label(seq, frame, shape) == target_id
                            high = torch.from_numpy(current.copy()).float()[None, None]
                            # Match add_new_mask's original-to-model mask resizing.
                            high = F.interpolate(high, (model.image_size, model.image_size),
                                mode="bilinear", align_corners=False, antialias=True) >= .5
                            gt_memory, gt_pos = model._encode_new_memory(vision, sizes,
                                high.float() * 20 - 10,
                                torch.tensor([[10. if current.any() else -10.]], dtype=torch.float32),
                                is_mask_from_pts=True)
                            common.require(len(gt_pos) == len(compact["maskmem_pos_enc"]) and
                                all(torch.equal(a, b) for a, b in zip(gt_pos, compact["maskmem_pos_enc"])),
                                "GT memory position differs")
                            record["gt_memory_calls"] += 1
                            target_low, presence = {}, {}
                            for horizon in (1, 3):
                                truth = read_label(seq, frame + horizon, shape) == target_id
                                target_low[horizon] = F.interpolate(
                                    torch.from_numpy(truth.copy()).float()[None, None],
                                    (256, 256), mode="nearest")
                                presence[horizon] = int(truth.any())
                            save_tensor(dest / ("anchor-%05d.pt" % frame),
                                dict(target_low=target_low, presence=presence,
                                     gt_memory=gt_memory.to(torch.bfloat16)),
                                sequence=name, frame=frame)
                        row["completed_frames"] += 1
                        record["completed_frames"] += 1
                        checkpoint()
                    common.require(row["completed_frames"] == count and
                        record["image_encoder_calls"] - encoder_before == count and
                        record["track_step_calls"] - steps_before == count,
                        "complete baseline and no extra native image/track calls")
                    del state, vision, pos, compact, history, logits, shared
            finally:
                model._track_step = original_track
                for hook in hooks:
                    hook.remove()
            row.update(status="completed", target_id=target_id, shape=list(shape), anchors=anchors,
                       seconds=time.perf_counter() - seq_started)
            record["completed_sequences"] += 1
            checkpoint()
            gc.collect()
        common.finish(runtime, record)
        common.require(record["completed_frames"] == 386 and record["completed_sequences"] == 6
                       and record["gt_memory_calls"] == 48, "complete campaign cache")
        for filename, digest in spec["source_sha256"].items():
            common.require(common.sha(filename) == digest, "source changed during cache build")
        common.require(common.sha(spec["data_index"]["path"]) == spec["data_index"]["sha256"],
                       "data index changed during cache build")
        record["status"] = "completed"
    except BaseException as error:
        record.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        checkpoint()
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    run(parser.parse_args().spec)
