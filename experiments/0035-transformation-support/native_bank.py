"""Source-prepared native bank capture; no model/data work on import.

CLI: --spec JSON. Runtime admission and process/resource supervision are external
to this worker. See NATIVE_BANK_SCOPE.md for the exact input/output contract.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


COHORT = (
    ("4691_cut_guava", 86), ("817_cut_onion", 90),
    ("9675_fold_clothes", 201), ("1934_tear_film", 69),
)
ARMS = ("B", "B+M1", "B+M2", "B+M1+M2")
COMMON_SHA256 = "751355a33b6ed468208d8f5aa92c8b4b72522f534159814928799e56634407c7"
COMPACT = {"maskmem_features", "maskmem_pos_enc", "pred_masks",
           "obj_ptr", "object_score_logits"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def streamed_state_hash(state, torch):
    """Exact qualified nf.state_hash descriptor; tensor bytes fed in 1MiB views."""
    def tensor_hash(value):
        value = value.detach().cpu().contiguous()
        raw = value.reshape(-1).view(torch.uint8).numpy()
        view = memoryview(raw)
        digest = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
        for start in range(0, len(view), 1048576):
            digest.update(view[start:start + 1048576])
        return digest.hexdigest()

    def describe(value):
        if isinstance(value, torch.Tensor):
            return {"tensor": tensor_hash(value)}
        if isinstance(value, dict):
            return [[describe(k), describe(v)] for k, v in value.items()]
        if isinstance(value, (list, tuple)):
            return [type(value).__name__, [describe(v) for v in value]]
        if isinstance(value, torch.device):
            return {"device": str(value)}
        if value is None or type(value) in (str, int, float, bool):
            return value
        raise TypeError("Unsupported native state value: " + type(value).__name__)
    return hashlib.sha256(json.dumps(describe(state), allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


class BoundedOutput:
    """Limit an existing binary file's extent; ZIP header seeks remain allowed."""
    def __init__(self, stream, limit):
        self.stream, self.limit = stream, limit

    def write(self, data):
        require(self.stream.tell() + len(data) <= self.limit, "NPZ output byte cap")
        return self.stream.write(data)

    def __getattr__(self, name):
        return getattr(self.stream, name)


def bound_file(item):
    require(type(item["bytes"]) is int and item["bytes"] > 0, "positive file size")
    path = Path(item["path"])
    require(path.is_absolute() and path.is_file(), "absolute existing input file")
    require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"],
            "input bytes changed: " + str(path))
    return path


def validate_sequence(sequence):
    """Metadata and opaque hashes only. Never accepts a later annotation list."""
    required = {"sequence", "role", "rgb", "initial"}
    require(required <= set(sequence) <= required | {"parent_id", "recording_id"},
            "sequence metadata keys")
    require(sequence["role"] == "TRAIN" and isinstance(sequence["sequence"], str)
            and sequence["sequence"] not in ("", ".", "..")
            and Path(sequence["sequence"]).name == sequence["sequence"], "sequence identity")
    rgb = sequence["rgb"]
    require(isinstance(rgb, list) and len(rgb) >= 2, "complete video descriptors")
    require([x["frame"] for x in rgb] == list(range(len(rgb)))
            and all(type(x["frame"]) is int for x in rgb), "ordered ordinal RGB frames")
    require(type(sequence["initial"]["frame"]) is int and sequence["initial"]["frame"] == 0,
            "only initial annotation allowed")
    for item in rgb:
        require(bound_file(item).suffix.lower() in (".jpg", ".jpeg"), "native JPEG input")
    require(bound_file(sequence["initial"]).suffix.lower() == ".png", "indexed initial PNG")


def initialize_record(output, spec_sha256, source_sha256, *, max_output_bytes=256 * 1024**2):
    """Pure record factory, also used by the parent's manufactured qualification."""
    require(type(max_output_bytes) is int and max_output_bytes > 0, "positive output cap")
    return dict(schema="evr0035.native-bank.v1", status="started",
        output=str(Path(output).absolute()), spec_sha256=spec_sha256,
        source_sha256=dict(source_sha256), monotonic_start=time.perf_counter(),
        max_output_bytes=max_output_bytes, bytes_written=0, partial_output_bytes=0,
        output_byte_scope="all completed/pending NPZ files; receipt/seal sizes checked separately within same cap",
        rows=[], outputs=[], timings=[], warnings=[], initial_label_decodes=[],
        noninitial_label_decodes=0, completed_frames=0, completed_sequences=0,
        calls={name + suffix: 0 for name in
               ("image_encoder", "track_step", "mask_decoder", "memory_encoder")
               for suffix in ("_intent", "_completed")},
        current_action=None, arm_names=list(ARMS), scientific_eligible=False)


def checkpoint(common, record):
    record["wall_seconds"] = time.perf_counter() - record["monotonic_start"]
    common.save(Path(record["output"]) / "receipt.json", record)
    receipt_bytes = (Path(record["output"]) / "receipt.json").stat().st_size
    if record["bytes_written"] + record["partial_output_bytes"] + receipt_bytes > record["max_output_bytes"]:
        record.update(status="failed", scientific_eligible=False, output_limit_exceeded=True)
        common.save(Path(record["output"]) / "receipt.json", record)
        raise ValueError("complete output byte cap including receipt")


def descriptor(common, root, path, kind, **metadata):
    return dict(path=str(path.relative_to(root)), bytes=path.stat().st_size,
                sha256=common.sha(path), kind=kind, **metadata)


def execute_sequence(runtime, common, sequence, output, record, *, compose_controls):
    """Capture one full clip; no fixed-cohort assertion here, no extra native call.

    `output` is a fresh clip directory inside record['output']; the production
    run() separately fixes all four names/lengths. This allows qualification on
    wholly manufactured clips without weakening the production population.
    """
    model, torch, np, Image = (runtime[k] for k in ("model", "torch", "np", "Image"))
    root, output = Path(record["output"]), Path(output)
    require(output.parent == root and not output.exists(), "fresh direct child clip output")
    validate_sequence(sequence)
    name, count = sequence["sequence"], len(sequence["rgb"])
    output.mkdir()
    row = dict(sequence=name, role=sequence["role"], frames=count, status="started",
               parent_id=sequence.get("parent_id"), recording_id=sequence.get("recording_id"),
               completed_frames=0, predictions=[], frame_records=[])
    record["rows"].append(row)
    began = time.perf_counter()
    before = dict(record["calls"])
    hooks, captured = [], {}
    state = iterator = None
    original_track = model._track_step

    def timed(name_, start, frame=None):
        record["timings"].append(dict(sequence=name, frame=frame, name=name_,
                                      seconds=time.perf_counter() - start))

    def finite(tensor, shape=None):
        require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu"
                and tensor.dtype == torch.float32
                and (shape is None or tuple(tensor.shape) == tuple(shape))
                and bool(torch.isfinite(tensor).all().item()), "native finite FP32 tensor contract")

    def entering(kind):
        def enter(_module, _args):
            record["calls"][kind + "_intent"] += 1
        return enter

    def completed(kind):
        def leave(_module, _args, _answer):
            record["calls"][kind + "_completed"] += 1
        return leave

    def observed(frame_idx, is_init_cond_frame, current_vision_feats,
                 current_vision_pos_embeds, feat_sizes, point_inputs, mask_inputs,
                 output_dict, num_frames, track_in_reverse, prev_sam_mask_logits):
        # The unchanged predictor passes the same per-object history dictionary
        # for both add_new_mask and ordinary propagation.
        matches = [i for i, history in state["output_dict_per_obj"].items()
                   if history is output_dict]
        require(len(matches) == 1, "unique native history-to-object association")
        object_id = int(state["obj_idx_to_id"][matches[0]])
        key = (int(frame_idx), object_id)
        require(key not in captured and 0 <= frame_idx < count and num_frames == count
                and not track_in_reverse and point_inputs is None and prev_sam_mask_logits is None
                and bool(is_init_cond_frame) == (frame_idx == 0)
                and (mask_inputs is not None) == (frame_idx == 0), "native route/capture identity")
        record["current_action"] = dict(sequence=name, frame=int(frame_idx),
                                        object_id=object_id, action="native_track_step")
        record["calls"]["track_step_intent"] += 1
        checkpoint(common, record)  # durable invocation intent, before native work
        answer = original_track(frame_idx, is_init_cond_frame, current_vision_feats,
            current_vision_pos_embeds, feat_sizes, point_inputs, mask_inputs,
            output_dict, num_frames, track_in_reverse, prev_sam_mask_logits)
        record["calls"]["track_step_completed"] += 1
        require(len(answer) == 4 and len(answer[1]) == 7, "native SAM result tuple")
        low, high, quality, selected_low, selected_high, pointer, presence = answer[1]
        bank_count = 1 if frame_idx == 0 else 3
        finite(low, (1, bank_count, 256, 256))
        finite(high, (1, bank_count, 1024, 1024))
        finite(quality, (1, bank_count))
        finite(selected_low, (1, 1, 256, 256))
        finite(selected_high, (1, 1, 1024, 1024))
        finite(pointer, (1, 256))
        finite(presence, (1, 1))
        selected = int(torch.argmax(quality[0]).item())
        require(torch.equal(selected_low, low[:, selected:selected + 1])
                and torch.equal(selected_high, high[:, selected:selected + 1]),
                "selected low/high differs from native argmax")
        captured[key] = dict(low=low.detach().clone(), quality=quality.detach().clone(),
            presence=presence.detach().clone(), selected=selected,
            alternatives=[i for i in range(bank_count) if i != selected],
            pointer_hash=runtime["nf"].tensor_hash(pointer, torch))
        # Never replace any returned value, selected output, pointer or history.
        return answer

    try:
        require(model.image_size == 1024 and model.num_maskmem == 7
                and model.memory_temporal_stride_for_eval == 1 and model.fill_hole_area == 0
                and not model.non_overlap_masks and not model.non_overlap_masks_for_mem_enc
                and model.use_mask_input_as_output_without_sam and model.use_obj_ptrs_in_encoder
                and model.multimask_output_in_sam and model.multimask_output_for_tracking
                and model.multimask_min_pt_num == 0 and model.multimask_max_pt_num == 1,
                "qualified native bank configuration")
        record["current_action"] = dict(sequence=name, frame=0, action="initial_annotation_decode")
        decode = dict(sequence=name, frame=0, sha256=sequence["initial"]["sha256"], status="started")
        record["initial_label_decodes"].append(decode)
        checkpoint(common, record)
        start = time.perf_counter()
        with Image.open(bound_file(sequence["initial"])) as image:
            require(image.format == "PNG" and image.mode in ("P", "L"), "indexed initial annotation")
            initial = np.array(image)
        require(initial.ndim == 2 and initial.dtype == np.uint8, "initial label type")
        object_ids = [int(v) for v in np.unique(initial) if 0 < int(v) < 255]
        require(object_ids and object_ids == list(range(1, object_ids[-1] + 1)),
                "positive initial IDs must be contiguous 1..max")
        shape, objects = tuple(initial.shape), len(object_ids)
        row.update(shape=list(shape), object_ids=object_ids, object_count=objects,
                   expected_object_frames=count * objects)
        decode.update(status="completed", object_ids=object_ids, original_255_pixels=int((initial == 255).sum()),
                      interpretation="255 is background; never a prompted object")
        # Native JPEG loader reports the final file's size, not a consistency
        # check over every file. Header checks add no model forward.
        for item in sequence["rgb"]:
            with Image.open(Path(item["path"])) as image:
                require(image.format == "JPEG" and image.size == (shape[1], shape[0]),
                        "all RGB dimensions must match the initial annotation")
        timed("initial_decode_and_header_checks", start)
        links = output / "native-input-links"
        links.mkdir()
        for item in sequence["rgb"]:
            (links / ("%05d.jpg" % item["frame"])).symlink_to(Path(item["path"]).resolve())
        model._track_step = observed
        for kind, module in (("image_encoder", model.image_encoder),
                             ("mask_decoder", model.sam_mask_decoder),
                             ("memory_encoder", model.memory_encoder)):
            hooks.extend((module.register_forward_pre_hook(entering(kind)),
                          module.register_forward_hook(completed(kind))))
        with torch.inference_mode(), runtime["guard"].capture_warnings(record):
            record["current_action"] = dict(sequence=name, action="native_init_state")
            checkpoint(common, record)
            start = time.perf_counter()
            state = model.init_state(str(links), offload_video_to_cpu=True,
                                     offload_state_to_cpu=True, async_loading_frames=False)
            require(state["num_frames"] == count and
                    (state["video_height"], state["video_width"]) == shape, "native video dimensions/count")
            timed("native_init_state", start)
            for link in links.iterdir():
                link.unlink()
            links.rmdir()
            start = time.perf_counter()
            for index, object_id in enumerate(object_ids):
                answer = model.add_new_mask(state, frame_idx=0, obj_id=object_id,
                                            mask=(initial == object_id))
                require(answer[0] == 0 and answer[1] == object_ids[:index + 1], "initial native ID order")
            del initial, answer
            timed("native_initial_prompts", start)
            iterator = iter(model.propagate_in_video(state))
            for frame in range(count):
                start = time.perf_counter()
                got, ids, original = next(iterator)
                require(got == frame and ids == object_ids, "complete native frame/ID order")
                finite(original, (objects, 1, *shape))
                timed("native_propagation", start, frame)
                start = time.perf_counter()
                bank_count = 1 if frame == 0 else 3
                capture = [captured.pop((frame, object_id)) for object_id in object_ids]
                require(not captured, "unconsumed capture from another native frame")
                selected_stack = []
                for index, item in enumerate(capture):
                    history = state["output_dict_per_obj"][index]
                    compact = history["cond_frame_outputs" if frame == 0 else "non_cond_frame_outputs"][frame]
                    require(set(compact) == COMPACT, "native compact output keys")
                    low = item["low"][:, item["selected"]:item["selected"] + 1]
                    require(torch.equal(low, compact["pred_masks"])
                            and torch.equal(item["presence"], compact["object_score_logits"])
                            and runtime["nf"].tensor_hash(compact["obj_ptr"], torch) == item["pointer_hash"],
                            "transparent capture/native stored state mismatch")
                    selected_stack.append(low)
                selected_stack = torch.cat(selected_stack, dim=0) if objects > 1 else selected_stack[0]
                _, resized_selected = model._get_orig_video_res_output(state, selected_stack)
                require(torch.equal(original, resized_selected), "same-stack original native B parity")
                # Every lane uses the native O×1 batch layout. The authoritative
                # selected lane remains the actual native yielded original.
                original_bank = np.empty((objects, 3, *shape), dtype=np.float32)
                original_bank[:, 0] = original[:, 0].cpu().numpy()
                for alternative in range(2):
                    if frame == 0:
                        original_bank[:, alternative + 1] = original_bank[:, 0]
                    else:
                        stack = torch.cat([item["low"][:, item["alternatives"][alternative]:
                                           item["alternatives"][alternative] + 1] for item in capture], dim=0)
                        _, resized = model._get_orig_video_res_output(state, stack)
                        finite(resized, (objects, 1, *shape))
                        original_bank[:, alternative + 1] = resized[:, 0].cpu().numpy()
                        del stack, resized
                maps = compose_controls(original_bank, np.asarray(object_ids, dtype=np.int64))
                require(set(maps) == set(ARMS), "four exact support arms")
                for value in maps.values():
                    require(value.shape == shape and value.dtype == np.int64
                            and np.isin(value, [0] + object_ids).all(), "categorical composition labels")
                if frame == 0:
                    require(all(np.array_equal(maps["B"], maps[a]) for a in ARMS),
                            "initial single bank must yield identical arms")
                frame_record = dict(sequence=name, frame=frame, bank_count=bank_count, selected_low_exact=True,
                    selected_original_exact=True,
                    selected_slots=[x["selected"] for x in capture],
                    alternative_slots=[x["alternatives"] for x in capture],
                    predicted_quality=[x["quality"][0].cpu().tolist() for x in capture],
                    object_score_logits=[float(x["presence"].item()) for x in capture],
                    native_positive_pixels=(original_bank[:, 0] > 0).sum(axis=(1, 2)).tolist(),
                    initial_diagnostic_only=(frame == 0))
                timed("parity_resize_and_composition", start, frame)
                start = time.perf_counter()
                path = output / ("frame-%05d.npz" % frame)
                pending = path.with_suffix(".npz.pending")
                labels64 = np.stack([maps[a] for a in ARMS])
                labels8 = labels64.astype(np.uint8)
                require(np.array_equal(labels64, labels8), "categorical uint8 conversion must be exact")
                remaining_bytes = (record["max_output_bytes"] - record["bytes_written"]
                                   - (root / "receipt.json").stat().st_size)
                require(remaining_bytes > 0, "output budget before NPZ")
                with pending.open("xb") as stream:
                    np.savez_compressed(BoundedOutput(stream, remaining_bytes),
                        low_bank=torch.cat([x["low"] for x in capture], dim=0).cpu().numpy(),
                        native_original_logits=original[:, 0].cpu().numpy(),
                        object_ids=np.asarray(object_ids, dtype=np.int64),
                        predicted_quality=np.asarray(frame_record["predicted_quality"], dtype=np.float32),
                        object_score_logits=np.asarray(frame_record["object_score_logits"], dtype=np.float32),
                        selected_slots=np.asarray(frame_record["selected_slots"], dtype=np.int64),
                        alternative_slots=np.asarray(frame_record["alternative_slots"], dtype=np.int64).reshape(objects, -1),
                        labels=labels8)
                os.replace(pending, path)
                item = descriptor(common, root, path, "native-bank-frame", sequence=name, frame=frame)
                record["bytes_written"] += item["bytes"]
                require(record["bytes_written"] <= record["max_output_bytes"], "output budget after NPZ")
                frame_record["output"] = item
                row["predictions"].append(item)
                record["outputs"].append(item)
                row["frame_records"].append(frame_record)
                row["completed_frames"] += 1
                record["completed_frames"] += 1
                timed("serialize_and_hash", start, frame)
                record["current_action"] = None
                checkpoint(common, record)
                # Do not retain high masks, vision features or a whole-video bank.
                del original_bank, maps, original, selected_stack, resized_selected, capture, compact, history, low
                del labels64, labels8
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise ValueError("native yielded an extra frame")
            start = time.perf_counter()
            row["final_native_state_sha256"] = streamed_state_hash(state, torch)
            timed("final_state_hash", start)
        row["calls"] = {key: record["calls"][key] - before[key] for key in before}
        for kind, expected in (("image_encoder", count), ("track_step", count * objects),
                               ("mask_decoder", count * objects), ("memory_encoder", count * objects)):
            require(row["calls"][kind + "_intent"] == row["calls"][kind + "_completed"] == expected,
                    "no extra native forwards: " + kind)
        require(row["completed_frames"] == count, "complete clip required")
        row.update(status="completed", seconds=time.perf_counter() - began)
        record["completed_sequences"] += 1
        checkpoint(common, record)
        return row
    except BaseException as error:
        record["partial_output_bytes"] += sum(p.stat().st_size for p in output.glob("*.npz.pending"))
        row.update(status="failed", error=dict(type=type(error).__name__, message=str(error)),
                   seconds=time.perf_counter() - began)
        checkpoint(common, record)
        raise
    finally:
        model._track_step = original_track
        for hook in hooks:
            hook.remove()
        captured.clear()
        if iterator is not None:
            iterator.close()
        state = iterator = None
        gc.collect()


def run(spec_path):
    spec_path = Path(spec_path).absolute()
    spec = json.loads(spec_path.read_text())
    common_path = Path(spec["common_path"]).absolute()
    sources = spec["source_sha256"]
    required = [Path(__file__).absolute(), common_path, Path(spec["metrics_path"]).absolute(),
                Path(__file__).with_name("NATIVE_BANK_SCOPE.md").absolute()]
    require(all(str(path) in sources and sha(path) == sources[str(path)] for path in required),
            "worker/common/metrics/scope source bindings")
    require(sha(common_path) == COMMON_SHA256, "unchanged qualified campaign common")
    require(all(sha(path) == digest for path, digest in sources.items()), "all source bindings")
    common = load_module("d35_native_common", common_path)
    out = Path(spec["output"]).absolute()
    require(not out.exists(), "fresh operation output")
    out.mkdir(parents=True)
    record = initialize_record(out, sha(spec_path), sources, max_output_bytes=spec["max_output_bytes"])
    record.update(data_index=spec["data_index"], fixed_cohort=[list(x) for x in COHORT],
                  initial_mask_semantics="IDs1..max; 0 and255 background",
                  output_composition="per-object included-slot max, global positive max, lowest-ID tie")
    checkpoint(common, record)
    try:
        start = time.perf_counter()
        index_path = bound_file(spec["data_index"])
        data = json.loads(index_path.read_text())
        sequences = data["sequences"]
        require([(s["sequence"], len(s["rgb"])) for s in sequences] == list(COHORT),
                "exact ordered four complete clips / 446 frames")
        for sequence in sequences:
            validate_sequence(sequence)
        record["timings"].append(dict(name="input_admission", seconds=time.perf_counter() - start))
        checkpoint(common, record)
        start = time.perf_counter()
        runtime = common.load_runtime(spec, record)
        metrics = load_module("d35_bank_metrics", Path(spec["metrics_path"]))
        record["timings"].append(dict(name="runtime_loading", seconds=time.perf_counter() - start))
        for sequence in sequences:
            execute_sequence(runtime, common, sequence, out / sequence["sequence"], record,
                             compose_controls=metrics.compose_controls)
        start = time.perf_counter()
        common.finish(runtime, record)
        require(record["completed_sequences"] == 4 and record["completed_frames"] == 446
                and record["calls"]["image_encoder_completed"] == 446
                and len(record["initial_label_decodes"]) == 4 and record["noninitial_label_decodes"] == 0,
                "complete fixed cohort/cost/initial-only label contract")
        require(sha(spec_path) == record["spec_sha256"], "operation specification changed")
        bound_file(spec["data_index"])
        for sequence in sequences:
            validate_sequence(sequence)
        require(all(sha(path) == digest for path, digest in sources.items()), "source changed during operation")
        record["timings"].append(dict(name="finish_and_input_rechecks", seconds=time.perf_counter() - start))
        record.update(status="completed", scientific_eligible=True, current_action=None)
    except BaseException as error:
        record.update(status="failed", scientific_eligible=False,
                      error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        checkpoint(common, record)
    try:
        seal = dict(schema="evr0035.native-bank-seal.v1", complete=True,
            receipt_sha256=sha(out / "receipt.json"), spec_sha256=record["spec_sha256"],
            source_sha256=sources, data_index=spec["data_index"], outputs=record["outputs"],
            completed_frames=446, completed_sequences=4)
        seal_bytes = len((json.dumps(seal, indent=2, allow_nan=False) + "\n").encode())
        require(record["bytes_written"] + (out / "receipt.json").stat().st_size
                + seal_bytes <= record["max_output_bytes"], "output budget before final seal")
        common.save(out / "seal.json", seal)
    except BaseException as error:
        record.update(status="failed", scientific_eligible=False,
                      error=dict(type=type(error).__name__, message=str(error)))
        checkpoint(common, record)
        raise
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    run(parser.parse_args().spec)
