"""Source-prepared, single-operation D38 prediction -> seal -> evaluation.

CLI: qualified Python -B -S run_complete.py SPEC.json. This file does not grant
execution permission. The default and maximum accepted complete-operation limit
is 1800 seconds; no subdivision, retry or optimized/compressed implementation.
Only the supplied first annotations are decoded during prediction. Existing GT
bytes may be hashed by the old admission before the offline guard; their labels
are decoded by evaluate_complete only after the immutable 446-frame seal.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

COHORT = (("4691_cut_guava", 86), ("817_cut_onion", 90),
          ("9675_fold_clothes", 201), ("1934_tear_film", 69))
ARMS = ("early_mean", "flat_histories", "late_mean", "output_ensemble")
BINDINGS = ("d35_spec", "d35_result", "d35_receipt", "d35_outer", "qualification_spec",
            "qualification_receipt", "qualification_outer", "qualification_outer_spec", "d37_spec",
            "d37_receipt", "d37_outer", "d37_seal", "d37_review", "d37_review_outer")
COMMON_SHA = "751355a33b6ed468208d8f5aa92c8b4b72522f534159814928799e56634407c7"
ADAPTER_SHA = "21e47d941d8518b387e2695f4a0863112adcb8c0505a487be2bb825f6fd62db9"
OPERATORS_SHA = "e18ad43a606776ab204c162e2cdf6b16d69bf4801491308ec18c6636a559a56e"
QUALIFIER_SHA = "1fc12075b8e3dc5f1dfd504bcc4ef06275497405c39388a2f87f0040031b1f1a"
D35_SPEC_SHA = "325961b52bed72c8710fee107265b02545002f9b7591fdd4119095841611a68d"
D35_RESULT_SHA = "7fe9592298d1cced864eaa6f75faf973e98c10ea3ce8c91100a678cc6cd60fe3"
CALL_KINDS = ("image_encoder", "mask_decoder", "memory_encoder", "memory_attention", "pointer_projection")
PHASES = ("native", "candidate_writes", "readouts")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def module(name, path):
    loader = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(loader)
    sys.modules[name] = value
    loader.loader.exec_module(value)
    return value


def load_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 32 * 1024**2,
            "bounded regular JSON required")
    return json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def checkpoint(common, out, record, *, enforce=True):
    record["wall_seconds"] = time.perf_counter() - record["monotonic_start"]
    common.save(out / "receipt.json", record)
    if enforce:
        require(record["wall_seconds"] < record["operation_wall_seconds"], "complete-operation wall budget")
        metadata_bytes = sum(p.stat().st_size for p in out.glob("*.json*"))
        require(record["bytes_written"] + record["partial_output_bytes"] + metadata_bytes
                <= record["max_output_bytes"], "complete output byte budget")


def admission(spec, record, common, admission35, here):
    """All subprocess-backed old admission occurs before the offline guard."""
    fingerprints = {}
    # Prior scientific results are opaque hash bindings here. Their numerical
    # contents are decoded only by the evaluator after the new prediction seal.
    bound = {key: admission35.bound_json(spec[key], fingerprints)
             for key in ("d35_spec", "d35_receipt", "qualification_spec", "qualification_receipt", "qualification_outer_spec")}
    for key in BINDINGS:
        item = spec[key]
        path = Path(item["path"])
        require(path.is_absolute() and path.is_file() and not path.is_symlink() and
                path.stat().st_size <= 32 * 1024**2 and sha(path) == item["sha256"],
                "opaque reference binding: " + key)
        fingerprints[str(path)] = item["sha256"]
    require(spec["d35_spec"]["sha256"] == D35_SPEC_SHA and
            spec["d35_result"]["sha256"] == D35_RESULT_SHA, "fixed completed D35 reference")
    oldspec = bound["d35_spec"]
    context = admission35.evaluation_context(oldspec)
    context["fingerprints"].update(fingerprints)
    oldreceipt = bound["d35_receipt"]
    result_path = Path(spec["d35_result"]["path"])
    require(Path(spec["d35_receipt"]["path"]) == result_path.parent / "receipt.json" and
            result_path.name == "evaluation.json" and
            oldreceipt["schema"] == "evr0035.evaluation.v1" and oldreceipt["status"] == "completed" and
            oldreceipt["spec_sha256"] == D35_SPEC_SHA and
            oldreceipt["source_sha256"] == oldspec["source_sha256"] and
            all(oldreceipt[k] == 446 for k in ("completed_frames", "parity_frames", "label_decodes", "label_decode_attempts")) and
            oldreceipt["completed_sequences"] == 4 and not oldreceipt["offline_attempts"] and
            oldreceipt["outputs"] == [dict(path="evaluation.json", sha256=D35_RESULT_SHA,
                                           bytes=result_path.stat().st_size)],
            "complete D35 receipt must bind the exact numerical result")
    admission35.clean_outer(spec["d35_outer"], spec["d35_receipt"]["sha256"], context["fingerprints"])
    require(oldspec["repo"] == spec["repo"] and oldspec["native_root"] == spec["native_root"]
            and oldspec["native_commit"] == spec["native_commit"], "same qualified native runtime")
    old_sources = context["prediction_receipt"]["source_sha256"]
    require(len(old_sources) == 70 and all(spec["source_sha256"].get(p) == h
            for p, h in old_sources.items()), "all original 70 scientific files unchanged")
    qspec, qualified = bound["qualification_spec"], bound["qualification_receipt"]
    outer = admission35.clean_outer(spec["qualification_outer"],
        spec["qualification_receipt"]["sha256"], context["fingerprints"])
    require(qualified["schema"] == "evr0038.native-qualification.v1" and
            qualified["status"] == "completed" and qualified["eligible"] is True and
            qualified["manufactured_only"] is True and qualified["scientific_payload_reads"] == 0 and
            qualified["spec_sha256"] == spec["qualification_spec"]["sha256"] and
            qualified["source_sha256"] == qspec["source_sha256"] and
            qualified["full_state_parity"] is True and qualified["side_state_unchanged"] is True and
            qualified["native_passes"] == 2 and
            qualified["observer_final_state_sha256"] == qualified["plain_final_state_sha256"] and
            [r["frame"] for r in qualified["native_output_parities"]] == list(range(10)) and
            all(r["exact"] is True for r in qualified["native_output_parities"]),
            "complete uncompressed manufactured qualification required")
    sides = qualified["side_frames"]
    require(len(sides) == 1 and sides[0]["frame"] == 8 and
            all(sides[0]["readouts"][key] is True for key in
                ("selected_context_exact", "selected_stack_exact", "selected_sam_exact")) and
            sides[0]["readouts"]["calls"] == dict(context_materializations=4, attention_stacks=5, sam_heads=6),
            "selected full-stack/all-SAM-field reuse was qualified")
    for name, digest in (("runtime_adapter.py", ADAPTER_SHA), ("operators.py", OPERATORS_SHA),
                         ("qualify_native.py", QUALIFIER_SHA)):
        key = str(here / name)
        require(qspec["source_sha256"].get(key) == spec["source_sha256"].get(key) == digest,
                "exact qualified uncompressed helper: " + name)
    for key in ("repo", "native_root", "native_commit", "common_path"):
        require(qspec[key] == spec[key], "qualification/runtime identity: " + key)
    qouter = bound["qualification_outer_spec"]
    require(Path(qspec["output"]) == Path(spec["qualification_receipt"]["path"]).parent and
            outer["spec_sha256"] == spec["qualification_outer_spec"]["sha256"] and
            qouter["command"] == [str(Path(context["installation"]) / "venv/bin/python"),
                "-B", "-S", str(here / "qualify_native.py"), "--spec", spec["qualification_spec"]["path"]] and
            qouter["repo"] == spec["repo"] and qouter["commit"] == outer["expected_commit"] and
            qouter["timeout"] == outer["hard_timeout_seconds"] == 60 and
            Path(qouter["operation_output"]) == Path(spec["qualification_outer"]["path"]).parent and
            qouter["result_file"] == spec["qualification_receipt"]["path"],
            "qualification raw argv/spec/output linkage (outer argv is redacted)")
    data_index = context["prediction_spec"]["data_index"]
    data = admission35.bound_json(data_index, context["fingerprints"])
    require([(s["sequence"], len(s["rgb"])) for s in data["sequences"]] == list(COHORT) and
            all(r["object_ids"] == [1] for r in context["prediction_receipt"]["rows"]),
            "all four complete one-initial-object clips")
    record.update(bindings={**{k: spec[k] for k in BINDINGS}, "data_index": data_index},
        data_index=data_index, opaque_gt_provenance_checked_before_prediction=True,
        opaque_gt_check_semantics="encoded bytes only; no label values decoded",
        selected_reuse_qualification=spec["qualification_receipt"],
        d35_native_reference=context["proof"]["predict_receipt"])
    return context, data["sequences"]


def finite(value, torch, shape):
    require(isinstance(value, torch.Tensor) and value.device.type == "cpu" and
            value.dtype == torch.float32 and tuple(value.shape) == tuple(shape) and
            bool(torch.isfinite(value).all()), "finite CPU FP32 tensor layout")


def compare_native(model, runtime, state, capture, original, oldrow, frame, context, native35, evaluate35, metrics):
    """Exact new native bank, yielded output, compact write and four old controls."""
    np, torch = runtime["np"], runtime["torch"]
    oldframe = oldrow["frame_records"][frame]
    path = evaluate35.frame_path(context["prediction_root"], oldrow["predictions"][frame])
    restored, controls = evaluate35.reconstruct_frame(path, oldrow, oldframe,
        np=np, torch=torch, metrics=metrics)
    with np.load(path, allow_pickle=False) as saved:
        low = saved["low_bank"]
        native_original = saved["native_original_logits"]
        quality = saved["predicted_quality"]
        presence = saved["object_score_logits"]
    sam = capture["sam_outputs"]
    require(np.array_equal(sam[0].numpy(), low) and
            np.array_equal(sam[2].numpy(), quality) and
            np.array_equal(sam[6].numpy().reshape(1), presence) and
            capture["selected_index"] == oldframe["selected_slots"][0], "D35 native bank/quality/presence parity")
    require(np.array_equal(original[:, 0].numpy(), native_original), "D35 yielded native original parity")
    bucket = "cond_frame_outputs" if frame == 0 else "non_cond_frame_outputs"
    compact = state["output_dict_per_obj"][0][bucket][frame]
    require(set(compact) == native35.COMPACT and torch.equal(compact["pred_masks"], sam[3]) and
            torch.equal(compact["obj_ptr"], sam[5]) and
            torch.equal(compact["object_score_logits"], sam[6]), "transparent compact native write")
    _, selected_original = model._get_orig_video_res_output(state, sam[3])
    require(torch.equal(original, selected_original), "native selected O-by-one resize parity")
    fresh = np.empty_like(restored)
    fresh[:, 0] = original[:, 0].numpy()
    alternatives = [j for j in range(sam[0].shape[1]) if j != capture["selected_index"]]
    require(alternatives == oldframe["alternative_slots"][0], "original alternative order")
    for j in range(2):
        if frame == 0:
            fresh[:, j + 1] = fresh[:, 0]
        else:
            _, value = model._get_orig_video_res_output(state, sam[0][:, alternatives[j]:alternatives[j] + 1])
            fresh[:, j + 1] = value[:, 0].numpy()
    fresh_controls = metrics.compose_controls(fresh, [1])
    require(np.array_equal(fresh, restored) and all(np.array_equal(fresh_controls[a], controls[a])
            for a in metrics.CONTROLS), "all original native controls exact")
    return dict(bank_exact=True, output_exact=True, input_exact=True,
                controls_exact=True, stored_write_exact=True,
                d35_output=oldrow["predictions"][frame])


def execute_sequence(runtime, common, sequence, oldrow, context, out, record, modules):
    native35, evaluate35, metrics, adapter, operators = modules
    model, torch, np, Image = (runtime[k] for k in ("model", "torch", "np", "Image"))
    native35.validate_sequence(sequence)
    name, count, shape = sequence["sequence"], len(sequence["rgb"]), tuple(oldrow["shape"])
    clip = out / name
    require(not clip.exists(), "fresh complete clip output")
    clip.mkdir()
    info = dict(sequence=name, frames=count, shape=list(shape), object_ids=[1],
        status="started", completed_frames=0, final_state_exact=False,
        expected_final_native_state_sha256=oldrow["final_native_state_sha256"],
        input_rgb=sequence["rgb"], input_initial=sequence["initial"], predictions=[])
    record["sequences"].append(info)
    start, before = time.perf_counter(), json.loads(json.dumps(record["calls"]))
    current = {"phase": "native", "frame": None}
    handles, links = [], clip / "native-input-links"
    state = iterator = prior = capture = None
    old_track, own_track = model._track_step, "_track_step" in model.__dict__
    old_attr = model.__dict__.get("_track_step")

    def mark(action, frame=None):
        record["current_action"] = dict(sequence=name, frame=frame, action=action)
        checkpoint(common, out, record)

    def observed_track(*args, **kwargs):
        frame = kwargs.get("frame_idx", args[0] if args else None)
        record["track_step_intent"] += 1
        mark("native_track_step", frame)
        answer = old_track(*args, **kwargs)
        record["track_step_completed"] += 1
        return answer

    def hook(kind, suffix):
        def called(*_args):
            record["calls"][current["phase"]][kind + suffix] += 1
        return called

    try:
        require(model.image_size == 1024 and model.num_maskmem == 7 and
                model.memory_temporal_stride_for_eval == 1 and model.fill_hole_area == 0 and
                not model.non_overlap_masks and not model.non_overlap_masks_for_mem_enc and
                model.use_mask_input_as_output_without_sam and model.use_obj_ptrs_in_encoder and
                model.multimask_output_in_sam and model.multimask_output_for_tracking and
                model.multimask_min_pt_num == 0 and model.multimask_max_pt_num == 1,
                "unchanged qualified native settings")
        mark("initial_annotation_decode", 0)
        with Image.open(native35.bound_file(sequence["initial"])) as image:
            require(image.format == "PNG" and image.mode in ("P", "L"), "indexed initial annotation")
            initial = np.array(image)
        require(initial.dtype == np.uint8 and initial.shape == shape and
                [int(x) for x in np.unique(initial) if 0 < int(x) < 255] == [1],
                "same initial dimensions and complete one-object population")
        record["initial_label_decodes"] += 1
        for item in sequence["rgb"]:
            with Image.open(item["path"]) as image:
                require(image.format == "JPEG" and image.size == (shape[1], shape[0]), "RGB dimensions")
        links.mkdir()
        for item in sequence["rgb"]:
            (links / ("%05d.jpg" % item["frame"])).symlink_to(Path(item["path"]).resolve())
        for kind, obj in (("image_encoder", model.image_encoder), ("mask_decoder", model.sam_mask_decoder),
                          ("memory_encoder", model.memory_encoder), ("memory_attention", model.memory_attention),
                          ("pointer_projection", model.obj_ptr_proj)):
            handles.extend((obj.register_forward_pre_hook(hook(kind, "_intent")),
                            obj.register_forward_hook(hook(kind, "_completed"))))
        model._track_step = observed_track
        with torch.inference_mode(), runtime["guard"].capture_warnings(record):
            mark("native_init_state")
            state = model.init_state(str(links), offload_video_to_cpu=True,
                offload_state_to_cpu=True, async_loading_frames=False)
            require(state["num_frames"] == count and
                    (state["video_height"], state["video_width"]) == shape, "complete native video")
            for link in links.iterdir():
                link.unlink()
            links.rmdir()
            with adapter.NativeObserver(model, torch) as observer:
                mark("initial_mask_prompt", 0)
                answer = model.add_new_mask(state, frame_idx=0, obj_id=1, mask=(initial == 1))
                require(answer[0] == 0 and answer[1] == [1], "native initial prompt output")
                del initial, answer
                iterator = iter(model.propagate_in_video(state))
                for frame in range(count):
                    current.update(phase="native", frame=frame)
                    mark("native_next_frame", frame)
                    began = time.perf_counter()
                    got, ids, original = next(iterator)
                    require(got == frame and ids == [1], "complete ordered native frames")
                    finite(original, torch, (1, 1, *shape))
                    capture = observer.pop(frame)
                    require(not observer.pending, "unconsumed native capture")
                    native_seconds = time.perf_counter() - began
                    mark("native_reference_parity", frame)
                    began = time.perf_counter()
                    parity = compare_native(model, runtime, state, capture, original,
                        oldrow, frame, context, native35, evaluate35, metrics)
                    parity_seconds = time.perf_counter() - began
                    write_info = None
                    if frame >= 2:
                        require(prior["frame"] == frame - 1, "causal most recent noninitial write")
                        current["phase"] = "candidate_writes"
                        mark("two_alternative_memory_encodings", frame)
                        writes, write_info = adapter.candidate_writes(model, torch, prior,
                            capture["history"]["non_cond_frame_outputs"][frame - 1])
                        require(write_info["calls"] == dict(memory_encoders=2, pointer_projections=2),
                                "exact candidate encoding/projection work")
                    else:
                        writes = None
                    current["phase"] = "readouts"
                    mark("four_full_uncompressed_readouts", frame)
                    low, read_info = adapter.readouts(model, torch, capture, writes, operators,
                        reuse_selected=True, selected_branch_qualified=True, identity_fastpath=False)
                    expected = dict(context_materializations=4, attention_stacks=4, sam_heads=5) if frame >= 2 else dict(
                        context_materializations=0, attention_stacks=0, sam_heads=0)
                    require(read_info["calls"] == expected and not read_info["identity_fastpath_used"],
                            "fixed complete uncompressed cost; no identity shortcut")
                    if frame >= 2:
                        require(all(read_info[k] is True for k in
                            ("selected_context_exact", "selected_stack_exact", "selected_sam_exact")),
                            "qualified selected reuse exact current context")
                        record["eligible_side_frames"] += 1
                    require(tuple(low) == ARMS, "fixed readout ordering")
                    mark("resize_pack_and_save", frame)
                    began = time.perf_counter()
                    masks = []
                    for arm in ARMS:
                        finite(low[arm], torch, (1, 1, 256, 256))
                        _, resized = model._get_orig_video_res_output(state, low[arm])
                        finite(resized, torch, (1, 1, *shape))
                        if frame < 2:
                            require(torch.equal(low[arm], capture["sam_outputs"][3]) and
                                    torch.equal(resized, original), "first two frames exact native no-op")
                        masks.append(np.packbits((resized[:, 0].numpy() > 0).reshape(1, -1), axis=-1, bitorder="big"))
                    arrays = dict(low_logits=np.stack([low[a][0].numpy() for a in ARMS]),
                        packed_masks=np.stack(masks), object_ids=np.asarray([1], dtype=np.int64),
                        shape=np.asarray(shape, dtype=np.int64))
                    path = clip / ("frame-%05d.npz" % frame)
                    pending = path.with_suffix(".npz.pending")
                    # Eight MiB is reserved for immutable manifests/live diagnostics.
                    remaining = record["max_output_bytes"] - record["bytes_written"] - 8 * 1024**2
                    require(remaining > 0, "output budget before frame archive")
                    with pending.open("xb") as stream:
                        np.savez_compressed(native35.BoundedOutput(stream, remaining), **arrays)
                    os.replace(pending, path)
                    desc = native35.descriptor(common, out, path, "d38-readout-frame", sequence=name, frame=frame)
                    record["bytes_written"] += desc["bytes"]
                    row = dict(sequence=name, frame=frame, filename=Path(sequence["rgb"][frame]["path"]).name,
                        shape=list(shape), object_ids=[1], output=desc, native_parity=parity,
                        native_seconds=native_seconds, parity_seconds=parity_seconds,
                        capture_seconds=capture["capture_seconds"], candidate_writes=write_info,
                        readouts=read_info, serialization_seconds=time.perf_counter() - began)
                    record["rows"].append(row)
                    record["outputs"].append(desc)
                    info["predictions"].append(desc)
                    info["completed_frames"] += 1
                    record["completed_frames"] += 1
                    record["current_action"] = None
                    checkpoint(common, out, record)
                    prior = capture
                    del writes, low, arrays, masks, resized, original
                current["phase"] = "native"
                try:
                    next(iterator)
                except StopIteration:
                    pass
                else:
                    raise ValueError("extra native frame")
                require(not observer.pending, "all captures consumed")
            mark("complete_native_state_parity")
            info["final_native_state_sha256"] = native35.streamed_state_hash(state, torch)
            require(info["final_native_state_sha256"] == info["expected_final_native_state_sha256"],
                    "entire native final state differs from D35")
            info["final_state_exact"] = True
        info["calls"] = {phase: {key: value - before[phase][key] for key, value in counts.items()}
                         for phase, counts in record["calls"].items()}
        verify_counts(info["calls"], count)
        info.update(status="completed", seconds=time.perf_counter() - start)
        record["completed_sequences"] += 1
        checkpoint(common, out, record)
    except BaseException as error:
        record["partial_output_bytes"] += sum(p.stat().st_size for p in clip.glob("*.npz.pending"))
        info.update(status="failed", seconds=time.perf_counter() - start,
                    error=dict(type=type(error).__name__, message=str(error)))
        checkpoint(common, out, record, enforce=False)
        raise
    finally:
        for handle in handles:
            handle.remove()
        if own_track:
            model._track_step = old_attr
        else:
            model.__dict__.pop("_track_step", None)
        if iterator is not None:
            iterator.close()
        state = iterator = prior = capture = None
        gc.collect()


def verify_counts(calls, count):
    side = count - 2
    expected = {
        "native": dict(image_encoder=count, mask_decoder=count, memory_encoder=count,
                       memory_attention=count - 1, pointer_projection=count),
        "candidate_writes": dict(image_encoder=0, mask_decoder=0, memory_encoder=2 * side,
                                 memory_attention=0, pointer_projection=2 * side),
        # Capture-only materialization dispatches call the module but not its
        # native stack; four real stacks plus four captures are observed here.
        "readouts": dict(image_encoder=0, mask_decoder=5 * side, memory_encoder=0,
                         memory_attention=8 * side, pointer_projection=5 * side),
    }
    for phase in PHASES:
        for kind in CALL_KINDS:
            require(calls[phase][kind + "_intent"] == calls[phase][kind + "_completed"]
                    == expected[phase][kind], "complete measured call accounting: " + phase + "/" + kind)


def run(spec_path):
    spec_path = Path(spec_path).resolve()
    spec_hash, spec = sha(spec_path), load_json(spec_path)
    require(spec.get("launch_admitted") is True and spec.get("preparation_only", False) is False,
            "prepared specification is not admitted for execution")
    here, repo = Path(__file__).resolve().parent, Path(spec["repo"]).resolve()
    out, common_path = Path(spec["output"]).absolute(), Path(spec["common_path"]).resolve()
    sources = spec["source_sha256"]
    limit = spec.get("operation_wall_seconds", 1800)
    require(type(limit) in (int, float) and math.isfinite(limit) and 0 < limit <= 1800,
            "complete operation remains limited to 1800 seconds")
    require(type(spec["max_output_bytes"]) is int and 16 * 1024**2 <= spec["max_output_bytes"] <= 2 * 1024**3,
            "bounded complete generated output cap")
    require(spec.get("seed", 0) == 0 and spec.get("identity_fastpath", False) is False,
            "fixed deterministic seed and full-work readouts")
    old = repo / "experiments/0035-transformation-support"
    require(Path(spec["protocol_path"]).resolve() == here / "FULL_COMPARISON.md",
            "fixed complete-comparison protocol")
    needed = [Path(__file__).resolve(), here / "runtime_adapter.py", here / "operators.py",
        here / "evaluate_complete.py", here / "ADMISSION.md", Path(spec["protocol_path"]).resolve(),
        common_path, *(old / x for x in ("admit.py", "native_bank.py", "evaluate.py", "metrics.py"))]
    require(all(str(p) in sources and sha(p) == sources[str(p)] for p in needed), "complete executor source closure")
    require(sha(common_path) == COMMON_SHA and sha(here / "runtime_adapter.py") == ADAPTER_SHA and
            sha(here / "operators.py") == OPERATORS_SHA, "unchanged qualified runtime/operators")
    require(all(Path(p).is_absolute() and sha(p) == h for p, h in sources.items()), "all source bindings")
    require(not out.exists(), "fresh single-operation output")
    out.mkdir(parents=True)
    common = module("d38_complete_common", common_path)
    record = dict(schema="evr0038.complete-operation.v1", status="started", eligible=False,
        scientific_eligible=False, clean_outer_required=True, output=str(out), spec_sha256=spec_hash,
        source_sha256=sources, monotonic_start=time.perf_counter(), operation_wall_seconds=limit,
        max_output_bytes=spec["max_output_bytes"], bytes_written=0, partial_output_bytes=0,
        phase="admission", current_action=None, warnings=[], offline_attempts=[], compatibility_probes=[],
        rows=[], sequences=[], outputs=[], arms=list(ARMS), completed_frames=0, completed_sequences=0,
        initial_label_decodes=0, noninitial_label_decodes=0, eligible_side_frames=0,
        track_step_intent=0, track_step_completed=0,
        calls={phase: {kind + suffix: 0 for kind in CALL_KINDS for suffix in ("_intent", "_completed")}
               for phase in PHASES}, phase_seconds={})
    checkpoint(common, out, record)
    try:
        free = shutil.disk_usage(out).free
        record["disk_preflight"] = dict(free_bytes=free, required_bytes=record["max_output_bytes"])
        require(free >= record["max_output_bytes"], "free disk for complete bounded output")
        admission35 = module("d38_complete_admission35", old / "admit.py")
        context, sequences = admission(spec, record, common, admission35, here)
        native35 = module("d38_complete_native35", old / "native_bank.py")
        for sequence in sequences:
            native35.validate_sequence(sequence)
        record["phase_seconds"]["admission"] = time.perf_counter() - record["monotonic_start"]
        record["phase"] = "runtime"
        checkpoint(common, out, record)
        began = time.perf_counter()
        runtime = common.load_runtime(spec, record)
        require(record["model_initial_hash"] == context["prediction_receipt"]["model_initial_hash"] ==
                context["prediction_receipt"]["model_final_hash"], "same D35 native model bytes")
        record["phase_seconds"]["runtime"] = time.perf_counter() - began
        evaluate35 = module("d38_complete_evaluate35", old / "evaluate.py")
        metrics = module("d38_complete_metrics35", old / "metrics.py")
        adapter = module("d38_complete_adapter", here / "runtime_adapter.py")
        operators = module("d38_complete_operators", here / "operators.py")
        evaluator = module("d38_complete_evaluator", here / "evaluate_complete.py")
        modules = native35, evaluate35, metrics, adapter, operators
        record["phase"] = "prediction"
        began = time.perf_counter()
        for sequence, oldrow in zip(sequences, context["prediction_receipt"]["rows"]):
            execute_sequence(runtime, common, sequence, oldrow, context, out, record, modules)
        record["phase_seconds"]["prediction"] = time.perf_counter() - began
        require(record["completed_frames"] == 446 and record["completed_sequences"] == 4 and
                record["eligible_side_frames"] == 438 and record["initial_label_decodes"] == 4 and
                record["noninitial_label_decodes"] == 0 and
                record["track_step_intent"] == record["track_step_completed"] == 446,
                "complete prediction/initial-only barrier")
        record["logical_work"] = dict(native_image_encoders=446, native_memory_encoders=446,
            native_sam_heads=446, native_attention_stacks=442, eligible_positions=438,
            alternate_memory_encoders=876, alternate_pointer_projections=876,
            extra_real_attention_stacks=1752, context_capture_dispatches=1752, extra_sam_heads=2190,
            selected_native_stack_and_sam_reuses=438, identity_fastpaths=0,
            scope="shared native image encoder; all side work and verification charged; no cached-cost advantage")
        began = time.perf_counter()
        common.finish(runtime, record)
        admission35.postcheck(context)
        require(sha(spec_path) == spec_hash and all(sha(p) == h for p, h in sources.items()), "preseal source/spec integrity")
        for sequence in sequences:
            native35.validate_sequence(sequence)
        for desc in record["outputs"]:
            evaluate35.frame_path(out, desc)
        record["phase_seconds"]["preseal_rechecks"] = time.perf_counter() - began
        snapshot = {key: record[key] for key in ("spec_sha256", "source_sha256", "bindings", "data_index", "rows",
            "sequences", "outputs", "arms", "completed_frames", "completed_sequences", "initial_label_decodes",
            "noninitial_label_decodes", "calls", "logical_work", "model_initial_hash", "model_final_hash")}
        snapshot.update(schema="evr0038.complete-prediction.v1", status="completed", frames=446,
                        bitorder="big", prediction_inputs="current/past RGB and supplied initial masks only")
        common.save(out / "prediction-receipt.json", snapshot)
        seal = dict(schema="evr0038.prediction-seal.v1", complete=True, frames=446, sequences=4,
            receipt_sha256=sha(out / "prediction-receipt.json"), spec_sha256=spec_hash,
            source_sha256=sources, bindings=record["bindings"], arms=list(ARMS), outputs=record["outputs"])
        common.save(out / "prediction-seal.json", seal)
        record["prediction_seal_sha256"] = sha(out / "prediction-seal.json")
        record["prediction_receipt_sha256"] = seal["receipt_sha256"]
        record["all_predictions_sealed_before_gt"] = True
        record["phase"] = "evaluation"
        checkpoint(common, out, record)
        began = time.perf_counter()
        report = evaluator.evaluate_complete(spec, out, record, context, np=runtime["np"], torch=runtime["torch"],
                                             Image=runtime["Image"], common=common)
        record["phase_seconds"]["evaluation"] = time.perf_counter() - began
        require(sha(out / "prediction-seal.json") == record["prediction_seal_sha256"] and
                sha(out / "prediction-receipt.json") == record["prediction_receipt_sha256"], "immutable prediction barrier")
        admission35.postcheck(context)
        require(sha(spec_path) == spec_hash and all(sha(p) == h for p, h in sources.items()), "final source/spec integrity")
        common.finish(runtime, record)
        require((out / "evaluation.json").is_file(), "complete evaluation output")
        record.update(evaluation_sha256=sha(out / "evaluation.json"), phase="completed", status="completed",
            eligible=True, scientific_eligible=False, current_action=None,
            eligibility_scope="child complete; scientific eligibility additionally requires clean complete external outer",
            final_artifacts=[native35.descriptor(common, out, out / filename, "complete-stage-artifact")
                for filename in ("prediction-receipt.json", "prediction-seal.json", "evaluation.json")])
        checkpoint(common, out, record)
        require(sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) <= record["max_output_bytes"],
                "all generated files within complete output cap")
    except BaseException as error:
        record.update(status="failed", eligible=False, scientific_eligible=False,
                      error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        checkpoint(common, out, record, enforce=False)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec")
    result = run(parser.parse_args().spec)
    print(json.dumps({key: result[key] for key in ("status", "eligible", "wall_seconds")}), flush=True)
