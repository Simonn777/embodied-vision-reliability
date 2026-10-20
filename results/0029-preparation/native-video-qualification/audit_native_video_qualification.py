"""Independent saved-output audit of the manufactured 25-frame qualification.

No model, RGB/PNG decoder, science video or query GT is imported/accessed.
NumPy is imported only by the CLI after the existing qualified offline guard.
The pure checker does not import native_video or qualifier verification code.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time

OWN = "experiments/0029-memory-information/"
FRAMES = (4, 7, 10, 13, 15, 18, 21, 24)
METHODS = ("signed_deletion", "baseline", "max_predicted_iou", "max_unsigned_change",
           "always_D1", "always_D3", "random_deletion", "random_on_signed", "native_bank",
           "prompt_direct", "prompt_signed", "reset_direct", "reset_signed")
COUNTS = dict(image_encoder_intent=25, image_encoder_completed=25, baseline_frames=25,
    track_step_intent=63, track_step_completed=63, verification_intent=8,
    verification_completed=8, deletion_intent=16, deletion_completed=16,
    prompt_intent=14, prompt_completed=14, native_bank_extra_calls=0)
SOURCE_PATHS = tuple(OWN + p for p in (
    "run_native_video_qualification.py", "qualify_native_video.py", "native_video.py",
    "diagnostic_core.py", "run_evidence.py")) + (
    "protocols/0029-native-video-comparison.md", "protocols/0029-signed-memory-diagnostic.md",
    "experiments/0027-affine-tangent/supervise.py", "experiments/0025c-rs-noise/supervise.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    def reject(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def digest(value):
    require(isinstance(value, str) and re.fullmatch("[a-f0-9]{64}", value), "invalid SHA256")
    return value


def bound_file(path, descriptor):
    path = Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            "missing/symlink saved artifact")
    require(type(descriptor["bytes"]) is int and path.stat().st_size == descriptor["bytes"]
            and sha(path) == descriptor["sha256"], "saved artifact hash/size changed")
    return path


def tensor_digest(array):
    """The producer's documented tensor-hash byte format; no torch import."""
    require(str(array.dtype) in ("float32", "float64", "float16"), "unexpected native array dtype")
    value = hashlib.sha256(str(("torch." + str(array.dtype), tuple(array.shape))).encode())
    value.update(array.tobytes(order="C"))
    return value.hexdigest()


def check_trace(trace, frame, removed=None, reset=False):
    spatial = [0] + [i for i in range(max(1, frame - 6), frame) if i != removed]
    pointers = [0] + [i for i in range(frame - 1, max(0, frame - 16), -1) if i != removed]
    if reset:
        spatial = pointers = [0]
    require(trace["frame"] == frame and trace["num_frames"] == 25
            and trace["pointer_limit"] == 16 and trace["pointer_distance_denominator"] == 15
            and trace["spatial_frames"] == spatial and trace["pointer_frames"] == pointers
            and trace["actual_batch_formula_exact"] is True, "carrier chronology/batch contract")
    expected = [("spatial", i) for i in spatial] + [("pointer", i) for i in pointers]
    require([(b["kind"], b["frame"]) for b in trace["blocks"]] == expected, "carrier order/coverage")
    cursor = pointer_tokens = 0
    for block in trace["blocks"]:
        require(type(block["start"]) is int and type(block["end"]) is int
                and block["start"] == cursor and block["end"] > cursor, "carrier token extents")
        cursor = block["end"]
        digest(block["values_sha256"]); digest(block["position_sha256"])
        if block["kind"] == "pointer":
            pointer_tokens += block["end"] - block["start"]
            require(block["temporal_distance"] == frame - block["frame"]
                    and block["temporal_distance_denominator"] == 15, "pointer time coordinates")
    require(cursor == trace["total_tokens"] and pointer_tokens == trace["num_obj_ptr_tokens"],
            "carrier token accounting")
    return {(b["kind"], b["frame"]): b for b in trace["blocks"]}


def check_witnesses(witness, index, frame):
    positive = [(224, 136), (252, 136), (224, 164), (252, 164)]
    negative = [(40, 40), (100, 40), (40, 300), (100, 300)][:3 if index == 0 else 4]
    require(witness["sequence"] == "dog-agility" and witness["frame"] == frame
            and witness["target_id"] == 1 and witness["initial_shape"] == [384, 512]
            and witness["current_shape"] == [384, 512]
            and witness["actionable"] is (index != 0), "manufactured witness identity")
    for key, points, label in (("positive", positive, 1), ("negative", negative, 0)):
        require(len(witness[key]) == len(points)
                and {tuple(p["source_xy"]) for p in witness[key]} == set(points),
                "manufactured witness set")
        for p in witness[key]:
            x, y = p["source_xy"]
            require(p["target_xy"] == [x + 8 * frame, y] and p["label"] == label
                    and p["rounded_current_xy"] == [int(x + 8 * frame), int(y)]
                    and p["confidence"] == 1, "manufactured correspondence changed")


def selection(row):
    bank = row["candidates"]
    token = "\0".join(("evr0029:random-deletion:2026-10-20", "dog-agility", "1", str(row["frame"])))
    raw = hashlib.sha256(token.encode()).hexdigest()
    random = {"branch": ("D1", "D3")[int(raw[-1], 16) & 1], "sha256": raw}
    choices = dict.fromkeys(METHODS, "B")
    if not row["actionable"]:
        return dict(actionable=False, choices=choices, signed_action=False, random=random, bank_losses=None)
    def signed(names):
        chosen = min(enumerate(names), key=lambda item: (bank[item[1]]["loss"], item[0]))[1]
        return chosen if bank["B"]["loss"] - bank[chosen]["loss"] > 1e-6 else "B"
    chosen = signed(("B", "D1", "D3"))
    choices.update(signed_deletion=chosen,
        max_predicted_iou=max(("B", "D1", "D3"), key=lambda n: bank[n]["predicted_iou"]),
        max_unsigned_change=max(("B", "D1", "D3"), key=lambda n: bank[n]["unsigned_change"]),
        always_D1="D1", always_D3="D3", random_deletion=random["branch"],
        random_on_signed=random["branch"] if chosen != "B" else "B",
        native_bank=signed(("B", "M0", "M1", "M2")), prompt_direct="P",
        prompt_signed=signed(("B", "P")), reset_direct="R", reset_signed=signed(("B", "R")))
    return dict(actionable=True, choices=choices, signed_action=chosen != "B", random=random,
                bank_losses={n: bank[n]["loss"] for n in ("B", "D1", "D3")})


def check_saved(operation, directory, sources, np):
    """Saved JSON/NPZ-only checker. The caller owns source/runtime/outer admission."""
    manufactured = directory / "manufactured"
    qualification = read(manufactured / "qualification-record.json")
    require(operation["qualification"] == qualification == operation["native_video_qualification"],
            "qualification aliases differ")
    require(qualification["schema"] == "evr0029.native-video-qualification.v1"
            and qualification["status"] == "completed" and qualification["complete"] is True
            and qualification["manufactured_only"] is True and qualification["scientific_eligible"] is False
            and qualification["davis_data_access"] is False
            and qualification["noninitial_annotation_access"] is False
            and qualification["source_sha256"] == sources, "manufactured provenance")
    fixture = qualification["fixture"]
    require(all(fixture[k] == v for k, v in {
        "internal_worker_sequence": "dog-agility", "frame_count": 25, "shape": [384, 512],
        "internal_identity_is_not_data_provenance": True, "right_shift_per_frame": 8,
        "padding": "zero", "jpeg_quality": 95, "jpeg_subsampling": 0,
        "initial_mask_yx": [118, 182, 207, 271], "query_frames": list(FRAMES),
        "first_query_omits_last_negative": True, "no_superpoint_or_lightglue_inference": True,
    }.items()), "manufactured fixture changed")
    inputs = qualification["inputs"]
    require(len(inputs) == 34 and [r["kind"] for r in inputs] ==
            ["manufactured_rgb"] * 25 + ["manufactured_initial_mask"] + ["manufactured_witness"] * 8,
            "manufactured input population")
    for i, item in enumerate(inputs):
        name = ("manufactured-frames/%05d.jpg" % i if i < 25 else "manufactured-initial.png"
                if i == 25 else "manufactured-witness-%05d.json" % FRAMES[i - 26])
        path = manufactured / name
        require(Path(item["path"]) == path, "input escaped manufactured directory")
        if i < 25:
            require(item["frame"] == i, "manufactured RGB order")
        bound_file(path, item)  # Opaque JPEG/PNG bytes only: no image decoder.
    worker_dir = manufactured / "worker"
    require(Path(qualification["worker_receipt"]["path"]) == worker_dir / "video-record.json",
            "worker receipt path")
    worker = read(bound_file(worker_dir / "video-record.json", qualification["worker_receipt"]))
    require(qualification["worker_record"] == {"native_video": worker}
            and worker["schema"] == "evr0029.native-video.v1"
            and worker["source_sha256"] == sources and worker["status"] == "completed"
            and worker["complete"] is True and worker["sequence"] == "dog-agility"
            and worker["frame_count"] == 25 and worker["query_frames"] == list(FRAMES)
            and worker["query_count"] == 8 and worker["actionable_count"] == 7
            and worker["noninitial_annotation_access"] is False and worker["calls"] == COUNTS
            and [b["frame"] for b in worker["baseline"]] == list(range(25))
            and [q["frame"] for q in worker["queries"]] == list(FRAMES)
            and worker["method_names"] == list(METHODS)
            and worker["observed_compact_memory_dtypes"] == ["torch.bfloat16"], "complete native worker")
    require(Path(worker["frames_dir"]) == manufactured / "manufactured-frames"
            and worker["initial_mask"] == {key: inputs[25][key] for key in ("path", "bytes", "sha256")}
            and {p.name for p in (manufactured / "manufactured-frames").iterdir()} ==
                {"%05d.jpg" % i for i in range(25)}, "worker manufactured-media bindings")
    descriptors = {d["filename"]: d for d in worker["outputs"]}
    expected = {"query-%05d-%s.npz" % (f, n) for i, f in enumerate(FRAMES)
                for n in ("B", "D1", "D3", "native-bank", *(("P", "R") if i else ()))}
    require(len(worker["outputs"]) == len(descriptors) == 46 and set(descriptors) == expected
            and {p.name for p in worker_dir.iterdir()} == expected | {"video-record.json"},
            "saved output inventory")
    checked_arrays = 0
    differences = 0
    cross_resize = []
    for index, row in enumerate(worker["queries"]):
        frame, actionable = FRAMES[index], index != 0
        witness = read(manufactured / ("manufactured-witness-%05d.json" % frame))
        check_witnesses(witness, index, frame)
        require(row["witnesses"] == witness and row["status"] == "completed"
                and row["sequence"] == "dog-agility" and row["target_id"] == 1
                and row["actionable"] is actionable
                and row["witness_counts"] == {"positive": 4, "negative": 4 if actionable else 3},
                "query input binding")
        require(row["input"] == {key: inputs[26 + index][key]
                                for key in ("path", "bytes", "sha256", "frame")},
                "worker witness-file binding")
        require(row["baseline_state_sha256_before"] == row["baseline_state_sha256_after"],
                "baseline state changed")
        digest(row["baseline_state_sha256_before"])
        arrays, logits = {}, {}
        names = ("B", "D1", "D3", "M0", "M1", "M2", *(("P", "R") if actionable else ()))
        require(set(row["candidates"]) == set(names), "candidate coverage")
        for name in names:
            candidate = row["candidates"][name]
            filename = "query-%05d-%s.npz" % (frame, "native-bank" if name.startswith("M") else name)
            desc = descriptors[filename]
            require(candidate["status"] == "completed" and candidate["output"] == desc
                    and candidate["array_key"] == "logits"
                    and candidate["channel"] == (int(name[1]) if name.startswith("M") else 0),
                    "candidate descriptor/channel")
            if filename not in arrays:
                path = bound_file(worker_dir / filename, desc)
                with np.load(path, allow_pickle=False) as archive:
                    require(set(archive.files) == set(desc["arrays"]), "NPZ array population")
                    values = {key: archive[key] for key in archive.files}
                for key, value in values.items():
                    metadata = desc["arrays"][key]
                    require(value.dtype.kind in "bifu" and np.isfinite(value).all()
                            and list(value.shape) == metadata["shape"]
                            and str(value.dtype) == metadata["dtype"]
                            and hashlib.sha256(value.tobytes(order="C")).hexdigest()
                            == metadata["raw_c_order_sha256"], "array finite/hash/shape/dtype")
                    checked_arrays += 1
                arrays[filename] = values
            values = arrays[filename]
            channel = candidate["channel"]
            require(values["logits"].shape == (1, 3 if name.startswith("M") else 1, 384, 512),
                    "original resolution")
            logits[name] = values["logits"][0, channel]
            selected = channel if name.startswith("M") else int(np.argmax(values["predicted_quality"][0]))
            require(candidate["predicted_iou"] == float(values["predicted_quality"][0, selected]),
                    "native quality value")
            if not name.startswith("M"):
                require(candidate["selected_native_index"] == selected
                        and candidate["low_sha256"] == tensor_digest(values["low_logits"]),
                        "selected native low tensor hash")
            if actionable:
                class_losses = []
                for key, sign in (("positive", -1), ("negative", 1)):
                    terms = []
                    for p in witness[key]:
                        x, y = p["target_xy"]  # This manufactured fixture uses integer pixels.
                        require(int(x) == x and int(y) == y, "noninteger manufactured witness")
                        z = sign * float(logits[name][int(y), int(x)])
                        terms.append(max(z, 0.) + math.log1p(math.exp(-abs(z))))
                    class_losses.append(math.fsum(terms) / len(terms))
                loss = sum(class_losses) / 2
                require(abs(loss - candidate["loss"]) <= 64 * np.finfo(np.float64).eps * max(1., abs(loss)),
                        "independent sparse logloss differs")
            else:
                require(candidate["loss"] is None, "nonactionable loss")
        baseline = arrays["query-%05d-B.npz" % frame]
        bank = arrays["query-%05d-native-bank.npz" % frame]
        selected = int(np.argmax(bank["predicted_quality"][0]))
        require(np.array_equal(baseline["low_logits"], bank["low_logits"][:, selected:selected + 1])
                and np.array_equal(baseline["predicted_quality"], bank["predicted_quality"])
                and worker["baseline"][frame]["original_logits_sha256"] == tensor_digest(baseline["logits"]),
                "baseline/native-bank selected-low and single-selected original parity")
        # Fixed prospectively before reading qualification output. The native
        # worker checks low_selected->B, not 3-channel vs 1-channel resize bits.
        selected_original = bank["logits"][:, selected:selected + 1]
        cross_resize.append(dict(frame=frame,
            exact_values=bool(np.array_equal(baseline["logits"], selected_original)),
            exact_masks=bool(np.array_equal(baseline["logits"] > 0, selected_original > 0)),
            max_absolute_difference=float(np.max(np.abs(
                baseline["logits"].astype(np.float64) - selected_original.astype(np.float64)))),
            hard_gate=False))
        require(row["native_bank"] == dict(count=3, captured_existing_call=True, extra_calls=0,
            selected_index=selected, selected_low_exact=True, selected_original_exact=True), "bank provenance")
        old = check_trace(row["baseline_trace"]["trace"], frame)
        require([b["name"] for b in row["branches"]] ==
                ["NOOP", "D1", "D3"] + (["P", "R"] if actionable else []), "branch coverage")
        for branch in row["branches"]:
            name, meta = branch["name"], branch["trace"]
            remove = frame - (1 if name == "D1" else 3) if name in ("D1", "D3") else None
            blocks = check_trace(meta["trace"], frame, removed=remove, reset=name == "R")
            require(branch["status"] == "completed" and branch["call_attempted"] is True
                    and branch["call_completed"] is True and branch["baseline_state_unchanged"] is True
                    and branch["state_sha256_before"] == row["baseline_state_sha256_before"]
                    and branch["fork_state_changed"] ==
                        (branch["state_sha256_before"] != branch["state_sha256_after"]),
                    "branch call/state isolation")
            require(all(meta[k] == row["baseline_trace"][k] for k in
                        ("vision_sha256", "vision_position_sha256", "feature_sizes")),
                    "branch current features changed")
            if name != "P":
                require(branch["state_sha256_after"] == branch["state_sha256_before"], "read-only state changed")
            if name in ("NOOP", "P"):
                require(meta["trace"] == row["baseline_trace"]["trace"], "full-memory trace changed")
            if name == "NOOP":
                require(branch["exact_baseline_original_parity"] is True
                        and branch["exact_baseline_low_parity"] is True, "no-op verification absent")
            if name in ("D1", "D3", "R"):
                require(set(blocks) == ({key for key in old if key[1] != remove}
                        if name != "R" else {("spatial", 0), ("pointer", 0)}), "carrier removal/backfill")
                diff = []
                for key, block in blocks.items():
                    prior = old[key]
                    require(block["values_sha256"] == prior["values_sha256"], "retained features changed")
                    if key[0] == "spatial":
                        require(block["position_sha256"] == prior["position_sha256"], "spatial position changed")
                    elif block["position_sha256"] != prior["position_sha256"]:
                        diff.append(dict(frame=key[1], full_position_sha256=prior["position_sha256"],
                            branch_position_sha256=block["position_sha256"],
                            temporal_distance=block["temporal_distance"], temporal_distance_denominator=15))
                if name != "R":
                    require(branch["deletion"]["removed_frame"] == remove
                            and branch["deletion"]["removed_kinds"] == ["spatial", "pointer"]
                            and branch["deletion"]["pointer_position_cross_batch_differences"] == diff,
                            "deletion description differs from raw trace")
                    differences += len(diff)
            if name == "P":
                require(meta["previous_mask_sha256"] == tensor_digest(np.clip(baseline["low_logits"], -32., 32.)),
                        "prompt prior differs from clamped baseline")
            else:
                require(meta["previous_mask_sha256"] is None, "unexpected prior logits")
            if name in ("P", "R"):
                points = witness["positive"] + witness["negative"]
                coords = np.asarray([p["target_xy"] for p in points], dtype=np.float32)
                coords = (coords[None] / np.asarray([512, 384], dtype=np.float32)
                          * worker["config"]["image_size"]).tolist()
                require(meta["point_coords"] == coords
                        and meta["point_labels"] == [[p["label"] for p in points]], "prompt inputs changed")
            else:
                require(meta["point_coords"] is None and meta["point_labels"] is None, "unexpected prompts")
        for name in ("B", "D1", "D3"):
            a, b = logits[name] > 0, logits["B"] > 0
            union = int(np.count_nonzero(a | b))
            change = 1. - int(np.count_nonzero(a & b)) / union if union else 0.
            require(row["candidates"][name]["unsigned_change"] == change, "unsigned change differs")
        require(row["selection"] == selection(row), "sealed choice differs")
    return dict(pass_all=True, manufactured_only=True, scientific_eligible=False, image_decodes=0,
        model_calls=0, queries=8, actionable_queries=7, calls=COUNTS, npz_files=46,
        numeric_arrays=checked_arrays, pointer_position_cross_batch_differences=differences,
        native_bank_cross_resize=cross_resize,
        independent_logloss_tolerance="64 * float64 eps * max(1, abs(recomputed_loss))",
        limits=[
            "Input JPEG/PNG hashes and frozen producer provenance are verified; pixels are not decoded/regenerated.",
            "No-op output tensors were not separately saved: parity is frozen-worker reported, plus exact saved traces/state hashes.",
            "Raw attention tensors and model weights are not decoded: actual-batch formula checks are inherited; saved carrier/time/hash relations are independently checked.",
            "No segmentation accuracy, query GT, scientific gain or runtime efficiency is assessed."])


def run(args):
    root, directory = Path(args.repo_root).resolve(), Path(args.qualification_root).resolve()
    out = Path(args.output_dir).absolute()
    require(not out.exists() and not out.is_symlink() and root not in out.resolve().parents,
            "fresh private audit directory required")
    out.mkdir(parents=True)
    began = time.perf_counter()
    record = dict(schema="evr0029.manufactured-video-audit.v1", status="started", eligible=False,
                  source_sha256=sha(__file__), offline_attempts=[], compatibility_probes=[])
    try:
        h = module("audit_installer_admission", root / OWN / "install_runtime.py")
        old = module("audit_old_native_admission", root / OWN / "run_evidence.py")
        sources = h.frozen(root, args.expected_commit, SOURCE_PATHS)
        installation, installed, native = old.native_contract(
            h, root, Path(args.native_root).resolve(), args.native_commit)
        command = [str(installation / "venv/bin/python"), "-S",
            str(root / OWN / "run_native_video_qualification.py"), "--repo-root", str(root),
            "--expected-commit", args.expected_commit, "--native-root", str(Path(args.native_root).resolve()),
            "--native-commit", args.native_commit, "--output-dir", str(directory)]
        binding = old.outer_contract(h, root, args.expected_commit, Path(args.qualification_spec).resolve(),
                                     directory / "receipt.json", command)
        operation = read(directory / "receipt.json")
        require(operation["schema"] == "evr0029.native-video-qualification-operation.v1"
                and operation["status"] == "completed" and operation["eligible"] is True
                and operation["manufactured_only"] is True and operation["scientific_eligible"] is False
                and operation["scientific_input_access"] is False and operation["offline_attempts"] == []
                and operation["expected_commit"] == args.expected_commit
                and operation["native_commit"] == args.native_commit
                and operation["source_sha256"] == sources and operation["native_binding"] == native,
                "clean source/native qualification admission")
        helper_hashes = {
            "qualify_helper": sha(root / "experiments/0022-frontend/qualify.py"),
            **{name: sha(root / OWN / (name + ".py")) for name in
               ("native_fixture", "native_video", "diagnostic_core")}}
        require(operation["qualification"]["observed_helper_sha256"] == helper_hashes,
                "qualification helper identity")
        require(sys.version_info[:3] == (3, 12, 13) and sys.flags.no_site == 1
                and Path(sys.executable).absolute() == installation / "venv/bin/python"
                and all(os.environ.get(k) == "4" for k in h.THREAD_KEYS), "qualified interpreter/threads")
        q = module("audit_offline_guard", root / "experiments/0022-frontend/qualify.py")
        q.install_guard(record["offline_attempts"], record["compatibility_probes"])
        sys.dont_write_bytecode = True
        import site
        site.main()
        import numpy as np
        require(Path(sys.prefix).resolve() == installation / "venv"
                and Path(np.__file__).resolve().is_relative_to(
                    installation / "venv/lib/python3.12/site-packages"), "qualified NumPy origin")
        record.update(qualification_binding=binding, native_binding=native, checked_sources=sources)
        record["checks"] = check_saved(operation, directory, sources, np)
        old.unchanged(h, binding["file_sha256"])
        old.unchanged(h, native["file_sha256"])
        require(all(sha(root / rel) == value for rel, value in sources.items())
                and installed["installed_inventory"] == h.inventory(installation)
                and not record["offline_attempts"], "audit source/runtime/offline recheck")
        record.update(status="completed", eligible=True)
    except BaseException as exc:
        record.update(status="failed", eligible=False, error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        record["wall_seconds"] = time.perf_counter() - began
        (out / "receipt.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: record[k] for k in ("status", "eligible", "wall_seconds")}), flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo-root", "expected-commit", "qualification-root", "qualification-spec",
                 "native-root", "native-commit", "output-dir"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())
