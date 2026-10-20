"""Two terminal learned heads on sealed cached observations; no GT or SAM imports."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ARMS = ("visual", "logit_only")
COHORT = (("dog-agility", 25, "TRAIN"), ("planes-water", 38, "TRAIN"),
          ("dogs-scale", 83, "TRAIN"), ("dog-gooses", 86, "TRAIN"),
          ("cat-girl", 89, "DEV"), ("upside-down", 65, "DEV"))
OLD_KEYS = {"cache_receipt", "cache_outer", "cache_operation", "cache_worker_spec", "mapping", "data_index"}
FIT_KEYS = {"fit_receipt", "fit_outer", "fit_operation", "fit_worker_spec"}
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def module(name, path):
    loader = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(loader)
    sys.modules[name] = value
    loader.loader.exec_module(value)
    return value


def relative(root, descriptor):
    rel = Path(descriptor["path"])
    path = (root / rel).resolve()
    require(not rel.is_absolute() and ".." not in rel.parts and path.is_relative_to(root.resolve()),
            "output escapes producer root")
    return dict(descriptor, path=str(path))


def check(descriptor, fingerprints):
    path = Path(descriptor["path"]).resolve()
    require(path.is_file() and path.stat().st_size == descriptor["bytes"]
            and sha(path) == descriptor["sha256"], "descriptor changed: " + str(path))
    require(str(path) not in fingerprints or fingerprints[str(path)] == descriptor["sha256"],
            "conflicting file identity")
    fingerprints[str(path)] = descriptor["sha256"]
    return path


def admit_fit(admission, bindings, declared_bindings, fingerprints, here):
    """Receipt/source admission only. Does not import train/head or load tensors."""
    cache, _ = admission.admit(bindings, "cache", fingerprints, legacy=True)
    fit, worker = admission.admit(bindings, "fit", fingerprints)
    require(cache["schema"] == "evr0030.development.baseline-cache.v1"
            and cache["completed_frames"] == 386 and cache["completed_sequences"] == 6
            and not cache["offline_attempts"] and cache["model_initial_hash"] == cache["model_final_hash"],
            "complete unchanged native cache")
    require(fit["schema"] == "evr0033.native-feature-fit.v1"
            and fit["completed_train_frames"] == len(fit["train_frames"]) == 228
            and fit["completed_arms"] == 2 and set(fit["arms"]) == set(ARMS)
            and not fit["offline_attempts"]
            and {(r["sequence"], r["frame"]) for r in fit["train_frames"]} ==
                {(n, f) for n, count, role in COHORT if role == "TRAIN" for f in range(1, count)},
            "complete two-arm TRAIN fit")
    require(fit["bindings"] == worker["bindings"] == {k: declared_bindings[k] for k in OLD_KEYS},
            "fit used different cache/mapping")
    operation = json.loads(bindings["fit_operation"].read_text())
    require(Path(operation["command"][-2]).resolve() == here / "train.py", "fit command script")
    for path, digest in worker["source_sha256"].items():
        require(sha(path) == digest and (path not in fingerprints or fingerprints[path] == digest),
                "fit source changed")
        fingerprints[path] = digest
    inventory = {d["path"]: d for d in fit["outputs"]}
    require(len(inventory) == len(fit["outputs"]), "duplicate fit outputs")
    selected = [fit["normalization"]] + [fit["arms"][a]["terminal_checkpoint"] for a in ARMS]
    for descriptor in selected:
        require(inventory[descriptor["path"]] == descriptor, "terminal artifact not in inventory")
        check(relative(bindings["fit_receipt"].parent, descriptor), fingerprints)
    for arm in ARMS:
        require(fit["arms"][arm]["status"] == "completed" and fit["arms"][arm]["epochs"] == 12
                and fit["arms"][arm]["attempted_frames"] == 2736
                and fit["arms"][arm]["zero_features"] == (arm == "logit_only")
                and fit["arms"][arm]["initial_state_sha256"] == fit["initial_state_sha256"],
                "terminal twelve-epoch arm required")
    return cache, fit, worker


def plans(cache, index, mapping):
    """Fixed metadata plan. Label paths are compared, never opened by prediction."""
    bases = {r["sequence"]: r for r in cache["rows"]}
    inputs = {r["sequence"]: r for r in index["sequences"]}
    require(len(bases) == len(cache["rows"]) == len(inputs) == len(index["sequences"]) == 6
            and set(bases) == set(inputs) == {n for n, _, _ in COHORT}
            and mapping["status"] == "completed" and mapping["member_count"] == 386,
            "complete six-parent metadata")
    answer = []
    for name, n, role in COHORT:
        base, source = bases[name], inputs[name]
        anchors = [1 + i * (n - 5) // 11 for i in range(12)] if role == "TRAIN" else []
        require(base["status"] == "completed" and base["completed_frames"] == base["frames"] == n
                and base["role"] == source["role"] == role
                and base["parent_id"] == source["parent_id"] == name and base["anchors"] == anchors
                and [d["frame"] for d in base["predictions"]] == [d["frame"] for d in source["labels"]]
                == list(range(n)), "parent metadata")
        labels = [d for d in mapping["mappings"] if d["sequence"] == name]
        require([d["frame"] for d in labels] == list(range(n))
                and all(d["role"] == role and all(d[k] == source["labels"][i][k]
                        for k in ("path", "bytes", "sha256")) for i, d in enumerate(labels)),
                "mapping/index label identity")
        frames = sorted({a + h for a in anchors for h in (1, 3)}) if role == "TRAIN" else list(range(n))
        answer.append(dict(sequence=name, parent_id=name, frames=n, role=role,
                           shape=base["shape"], target_id=base["target_id"], anchors=anchors,
                           requested_frames=frames))
    require(sum(len(r["requested_frames"]) for r in answer) == 241, "241 fixed frame identities")
    return answer


def run(spec_path):
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text())
    here, evaluation = Path(__file__).resolve().parent, Path(spec["evaluation_root"])
    common_path = Path(spec["campaign_root"]) / "common.py"
    require(spec["source_sha256"].get(str(common_path)) == sha(common_path), "serialization helper changed")
    c = module("feature_prediction_common", common_path)
    spec, out, record, began = c.start(spec_path, "native-feature-prediction")
    record.update(schema="evr0033.native-feature-prediction.v1", bindings=spec["bindings"],
        rows=[], outputs=[], arms=list(ARMS), current_label_decodes=0, native_calls=0,
        image_encoder_calls=0, native_weights_loaded=False, final_test_access=False,
        completed_frames=0, completed_arm_predictions=0, native_parity_checks=0,
        zero_head_parity_checks=0, learned_head_calls={a: 0 for a in ARMS}, offline_attempts=[],
        compatibility_probes=[], warnings=[])
    fingerprints = dict(spec["source_sha256"]); fingerprints[str(spec_path)] = record["spec_sha256"]
    def checkpoint():
        record["wall_seconds"] = time.perf_counter() - began
        c.save(out / "receipt.json", record)
    checkpoint()
    try:
        guard_path = Path(spec["guard_source"]).resolve()
        required = {str(here / n) for n in ("predict.py", "head.py", "train.py", "SCOPE.md", "ADMISSION.md")}
        required |= {str(common_path), str(guard_path)}
        required |= {str(evaluation / n) for n in ("evaluate_jf.py", "evaluate_baseline.py", "evaluate_rollout.py")}
        require(required <= set(fingerprints) and set(spec["bindings"]) == OLD_KEYS | FIT_KEYS,
                "source and ten metadata bindings")
        for path, digest in fingerprints.items():
            require(sha(path) == digest, "source/spec changed")
        require(sys.flags.no_site == 1 and sys.version_info[:3] == (3, 12, 13)
                and Path(sys.executable).absolute() == Path(spec["python"]).absolute()
                and all(os.environ.get(k) == "4" for k in THREAD_KEYS), "qualified offline interpreter/threads")
        bindings = {k: check(d, fingerprints) for k, d in spec["bindings"].items()}
        guard = module("feature_prediction_guard", guard_path)
        guard.install_guard(record["offline_attempts"], record["compatibility_probes"])
        import site
        site.main()
        with guard.capture_warnings(record):
            import torch
            torch.set_num_threads(4); torch.set_num_interop_threads(1)
            torch.manual_seed(0)
            import numpy as np
            sys.path.insert(0, str(evaluation))
            admission = module("feature_prediction_admission", evaluation / "evaluate_rollout.py")
            admission.e.cv2.setNumThreads(0)
            head = module("feature_prediction_head", here / "head.py")
        require(torch.get_num_threads() == 4 and torch.get_num_interop_threads() == 1
                and admission.e.cv2.getNumThreads() == 1, "actual execution threads")
        record["runtime"] = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                                 torch_threads=4, torch_interop_threads=1, opencv_threads=1)
        cache, fit, worker = admit_fit(admission, bindings, spec["bindings"], fingerprints, here)
        require(cache["data_index"]["sha256"] == sha(bindings["data_index"]), "cache/index binding")
        frame_plans = plans(cache, json.loads(bindings["data_index"].read_text()),
                            json.loads(bindings["mapping"].read_text()))
        cache_root, fit_root = bindings["cache_receipt"].parent, bindings["fit_receipt"].parent
        inventory = {d["path"]: d for d in cache["outputs"]}
        require(len(inventory) == len(cache["outputs"]), "duplicate cache paths")
        bases = {r["sequence"]: r for r in cache["rows"]}
        selected = {}
        for plan in frame_plans:
            name = plan["sequence"]
            for frame in plan["requested_frames"]:
                native = bases[name]["predictions"][frame]
                require(inventory[native["path"]] == native, "native prediction inventory")
                selected[name, frame] = (
                    check(relative(cache_root, inventory[f"{name}/frame-{frame:05d}.pt"]), fingerprints),
                    check(relative(cache_root, native), fingerprints))
        norm_path = fit_root / fit["normalization"]["path"]
        norm = json.loads(norm_path.read_text())
        require(norm["source_frames"] == 228 and norm["population_pixels"] == 228 * 65536,
                "TRAIN-only normalization population")
        mean, std = (torch.tensor(norm[k], dtype=torch.float32) for k in ("mean", "std"))
        require(mean.shape == std.shape == (32,) and bool(torch.isfinite(mean).all())
                and bool(torch.isfinite(std).all()) and bool((std >= 1e-6).all()), "normalization values")
        models, zeros, states, configurations = {}, {}, {}, []
        for arm in ARMS:
            package = torch.load(fit_root / fit["arms"][arm]["terminal_checkpoint"]["path"],
                                 map_location="cpu", weights_only=True)
            require(package["schema"] == "evr0033.native-feature-checkpoint.v1"
                    and package["arm"] == arm and package["epochs"] == 12
                    and package["optimizer_steps"] == fit["arms"][arm]["optimizer_steps"]
                    and package["config"] == fit["config"]
                    and package["initial_state_sha256"] == fit["initial_state_sha256"]
                    and torch.equal(package["normalization"]["mean"], mean)
                    and torch.equal(package["normalization"]["std"], std), "terminal checkpoint contract")
            models[arm] = head.NativeFeatureHead(mean, std, zero_features=arm == "logit_only")
            models[arm].load_state_dict(package["state_dict"], strict=True)
            models[arm].eval().requires_grad_(False)
            zeros[arm] = head.NativeFeatureHead(mean, std, zero_features=arm == "logit_only").eval().requires_grad_(False)
            require(sum(p.numel() for p in models[arm].parameters()) == 1377
                    and torch.equal(models[arm].train_mean.reshape(-1), mean)
                    and torch.equal(models[arm].train_std.reshape(-1), std), "head architecture/statistics")
            states[arm] = {k: v.detach().clone() for k, v in models[arm].state_dict().items()}
            require(all(v.dtype == torch.float32 and bool(torch.isfinite(v).all())
                        for v in states[arm].values()), "finite float32 head state")
            configurations.append(package["config"])
        require(configurations[0] == configurations[1], "arm scientific configurations differ")
        record.update(normalization=fit["normalization"],
            terminal_checkpoints={a: fit["arms"][a]["terminal_checkpoint"] for a in ARMS},
            fit_initial_state_sha256=fit["initial_state_sha256"], head_config=configurations[0],
            native_state_hash_reference=cache["model_final_hash"], native_weights_unchanged_by_reference=True)
        with torch.inference_mode(), guard.capture_warnings(record):
            for plan in frame_plans:
                row = {k: v for k, v in plan.items() if k != "requested_frames"}
                row.update(status="started", predictions={a: [] for a in ARMS}, frame_records=[])
                record["rows"].append(row)
                directory = out / plan["sequence"]; directory.mkdir()
                for frame in plan["requested_frames"]:
                    started = time.perf_counter()
                    detail = dict(frame=frame, status="started", arms={})
                    row["frame_records"].append(detail); checkpoint()
                    tensor_path, native_path = selected[plan["sequence"], frame]
                    packed = torch.load(tensor_path, weights_only=True, map_location="cpu")
                    low, features = packed["out"]["pred_masks"], packed["vision"][0]
                    require(low.shape == (1, 1, 256, 256) and features.shape == (65536, 1, 32)
                            and low.dtype == features.dtype == torch.float32
                            and bool(torch.isfinite(low).all()) and bool(torch.isfinite(features).all()),
                            "finite native observation shapes")
                    features = features.permute(1, 2, 0).reshape(1, 32, 256, 256)
                    presence = packed["out"]["object_score_logits"]
                    require(presence.numel() == 1 and bool(torch.isfinite(presence).all()), "native presence")
                    native = admission.e._prediction(native_path, [plan["target_id"]], tuple(plan["shape"]))[0]
                    full = head.original_logits(low, plan["shape"])[0, 0]
                    require(np.array_equal((full > 0).numpy(), native), "native original-resolution parity")
                    record["native_parity_checks"] += 1
                    reason = ("DEV_initial" if plan["role"] == "DEV" and frame == 0 else
                              "native_nonpositive_presence" if float(presence.item()) <= 0 else
                              "native_empty" if not native.any() else None)
                    detail.update(noop_reason=reason, native_presence=float(presence.item()), native_parity=True)
                    for arm in ARMS:
                        zero = zeros[arm](features, low)
                        require(torch.equal(zero, low) and np.array_equal(
                            (head.original_logits(zero, plan["shape"])[0, 0] > 0).numpy(), native),
                            "zero-head original-resolution native parity")
                        record["zero_head_parity_checks"] += 1
                        began_arm = time.perf_counter()
                        if reason is None:
                            record["learned_head_calls"][arm] += 1
                            checkpoint()
                            changed = models[arm](features, low)
                            answer = (head.original_logits(changed, plan["shape"])[0, 0] > 0).numpy()
                        else:
                            answer = native.copy()
                        path = directory / f"{arm}-{frame:05d}.npz"
                        with path.open("xb") as stream:
                            np.savez_compressed(stream, masks=answer[None],
                                                object_ids=np.array([plan["target_id"]], dtype=np.int64))
                        descriptor = dict(path=str(path.relative_to(out)), sha256=sha(path),
                            bytes=path.stat().st_size, kind="prediction", sequence=plan["sequence"], frame=frame, arm=arm)
                        row["predictions"][arm].append(descriptor); record["outputs"].append(descriptor)
                        detail["arms"][arm] = dict(zero_head_parity=True, head_called=reason is None,
                            changed_pixels=int(np.count_nonzero(answer != native)),
                            output_foreground_pixels=int(answer.sum()), seconds=time.perf_counter() - began_arm)
                        record["completed_arm_predictions"] += 1
                    detail.update(status="completed", wall_seconds=time.perf_counter() - started)
                    record["completed_frames"] += 1; checkpoint()
                    del packed, low, features, full, presence, native, zero, answer
                row["status"] = "completed"; checkpoint()
        require(record["completed_frames"] == record["native_parity_checks"] == 241
                and record["completed_arm_predictions"] == record["zero_head_parity_checks"] == 482
                and len(record["outputs"]) == 482 and not record["offline_attempts"]
                and not any(n == "sam2" or n.startswith("sam2.") for n in sys.modules),
                "complete offline two-arm prediction without SAM imports")
        require(all(torch.equal(v, models[a].state_dict()[k]) for a in ARMS for k, v in states[a].items()),
                "learned checkpoint state changed")
        for path, digest in fingerprints.items():
            require(sha(path) == digest, "source/input changed during prediction")
        record.update(status="completed", input_sha256=fingerprints, learned_weights_unchanged=True)
        checkpoint()
        c.save(out / "seal.json", dict(schema="evr0033.native-feature-seal.v1", complete=True,
            receipt_sha256=sha(out / "receipt.json"), source_sha256=record["source_sha256"],
            spec_sha256=record["spec_sha256"], frame_count=241, arm_prediction_count=482,
            outputs=record["outputs"], arms=list(ARMS)))
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        checkpoint()
        raise


if __name__ == "__main__":
    run(sys.argv[1])
