"""One-shot, offline manufactured-input qualification; never a pose benchmark.

Imports only the standard library until receipts and local files are verified.
This file does not install dependencies, acquire assets, or execute on import.
"""
import argparse
import contextlib
import errno
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[2]
SP_URL = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth"
LG_URL = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth"
MOGE_OPTIONS = dict(num_tokens=1200, use_fp16=False, apply_mask=False, force_projection=True)
SP_OPTIONS = dict(max_num_keypoints=2048)
LG_OPTIONS = dict(features="superpoint", flash=False, mp=False,
                  depth_confidence=-1, width_confidence=-1)
PINS = {"moge": "925b8ed835a7a9cdb7578ba15c658a0afc969030",
        "utils3d": "3fab839f0be9931dac7c8488eb0e1600c236e183",
        "lightglue": "eb42fee2d71449efb0aa5c10549752b5d75384d8"}


class QualificationError(RuntimeError):
    def __init__(self, code, details=None):
        super().__init__(code)
        self.details = details


def require(condition, code):
    if not condition:
        raise QualificationError(code)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    with Path(path).open() as f:
        return json.load(f, parse_constant=lambda _: (_ for _ in ()).throw(QualificationError("nonfinite_json")))


def frozen_bindings(args):
    helper = Path(__file__).with_name("supervise.py")
    spec = importlib.util.spec_from_file_location("_evr0022_qualification_supervisor", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = [Path(__file__), helper, helper.with_name("install_runtime.py"),
             helper.with_name("acquire.py"), helper.with_name("acquire_operation.py"),
             args.protocol, args.asset_manifest, args.wheel_assets_manifest, args.wheel_manifest,
             ROOT / "docs/literature/0022-runtime-source-contract.md"]
    return module.verify_frozen(args.expected_commit, paths)


def safe_child(root, relative):
    p = Path(relative)
    require(not p.is_absolute() and p.parts and all(x not in (".", "..") for x in p.parts), "unsafe_relative_path")
    out = Path(root) / p
    require(all(not parent.is_symlink() for parent in [out, *out.parents] if parent != Path(root).parent), "symlink_path")
    require(out.resolve().is_relative_to(Path(root).resolve()), "path_escape")
    return out


def verify_installation(args, installed, frozen):
    outer = read_json(args.installation_supervision)
    require(installed.get("expected_commit") == outer.get("expected_commit") == args.expected_commit,
            "installation_commit")
    required = [Path(__file__).with_name(x) for x in ("install_runtime.py", "acquire.py", "supervise.py")]
    required += [args.asset_manifest, args.wheel_assets_manifest, args.protocol]
    keys = {str(p.resolve().relative_to(ROOT)) for p in required}
    source = installed.get("source_sha256")
    require(isinstance(source, dict) and keys <= set(source) and set(source) <= set(frozen)
            and all(frozen[k] == v for k, v in source.items()), "installation_source_binding")
    outer_keys = {str(Path(__file__).with_name(x).resolve().relative_to(ROOT))
                  for x in ("supervise.py", "install_runtime.py")}
    require(outer.get("source_sha256") == {k: frozen[k] for k in outer_keys}, "installation_supervisor_binding")
    require(outer.get("schema") == "evr0022.supervision.v1" and outer.get("operation") == "installation"
            and outer.get("status") == "completed" and outer.get("exit_code") == 0
            and outer.get("termination") is None and outer.get("remaining_descendant_pids") == []
            and outer.get("cleanup_errors") == [], "installation_supervision_failed")
    wall, rss = outer.get("wall_seconds"), outer.get("peak_group_rss_kib")
    require(type(wall) in (int, float) and math.isfinite(wall) and 0 <= wall < 1800
            and type(rss) is int and 0 <= rss <= 8 * 1024 * 1024
            and outer.get("hard_timeout_seconds") == 1800
            and outer.get("group_rss_limit_kib") == 8 * 1024 * 1024, "installation_resources")
    log_hashes = {}
    for name in ("stdout", "stderr"):
        p = safe_child(args.installation_supervision.parent, name + ".log")
        log_hashes[name] = sha(p)
        require(log_hashes[name] == outer.get(name + "_sha256"), "installation_log_changed")
    command = outer.get("command")
    def display(value):
        return str(value).replace(str(sys.executable), "$PYTHON").replace(str(ROOT), "$REPOSITORY").replace(str(ROOT.parent), "$PRIVATE")
    require(isinstance(command, list) and len(command) >= 2 and command[0] == "$PYTHON"
            and command[1] == display(Path(__file__).with_name("install_runtime.py")), "installation_child_script")
    flags = {}
    require(len(command[2:]) % 2 == 0, "installation_child_arguments")
    for key, value in zip(command[2::2], command[3::2]):
        require(key.startswith("--") and key not in flags, "installation_child_arguments")
        flags[key] = value
    for key, value in {"--destination": args.installation, "--source-assets": args.assets_dir,
                       "--source-manifest": args.asset_manifest, "--wheel-manifest": args.wheel_assets_manifest,
                       "--expected-commit": args.expected_commit}.items():
        observed = flags.get(key)
        require(isinstance(observed, str), "installation_child_input")
        if key == "--expected-commit":
            require(observed == value, "installation_child_input")
        else:
            expanded = observed.replace("$REPOSITORY", str(ROOT)).replace("$PRIVATE", str(ROOT.parent))
            child_path = Path(expanded)
            child_path = child_path if child_path.is_absolute() else ROOT / child_path
            require(child_path.resolve() == Path(value).resolve(), "installation_child_input")
    return {"installation_supervision_sha256": sha(args.installation_supervision),
            "installation_supervision_log_sha256": log_hashes}


def verify_files(args, frozen):
    """No model import. Bind exact manifests, acquired files and installation."""
    manifest, wheels = read_json(args.asset_manifest), read_json(args.wheel_manifest)
    wheel_assets = read_json(args.wheel_assets_manifest)
    ledger = read_json(args.acquisition_receipt)
    install_path = args.installation / "install-receipt.json"
    installed = read_json(install_path)
    supervision_binding = verify_installation(args, installed, frozen)
    require(list(sys.version_info[:3]) == [3, 9, 6], "python_version")
    require(ledger.get("status") == installed.get("status") == "completed", "incomplete_acquisition_or_install")
    require(installed.get("schema") == "evr0022.install.v1", "install_schema")
    require(ledger.get("manifest_sha256") == canonical(manifest), "asset_ledger_manifest")
    expected = {"source_manifest_sha256": sha(args.asset_manifest),
                "source_manifest_canonical_sha256": canonical(manifest),
                "wheel_manifest_sha256": sha(args.wheel_assets_manifest),
                "wheel_manifest_canonical_sha256": canonical(wheel_assets),
                "source_acquisition_ledger_sha256": sha(args.acquisition_receipt)}
    require(all(installed.get(k) == v for k, v in expected.items()), "install_input_binding")
    expected_wheels = {x["filename"]: x for x in wheels["wheels"]}
    require(len(wheel_assets["assets"]) == len(expected_wheels), "wheel_manifest_coverage")
    for row in wheel_assets["assets"]:
        m = expected_wheels.get(row["filename"], {})
        require(row.get("bytes") == m.get("size") and row.get("sha256") == m.get("sha256")
                and row.get("url") == m.get("url"), "wheel_metadata_asset_binding")
    for dep in wheels["reuse_installed"]:
        dist = importlib.metadata.distribution(dep["name"])
        require(dist.version == dep["version"] and hashlib.sha256(dist.read_text("METADATA").encode()).hexdigest()
                == dep["installed_metadata_text_sha256"], "reused_distribution_changed")
    records = {x["id"]: x for x in ledger["assets"]}
    require(len(records) == len(ledger["assets"]) == len(manifest["assets"]), "asset_coverage")
    files = {}
    for asset in manifest["assets"]:
        p = safe_child(args.assets_dir, asset["filename"])
        rec = records[asset["id"]]
        require(rec.get("status") in ("downloaded_verified", "reused_verified", "downloaded_size_checked_first_hash"), "asset_status")
        require(p.is_file() and p.stat().st_size == asset["bytes"], "asset_bytes")
        digest = sha(p)
        require(rec.get("sha256") == digest and asset.get("sha256", digest) == digest, "asset_hash")
        if "git_blob_sha1" in asset:
            require(hashlib.sha1(b"blob " + str(asset["bytes"]).encode() + b"\0" + p.read_bytes()).hexdigest() == asset["git_blob_sha1"], "asset_git_blob")
        files[asset["filename"]] = p
    inventory = installed.get("installed_files")
    sizes = installed.get("installed_file_bytes")
    require(isinstance(inventory, dict) and inventory and isinstance(sizes, dict), "install_inventory")
    require(set(sizes) == set(inventory), "inventory_size_coverage")
    for rel, digest in inventory.items():
        p = safe_child(args.installation, rel)
        require(p.is_file() and p.stat().st_size == sizes[rel] and sha(p) == digest, "installed_file_changed")
    actual = {str(p.relative_to(args.installation)) for base in ("sources", "runtime")
              for p in (args.installation / base).rglob("*") if p.is_file()}
    require(actual == {x for x in inventory if x.startswith(("sources/", "runtime/"))}, "unlisted_runtime_file")
    layout = manifest["source_layout"]
    require({x["install_prefix"] for x in layout} == set(PINS), "source_prefixes")
    for row in layout:
        require(row["revision"] == PINS[row["install_prefix"]], "source_revision")
        rel = "sources/" + row["install_prefix"] + "/" + row["relative_path"]
        require(inventory.get(rel) == sha(files[row["filename"]]), "source_copy_binding")
    model_names = ("moge2-small-model.pt", "superpoint_v1.pth", "superpoint_lightglue.pth")
    require(all(x in files for x in model_names), "missing_checkpoint")
    return files, installed, wheels, dict(expected, install_receipt_sha256=sha(install_path),
        **supervision_binding,
        dependency_metadata_sha256=sha(args.wheel_manifest),
        protocol_sha256=sha(args.protocol), source_sha256=sha(__file__),
        checkpoint_sha256={x: sha(files[x]) for x in model_names})


def optional_numpy_sve_probe(event, arguments):
    """Recognize only NumPy's optional, absent-on-macOS lscpu import probe."""
    if (sys.platform != "darwin" or event != "subprocess.Popen" or len(arguments) != 4
            or arguments[0] != "lscpu" or arguments[1] != ["lscpu"]
            or arguments[2] is not None or arguments[3] is not None):
        return False
    frame = sys._getframe(1)
    try:
        while frame is not None:
            if frame.f_code.co_name == "run" and frame.f_globals.get("__name__") == "subprocess":
                local = frame.f_locals
                caller = frame.f_back
                module = sys.modules.get("numpy.testing._private.utils")
                return (local.get("popenargs") == ("lscpu",)
                        and local.get("capture_output") is True
                        and local.get("input") is None and local.get("timeout") is None
                        and local.get("check") is False
                        and local.get("kwargs") == {"text": True, "stdout": -1, "stderr": -1}
                        and caller is not None and caller.f_code.co_name == "check_support_sve"
                        and caller.f_globals.get("__name__") == "numpy.testing._private.utils"
                        and caller.f_code.co_filename.endswith("/numpy/testing/_private/utils.py")
                        and module is not None and module.__dict__ is caller.f_globals
                        and getattr(module, "__file__", None) == caller.f_code.co_filename)
            frame = frame.f_back
        return False
    finally:
        del frame


def install_guard(attempts, compatibility_probes=None):
    """Python audit hook persists for this dedicated process; no external child/network."""
    forbidden = {"socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.gethostbyname",
                 "socket.gethostbyaddr", "socket.sendto", "subprocess.Popen", "os.system",
                 "os.exec", "os.posix_spawn", "os.fork", "os.forkpty", "builtins.input"}
    def audit(event, arguments):
        if event in forbidden:
            if compatibility_probes is not None and optional_numpy_sve_probe(event, arguments):
                compatibility_probes.append({"event": event, "command": ["lscpu"],
                    "caller": "numpy.testing._private.utils.check_support_sve",
                    "outcome": "suppressed_optional_probe", "exception": "FileNotFoundError",
                    "reason": "macOS optional SVE probe treated as unavailable; no process executed"})
                raise FileNotFoundError(errno.ENOENT, "Optional NumPy SVE probe unavailable on macOS", "lscpu")
            attempts.append({"event": event})
            raise QualificationError("offline_guard_" + event)
    sys.addaudithook(audit)


@contextlib.contextmanager
def capture_warnings(record):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            yield
        finally:
            record["warnings"].extend({"category": type(w.message).__name__, "message": str(w.message)} for w in caught)


def validate_config(config, classes):
    require(isinstance(config, dict), "moge_config_not_dict")
    def check(values, cls, name):
        require(isinstance(values, dict), name + "_config_not_dict")
        sig = inspect.signature(cls.__init__)
        allowed = {k for k, v in sig.parameters.items() if k != "self" and
                   v.kind not in (v.VAR_KEYWORD, v.VAR_POSITIONAL)}
        require(set(values) <= allowed, name + "_unknown_config")
        required = {k for k, v in sig.parameters.items() if k in allowed and v.default is v.empty}
        require(required <= set(values), name + "_missing_config")
    check(config, classes["model"], "moge")
    check(config["encoder"], classes["encoder"], "encoder")
    for key in ("neck", "points_head", "mask_head", "normal_head"):
        if config.get(key) is not None:
            check(config[key], classes["conv"], key)
    if config.get("scale_head") is not None:
        check(config["scale_head"], classes["mlp"], "scale_head")
    canonical(config)  # configuration must be finite JSON, no opaque objects


def rename_lightglue(state, layers):
    out = {}
    for key, value in state.items():
        new = key
        for i in range(layers):
            new = new.replace("self_attn.%d" % i, "transformers.%d.self_attn" % i)
            new = new.replace("cross_attn.%d" % i, "transformers.%d.cross_attn" % i)
        require(new not in out, "legacy_key_collision")
        out[new] = value
    return out


def strict_state(model, state, torch, allow_confidence=False):
    require(isinstance(state, dict) and all(isinstance(k, str) and isinstance(v, torch.Tensor)
                                          for k, v in state.items()), "invalid_state_dict")
    expected = model.state_dict()
    missing, unexpected = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    allowed = {"confidence_thresholds"} if allow_confidence else set()
    if unexpected or not set(missing) <= allowed:
        raise QualificationError("state_key_mismatch", {"missing_keys": missing,
                                 "unexpected_keys": unexpected, "allowed_missing_buffers": sorted(allowed)})
    learned = dict(model.named_parameters())
    require(not (set(learned) & set(missing)), "missing_learned_weight")
    enriched = dict(state)
    if allow_confidence:
        deterministic = torch.tensor([model.confidence_threshold(i) for i in range(model.conf.n_layers)], dtype=expected["confidence_thresholds"].dtype)
        require(torch.equal(expected["confidence_thresholds"], deterministic), "confidence_buffer_formula")
        if "confidence_thresholds" in state:
            require(torch.equal(state["confidence_thresholds"].cpu(), deterministic), "checkpoint_confidence_buffer")
        enriched["confidence_thresholds"] = deterministic
    for key, value in enriched.items():
        require(tuple(value.shape) == tuple(expected[key].shape), "state_shape_mismatch")
        require(bool(torch.isfinite(value).all()), "nonfinite_weight")
    model.load_state_dict(enriched, strict=True)
    loaded = model.state_dict()
    require(all(torch.equal(loaded[k].cpu(), v.to(dtype=loaded[k].dtype, device="cpu"))
                for k, v in enriched.items()), "loaded_weight_value_mismatch")
    return {"missing_keys": missing, "unexpected_keys": unexpected,
            "allowed_missing_buffers": sorted(allowed), "learned_keys": sorted(learned),
            "state_shapes": {k: list(v.shape) for k, v in loaded.items()},
            "strict_after_deterministic_buffer_completion": True}


@contextlib.contextmanager
def local_hub(torch, paths, calls):
    original = torch.hub.load_state_dict_from_url
    def local(url, *a, **kw):
        require(url in paths, "unapproved_hub_url")
        require(not a and set(kw) <= {"file_name"}, "unexpected_hub_options")
        calls.append({"asset": paths[url].name, "url": url, "weights_only": True})
        return torch.load(paths[url], map_location="cpu", weights_only=True)
    torch.hub.load_state_dict_from_url = local
    try:
        yield
    finally:
        torch.hub.load_state_dict_from_url = original


def fixture(np):
    """Fixed before outputs: no photometric/feature-based fixture selection."""
    h, w = 384, 512
    y, x = np.mgrid[:h, :w]
    gradient = np.stack((x / (w - 1), y / (h - 1), (x + y) / (w + h - 2)), axis=-1)
    a = (0.55 * gradient + 0.35 * np.random.default_rng(0).random((h, w, 3)) + 0.05)
    squares = [(36, 46, 42, [0.95, 0.1, 0.2]), (118, 207, 64, [0.1, 0.9, 0.3]),
               (265, 352, 70, [0.2, 0.25, 0.95]), (75, 395, 35, [0.92, 0.92, 0.2])]
    for yy, xx, side, color in squares:
        a[yy:yy + side, xx:xx + side] = color
    a = np.ascontiguousarray(a.astype(np.float32))
    b = np.zeros_like(a); b[:, 12:] = a[:, :-12]
    return [a, b], {"seed": 0, "shape_hwc": [h, w, 3], "dtype": "float32",
        "formula": "0.55*RGB_gradient + 0.35*default_rng(0).random + 0.05; overwrite four squares",
        "squares_y_x_side_rgb": squares, "second_image": "right shift 12 pixels, zero left edge"}


def configure_execution(moge, torch, cv2):
    """Apply the declared authored attention path after strict weight loading."""
    moge.enable_pytorch_native_sdpa()
    cv2.setNumThreads(1)
    settings = {"moge_native_sdpa_enabled": True,
                "torch_intraop_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
                "opencv_threads": cv2.getNumThreads()}
    require(settings["torch_intraop_threads"] == 4 and settings["torch_interop_threads"] == 1
            and settings["opencv_threads"] == 1, "execution_thread_configuration")
    return settings


def save_arrays(path, arrays, np):
    with path.open("xb") as f:
        np.savez(f, **arrays)
    return {"filename": path.name, "sha256": sha(path), "bytes": path.stat().st_size,
            "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                            "raw_c_order_sha256": hashlib.sha256(v.tobytes(order="C")).hexdigest()}
                       for k, v in arrays.items()}}


def qualify(args):
    require(not args.output_dir.exists(), "output_directory_exists")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record = {"schema": "evr0022.qualification.v1", "status": "started", "eligible": False,
              "stage": "verification", "source_sha256": sha(__file__), "settings": {
                  "moge": MOGE_OPTIONS, "superpoint": SP_OPTIONS, "extract_resize": None,
                  "lightglue": LG_OPTIONS, "device": "cpu", "dtype": "float32", "threads": 4},
              "timings": {}, "warnings": [], "offline_attempts": [], "compatibility_probes": [],
              "local_hub_calls": []}
    def save():
        tmp = args.output_dir / "receipt.pending"
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(tmp, args.output_dir / "receipt.json")
    def stage(name):
        record["stage"] = name; save()
    save()
    try:
        frozen = frozen_bindings(args)
        record.update(expected_commit=args.expected_commit, frozen_files=frozen)
        files, installed, wheels, bindings = verify_files(args, frozen)
        record["bindings"] = bindings
        record["timings"]["verification_seconds"] = time.perf_counter() - started
        os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          XFORMERS_DISABLED="1", PYTHONDONTWRITEBYTECODE="1")
        sys.dont_write_bytecode = True
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = "4"
        sys.path[:0] = [str(args.installation / "runtime")] + [str(args.installation / "sources" / k) for k in PINS]
        install_guard(record["offline_attempts"], record["compatibility_probes"])
        record["offline_guard_scope"] = "Python audit events for socket/DNS/process/input; not an OS sandbox for arbitrary native syscalls"
        stage("imports"); t0 = time.perf_counter()
        import numpy as np
        import torch
        import cv2
        import kornia
        from kornia.config import InstallationMode, kornia_config
        kornia_config.lazyloader.installation_mode = InstallationMode.RAISE
        from moge.model.v2 import MoGeModel
        from moge.model.modules import DINOv2Encoder, ConvStack, MLP
        from lightglue import SuperPoint, LightGlue
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        versions = {x["name"]: x["version"] for x in wheels["wheels"]}
        reuse = wheels["reuse_installed"]
        reuse = reuse.values() if isinstance(reuse, dict) else reuse
        versions.update({x["name"]: x["version"] for x in reuse})
        require(all(importlib.metadata.version(k) == v for k, v in versions.items()), "runtime_version")
        record["runtime_versions"] = versions
        for package in PINS:
            module_file = Path(sys.modules[package].__file__).resolve()
            require(module_file.is_relative_to((args.installation / "sources" / package).resolve()), "source_import_origin")
        record["timings"]["imports_seconds"] = time.perf_counter() - t0
        require(not record["offline_attempts"], "blocked_import_operation")
        stage("model_loading"); t0 = time.perf_counter()
        with capture_warnings(record):
            checkpoint = torch.load(files["moge2-small-model.pt"], map_location="cpu", weights_only=True)
            require(isinstance(checkpoint, dict) and "model_config" in checkpoint and "model" in checkpoint, "moge_checkpoint_structure")
            config = checkpoint["model_config"]
            canonical(config)
            record["moge_config"] = config
            validate_config(config, dict(model=MoGeModel, encoder=DINOv2Encoder, conv=ConvStack, mlp=MLP))
            record["moge_config_sha256"] = canonical(config)
            moge = MoGeModel(**config)
            record["loads"] = {"moge": strict_state(moge, checkpoint["model"], torch)}
            del checkpoint
            record["timings"]["moge_load_seconds"] = time.perf_counter() - t0
            paths = {SP_URL: files["superpoint_v1.pth"], LG_URL: files["superpoint_lightglue.pth"]}
            with local_hub(torch, paths, record["local_hub_calls"]):
                t0 = time.perf_counter(); sp = SuperPoint(**SP_OPTIONS)
                state = torch.load(paths[SP_URL], map_location="cpu", weights_only=True)
                record["loads"]["superpoint"] = strict_state(sp, state, torch); del state
                record["timings"]["superpoint_load_seconds"] = time.perf_counter() - t0
                t0 = time.perf_counter(); lg = LightGlue(**LG_OPTIONS)
                state = rename_lightglue(torch.load(paths[LG_URL], map_location="cpu", weights_only=True), lg.conf.n_layers)
                record["loads"]["lightglue"] = strict_state(lg, state, torch, allow_confidence=True); del state
                record["timings"]["lightglue_load_seconds"] = time.perf_counter() - t0
        require(not any("ignored" in w["message"].lower() or "invalid arguments" in w["message"].lower() for w in record["warnings"]), "ignored_model_config")
        require([x["url"] for x in record["local_hub_calls"]] == [SP_URL, LG_URL], "hub_call_count")
        for model in (moge, sp, lg):
            model.eval().to(device="cpu", dtype=torch.float32)
            require(all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters()), "model_device_dtype")
        record["execution_configuration"] = configure_execution(moge, torch, cv2)
        images, record["fixture"] = fixture(np)
        record["input_arrays"] = save_arrays(args.output_dir / "inputs.npz", {"image0": images[0], "image1": images[1]}, np)
        tensors = [torch.from_numpy(x.transpose(2, 0, 1).copy()) for x in images]
        record["depth_outputs"], record["feature_outputs"] = [], []
        features = []
        with torch.inference_mode():
            for i, tensor in enumerate(tensors):
                stage("moge_image%d" % i); t0 = time.perf_counter()
                with capture_warnings(record):
                    out = moge.infer(tensor, **MOGE_OPTIONS)
                record["timings"]["moge_image%d_seconds" % i] = time.perf_counter() - t0
                require({"depth", "mask", "points", "intrinsics"} <= set(out), "missing_depth_outputs")
                raw = {k: v.detach().cpu().numpy() for k, v in out.items()}
                cached = save_arrays(args.output_dir / ("depth%d.npz" % i), raw, np)
                record["depth_outputs"].append(cached)
                d, mask = raw["depth"], raw["mask"]
                require(d.shape == mask.shape == (384, 512) and d.dtype == np.float32 and mask.dtype == np.bool_, "depth_shape_dtype")
                require(raw["points"].shape == (384, 512, 3) and raw["intrinsics"].shape == (3, 3), "depth_geometry_shape")
                require(np.isfinite(raw["intrinsics"]).all(), "nonfinite_intrinsics")
                require(bool(mask.any()), "zero_valid_depth")
                require(np.isfinite(d[mask]).all() and (d[mask] > 0).all() and np.isfinite(raw["points"][mask]).all(), "invalid_valid_depth")
                if "normal" in raw:
                    require(raw["normal"].shape == (384, 512, 3) and np.isfinite(raw["normal"][mask]).all(), "invalid_valid_normals")
                cached.update(valid_pixels=int(mask.sum()), valid_depth_min=float(d[mask].min()),
                              valid_depth_max=float(d[mask].max()), raw_nonfinite_pixels=int((~np.isfinite(d)).sum()))
                del out, raw
            for i, tensor in enumerate(tensors):
                stage("superpoint_image%d" % i); t0 = time.perf_counter()
                feature = sp.extract(tensor, resize=None)
                record["timings"]["superpoint_image%d_seconds" % i] = time.perf_counter() - t0
                raw = {k: v.detach().cpu().numpy() for k, v in feature.items()}
                cached = save_arrays(args.output_dir / ("features%d.npz" % i), raw, np)
                record["feature_outputs"].append(cached)
                k, desc = raw["keypoints"], raw["descriptors"]
                require(k.ndim == 3 and k.shape[0] == 1 and k.shape[2] == 2 and 0 < k.shape[1] <= 2048, "keypoint_count_shape")
                require(desc.shape == (1, k.shape[1], 256) and k.dtype == desc.dtype == np.float32
                        and all(np.isfinite(v).all() for v in raw.values()), "feature_shape_finiteness")
                require((k >= 0).all() and (k[..., 0] < 512).all() and (k[..., 1] < 384).all(), "keypoint_bounds")
                require(np.array_equal(raw["image_size"], [[512, 384]]), "image_size_mapping")
                cached["keypoint_count"] = k.shape[1]
                features.append(feature)
            stage("lightglue_pair"); t0 = time.perf_counter()
            matches = lg({"image0": features[0], "image1": features[1]})
            record["timings"]["lightglue_pair_seconds"] = time.perf_counter() - t0
            raw = {k: (v[0].detach().cpu().numpy() if isinstance(v, list) else v.detach().cpu().numpy())
                   for k, v in matches.items() if k != "stop"}
            raw["stop"] = np.asarray(matches["stop"], dtype=np.int64)
            record["match_output"] = save_arrays(args.output_dir / "matches.npz", raw, np)
            require(all(np.isfinite(v).all() for v in raw.values()), "nonfinite_matching_output")
            pairs, scores = raw["matches"], raw["scores"]
            if pairs.ndim == 3:
                require(pairs.shape[0] == 1, "match_batch"); pairs, scores = pairs[0], scores[0]
            require(pairs.ndim == 2 and pairs.shape[1] == 2 and np.issubdtype(pairs.dtype, np.integer), "match_shape_dtype")
            require(len(pairs) > 0 and scores.shape == (len(pairs),) and np.isfinite(scores).all(), "zero_or_invalid_matches")
            n0, n1 = [x["keypoint_count"] for x in record["feature_outputs"]]
            require((pairs >= 0).all() and (pairs[:, 0] < n0).all() and (pairs[:, 1] < n1).all(), "match_bounds")
            require(len(np.unique(pairs[:, 0])) == len(pairs) == len(np.unique(pairs[:, 1])), "nonmutual_matches")
            for name, n, other in (("matches0", n0, n1), ("matches1", n1, n0)):
                require(raw[name].shape == (1, n) and (raw[name] >= -1).all() and (raw[name] < other).all(), "dense_match_bounds")
            require(np.array_equal(raw["matches0"][0, pairs[:, 0]], pairs[:, 1]) and
                    np.array_equal(raw["matches1"][0, pairs[:, 1]], pairs[:, 0]), "compact_dense_match_parity")
            record["match_output"].update(match_count=len(pairs), stop=int(matches["stop"]))
        stage("postcheck")
        require(not record["offline_attempts"], "offline_guard_attempted")
        _, _, _, final_bindings = verify_files(args, frozen)
        require(final_bindings == bindings, "input_changed_during_qualification")
        require(all(sha(ROOT / rel) == digest for rel, digest in frozen.items()), "frozen_source_changed")
        record.update(status="completed", eligible=True, stage="completed")
    except BaseException as exc:
        record.update(status="failed", eligible=False, failed_stage=record["stage"],
                      error=(str(exc) if isinstance(exc, QualificationError) else type(exc).__name__))
        if isinstance(exc, QualificationError) and exc.details is not None:
            record["error_details"] = exc.details
        raise
    finally:
        record["wall_seconds"] = time.perf_counter() - started
        save()
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--installation", type=Path, required=True)
    p.add_argument("--installation-supervision", type=Path, required=True)
    p.add_argument("--asset-manifest", type=Path, default=ROOT / "configs/0022-source-model-assets.json")
    p.add_argument("--wheel-manifest", type=Path, default=ROOT / "results/0022-preparation/dependency-wheels.json")
    p.add_argument("--wheel-assets-manifest", type=Path, default=ROOT / "configs/0022-wheel-assets.json")
    p.add_argument("--expected-commit", required=True)
    p.add_argument("--acquisition-receipt", type=Path, required=True)
    p.add_argument("--assets-dir", type=Path, required=True)
    p.add_argument("--protocol", type=Path, default=ROOT / "protocols/0022-traceable-frontend-assets.md")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    result = qualify(args)
    print(json.dumps({"status": result["status"], "eligible": result["eligible"], "wall_seconds": result["wall_seconds"]}))


if __name__ == "__main__":
    main()
