"""One native TAPNext++ CPU capability attempt behind the frozen supervisor.

Importing this module imports no ML packages and runs nothing. The caller must
supply exact package versions from a successful reviewed cloud import preflight,
and an outer 1800 s timeout / 5 s kill-after supervisor. This module never
acquires assets, installs packages, changes a state, or performs a warmup forward.
"""
import hashlib
import importlib
import importlib.metadata
from importlib.machinery import PathFinder
import json
import os
import platform
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone

SOURCE_REV = "730cda1c730877cfedbe01bf87fb1cadb78a565d"
HF_REV = "5d3fb48e76c5422841e38501514121e251beabb7"
WEIGHT_BYTES = 2532282370
WEIGHT_SHA = "cb96a43444ccb4fbdb25d800b88c7ba196179a526e78f01b021a16b1c1eff6da"
NPZ_SHA = "162350eb76f8399df95e7ded23eefd82793ea03db05dd2fe5101ef1027dfddf0"
SOURCES = {
    "tapnet.tapnext.tapnext_torch": "48072289b22439267b3a51eb4cde5e2f81f38c840b6c95b659d60032a07c691a",
    "tapnet.tapnext.tapnext_lru_modules": "2a475419c2f7c67db9acdf412108fb3b025d168f60e9e087fe35d65157adf64f",
    "tapnet.tapnext.pscan": "7df34bf5e8318e81bb330ded14fc5e675e7485ea39fa9f0d77fa470f85796a2d",
    "tapnet.tapnextpp.votsp2026.model": "e1219e2341c33e0f9124fe02d09a521ac7c91bc27505fcbebbf05741f825e46b",
    "tapnet.tapnextpp.votsp2026.utils": "0336c526dc2b91244c3e5b803f84b65c6323ede138340a4107441a6b9c9fdbf8",
}
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def verify_source_tree(source_root):
    """Reject unbound parent initializers before any TAPNet code executes."""
    source_root = Path(source_root)
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError("Expected a regular independent source directory")
    files = {name.replace(".", "/") + ".py": digest for name, digest in SOURCES.items()}
    dirs = {str(parent) for name in files for parent in Path(name).parents if str(parent) != "."}
    observed = set()
    for path in source_root.rglob("*"):
        name = path.relative_to(source_root).as_posix()
        if path.is_symlink():
            raise ValueError("Source symlink is not a frozen input")
        if path.is_dir() and name in dirs:
            continue
        if not path.is_file() or name not in files or sha(path) != files[name]:
            raise ValueError("Unexpected or modified source tree entry: " + name)
        observed.add(name)
    if observed != set(files):
        raise ValueError("Pinned source file missing")
    if any(name == "tapnet" or name.startswith("tapnet.") for name in sys.modules):
        raise ValueError("TAPNet must not already be imported")
    # A later installed regular package can override an earlier namespace path.
    # Inspect the top-level spec without importing it or executing an initializer.
    spec = PathFinder.find_spec("tapnet", sys.path)
    wanted = (source_root / "tapnet").resolve()
    if (spec is None or spec.loader is not None or spec.origin is not None
            or [Path(p).resolve() for p in spec.submodule_search_locations or []] != [wanted]):
        raise ValueError("Only the frozen TAPNet namespace may resolve")


def read_fixed_inputs(npz_path, np):
    """Interpret only video frames 0..7 and point/visibility entries (0..7, 0).

    The cached archive is np.savez/ZIP_STORED. Header parsing is bounded and uses
    NumPy's NPY format parser; no pickle and no wholesale GT-array np.load call.
    A seek may traverse uninterpreted ZIP bytes. SHA checks read all file bytes.
    """
    def header(archive, name, shape, kinds):
        info = archive.getinfo(name)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError("Expected stored cached NPZ; no decompression fallback")
        stream = archive.open(info)
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            got, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            got, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("Unexpected NPY version")
        if got != shape or fortran or dtype.hasobject or dtype.kind not in kinds:
            raise ValueError("Unexpected numeric cache array schema")
        offset = stream.tell()
        if offset + int(np.prod(shape)) * dtype.itemsize != info.file_size:
            raise ValueError("NPY member byte size differs")
        return stream, offset, dtype

    with zipfile.ZipFile(npz_path) as archive:
        if sorted(archive.namelist()) != ["occluded.npy", "points.npy", "video.npy"]:
            raise ValueError("Unexpected cached NPZ members")
        stream, _, video_dtype = header(archive, "video.npy", (250, 256, 256, 3), "u")
        if video_dtype.itemsize != 1:
            stream.close()
            raise ValueError("Expected uint8 RGB")
        with stream:
            raw = stream.read(8 * 256 * 256 * 3)
            if len(raw) != 8 * 256 * 256 * 3:
                raise ValueError("Truncated fixed RGB prefix")
        frames = np.frombuffer(raw, dtype=np.uint8).reshape(8, 256, 256, 3).copy()
        stream, offset, dtype = header(archive, "points.npy", (30, 250, 2), "f")
        if dtype.itemsize not in (4, 8):
            raise ValueError("Unexpected point precision")
        queries = []
        with stream:
            for point in range(8):
                stream.seek(offset + point * 250 * 2 * dtype.itemsize)
                value = stream.read(2 * dtype.itemsize)
                if len(value) != 2 * dtype.itemsize:
                    raise ValueError("Truncated initial query")
                queries.append(np.frombuffer(value, dtype=dtype).copy())
        stream, offset, dtype = header(archive, "occluded.npy", (30, 250), "b")
        if dtype.itemsize != 1:
            raise ValueError("Unexpected visibility precision")
        with stream:
            for point in range(8):
                stream.seek(offset + point * 250)
                if stream.read(1) != b"\x00":
                    raise ValueError("Prespecified query is not visible at frame zero")
    xy = np.asarray(queries, dtype=np.float32) * np.float32(256)
    if not np.isfinite(xy).all():
        raise ValueError("Nonfinite initial query")
    return frames, xy


def run_capability(*, checkpoint, development_npz, source_root, output,
                   marker, expected_packages, overlay_root, base_runtime_root,
                   provenance):
    """Execute only through the separately reviewed, frozen-source supervisor.

    expected_packages must have exactly torch/numpy/torchvision/einops/Pillow with exact
    versions recorded by a prior successful import-only preflight. Missing or
    incompatible dependencies stop this candidate; this function never installs.
    The only scientific forwards are the eight track_frame calls below.
    """
    output = Path(output); output.mkdir(exist_ok=False)
    checkpoint = Path(checkpoint).resolve(); development_npz = Path(development_npz).resolve()
    source_root = Path(source_root); marker = Path(marker)
    overlay_root = Path(overlay_root).resolve(); base_runtime_root = Path(base_runtime_root).resolve()
    started = time.perf_counter(); old_alarm_handler = signal.getsignal(signal.SIGALRM)
    receipt = {"status": "preflight", "started_utc": utc(), "scope": "native CPU capability only",
               "python": sys.version, "platform": platform.platform(), "command": sys.argv,
               "source_sha256": SOURCES, "checkpoint_sha256": WEIGHT_SHA,
               "development_npz_sha256": NPZ_SHA,
               "source_revision": SOURCE_REV, "checkpoint_revision": HF_REV,
               "adapter_sha256": sha(__file__), "forwards_attempted": 0,
               "frames_completed": 0, "frames": [], "phases_seconds": {},
               "acquisition": "none; existing hash-verified local assets only",
               "provenance": provenance}
    def persist():
        receipt["elapsed_seconds"] = time.perf_counter() - started
        receipt["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save(output / "receipt.json", receipt)
    def deadline(signum, frame):
        raise TimeoutError("1790 second inner wall deadline")
    try:
        if sys.platform != "linux" or sys.version_info < (3, 10):
            raise ValueError("This reviewed runtime target is Linux Python >=3.10")
        if any(os.environ.get(name) != "4" for name in THREAD_VARS):
            raise ValueError("Launcher must set all five thread variables to 4 before imports")
        affinity = sorted(os.sched_getaffinity(0))[:4]
        if len(affinity) != 4:
            raise ValueError("Fewer than four allowed CPUs; await reviewed preflight")
        os.sched_setaffinity(0, affinity)
        resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
        signal.signal(signal.SIGALRM, deadline); signal.alarm(1790)
        receipt["limits"] = {"affinity": affinity, "AS_bytes": 8 * 1024**3,
                             "outer_timeout_required_seconds": 1800,
                             "outer_kill_after_required_seconds": 5}
        if set(expected_packages) != {"torch", "numpy", "torchvision", "einops", "Pillow"}:
            raise ValueError("Exact preflight package pins are required")
        if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or not sys.dont_write_bytecode:
            raise ValueError("Launcher must keep the old runtime bytecode read-only")
        if any(name in sys.modules for name in ("torch", "numpy", "torchvision", "einops", "PIL")):
            raise ValueError("Use a fresh process; do not repurpose a loaded runtime")
        if not overlay_root.is_dir() or overlay_root == base_runtime_root or overlay_root.is_relative_to(base_runtime_root):
            raise ValueError("Require a separate existing dependency overlay")
        if any((overlay_root / name).exists() for name in ("torch", "numpy")):
            raise ValueError("Do not install or shadow old torch/numpy in the overlay")
        sys.path.insert(0, str(overlay_root))
        actual = {name: importlib.metadata.version(name) for name in expected_packages}
        if actual != expected_packages or actual["torch"] != "2.10.0+cpu":
            raise ValueError("Preflight package versions absent or changed")
        if checkpoint.stat().st_size != WEIGHT_BYTES or sha(checkpoint) != WEIGHT_SHA:
            raise ValueError("Official 256px checkpoint size or SHA differs")
        if sha(development_npz) != NPZ_SHA:
            raise ValueError("Existing development video-0 NPZ changed")
        sys.path.insert(0, str(source_root))
        verify_source_tree(source_root)
        source_root = source_root.resolve()
        import numpy as np
        import torch
        import torchvision
        import einops
        import PIL
        for name, package in (("torch", torch), ("numpy", np), ("torchvision", torchvision),
                              ("einops", einops), ("Pillow", PIL)):
            if str(package.__version__) != expected_packages[name]:
                raise ValueError("Imported module version differs: " + name)
        for package in (torch, np):
            if not Path(package.__file__).resolve().is_relative_to(base_runtime_root):
                raise ValueError("Old torch/numpy runtime identity changed")
        for package in (torchvision, einops, PIL):
            if not Path(package.__file__).resolve().is_relative_to(overlay_root):
                raise ValueError("Extra dependency was not loaded from independent overlay")
        if torch.version.cuda is not None or torch.get_default_dtype() != torch.float32:
            raise ValueError("CPU float32 runtime required")
        torch.set_num_threads(4); torch.set_num_interop_threads(4)
        torch.manual_seed(20261018); np.random.seed(20261018)
        torch.use_deterministic_algorithms(True)
        verify_source_tree(source_root)
        wrapper_module = importlib.import_module("tapnet.tapnextpp.votsp2026.model")
        for module, digest in SOURCES.items():
            loaded = importlib.import_module(module)
            expected = source_root / (module.replace(".", "/") + ".py")
            if Path(loaded.__file__).resolve() != expected or sha(expected) != digest:
                raise ValueError("Unexpected actual imported source: " + module)
        receipt["cpu_model"] = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
        receipt["cgroup_memory_max"] = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        receipt["packages"] = actual
        receipt["imports"] = {m.__name__: {"path": m.__file__, "sha256": sha(m.__file__)}
                              for m in [np, torch, torchvision, einops, PIL]}
        receipt["thread_counts"] = [torch.get_num_threads(), torch.get_num_interop_threads()]
        frames, queries_xy = read_fixed_inputs(development_npz, np)
        receipt["phases_seconds"]["input_verification_and_imports"] = time.perf_counter() - started
        with marker.open("x") as handle:
            json.dump({"started_utc": receipt["started_utc"], "adapter_sha256": sha(__file__),
                       "provenance": provenance,
                       "output": str(output.resolve()), "pid": os.getpid()}, handle)
        receipt["marker_sha256"] = sha(marker); receipt["status"] = "loading"; receipt["active_phase"] = "loading"; persist()
        load_start = time.perf_counter()
        model = wrapper_module.TAPNextPP.from_checkpoint(
            checkpoint, device="cpu", half_precision=False, compile_model=False,
            input_resolution=256)
        receipt["phases_seconds"]["model_construct_and_checkpoint_load"] = time.perf_counter() - load_start
        if model._model.training or any(p.device.type != "cpu" or p.dtype != torch.float32
                                       for p in model.parameters()):
            raise ValueError("Native CPU float32 eval mode was not established")
        receipt["parameter_count"] = sum(p.numel() for p in model.parameters())
        receipt["status"] = "inference"; receipt["active_phase"] = "inference"; persist()
        inference_start = time.perf_counter()
        captured = []
        def capture(module, inputs, result):
            # Read-only output tap: no return replacement, no extra model call.
            captured.append((tuple(x.detach().cpu().numpy().copy() for x in result[:3]), result[3]))
        hook = model._model.register_forward_hook(capture)
        state = None; first_query_hash = None
        def tensor_info(tensor, expected_shape):
            if tuple(tensor.shape) != expected_shape or tensor.device.type != "cpu" or tensor.dtype != torch.float32:
                raise ValueError("Unexpected native state schema")
            array = tensor.detach().contiguous().numpy()
            finite = bool(np.isfinite(array).all())
            if not finite:
                raise ValueError("Nonfinite native state")
            return {"shape": list(array.shape), "dtype": str(array.dtype), "device": "cpu",
                    "finite": finite, "bytes": array.nbytes,
                    "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest()}
        with torch.inference_mode():
            for t in range(8):
                frame_start = time.perf_counter(); captured.clear()
                frame_bgr = frames[t, :, :, ::-1].copy()  # official wrapper converts BGR back to RGB
                receipt["forwards_attempted"] += 1; persist()
                call_start = time.perf_counter()
                positions, visible, state = model.track_frame(
                    frame_bgr, query_points_xy=queries_xy if t == 0 else None,
                    state=state, autocast=False)
                forward_seconds = time.perf_counter() - call_start
                if len(captured) != 1 or captured[0][1] is not state:
                    raise ValueError("Expected exactly one native forward/state")
                (tracks, track_logits, vis_logits), _ = captured.pop()
                arrays = {"tracks_model_yx": tracks, "coordinate_logits": track_logits,
                          "visibility_logits": vis_logits, "positions_xy": positions,
                          "visible": visible}
                expected_shapes = [(1, 1, 8, 2), (1, 1, 8, 512), (1, 1, 8, 1), (8, 2), (8,)]
                for (name, value), shape in zip(arrays.items(), expected_shapes):
                    if value.shape != shape or value.dtype != (np.bool_ if name == "visible" else np.float32) or not np.isfinite(value).all():
                        raise ValueError("Unexpected native output schema/finite: " + name)
                if not np.array_equal(positions, tracks[0, 0, :, ::-1]) or not np.array_equal(visible, vis_logits[0, 0, :, 0] > 0):
                    raise ValueError("Official wrapper/raw output disagreement")
                if state.step != t + 1 or len(state.hidden_state) != 12:
                    raise ValueError("Unexpected native state step/depth")
                q_info = tensor_info(state.query_points, (1, 8, 3))
                if first_query_hash is None:
                    first_query_hash = q_info["sha256"]
                if q_info["sha256"] != first_query_hash:
                    raise ValueError("Original query state changed")
                layers = []
                for layer in state.hidden_state:
                    layers.append({"rg_lru_state": tensor_info(layer.rg_lru_state, (1032, 768)),
                                   "conv1d_state": tensor_info(layer.conv1d_state, (1032, 3, 768))})
                frame_path = output / ("frame-%03d.npz" % t)
                with frame_path.open("xb") as handle:
                    np.savez(handle, **arrays)
                receipt["frames"].append({"frame": t, "track_frame_seconds": forward_seconds,
                    "whole_frame_with_validation_save_seconds": time.perf_counter() - frame_start,
                    "state": {"step": state.step, "query_points": q_info, "hidden_state": layers},
                    "output_file": frame_path.name, "output_sha256": sha(frame_path)})
                receipt["frames_completed"] = t + 1; persist()
        hook.remove()
        receipt["phases_seconds"]["inference_loop_with_validation_and_artifacts"] = time.perf_counter() - inference_start
        receipt["status"] = "completed_native_capability"
        receipt["scientific_boundary"] = "No tracking accuracy, intervention benefit, or 250-frame feasibility measured"
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_alarm_handler)
        receipt["finished_utc"] = utc(); persist()
    return receipt
