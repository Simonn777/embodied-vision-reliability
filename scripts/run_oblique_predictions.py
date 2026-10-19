"""One cache-only four-view prediction attempt; never reads evaluation labels.

The caller supplies a prepared, label-free NPZ and a fresh output directory.
An external reviewed supervisor MUST enforce the 900 s wall limit, <=4 CPUs,
8 GiB address space. Reviewed scientific code makes no network requests;
there is no claim of OS network-namespace isolation. There is no warmup, retry, resume, fusion,
threshold fitting, asset acquisition or package installation in this module.
Importing this module imports no numerical/ML packages and executes no model.

Backend contract: forward(rgb_uint8, initial_query_tyx_or_none, state) returns
(arrays, next_state), where arrays are the three float32 output arrays below.
The initial query has shape (1,8,3) and order [t,y,x], with t=0; state.step is
the number of observed frames, not the original frame number. Native execution
uses the unmodified official frame wrapper and a read-only raw-output hook.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import importlib.util
from importlib.machinery import PathFinder
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
import zipfile


SOURCE_REV = "730cda1c730877cfedbe01bf87fb1cadb78a565d"
HF_REV = "5d3fb48e76c5422841e38501514121e251beabb7"
WEIGHT_BYTES = 2532282370
WEIGHT_SHA = "cb96a43444ccb4fbdb25d800b88c7ba196179a526e78f01b021a16b1c1eff6da"
SOURCES = {
    "tapnet.tapnext.tapnext_torch": "48072289b22439267b3a51eb4cde5e2f81f38c840b6c95b659d60032a07c691a",
    "tapnet.tapnext.tapnext_lru_modules": "2a475419c2f7c67db9acdf412108fb3b025d168f60e9e087fe35d65157adf64f",
    "tapnet.tapnext.pscan": "7df34bf5e8318e81bb330ded14fc5e675e7485ea39fa9f0d77fa470f85796a2d",
    "tapnet.tapnextpp.votsp2026.model": "e1219e2341c33e0f9124fe02d09a521ac7c91bc27505fcbebbf05741f825e46b",
    "tapnet.tapnextpp.votsp2026.utils": "0336c526dc2b91244c3e5b803f84b65c6323ede138340a4107441a6b9c9fdbf8",
}
PACKAGES = {"torch": "2.10.0+cpu", "numpy": "2.0.2",
            "torchvision": "0.25.0+cpu", "einops": "0.8.2", "Pillow": "13.0.0"}
RUNTIME_INIT_SHA = {
    "torch": "51c90fe34a7cf869517d1bab4cd114498790e169ec668f1a154c35ec64117a5e",
    "numpy": "c09e25b58f6b2f8e2cb3c158168f902d447f8171e5ea6513c0aca41ecbda7c2b",
}
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
INPUT_SCHEMA = {"video": ((16, 256, 256, 3), "uint8"),
                "query_xy": ((8, 2), "float32"),
                "point_ids": ((8,), "int64"), "frame_ids": ((16,), "int64")}
OUTPUT_SCHEMA = {"coordinate_logits": (8, 512), "positions_xy": (8, 2),
                 "visibility_logits": (8,)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("Expected a lowercase SHA-256 digest")
    return value


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _save_json(path, payload, *, exclusive=False):
    if exclusive:
        with path.open("x") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
    else:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("x") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)


def _geometry():
    path = Path(__file__).with_name("oblique_view_geometry.py")
    spec = importlib.util.spec_from_file_location("_oblique_prediction_geometry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_prepared_input(path, expected_sha256):
    """Accept exactly four bounded, stored numeric members, with no label key.

    Schema headers are checked before payload reads. NPZ identity is checked
    again after loading, so replacing a file during this read fails closed.
    """
    import numpy as np

    path = Path(path)
    _digest(expected_sha256)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024**2:
        raise ValueError("Expected a regular prepared NPZ smaller than 4 MiB")
    if sha(path) != expected_sha256:
        raise ValueError("Prepared input SHA-256 differs")
    with zipfile.ZipFile(path) as archive:
        expected_names = sorted(name + ".npy" for name in INPUT_SCHEMA)
        if sorted(archive.namelist()) != expected_names:
            raise ValueError("Prepared input must have exactly the four label-free keys")
        for name, (shape, dtype_name) in INPUT_SCHEMA.items():
            info = archive.getinfo(name + ".npy")
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError("Prepared NPZ must use ZIP_STORED")
            with archive.open(info) as member:
                version = np.lib.format.read_magic(member)
                if version == (1, 0):
                    got_shape, fortran, dtype = np.lib.format.read_array_header_1_0(member)
                elif version == (2, 0):
                    got_shape, fortran, dtype = np.lib.format.read_array_header_2_0(member)
                else:
                    raise ValueError("Unexpected NPY version")
                if got_shape != shape or fortran or dtype != np.dtype(dtype_name):
                    raise ValueError("Prepared input schema differs: " + name)
                if member.tell() + int(np.prod(shape)) * dtype.itemsize != info.file_size:
                    raise ValueError("Prepared member byte count differs: " + name)
    with np.load(path, allow_pickle=False) as loaded:
        values = {name: loaded[name].copy() for name in INPUT_SCHEMA}
    if sha(path) != expected_sha256:
        raise ValueError("Prepared input changed during loading")
    if not np.array_equal(values["frame_ids"], np.arange(0, 256, 16, dtype=np.int64)):
        raise ValueError("Only the frozen 16 observed frame IDs are permitted")
    queries = values["query_xy"]
    if not np.isfinite(queries).all() or np.any((queries < 0) | (queries > 256)):
        raise ValueError("Initial queries must be finite original-canvas coordinates")
    ids = values["point_ids"]
    if len(np.unique(ids)) != 8 or np.any(ids < 0):
        raise ValueError("Eight distinct nonnegative point IDs are required")
    # A backend only receives a newly rendered current frame and a fresh query
    # array. It cannot obtain future frames or the prepared archive from here.
    for value in values.values():
        value.setflags(write=False)
    return values


def _validate_outputs(arrays):
    import numpy as np

    if not isinstance(arrays, dict) or set(arrays) != set(OUTPUT_SCHEMA):
        raise ValueError("Unexpected prediction array names")
    for name, shape in OUTPUT_SCHEMA.items():
        value = arrays[name]
        if (not isinstance(value, np.ndarray) or value.shape != shape
                or value.dtype != np.float32 or not np.isfinite(value).all()):
            raise ValueError("Unexpected prediction shape, dtype or finite values: " + name)


def run_predictions(*, prepared_input, input_sha256, group_id, output,
                    protocol_sha256, backend_factory):
    """Execute exactly 64 calls on success; any failure is terminal for this path.

    backend_factory is injectable only for software fixture tests. The CLI
    always constructs NativeBackend with the fixed upstream identities.
    """
    output = Path(output)
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    receipt = {
        "schema_version": 1, "status": "started", "started_utc": _utc(),
        "group_id": group_id, "input_sha256": input_sha256,
        "protocol_sha256": protocol_sha256, "runner_sha256": sha(__file__),
        "geometry_sha256": sha(Path(__file__).with_name("oblique_view_geometry.py")),
        "python": sys.version, "platform": platform.platform(), "command": sys.argv,
        "forwards_attempted": 0, "completed_predictions": 0, "frames": [],
        "acquisition": "none; hash-verified existing cache only",
        "scope": "predictions only; no evaluation labels, fusion, scoring or model-state edits",
        "observation_policy": "16 observed frames at original IDs 0,16,...,240; state steps 0..15",
    }
    _save_json(output / "started.json", {
        "started_utc": receipt["started_utc"], "pid": os.getpid(),
        "group_id": group_id, "input_sha256": input_sha256,
        "protocol_sha256": protocol_sha256, "runner_sha256": receipt["runner_sha256"],
    }, exclusive=True)

    def persist():
        receipt["wall_seconds"] = time.perf_counter() - started
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt["peak_rss_bytes"] = int(rss if sys.platform == "darwin" else rss * 1024)
        _save_json(output / "receipt.json", receipt)

    backend = None
    try:
        persist()
        if group_id not in ("g-036", "g-002"):
            raise ValueError("Group is outside the two prospective development groups")
        _digest(protocol_sha256)
        inputs = read_prepared_input(prepared_input, input_sha256)
        import numpy as np
        geometry = _geometry()
        receipt["point_ids"] = inputs["point_ids"].tolist()
        receipt["frame_ids"] = inputs["frame_ids"].tolist()
        receipt["query_xy_original"] = inputs["query_xy"].tolist()
        receipt["view_order"] = [item[0] for item in geometry.FIXED_VIEW_SPECS]
        receipt["views"] = {}
        receipt["status"] = "loading_backend"
        persist()
        backend = backend_factory()
        receipt["backend"] = backend.metadata
        receipt["status"] = "collecting_predictions"
        persist()
        for view, transform in geometry.fixed_view_affines().items():
            query_xy = geometry.map_points(inputs["query_xy"], transform)
            initial_query = np.zeros((1, 8, 3), dtype=np.float32)
            initial_query[0, :, 1:] = query_xy[:, ::-1]
            receipt["views"][view] = {"original_to_view": transform.tolist(),
                                      "query_xy": query_xy.tolist()}
            state = None  # independent causal trajectory; model weights are shared
            for observation_index, frame_id in enumerate(inputs["frame_ids"]):
                frame_started = time.perf_counter()
                frame = geometry.warp_rgb(inputs["video"][observation_index], transform)
                receipt["active_call"] = {"view": view, "frame_index": int(frame_id),
                                          "observation_index": observation_index}
                receipt["forwards_attempted"] += 1
                persist()  # durable attempted count before the only model call
                forward_started = time.perf_counter()
                arrays, state = backend.forward(
                    frame, initial_query.copy() if observation_index == 0 else None, state)
                forward_seconds = time.perf_counter() - forward_started
                _validate_outputs(arrays)
                if getattr(state, "step", None) != observation_index + 1:
                    raise ValueError("Causal state step must count observed frames only")
                name = f"view-{view}-frame-{observation_index:03d}.npz"
                path = output / name
                with path.open("xb") as stream:
                    np.savez(stream, **arrays)
                receipt["frames"].append({
                    "view": view, "frame_index": int(frame_id),
                    "observation_index": observation_index, "state_step": int(state.step),
                    "output_file": name, "output_sha256": sha(path),
                    "forward_seconds": forward_seconds,
                    "frame_seconds": time.perf_counter() - frame_started,
                })
                receipt["completed_predictions"] += 1
                persist()
            state = None
        if receipt["forwards_attempted"] != 64 or receipt["completed_predictions"] != 64:
            raise ValueError("Expected exactly four independent 16-frame trajectories")
        # close() only removes the read-only hook; it cannot run another forward.
        backend.close()
        backend = None
        receipt.pop("active_call", None)
        receipt["status"] = "completed_predictions"
        receipt["finished_utc"] = _utc()
        persist()
        _save_json(output / "completed.json", {
            "receipt_sha256": sha(output / "receipt.json"), "completed_predictions": 64,
            "finished_utc": receipt["finished_utc"],
        }, exclusive=True)
        return receipt
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error),
                       traceback=traceback.format_exc(), finished_utc=_utc())
        persist()
        raise
    finally:
        if backend is not None:
            backend.close()


def verify_source_tree(source_root):
    """Verify only the five frozen namespace files before executing TAPNet code."""
    root = Path(source_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Expected an independent regular source directory")
    files = {name.replace(".", "/") + ".py": digest for name, digest in SOURCES.items()}
    directories = {str(p) for name in files for p in Path(name).parents if str(p) != "."}
    observed = set()
    for path in root.rglob("*"):
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError("Source symlink is not a frozen input")
        if path.is_dir() and name in directories:
            continue
        if not path.is_file() or name not in files or sha(path) != files[name]:
            raise ValueError("Unexpected or modified source entry: " + name)
        observed.add(name)
    if observed != set(files):
        raise ValueError("Pinned source file missing")
    if any(name == "tapnet" or name.startswith("tapnet.") for name in sys.modules):
        raise ValueError("A fresh process without prior TAPNet imports is required")
    spec = PathFinder.find_spec("tapnet", sys.path)
    if (spec is None or spec.loader is not None or spec.origin is not None
            or [Path(p).resolve() for p in spec.submodule_search_locations or []]
            != [(root / "tapnet").resolve()]):
        raise ValueError("Only the frozen TAPNet namespace may resolve")


def verify_asset_receipt(path, expected_sha256):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024**2:
        raise ValueError("Expected a regular bounded prior asset receipt")
    if sha(path) != _digest(expected_sha256):
        raise ValueError("Prior asset receipt identity differs")
    prior = json.loads(path.read_text())
    if (prior.get("status") != "completed_native_capability"
            or prior.get("source_sha256") != SOURCES
            or prior.get("checkpoint_sha256") != WEIGHT_SHA
            or prior.get("source_revision") != SOURCE_REV
            or prior.get("checkpoint_revision") != HF_REV
            or prior.get("packages") != PACKAGES):
        raise ValueError("Prior native receipt does not establish these frozen asset identities")


class NativeBackend:
    """Official CPU wrapper with a read-only tap, no weight or state replacement."""

    def __init__(self, *, source_root, checkpoint, overlay_root, base_runtime_root,
                 asset_receipt, asset_receipt_sha256):
        if sys.platform != "linux" or platform.python_version() != "3.12.14":
            raise ValueError("Frozen native target is Linux Python 3.12.14")
        if any(os.environ.get(name) != "4" for name in THREAD_VARS):
            raise ValueError("Supervisor must set all five thread variables to 4 before imports")
        if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or not sys.dont_write_bytecode:
            raise ValueError("Supervisor must disable bytecode writes before imports")
        affinity = sorted(os.sched_getaffinity(0))
        limits = resource.getrlimit(resource.RLIMIT_AS)
        if not 1 <= len(affinity) <= 4 or any(v < 0 or v > 8 * 1024**3 for v in limits):
            raise ValueError("Supervisor must enforce <=4 CPUs and <=8 GiB address space")
        if "torch" in sys.modules:
            raise ValueError("Use a fresh process without a prior Torch import")
        root = Path(source_root)
        checkpoint = Path(checkpoint)
        overlay = Path(overlay_root).resolve()
        base = Path(base_runtime_root).resolve()
        if (not overlay.is_dir() or not base.is_dir() or overlay == base
                or overlay.is_relative_to(base)
                or any((overlay / name).exists() for name in ("torch", "numpy"))):
            raise ValueError("Require separate existing overlay and original Torch/NumPy runtime")
        verify_asset_receipt(asset_receipt, asset_receipt_sha256)
        if (checkpoint.is_symlink() or not checkpoint.is_file()
                or checkpoint.stat().st_size != WEIGHT_BYTES or sha(checkpoint) != WEIGHT_SHA):
            raise ValueError("Official 256px checkpoint identity differs")
        for name, digest in RUNTIME_INIT_SHA.items():
            if sha(base / name / "__init__.py") != digest:
                raise ValueError("Base runtime initializer identity differs: " + name)
        sys.path.insert(0, str(root))
        verify_source_tree(root)
        actual = {name: importlib.metadata.version(name) for name in PACKAGES}
        if actual != PACKAGES:
            raise ValueError("Frozen package versions absent or changed")
        self.np = importlib.import_module("numpy")
        self.torch = importlib.import_module("torch")
        modules = {name: importlib.import_module("PIL" if name == "Pillow" else name)
                   for name in PACKAGES}
        for name, module in modules.items():
            expected_root = base if name in ("torch", "numpy") else overlay
            if (str(module.__version__) != PACKAGES[name]
                    or not Path(module.__file__).resolve().is_relative_to(expected_root)):
                raise ValueError("Imported package identity differs: " + name)
        torch = self.torch
        if torch.version.cuda is not None or torch.get_default_dtype() != torch.float32:
            raise ValueError("CPU float32 runtime required")
        torch.set_num_threads(4)
        torch.set_num_interop_threads(4)
        torch.manual_seed(20261018)
        self.np.random.seed(20261018)
        torch.use_deterministic_algorithms(True)
        verify_source_tree(root)
        wrapper = importlib.import_module("tapnet.tapnextpp.votsp2026.model")
        for name, digest in SOURCES.items():
            loaded = importlib.import_module(name)
            expected = (root / (name.replace(".", "/") + ".py")).resolve()
            if Path(loaded.__file__).resolve() != expected or sha(expected) != digest:
                raise ValueError("Unexpected actual imported source: " + name)
        self.model = wrapper.TAPNextPP.from_checkpoint(
            checkpoint, device="cpu", half_precision=False,
            compile_model=False, input_resolution=256)
        if self.model._model.training or any(
                p.device.type != "cpu" or p.dtype != torch.float32 for p in self.model.parameters()):
            raise ValueError("Native CPU float32 eval mode was not established")
        self._captured = []

        def capture(module, inputs, result):
            self._captured.append((
                tuple(value.detach().cpu().numpy().copy() for value in result[:3]), result[3]))

        self._hook = self.model._model.register_forward_hook(capture)
        self._initial_query = None
        self.metadata = {
            "implementation": "official track_frame plus read-only raw forward hook",
            "source_revision": SOURCE_REV, "checkpoint_revision": HF_REV,
            "source_sha256": SOURCES.copy(), "checkpoint_sha256": WEIGHT_SHA,
            "checkpoint_bytes": WEIGHT_BYTES, "asset_receipt_sha256": asset_receipt_sha256,
            "packages": actual, "cpu_affinity": affinity, "RLIMIT_AS_bytes": list(limits),
            "threads": [torch.get_num_threads(), torch.get_num_interop_threads()],
            "imports": {name: {"path": str(module.__file__), "sha256": sha(module.__file__)}
                        for name, module in modules.items()},
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "seed": 20261018, "autocast": False, "compile": False,
        }

    def forward(self, rgb, query_tyx, state):
        np, torch = self.np, self.torch
        if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
            raise ValueError("Native wrapper input must be uint8 RGB")
        if state is None:
            if (query_tyx is None or query_tyx.shape != (1, 8, 3)
                    or query_tyx.dtype != np.float32 or not np.isfinite(query_tyx).all()
                    or not np.array_equal(query_tyx[0, :, 0], np.zeros(8, dtype=np.float32))):
                raise ValueError("First observation needs eight [0,y,x] queries")
            self._initial_query = query_tyx.copy()
            query_xy = query_tyx[0][:, [2, 1]].copy()
            expected_step = 1
        else:
            if query_tyx is not None:
                raise ValueError("Later observations must reuse the native causal query state")
            query_xy = None
            expected_step = state.step + 1
        self._captured.clear()
        with torch.inference_mode():
            positions, visible, next_state = self.model.track_frame(
                rgb[:, :, ::-1].copy(), query_points_xy=query_xy,
                state=state, autocast=False)
        if len(self._captured) != 1 or self._captured[0][1] is not next_state:
            raise ValueError("Expected exactly one native core forward and its unchanged state")
        (tracks, logits, visibility), _ = self._captured.pop()
        for name, array, shape in (
                ("tracks", tracks, (1, 1, 8, 2)), ("logits", logits, (1, 1, 8, 512)),
                ("visibility", visibility, (1, 1, 8, 1)), ("positions", positions, (8, 2))):
            if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError("Invalid native output: " + name)
        if (visible.shape != (8,) or visible.dtype != np.bool_
                or not np.array_equal(positions, tracks[0, 0, :, ::-1])
                or not np.array_equal(visible, visibility[0, 0, :, 0] > 0)):
            raise ValueError("Official wrapper and raw output disagree")
        if next_state.step != expected_step or len(next_state.hidden_state) != 12:
            raise ValueError("Unexpected native state step or depth")

        def state_array(tensor, shape):
            if tuple(tensor.shape) != shape or tensor.device.type != "cpu" or tensor.dtype != torch.float32:
                raise ValueError("Native state schema differs")
            array = tensor.detach().numpy()
            if not np.isfinite(array).all():
                raise ValueError("Nonfinite native state")
            return array

        if not np.array_equal(state_array(next_state.query_points, (1, 8, 3)), self._initial_query):
            raise ValueError("Initial native [t,y,x] query changed")
        for layer in next_state.hidden_state:
            state_array(layer.rg_lru_state, (1032, 768))
            state_array(layer.conv1d_state, (1032, 3, 768))
        return {"coordinate_logits": logits[0, 0].copy(), "positions_xy": positions.copy(),
                "visibility_logits": visibility[0, 0, :, 0].copy()}, next_state

    def close(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        self._captured.clear()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared-input", "input-sha256", "group-id", "output", "protocol-sha256",
                 "source-root", "checkpoint", "overlay-root", "base-runtime-root",
                 "asset-receipt", "asset-receipt-sha256"):
        parser.add_argument("--" + name, required=True)
    args = vars(parser.parse_args(argv))
    # These are existing dependencies only; no installation/import fallback.
    # This precedes the first NumPy import used to validate prepared inputs.
    for name, digest in RUNTIME_INIT_SHA.items():
        if sha(Path(args["base_runtime_root"]) / name / "__init__.py") != digest:
            raise ValueError("Base runtime initializer differs before numerical imports: " + name)
    sys.path.insert(0, str(Path(args["base_runtime_root"]).resolve()))
    sys.path.insert(0, str(Path(args["overlay_root"]).resolve()))
    backend_keys = ("source_root", "checkpoint", "overlay_root", "base_runtime_root",
                    "asset_receipt", "asset_receipt_sha256")
    backend_options = {name: args.pop(name) for name in backend_keys}
    run_predictions(**args, backend_factory=lambda: NativeBackend(**backend_options))


if __name__ == "__main__":
    main()
