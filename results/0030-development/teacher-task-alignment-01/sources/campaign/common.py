"""Small development runtime: immutable native admission, offline guard and receipts."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_runtime(spec, record):
    root, native = Path(spec["repo"]), Path(spec["native_root"])
    own = root / "experiments/0029-memory-information"
    for filename, expected in spec["source_sha256"].items():
        require(sha(filename) == expected, "changed development source: " + filename)
    h = module("campaign_install", own / "install_runtime.py")
    old = module("campaign_admission", own / "run_evidence.py")
    q = module("campaign_guard", root / "experiments/0022-frontend/qualify.py")
    nf = module("campaign_native", own / "native_fixture.py")
    installation, _, binding = old.native_contract(h, root, native, spec["native_commit"])
    record["native_binding"] = binding
    require(sys.flags.no_site == 1 and sys.version_info[:3] == (3, 12, 13)
            and Path(sys.executable).absolute() == installation / "venv/bin/python", "qualified interpreter")
    require(all(os.environ.get(k) == "4" for k in h.THREAD_KEYS), "thread envelope")
    q.install_guard(record.setdefault("offline_attempts", []), record.setdefault("compatibility_probes", []))
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      XFORMERS_DISABLED="1", PYTHONDONTWRITEBYTECODE="1")
    import site
    site.main()
    sys.path.insert(0, str(installation / "sources/sam2"))
    import numpy as np
    import torch
    import cv2
    from PIL import Image
    from sam2.build_sam import build_sam2_video_predictor
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(spec.get("seed", 0))
    cv2.setNumThreads(0)
    model = build_sam2_video_predictor(nf.CONFIG, ckpt_path=None, device="cpu", mode="eval",
        apply_postprocessing=False, hydra_overrides_extra=list(nf.OVERRIDES), vos_optimized=False)
    weights = torch.load(installation / "weights/sam2.1_hiera_tiny.pt", map_location="cpu", weights_only=True)
    record["strict_load"] = q.strict_state(model, weights["model"], torch)
    del weights
    model.eval().to(device="cpu", dtype=torch.float32).requires_grad_(False)
    require(all(p.device.type == "cpu" and p.dtype == torch.float32 and not p.requires_grad
                for p in model.parameters()), "frozen CPU model")
    record["model_initial_hash"] = nf.state_hash(model.state_dict(), torch)
    return dict(model=model, torch=torch, np=np, cv2=cv2, Image=Image, nf=nf, guard=q)


def finish(runtime, record):
    model, torch = runtime["model"], runtime["torch"]
    record["model_final_hash"] = runtime["nf"].state_hash(model.state_dict(), torch)
    require(record["model_final_hash"] == record["model_initial_hash"], "native weights changed")
    require(not record["offline_attempts"], "network access attempted")
    require(all(p.grad is None for p in model.parameters()), "native gradient accumulated")


def ordinary(value, torch):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: ordinary(v, torch) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(ordinary(v, torch) for v in value)
    return value


def start(spec_path, kind):
    spec = json.loads(Path(spec_path).read_text())
    out = Path(spec["output"])
    require(not out.exists(), "fresh operation output required")
    out.mkdir(parents=True)
    record = dict(schema="evr0030.development." + kind + ".v1", status="started",
                  spec_sha256=sha(spec_path), source_sha256=spec["source_sha256"],
                  seed=spec.get("seed", 0), rows=[], warnings=[])
    began = time.perf_counter()
    save(out / "receipt.json", record)
    return spec, out, record, began
