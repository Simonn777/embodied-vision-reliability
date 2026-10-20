"""Isolated saved-checkpoint pointer-position arithmetic; no model construction."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

P = Path(__file__).resolve().parents[3]
R = P / "work/embodied-vision-reliability"
BASE = P / "work/0029-runtime-02"
OUT = Path(__file__).resolve().parent
COMMIT = "37b54982bcc3b22b56a5817b6881f22489eba8d5"
FIXTURE = "experiments/0029-memory-information/native_fixture.py"
HELPER = "experiments/0022-frontend/qualify.py"
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def function(path, name, namespace):
    # Execute only the exact original function AST, not the containing model module.
    tree = ast.parse(path.read_bytes(), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    require(len(nodes) == 1, "ambiguous original function")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def main(record):
    install = BASE / "installation"
    irpath = install / "install-receipt.json"
    qrpath = BASE / "qualification/receipt.json"
    fpath = BASE / "qualification/native-fixture/fixture-record.json"
    inst, qual, saved = [json.loads(x.read_text()) for x in (irpath, qrpath, fpath)]
    require(sha(irpath) == "bad95faf8079618e05007e1369d89b76b493bdb5d6847de560ea524daca835e4", "install receipt")
    require(sha(qrpath) == "f1b3024fbbd7cc0a7a94d2fecd4400dacc32c8edf51fafd0d7dc4dc8d08a7570", "qualification receipt")
    require(saved == qual["native_fixture"], "fixture receipt copy")
    require(inst["status"] == qual["status"] == saved["status"] == "completed", "completed inputs")
    paths = [install / rel for rel in (
        "weights/sam2.1_hiera_tiny.pt",
        "sources/sam2/sam2/modeling/sam2_utils.py",
        "sources/sam2/sam2/modeling/sam2_base.py",
        "sources/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml")]
    bindings = {str(x.relative_to(P)): sha(x) for x in (irpath, qrpath, fpath, *paths)}
    for path in paths:
        row = inst["installed_inventory"][str(path.relative_to(install))]
        require(row["type"] == "file" and row["sha256"] == sha(path)
                and row["bytes"] == path.stat().st_size, "inventory input")
    for rel in (FIXTURE, HELPER):
        frozen = subprocess.run(["git", "-C", str(R), "show", COMMIT + ":" + rel],
                                capture_output=True, check=True).stdout
        expected = qual["source_sha256"][rel]
        require(hashlib.sha256(frozen).hexdigest() == sha(R / rel) == expected, "frozen helper")
        bindings[str((R / rel).relative_to(P))] = expected
    require("proj_tpos_enc_in_obj_ptrs: true" in paths[3].read_text(), "projection enabled")
    record["input_sha256"] = bindings
    require(sys.flags.no_site == 1 and all(os.environ.get(k) == "4" for k in THREAD_KEYS), "launch contract")
    spec = importlib.util.spec_from_file_location("original_0022_guard", R / HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.install_guard(record["offline_attempts"], record["compatibility_probes"])
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      PYTHONDONTWRITEBYTECODE="1")
    sys.dont_write_bytecode = True
    runtime = install / "venv/lib/python3.12/site-packages"
    sys.path.insert(0, str(runtime))
    with helper.capture_warnings(record):
        import torch
    require(Path(torch.__file__).resolve().is_relative_to(runtime.resolve()), "Torch origin")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    require(torch.get_num_threads() == 4 and torch.get_num_interop_threads() == 1, "Torch threads")
    record["runtime"] = {"torch_version": torch.__version__, "intra_threads": 4, "inter_threads": 1,
                         "device": "cpu", "dtype": "torch.float32"}
    tensor_hash = function(R / FIXTURE, "tensor_hash", {"hashlib": hashlib})
    sine = function(paths[1], "get_1d_sine_pe", {"torch": torch})
    with helper.capture_warnings(record):
        checkpoint = torch.load(paths[0], map_location="cpu", weights_only=True)
    keys = ["obj_ptr_tpos_proj.weight", "obj_ptr_tpos_proj.bias"]
    weight, bias = [checkpoint["model"][key].detach().clone() for key in keys]
    del checkpoint
    require(weight.shape == (64, 256) and bias.shape == (64,)
            and weight.dtype == bias.dtype == torch.float32
            and torch.isfinite(weight).all().item() and torch.isfinite(bias).all().item(), "projection tensor contract")
    record["checkpoint_tensor_keys"] = keys
    record["projection_tensors"] = {
        key: {"shape": list(t.shape), "dtype": str(t.dtype), "tensor_sha256": tensor_hash(t, torch)}
        for key, t in zip(keys, (weight, bias))}
    def saved_hash(trace):
        blocks = [b for b in trace["blocks"] if b["kind"] == "pointer" and b["frame"] == 0]
        require(len(blocks) == 1, "saved pointer0")
        return blocks[0]["position_sha256"]
    expected = {"full": saved_hash(saved["probes"][0]["trace"]),
                "initial_only": saved_hash(saved["initial_memory_prompt_trace"])}
    results = {}
    with torch.inference_mode():
        for name, distances in (("full", [7, 1, 2, 3, 4, 5, 6]), ("initial_only", [7])):
            x = torch.tensor(distances, device="cpu") / 8
            encoded = sine(x, dim=256)
            projected = torch.nn.functional.linear(encoded, weight, bias)
            first = projected.unsqueeze(1).expand(-1, 1, 64)[0:1].expand(4, -1, -1)
            results[name] = (encoded[0].clone(), first.clone())
            observed = tensor_hash(first, torch)
            record["branches"][name] = {
                "integer_distances": distances, "denominator": 8, "normalized_input": x.tolist(),
                "encoded_shape": list(encoded.shape), "encoded_first": encoded[0].tolist(),
                "encoded_first_tensor_sha256": tensor_hash(encoded[0], torch),
                "projected_shape": list(projected.shape), "expanded_first": first.tolist(),
                "expected_position_sha256": expected[name], "position_sha256": observed,
                "exact_saved_hash_match": observed == expected[name]}
        encoding_difference = results["full"][0] - results["initial_only"][0]
        difference = results["full"][1] - results["initial_only"][1]
        record["comparison"] = {
            "encoded_first_bitwise_equal": torch.equal(results["full"][0], results["initial_only"][0]),
            "encoded_first_max_abs_difference": float(encoding_difference.abs().max()),
            "position_bitwise_equal": torch.equal(results["full"][1], results["initial_only"][1]),
            "position_max_abs_difference": float(difference.abs().max()),
            "position_different_values": int(torch.count_nonzero(difference)),
            "both_saved_hashes_reproduced": all(v["exact_saved_hash_match"] for v in record["branches"].values())}
    require(record["comparison"]["both_saved_hashes_reproduced"], "saved position hashes not reproduced")
    require(not record["offline_attempts"], "forbidden operation")
    require(all(sha(P / rel) == value for rel, value in bindings.items()), "input source changed")
    require("sam2" not in sys.modules and "lightglue" not in sys.modules, "no model module import")
    record.update(status="completed", inputs_unchanged=True)


if __name__ == "__main__":
    started = time.perf_counter()
    record = {"schema": "evr0029.pointer-arithmetic-diagnostic.v1", "status": "started",
              "source_sha256": sha(__file__), "source_commit": COMMIT,
              "scope": "original sine function and F.linear on two saved checkpoint tensors; no model construction, inference, image or benchmark",
              "offline_attempts": [], "compatibility_probes": [], "warnings": [], "branches": {}}
    try:
        main(record)
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record["wall_seconds"] = time.perf_counter() - started
        with (OUT / "diagnostic.json").open("x") as f:
            json.dump(record, f, indent=2, allow_nan=False)
            f.write("\n")
        print(json.dumps({k: record[k] for k in ("status", "wall_seconds")}), flush=True)
