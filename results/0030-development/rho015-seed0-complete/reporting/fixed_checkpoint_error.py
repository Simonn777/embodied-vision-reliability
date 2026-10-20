"""Post-hoc TRAIN-only target fit; no SAM2, future masks, fitting, or DEV access."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

ARMS = ("multistep", "one_step", "gt_memory")
PARENTS = ("dog-agility", "planes-water", "dogs-scale", "dog-gooses")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check(descriptor):
    p = Path(descriptor["path"])
    require(p.is_file() and p.stat().st_size == descriptor["bytes"] and
            sha(p) == descriptor["sha256"], "artifact changed: " + str(p))
    return p


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def summary(rows):
    result = {k: math.fsum(r[k] for r in rows) / len(rows)
              for k in ("E", "E0", "prediction_energy")}
    result["E_over_E0"] = result["E"] / result["E0"] if result["E0"] else None
    cosine = [r["gram_cosine"] for r in rows if r["gram_cosine"] is not None]
    result.update(count=len(rows), direction_defined_count=len(cosine),
                  mean_defined_gram_cosine=math.fsum(cosine) / len(cosine) if cosine else None)
    return result


def run(spec_path):
    began = time.perf_counter()
    spec = json.loads(Path(spec_path).read_text())
    out = Path(spec["output"]); out.mkdir(exist_ok=False)
    record = dict(schema="evr0030.fixed-checkpoint-fit.v1", status="started", arms=[],
                  spec_sha256=sha(spec_path), model_calls=0, sam2_calls=0, head_forward_calls=0,
                  fitting_updates=0, train_only=True, dev_access=False, final_test_access=False,
                  source_sha256=spec["source_sha256"])
    try:
        require(sys.version_info[:3] == (3, 12, 13), "qualified interpreter")
        root = Path(spec["family_root"])
        require(set(spec["source_sha256"]) == {str(Path(__file__).resolve()),
                str(root / "family.py"), str(root / "head.py")}, "exact analysis/family/head sources")
        for path, digest in spec["source_sha256"].items():
            require(sha(path) == digest, "source changed")
        target_path = check(spec["targets"]); teacher_path = check(spec["teacher_receipt"])
        teacher = json.loads(teacher_path.read_text())
        require(teacher["status"] == "completed" and len(teacher["rows"]) == 48 and
                teacher["targets"]["sha256"] == spec["targets"]["sha256"] and
                teacher_path.parent / teacher["targets"]["path"] == target_path, "complete teacher")
        require([x["arm"] for x in spec["checkpoints"]] == list(ARMS), "all three fixed arms")
        for entry in spec["checkpoints"]:
            checkpoint = check(entry["checkpoint"]); receipt = json.loads(check(entry["receipt"]).read_text())
            final = receipt["checkpoints"][-1]
            require(receipt["status"] == "completed" and receipt["arm"] == entry["arm"] and
                    receipt["rho"] == teacher["rho"] and final["step"] == 288 and
                    final["sha256"] == entry["checkpoint"]["sha256"] and
                    Path(entry["receipt"]["path"]).parent / final["path"] == checkpoint, "final regressor")
        import torch
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        family = load_module("fit_analysis_family", root / "family.py")
        heads = load_module("fit_analysis_head", root / "head.py")
        data = torch.load(target_path, map_location="cpu", weights_only=True)
        identities = [(x["sequence"], x["anchor"]) for x in data]
        require(len(data) == len(set(identities)) == 48 and
                all(sum(seq == parent for seq, _ in identities) == 12 for parent in PARENTS) and
                all(x["context"]["rho"] == teacher["rho"] for x in data), "fixed TRAIN targets")
        for entry in spec["checkpoints"]:
            head = heads.build_head(torch=torch).eval().requires_grad_(False)
            state = torch.load(entry["checkpoint"]["path"], map_location="cpu", weights_only=True)
            require(state["step"] == 288, "checkpoint update count")
            head.load_state_dict(state["head"], strict=True)
            rows = []
            with torch.no_grad():
                for item in data:
                    context = item["context"]; target = item[entry["arm"]]
                    raw = head(item["features"]).reshape(64, 2)
                    q = family.project(context, raw * (context["sigma"] * context["rho"]).float()[:, None],
                                       torch=torch)
                    record["head_forward_calls"] += 1
                    energy = lambda v: float(family.coefficient_energy(context, v, torch=torch))
                    error, zero, pred = energy(q-target), energy(target), energy(q)
                    radius2 = context["radius"] ** 2
                    require(radius2 > 0 or error == zero == pred == 0, "nonzero zero-radius target")
                    cross = float(torch.einsum("ci,cij,cj->", q.double(), context["gram"], target.double()))
                    cosine = cross / math.sqrt(pred * zero) if pred > 0 and zero > 0 else None
                    require(cosine is None or abs(cosine) <= 1 + 1e-12, "invalid Gram angle")
                    rows.append(dict(sequence=item["sequence"], anchor=item["anchor"],
                        E=error / radius2 if radius2 else 0., E0=zero / radius2 if radius2 else 0.,
                        prediction_energy=pred / radius2 if radius2 else 0.,
                        E_over_E0=error / zero if zero else None,
                        gram_cosine=max(-1., min(1., cosine)) if cosine is not None else None,
                        gram_cosine_unclipped=cosine))
            parents = {p: summary([r for r in rows if r["sequence"] == p]) for p in PARENTS}
            record["arms"].append(dict(arm=entry["arm"], checkpoint=entry["checkpoint"],
                                       rows=rows, parents=parents, equal_parent=summary(rows)))
            save(out / "receipt.json", record)
        for path, digest in spec["source_sha256"].items():
            require(sha(path) == digest, "source changed after analysis")
        for d in [spec["targets"], spec["teacher_receipt"]] + [
                x[k] for x in spec["checkpoints"] for k in ("checkpoint", "receipt")]:
            check(d)
        require(record["head_forward_calls"] == 144, "complete three-arm coverage")
        record.update(status="completed", rho=teacher["rho"], target_count=48,
                      interpretation="TRAIN fit only; ratios use all errors, direction means show defined count")
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc)); raise
    finally:
        record["wall_seconds"] = time.perf_counter() - began
        save(out / "receipt.json", record)


if __name__ == "__main__":
    run(sys.argv[1])
