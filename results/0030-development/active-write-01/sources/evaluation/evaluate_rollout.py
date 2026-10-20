"""Evaluate a complete causal DEV rollout; metric code is shared unchanged."""
import json
from pathlib import Path
import sys
import time
from evaluate_baseline import sha, require, save
import evaluate_jf as e


def admit(binding, prefix, fingerprints, legacy=False):
    read = lambda suffix: json.loads(binding[prefix + suffix].read_text())
    record, outer, operation, worker = (read(s) for s in ("_receipt", "_outer", "_operation", "_worker_spec"))
    expected = ("evr0026.operation.v1", "7c7798fe76756ace692fe9d9a83755d758690c62f0a6cc7f11af655dd5c60efe") if legacy else (
        "evr0029.exit-checked-operation.v1", "8f7bcbb8b7d31031c709f72ea2f49da1913ced7230154e45e0d165d00457c9a8")
    require((outer["schema"], outer["supervisor_sha256"]) == expected and
            outer["monitor_sha256"] == "4b8b0efa4fd713a032967c9b34539eac128184ef6db4e8535e83d785c6bbebcd" and
            outer["status"] == record["status"] == "completed" and outer["exit_code"] == 0 and
            outer["termination"] is None and not outer["remaining_descendant_pids"] and
            not outer["cleanup_errors"] and outer["wall_seconds"] <= 1800 and
            outer["peak_group_rss_kib"] <= 8388608, "unclean " + prefix)
    if not legacy:
        require(not outer["signal_attempts"] and outer["triggering_snapshot"] is None and
                not outer["post_exit_snapshot"]["rows"] and len(outer["cleanup_snapshots"]) == 1 and
                outer["cleanup_snapshots"][0]["label"] == "final" and
                not outer["cleanup_snapshots"][0]["rows"], "exit-check cleanup")
    require(outer["output_sha256"] == sha(binding[prefix + "_receipt"]) and
            outer["spec_sha256"] == sha(binding[prefix + "_operation"]) and
            record["spec_sha256"] == sha(binding[prefix + "_worker_spec"]) and
            Path(operation["result_file"]) == binding[prefix + "_receipt"] and
            Path(operation["command"][-1]) == binding[prefix + "_worker_spec"] and
            Path(worker["output"]) == binding[prefix + "_receipt"].parent and
            record["source_sha256"] == worker["source_sha256"], "operation binding")
    for name, digest in outer["log_sha256"].items():
        p = binding[prefix + "_outer"].parent / name
        require(sha(p) == digest, "log changed"); fingerprints[str(p)] = digest
    return record, worker


def run(spec_path):
    began = time.perf_counter(); spec = json.loads(Path(spec_path).read_text())
    out = Path(spec["output"]); out.mkdir(exist_ok=False)
    record = dict(schema="evr0030.rollout-evaluation.v1", status="started", model_calls=0,
                  final_test_access=False, spec_sha256=sha(spec_path), source_sha256=spec["source_sha256"])
    try:
        fingerprints = dict(spec["source_sha256"])
        for d in spec["bindings"].values():
            e._descriptor(d); fingerprints[d["path"]] = d["sha256"]
        for path, digest in fingerprints.items():
            require(sha(path) == digest, "source/binding changed")
        binding = {k: Path(v["path"]) for k, v in spec["bindings"].items()}
        cache, _ = admit(binding, "cache", fingerprints, legacy=True)
        rollout, worker = admit(binding, "rollout", fingerprints)
        require(rollout["schema"] == "evr0030.development.own-rollout.v1" and rollout["frames"] == 154 and
                cache["completed_frames"] == 386 and cache["completed_sequences"] == 6 and
                rollout["arm"] == worker["arm"] and not rollout["offline_attempts"] and
                rollout["model_initial_hash"] == rollout["model_final_hash"] == cache["model_final_hash"] and
                worker["cache_receipt_sha256"] == sha(binding["cache_receipt"]), "rollout/cache contract")
        if worker.get("checkpoint") is not None:
            d = worker["checkpoint"]; require(sha(d["path"]) == d["sha256"], "checkpoint changed")
            fingerprints[d["path"]] = d["sha256"]
        mapping = json.loads(binding["mapping"].read_text()); index = json.loads(binding["data_index"].read_text())
        require(mapping["status"] == "completed" and mapping["member_count"] == 386 and
                cache["data_index"]["sha256"] == sha(binding["data_index"]), "mapping/cache metadata")
        require([r["sequence"] for r in rollout["rows"]] == worker["dev_sequences"] ==
                ["cat-girl", "upside-down"], "complete fixed DEV cohort")
        inputs = {r["sequence"]: r for r in index["sequences"]}; bases = {r["sequence"]: r for r in cache["rows"]}
        reports = []
        require(rollout["outputs"] == [p for r in rollout["rows"] for p in r["predictions"]], "output inventory")
        for p in rollout["outputs"]:
            e._descriptor(dict(p, path=str(binding["rollout_receipt"].parent / p["path"])))
        for row in rollout["rows"]:
            name = row["sequence"]; source, base = inputs[name], bases[name]; n = base["frames"]
            labels = [m for m in mapping["mappings"] if m["sequence"] == name]
            require(source["role"] == "DEV" and source["parent_id"] == row["parent_id"] == name and
                    all(row[k] == base[k] for k in ("target_id", "shape", "frames")) and
                    [p["frame"] for p in row["predictions"]] == [m["frame"] for m in labels] == list(range(n)) and
                    all(m["role"] == "DEV" and all(m[k] == source["labels"][i][k] for k in
                        ("path", "sha256", "bytes")) for i, m in enumerate(labels)), "complete DEV identity")
            predictions = [dict(p, path=str(binding["rollout_receipt"].parent / p["path"])) for p in row["predictions"]]
            reports.append(e.evaluate_sequence(dict(sequence=name, parent_id=name, shape_hw=row["shape"],
                object_mode="target", target_id=row["target_id"], frame_ids=[f"{i:05d}" for i in range(n)],
                predictions=predictions, annotations=labels)))
        save(out / "evaluation.json", e.aggregate(reports))
        for path, digest in fingerprints.items():
            require(sha(path) == digest, "input/source changed")
        record.update(status="completed", arm=rollout["arm"], raw_frames=154, included_frames=150,
                      evaluation_sha256=sha(out / "evaluation.json"), bindings=spec["bindings"])
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc)); raise
    finally:
        record["wall_seconds"] = time.perf_counter() - began; save(out / "receipt.json", record)


if __name__ == "__main__":
    e.cv2.setNumThreads(0)
    run(sys.argv[1])
