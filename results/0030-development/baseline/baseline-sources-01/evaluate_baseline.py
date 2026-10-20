"""Evaluate sealed baseline on the two exposed DEV parents; never loads a model."""
import hashlib
import json
from pathlib import Path
import sys
import time
import evaluate_jf as e


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(spec_path):
    started = time.perf_counter()
    spec = json.loads(Path(spec_path).read_text())
    output = Path(spec["output"])
    output.mkdir(exist_ok=False)
    record = dict(schema="evr0030.baseline-evaluation.v1", status="started",
                  model_calls=0, final_test_access=False, source_sha256=spec["source_sha256"],
                  spec_sha256=sha(spec_path), completed_sequences=[])
    try:
        fingerprints = dict(spec["source_sha256"])
        for d in spec["bindings"].values():
            e._descriptor(d); fingerprints[d["path"]] = d["sha256"]
        for name, digest in fingerprints.items():
            require(sha(name) == digest, "source or metadata binding mismatch")
        binding = {k: Path(v["path"]) for k, v in spec["bindings"].items()}
        cache = json.loads(binding["cache_receipt"].read_text())
        outer = json.loads(binding["cache_outer"].read_text())
        operation = json.loads(binding["cache_operation"].read_text())
        mapping = json.loads(binding["mapping"].read_text())
        index = json.loads(binding["data_index"].read_text())
        require(outer["status"] == "completed" and outer["exit_code"] == 0 and
                outer["termination"] is None and not outer["remaining_descendant_pids"] and
                not outer["cleanup_errors"] and not outer["signal_attempts"] and
                not outer["post_exit_snapshot"]["rows"], "cache outer not clean")
        require(outer["wall_seconds"] <= 1800 and outer["peak_group_rss_kib"] <= 8388608 and
                outer["output_sha256"] == sha(binding["cache_receipt"]) and
                outer["spec_sha256"] == sha(binding["cache_operation"]) and
                Path(operation["result_file"]) == binding["cache_receipt"], "cache operation binding")
        for name, digest in outer["log_sha256"].items():
            p = binding["cache_outer"].parent / name
            require(sha(p) == digest, "cache log changed"); fingerprints[str(p)] = digest
        require(cache["status"] == mapping["status"] == "completed" and
                cache["completed_frames"] == mapping["member_count"] == 386 and
                cache["completed_sequences"] == 6 and cache["data_index"]["sha256"] ==
                sha(binding["data_index"]), "complete cache/mapping required")
        rows = {r["sequence"]: r for r in cache["rows"]}
        inputs = {r["sequence"]: r for r in index["sequences"]}
        reports = []
        for name in ("cat-girl", "upside-down"):
            row, source = rows[name], inputs[name]
            n = row["frames"]
            require(row["status"] == "completed" and source["role"] == "DEV" and
                    source["parent_id"] == name and len(row["predictions"]) == n and
                    [p["frame"] for p in row["predictions"]] == list(range(n)), "complete DEV identity")
            labels = [m for m in mapping["mappings"] if m["sequence"] == name]
            require([m["frame"] for m in labels] == list(range(n)) and
                    all(m["role"] == "DEV" and all(m[k] == source["labels"][i][k]
                        for k in ("path", "sha256", "bytes")) for i, m in enumerate(labels)), "label mapping")
            predictions = [dict(p, path=str(binding["cache_receipt"].parent / p["path"]))
                           for p in row["predictions"]]
            report = e.evaluate_sequence(dict(sequence=name, parent_id=name, shape_hw=row["shape"],
                object_mode="target", target_id=row["target_id"], frame_ids=[f"{i:05d}" for i in range(n)],
                predictions=predictions, annotations=labels))
            reports.append(report); record["completed_sequences"].append(name)
        require(sum(r["frame_count"] for r in reports) == 154, "full DEV frame coverage")
        save(output / "evaluation.json", e.aggregate(reports))
        for name, digest in fingerprints.items():
            require(sha(name) == digest, "input/source changed during evaluation")
        record.update(status="completed", raw_frames=154, included_frames=150,
                      evaluation_sha256=sha(output / "evaluation.json"), bindings=spec["bindings"])
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        record["wall_seconds"] = time.perf_counter() - started
        save(output / "receipt.json", record)


if __name__ == "__main__":
    e.cv2.setNumThreads(0)
    run(sys.argv[1])
