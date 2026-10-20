"""Admission for the user's single complete D38 60-minute exception; no I/O on import."""
import hashlib
import json
import math
from pathlib import Path

AUTHORIZATION_SHA = "05bc131cc974df1df0d267ae727b5c8a641188839960b501391c79c2dd0175ce"
OPERATION = "d38-complete-01"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def admit(spec, *, supervision=False):
    key = "timeout" if supervision else "operation_wall_seconds"
    limit = spec.get(key, 1800)
    require(type(limit) in (int, float) and math.isfinite(limit) and 0 < limit <= 3600,
            "finite positive operation limit at most 3600 seconds")
    if limit <= 1800:
        return None
    require(limit == 3600 and spec.get("operation_id") == OPERATION,
            "only the one named complete operation has the exception")
    here = Path(__file__).resolve().parent
    expected_path = here / "resource-authorization-01.json"
    desc = spec.get("resource_exception", {})
    require(desc.get("path") == str(expected_path) and desc.get("sha256") == AUTHORIZATION_SHA
            and expected_path.is_file() and not expected_path.is_symlink(), "bound direct-user authorization required")
    raw = expected_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == AUTHORIZATION_SHA, "authorization changed")
    authorization = json.loads(raw)
    require(authorization["operation_id"] == OPERATION and authorization["max_complete_operations"] == 1
            and authorization["hard_timeout_seconds"] == 3600 and authorization["cpu_threads"] == 4
            and authorization["group_rss_limit_kib"] == 8388608
            and authorization["additional_fees"] is False and authorization["new_acquisition"] is False
            and (authorization["complete_clips"], authorization["complete_frames"], authorization["arms"]) == (4, 446, 9),
            "unchanged one-operation resource and scientific scope")
    output_key = "operation_output" if supervision else "output"
    basename = "supervision_output_basename" if supervision else "scientific_output_basename"
    output = Path(spec[output_key])
    require(output.is_absolute() and output.name == authorization[basename], "fixed fresh operation output identity")
    if supervision:
        command = spec["command"]
        require(len(command) == 5 and command[1:3] == ["-B", "-S"]
                and command[3] == str(here / "run_complete.py"), "authorized complete worker argv")
        worker_path = Path(command[4])
        require(worker_path.is_absolute() and worker_path.is_file() and not worker_path.is_symlink(),
                "bound complete worker specification")
        worker_raw = worker_path.read_bytes()
        require(hashlib.sha256(worker_raw).hexdigest() == spec.get("worker_spec_sha256"), "worker spec changed")
        worker = json.loads(worker_raw)
        require(worker["resource_exception"] == desc and worker["operation_id"] == OPERATION
                and worker["operation_wall_seconds"] == limit and worker["repo"] == spec["repo"]
                and worker["launch_admitted"] is True and worker["preparation_only"] is False
                and command[0] == str(Path(worker["native_root"]) / "installation/venv/bin/python")
                and spec["result_file"] == str(Path(worker["output"]) / "receipt.json")
                and Path(worker["output"]).parent == output.parent, "same complete worker/resource/output identity")
        admit(worker)
    return desc
