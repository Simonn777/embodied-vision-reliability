"""Authorized D38 supervisor; original process lifecycle and RSS monitor unchanged.

Original 0027 supervisor/0025c monitor are unchanged. This is a new receipt
schema and cannot turn a previously failed operation into a clean operation.
The exit time recorded here is first observed by poll/wait, not kernel exit time.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

MONITOR = "experiments/0025c-rs-noise/supervise.py"
RSS_LIMIT = 8 * 1024 * 1024


def require(value, message):
    if not value:
        raise ValueError(message)


def load_monitor(root):
    path = root / MONITOR
    spec = importlib.util.spec_from_file_location("unchanged_process_tree_monitor", path)
    monitor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monitor)
    return monitor


def run_spec(spec_path, *, monitor=None):
    """The optional monitor object is only for manufactured process tests."""
    spec_path = Path(spec_path).resolve()
    specification = json.loads(spec_path.read_text())
    root = Path(specification["repo"]).resolve()
    output = Path(specification["operation_output"]).absolute()
    require(not output.exists() and not output.is_symlink(), "fresh supervision directory required")
    require(output.resolve() != root and root not in output.resolve().parents, "private supervision output")
    command = specification["command"]
    require(isinstance(command, list) and command and
            all(isinstance(v, str) and "\0" not in v for v in command), "command must be explicit argv")
    limit = specification["timeout"]
    require(type(limit) in (int, float) and 0 < limit <= 3600, "wall limit")
    limit_path = Path(__file__).resolve().with_name("resource_limit.py")
    loader = importlib.util.spec_from_file_location("d38_authorized_limit", limit_path)
    limit_module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(limit_module)
    authorization = limit_module.admit(specification, supervision=True)
    monitor = load_monitor(root) if monitor is None else monitor
    env = dict(os.environ)
    env.update(specification["environment"])
    require(all(env.get(k) == "4" for k in monitor.THREAD_KEYS), "four-thread environment")
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    output.mkdir(parents=True, exist_ok=False)
    record = dict(schema="evr0038.authorized-exit-checked-operation.v1", resource_exception=authorization, status="started",
        command=[monitor.display(v) for v in command], spec_sha256=monitor.sha256(spec_path),
        supervisor_sha256=monitor.sha256(__file__), monitor_sha256=monitor.sha256(root / MONITOR),
        expected_commit=specification["commit"], hard_timeout_seconds=limit,
        group_rss_limit_kib=RSS_LIMIT, thread_environment={k: env[k] for k in monitor.THREAD_KEYS},
        child_pid=None, declared_child_pgid=None, observed_child_pgid=None,
        pgid_observation_error=None, exit_observed_at_seconds=None,
        exit_observation_scope="First successful poll/wait observation, not kernel exit timestamp",
        pre_poll_snapshot=None, post_exit_snapshot=None, triggering_snapshot=None,
        cleanup_snapshots=[], signal_attempts=[],
        sampling_scope="Unchanged 0025c PID/PPID/PGID/RSS samples; no process STAT or start-time identity")
    receipt = output / "receipt.json"
    monitor.save(receipt, record)
    start = time.perf_counter()
    groups, peak, termination, code, child = set(), 0, None, None, None
    remaining, cleanup_errors, interrupted = [], [], None

    def sample(label):
        nonlocal peak
        before = time.perf_counter() - start
        known = sorted(groups)
        rows, new = monitor.snapshot(child.pid, groups)
        groups.update(new)
        peak = max(peak, sum(row[3] for row in rows))
        return dict(label=label, started_seconds=before,
            finished_seconds=time.perf_counter() - start, known_groups=known,
            discovered_groups=sorted(new), rows=[list(row) for row in rows])

    def observe_exit():
        nonlocal code
        value = child.poll()
        if value is not None:
            code = value
            if record["exit_observed_at_seconds"] is None:
                record["exit_observed_at_seconds"] = time.perf_counter() - start
        return value

    def send(group, sig):
        row = dict(pgid=group, signal=sig.name, attempted_seconds=time.perf_counter() - start)
        record["signal_attempts"].append(row)
        try:
            require(group != os.getpgrp(), "refusing to signal supervisor process group")
            os.killpg(group, sig)
            row["result"] = "sent"
        except ProcessLookupError:
            row["result"] = "not_found"
        except (OSError, ValueError) as exc:
            row.update(result="error", error_type=type(exc).__name__, error=str(exc))
            cleanup_errors.append(sig.name + ":" + type(exc).__name__)
        row["finished_seconds"] = time.perf_counter() - start

    def clean_groups():
        send(child.pid, signal.SIGSTOP)
        try:
            record["cleanup_snapshots"].append(sample("before_group_kill"))
        except Exception as exc:
            cleanup_errors.append("cleanup_sample:" + type(exc).__name__)
        for group in sorted(groups):
            send(group, signal.SIGKILL)

    with (output / "stdout.log").open("xb") as stdout, (output / "stderr.log").open("xb") as stderr:
        try:
            child = subprocess.Popen(command, cwd=root, env=env, stdout=stdout, stderr=stderr,
                                     start_new_session=True)
            groups.add(child.pid)
            record.update(child_pid=child.pid, declared_child_pgid=child.pid)
            try:
                record["observed_child_pgid"] = os.getpgid(child.pid)
                require(record["observed_child_pgid"] == child.pid, "child session/process group differs")
            except ProcessLookupError:
                record["pgid_observation_error"] = "ProcessLookupError"
            monitor.save(receipt, record)
            while True:
                before = sample("before_poll")
                record["pre_poll_snapshot"] = before
                if peak > RSS_LIMIT:
                    termination = "group_rss_limit"; record["triggering_snapshot"] = before; break
                if time.perf_counter() - start >= limit:
                    termination = "timeout"; record["triggering_snapshot"] = before; break
                if observe_exit() is not None:
                    # Crucial change: do not classify the earlier sample as
                    # evidence that a descendant remains AFTER parent exit.
                    after = sample("after_exit_observed")
                    record["post_exit_snapshot"] = after
                    if any(row[0] != child.pid for row in after["rows"]):
                        termination = "descendants_remain_after_fresh_exit_sample"
                        record["triggering_snapshot"] = after
                    break
                time.sleep(.1)
        except BaseException as exc:
            termination = type(exc).__name__ + ": " + monitor.display(exc)
            if not isinstance(exc, Exception):
                interrupted = exc
        finally:
            if child is not None:
                if observe_exit() is None or termination is not None or child.returncode != 0:
                    clean_groups()
                try:
                    code = child.wait(timeout=5)
                    if record["exit_observed_at_seconds"] is None:
                        record["exit_observed_at_seconds"] = time.perf_counter() - start
                except Exception as exc:
                    cleanup_errors.append("child_wait:" + type(exc).__name__)
                    termination = termination or "child_wait_failed"
                try:
                    final = sample("final")
                    record["cleanup_snapshots"].append(final)
                    remaining = [row[0] for row in final["rows"] if row[0] != child.pid]
                    if remaining and termination is None:
                        termination = "descendants_in_final_sample"
                        record["triggering_snapshot"] = final
                        clean_groups()
                        final = sample("after_final_cleanup")
                        record["cleanup_snapshots"].append(final)
                        remaining = [row[0] for row in final["rows"] if row[0] != child.pid]
                except Exception as exc:
                    cleanup_errors.append("final_sample:" + type(exc).__name__)
                    termination = termination or "final_sample_failed"
            stdout.flush(); stderr.flush()
    elapsed = time.perf_counter() - start
    if peak > RSS_LIMIT:
        termination = termination or "group_rss_limit_at_exit"
    if elapsed >= limit:
        termination = termination or "timeout_at_exit"
    record.update(status="completed" if code == 0 and termination is None and not remaining
                  and not cleanup_errors else "failed", exit_code=code, termination=termination,
        wall_seconds=elapsed, peak_group_rss_kib=peak, remaining_descendant_pids=remaining,
        cleanup_errors=cleanup_errors,
        log_sha256={name: monitor.sha256(output / name) for name in ("stdout.log", "stderr.log")})
    result = specification.get("result_file")
    if result is not None and Path(result).is_file():
        record["output_sha256"] = monitor.sha256(result)
    monitor.save(receipt, record)
    if time.perf_counter() - start >= limit and record["status"] == "completed":
        record.update(status="failed", termination="timeout_at_final_receipt",
                      wall_seconds=time.perf_counter() - start)
        monitor.save(receipt, record)
    if interrupted is not None:
        raise interrupted
    return record


if __name__ == "__main__":
    value = run_spec(sys.argv[1])
    print(json.dumps(value, indent=2), flush=True)
    raise SystemExit(0 if value["status"] == "completed" else 1)
