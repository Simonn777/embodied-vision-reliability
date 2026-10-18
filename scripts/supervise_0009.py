#!/usr/bin/env python3
"""Linux-only external resource supervisor for the single 0009 collection.

The command is the frozen collector, never an arbitrary shell string. This
wrapper records wait4 evidence even when the collector is killed. It does not
evaluate outcomes, resume cases, install dependencies, or make network requests.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time


LIMIT_SECONDS = 900.0
ADDRESS_SPACE_BYTES = 8 * 1024**3
THREADS = 4


class _ReceivedSignal(Exception):
    def __init__(self, number):
        self.number = number


def supervise(command, output, *, seconds=LIMIT_SECONDS, cpu_ids=None):
    """Run a list argv with strict process limits; write a fresh receipt directory.

    The smaller seconds override is for artificial supervisor tests only. The
    CLI has no timeout override. Linux ru_maxrss is KiB. The child forms its own
    process group; on deadline the entire group is killed before wait4 reaping.
    """
    if not sys.platform.startswith("linux"):
        raise RuntimeError("0009 scientific supervisor requires Linux")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not (0 < seconds <= LIMIT_SECONDS):
        raise ValueError("deadline must be positive and at most 900 seconds")
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        raise ValueError("command must be nonempty string argv")
    available = sorted(os.sched_getaffinity(0))
    selected = available[:THREADS] if cpu_ids is None else list(cpu_ids)
    if (not selected or len(selected) > THREADS or
            any(type(cpu) is not int for cpu in selected) or len(set(selected)) != len(selected) or
            not set(selected) <= set(available)):
        raise ValueError("invalid CPU affinity")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(THREADS)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    launch_mask = None

    def limits():
        os.setsid()
        os.sched_setaffinity(0, selected)
        resource.setrlimit(resource.RLIMIT_AS, (ADDRESS_SPACE_BYTES, ADDRESS_SPACE_BYTES))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        signal.pthread_sigmask(signal.SIG_SETMASK, launch_mask)

    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    start = time.monotonic()
    timed_out = False
    interrupted = False
    interrupt_signal = None
    proc = None
    status = usage = None
    failure_type = failure_stage = reap_error = None
    fatal = None
    command_record = {"argv": command, "cwd": os.getcwd(), "started_utc": started_utc,
                      "wall_limit_seconds": seconds, "address_space_bytes": ADDRESS_SPACE_BYTES,
                      "cpu_affinity": selected, "numeric_threads": THREADS}
    (output / "command.json").write_text(json.dumps(command_record, indent=2) + "\n")
    handlers = {}

    def receive_signal(number, _frame):
        raise _ReceivedSignal(number)

    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            handlers[number] = signal.signal(number, receive_signal)
        with (output / "stdout.log").open("xb") as stdout, (output / "stderr.log").open("xb") as stderr:
            # Defer these signals until Popen returns a PID we can clean up.
            # The child restores the inherited mask before exec as well.
            launch_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
            try:
                proc = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr, preexec_fn=limits)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, launch_mask)
            while True:
                pid, child_status, child_usage = os.wait4(proc.pid, os.WNOHANG)
                if pid:
                    status, usage = child_status, child_usage
                    break
                elapsed = time.monotonic() - start
                if elapsed >= seconds:
                    timed_out = True
                    break
                time.sleep(min(0.02, seconds - elapsed))
    except (_ReceivedSignal, KeyboardInterrupt) as exc:
        interrupted = True
        interrupt_signal = exc.number if isinstance(exc, _ReceivedSignal) else signal.SIGINT
    except Exception as exc:
        failure_type, failure_stage = type(exc).__name__, "launch" if proc is None else "supervision"
    except BaseException as exc:
        # Preserve unusual control-flow exceptions, after cleanup and evidence.
        failure_type, failure_stage = type(exc).__name__, "supervision"
        fatal = exc
    finally:
        # A second termination signal must not interrupt kill/reap and orphan work.
        for number in handlers:
            signal.signal(number, signal.SIG_IGN)
        if proc is not None:
            # Also terminate any surviving descendants after a normal leader exit.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if status is None:
                try:
                    _, status, usage = os.wait4(proc.pid, 0)
                except ChildProcessError as exc:
                    reap_error = type(exc).__name__
            if status is not None:
                proc.returncode = os.waitstatus_to_exitcode(status)
        for number, handler in handlers.items():
            signal.signal(number, handler)
    wall = time.monotonic() - start
    child_exit = proc.returncode if proc is not None and status is not None else None
    supervisor_exit = (None if fatal is not None else 124 if timed_out else
                       128 + interrupt_signal if interrupted else
                       125 if failure_type or reap_error or child_exit is None else
                       child_exit if child_exit >= 0 else 128 - child_exit)
    receipt = dict(command_record)
    receipt.update({"ended_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "raw_wait_status": status, "child_exit_code": child_exit,
                    "wall_seconds": wall, "user_cpu_seconds": usage.ru_utime if usage else None,
                    "system_cpu_seconds": usage.ru_stime if usage else None,
                    "max_rss_kib": usage.ru_maxrss if usage else None,
                    "timed_out": timed_out, "supervisor_interrupted": interrupted,
                    "interrupt_signal": interrupt_signal, "failure_type": failure_type,
                    "failure_stage": failure_stage, "reap_error": reap_error,
                    "supervisor_exit_code": supervisor_exit})
    receipt["logs"] = {}
    for name in ("stdout.log", "stderr.log"):
        path = output / name
        data = path.read_bytes() if path.is_file() else None
        receipt["logs"][name] = ({"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                                 if data is not None else None)
    (output / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if fatal is not None:
        raise fatal
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, help="Existing cloud interpreter")
    parser.add_argument("--freeze-commit", required=True)
    parser.add_argument("--output", required=True, help="Fresh scientific output directory")
    parser.add_argument("--supervision-output", required=True, help="Fresh external receipt directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    command = [str(Path(args.python).resolve()), str(root / "scripts/run_geometric_stage0.py"),
               "collect", "--freeze-commit", args.freeze_commit, "--output", args.output]
    receipt = supervise(command, args.supervision_output)
    binding = bind_collection(args.output, args.supervision_output)
    print(json.dumps({key: receipt[key] for key in ("child_exit_code", "supervisor_exit_code",
          "timed_out", "wall_seconds", "max_rss_kib")}), flush=True)
    return binding["effective_exit_code"]


def bind_collection(output, supervision_output):
    """Bind a closed execution receipt and selection lock without rewriting either."""
    folder = Path(supervision_output)
    execution = (folder / "execution.json").read_bytes()
    receipt = json.loads(execution)
    lock = Path(output) / "selection-lock.json"
    record = None
    if not any(p.is_symlink() for p in (lock, *lock.parents)) and lock.is_file():
        data = lock.read_bytes()
        record = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    code = receipt["supervisor_exit_code"]
    if type(code) is not int:
        raise ValueError("cannot bind an execution without a known supervisor return code")
    if receipt["child_exit_code"] == 0 and record is None:
        code = 125
    binding = {"schema": "evr0009.collection-binding.v1",
               "execution_sha256": hashlib.sha256(execution).hexdigest(),
               "selection_lock": record,
               "lock_status": "present" if record is not None else "missing_or_unsafe",
               "effective_exit_code": code}
    with (folder / "collection-binding.json").open("x") as stream:
        stream.write(json.dumps(binding, indent=2) + "\n")
    return binding


if __name__ == "__main__":
    raise SystemExit(main())
