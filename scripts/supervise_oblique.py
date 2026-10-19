"""Frozen, cache-only 0011 phases with exclusive markers and bounded Linux CPU."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import re
import signal
import subprocess
import sys
import time

PHASES = ("prepare", "g-036", "g-002", "derive", "score")
LIMITS = (300, 900, 900, 1800, 1800)
MEMORY = 8 * 1024**3
REQUIRED_FILES = (
    "scripts/supervise_oblique.py", "scripts/prepare_oblique_development.py",
    "scripts/run_oblique_predictions.py", "scripts/evaluate_oblique_saved.py",
    "scripts/oblique_fusion.py", "scripts/marginal_coupling.py", "scripts/oblique_view_geometry.py",
    "src/evr/__init__.py", "src/evr/bootstapir_dev.py", "src/evr/pilot_identity.py",
    "configs/0011-oblique-development.json", "protocols/0011-oblique-coupling-development.md",
    "results/0002-preflight/identity/manifest.json",
    "results/0005-recovered-capability/native/run/receipt.json",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024**2), b""):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def read(path):
    return json.loads(Path(path).read_text())


def verify_freeze(repo, expected):
    freeze = repo / "protocols/0011-oblique-development-freeze.json"
    if sha(freeze) != expected:
        raise ValueError("published freeze receipt differs")
    data = read(freeze)
    if (not isinstance(data.get("files"), dict) or not set(REQUIRED_FILES) <= set(data["files"]) or
        not isinstance(data.get("source_commit"), str) or not re.fullmatch("[0-9a-f]{40}", data["source_commit"])):
        raise ValueError("incomplete freeze/source identity")
    for name, digest in data["files"].items():
        p = repo / name
        if not p.resolve().is_relative_to(repo.resolve()) or p.is_symlink() or sha(p) != digest:
            raise ValueError("frozen source differs: " + name)
    return data


def verify_prior_phases(output, phase, freeze_sha):
    for prior in PHASES[:PHASES.index(phase)]:
        folder = output / "supervision" / prior
        previous = read(folder / "execution.json")
        binding = read(folder / "source-binding.json")
        if (previous["supervisor_exit_code"] != 0 or previous["child_exit_code"] != 0 or
            binding.get("freeze_sha256") != freeze_sha or
            binding.get("execution_sha256") != sha(folder / "execution.json")):
            raise ValueError("prior phase failed or binding changed; no continuation")


def clean_environment(repo, base):
    env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
           "PYTHONHASHSEED": "20261018", "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": ""}
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in (
        base / "tapnextpp-cache/overlay", base / "bootstapir-cache/runtime-fixed",
        base / "bootstapir-cache/runtime", repo / "src", repo / "scripts"))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "4"
    return env


def command_for(phase, repo, base, output, config):
    python = config["python"]
    prep = output / "prepared"
    if phase == "prepare":
        return [python, str(repo / "scripts/prepare_oblique_development.py"),
                "--archive", str(base / "bootstapir-cache/rgb.zip"),
                "--manifest", str(repo / "results/0002-preflight/identity/manifest.json"),
                "--output", str(prep)]
    if phase in ("g-036", "g-002"):
        preparation = read(prep / "receipt.json")
        if preparation["status"] != "completed_preparation":
            raise ValueError("preparation not completed")
        inp = prep / (phase + ".predictor.npz")
        digest = preparation["files"][inp.name]["sha256"]
        if sha(inp) != digest:
            raise ValueError("prepared predictor changed")
        return [python, str(repo / "scripts/run_oblique_predictions.py"),
                "--prepared-input", str(inp), "--input-sha256", digest,
                "--group-id", phase, "--output", str(output / "predictions" / phase),
                "--source-root", str(base / "tapnextpp-recovery/source"),
                "--checkpoint", str(base / "tapnextpp-recovery/weights/tapnextpp_ckpt.pt"),
                "--overlay-root", str(base / "tapnextpp-cache/overlay"),
                "--base-runtime-root", str(base / "bootstapir-cache/runtime-fixed"),
                "--asset-receipt", str(base / "scratch/0005-recovered-native-capability/run/receipt.json"),
                "--asset-receipt-sha256", config["asset_receipt_sha256"],
                "--protocol-sha256", sha(repo / "protocols/0011-oblique-coupling-development.md")]
    return [python, str(repo / "scripts/evaluate_oblique_saved.py"), phase,
            "--input", str(output / ("predictions" if phase == "derive" else "derived")),
            "--preparation", str(prep),
            "--output", str(output / ("derived" if phase == "derive" else "evaluation"))]


def supervise(command, output, *, cwd, env, seconds):
    """External wait4 receipt survives child failure, timeout and interruption."""
    if sys.platform != "linux" or seconds not in LIMITS:
        raise ValueError("frozen Linux phase deadline required")
    selected = sorted(os.sched_getaffinity(0))[:4]
    if len(selected) != 4:
        raise ValueError("four available CPUs required")
    output.mkdir(exist_ok=False)
    record = {"argv": command, "cwd": str(cwd), "environment": env,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "wall_limit_seconds": seconds, "memory_limit_bytes": MEMORY,
              "cpu_affinity": selected, "threads": 4}
    write(output / "started.json", record)
    start, proc, status, usage = time.monotonic(), None, None, None
    timed_out, interrupted, launch_error = False, None, None
    old_handlers = {}
    def terminate(signum, frame):
        raise InterruptedError(signum)
    def limits():
        os.setsid()
        os.sched_setaffinity(0, selected)
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY, MEMORY))
        resource.setrlimit(resource.RLIMIT_CORE, (0,0))
        signal.pthread_sigmask(signal.SIG_SETMASK, [])
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, terminate)
        with (output / "stdout.log").open("xb") as stdout, (output / "stderr.log").open("xb") as stderr:
            mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
            try:
                proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout,
                                        stderr=stderr, preexec_fn=limits)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, mask)
            while True:
                mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
                try:
                    pid, st, us = os.wait4(proc.pid, os.WNOHANG)
                    if pid:
                        status, usage = st, us
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, mask)
                if pid:
                    break
                if time.monotonic() - start >= seconds:
                    timed_out = True
                    break
                time.sleep(.05)
    except (InterruptedError, KeyboardInterrupt) as exc:
        interrupted = str(exc) or type(exc).__name__
    except BaseException as exc:
        launch_error = {"type": type(exc).__name__, "error": str(exc)}
    finally:
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if status is None:
                _, status, usage = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(status)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    code = 124 if timed_out else 125 if interrupted or launch_error or status is None else os.waitstatus_to_exitcode(status)
    record.update(child_exit_code=None if status is None else os.waitstatus_to_exitcode(status),
                  supervisor_exit_code=code, timed_out=timed_out, interrupted=interrupted,
                  launch_error=launch_error, raw_wait_status=status,
                  wall_seconds=time.monotonic()-start,
                  max_rss_kib=None if usage is None else usage.ru_maxrss,
                  user_cpu_seconds=None if usage is None else usage.ru_utime,
                  system_cpu_seconds=None if usage is None else usage.ru_stime,
                  ended_utc=datetime.now(timezone.utc).isoformat())
    record["logs"] = {n: {"bytes": (output/n).stat().st_size, "sha256": sha(output/n)}
                      for n in ("stdout.log", "stderr.log") if (output/n).exists()}
    write(output / "execution.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--freeze-sha256", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    freeze = verify_freeze(repo, args.freeze_sha256)
    config = read(repo / "configs/0011-oblique-development.json")
    base = Path(config["cloud_base"])
    output = base / config["output_relative"]
    if args.phase == "prepare":
        output.mkdir(exist_ok=False)
        (output / "predictions").mkdir()
        (output / "supervision").mkdir()
    if not output.is_dir() or output.is_symlink():
        raise ValueError("missing/unsafe once-only root")
    with (output / "active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        index = PHASES.index(args.phase)
        verify_prior_phases(output, args.phase, args.freeze_sha256)
        command = command_for(args.phase, repo, base, output, config)
        control = output / "supervision" / args.phase
        receipt = supervise(command, control, cwd=repo, env=clean_environment(repo, base), seconds=LIMITS[index])
        write(control / "source-binding.json", {"freeze_sha256": args.freeze_sha256,
              "source_commit": freeze["source_commit"], "execution_sha256": sha(control / "execution.json")})
        print(json.dumps({k:receipt[k] for k in ("supervisor_exit_code", "child_exit_code", "wall_seconds", "max_rss_kib")}))
        return receipt["supervisor_exit_code"] if receipt["supervisor_exit_code"] >= 0 else 128 - receipt["supervisor_exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
