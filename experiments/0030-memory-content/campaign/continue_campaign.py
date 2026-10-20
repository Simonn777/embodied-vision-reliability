"""Serial continuation of the declared first development configuration.

Each child is a separately bounded complete operation under the existing exit
supervisor. No hidden retries; stop on any failed operation. Parent owns analysis.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def clean(path):
    value = read(path)
    if (value["status"] != "completed" or value["exit_code"] != 0 or value["termination"] is not None
            or value["remaining_descendant_pids"] or value["cleanup_errors"]):
        raise ValueError("unclean complete operation: " + str(path))
    return value


def main():
    root = Path(__file__).resolve().parent
    native = root.parent.parent / "0029-runtime-02"
    repo = root.parent.parent / "embodied-vision-reliability"
    base = read(root / "teacher-rho015-spec-01.json")
    outer_base = read(root / "teacher-rho015-outer-spec-01.json")
    supervisor = repo / "experiments/0029-memory-information/supervise_exit_checked.py"
    receipt = root / "continuation-02.json"
    if receipt.exists():
        raise ValueError("fresh continuation receipt required; no automatic restart")
    state = dict(status="started", operations=[], metrics={}, source_sha256=sha(__file__),
                 supersedes="continuation-01 stopped before dispatch to add sequential metrics")
    save(receipt, state)

    def run(name, script, changes, *, no_site=True, extra_sources=()):
        spec = dict(base, output=str(root / name), **changes)
        spec["source_sha256"] = dict(base["source_sha256"])
        spec["source_sha256"][str(script)] = sha(script)
        for path in extra_sources:
            spec["source_sha256"][str(path)] = sha(path)
        spec_path = root / (name + "-spec.json")
        outer_path = root / (name + "-outer-spec.json")
        if spec_path.exists() or outer_path.exists() or Path(spec["output"]).exists():
            raise ValueError("refusing overwrite/retry: " + name)
        save(spec_path, spec)
        command = [str(native / "installation/venv/bin/python")]
        if no_site:
            command.append("-S")
        command += ["-B", "-W", "error", str(script), str(spec_path)]
        outer = dict(outer_base, command=command,
            operation_output=str(root / (name + "-supervision")),
            result_file=str(Path(spec["output"]) / "receipt.json"))
        save(outer_path, outer)
        row = dict(name=name, status="started", spec_sha256=sha(spec_path),
                   outer_spec_sha256=sha(outer_path))
        state["operations"].append(row); save(receipt, state)
        print(json.dumps(row), flush=True)
        with (root / (name + "-launcher.log")).open("xb") as stdout, (root / (name + "-launcher.stderr.log")).open("xb") as stderr:
            result = subprocess.run(["/opt/homebrew/bin/python3.12", "-B", str(supervisor), str(outer_path)],
                                    stdout=stdout, stderr=stderr)
        supervision = clean(Path(outer["operation_output"]) / "receipt.json")
        product = read(outer["result_file"])
        if result.returncode != 0 or product["status"] != "completed" or supervision["output_sha256"] != sha(outer["result_file"]):
            raise ValueError("failed/inconsistent child: " + name)
        if changes.get("direct_comparison") == "cost_matched" and supervision["wall_seconds"] > changes["complete_cost_budget_seconds"]:
            raise ValueError("complete outer direct cost exceeded matched budget")
        row.update(status="completed", wall_seconds=supervision["wall_seconds"],
                   peak_group_rss_kib=supervision["peak_group_rss_kib"], result_sha256=sha(outer["result_file"]))
        save(receipt, state); print(json.dumps(row), flush=True)
        return product, supervision, Path(spec["output"])

    try:
        teacher_outer_path = root / "teacher-rho015-supervision-01/receipt.json"
        while read(teacher_outer_path)["status"] == "started":
            time.sleep(2)
        teacher_outer = clean(teacher_outer_path)
        teacher = read(root / "teacher-rho015-01/receipt.json")
        if teacher["status"] != "completed" or teacher_outer["output_sha256"] != sha(root / "teacher-rho015-01/receipt.json"):
            raise ValueError("teacher did not complete")
        targets = root / "teacher-rho015-01" / teacher["targets"]["path"]
        if sha(targets) != teacher["targets"]["sha256"]:
            raise ValueError("teacher target mismatch")
        fits = {}
        for arm in ("multistep", "one_step", "gt_memory"):
            fits[arm] = run("fit-"+arm+"-rho015-seed0-01", root / "train.py",
                dict(mode="regress", arm=arm, targets=str(targets), targets_sha256=sha(targets)))
        def rollout(arm, checkpoint=None):
            name = "rollout-"+arm+"-rho015-seed0-01"
            result = run(name, root / "rollout.py",
                dict(mode="rollout", arm=arm, dev_sequences=["cat-girl", "upside-down"], checkpoint=checkpoint))
            evaluation = root.parent / "evaluation"
            bindings = dict(read(evaluation / "baseline-spec-02.json")["bindings"])
            for key, path in dict(rollout_receipt=root/name/"receipt.json",
                rollout_outer=root/(name+"-supervision")/"receipt.json",
                rollout_operation=root/(name+"-outer-spec.json"),
                rollout_worker_spec=root/(name+"-spec.json")).items():
                bindings[key] = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))
            metric, _, folder = run("evaluate-"+arm+"-rho015-seed0-01", evaluation / "evaluate_rollout.py",
                dict(mode="metric", bindings=bindings), no_site=False,
                extra_sources=[evaluation/"evaluate_jf.py", evaluation/"evaluate_baseline.py"])
            score = read(folder / "evaluation.json")
            state["metrics"][arm] = dict(mean=score["mean"], parents=score["parents"],
                                         evaluation_sha256=metric["evaluation_sha256"])
            save(receipt, state)
            print(json.dumps(dict(arm=arm, development_metrics=state["metrics"][arm])), flush=True)
            return result
        rollout("native")
        for arm, (fit, _, folder) in fits.items():
            ckpt = fit["checkpoints"][-1]
            rollout(arm, dict(ckpt, path=str(folder / ckpt["path"])))
        budget = teacher_outer["wall_seconds"] + fits["multistep"][1]["wall_seconds"]
        state["direct_complete_cost_budget_seconds"] = budget; save(receipt, state)
        for comparison in ("cost_matched", "update_matched"):
            fit, _, folder = run("fit-direct-"+comparison+"-rho015-seed0-01", root / "train.py",
                dict(mode="direct", direct_comparison=comparison, complete_cost_budget_seconds=budget))
            ckpt = fit["checkpoints"][-1]
            rollout("direct-"+comparison, dict(ckpt, path=str(folder / ckpt["path"])))
        state["status"] = "completed"
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save(receipt, state)


if __name__ == "__main__":
    main()
