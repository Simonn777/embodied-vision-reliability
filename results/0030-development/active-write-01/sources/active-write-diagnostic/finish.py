"""Wait for the admitted fit, then run its final checkpoint and unchanged metrics once."""
import importlib.util
import json
from pathlib import Path
import subprocess
import time


def main():
    root = Path(__file__).resolve().parent
    campaign = root.parent / "campaign"
    module_spec = importlib.util.spec_from_file_location("completed_continuation", campaign / "continue_campaign.py")
    helper = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(helper)
    read, save, sha, clean = helper.read, helper.save, helper.sha, helper.clean
    path = root / "continuation-01.json"
    if path.exists():
        raise ValueError("fresh continuation required; no retry")
    state = dict(status="waiting_for_fit", operations=[], source_sha256=sha(__file__),
                 helper_sha256=sha(campaign / "continue_campaign.py"))
    save(path, state)
    fit_spec = read(root / "fit-spec-01.json")
    outer_base = read(root / "fit-outer-spec-01.json")
    base = read(campaign / "teacher-rho015-spec-01.json")
    supervisor = Path(base["repo"]) / "experiments/0029-memory-information/supervise_exit_checked.py"

    def descriptor(p):
        return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))

    def run(name, script, updates, *, no_site=True, extras=()):
        spec = dict(base, output=str(root / name), **updates)
        spec["source_sha256"] = dict(base["source_sha256"])
        for source in (Path(__file__).resolve(), script, root / "ADMISSION.md", *extras):
            spec["source_sha256"][str(source)] = sha(source)
        worker, outer_path = root / (name + "-spec.json"), root / (name + "-outer-spec.json")
        if worker.exists() or outer_path.exists() or Path(spec["output"]).exists():
            raise ValueError("no overwrite: " + name)
        save(worker, spec)
        command = [str(Path(base["native_root"]) / "installation/venv/bin/python")]
        command += (["-S"] if no_site else []) + ["-B", "-W", "error", str(script), str(worker)]
        outer = dict(outer_base, command=command, operation_output=str(root / (name + "-supervision")),
                     result_file=str(root / name / "receipt.json"))
        save(outer_path, outer)
        row = dict(name=name, status="started", worker=descriptor(worker), operation=descriptor(outer_path))
        state["operations"].append(row); save(path, state)
        with (root / (name + "-launcher.log")).open("xb") as stdout, (root / (name + "-launcher.stderr.log")).open("xb") as stderr:
            result = subprocess.run(["/opt/homebrew/bin/python3.12", "-B", str(supervisor), str(outer_path)], stdout=stdout, stderr=stderr)
        supervision = clean(Path(outer["operation_output"]) / "receipt.json")
        product = read(outer["result_file"])
        if result.returncode or product["status"] != "completed" or supervision["output_sha256"] != sha(outer["result_file"]):
            raise ValueError("failed/inconsistent operation: " + name)
        row.update(status="completed", wall_seconds=supervision["wall_seconds"],
                   peak_group_rss_kib=supervision["peak_group_rss_kib"], result=descriptor(Path(outer["result_file"])))
        save(path, state)

    try:
        began = time.perf_counter()
        fit_outer_path = root / "fit-supervision-01/receipt.json"
        while read(fit_outer_path)["status"] == "started":
            if time.perf_counter() - began > 1800:
                raise TimeoutError("fit did not finish; no downstream dispatch")
            time.sleep(15)
        outer = clean(fit_outer_path)
        fit_path = root / "fit-01/receipt.json"
        fit = read(fit_path)
        if (fit["status"] != "completed" or outer["output_sha256"] != sha(fit_path)
                or outer["wall_seconds"] > fit_spec["complete_cost_budget_seconds"]
                or outer["spec_sha256"] != sha(root / "fit-outer-spec-01.json")
                or fit["spec_sha256"] != sha(root / "fit-spec-01.json")):
            raise ValueError("fit failed complete cost/integrity check")
        for source, expected in fit_spec["source_sha256"].items():
            if sha(source) != expected:
                raise ValueError("fit source changed")
        checkpoint = fit["checkpoints"][-1]
        checkpoint_path = root / "fit-01" / checkpoint["path"]
        if descriptor(checkpoint_path)["sha256"] != checkpoint["sha256"]:
            raise ValueError("final checkpoint changed")
        state.update(status="started", fit=descriptor(fit_path), fit_outer=descriptor(fit_outer_path))
        save(path, state)
        run("rollout-01", campaign / "rollout.py", dict(mode="rollout", arm="direct_all_active",
            dev_sequences=["cat-girl", "upside-down"], checkpoint=dict(checkpoint, path=str(checkpoint_path))))
        evaluation = root.parent / "evaluation"
        bindings = dict(read(evaluation / "baseline-spec-02.json")["bindings"])
        for key, source in dict(rollout_receipt=root/"rollout-01/receipt.json",
            rollout_outer=root/"rollout-01-supervision/receipt.json", rollout_operation=root/"rollout-01-outer-spec.json",
            rollout_worker_spec=root/"rollout-01-spec.json").items():
            bindings[key] = descriptor(source)
        run("evaluate-01", evaluation / "evaluate_rollout.py", dict(mode="metric", bindings=bindings),
            no_site=False, extras=(evaluation/"evaluate_jf.py", evaluation/"evaluate_baseline.py"))
        score = read(root / "evaluate-01/evaluation.json")
        references = {}
        for name, entry in fit_spec["development_comparison_references"].items():
            if sha(entry["path"]) != entry["sha256"]:
                raise ValueError("development comparison changed")
            references[name] = read(entry["path"])
        parents = {x["parent_id"]: x["mean"]["JF"] for x in score["parents"]}
        native = {x["parent_id"]: x["mean"]["JF"] for x in references["native"]["parents"]}
        if set(parents) != set(native) or len(parents) != 2:
            raise ValueError("incomplete development parents")
        better_mean = score["mean"]["JF"] > references["gt_memory"]["mean"]["JF"]
        better_parents = all(parents[p] > native[p] for p in parents)
        state.update(status="completed", score=dict(mean=score["mean"], parents=score["parents"]),
            development_decision=dict(exceeds_gt_control_mean=better_mean, both_parents_exceed_native=better_parents,
                action="replicate_at_seed1" if better_mean and better_parents else "close_exact_channel_affine_family",
                final_evidence_gate_met=False))
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save(path, state)


if __name__ == "__main__":
    main()
