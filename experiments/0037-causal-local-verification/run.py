"""One complete causal local-development comparison; --spec binds all inputs.

Stages: causal features -> feature seal -> cross-fit targets -> fits -> all
predictions seal -> metrics. No native model is loaded or modified.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

COHORT = (("4691_cut_guava", 86), ("817_cut_onion", 90),
          ("9675_fold_clothes", 201), ("1934_tear_film", 69))
FIXED = ("B", "B+M1", "B+M2", "B+M1+M2")
LEARNED = ("ordinary_mlp", "no_ancestry_mlp", "negative_gate_mlp")
LOCAL = ("fixed_negative", "positive_required") + LEARNED
POLICIES = ("author_background", "void_ignore")
D35_SPEC_SHA = "325961b52bed72c8710fee107265b02545002f9b7591fdd4119095841611a68d"
D35_RESULT_SHA = "7fe9592298d1cced864eaa6f75faf973e98c10ea3ce8c91100a678cc6cd60fe3"
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 32*1024**2,
            "bounded regular JSON required")
    return json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def fold_samples(training, held, np):
    """The excluded clip contributes no X, targets, weights or normalization."""
    require(set(training) == {name for name,_ in COHORT} and held in training, "complete fold keys")
    sources = [name for name,_ in COHORT if name != held]
    x = np.concatenate([training[s]["x"] for s in sources])
    y = np.concatenate([training[s]["y"] for s in sources])
    w = np.concatenate([training[s]["w"] for s in sources])/3.0
    return sources, x, y, w


def unpack_counts(packed, policy):
    counts = {k.split("__",1)[1]: v for k,v in packed.items() if k.startswith(policy+"__")}
    counts["total_gt"] = int(counts["total_gt"].item())
    return counts


def continuation(primary):
    methods = primary["methods"]
    candidate = methods["negative_gate_mlp"]
    gains_fixed = {m: candidate[m] - max(methods[a][m] for a in FIXED) for m in ("J", "Jtr")}
    gains_controls = {a: {m: candidate[m]-methods[a][m] for m in ("J", "Jtr")}
                      for a in ("fixed_negative", "ordinary_mlp", "no_ancestry_mlp")}
    per_clip = {r["sequence"]: {a: {m: r["methods"]["negative_gate_mlp"][m]-r["methods"][a][m]
                for m in ("J", "Jtr")} for a in ("B", "B+M1+M2")}
                for r in primary["per_sequence"]}
    clauses = {"over_fixed_3pp_"+m: gains_fixed[m] >= .03 for m in gains_fixed}
    clauses.update({"over_"+a+"_0.5pp_"+m: gains_controls[a][m] >= .005
                    for a in gains_controls for m in ("J", "Jtr")})
    clauses.update(three_clips_positive_Jtr_over_union=sum(
        v["B+M1+M2"]["Jtr"] > 0 for v in per_clip.values()) >= 3,
        no_clip_native_loss_over_0_5pp=all(v["B"][m] >= -.005
        for v in per_clip.values() for m in ("J", "Jtr")))
    return dict(clauses=clauses, numerical_passed=all(clauses.values()),
                gains_over_strongest_fixed=gains_fixed, gains_over_controls=gains_controls,
                per_clip_gains=per_clip, clean_outer_and_independent_review_required=True)


def run(spec_path):
    spec_path = Path(spec_path).resolve()
    spec_hash, spec = sha(spec_path), load(spec_path)
    out = Path(spec["output"]).absolute()
    require(not out.exists(), "fresh complete-operation output")
    out.mkdir(parents=True)
    started = time.monotonic()
    record = dict(schema="evr0037.local-comparison-receipt.v1", status="started",
        spec_sha256=spec_hash, source_sha256=spec["source_sha256"], phase="admission",
        completed_feature_frames=0, completed_target_frames=0, completed_prediction_frames=0,
        completed_metric_frames=0, completed_fits=0, flow_calls=0, model_calls=0,
        initial_label_decodes=0, noninitial_label_decodes=0, offline_attempts=[],
        compatibility_probes=[], bytes_written=0, outputs=[], phase_seconds={})
    save(out/"receipt.json", record)
    files = {}

    def checkpoint(phase):
        record.update(phase=phase, wall_seconds=time.monotonic()-started)
        save(out/"receipt.json", record)

    def artifact(path, kind):
        path = Path(path)
        desc = dict(path=str(path.relative_to(out)), bytes=path.stat().st_size,
                    sha256=sha(path), kind=kind)
        require(desc["path"] not in files, "artifact must be immutable")
        files[desc["path"]] = desc
        record["outputs"].append(desc)
        record["bytes_written"] += desc["bytes"]
        require(record["bytes_written"] <= 3*1024**3, "fixed 3GiB output cap")
        return desc

    def read_npz(desc):
        path = out/desc["path"]
        require(sha(path) == desc["sha256"] and path.stat().st_size == desc["bytes"], "cached output changed")
        with np.load(path, allow_pickle=False) as f:
            return {k: f[k] for k in f.files}

    try:
        here = Path(__file__).resolve().parent
        require(spec["schema"] == "evr0037.local-comparison-spec.v1", "spec schema")
        require(all(sha(p) == h for p,h in spec["source_sha256"].items()), "source closure")
        for name in ("run.py", "core.py", "models.py", "features.py", "ADMISSION.md"):
            require(str(here/name) in spec["source_sha256"], "bound executing source")
        require(spec["d35_spec"]["sha256"] == D35_SPEC_SHA and
                sha(spec["d35_spec"]["path"]) == D35_SPEC_SHA, "exact D35 admission")
        require(spec["d35_result"]["sha256"] == D35_RESULT_SHA and
                sha(spec["d35_result"]["path"]) == D35_RESULT_SHA, "exact D35 parity results")
        oldspec, oldresult = load(spec["d35_spec"]["path"]), load(spec["d35_result"]["path"])
        d35 = Path(oldspec["metrics_path"]).parent
        admission = module("d37_admit", d35/"admit.py")
        context = admission.evaluation_context(oldspec)
        for p,h in oldspec["source_sha256"].items():
            require(spec["source_sha256"].get(p) == h, "old source closure unchanged")
        require(sys.flags.no_site == 1 and sys.version_info[:3] == (3,12,13)
                and Path(sys.executable).absolute() == Path(context["installation"])/"venv/bin/python",
                "qualified interpreter")
        require(all(os.environ.get(k) == "4" for k in THREADS), "four-thread envelope")
        guard = module("d37_guard", oldspec["guard_source"])
        guard.install_guard(record["offline_attempts"], record["compatibility_probes"])
        os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        import site
        site.main()
        import numpy as np
        import torch
        import cv2
        from PIL import Image
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(0)
        require(np.__version__ == "2.2.6" and cv2.__version__ == "4.11.0", "qualified array/flow runtime")
        sys.path.insert(0, str(here))
        import features
        import core
        import models
        metrics = module("d37_metrics", d35/"metrics.py")
        restore = module("d37_restore", d35/"evaluate.py")
        producer = context["prediction_receipt"]
        index = load(context["prediction_spec"]["data_index"]["path"])
        require([(r["sequence"], r["frames"]) for r in producer["rows"]] == list(COHORT), "whole fixed cohort")
        record.update(feature_names=list(core.FEATURE_NAMES), tracking_columns=list(core.TRACK_COLUMNS),
                      inherited_native_seconds=686.546369958, all_clips_exposed=True,
                      recording_independence_established=False, input_bindings=context["bindings"])
        frame_rows = []
        phase_start = time.monotonic()
        checkpoint("features")
        for row, data in zip(producer["rows"], index["sequences"]):
            name = row["sequence"]
            require(name == data["sequence"] and row["object_ids"] == [1], "complete single initial object")
            folder = out/name
            folder.mkdir()
            state = previous = None
            for frame, (desc, native_record, rgb_desc) in enumerate(zip(row["predictions"], row["frame_records"], data["rgb"])):
                require(rgb_desc["frame"] == frame and sha(rgb_desc["path"]) == rgb_desc["sha256"], "RGB bytes")
                with Image.open(rgb_desc["path"]) as im:
                    require(im.format == "JPEG" and im.mode == "RGB", "RGB JPEG input")
                    rgb = np.asarray(im).copy()
                require(tuple(rgb.shape[:2]) == tuple(row["shape"]), "every RGB preserves admitted dimensions")
                path = restore.frame_path(context["prediction_root"], desc)
                logits, controls = restore.reconstruct_frame(path, row, native_record, np=np, torch=torch, metrics=metrics)
                if frame == 0:
                    gtpath = context["annotation_paths"][name][0]
                    require(sha(gtpath) == context["fingerprints"][str(gtpath)], "initial prompt bytes")
                    with Image.open(gtpath) as im:
                        initial = np.asarray(im).copy()
                    state, previous, evidence = features.initialize(rgb, initial)
                    record["initial_label_decodes"] += 1
                else:
                    state, previous, evidence = features.advance(previous, rgb, state)
                    record["flow_calls"] += 2
                x, ids, union_counts, metadata = core.build_features(previous, evidence, logits[0], controls, frame)
                require(np.isfinite(x).all() and x.dtype == np.float32 and x.shape[1] == len(core.FEATURE_NAMES), "finite declared feature matrix")
                cache = folder/f"features-{frame:05d}.npz"
                np.savez_compressed(cache, X=x, union_counts=union_counts)
                saved = artifact(cache, "causal-cell-features")
                frame_rows.append(dict(sequence=name, frame=frame, shape=row["shape"], small_shape=list(previous.shape[:2]),
                    feature=saved, native_descriptor=desc, metadata=metadata, transport=evidence["metadata"],
                    coverage={k: float(np.mean(state == v)) for k,v in (("unknown",0),("foreground",1),("background",2))}))
                record["completed_feature_frames"] += 1
                if frame % 20 == 0:
                    checkpoint("features")
                del logits, controls, rgb, x, ids
        require(record["completed_feature_frames"] == 446 and record["flow_calls"] == 884
                and record["noninitial_label_decodes"] == 0, "complete causal features before metric labels")
        save(out/"feature-seal.json", dict(complete=True, frames=446, noninitial_label_decodes=0,
             source_sha256=spec["source_sha256"], features=[r["feature"] for r in frame_rows],
             monotonic_seconds=time.monotonic()-started))
        artifact(out/"feature-seal.json", "feature-seal")
        record["phase_seconds"]["features"] = time.monotonic()-phase_start

        # These labels train other clips' folds. They never enter feature construction.
        phase_start = time.monotonic()
        checkpoint("training-targets")
        training = {name: dict(x=[], y=[], w=[], frame=[]) for name,_ in COHORT}
        fixed_scores = {p: {name: [] for name,_ in COHORT} for p in POLICIES}
        for r in frame_rows:
            name, frame = r["sequence"], r["frame"]
            feature = read_npz(r["feature"])
            native_path = restore.frame_path(context["prediction_root"], r["native_descriptor"])
            with np.load(native_path, allow_pickle=False) as f:
                labels = f["labels"]
            controls = {a: labels[i] for i,a in enumerate(FIXED)}
            gtpath = context["annotation_paths"][name][frame]
            require(sha(gtpath) == context["fingerprints"][str(gtpath)], "training annotation bytes")
            with Image.open(gtpath) as im:
                truth = np.asarray(im).copy()
            if frame:
                record["noninitial_label_decodes"] += 1
            else:
                record["initial_label_decodes"] += 1
            ids = features.cell_ids(tuple(r["shape"]), tuple(r["small_shape"]))
            counts = core.label_counts(truth, controls, ids, n_cells=len(feature["X"]))
            flattened = {p+"__"+k: np.asarray(v) for p in POLICIES for k,v in counts[p].items()}
            path = out/name/f"targets-{frame:05d}.npz"
            np.savez_compressed(path, **flattened)
            r["targets"] = artifact(path, "cross-fit-target-counts")
            for p, oldkey in (("author_background", "primary"), ("void_ignore", "void_ignore_sensitivity")):
                c = counts[p]
                scores = {a: [float(c["fixed_tp"][i]/(c["total_gt"]+c["fixed_fp"][i]))
                             if c["total_gt"]+c["fixed_fp"][i] else 1.0] for i,a in enumerate(FIXED)}
                oldrow = next(v for v in oldresult[oldkey]["per_sequence"] if v["sequence"] == name)
                require(all(scores[a][0] == oldrow["methods"][a]["per_object"][0]["frame_J"][frame]
                            for a in FIXED), "exact D35 original-resolution fixed-control parity")
                fixed_scores[p][name].append(scores)
            if frame:
                chosen = core.sample_indices(feature["union_counts"])
                if len(chosen):
                    c = counts["author_background"]
                    areas = feature["union_counts"][chosen].astype(np.float64)
                    require(np.all(areas > 0), "only nonempty union cells sampled")
                    nframes = dict(COHORT)[name]-1
                    training[name]["x"].append(feature["X"][chosen])
                    training[name]["y"].append((c["union_tp"][chosen]/areas).astype(np.float32))
                    training[name]["w"].append(areas/math.fsum(areas)/nframes)
                    training[name]["frame"].append(np.full(len(chosen), frame, np.int64))
            record["completed_target_frames"] += 1
        for name,_ in COHORT:
            values = training[name]
            if values["x"]:
                training[name] = {k: np.concatenate(v) for k,v in values.items()}
            else:
                training[name] = dict(x=np.empty((0,len(core.FEATURE_NAMES)),np.float32),
                    y=np.empty(0,np.float32), w=np.empty(0,np.float64), frame=np.empty(0,np.int64))
        record["phase_seconds"]["training_targets"] = time.monotonic()-phase_start

        phase_start = time.monotonic()
        checkpoint("fitting")
        modeldir = out/"models"
        modeldir.mkdir()
        fitted = {}
        for held,_ in COHORT:
            sources, x, y, w = fold_samples(training, held, np)
            fitted[held] = {}
            for arm in LEARNED:
                current = x.copy()
                if arm == "no_ancestry_mlp":
                    current[:, core.TRACK_COLUMNS] = 0
                active = x[:,19] >= .75 if arm == "negative_gate_mlp" else None
                model, trace = models.fit(current, y, w, active=active)
                model.update(excluded_clip=held, training_clips=sources, arm=arm,
                    feature_names=list(core.FEATURE_NAMES), selected_frame_counts={s: len(set(training[s]["frame"].tolist())) for s in sources})
                trace.update(excluded_clip=held, training_clips=sources, arm=arm)
                for kind, value in (("model",model),("trace",trace)):
                    path = modeldir/f"{held}-{arm}-{kind}.json"
                    save(path, value)
                    artifact(path, "fit-"+kind)
                fitted[held][arm] = model
                record["completed_fits"] += 1
                checkpoint("fitting")
        require(record["completed_fits"] == 12, "all four folds and three learned arms")
        del training, x, y, w, current
        record["phase_seconds"]["fitting"] = time.monotonic()-phase_start

        phase_start = time.monotonic()
        checkpoint("prediction")
        for r in frame_rows:
            x = read_npz(r["feature"])["X"]
            actions = core.actions_fixed(x)
            for arm in LEARNED:
                current = x.copy()
                if arm == "no_ancestry_mlp":
                    current[:,core.TRACK_COLUMNS] = 0
                probability = models.predict(fitted[r["sequence"]][arm], current)
                require(probability.shape == (len(x),) and np.isfinite(probability).all(), "complete finite prediction")
                keep = probability >= .5
                if arm == "negative_gate_mlp":
                    keep = keep | (x[:,19] < .75)
                actions[arm] = (keep.copy(), keep.copy())
            if r["frame"] == 0:
                actions = {a: (np.ones(len(x),bool), np.ones(len(x),bool)) for a in LOCAL}
            require(set(actions) == set(LOCAL), "all local arms")
            path = out/r["sequence"]/f"prediction-{r['frame']:05d}.npz"
            np.savez_compressed(path, keep=np.stack([np.stack(actions[a]) for a in LOCAL]))
            r["prediction"] = artifact(path, "cell-predictions")
            record["completed_prediction_frames"] += 1
        require(record["completed_prediction_frames"] == 446, "complete prediction population")
        save(out/"prediction-seal.json", dict(complete=True, frames=446, arms=list(LOCAL),
            predictions=[r["prediction"] for r in frame_rows], source_sha256=spec["source_sha256"],
            own_fold_targets_passed_to_predict=False, monotonic_seconds=time.monotonic()-started))
        artifact(out/"prediction-seal.json", "prediction-seal")
        record["phase_seconds"]["prediction"] = time.monotonic()-phase_start

        phase_start = time.monotonic()
        checkpoint("evaluation")
        scores = {p: {name: [] for name,_ in COHORT} for p in POLICIES}
        removals = []
        for r in frame_rows:
            packed = read_npz(r["targets"])
            keep = read_npz(r["prediction"])["keep"]
            require(keep.dtype == bool and keep.shape[:2] == (len(LOCAL),2), "declared local actions")
            for policy in POLICIES:
                counts = unpack_counts(packed, policy)
                values = dict(fixed_scores[policy][r["sequence"]][r["frame"]])
                details = {}
                for i,arm in enumerate(LOCAL):
                    result = core.score_actions(counts, keep[i,0], keep[i,1])
                    values[arm] = [result["J"]]
                    details[arm] = result
                scores[policy][r["sequence"]].append(values)
                removals.append(dict(sequence=r["sequence"], frame=r["frame"], policy=policy, arms=details))
            record["completed_metric_frames"] += 1
        reports = {p: metrics.aggregate_sequences([metrics.summarize_sequence(
            name, scores[p][name], [1], void_policy=p) for name,_ in COHORT]) for p in POLICIES}
        report = dict(schema="evr0037.local-comparison.v1", primary=reports["author_background"],
            void_ignore_sensitivity=reports["void_ignore"], continuation=continuation(reports["author_background"]),
            sensitivity_continuation=continuation(reports["void_ignore"]), removals=removals,
            all_four_clips_exposed_development=True, recording_independence_established=False,
            novelty_established=False, final_paper_gate_met=False)
        save(out/"report.json", report)
        artifact(out/"report.json", "complete-evaluation")
        save(out/"frame-inventory.json", frame_rows)
        artifact(out/"frame-inventory.json", "complete-frame-inventory")
        record["phase_seconds"]["evaluation"] = time.monotonic()-phase_start
        checkpoint("postcheck")
        admission.postcheck(context)
        require(all(sha(Path(context["prediction_root"])/d["path"]) == d["sha256"]
                    for row in producer["rows"] for d in row["predictions"]), "all native cache bytes unchanged")
        require(all(sha(p) == h for p,h in spec["source_sha256"].items()) and sha(spec_path) == spec_hash,
                "source/spec unchanged")
        require(all(sha(out/d["path"]) == d["sha256"] for d in record["outputs"]), "all outputs unchanged")
        require(not record["offline_attempts"] and record["completed_metric_frames"] == 446
                and record["completed_target_frames"] == 446 and record["noninitial_label_decodes"] == 442
                and record["initial_label_decodes"] == 8,
                "complete offline evaluation")
        record.update(status="completed", phase="complete", scientific_eligible_pending_outer=True,
            numerical_passed=report["continuation"]["numerical_passed"], source_input_postcheck=True)
    except BaseException as exc:
        record.update(status="failed", error=dict(type=type(exc).__name__, message=str(exc)))
        if hasattr(exc, "trace"):
            save(out/"failed-fit-trace.json", exc.trace)
        if getattr(exc, "model", None) is not None:
            save(out/"failed-fit-partial-model.json", exc.model)
        raise
    finally:
        record["wall_seconds"] = time.monotonic()-started
        save(out/"receipt.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    result = run(parser.parse_args().spec)
    print(json.dumps({k: result[k] for k in ("status", "wall_seconds", "completed_metric_frames", "numerical_passed")}))
