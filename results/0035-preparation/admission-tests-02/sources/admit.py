"""Byte-bound all-four prediction barrier; stdlib only, no model or array reads."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

COHORT = (("4691_cut_guava", 86), ("817_cut_onion", 90),
          ("9675_fold_clothes", 201), ("1934_tear_film", 69))
MANIFEST_SHA256 = "141371225d59c0b1ef483ee3b7565aace037033df072eabb2839e8d4f19bfb07"
SUPERVISOR_SHA256 = "8f7bcbb8b7d31031c709f72ea2f49da1913ced7230154e45e0d165d00457c9a8"
MONITOR_SHA256 = "4b8b0efa4fd713a032967c9b34539eac128184ef6db4e8535e83d785c6bbebcd"
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
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


def bound_json(item, fingerprints):
    path = Path(item["path"])
    require(path.is_absolute() and path.is_file() and not path.is_symlink()
            and path.stat().st_size <= 32 * 1024**2, "bounded regular JSON input")
    require(sha(path) == item["sha256"], "JSON input changed: " + str(path))
    fingerprints[str(path)] = item["sha256"]
    return json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def relative(root, name):
    path = Path(name)
    require(not path.is_absolute() and path.parts and all(p not in (".", "..") for p in path.parts),
            "safe relative artifact")
    target = Path(root) / path
    require(target.resolve().is_relative_to(Path(root).resolve()), "artifact escapes root")
    return target


def clean_outer(item, expected_output, fingerprints):
    outer = bound_json(item, fingerprints)
    require(outer["schema"] == "evr0029.exit-checked-operation.v1"
            and outer["supervisor_sha256"] == SUPERVISOR_SHA256
            and outer["monitor_sha256"] == MONITOR_SHA256
            and outer["thread_environment"] == {key: "4" for key in THREAD_KEYS}
            and type(outer["hard_timeout_seconds"]) in (int, float)
            and math.isfinite(outer["hard_timeout_seconds"]) and 0 < outer["hard_timeout_seconds"] <= 1800
            and outer["group_rss_limit_kib"] == 8 * 1024 * 1024
            and outer["status"] == "completed" and outer["exit_code"] == 0
            and outer["termination"] is None and not outer["remaining_descendant_pids"]
            and not outer["cleanup_errors"]
            and type(outer["wall_seconds"]) in (int, float)
            and math.isfinite(outer["wall_seconds"]) and 0 <= outer["wall_seconds"] < 1800
            and type(outer["peak_group_rss_kib"]) is int
            and 0 <= outer["peak_group_rss_kib"] <= 8 * 1024 * 1024
            and set(outer["log_sha256"]) == {"stdout.log", "stderr.log"}
            and outer["output_sha256"] == expected_output, "clean complete outer operation required")
    root = Path(item["path"]).parent
    for name, digest in outer["log_sha256"].items():
        p = relative(root, name)
        require(p.is_file() and sha(p) == digest, "outer log changed")
        fingerprints[str(p)] = digest
    return outer


def acquisition(item, outer_item, phase, manifest, fingerprints, repo):
    record = bound_json(item, fingerprints)
    clean_outer(outer_item, item["sha256"], fingerprints)
    require(record["schema"] == "evr0035.vost-acquisition.v1" and record["status"] == "completed"
            and record["phase"] == phase and record["index_sha256"]["manifest"] == MANIFEST_SHA256
            and record["retry_count"] == 0 and record["hidden_retries"] is False
            and record["full_archive_fallback"] is False and record["media_decoded"] is False,
            "complete frozen phase acquisition")
    wanted = [r for r in manifest["members"] if r["phase"] == phase]
    require([r["filename"] for r in record["members"]] == [r["filename"] for r in wanted], "exact acquired population")
    root = Path(item["path"]).parent
    output = {}
    for saved, expected in zip(record["members"], wanted):
        require(all(saved[k] == v for k, v in expected.items()), "acquired metadata changed")
        p = relative(root, saved["saved_relative"])
        require(not p.is_symlink() and p.stat().st_size == saved["bytes"] == saved["uncompressed_bytes"]
                and sha(p) == saved["sha256"], "acquired member changed")
        fingerprints[str(p)] = saved["sha256"]
        output[(saved["sequence"], saved["frame_index"])] = dict(
            path=str(p), frame=saved["frame_index"], bytes=saved["bytes"], sha256=saved["sha256"])
    for name, digest in record["source_sha256"].items():
        p = relative(repo, name)
        require(sha(p) == digest, "acquisition source changed")
        fingerprints[str(p)] = digest
    return record, output


def verify_prediction_proof(path, expected_manifest_sha256=MANIFEST_SHA256, *, check_outputs=True):
    """The acquired-metric caller binds proof SHA; this checks every referenced seal."""
    path = Path(path).absolute()
    fingerprints = {str(path): sha(path)}
    proof = json.loads(path.read_text())
    require(proof["schema"] == "evr0035.prediction-proof.v1", "prediction proof schema")
    require(proof["manifest"]["sha256"] == expected_manifest_sha256 == MANIFEST_SHA256,
            "fixed manifest required")
    manifest = bound_json(proof["manifest"], fingerprints)
    require([(c["sequence"], c["frames"]) for c in manifest["clips"]] == list(COHORT), "fixed cohort")
    for source, digest in proof["source_sha256"].items():
        require(sha(source) == digest, "proof source changed")
        fingerprints[source] = digest
    qualification = bound_json(proof["qualification_receipt"], fingerprints)
    require(qualification["status"] == "completed" and qualification["eligible"] is True
            and qualification["manufactured_only"] is True
            and qualification["scientific_input_access"] is False
            and qualification["full_state_parity"] is True
            and len(qualification["native_output_parities"]) == 7
            and all(r["exact"] for r in qualification["native_output_parities"]),
            "complete manufactured native qualification")
    clean_outer(proof["qualification_outer"], proof["qualification_receipt"]["sha256"], fingerprints)
    spec = bound_json(proof["predict_spec"], fingerprints)
    receipt = bound_json(proof["predict_receipt"], fingerprints)
    seal = bound_json(proof["predict_seal"], fingerprints)
    clean_outer(proof["predict_outer"], proof["predict_receipt"]["sha256"], fingerprints)
    require(spec["source_sha256"] == receipt["source_sha256"] == seal["source_sha256"]
            == proof["source_sha256"] == qualification["source_sha256"], "one frozen scientific source closure")
    require(receipt["schema"] == "evr0035.native-bank.v1" and receipt["status"] == "completed"
            and receipt["scientific_eligible"] is True and receipt["completed_frames"] == 446
            and receipt["completed_sequences"] == 4 and receipt["noninitial_label_decodes"] == 0
            and len(receipt["initial_label_decodes"]) == 4
            and receipt["spec_sha256"] == proof["predict_spec"]["sha256"], "complete prediction receipt")
    require(seal["schema"] == "evr0035.native-bank-seal.v1" and seal["complete"] is True
            and seal["receipt_sha256"] == proof["predict_receipt"]["sha256"]
            and seal["spec_sha256"] == receipt["spec_sha256"]
            and seal["outputs"] == receipt["outputs"]
            and seal["completed_frames"] == 446 and seal["completed_sequences"] == 4,
            "complete prediction seal")
    index = bound_json(spec["data_index"], fingerprints)
    require(seal["data_index"] == receipt["data_index"] == spec["data_index"], "bound data index")
    require([(r["sequence"], len(r["rgb"])) for r in index["sequences"]] == list(COHORT), "complete data index")
    initial_acq, initial_files = acquisition(proof["initial_acquisition"], proof["initial_outer"],
                                             "initial", manifest, fingerprints, spec["repo"])
    rgb_acq, rgb_files = acquisition(proof["rgb_acquisition"], proof["rgb_outer"],
                                     "rgb", manifest, fingerprints, spec["repo"])
    require(rgb_acq["initial_admission"]["receipt_sha256"] == proof["initial_acquisition"]["sha256"],
            "RGB follows the same initial acquisition")
    inspection = bound_json(proof["initial_inspection"], fingerprints)
    clean_outer(proof["initial_inspection_outer"], proof["initial_inspection"]["sha256"], fingerprints)
    require(inspection["status"] == "completed" and inspection["eligible"] is True
            and inspection["initial_only"] is True and inspection["noninitial_labels_read"] == 0
            and inspection["model_calls"] == 0 and all(inspection["checks"].values())
            and inspection["prediction_output_cap_bytes"] == spec["max_output_bytes"], "initial-only resource admission")
    require(inspection["source_sha256"] == proof["source_sha256"], "initial inspection source closure")
    require(inspection["fingerprints"].get(proof["initial_acquisition"]["path"])
            == proof["initial_acquisition"]["sha256"]
            and rgb_acq["initial_admission"]["inspection_sha256"] == proof["initial_inspection"]["sha256"]
            and rgb_acq["initial_admission"]["inspection_outer_sha256"] == proof["initial_inspection_outer"]["sha256"],
            "resource inspection belongs to this initial acquisition and RGB phase")
    for sequence in index["sequences"]:
        name = sequence["sequence"]
        require(all(sequence["initial"][k] == v for k, v in initial_files[(name, 0)].items()), "initial prompt provenance")
        for desc in sequence["rgb"]:
            require(all(desc[k] == v for k, v in rgb_files[(name, desc["frame"])].items()), "RGB provenance")
    require([(r["sequence"], r["frames"]) for r in receipt["rows"]] == list(COHORT), "complete prediction rows")
    flat = []
    object_frames = 0
    for row, (_, count) in zip(receipt["rows"], COHORT):
        ids = row["object_ids"]
        require(ids == list(range(1, len(ids)+1)) and ids and len(ids) <= 254
                and row["status"] == "completed" and row["completed_frames"] == count
                and len(row["frame_records"]) == count, "all initial IDs and frames")
        require([r["frame"] for r in row["frame_records"]] == list(range(count))
                and all(r["selected_low_exact"] and r["selected_original_exact"] for r in row["frame_records"]),
                "complete native parities")
        require([(p["sequence"], p["frame"]) for p in row["predictions"]]
                == [(row["sequence"], i) for i in range(count)], "complete prediction inventory")
        object_frames += count * len(ids)
        flat.extend(row["predictions"])
        inspected = next(r for r in inspection["rows"] if r["sequence"] == row["sequence"])
        require(row["shape"] == inspected["shape"] and ids == inspected["object_ids"], "initial IDs/dimensions changed")
    require(flat == receipt["outputs"] and len({p["path"] for p in flat}) == 446, "exact flat output inventory")
    for kind, expected in (("image_encoder", 446), ("track_step", object_frames),
                           ("mask_decoder", object_frames), ("memory_encoder", object_frames)):
        require(receipt["calls"][kind+"_intent"] == receipt["calls"][kind+"_completed"] == expected,
                "native call accounting")
    root = Path(spec["output"])
    require(root == Path(proof["predict_receipt"]["path"]).parent == Path(receipt["output"]), "prediction root")
    for output in flat:
        p = relative(root, output["path"])
        require(p.is_file() and not p.is_symlink() and p.stat().st_size == output["bytes"], "prediction file inventory")
        if check_outputs:
            require(sha(p) == output["sha256"], "prediction byte seal changed")
    return dict(eligible=True, manifest_sha256=MANIFEST_SHA256, proof_sha256=fingerprints[str(path)],
                proof=proof, manifest=manifest, prediction_receipt=receipt, prediction_root=str(root),
                prediction_spec=spec, fingerprints=fingerprints, object_frames=object_frames,
                output_bytes=sum(p["bytes"] for p in flat), outputs_verified=check_outputs)


def postcheck(context):
    require(all(sha(path) == digest for path, digest in context["fingerprints"].items()),
            "bound source/receipt/metadata changed during operation")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def evaluation_context(spec):
    """No arrays/model; require all-four clean seals before reading metric media."""
    fingerprints = {}
    proof = bound_json(spec["prediction_proof"], fingerprints)
    context = verify_prediction_proof(spec["prediction_proof"]["path"], check_outputs=False)
    fingerprints.update(context["fingerprints"])
    require(spec["source_sha256"] == proof["source_sha256"], "evaluation uses frozen scientific source closure")
    for key in ("initial_acquisition", "initial_outer"):
        require(spec[key] == proof[key], "same initial annotation acquisition")
    initial, initial_paths = acquisition(spec["initial_acquisition"], spec["initial_outer"],
                                         "initial", context["manifest"], fingerprints, spec["repo"])
    metric, metric_paths = acquisition(spec["metric_acquisition"], spec["metric_outer"],
                                       "metric", context["manifest"], fingerprints, spec["repo"])
    require(metric["gt_admission"]["eligible"] is True
            and metric["gt_admission"]["prediction_proof_sha256"] == spec["prediction_proof"]["sha256"]
            and metric["gt_admission"]["manifest_sha256"] == MANIFEST_SHA256
            and metric["initial_admission"]["receipt_sha256"] == spec["initial_acquisition"]["sha256"],
            "metric transport followed the exact complete prediction proof")
    annotations = {name: {} for name, _ in COHORT}
    for (name, frame), desc in {**initial_paths, **metric_paths}.items():
        require(frame not in annotations[name], "duplicate annotation ordinal")
        annotations[name][frame] = desc["path"]
    require(all(sorted(annotations[name]) == list(range(count)) for name, count in COHORT), "complete metric labels")
    root = Path(spec["repo"])
    h = module("eval_installed_admission", root / "experiments/0029-memory-information/install_runtime.py")
    old = module("eval_native_admission", root / "experiments/0029-memory-information/run_evidence.py")
    installation, _, binding = old.native_contract(h, root, Path(spec["native_root"]), spec["native_commit"])
    fingerprints.update(binding["file_sha256"])
    context.update(installation=installation, initial_acquisition=initial, metric_acquisition=metric,
                   annotation_paths=annotations, fingerprints=fingerprints,
                   bindings={"prediction_proof": spec["prediction_proof"], "native": binding})
    return context
