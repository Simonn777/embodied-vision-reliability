"""D38 saved prediction evaluation inside ONE complete operation.

The caller admits the unchanged D35 context before its permanent offline guard.
That admission may hash opaque annotation bytes; it decodes no labels. This
helper requires all new predictions, native/input/final-state parities and the
immutable prediction seal before any archive or numerical GT decoding here.
The current operation has no completed outer yet: final eligibility requires
its separately verified clean outer. No model/production imports on import.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

COHORT = (("4691_cut_guava", 86), ("817_cut_onion", 90),
          ("9675_fold_clothes", 201), ("1934_tear_film", 69))
FIXED = ("B", "B+M1", "B+M2", "B+M1+M2")
READOUTS = ("early_mean", "flat_histories", "late_mean", "output_ensemble")
ARMS = FIXED + ("fixed_negative",) + READOUTS
CONTROLS = tuple(a for a in ARMS if a != "late_mean")
POLICIES = ("author_background", "void_ignore")
REPORT_KEYS = ("primary", "void_ignore_sensitivity")
NPZ_KEYS = {"low_logits", "packed_masks", "object_ids", "shape"}
D35_SPEC_SHA = "325961b52bed72c8710fee107265b02545002f9b7591fdd4119095841611a68d"
D35_RESULT_SHA = "7fe9592298d1cced864eaa6f75faf973e98c10ea3ce8c91100a678cc6cd60fe3"
REQUIRED_LOCAL_SOURCES = ("evaluate_complete.py", "run_complete.py", "ADMISSION.md")


def require(ok, message):
    if not ok:
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
            "bounded regular JSON")
    return json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def bound(desc, fingerprints):
    path = Path(desc["path"])
    require(path.is_absolute() and sha(path) == desc["sha256"], "bound JSON changed")
    fingerprints[str(path)] = desc["sha256"]
    return load(path)


def artifact(root, desc, fingerprints):
    root, relative = Path(root).resolve(), Path(desc["path"])
    path = (root / relative).resolve()
    require(not relative.is_absolute() and ".." not in relative.parts and path.is_relative_to(root)
            and not (root / relative).is_symlink() and path.is_file(), "safe regular artifact")
    require(type(desc["bytes"]) is int and desc["bytes"] > 0
            and path.stat().st_size == desc["bytes"] and sha(path) == desc["sha256"],
            "artifact bytes changed")
    fingerprints[str(path)] = desc["sha256"]
    return path


def module(name, path):
    definition = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(definition)
    sys.modules[name] = value
    definition.loader.exec_module(value)
    return value


def validate_population(prediction, seal, old_producer, input_index):
    """Metadata-only complete barrier, including every endpoint and parity flag."""
    require(prediction["schema"] == "evr0038.complete-prediction.v1"
            and prediction["status"] == "completed"
            and prediction["frames"] == prediction["completed_frames"] == 446
            and prediction["completed_sequences"] == 4 and prediction["arms"] == list(READOUTS),
            "complete nine-arm input population")
    require(seal["schema"] == "evr0038.prediction-seal.v1" and seal["complete"] is True
            and seal["frames"] == 446 and seal["sequences"] == 4
            and all(seal[k] == prediction[k] for k in
                    ("outputs", "arms", "source_sha256", "spec_sha256", "bindings")),
            "immutable all-four prediction seal")
    require([(r["sequence"], r["frames"]) for r in prediction["sequences"]] == list(COHORT)
            and [(r["sequence"], r["frames"]) for r in old_producer["rows"]] == list(COHORT)
            and [(r["sequence"], len(r["rgb"])) for r in input_index["sequences"]] == list(COHORT),
            "same complete original clips")
    wanted = [(name, frame) for name, count in COHORT for frame in range(count)]
    require([(r["sequence"], r["frame"]) for r in prediction["rows"]] == wanted,
            "all446 unique chronological rows")
    old = {r["sequence"]: r for r in old_producer["rows"]}
    index = {r["sequence"]: r for r in input_index["sequences"]}
    for seq in prediction["sequences"]:
        ref = old[seq["sequence"]]
        require(seq["status"] == "completed" and seq["completed_frames"] == seq["frames"]
                and seq["object_ids"] == ref["object_ids"] == [1]
                and seq["shape"] == ref["shape"] and seq["final_state_exact"] is True
                and seq["final_native_state_sha256"] == seq["expected_final_native_state_sha256"]
                    == ref["final_native_state_sha256"],
                "all native final-state/input identities")
    for row in prediction["rows"]:
        ref = old[row["sequence"]]
        require(row["object_ids"] == ref["object_ids"] and row["shape"] == ref["shape"]
                and row["filename"] == Path(index[row["sequence"]]["rgb"][row["frame"]]["path"]).name
                and all(row["native_parity"].get(k) is True for k in
                        ("bank_exact", "output_exact", "input_exact", "controls_exact", "stored_write_exact")),
                "per-frame native/input parity")
    outputs = [r["output"] for r in prediction["rows"]]
    require(outputs == prediction["outputs"] and len({d["path"] for d in outputs}) == 446,
            "complete unique archive inventory")


def reference_scores(d35, d37):
    """Validate identity/policy/complete scalar reference; never select by GT."""
    result = {}
    for policy, key in zip(POLICIES, REPORT_KEYS):
        old, prior = d35[key], d37[key]
        require(old["void_policy"] == prior["void_policy"] == policy
                and old["sequence_count"] == prior["sequence_count"] == 4
                and old["object_count"] == prior["object_count"] == 4, "reference metric population")
        require([(r["sequence"], r["frame_count"]) for r in old["per_sequence"]] == list(COHORT)
                and [(r["sequence"], r["frame_count"]) for r in prior["per_sequence"]] == list(COHORT),
                "reference complete frame populations")
        result[policy] = {}
        for a, b in zip(old["per_sequence"], prior["per_sequence"]):
            require(a["object_ids"] == b["object_ids"] == [1]
                    and a["frame_positions"] == b["frame_positions"], "same reference IDs/endpoints")
            values = {}
            for arm in FIXED + ("fixed_negative",):
                obj = b["methods"][arm]["per_object"]
                require(len(obj) == 1 and obj[0]["object_id"] == 1, "reference object identity")
                raw = obj[0]["frame_J"]
                require(len(raw) == a["frame_count"] and all(type(v) in (int, float)
                        and math.isfinite(v) and 0 <= v <= 1 for v in raw), "finite complete reference scores")
                if arm in FIXED:
                    require(raw == a["methods"][arm]["per_object"][0]["frame_J"], "D37/D35 fixed parity")
                values[arm] = raw
            result[policy][a["sequence"]] = values
    return result


def continuation(primary):
    methods = primary["methods"]
    require(set(methods) == set(ARMS) and primary["sequence_count"] == primary["object_count"] == 4,
            "nine arms and four sequence-objects")
    require([(r["sequence"], r["frame_count"]) for r in primary["per_sequence"]] == list(COHORT),
            "complete gate cohort")
    strongest = {m: max(CONTROLS, key=lambda a: methods[a][m]) for m in ("J", "Jtr")}
    gains = {m: methods["late_mean"][m] - methods[strongest[m]][m] for m in strongest}
    per_clip = {r["sequence"]: {a: {m: r["methods"]["late_mean"][m]-r["methods"][a][m]
                for m in ("J", "Jtr")} for a in ("B", "B+M1+M2")}
                for r in primary["per_sequence"]}
    clauses = {"over_every_control_3pp_"+m: gains[m] >= .03 for m in ("J", "Jtr")}
    clauses.update(three_clips_positive_Jtr_over_union=sum(
        v["B+M1+M2"]["Jtr"] > 0 for v in per_clip.values()) >= 3,
        no_clip_native_loss_over_0_5pp=all(v["B"][m] >= -.005
            for v in per_clip.values() for m in ("J", "Jtr")))
    return dict(clauses=clauses, numerical_passed=all(clauses.values()),
        strongest_control_by_metric=strongest, gains_over_strongest_control=gains,
        per_clip_gains=per_clip, control_tie_order=list(CONTROLS),
        clean_current_outer_and_independent_review_required=True,
        interpretation="exposed-development shadow readout screen, not a paper or recurrent-policy result")


def reconstruct_readouts(path, row, *, np, torch):
    """Exact native O×1 resize per arm, checked against all packed bits."""
    shape, ids = tuple(row["shape"]), row["object_ids"]
    require(ids == [1] and len(shape) == 2 and min(shape) > 0, "qualified one-object original shape")
    pixels = shape[0]*shape[1]
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == NPZ_KEYS, "exact readout NPZ schema")
        saved = {key: archive[key] for key in NPZ_KEYS}
    expected = {"low_logits": (np.dtype("float32"), (4, 1, 256, 256)),
        "packed_masks": (np.dtype("uint8"), (4, 1, (pixels+7)//8)),
        "object_ids": (np.dtype("int64"), (1,)), "shape": (np.dtype("int64"), (2,))}
    for key, (dtype, dimensions) in expected.items():
        require(saved[key].dtype == dtype and saved[key].shape == dimensions
                and np.isfinite(saved[key]).all(), "finite dtype/shape: "+key)
    require(saved["object_ids"].tolist() == ids and saved["shape"].tolist() == list(shape), "archive identity")
    bits = np.unpackbits(saved["packed_masks"], axis=-1, bitorder="big")
    require(not bits[..., pixels:].any(), "zero packed padding bits")
    predictions = {}
    with torch.inference_mode():
        for i, arm in enumerate(READOUTS):
            low = torch.from_numpy(saved["low_logits"][i]).reshape(1, 1, 256, 256).contiguous()
            original = low if shape == (256, 256) else torch.nn.functional.interpolate(
                low, size=shape, mode="bilinear", align_corners=False)
            mask = original[0, 0].numpy() > 0
            require(np.array_equal(mask, bits[i, 0, :pixels].reshape(shape).astype(bool)),
                    "sealed readout mask differs from original-layout resize")
            predictions[arm] = mask.astype(np.int64)
    return predictions


def validate_d35_receipt(spec, oldspec, receipt):
    """The historical outer seals the receipt, not its evaluation.json child."""
    require(Path(spec["d35_receipt"]["path"]) == Path(spec["d35_result"]["path"]).parent/"receipt.json"
            and receipt["schema"] == "evr0035.evaluation.v1" and receipt["status"] == "completed"
            and receipt["spec_sha256"] == spec["d35_spec"]["sha256"]
            and receipt["source_sha256"] == oldspec["source_sha256"]
            and receipt["completed_frames"] == 446 and receipt["completed_sequences"] == 4,
            "complete original D35 evaluation receipt")
    outputs = receipt["outputs"]
    require(len(outputs) == 1 and outputs[0]["path"] == "evaluation.json"
            and outputs[0]["sha256"] == spec["d35_result"]["sha256"]
            and type(outputs[0]["bytes"]) is int and outputs[0]["bytes"] > 0,
            "D35 receipt binds its sole complete evaluation output")


def admit_references(spec, admission, fingerprints):
    """Only old complete scalar reports, their clean outers and review proof."""
    require(spec["d35_spec"]["sha256"] == D35_SPEC_SHA
            and spec["d35_result"]["sha256"] == D35_RESULT_SHA, "fixed D35 identities")
    oldspec, oldresult = bound(spec["d35_spec"], fingerprints), bound(spec["d35_result"], fingerprints)
    d35receipt = bound(spec["d35_receipt"], fingerprints)
    admission.clean_outer(spec["d35_outer"], spec["d35_receipt"]["sha256"], fingerprints)
    validate_d35_receipt(spec, oldspec, d35receipt)
    worker, receipt = bound(spec["d37_spec"], fingerprints), bound(spec["d37_receipt"], fingerprints)
    admission.clean_outer(spec["d37_outer"], spec["d37_receipt"]["sha256"], fingerprints)
    require(receipt["status"] == "completed" and receipt["phase"] == "complete"
            and receipt["spec_sha256"] == spec["d37_spec"]["sha256"]
            and receipt["source_input_postcheck"] is True and not receipt["offline_attempts"]
            and all(receipt[k] == 446 for k in ("completed_feature_frames", "completed_target_frames",
                      "completed_prediction_frames", "completed_metric_frames"))
            and receipt["completed_fits"] == 12 and worker["d35_spec"] == spec["d35_spec"]
            and worker["d35_result"] == spec["d35_result"], "complete matching D37 reference")
    root = Path(worker["output"])
    require(Path(spec["d37_receipt"]["path"]).parent == root, "D37 receipt root")
    descriptors = {d["path"]: d for d in receipt["outputs"]}
    report_path = artifact(root, descriptors["report.json"], fingerprints)
    sealed = bound(spec["d37_seal"], fingerprints)
    require(Path(spec["d37_seal"]["path"]) == root/"prediction-seal.json"
            and spec["d37_seal"]["sha256"] == descriptors["prediction-seal.json"]["sha256"]
            and sealed["complete"] is True and sealed["frames"] == 446
            and "fixed_negative" in sealed["arms"]
            and sealed["source_sha256"] == receipt["source_sha256"] == worker["source_sha256"],
            "D37 sealed whole-population reference")
    review_receipt = bound(spec["d37_review"], fingerprints)
    admission.clean_outer(spec["d37_review_outer"], spec["d37_review"]["sha256"], fingerprints)
    require(review_receipt["schema"] == "evr0037.independent-review-receipt.v1"
            and review_receipt["status"] == "completed" and review_receipt["passed"] is True
            and not review_receipt["errors"] and review_receipt["checks"] > 0,
            "complete independent D37 review")
    review = bound(dict(path=str(Path(spec["d37_review"]["path"]).parent/"review.json"),
                        sha256=review_receipt["review_sha256"]), fingerprints)
    require(review["status"] == "completed" and review["passed"] is True and not review["errors"],
            "independent review result")
    for path in (spec["d37_spec"]["path"], spec["d37_receipt"]["path"], str(report_path), spec["d35_result"]["path"]):
        require(review["source_and_input_hashes"].get(path) == fingerprints[path], "review/reference linkage")
    for path, digest in worker["source_sha256"].items():
        require(sha(path) == digest, "inherited D37 source changed")
        fingerprints[path] = digest
    return oldspec, oldresult, load(report_path)


def evaluate_complete(spec, out, record, context, *, np, torch, Image, common):
    """Caller owns runtime/guard/final outer; this function performs no native call."""
    out = Path(out).resolve()
    began = time.perf_counter()
    fingerprints = {}
    state = dict(status="started", model_calls=0, native_calls=0, label_decode_attempts=0,
        label_decodes=0, noninitial_label_decodes=0, completed_frames=0, parity_frames=0,
        completed_sequences=0, opaque_annotation_hashes_allowed_before_prediction_seal=True,
        numerical_gt_admitted_after_prediction_seal=False)
    record["evaluation"] = state

    def checkpoint():
        state["wall_seconds"] = time.perf_counter()-began
        common.save(out/"receipt.json", record)

    checkpoint()
    try:
        sources = spec["source_sha256"]
        here = Path(__file__).resolve().parent
        require(all(str(here/name) in sources for name in REQUIRED_LOCAL_SOURCES), "evaluation source closure")
        require(Path(spec["protocol_path"]).resolve() == here/"FULL_COMPARISON.md"
                and str(here/"FULL_COMPARISON.md") in sources, "complete comparison protocol binding")
        require(all(sha(path) == digest for path, digest in sources.items()), "source bytes before evaluation")
        prediction_path, seal_path = out/"prediction-receipt.json", out/"prediction-seal.json"
        require(record["all_predictions_sealed_before_gt"] is True
                and record["noninitial_label_decodes"] == 0
                and sha(seal_path) == record["prediction_seal_sha256"]
                and sha(prediction_path) == record["prediction_receipt_sha256"],
                "producer completed immutable pre-GT barrier")
        seal = bound(dict(path=str(seal_path), sha256=sha(seal_path)), fingerprints)
        prediction = bound(dict(path=str(prediction_path), sha256=seal["receipt_sha256"]), fingerprints)
        require(prediction["source_sha256"] == sources
                and prediction["spec_sha256"] == record["spec_sha256"]
                and all(prediction["bindings"].get(k) == spec[k] for k in
                    ("d35_spec", "d35_result", "d35_receipt", "d35_outer", "qualification_spec", "qualification_receipt",
                     "qualification_outer", "qualification_outer_spec", "d37_spec", "d37_receipt", "d37_outer", "d37_seal",
                     "d37_review", "d37_review_outer")), "current frozen prediction identity")
        index = bound(context["prediction_spec"]["data_index"], fingerprints)
        require(prediction["data_index"] == context["prediction_spec"]["data_index"], "same RGB/initial index")
        validate_population(prediction, seal, context["prediction_receipt"], index)
        for desc in prediction["outputs"]:
            artifact(out, desc, fingerprints)
        state.update(prediction_seal_sha256=fingerprints[str(seal_path)], prediction_archive_hash_checks=446)
        oldspec = bound(spec["d35_spec"], fingerprints)
        d35 = Path(oldspec["metrics_path"]).parent
        require(all(sources.get(p) == h for p,h in oldspec["source_sha256"].items()), "unchanged original D35 sources")
        admission = module("d38_evaluation_admit", d35/"admit.py")
        oldspec, oldresult, d37 = admit_references(spec, admission, fingerprints)
        refs = reference_scores(oldresult, d37)
        require(context["proof_sha256"] == oldspec["prediction_proof"]["sha256"]
                and set(context["annotation_paths"]) == {n for n,_ in COHORT}, "passed admitted D35 context")
        admission.postcheck(context)
        metrics = module("d38_original_metrics", d35/"metrics.py")
        restore = module("d38_original_reconstruct", d35/"evaluate.py")
        state["numerical_gt_admitted_after_prediction_seal"] = True
        checkpoint()
        frames = {p: {n: [] for n,_ in COHORT} for p in POLICIES}
        void_counts = {n: [] for n,_ in COHORT}
        old_rows = {r["sequence"]: r for r in context["prediction_receipt"]["rows"]}
        for row in prediction["rows"]:
            name, frame = row["sequence"], row["frame"]
            new_path = artifact(out, row["output"], fingerprints)
            readouts = reconstruct_readouts(new_path, row, np=np, torch=torch)
            oldrow = old_rows[name]
            old_path = restore.frame_path(context["prediction_root"], oldrow["predictions"][frame])
            _, controls = restore.reconstruct_frame(old_path, oldrow, oldrow["frame_records"][frame],
                                                    np=np, torch=torch, metrics=metrics)
            if frame < 2:
                require(all(np.array_equal(mask, controls["B"]) for mask in readouts.values()), "t0/1 native no-op")
            state["parity_frames"] += 1
            gt_path = Path(context["annotation_paths"][name][frame])
            require(sha(gt_path) == context["fingerprints"][str(gt_path)], "GT bytes changed before decode")
            state["label_decode_attempts"] += 1
            checkpoint()
            with Image.open(gt_path) as image:
                require(image.format == "PNG" and image.mode in ("P", "L"), "indexed GT PNG")
                truth = np.array(image)
            require(truth.dtype == np.uint8 and truth.shape == tuple(row["shape"]), "GT shape/type")
            if frame == 0:
                require(list(metrics.initial_object_ids(truth)) == row["object_ids"], "all initial IDs unchanged")
            state["label_decodes"] += 1
            state["noninitial_label_decodes"] += int(frame > 0)
            void_counts[name].append(int(np.count_nonzero(truth == 255)))
            for policy in POLICIES:
                values = metrics.score_frame(truth, {**controls, **readouts}, row["object_ids"], void_policy=policy)
                require(all(values[a] == [refs[policy][name][a][frame]] for a in FIXED), "all-frame D35/D37 fixed parity")
                values["fixed_negative"] = [refs[policy][name]["fixed_negative"][frame]]
                frames[policy][name].append({a: values[a] for a in ARMS})
            state["completed_frames"] += 1
            if frame == oldrow["frames"]-1:
                state["completed_sequences"] += 1
            del truth, controls, readouts
        require(state["completed_frames"] == state["label_decodes"] == state["label_decode_attempts"] ==
                state["parity_frames"] == 446 and state["completed_sequences"] == 4
                and state["noninitial_label_decodes"] == 442, "complete full-denominator evaluation")
        reports = {p: metrics.aggregate_sequences([metrics.summarize_sequence(n, frames[p][n], [1],
                    void_policy=p) for n,_ in COHORT]) for p in POLICIES}
        primary = reports[POLICIES[0]]
        require(sum(len(r["frame_positions"]["J"]) for r in primary["per_sequence"]) == 438
                and sum(len(r["frame_positions"]["Jtr"]) for r in primary["per_sequence"]) == 110,
                "fixed438 interiors/110tail population")
        report = dict(schema="evr0038.complete-evaluation.v1", primary=primary,
            void_ignore_sensitivity=reports[POLICIES[1]], continuation=continuation(primary),
            sensitivity_continuation=continuation(reports[POLICIES[1]]), original_255_pixels=void_counts,
            arms=list(ARMS), fixed_negative_is_sealed_numerical_reference=True,
            fixed_negative_recomputed=False, matched_pipeline_efficiency_claim=False,
            prediction_seal_sha256=state["prediction_seal_sha256"],
            all_four_clips_exposed_development=True, recording_independence_established=False,
            recurrent_policy_evaluated=False, official_benchmark_result=False, F_computed=False,
            novelty_established=False, final_paper_gate_met=False, current_clean_outer_still_required=True,
            metric_scale="[0,1]; times100 is percentage points", source_sha256=sources,
            reference_bindings={k: spec[k] for k in ("d35_spec", "d35_result", "d35_receipt", "d35_outer", "d37_spec",
                "d37_receipt", "d37_outer", "d37_seal", "d37_review", "d37_review_outer")})
        admission.postcheck(context)
        require(all(sha(p) == h for p,h in fingerprints.items())
                and all(sha(p) == h for p,h in sources.items()), "source/prediction/reference postcheck")
        common.save(out/"evaluation.json", report)
        state.update(status="completed", numerical_passed=report["continuation"]["numerical_passed"],
            output=dict(path="evaluation.json", sha256=sha(out/"evaluation.json"),
                        bytes=(out/"evaluation.json").stat().st_size))
        return report
    except BaseException as exc:
        state.update(status="failed", error=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        checkpoint()
