"""Admit one complete native video after the complete frozen evidence screen.

Only stdlib is imported before source, old evidence, native qualification and
all 392 RGB/initial-mask files have been verified. No query annotation is
accepted. Old evidence sources remain byte-identical at their own commit; the
new scientific wrapper/worker use a separate commit and additional protocol.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

OWN = "experiments/0029-memory-information/"
PROTOCOL = "protocols/0029-native-video-comparison.md"
OLD_PROTOCOL = "protocols/0029-signed-memory-diagnostic.md"
MANIFEST = "configs/0029-davis-development.json"
SUPERVISOR = "experiments/0027-affine-tangent/supervise.py"
MONITOR = "experiments/0025c-rs-noise/supervise.py"
OLD_HELPER = "experiments/0022-frontend/qualify.py"
EVIDENCE_COMMIT = "c09e240009869402e2b97a55af42617eea3c6405"
EVIDENCE_SOURCES = tuple(OWN + name for name in (
    "run_evidence.py", "frontend_evidence.py", "diagnostic_core.py", "acquire_davis.py",
)) + (OLD_PROTOCOL, MANIFEST, SUPERVISOR, MONITOR)
SOURCES = EVIDENCE_SOURCES + (
    OWN + "run_video.py", OWN + "native_video.py", OWN + "install_runtime.py",
    OWN + "run_native_video_qualification.py", OWN + "qualify_native_video.py", PROTOCOL,
)
QUALIFICATION_SOURCES = tuple(OWN + name for name in (
    "run_native_video_qualification.py", "qualify_native_video.py", "native_video.py",
    "diagnostic_core.py", "run_evidence.py",
)) + (PROTOCOL, OLD_PROTOCOL, SUPERVISOR, MONITOR)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def artifact(h, directory, descriptor, filename):
    """Check opaque saved arrays or a witness, without decoding any NPZ."""
    h.require(isinstance(descriptor, dict) and descriptor.get("filename") == filename
              and Path(filename).name == filename, "artifact filename")
    path = directory / filename
    h.require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
              and type(descriptor.get("bytes")) is int
              and path.stat().st_size == descriptor["bytes"]
              and h.sha(path) == descriptor.get("sha256"), "evidence/prediction artifact changed")
    return path


def evidence_arguments(h, root, commit, evidence_root, spec_path):
    """Derive every prior input from the exact, subsequently outer-bound argv."""
    spec = h.read_json(spec_path)
    command = spec.get("command")
    h.require(isinstance(command, list) and all(isinstance(x, str) for x in command)
              and len(command) == 23
              and command[1:3] == ["-S", str(root / OWN / "run_evidence.py")],
              "evidence command prefix/length")
    tokens = command[3:]
    h.require(len(set(tokens[::2])) == 10, "duplicate evidence flags")
    flags = dict(zip(tokens[::2], tokens[1::2]))
    names = ("repo-root", "expected-commit", "native-root", "native-commit",
             "metadata-root", "rgb-root", "rgb-spec", "initial-root", "initial-spec", "output-dir")
    h.require(set(flags) == {"--" + name for name in names}
              and flags["--expected-commit"] == commit
              and flags["--repo-root"] == str(root)
              and flags["--output-dir"] == str(evidence_root), "evidence argument identity")
    parsed = {}
    for name in names:
        value = flags["--" + name]
        if name.endswith("commit"):
            parsed[name.replace("-", "_")] = value
        else:
            path = Path(value)
            h.require(path.is_absolute() and path.resolve() == path, "noncanonical evidence path")
            parsed[name.replace("-", "_")] = path
    h.require(command[0] == str(parsed["native_root"] / "installation/venv/bin/python"),
              "evidence interpreter differs from native installation")
    return parsed, command


def evidence_inventory(h, plan, directory, result, verified):
    """Recheck all 54/48 rows and their exact output inventory before any ML."""
    h.require(result.get("schema") == "evr0029.frontend-evidence.v1"
              and result.get("status") == "completed" and result.get("complete") is True
              and result.get("sam_calls") == 0
              and result.get("noninitial_annotation_access") is False
              and result.get("sam_sequence_admitted") is False
              and result.get("calls") == {
                  "superpoint_intent": 54, "superpoint_completed": 54,
                  "lightglue_intent": 48, "lightglue_completed": 48},
              "evidence screen incomplete/wrong scope")
    cohorts = plan["cohorts"]
    image_keys = [(c["sequence"], f) for c in cohorts for f in [0, *c["query_frame_indices"]]]
    query_keys = [(c["sequence"], f) for c in cohorts for f in c["query_frame_indices"]]
    images, queries, masks = result.get("images", []), result.get("queries", []), result.get("initial_masks", [])
    h.require(len(image_keys) == len(images) == 54 and len(query_keys) == len(queries) == 48
              and [(r["sequence"], r["frame"]) for r in images] == image_keys
              and [(r["sequence"], r["frame"]) for r in queries] == query_keys
              and [(r["sequence"], r["frame"]) for r in masks] == [(c["sequence"], 0) for c in cohorts]
              and all(r.get("status") == "completed" for r in images + queries + masks),
              "incomplete/reordered evidence identities")
    names = {"evidence-record.json"}
    fingerprint = {str(directory / "evidence-record.json"): h.sha(directory / "evidence-record.json")}
    image_map, target_map = {}, {}
    for rows, phase in ((images, "predictive_rgb"), (masks, "initial_supervision")):
        for row in rows:
            source = verified[phase][(row["sequence"], row["frame"])]
            expected = {k: (str(source[k]) if k == "path" else source[k])
                        for k in ("path", "sha256", "bytes")}
            if phase == "predictive_rgb":
                expected["frame"] = row["frame"]
            h.require(row.get("input") == expected, "evidence media source changed")
            if phase == "initial_supervision":
                target = row.get("target_id")
                h.require(type(target) is int and target > 0 and target != 255, "invalid target identity")
                target_map[row["sequence"]] = row
            else:
                filename = "features-%s-%05d.npz" % (row["sequence"], row["frame"])
                path = artifact(h, directory, row["output"], filename)
                names.add(filename)
                fingerprint[str(path)] = row["output"]["sha256"]
                image_map[(row["sequence"], row["frame"])] = row
    witnesses = {}
    counts = {c["sequence"]: 0 for c in cohorts}
    for row in queries:
        sequence, frame = row["sequence"], row["frame"]
        for key, prefix, extension in (("matches", "matches", ".npz"), ("witnesses", "witnesses", ".json")):
            filename = "%s-%s-%05d%s" % (prefix, sequence, frame, extension)
            path = artifact(h, directory, row[key], filename)
            names.add(filename)
            fingerprint[str(path)] = row[key]["sha256"]
        path = directory / row["witnesses"]["filename"]
        item = h.read_json(path)
        positive, negative = item.get("positive"), item.get("negative")
        h.require(isinstance(positive, list) and isinstance(negative, list)
                  and max(len(positive), len(negative)) <= 32, "witness class lists")
        actionable = len(positive) >= 4 and len(negative) >= 4
        mask = target_map[sequence]
        h.require(item.get("sequence") == sequence and item.get("frame") == frame
                  and item.get("target_id") == row.get("target_id") == mask["target_id"]
                  and item.get("initial_shape") == mask["shape"]
                  and item.get("current_shape") == image_map[(sequence, frame)]["shape"][:2]
                  and type(item.get("actionable")) is bool and type(row.get("actionable")) is bool
                  and item["actionable"] == row["actionable"] == actionable
                  and item.get("counts") == row.get("witness_counts")
                  and item["counts"].get("positive_retained") == len(positive)
                  and item["counts"].get("negative_retained") == len(negative)
                  and all(p.get("label") == 1 for p in positive)
                  and all(p.get("label") == 0 for p in negative), "witness identity/counts")
        counts[sequence] += int(actionable)
        witnesses[(sequence, frame)] = {"frame": frame, "path": path,
            "sha256": row["witnesses"]["sha256"], "bytes": row["witnesses"]["bytes"]}
    coverage = {"queries": 48, "videos": 6, "actionable_queries": sum(counts.values()),
                "actionable_queries_by_video": counts,
                "videos_with_evidence": sum(n > 0 for n in counts.values()),
                "pass": sum(n > 0 for n in counts.values()) >= 4}
    h.require(result.get("coverage") == coverage and coverage["pass"],
              "fixed evidence coverage does not admit native video stage")
    h.require({p.name for p in directory.iterdir()} == names, "unexpected/missing evidence output")
    return witnesses, coverage, fingerprint


def admit_evidence(h, old, root, commit, evidence_root, spec_path):
    h.require(commit == EVIDENCE_COMMIT, "unadmitted evidence freeze")
    prior, command = evidence_arguments(h, root, commit, evidence_root, spec_path)
    binding = old.outer_contract(h, root, commit, spec_path, evidence_root / "receipt.json", command)
    record = h.read_json(evidence_root / "receipt.json")
    sources = h.frozen(root, commit, EVIDENCE_SOURCES)
    h.require(record.get("schema") == "evr0029.evidence-operation.v1"
              and record.get("status") == "completed" and record.get("eligible") is True
              and record.get("expected_commit") == commit
              and record.get("native_commit") == prior["native_commit"]
              and record.get("noninitial_annotation_access") is False
              and record.get("sam_sequence_invocations") == 0
              and record.get("offline_attempts") == []
              and record.get("source_sha256") == sources, "evidence operation not eligible")
    installation, installed, native_binding = old.native_contract(
        h, root, prior["native_root"], prior["native_commit"])
    h.require(record.get("native_binding") == native_binding, "evidence native provenance changed")
    acquisition = old.module("evr0029_video_acquisition", root / OWN / "acquire_davis.py")
    plan, _ = acquisition.load_plan(root / MANIFEST, prior["metadata_root"],
                                    acquisition.load_helper(root / acquisition.HELPER))
    old.distinct_media_roots(h, prior["rgb_root"], prior["initial_root"])
    verified, input_bindings = {}, {}
    for phase, prefix in (("predictive_rgb", "rgb"), ("initial_supervision", "initial")):
        verified[phase], input_bindings[phase] = old.media_contract(
            h, root, commit, plan, phase, prior[prefix + "_root"], prior[prefix + "_spec"],
            prior["metadata_root"])
    h.require(record.get("input_bindings") == input_bindings, "evidence input bindings changed")
    directory = evidence_root / "evidence"
    result = h.read_json(directory / "evidence-record.json")
    h.require(record.get("evidence") == result == record.get("frontend_evidence")
              and result.get("source_sha256") == sources, "evidence inner/outer record differs")
    witnesses, coverage, files = evidence_inventory(h, plan, directory, result, verified)
    files.update(binding["file_sha256"])
    files.update(native_binding["file_sha256"])
    for item in input_bindings.values():
        files.update(item["file_sha256"])
    files.update(old.snapshot(h, [root / p for p in sources]))
    for name, expected in plan["source_bindings"].items():
        path = prior["metadata_root"] / name
        h.require(path.stat().st_size == expected["bytes"] and h.sha(path) == expected["sha256"],
                  "cached plan metadata changed")
        files[str(path)] = expected["sha256"]
    for phase in verified.values():
        files.update({str(row["path"]): row["sha256"] for row in phase.values()})
    return dict(prior=prior, plan=plan, verified=verified, witnesses=witnesses,
        installation=installation, installed=installed,
        binding=dict(evidence_commit=commit, native_commit=prior["native_commit"],
            evidence_outer=binding, evidence_source_sha256=sources, native=native_binding,
            input_bindings=input_bindings, coverage=coverage, file_sha256=files))


def video_inputs(h, root, source_sha256, admitted, sequence):
    cohorts = [c for c in admitted["plan"]["cohorts"] if c["sequence"] == sequence]
    h.require(len(cohorts) == 1, "sequence outside fixed six")
    cohort = cohorts[0]
    frames = [admitted["verified"]["predictive_rgb"][(sequence, i)]
              for i in range(cohort["frame_count"])]
    directory = frames[0]["path"].parent
    h.require(all(p["path"].parent == directory for p in frames)
              and [p["path"].stem for p in frames] == cohort["frame_stems"]
              and sorted(directory.iterdir()) == sorted(p["path"] for p in frames),
              "full video directory contains missing/extra/unordered media")
    initial = admitted["verified"]["initial_supervision"][(sequence, 0)]
    installation = admitted["installation"]
    return dict(qualify_helper=root / OLD_HELPER, native_fixture=root / OWN / "native_fixture.py",
        diagnostic_core=root / OWN / "diagnostic_core.py", sam2_source=installation / "sources/sam2",
        runtime=installation / "venv/lib/python3.12/site-packages",
        sam2_checkpoint=installation / "weights/sam2.1_hiera_tiny.pt", source_sha256=source_sha256,
        video=dict(sequence=sequence, frame_count=cohort["frame_count"], frames_dir=directory,
            initial_mask={k: initial[k] for k in ("path", "sha256", "bytes")},
            witnesses=[admitted["witnesses"][(sequence, f)] for f in cohort["query_frame_indices"]]))


def prediction_inventory(h, output, result, video):
    expected = video["witnesses"]
    h.require(result.get("schema") == "evr0029.native-video.v1"
              and result.get("status") == "completed" and result.get("complete") is True
              and result.get("sequence") == video["sequence"]
              and result.get("frame_count") == video["frame_count"]
              and result.get("query_count") == 8
              and result.get("noninitial_annotation_access") is False
              and [r.get("frame") for r in result.get("baseline", [])] == list(range(video["frame_count"]))
              and [(r.get("sequence"), r.get("frame")) for r in result.get("queries", [])]
                  == [(video["sequence"], w["frame"]) for w in expected]
              and all(r.get("status") == "completed" for r in result["queries"]),
              "incomplete native video result")
    path = output / "video-record.json"
    h.require(h.read_json(path) == result, "saved native video record differs")
    files = {"video-record.json": {"sha256": h.sha(path), "bytes": path.stat().st_size}}
    for item in result.get("outputs", []):
        name = item.get("filename")
        h.require(isinstance(name, str) and name not in files, "duplicate/invalid output filename")
        artifact(h, output, item, name)
        files[name] = {"sha256": item["sha256"], "bytes": item["bytes"]}
    h.require(set(files) == {p.name for p in output.iterdir()}, "unbound native output file")
    actionable_count = 0
    used = {"video-record.json"}
    for row, witness in zip(result["queries"], expected):
        item = h.read_json(witness["path"])
        h.require(row.get("input") == {**witness, "path": str(witness["path"])}
                  and row.get("witnesses") == item
                  and row.get("actionable") is item["actionable"]
                  and row.get("target_id") == item["target_id"], "worker changed admitted witness")
        actionable_count += int(item["actionable"])
        names = {"B", "D1", "D3", "M0", "M1", "M2"} | ({"P", "R"} if row["actionable"] else set())
        h.require(set(row.get("candidates", {})) == names
                  and isinstance(row.get("selection"), dict), "incomplete candidate bank/selection")
        for name, candidate in row["candidates"].items():
            descriptor = candidate.get("output", {})
            expected_name = "query-%05d-%s.npz" % (
                row["frame"], "native-bank" if name.startswith("M") else name)
            h.require(files.get(descriptor.get("filename")) == {
                "sha256": descriptor.get("sha256"), "bytes": descriptor.get("bytes")}
                and descriptor.get("filename") == expected_name
                and candidate.get("status") == "completed"
                and candidate.get("array_key") == "logits"
                and candidate.get("channel") == (int(name[1:]) if name.startswith("M") else 0),
                "candidate array outside sealed inventory")
            used.add(expected_name)
        h.require(row["selection"].get("actionable") is item["actionable"]
                  and set(row["selection"].get("choices", {})) == set(result["method_names"])
                  and all(name in names for name in row["selection"]["choices"].values()),
                  "invalid sealed branch choice")
    expected_calls = dict(image_encoder_intent=video["frame_count"],
        image_encoder_completed=video["frame_count"], baseline_frames=video["frame_count"],
        track_step_intent=video["frame_count"] + 24 + 2 * actionable_count,
        track_step_completed=video["frame_count"] + 24 + 2 * actionable_count,
        verification_intent=8, verification_completed=8, deletion_intent=16, deletion_completed=16,
        prompt_intent=2 * actionable_count, prompt_completed=2 * actionable_count,
        native_bank_extra_calls=0)
    h.require(set(files) == used and result.get("calls") == expected_calls
              and result.get("actionable_count") == actionable_count, "output/call completeness")
    return files


def qualification_contract(h, old, root, commit, admitted, qualification_root, spec_path):
    """Require the separate 25-frame general-worker qualification, not just cond0 tests."""
    prior = admitted["prior"]
    command = [str(admitted["installation"] / "venv/bin/python"), "-S",
        str(root / OWN / "run_native_video_qualification.py"),
        "--repo-root", str(root), "--expected-commit", commit,
        "--native-root", str(prior["native_root"]), "--native-commit", prior["native_commit"],
        "--output-dir", str(qualification_root)]
    receipt_path = qualification_root / "receipt.json"
    binding = old.outer_contract(h, root, commit, spec_path, receipt_path, command)
    record = h.read_json(receipt_path)
    sources = h.frozen(root, commit, QUALIFICATION_SOURCES)
    h.require(record.get("schema") == "evr0029.native-video-qualification-operation.v1"
              and record.get("status") == "completed" and record.get("eligible") is True
              and record.get("manufactured_only") is True
              and record.get("scientific_eligible") is False
              and record.get("scientific_input_access") is False
              and record.get("expected_commit") == commit
              and record.get("native_commit") == prior["native_commit"]
              and record.get("source_sha256") == sources
              and record.get("native_binding") == admitted["binding"]["native"]
              and record.get("offline_attempts") == [], "general worker qualification not admitted")
    directory = qualification_root / "manufactured"
    inner_path = directory / "qualification-record.json"
    result = h.read_json(inner_path)
    h.require(record.get("qualification") == result == record.get("native_video_qualification")
              and result.get("schema") == "evr0029.native-video-qualification.v1"
              and result.get("status") == "completed" and result.get("complete") is True
              and result.get("manufactured_only") is True and result.get("scientific_eligible") is False
              and result.get("davis_data_access") is False
              and result.get("noninitial_annotation_access") is False
              and result.get("source_sha256") == sources, "manufactured qualification record changed")
    qualifier = module("evr0029_video_qualification_checks", root / OWN / "qualify_native_video.py")
    queries = list(qualifier.QUERY_FRAMES)
    expected = [("manufactured_rgb", f, directory / "manufactured-frames" / ("%05d.jpg" % f))
                for f in range(25)]
    expected += [("manufactured_initial_mask", None, directory / "manufactured-initial.png")]
    expected += [("manufactured_witness", f, directory / ("manufactured-witness-%05d.json" % f))
                 for f in queries]
    inputs = result.get("inputs", [])
    h.require(len(inputs) == len(expected), "manufactured input inventory")
    files = dict(binding["file_sha256"])
    witnesses = []
    initial = None
    for item, (kind, frame, path) in zip(inputs, expected):
        h.require(item.get("kind") == kind and item.get("frame") == frame
                  and item.get("path") == str(path), "manufactured input identity")
        artifact(h, path.parent, {**item, "filename": path.name}, path.name)
        files[str(path)] = item["sha256"]
        media = {k: item[k] for k in ("path", "sha256", "bytes")}
        if kind == "manufactured_initial_mask":
            initial = media
        elif kind == "manufactured_witness":
            witnesses.append({"frame": frame, **media})
    worker_path = directory / "worker/video-record.json"
    descriptor = result.get("worker_receipt", {})
    h.require(descriptor.get("path") == str(worker_path), "qualified worker record path")
    artifact(h, worker_path.parent, {**descriptor, "filename": worker_path.name}, worker_path.name)
    worker = h.read_json(worker_path)
    h.require(result.get("worker_record", {}).get("native_video") == worker
              and worker.get("source_sha256") == sources
              and worker.get("initial_mask") == initial
              and worker.get("frames_dir") == str(directory / "manufactured-frames")
              and sorted((directory / "manufactured-frames").iterdir())
                  == sorted(path for kind, _, path in expected if kind == "manufactured_rgb"),
              "qualified worker record/input binding")
    verification = qualifier.verify_worker(worker)  # Pure saved-record checks; no imports of ML.
    h.require(result.get("verification") == verification
              and verification.get("technical_checks_passed") is True,
              "general worker technical checks incomplete")
    inventory = prediction_inventory(h, directory / "worker", worker, dict(
        sequence=qualifier.INTERNAL_SEQUENCE, frame_count=25, witnesses=witnesses))
    files.update({str(directory / "worker" / name): entry["sha256"]
                  for name, entry in inventory.items()})
    files[str(inner_path)] = h.sha(inner_path)
    helper_paths = {"qualify_helper": root / OLD_HELPER,
                    "native_fixture": root / OWN / "native_fixture.py",
                    "native_video": root / OWN / "native_video.py",
                    "diagnostic_core": root / OWN / "diagnostic_core.py"}
    h.require(result.get("observed_helper_sha256") == {k: h.sha(p) for k, p in helper_paths.items()},
              "manufactured helper sources changed")
    files.update(old.snapshot(h, [root / rel for rel in sources]))
    binding.update(source_sha256=sources, verification=verification, file_sha256=files)
    return binding


def run(args):
    root = Path(args.repo_root).resolve()
    out = Path(args.output_dir).absolute()
    if out.exists() or out.is_symlink() or root == out.resolve() or root in out.resolve().parents:
        raise ValueError("fresh private video output required")
    out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    record = dict(schema="evr0029.video-operation.v1", status="started", eligible=False,
        expected_commit=args.expected_commit, evidence_commit=args.evidence_commit,
        sequence=args.sequence, noninitial_annotation_access=False, sam_sequence_invocations=0,
        offline_attempts=[], compatibility_probes=[])
    def save():
        record["wall_seconds"] = time.perf_counter() - began
        pending = out / "receipt.pending"
        pending.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        os.replace(pending, out / "receipt.json")
    save()
    try:
        h = module("evr0029_video_install", root / OWN / "install_runtime.py")
        bindings = h.frozen(root, args.expected_commit, SOURCES)
        h.require(h.sha(Path(__file__)) == bindings[OWN + "run_video.py"], "wrong executing wrapper")
        record["source_sha256"] = bindings
        old = module("evr0029_video_admission", root / OWN / "run_evidence.py")
        admitted = admit_evidence(h, old, root, args.evidence_commit,
                                  Path(args.evidence_root).resolve(), Path(args.evidence_spec).resolve())
        record["evidence_binding"] = admitted["binding"]
        record["qualification_binding"] = qualification_contract(h, old, root, args.expected_commit,
            admitted, Path(args.qualification_root).resolve(), Path(args.qualification_spec).resolve())
        inputs = video_inputs(h, root, bindings, admitted, args.sequence)
        record["frame_count"] = inputs["video"]["frame_count"]
        record["query_frames"] = [w["frame"] for w in inputs["video"]["witnesses"]]
        save()
        installation = admitted["installation"]
        h.require(sys.version_info[:3] == (3, 12, 13) and sys.flags.no_site == 1
                  and Path(sys.executable).absolute() == installation / "venv/bin/python",
                  "pinned guarded interpreter required")
        q = module("evr0029_video_offline", root / OLD_HELPER)
        q.install_guard(record["offline_attempts"], record["compatibility_probes"])
        sys.dont_write_bytecode = True
        for key in h.THREAD_KEYS:
            h.require(os.environ.get(key) == "4", "thread environment")
        os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          XFORMERS_DISABLED="1", PYTHONDONTWRITEBYTECODE="1")
        import site
        site.main()
        h.require(Path(sys.prefix).resolve() == installation / "venv", "wrong native runtime prefix")
        sys.path[:0] = [str(installation / "sources/sam2"), str(root / OWN)]
        worker = module("evr0029_native_video_worker", root / OWN / "native_video.py")
        record["sam_sequence_invocations"] = 1
        save()
        result = worker.run(inputs, out / "predictions", record)
        record["native_video"] = result
        files = prediction_inventory(h, out / "predictions", result, inputs["video"])
        h.require(result.get("source_sha256") == bindings, "worker source binding changed")
        h.require(not record["offline_attempts"], "forbidden process/network attempt")
        h.require(admitted["installed"]["installed_inventory"] == h.inventory(installation),
                  "qualified runtime changed")
        old.unchanged(h, admitted["binding"]["file_sha256"])
        old.unchanged(h, record["qualification_binding"]["file_sha256"])
        h.require(all(h.sha(root / rel) == digest for rel, digest in bindings.items()), "source changed")
        # Includes exact directory membership, preventing an extra JPEG after admission.
        video_inputs(h, root, bindings, admitted, args.sequence)
        h.require(time.perf_counter() - began <= 1800, "complete video wall limit")
        record.update(status="completed", eligible=True, query_count=8, outputs=files)
        save()
        seal = dict(schema="evr0029.video-seal.v1", status="completed",
            expected_commit=args.expected_commit, evidence_commit=args.evidence_commit,
            source_sha256=bindings, sequence=args.sequence, frame_count=record["frame_count"],
            query_count=8, receipt_sha256=h.sha(out / "receipt.json"),
            files={"predictions/" + name: entry for name, entry in files.items()})
        with (out / "seal.json").open("x") as stream:
            json.dump(seal, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except BaseException as error:
        record.update(status="failed", eligible=False, error_type=type(error).__name__, error=str(error))
        save()
        raise
    print(json.dumps({k: record[k] for k in ("status", "eligible", "sequence", "wall_seconds")}), flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo-root", "expected-commit", "evidence-root", "evidence-spec",
                 "evidence-commit", "qualification-root", "qualification-spec", "sequence", "output-dir"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())
