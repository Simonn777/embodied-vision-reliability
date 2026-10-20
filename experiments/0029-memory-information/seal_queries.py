"""Verify all six completed video operations before admitting any query GT.

This stdlib-only verifier hashes every NPZ without decoding it. Array finiteness
is inherited from the frozen native worker's actual np.isfinite checks and
unchanged artifact bytes, not independently re-established here. No mask, RGB,
model, or query annotation is decoded. Saved scalar branch choices are checked.
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

OWN = "experiments/0029-memory-information/"
SELF = OWN + "seal_queries.py"
GT_WRAPPER = OWN + "acquire_query_gt.py"
COHORT = (("dogs-scale", 83), ("upside-down", 65), ("dog-agility", 25),
          ("cat-girl", 89), ("planes-water", 38), ("dog-gooses", 86))
METHODS = ("signed_deletion", "baseline", "max_predicted_iou", "max_unsigned_change",
           "always_D1", "always_D3", "random_deletion", "random_on_signed", "native_bank",
           "prompt_direct", "prompt_signed", "reset_direct", "reset_signed")
RANDOM_SALT = "evr0029:random-deletion:2026-10-20"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def merge(h, destination, incoming):
    for path, digest in incoming.items():
        h.require(path not in destination or destination[path] == digest, "conflicting file binding")
        destination[path] = digest


def video_arguments(h, root, commit, evidence_commit, qualification_commit, sequence, spec_path):
    spec = h.read_json(spec_path)
    command = spec.get("command")
    h.require(isinstance(command, list) and len(command) == 23
              and all(isinstance(x, str) for x in command)
              and command[1:3] == ["-S", str(root / OWN / "run_video.py")],
              "video command prefix/length")
    names = ("repo-root", "expected-commit", "evidence-root", "evidence-spec",
             "evidence-commit", "qualification-root", "qualification-spec", "qualification-commit",
             "sequence", "output-dir")
    flags = dict(zip(command[3::2], command[4::2]))
    h.require(len(set(command[3::2])) == 10 and set(flags) == {"--" + n for n in names}
              and flags["--repo-root"] == str(root)
              and flags["--expected-commit"] == commit
              and flags["--evidence-commit"] == evidence_commit
              and flags["--qualification-commit"] == qualification_commit
              and flags["--sequence"] == sequence, "video argument identity")
    parsed = {}
    for name in names:
        value = flags["--" + name]
        if name in ("expected-commit", "evidence-commit", "qualification-commit", "sequence"):
            parsed[name.replace("-", "_")] = value
        else:
            path = Path(value)
            h.require(path.is_absolute() and path.resolve() == path, "noncanonical video path")
            parsed[name.replace("-", "_")] = path
    return parsed, command


def prepare(root, commit, proof_path):
    """Bind code/proof/spec metadata only; does not grant query-GT access."""
    root, proof_path = Path(root).resolve(), Path(proof_path).resolve()
    h = module("evr0029_seal_installer", root / OWN / "install_runtime.py")
    video = module("evr0029_seal_video", root / OWN / "run_video.py")
    sources = h.frozen(root, commit, (*video.SOURCES, SELF, GT_WRAPPER))
    scientific_sources = video.unchanged_scientific_sources(h, root)
    h.require(h.sha(Path(__file__)) == sources[SELF], "wrong executing seal verifier")
    old = module("evr0029_seal_evidence", root / OWN / "run_evidence.py")
    proof = h.read_json(proof_path)
    h.require(set(proof) == {"schema", "expected_commit", "evidence_commit", "qualification_commit", "replication", "videos"}
              and proof.get("schema") == "evr0029.prediction-proof.v2"
              and proof.get("expected_commit") == commit
              and commit != video.QUALIFICATION_COMMIT
              and proof.get("evidence_commit") == video.EVIDENCE_COMMIT
              and proof.get("qualification_commit") == video.QUALIFICATION_COMMIT
              and proof.get("replication") == video.REPLICATION
              and isinstance(proof["videos"], list) and len(proof["videos"]) == 6
              and [r.get("sequence") for r in proof["videos"]] == [n for n, _ in COHORT]
              and all(set(r) == {"sequence", "spec_path"} for r in proof["videos"]),
              "fixed six-video prospective proof")
    layouts = []
    files = old.snapshot(h, [root / p for p in sources] + [proof_path])
    for row in proof["videos"]:
        spec = Path(row["spec_path"])
        h.require(spec.is_absolute() and spec.resolve() == spec and not spec.is_symlink(),
                  "video spec path")
        arguments, command = video_arguments(h, root, commit, proof["evidence_commit"],
                                             proof["qualification_commit"], row["sequence"], spec)
        layouts.append(dict(sequence=row["sequence"], spec=spec, arguments=arguments, command=command))
        files[str(spec)] = h.sha(spec)
    h.require(len({str(row["spec"]) for row in layouts}) == 6
              and len({str(row["arguments"]["output_dir"]) for row in layouts}) == 6,
              "reused video spec/output")
    shared = ("evidence_root", "evidence_spec", "qualification_root", "qualification_spec")
    h.require(all(all(row["arguments"][k] == layouts[0]["arguments"][k] for k in shared)
                  for row in layouts), "videos use different evidence/qualification chains")
    return dict(h=h, video=video, old=old, root=root, commit=commit, proof_path=proof_path,
                proof=proof, sources=sources, scientific_sources=scientific_sources,
                layouts=layouts, files=files)


def check_choices(h, row):
    """Reconstruct the fixed scalar policy; no labels or candidate arrays enter."""
    bank = row["candidates"]
    for value in bank.values():
        x = value.get("predicted_iou")
        h.require(type(x) in (int, float) and math.isfinite(x), "nonfinite predicted quality")
    losses, ious, changes = {}, {}, {}
    for name in ("B", "D1", "D3"):
        changes[name] = bank[name].get("unsigned_change")
        h.require(type(changes[name]) in (int, float) and math.isfinite(changes[name])
                  and 0 <= changes[name] <= 1, "unsigned change")
    h.require(changes["B"] == 0, "baseline unsigned change")
    digest = hashlib.sha256("\0".join((RANDOM_SALT, row["sequence"], str(row["target_id"]),
                                      str(row["frame"]))).encode()).hexdigest()
    random = {"branch": ("D1", "D3")[int(digest[-1], 16) & 1], "sha256": digest}
    choices = {name: "B" for name in METHODS}
    expected = dict(actionable=row["actionable"], choices=choices, signed_action=False,
                    random=random, bank_losses=None)
    if row["actionable"]:
        for name, candidate in bank.items():
            value = candidate.get("loss")
            h.require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                      "missing/nonfinite candidate loss")
        losses = {name: bank[name]["loss"] for name in ("B", "D1", "D3")}
        ious = {name: bank[name]["predicted_iou"] for name in losses}
        def signed(names):
            best = min(range(len(names)), key=lambda i: (bank[names[i]]["loss"], i))
            return names[best] if bank["B"]["loss"] - bank[names[best]]["loss"] > 1e-6 else "B"
        chosen = signed(("B", "D1", "D3"))
        choices.update(signed_deletion=chosen,
            max_predicted_iou=max(ious, key=ious.get), max_unsigned_change=max(changes, key=changes.get),
            always_D1="D1", always_D3="D3", random_deletion=random["branch"],
            random_on_signed=random["branch"] if chosen != "B" else "B",
            native_bank=signed(("B", "M0", "M1", "M2")),
            prompt_direct="P", prompt_signed=signed(("B", "P")),
            reset_direct="R", reset_signed=signed(("B", "R")))
        expected.update(signed_action=chosen != "B", bank_losses=losses)
    else:
        h.require(all(value.get("loss") is None for value in bank.values()),
                  "nonactionable row carries a computed loss")
    h.require(row.get("selection") == expected, "sealed branch choices differ from frozen rule")


def recheck(context):
    """No subprocesses, network, decoded arrays or scientific recomputation."""
    h, old = context["h"], context["old"]
    old.unchanged(h, context["files"])
    h.require(context["admitted"]["installed"]["installed_inventory"]
              == h.inventory(context["admitted"]["installation"]), "qualified runtime changed")
    for layout in context["layouts"]:
        context["video"].video_inputs(h, context["root"], context["video_sources"],
                                      context["admitted"], layout["sequence"])


def require_runtime(context):
    h = context["h"]
    h.require(sys.version_info[:3] == (3, 12, 13) and sys.flags.no_site == 1
              and Path(sys.executable).absolute()
                  == context["admitted"]["installation"] / "venv/bin/python",
              "qualified CPython 3.12.13 -S required")
    h.require(all(os.environ.get(k) == "4" for k in h.THREAD_KEYS), "four-thread environment")


def verify_queries(root, commit, proof_path, *, prepared=None):
    """Return deterministic JSON admission and an internal hash-recheck context."""
    context = prepared if prepared is not None else prepare(root, commit, proof_path)
    h, v, old = context["h"], context["video"], context["old"]
    h.require(context["root"] == Path(root).resolve() and context["commit"] == commit
              and context["proof_path"] == Path(proof_path).resolve(), "prepared proof identity")
    root = context["root"]
    first = context["layouts"][0]["arguments"]
    admitted = v.admit_evidence(h, old, root, context["proof"]["evidence_commit"],
                                first["evidence_root"], first["evidence_spec"])
    qualification = v.qualification_contract(h, old, root, context["proof"]["qualification_commit"], admitted,
        first["qualification_root"], first["qualification_spec"])
    context["admitted"] = admitted
    context.update(old_admission=old, new_outer_contract=v.new_outer_contract, plan=admitted["plan"],
                   installation=admitted["installation"], installed=admitted["installed"],
                   prior={**admitted["prior"], "evidence_root": first["evidence_root"],
                          "evidence_spec": first["evidence_spec"],
                          "qualification_root": first["qualification_root"],
                          "qualification_spec": first["qualification_spec"],
                          "qualification_commit": context["proof"]["qualification_commit"]},
                   videos=[], rows=[], query_prediction_dirs=[])
    require_runtime(context)
    sources = h.frozen(root, commit, v.SOURCES)
    context["video_sources"] = sources
    merge(h, context["files"], admitted["binding"]["file_sha256"])
    merge(h, context["files"], qualification["file_sha256"])
    rows, summaries, total_calls = [], [], {}
    for layout, (_, count) in zip(context["layouts"], COHORT):
        args, command = layout["arguments"], layout["command"]
        output = args["output_dir"]
        h.require(command[0] == str(admitted["installation"] / "venv/bin/python"),
                  "video did not use qualified interpreter")
        outer = v.new_outer_contract(h, root, commit, layout["spec"], output / "seal.json", command)
        record, seal = h.read_json(output / "receipt.json"), h.read_json(output / "seal.json")
        h.require(record.get("schema") == "evr0029.video-operation.v2"
                  and record.get("status") == "completed" and record.get("eligible") is True
                  and record.get("expected_commit") == commit
                  and record.get("evidence_commit") == context["proof"]["evidence_commit"]
                  and record.get("qualification_commit") == context["proof"]["qualification_commit"]
                  and record.get("replication") == v.REPLICATION
                  and record.get("sequence") == layout["sequence"]
                  and record.get("frame_count") == count and record.get("query_count") == 8
                  and record.get("noninitial_annotation_access") is False
                  and record.get("sam_sequence_invocations") == 1
                  and record.get("offline_attempts") == []
                  and record.get("source_sha256") == sources
                  and record.get("scientific_source_sha256") == context["scientific_sources"]
                  and record.get("evidence_binding") == admitted["binding"]
                  and record.get("qualification_binding") == qualification, "video operation binding")
        inputs = v.video_inputs(h, root, sources, admitted, layout["sequence"])
        worker = record["native_video"]
        inventory = v.prediction_inventory(h, output / "predictions", worker, inputs["video"])
        expected_files = {"predictions/" + name: item for name, item in inventory.items()}
        h.require(worker.get("source_sha256") == sources and worker.get("method_names") == list(METHODS)
                  and worker.get("descriptive_controls") == ["ungated_D1", "ungated_D3"]
                  and record.get("query_frames") == [w["frame"] for w in inputs["video"]["witnesses"]]
                  and record.get("outputs") == inventory
                  and seal.get("schema") == "evr0029.video-seal.v2"
                  and seal.get("status") == "completed" and seal.get("expected_commit") == commit
                  and seal.get("evidence_commit") == context["proof"]["evidence_commit"]
                  and seal.get("qualification_commit") == context["proof"]["qualification_commit"]
                  and seal.get("replication") == v.REPLICATION
                  and seal.get("source_sha256") == sources and seal.get("sequence") == layout["sequence"]
                  and seal.get("frame_count") == count and seal.get("query_count") == 8
                  and seal.get("receipt_sha256") == h.sha(output / "receipt.json")
                  and seal.get("files") == expected_files
                  and {p.name for p in output.iterdir()} == {"receipt.json", "seal.json", "predictions"},
                  "video seal/output completeness")
        for row in worker["queries"]:
            check_choices(h, row)
            context["rows"].append(row)
            context["query_prediction_dirs"].append(output / "predictions")
            rows.append(dict(sequence=row["sequence"], frame=row["frame"], target_id=row["target_id"],
                             actionable=row["actionable"], choices=row["selection"]["choices"]))
        for name, value in worker["calls"].items():
            h.require(type(value) is int and value >= 0, "native call counter type")
            total_calls[name] = total_calls.get(name, 0) + value
        merge(h, context["files"], outer["file_sha256"])
        merge(h, context["files"], old.snapshot(h, [output / "receipt.json"]))
        merge(h, context["files"], {str(output / name): item["sha256"]
                                    for name, item in expected_files.items()})
        summaries.append(dict(sequence=layout["sequence"], frame_count=count, query_count=8,
            spec_path=str(layout["spec"]), output=str(output),
            outer_sha256=outer["outer_sha256"], seal_sha256=h.sha(output / "seal.json"),
            receipt_sha256=h.sha(output / "receipt.json"), calls=worker["calls"]))
        context["videos"].append(dict(sequence=layout["sequence"], root=output,
            spec=layout["spec"], record=record, seal=seal, outer=outer,
            inputs=inputs, prediction_dir=output / "predictions"))
    h.require(len(rows) == 48 and len({(r["sequence"], r["frame"]) for r in rows}) == 48
              and total_calls["baseline_frames"] == 386
              and total_calls["image_encoder_intent"] == total_calls["image_encoder_completed"] == 386
              and total_calls["verification_completed"] == 48 and total_calls["deletion_completed"] == 96,
              "fixed whole-cohort call/identity coverage")
    recheck(context)
    admission = dict(schema="evr0029.query-admission.v2", eligible=True, expected_commit=commit,
        evidence_commit=context["proof"]["evidence_commit"], source_sha256=context["sources"],
        qualification_commit=context["proof"]["qualification_commit"], replication=v.REPLICATION,
        scientific_source_sha256=context["scientific_sources"],
        proof_sha256=h.sha(context["proof_path"]), manifest_sha256=h.sha(root / v.MANIFEST),
        video_count=6, frame_count=386, query_count=48, videos=summaries, queries=rows, calls=total_calls,
        evidence_coverage=admitted["binding"]["coverage"],
        qualification_receipt_sha256=qualification["receipt_sha256"],
        file_sha256=dict(context["files"]), query_gt_access=False,
        finite_array_scope="Inherited frozen-worker np.isfinite checks plus unchanged NPZ hashes; no independent array decoding.")
    return admission, context


def run(args):
    out, root = Path(args.output_dir).absolute(), Path(args.repo_root).resolve()
    if out.exists() or out.is_symlink() or out.resolve() == root or root in out.resolve().parents:
        raise ValueError("fresh private admission output required")
    out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    record = dict(schema="evr0029.query-admission-operation.v1", status="started", eligible=False,
                  expected_commit=args.expected_commit, query_gt_access=False)
    def save():
        record["wall_seconds"] = time.perf_counter() - began
        pending = out / "receipt.pending"
        pending.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        os.replace(pending, out / "receipt.json")
    save()
    try:
        admission, context = verify_queries(root, args.expected_commit, args.proof)
        context["h"].require(time.perf_counter() - began <= 1800, "whole admission time cap")
        record.update(status="completed", eligible=True, admission=admission)
        save()
        with (out / "seal.json").open("x") as stream:
            json.dump(dict(schema="evr0029.query-admission-seal.v1", status="completed",
                expected_commit=args.expected_commit, query_count=48,
                receipt_sha256=context["h"].sha(out / "receipt.json")), stream, indent=2)
            stream.write("\n")
    except BaseException as exc:
        record.update(status="failed", eligible=False, error_type=type(exc).__name__, error=str(exc))
        save()
        raise
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo-root", "expected-commit", "proof", "output-dir"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())
