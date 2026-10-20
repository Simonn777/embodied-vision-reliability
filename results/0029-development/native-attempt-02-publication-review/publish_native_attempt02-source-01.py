"""Prepare an auditable text-only publication from completed attempt02.

DRAFT: do not execute before parent authorization and the independent audit.
No model, scientific metric, decoder, network or NPZ reader is called. Provenance
helpers only hash opaque media/arrays. Output must be a fresh PRIVATE staging
folder; parent copies it into the repository after review. Originals stay intact.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

OWN = "experiments/0029-memory-information/"
EXPECTED_COMMIT = "098dee64878b5ef33edbac12223f86da176e091e"
AUDITOR_SHA256 = "900862d5fd657a649089d97a282267f523914259d200426242be5d1ebd059aed"
COHORT = (("dogs-scale", 83), ("upside-down", 65), ("dog-agility", 25),
          ("cat-girl", 89), ("planes-water", 38), ("dog-gooses", 86))
METHODS = ("signed_deletion", "baseline", "max_predicted_iou", "max_unsigned_change",
           "always_D1", "always_D3", "random_deletion", "random_on_signed", "native_bank",
           "prompt_direct", "prompt_signed", "reset_direct", "reset_signed")
PUBLIC_DESTINATION = "results/0029-development/native-attempt-02"
PRIVATE_MARKERS = ("/" + "Users" + "/", "/" + "home" + "/")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    def reject(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def path_arg(value):
    path = Path(value).resolve()
    require(path.is_absolute(), "absolute input path required")
    return path


class Exporter:
    def __init__(self, root, output):
        self.root, self.output = root, output
        self.files, self.originals = [], {}
        self.aliases = ((str(root), "$REPOSITORY"), (str(root.parent), "$PRIVATE"),
                        (str(root.parent.parent), "$PROJECT"),
                        (str(Path.home()), "$USER_HOME"))

    def redact(self, raw):
        text = raw.decode("utf-8")
        for prefix, alias in self.aliases:
            text = text.replace(prefix, alias)
        require(not any(marker in text for marker in PRIVATE_MARKERS), "unredacted private path")
        return text.encode()

    def identity(self, path):
        path = Path(path).resolve()
        digest = sha(path)
        require(str(path) not in self.originals or self.originals[str(path)] == digest,
                "source changed during publication")
        self.originals[str(path)] = digest
        return dict(origin=self.redact(str(path).encode()).decode(),
                    original_sha256=digest, original_bytes=path.stat().st_size)

    def put(self, name, raw, origins, transformation):
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts
                and relative.suffix in (".json", ".log", ".csv", ".md", ".py"),
                "only named generated text files may be published")
        destination = self.output / relative
        require(not destination.exists(), "duplicate public path")
        public = self.redact(raw)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(public)
        self.files.append(dict(path=relative.as_posix(), origins=origins,
            public_sha256=sha(destination), public_bytes=len(public),
            transformation=transformation, omissions=[]))

    def copy(self, source, name):
        require(Path(source).suffix in (".json", ".log", ".md", ".py"), "binary/media copy forbidden")
        origin = self.identity(source)
        raw = Path(source).read_bytes()
        self.put(name, raw, [origin], "Byte-identical except declared literal private-path aliases; no fields removed.")

    def derive_json(self, name, value, source, pointer):
        origin = self.identity(source)
        origin["json_pointer"] = pointer
        self.put(name, encoded(value), [origin], "Complete indicated JSON subtree; path aliases only.")

    def table(self, name, rows, source, description):
        require(rows, "empty derived table")
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        self.put(name, stream.getvalue().encode(), [self.identity(source)],
                 description + " Numeric values are not rounded; no outcome rows omitted.")

    def operation(self, spec_path, label, result_path):
        spec = read(spec_path)
        outer_root = Path(spec["operation_output"])
        outer_path = outer_root / "receipt.json"
        outer = read(outer_path)
        self.copy(spec_path, label + "/spec.json")
        self.copy(result_path, label + "/receipt.json")
        for name in ("receipt.json", "stdout.log", "stderr.log"):
            self.copy(outer_root / name, label + "/supervision/" + name)
        return dict(attempt="02", phase=label, status=outer["status"],
            source_commit=outer["expected_commit"], exit_code=outer["exit_code"],
            termination=outer.get("termination"), wall_seconds=outer["wall_seconds"],
            peak_group_rss_kib=outer["peak_group_rss_kib"],
            spec_sha256=sha(spec_path), outer_sha256=sha(outer_path),
            artifact_sha256=sha(result_path), cost_scope="disjoint actual outer operation")


def metadata_contract(args):
    require(args.expected_commit == EXPECTED_COMMIT, "unexpected attempt02 freeze")
    ci, revision = read(args.source_ci), read(args.spec_revision)
    require(ci.get("headSha") == args.expected_commit and ci.get("status") == "completed"
            and ci.get("conclusion") == "success" and isinstance(ci.get("url"), str),
            "exact-source CI not successful")
    require(revision.get("schema") == "evr0029.unexecuted-specification-revision.v1"
            and revision.get("to_commit") == args.expected_commit
            and revision.get("scientific_operation_executed") is False
            and revision.get("no_operation_outputs_before_or_after") is True,
            "prospective specification revision missing")
    base = Path(revision["root"]).resolve()
    require(base == args.proof.parent, "revision/proof root differs")
    expected = {"prediction-proof.json", "gt-spec.json", "evaluation-spec.json", "audit-spec-01.json"}
    expected.update(name + "/spec.json" for name, _ in COHORT)
    require(set(revision.get("new_manifest", {})) == expected, "fixed ten specification files required")
    for relative, digest in revision["new_manifest"].items():
        require(sha(base / relative) == digest, "prospective specification changed: " + relative)
    require(args.gt_spec == base / "gt-spec.json"
            and args.evaluation_spec == base / "evaluation-spec.json"
            and args.audit_spec == base / "audit-spec-01.json", "wrong stage specification")
    return ci, revision


def admitted_results(args):
    require(sha(args.audit_source) == AUDITOR_SHA256, "reviewed private auditor source changed")
    auditor = module("evr0029_publication_provenance_only", args.audit_source)
    # ONLY admission: no numerical_preflight/arithmetic/load_arrays/run call.
    seals, context, evaluation, _, fingerprints, evaluation_outer = auditor.admission(args)
    h, old = context["h"], context["old_admission"]
    command = [str(context["installation"] / "venv/bin/python"), "-S", str(args.audit_source),
        "--repo-root", str(args.repo_root), "--expected-commit", args.expected_commit,
        "--proof", str(args.proof), "--gt-root", str(args.gt_root), "--gt-spec", str(args.gt_spec),
        "--evaluation-root", str(args.evaluation_root), "--evaluation-spec", str(args.evaluation_spec),
        "--output-dir", str(args.audit_root)]
    audit_outer = context["new_outer_contract"](h, args.repo_root, args.expected_commit,
        args.audit_spec, args.audit_root / "receipt.json", command)
    audit = read(args.audit_root / "receipt.json")
    require(audit.get("schema") == "evr0029.saved-query-audit.v1"
            and audit.get("status") == "completed" and audit.get("eligible") is True
            and audit.get("expected_commit") == args.expected_commit
            and audit.get("auditor_sha256") == AUDITOR_SHA256
            and audit.get("errors") == [] and type(audit.get("checks")) is int and audit["checks"] > 0
            and audit.get("query_masks_decoded") == 48 and audit.get("model_calls") == 0
            and audit.get("offline_attempts") == [] and audit.get("admission_complete") is True
            and audit.get("all_prediction_arithmetic_preflight_complete") is True
            and audit.get("source_sha256") == context["sources"]
            and audit.get("file_sha256") == fingerprints
            and audit.get("prediction_admission") == evaluation["prediction_admission"]
            and audit.get("evaluation_outer") == evaluation_outer
            and audit.get("independent_gate") == evaluation["evaluation"]["gate"],
            "independent saved audit incomplete or mismatched")
    for path, digest in audit_outer["file_sha256"].items():
        require(path not in fingerprints or fingerprints[path] == digest, "conflicting audit binding")
        fingerprints[path] = digest
    result = evaluation["evaluation"]
    require(result.get("schema") == "evr0029.query-evaluation.v1" and result.get("eligible") is True
            and result.get("denominator") == 48 and result.get("gt_accessed") is True
            and result.get("gate") == result["summary"]["gate"], "incomplete evaluation")
    expected = [(name, 4 + (2*j*(n-5)+7)//14) for name, n in COHORT for j in range(8)]
    for rows in (result["rows"], result["summary"]["rows"]):
        require([(r["sequence"], r["frame"]) for r in rows] == expected
                and all(r["status"] == "completed" for r in rows), "fixed48 outcome coverage")
    for row in result["rows"]:
        require(set(row["selected_branches"]) == set(METHODS), "all13 methods required")
    require(list(result["summary"]["videos"]) == [name for name, _ in COHORT], "six-video order changed")
    return seals, context, evaluation, audit, fingerprints


def tables(exporter, receipt, source):
    e, s = receipt["evaluation"], receipt["evaluation"]["summary"]
    methods = [dict(method=name, pooled_j=s["pooled"][name], **{
        sequence: s["videos"][sequence]["means"][name] for sequence, _ in COHORT}) for name in METHODS]
    exporter.table("methods.csv", methods, source, "All13 frozen method means, pooled and six videos.")
    videos = []
    for sequence, count in COHORT:
        v = s["videos"][sequence]
        videos.append(dict(sequence=sequence, frame_count=count, query_count=8,
            actionable_queries=e["coverage"]["actionable_queries_by_video"][sequence],
            signed_j=v["means"]["signed_deletion"], baseline_j=v["means"]["baseline"],
            best_control=v["best_control"], best_control_j=v["means"][v["best_control"]],
            signed_margin=v["signed_margin"], positive=v["signed_margin"] > 0,
            above_minus_002=v["signed_margin"] >= -.02))
    exporter.table("sequences.csv", videos, source, "All six video margins against that video's strongest named control.")
    queries = []
    for row, metric in zip(e["rows"], s["rows"]):
        value = {key: row[key] for key in ("sequence", "frame", "target_id", "status", "actionable")}
        for name in METHODS:
            value[name + "_branch"] = row["selected_branches"][name]
            value[name + "_j"] = metric["j"][name]
        for name in ("B", "D1", "D3", "M0", "M1", "M2", "P", "R"):
            value[name + "_candidate_j"] = row["candidate_j"].get(name)
        value.update(signed_j_delta=row["signed_harm"]["j_delta"],
                     signed_drop_gt_005=row["signed_harm"]["drop_gt_005"])
        queries.append(value)
    exporter.table("queries.csv", queries, source, "All48 fixed queries and all624 deployed method outcomes; absent conditional P/R is blank.")
    gates = [dict(clause="coverage", value=e["coverage"]["videos_with_evidence"], threshold=4,
                  relation=">=", passed=e["gate"]["coverage"])]
    for clause, key, threshold, relation in (
        ("candidate_information", "candidate_gain", .02, ">="),
        ("pooled_selection", "signed_gain", .02, ">="),
        ("four_positive_videos", "positive_videos", 4, ">="),
        ("no_video_below_minus_002", "worst_video_margin", -.02, ">=")):
        gates.append(dict(clause=clause, value=s[key], threshold=threshold, relation=relation, passed=e["gate"][clause]))
    exporter.table("gate.csv", gates, source, "Saved five gate components; no rounded-value decision or retuning.")
    headroom = []
    for scope in ("pooled", *(name for name, _ in COHORT)):
        values = s["pooled"] if scope == "pooled" else s["videos"][scope]["means"]
        description = e["diagnostics"]["pooled"] if scope == "pooled" else e["diagnostics"]["videos"][scope]
        headroom.append(dict(scope=scope, baseline_j=values["baseline"],
            deletion_admissible_j=values["oracle_admissible"],
            deletion_unrestricted_j=values["oracle_unrestricted"],
            ungated_D1_j=values["ungated_D1"], ungated_D3_j=values["ungated_D3"],
            **description["native_bank"]))
    exporter.table("headroom.csv", headroom, source, "Separate deletion/native oracle descriptions, never deployable controls.")
    cost_columns = [dict(unit=unit, column=name, sum=value["sum"], reported_queries=value["reported_queries"])
        for unit, values in e["diagnostics"]["pooled"]["costs"].items() for name, value in values.items()]
    exporter.table("query-cost-columns.csv", cost_columns, source, "Each named cost column only; overlapping columns are not totaled.")


def historical_costs(exporter, directory):
    manifest_path = directory / "publication-manifest.json"
    manifest = read(manifest_path)
    require(manifest.get("schema") == "evr0029.incomplete-native-publication.v1"
            and manifest.get("query_gt_acquired") is False and manifest.get("query_gt_decoded") is False,
            "attempt01 historical publication contract")
    known = {row["path"]: row for row in manifest["files"]}
    rows = []
    for sequence, _ in COHORT:
        name = sequence + "/supervision/receipt.json"
        if name not in known:
            require(sequence in ("planes-water", "dog-gooses"), "historical operation missing")
            rows.append(dict(attempt="01", phase="video/" + sequence, status="not_run", source_commit=None,
                exit_code=None, termination=None, wall_seconds=None, peak_group_rss_kib=None,
                spec_sha256=None, outer_sha256=None, artifact_sha256=None, cost_scope="not executed; no zero cost invented"))
            continue
        path = directory / name
        require(sha(path) == known[name]["public_sha256"]
                and path.stat().st_size == known[name]["public_bytes"], "historical outer publication changed")
        value = read(path)
        exporter.identity(path)
        rows.append(dict(attempt="01", phase="video/" + sequence, status=value["status"],
            source_commit=value["expected_commit"], exit_code=value["exit_code"], termination=value.get("termination"),
            wall_seconds=value["wall_seconds"], peak_group_rss_kib=value["peak_group_rss_kib"],
            spec_sha256=value["spec_sha256"], outer_sha256=known[name]["original_sha256"],
            artifact_sha256=value.get("output_sha256"), cost_scope="historical disjoint actual outer; no efficacy admission"))
    exporter.derive_json("attempt01-reference.json", dict(status="technically_ineligible",
        public_directory=directory.relative_to(exporter.root).as_posix(),
        manifest_sha256=sha(manifest_path), operations=rows,
        caveat="Original failure and unresolved cause remain unchanged. No old predictions enter attempt02."), manifest_path, "/files")
    return rows


def run(args):
    for name in ("repo_root", "proof", "gt_root", "gt_spec", "evaluation_root", "evaluation_spec",
                 "audit_root", "audit_spec", "audit_source", "source_ci", "spec_revision", "attempt01_publication"):
        setattr(args, name, path_arg(getattr(args, name)))
    root, output = args.repo_root, Path(args.output_dir).absolute()
    require(not output.exists() and not output.is_symlink()
            and output.resolve() != root and root not in output.resolve().parents,
            "fresh PRIVATE publication staging required; do not edit repository during admission")
    self_hash = sha(__file__)
    metadata_hashes = {str(path): sha(path) for path in (args.source_ci, args.spec_revision)}
    ci, revision = metadata_contract(args)
    seals, context, receipt, audit, fingerprints = admitted_results(args)
    fingerprints.update(metadata_hashes)
    output.mkdir(parents=True, exist_ok=False)
    exporter = Exporter(root, output)
    evaluation_path = args.evaluation_root / "receipt.json"
    exporter.copy(args.source_ci, "source-ci.json")
    exporter.copy(args.spec_revision, "specification-revision.json")
    exporter.copy(args.proof, "prediction-proof.json")
    exporter.copy(args.audit_source, "audit-source.py")
    exporter.copy(Path(__file__), "publication-source.py")
    exporter.derive_json("evaluation.json", receipt["evaluation"], evaluation_path, "/evaluation")
    exporter.derive_json("diagnostics.json", receipt["evaluation"]["diagnostics"], evaluation_path, "/evaluation/diagnostics")
    tables(exporter, receipt, evaluation_path)
    operations = []
    for video in context["videos"]:
        name = video["sequence"]
        operations.append(exporter.operation(video["spec"], "videos/" + name, video["root"] / "receipt.json"))
        exporter.copy(video["root"] / "seal.json", "videos/" + name + "/seal.json")
        exporter.copy(video["prediction_dir"] / "video-record.json", "videos/" + name + "/worker-record.json")
    for phase in ("gt", "evaluation", "audit"):
        directory, spec = getattr(args, phase + "_root"), getattr(args, phase + "_spec")
        operations.append(exporter.operation(spec, phase, directory / "receipt.json"))
    exporter.copy(args.gt_root / "acquisition/receipt.json", "gt/acquisition-receipt.json")
    historical = historical_costs(exporter, args.attempt01_publication)
    # Operations are generated from multiple original receipts, so list every origin.
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(operations[0])); writer.writeheader(); writer.writerows(historical + operations)
    exporter.put("operations.csv", stream.getvalue().encode(),
        [exporter.identity(Path(path)) for path in sorted(exporter.originals) if path.endswith("receipt.json")],
        "All actual old/new outer durations plus explicit not-run rows; no nested stage sums.")
    e, s = receipt["evaluation"], receipt["evaluation"]["summary"]
    state = "PASS" if e["gate"]["pass"] else "FAIL"
    readme = f'''# 0029 attempt02: fixed development diagnostic

All six new video operations and all48 fixed queries completed; independent saved audit passed ({audit['checks']} checks,0errors). The frozen scientific gate is **{state}**. This is custom fixed-query foreground J, not official DAVIS J&F or held-out efficacy.

Signed J: {s['pooled']['signed_deletion']:.9f}; baseline J: {s['pooled']['baseline']:.9f}; strongest pooled named control: {s['best_pooled_control']} ({s['pooled'][s['best_pooled_control']]:.9f}). Signed gain: {100*s['signed_gain']:.9f} percentage points. Admissible deletion-oracle gain: {100*s['candidate_gain']:.9f} points. Positive-video margins: {s['positive_videos']}/6; worst margin: {100*s['worst_video_margin']:.9f} points. Exact raw values/Booleans are in gate.csv and evaluation.json, never inferred from rounded text.

methods.csv contains all13 methods; sequences.csv all6 video-specific strongest-control margins; queries.csv all48 fixed queries/624 deployed outcomes. headroom.csv separates deletion and native-bank oracles from deployable controls. Full diagnostics retain witness precision including void255, signed-loss/J signs, mask no-ops and harmful signed drops strictly>.05. Nonactionable queries remain baseline; no successful subset is substituted. Query dependence precludes treating48 as48 independent trials.

Source freeze: {args.expected_commit}; source CI: {ci['url']}. Original evidence c09 and qualification ec94 remain distinct immutable chains. Attempt01 remains technically ineligible and is referenced separately, including its failed outer cost; it contributes no predictions or efficacy value. operations.csv keeps disjoint old/new operation costs. Shared preparation remains in prior publications and is not recharged. Query-stage columns are overlapping descriptions, not extra total latency. No complete-pipeline latency advantage is claimed.

Full generated JSON/log/spec fields are retained with literal path aliases only. RGB, mask PNGs, candidate NPZ, checkpoints and runtime binaries are not copied. publication-manifest.json maps immutable original hashes/bytes to public hashes/bytes and records every derivation. Embedded original hashes still identify private original artifacts, not sanitized copies. Auditor scalar-replay differences remain visible. This is necessary development evidence only; it does not establish novelty, closed-loop improvement, a dense guarantee, final held-out admission or a paper claim.
'''
    exporter.put("README.md", readme.encode(), [exporter.identity(evaluation_path), exporter.identity(args.audit_root / "receipt.json")],
                 "Generated factual summary of saved admitted scalar results; no scientific recomputation.")
    # Recheck all admitted sources, payload fingerprints and runtime after packaging.
    seals.recheck(context)
    context["old_admission"].unchanged(context["h"], fingerprints)
    require(sha(__file__) == self_hash, "publisher source changed")
    for path, digest in exporter.originals.items():
        require(sha(path) == digest, "publication input changed")
    for entry in exporter.files:
        path = output / entry["path"]
        require(sha(path) == entry["public_sha256"] and path.stat().st_size == entry["public_bytes"], "public file changed")
        require(not any(marker.encode() in path.read_bytes() for marker in PRIVATE_MARKERS), "privacy scan failed")
    manifest = dict(schema="evr0029.native-attempt02-publication.v1", source_commit=args.expected_commit,
        proposed_repository_destination=PUBLIC_DESTINATION, completed=True,
        scientific_gate_passed=e["gate"]["pass"], manuscript_admission=False,
        publisher_sha256=self_hash, auditor_sha256=AUDITOR_SHA256,
        proof_sha256=sha(args.proof), source_ci_sha256=sha(args.source_ci),
        specification_revision_sha256=sha(args.spec_revision),
        files=exporter.files, no_binary_media_or_npz_copied=True,
        private_paths_aliased=True, all_fields_preserved=True,
        cost_scope="Disjoint outer rows only. Shared preparation linked, query-stage costs descriptive. No latency advantage.")
    with (output / "publication-manifest.json").open("xb") as stream:
        stream.write(exporter.redact(encoded(manifest)))
    print(json.dumps(dict(status="staged_for_independent_review", output=str(output),
        files=len(exporter.files)+1, query_count=48, method_count=13,
        scientific_gate_passed=e["gate"]["pass"], manifest_sha256=sha(output / "publication-manifest.json"))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo-root", "expected-commit", "proof", "gt-root", "gt-spec", "evaluation-root",
                 "evaluation-spec", "audit-root", "audit-spec", "audit-source", "source-ci",
                 "spec-revision", "attempt01-publication", "output-dir"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())
