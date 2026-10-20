"""Narrow saved-text follow-up: metadata correction and unchanged package bytes."""
import hashlib
import json
from pathlib import Path

PREP = Path(__file__).resolve().parent
PROJECT = PREP.parent.parent
REPO = PREP.parent / "embodied-vision-reliability"
OLD = PREP / "native-attempt02-publication-01"
NEW = PREP / "native-attempt02-publication-02"
COMMIT = "098dee64878b5ef33edbac12223f86da176e091e"
OLD_SOURCE = "4e26407c3c738ab34c6170657fc8836132e53ca93b84d4afe3c3eb56e6cc4286"
NEW_SOURCE = "42a8c22a070e40757fd1f74324645fe0a49cc6af34ace7905bb4235745e8402d"
OLD_MANIFEST = "201eb7e8165f27c7f4cad7e9e18534509ead7b644661e6fe452592d8ed93c319"
NEW_MANIFEST = "924563c9236d65cb2fc121cda2a85eca69008f9e7770d8dfd23fb675c82dede8"
ALIASES = [(str(REPO), "$REPOSITORY"), (str(PREP.parent), "$PRIVATE"),
           (str(PROJECT), "$PROJECT"), (str(Path.home()), "$USER_HOME")]
issues, checks = [], []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def check(label, value):
    checks.append(label)
    if not value:
        issues.append(label)


def alias(value):
    for path, replacement in ALIASES:
        value = value.replace(path, replacement)
    return value


def resolve(value):
    for path, replacement in ALIASES:
        if value == replacement or value.startswith(replacement + "/"):
            return Path(path + value[len(replacement):])
    raise ValueError("Not an admitted aliased origin")


old = read(OLD / "publication-manifest.json")
new = read(NEW / "publication-manifest.json")
check("original manifest retained", sha(OLD / "publication-manifest.json") == OLD_MANIFEST)
check("new manifest bound", sha(NEW / "publication-manifest.json") == NEW_MANIFEST)
check("old publisher snapshot retained", sha(PREP / "publish_native_attempt02-source-01.py") == OLD_SOURCE)
check("corrected publisher bound", sha(PREP / "publish_native_attempt02.py") == NEW_SOURCE)
first = PREP / "attempt02-publication-review.json"
check("first FAIL report retained", sha(first) == "d0d8247d7cc40bb4718678a9e7f1e2b40392cc7579e5bf165fdd191ca6921bb7"
      and read(first)["status"] == "FAIL")

inventory = lambda root: {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
old_names, new_names = inventory(OLD), inventory(NEW)
check("same 75 files", old_names == new_names and len(new_names) == 75)
changed = sorted(name for name in old_names & new_names if sha(OLD / name) != sha(NEW / name))
check("only two permitted files changed", changed == ["publication-manifest.json", "publication-source.py"])
check("historical reference bytes unchanged", (OLD / "attempt01-reference.json").read_bytes() == (NEW / "attempt01-reference.json").read_bytes())
old_entries = {x["path"]: x for x in old["files"]}
new_entries = {x["path"]: x for x in new["files"]}
check("74 entries and exact inventory", len(new_entries) == len(new["files"]) == 74
      and set(new_entries) == new_names - {"publication-manifest.json"})
for name, entry in new_entries.items():
    check("public binding: " + name, sha(NEW / name) == entry["public_sha256"]
          and (NEW / name).stat().st_size == entry["public_bytes"])
    if name not in ("publication-source.py", "attempt01-reference.json"):
        check("entry unchanged: " + name, entry == old_entries[name])
check("top-level metadata unchanged except publisher", {k: v for k, v in old.items() if k not in ("files", "publisher_sha256")}
      == {k: v for k, v in new.items() if k not in ("files", "publisher_sha256")})
check("publisher public copy exact", sha(NEW / "publication-source.py") == new["publisher_sha256"] == NEW_SOURCE)

affected = new_entries["attempt01-reference.json"]
history = REPO / "results/0029-development/native-attempt-01"
expected_origins = [history / "publication-manifest.json"] + [
    history / name / "supervision/receipt.json"
    for name in ("dogs-scale", "upside-down", "dog-agility", "cat-girl")]
check("five explicit actual origins", [resolve(x["origin"]) for x in affected["origins"]] == expected_origins)
for origin in affected["origins"]:
    path = resolve(origin["origin"])
    check("affected origin binding: " + origin["origin"], sha(path) == origin["original_sha256"]
          and path.stat().st_size == origin["original_bytes"] and "json_pointer" not in origin)
check("constructed summary accurately described", affected["transformation"].startswith("Constructed historical summary, not a source JSON subtree.")
      and "null measurement fields" in affected["transformation"] and affected["omissions"] == [])
check("affected metadata changes only origins and description",
      {k: v for k, v in affected.items() if k not in ("origins", "transformation")}
      == {k: v for k, v in old_entries["attempt01-reference.json"].items() if k not in ("origins", "transformation")})
reference = read(NEW / "attempt01-reference.json")
historical_manifest = read(history / "publication-manifest.json")
historical_entries = {x["path"]: x for x in historical_manifest["files"]}
for row in reference["operations"]:
    name = row["phase"].split("/")[1] + "/supervision/receipt.json"
    if row["status"] == "not_run":
        check("explicit unexecuted nulls: " + name, all(row[k] is None for k in (
            "source_commit", "exit_code", "termination", "wall_seconds", "peak_group_rss_kib",
            "spec_sha256", "outer_sha256", "artifact_sha256")))
        continue
    receipt = read(history / name)
    mapping = {"status": "status", "source_commit": "expected_commit", "exit_code": "exit_code",
               "termination": "termination", "wall_seconds": "wall_seconds",
               "peak_group_rss_kib": "peak_group_rss_kib", "spec_sha256": "spec_sha256",
               "artifact_sha256": "output_sha256"}
    check("declared field mapping: " + name, all(row[k] == receipt[v] for k, v in mapping.items())
          and row["outer_sha256"] == historical_entries[name]["original_sha256"])

spec_path = PREP / "native-attempt02-publication-spec-02.json"
outer_dir = PREP / "native-attempt02-publication-supervision-02"
spec, outer = read(spec_path), read(outer_dir / "receipt.json")
check("clean publisher outer", outer["status"] == "completed" and outer["exit_code"] == 0
      and outer["termination"] is None and outer["remaining_descendant_pids"] == []
      and outer["cleanup_errors"] == [] and outer["signal_attempts"] == [])
check("fresh exit and final snapshots", outer["post_exit_snapshot"]["rows"] == []
      and outer["post_exit_snapshot"]["started_seconds"] >= outer["exit_observed_at_seconds"]
      and outer["cleanup_snapshots"][-1]["rows"] == [])
check("spec command and output binding", sha(spec_path) == outer["spec_sha256"]
      and [alias(x) for x in spec["command"]] == outer["command"]
      and Path(spec["result_file"]) == NEW / "publication-manifest.json"
      and outer["output_sha256"] == NEW_MANIFEST
      and outer["expected_commit"] == spec["commit"] == COMMIT)
check("source and resource declaration", spec["publisher_sha256"] == NEW_SOURCE
      and outer["hard_timeout_seconds"] == spec["timeout"] == 1800
      and 0 <= outer["wall_seconds"] <= 1800
      and 0 <= outer["peak_group_rss_kib"] <= outer["group_rss_limit_kib"] == 8388608
      and all(v == "4" for v in outer["thread_environment"].values()))
for name, digest in outer["log_sha256"].items():
    check("outer log: " + name, sha(outer_dir / name) == digest)
for key, path in [("supervisor_sha256", REPO / "experiments/0029-memory-information/supervise_exit_checked.py"),
                  ("monitor_sha256", REPO / "experiments/0025c-rs-noise/supervise.py")]:
    check(key, sha(path) == outer[key])
check("scientific failure unchanged", new["scientific_gate_passed"] is False and new["manuscript_admission"] is False)

result = dict(schema="evr0029.publication-followup-review.v1", status="FAIL" if issues else "PASS",
              checks=len(checks), issues=issues, changed_files=changed, unchanged_files=len(new_names)-len(changed),
              first_review_status="FAIL", first_review_sha256=sha(first),
              original_manifest_sha256=OLD_MANIFEST, corrected_manifest_sha256=NEW_MANIFEST,
              corrected_publisher_sha256=NEW_SOURCE, script_sha256=sha(__file__),
              publication_outer_sha256=sha(outer_dir / "receipt.json"),
              scientific_gate_passed=False,
              scope="Only affected historical-summary provenance, publisher outer chain, and unchanged byte coverage. No metrics, scientific audit, media/NPZ, model, or fit rerun.")
with (PREP / "attempt02-publication-review-02.json").open("x") as stream:
    json.dump(result, stream, indent=2); stream.write("\n")
print(json.dumps(result, indent=2))
raise SystemExit(bool(issues))
