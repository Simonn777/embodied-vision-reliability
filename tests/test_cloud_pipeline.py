"""Adversarial checks for the trusted cloud publication boundary."""

import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


spec = importlib.util.spec_from_file_location("cloud_pipeline", Path(__file__).resolve().parents[1] / "scripts/cloud_pipeline.py")
cloud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cloud)
DATE = "2026-10-16"


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, stderr=subprocess.DEVNULL).decode().strip()


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "--quiet")
    git(repo, "config", "user.email", "cloud-test@example.invalid")
    git(repo, "config", "user.name", "Cloud boundary tests")
    files = {"README.md": "original\n", "src/base.py": "x = 1\n", "notes/2026-10-16.md": "old note\n",
             "protocols/0001.md": "frozen protocol\n", "results/0001/result.json": "{}\n",
             "AGENTS.md": "trusted rules\n", "scripts/cloud_pipeline.py": "trusted guard\n",
             "tests/test_cloud_pipeline.py": "trusted tests\n", "pyproject.toml": "trusted dependencies\n"}
    for name, value in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    return repo, git(repo, "rev-parse", "HEAD")


def write(repo, name, content):
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if isinstance(content, bytes) else content.encode())
    return path


def make_bundle(repository, tmp_path, changes=None, preflight=False):
    repo, base = repository
    for name, value in (changes or {"README.md": "updated\n"}).items():
        write(repo, name, value)
    output = tmp_path / "proposal"
    cloud.bundle(repo, output, base, DATE, preflight)
    return output


def clean_checkout(repository, tmp_path):
    repo, _ = repository
    destination = tmp_path / "candidate"
    git(tmp_path, "clone", "--quiet", "--no-hardlinks", str(repo), str(destination))
    return destination


def test_round_trip_validates_actual_tree_and_stages_only_regular_changes(repository, tmp_path):
    repo, base = repository
    output = make_bundle(repository, tmp_path, {"README.md": "updated\n", "src/new.py": "x=2\n",
                                               f"notes/{DATE}.md": "old note\nnew evidence\n"})
    candidate = clean_checkout(repository, tmp_path)
    before = git(candidate, "write-tree")
    report = cloud.validate(candidate, output, base, DATE)
    assert report["paths"] == ["README.md", f"notes/{DATE}.md", "src/new.py"]
    assert git(candidate, "write-tree") == before
    assert (candidate / "README.md").read_text() == "original\n"
    cloud.validate(candidate, output, base, DATE, apply=True)
    assert (candidate / "src/new.py").read_text() == "x=2\n"
    assert git(candidate, "diff", "--cached", "--name-only").splitlines() == report["paths"]


def test_empty_patch_only_valid_in_explicit_preflight(repository, tmp_path):
    repo, base = repository
    with pytest.raises(cloud.Rejected, match="empty"):
        cloud.bundle(repo, tmp_path / "production", base, DATE)
    output = tmp_path / "preflight"
    report = cloud.bundle(repo, output, base, DATE, preflight=True)
    assert report["file_count"] == 0
    assert cloud.validate(repo, output, base, DATE, preflight=True)["valid"]
    with pytest.raises(cloud.Rejected, match="Manifest"):
        cloud.validate(repo, output, base, DATE)


def test_no_material_change_can_exit_success_without_creating_an_artifact(repository, tmp_path):
    repo, base = repository
    destination = tmp_path / "no-change"
    report = cloud.bundle(repo, destination, base, DATE, allow_no_change=True)
    assert report["has_changes"] is False and report["reason"] == "no_material_change"
    assert not destination.exists()


def test_bundle_cli_writes_no_change_outputs_without_api_or_artifact(repository, tmp_path, capsys):
    repo, base = repository
    destination = tmp_path / "no-change"
    output = tmp_path / "github-output"
    assert cloud.main(["bundle", "--repo", str(repo), "--output", str(destination),
                       "--base-sha", base, "--date", DATE, "--allow-no-change",
                       "--github-output", str(output)]) == 0
    assert "has_changes=false\n" in output.read_text()
    assert "reason=no_material_change\n" in output.read_text()
    assert json.loads(capsys.readouterr().out)["has_changes"] is False
    assert not destination.exists()


@pytest.mark.parametrize("path", [".github/workflows/owned.yml", ".research/cloud-runs/2026-10-16.json",
    ".codex/config.toml", "AGENTS.md", "docs/AGENTS.md", "scripts/cloud_pipeline.py",
    "tests/test_cloud_pipeline.py", "pyproject.toml", "docs/RESEARCH_CHARTER.md",
    "docs/CONTRIBUTIONS.md", "requirements.txt", "docs/requirements-extra.txt",
    "docs/CLOUD_EXECUTION.md",
    "src/../README.md", "/tmp/outside", "src//a.py", "src/./a.py", "src/a\\b.py",
    "src/a\nb.py", "src/a\rb.py", "src/.git/config", "docs/node_modules/code.js"])
def test_sensitive_and_ambiguous_paths_are_rejected(path):
    with pytest.raises(cloud.Rejected):
        cloud.allowed_path(path)


@pytest.mark.parametrize("path", ["protocols/0001.md", "results/0001/result.json"])
def test_published_protocols_and_results_cannot_be_rewritten(repository, tmp_path, path):
    with pytest.raises(cloud.Rejected, match="immutable"):
        make_bundle(repository, tmp_path, {path: "replacement"})


def test_old_result_directory_cannot_acquire_new_evidence_files(repository, tmp_path):
    with pytest.raises(cloud.Rejected, match="run directories are immutable"):
        make_bundle(repository, tmp_path, {"results/0001/new-claim.json": "{}"})


def test_current_note_is_append_only_and_other_dates_are_frozen(repository, tmp_path):
    with pytest.raises(cloud.Rejected, match="prefix"):
        make_bundle(repository, tmp_path, {f"notes/{DATE}.md": "rewritten\n"})
    repo, base = repository
    write(repo, f"notes/{DATE}.md", "old note\n")
    write(repo, "notes/2026-10-15.md", "backfilled\n")
    with pytest.raises(cloud.Rejected, match="current day's"):
        cloud.bundle(repo, tmp_path / "other", base, DATE)


@pytest.mark.parametrize("kind", ["delete", "rename", "symlink", "executable"])
def test_forbidden_filesystem_changes_cannot_be_bundled(repository, tmp_path, kind):
    repo, base = repository
    path = repo / "src/base.py"
    if kind == "delete":
        path.unlink()
    elif kind == "rename":
        path.rename(repo / "src/renamed.py")
    elif kind == "symlink":
        path.unlink()
        path.symlink_to("../README.md")
    else:
        path.chmod(0o755)
    with pytest.raises(cloud.Rejected):
        cloud.bundle(repo, tmp_path / "proposal", base, DATE)


def test_new_binary_result_uses_bounded_literals_and_round_trips(repository, tmp_path):
    repo, base = repository
    data = b"\x00binary example\xff" * 100
    output = make_bundle(repository, tmp_path, {"results/0002/example.bin": data})
    assert b"GIT binary patch" in (output / "changes.patch").read_bytes()
    candidate = clean_checkout(repository, tmp_path)
    cloud.validate(candidate, output, base, DATE, apply=True)
    assert (candidate / "results/0002/example.bin").read_bytes() == data


def test_new_protocol_and_results_require_separate_publications(repository, tmp_path):
    with pytest.raises(cloud.Rejected, match="prior run"):
        make_bundle(repository, tmp_path, {"protocols/0002.md": "new", "results/0002/value.json": "{}"})


@pytest.mark.parametrize("path", ["src/base.py", "scripts/new_experiment.py"])
def test_source_must_be_published_before_new_results(repository, tmp_path, path):
    with pytest.raises(cloud.Rejected, match="source changes in a prior run"):
        make_bundle(repository, tmp_path, {path: "x=42\n", "results/0002/value.json": "{}"})


@pytest.mark.parametrize("line", [b"new file mode 120000", b"new file mode 100755", b"new file mode 160000",
                                   b"deleted file mode 100644", b"old mode 100644", b"rename from src/a.py",
                                   b"copy from src/a.py", b"delta 100"])
def test_untrusted_patch_modes_and_binary_deltas_rejected_before_git(line):
    patch = b"diff --git a/src/a.py b/src/a.py\n" + line + b"\n"
    with pytest.raises(cloud.Rejected):
        cloud.patch_paths(patch)


def test_binary_expansion_size_bound_is_checked_before_git(monkeypatch):
    monkeypatch.setattr(cloud, "MAX_BYTES", 1000)
    with pytest.raises(cloud.Rejected, match="Expanded"):
        cloud.patch_paths(b"diff --git a/results/new.bin b/results/new.bin\nGIT binary patch\nliteral 1000\n")


def test_expanded_aggregate_limit_not_just_patch_size(repository, tmp_path, monkeypatch):
    output = make_bundle(repository, tmp_path, {"docs/a.bin": b"\x00" * 3000, "docs/b.bin": b"\x00" * 3000})
    repo, base = repository
    assert (output / "changes.patch").stat().st_size < 5000
    monkeypatch.setattr(cloud, "MAX_BYTES", 5000)
    with pytest.raises(cloud.Rejected, match="Expanded"):
        cloud.validate(repo, output, base, DATE)


@pytest.mark.parametrize("field,value", [("base_sha", "0" * 40), ("date", "2026-10-17"),
                                         ("patch_sha256", "0" * 64), ("mode", "preflight"),
                                         ("schema_version", True), ("extra", "unexpected")])
def test_manifest_cannot_change_workflow_authority(repository, tmp_path, field, value):
    output = make_bundle(repository, tmp_path)
    path = output / "manifest.json"
    manifest = json.loads(path.read_text()); manifest[field] = value
    path.write_text(json.dumps(manifest))
    repo, base = repository
    with pytest.raises(cloud.Rejected, match="Manifest"):
        cloud.validate(repo, output, base, DATE)


def test_patch_tampering_and_extra_artifact_files_fail_closed(repository, tmp_path):
    output = make_bundle(repository, tmp_path)
    repo, base = repository
    with (output / "changes.patch").open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(cloud.Rejected, match="Manifest"):
        cloud.validate(repo, output, base, DATE)
    (output / "unreviewed.txt").write_text("extra")
    with pytest.raises(cloud.Rejected, match="exactly"):
        cloud.validate(repo, output, base, DATE)


def test_symlink_bundle_member_is_rejected(repository, tmp_path):
    output = make_bundle(repository, tmp_path)
    path = output / "changes.patch"
    target = tmp_path / "external.patch"
    path.rename(target); path.symlink_to(target)
    repo, base = repository
    with pytest.raises(cloud.Rejected, match="regular"):
        cloud.validate(repo, output, base, DATE)


def test_credential_like_content_is_not_published(repository, tmp_path):
    with pytest.raises(cloud.Rejected, match="Credential"):
        make_bundle(repository, tmp_path, {"docs/leak.txt": b"sk-" + b"A" * 30})


def test_stale_base_is_rejected_even_for_preflight(repository, tmp_path):
    repo, base = repository
    output = tmp_path / "preflight"
    cloud.bundle(repo, output, base, DATE, preflight=True)
    write(repo, "README.md", "new upstream\n")
    git(repo, "add", "README.md"); git(repo, "commit", "-qm", "new base")
    with pytest.raises(cloud.Rejected, match="HEAD"):
        cloud.validate(repo, output, base, DATE, preflight=True)


@pytest.mark.parametrize("moment,eligible,date", [
    ("2026-10-16T03:59:59Z", False, "2026-10-15"),
    ("2026-10-16T04:00:00Z", True, "2026-10-16"),
    ("2026-11-01T05:30:00Z", True, "2026-11-01"),
    ("2026-11-01T06:30:00Z", True, "2026-11-01"),
    ("2027-10-16T03:59:59Z", True, "2027-10-15"),
    ("2027-10-16T04:00:00Z", False, "2027-10-16"),
])
def test_gate_uses_new_york_dates_and_handles_dst(repository, moment, eligible, date):
    repo, _ = repository
    result = cloud.gate(repo, "research", "true", "true", moment)
    assert result["eligible"] == result["run_ai"] == result["run_publish"] == eligible
    assert result["date"] == date


def test_gate_has_separate_paid_activation_and_no_api_preflight(repository):
    repo, _ = repository
    moment = "2026-10-16T12:00:00Z"
    assert cloud.gate(repo, "research", "false", "true", moment)["reason"] == "research_disabled"
    assert cloud.gate(repo, "research", "true", "false", moment)["reason"] == "api_key_missing"
    preflight = cloud.gate(repo, "preflight", "false", "false", moment)
    assert preflight["eligible"] and not preflight["run_ai"] and not preflight["run_publish"]


def test_publisher_only_ledger_is_exclusive_and_date_checked(repository, monkeypatch):
    repo, base = repository
    monkeypatch.setattr(cloud, "current_date", lambda: dt.date(2026, 10, 16))
    result = cloud.record(repo, DATE, base, "12345", "0" * 64)
    assert result["ledger"] == f".research/cloud-runs/{DATE}.json"
    with pytest.raises(FileExistsError):
        cloud.record(repo, DATE, base, "12346", "0" * 64)
    assert cloud.gate(repo, "research", "true", "true", "2026-10-16T12:00:00Z")["reason"] == "already_published_today"
    with pytest.raises(cloud.Rejected, match="today"):
        cloud.record(repo, "2026-10-17", base, "12346", "0" * 64)
    with pytest.raises(cloud.Rejected, match="authorized"):
        cloud.record(repo, "2027-10-16", base, "12346", "0" * 64)
