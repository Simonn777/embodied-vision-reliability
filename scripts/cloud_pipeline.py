#!/usr/bin/env python3
"""Trusted boundary for cloud proposals; never execute code from a proposal.

Run this file from the frozen base revision with ``python -I``. Validation uses
a temporary Git index, not candidate Python imports or candidate test results.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile
from zoneinfo import ZoneInfo


MAX_BYTES = 20 * 1024 * 1024  # Strictly below this bound.
MAX_FILES = 200
START = dt.date(2026, 10, 16)
END = dt.date(2027, 10, 15)
SHA = re.compile(r"[0-9a-f]{40}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
PROTECTED = {
    "scripts/cloud_pipeline.py", "tests/test_cloud_pipeline.py", "pyproject.toml",
    "docs/RESEARCH_CHARTER.md", "docs/CONTRIBUTIONS.md", "docs/CLOUD_EXECUTION.md",
}
TOP_FILES = {"README.md", "CURRENT_STATE.md", "NEXT_ACTIONS.md"}
DIRS = {"src", "scripts", "tests", "docs", "notes", "protocols", "results"}
SECRET_PATTERNS = (
    re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"\b(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{20,}"),
    re.compile(rb"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
)


class Rejected(ValueError):
    pass


def git(repo, *args, data=None, index=None, check=True):
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0")
    if index is not None:
        env["GIT_INDEX_FILE"] = str(index)
    result = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false",
         "-c", "core.autocrlf=false", *args], cwd=repo, env=env, input=data,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
    )
    if check and result.returncode:
        # Git diagnostics can contain hostile patch content; keep logs bounded.
        raise Rejected(f"Git operation {args[0]} failed (exit {result.returncode})")
    return result


def head(repo):
    return git(repo, "rev-parse", "HEAD").stdout.decode().strip()


def check_base(repo, base):
    if not SHA.fullmatch(base) or head(repo) != base:
        raise Rejected("The checkout HEAD must exactly match the manifest base SHA")


def parse_date(value):
    try:
        date = dt.date.fromisoformat(value)
    except (ValueError, TypeError):
        raise Rejected("Invalid ISO date") from None
    if date.isoformat() != value or not START <= date <= END:
        raise Rejected("Date is outside the authorized research year")
    return date


def allowed_path(value):
    parts = PurePosixPath(value).parts
    if not parts or value != "/".join(parts) or any(not COMPONENT.fullmatch(p) for p in parts):
        raise Rejected("Proposal contains an unsafe path")
    if any(p.casefold() in {"agents.md", "node_modules", "__pycache__"} for p in parts):
        raise Rejected("Proposal touches a protected path")
    if value in PROTECTED or (len(parts) == 1 and value not in TOP_FILES):
        raise Rejected("Proposal touches a protected file")
    if len(parts) > 1 and parts[0] not in DIRS:
        raise Rejected("Proposal directory is not allowed")
    if any(p.casefold().startswith(("requirements", "package-lock", "uv.lock", "poetry.lock")) for p in parts):
        raise Rejected("Dependency changes require a separately reviewed update")
    return value


def read_regular(path, limit=MAX_BYTES):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size >= limit:
        raise Rejected("Expected a bounded regular file")
    with path.open("rb") as handle:
        content = handle.read(limit)
    if len(content) >= limit:
        raise Rejected("File exceeds the size limit")
    return content


def tree_entries(repo, base):
    entries = {}
    for record in git(repo, "ls-tree", "-rz", base).stdout.split(b"\0"):
        if not record:
            continue
        metadata, name = record.split(b"\t", 1)
        mode, kind, digest = metadata.decode().split()
        entries[name.decode("utf-8")] = (mode, kind, digest)
    return entries


def check_ancestors(repo, path, old):
    for parent in PurePosixPath(path).parents:
        name = str(parent)
        if name == ".":
            continue
        if name in old or (repo / name).is_symlink():
            raise Rejected("A proposal path has a non-directory or symlink ancestor")


def check_content(repo, path, content, old, date):
    allowed_path(path)
    check_ancestors(repo, path, old)
    if path in old and old[path][0] != "100644":
        raise Rejected("Only ordinary non-executable files can change")
    if path in old and path.split("/")[0] in {"protocols", "results"}:
        raise Rejected("Previously published protocols and results are immutable")
    if path.startswith("results/"):
        parts = path.split("/")
        if len(parts) < 3:
            raise Rejected("New results must use a new run directory")
        prefix = "/".join(parts[:2]) + "/"
        if any(name.startswith(prefix) for name in old):
            raise Rejected("Previously published result run directories are immutable")
    if path.startswith("notes/"):
        if path != f"notes/{date}.md":
            raise Rejected("Only the current day's note may be added or appended")
        if path in old:
            previous = git(repo, "cat-file", "blob", old[path][2]).stdout
            if not content.startswith(previous):
                raise Rejected("Existing notes must be preserved byte for byte as a prefix")
    if any(pattern.search(content) for pattern in SECRET_PATTERNS):
        raise Rejected("Credential-like content is not permitted in proposals")


def patch_paths(patch):
    """Reject ambiguous headers before Git processes the binary/text patch."""
    if not patch:
        return set()
    if len(patch) >= MAX_BYTES or not patch.startswith(b"diff --git "):
        raise Rejected("Patch is too large or is not a canonical Git diff")
    current = None
    paths = set()
    literal_total = 0
    for line in patch.splitlines():
        if line.startswith(b"diff --git "):
            match = re.fullmatch(rb"diff --git a/([^\s]+) b/([^\s]+)", line)
            if not match or match[1] != match[2]:
                raise Rejected("Quoted, renamed, or ambiguous patch paths are forbidden")
            try:
                current = allowed_path(match[1].decode("ascii"))
            except UnicodeDecodeError:
                raise Rejected("Cloud proposal paths must use ASCII") from None
            if current in paths:
                raise Rejected("Duplicate patch sections are forbidden")
            paths.add(current)
        elif line.startswith((b"--- ", b"+++ ")):
            if current is None:
                raise Rejected("Missing patch header")
            expected = (b"a/" if line.startswith(b"--- ") else b"b/") + current.encode()
            if line[4:] != expected and not (line.startswith(b"--- ") and line[4:] == b"/dev/null"):
                raise Rejected("Patch file headers do not match the allowed path")
        elif line.startswith((b"deleted file mode", b"old mode", b"new mode", b"rename ", b"copy ", b"delta ")):
            raise Rejected("Deletes, mode changes, renames, copies, and binary deltas are forbidden")
        elif line.startswith(b"new file mode ") and line != b"new file mode 100644":
            raise Rejected("New files must have mode 100644")
        elif line.startswith(b"literal "):
            try:
                size = int(line[8:])
            except ValueError:
                raise Rejected("Invalid binary literal size") from None
            if not 0 <= size < MAX_BYTES:
                raise Rejected("Expanded binary patch exceeds the file-size bound")
            literal_total += size
            if literal_total >= MAX_BYTES:
                raise Rejected("Expanded binary patch literals reach the cumulative size bound")
    if not paths or len(paths) > MAX_FILES:
        raise Rejected("Invalid changed-file count")
    return paths


def validate(repo, bundle, base, date, preflight=False, apply=False):
    check_base(repo, base)
    parse_date(date)
    if bundle.is_symlink() or not bundle.is_dir() or {p.name for p in bundle.iterdir()} != {"manifest.json", "changes.patch"}:
        raise Rejected("Bundle must contain exactly manifest.json and changes.patch")
    manifest_bytes = read_regular(bundle / "manifest.json", 8192)
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise Rejected("Duplicate manifest keys")
            result[key] = value
        return result
    manifest = json.loads(manifest_bytes, object_pairs_hook=unique_pairs)
    patch = read_regular(bundle / "changes.patch")
    if len(manifest_bytes) + len(patch) >= MAX_BYTES:
        raise Rejected("The complete proposal bundle reaches the 20 MiB limit")
    expected = {"schema_version": 1, "base_sha": base, "date": date,
                "patch_sha256": hashlib.sha256(patch).hexdigest(),
                "mode": "preflight" if preflight else "research"}
    if manifest != expected or type(manifest.get("schema_version")) is not int:
        raise Rejected("Manifest, expected base/date/mode, and patch digest must match")
    paths = patch_paths(patch)
    if not paths and not preflight:
        raise Rejected("Empty research proposals are not publishable")
    if any(p.startswith("protocols/") for p in paths) and any(p.startswith("results/") for p in paths):
        raise Rejected("Publish a new protocol in a prior run before adding results")
    if any(p.startswith(("src/", "scripts/")) for p in paths) and any(p.startswith("results/") for p in paths):
        raise Rejected("Publish source changes in a prior run before adding results")
    old = tree_entries(repo, base)
    total = 0
    with tempfile.TemporaryDirectory(prefix="evr-index-") as directory:
        index = Path(directory) / "index"
        git(repo, "read-tree", base, index=index)
        if patch:
            git(repo, "apply", "--cached", "--binary", "--check", "-", data=patch, index=index)
            git(repo, "apply", "--cached", "--binary", "-", data=patch, index=index)
        raw = git(repo, "diff", "--cached", "--raw", "--no-abbrev", "--no-ext-diff", "--no-renames", "-z", base, index=index).stdout.split(b"\0")
        actual = set()
        for position in range(0, len(raw) - 1, 2):
            metadata, name = raw[position], raw[position + 1].decode("utf-8")
            old_mode, new_mode, _old_sha, new_sha, status = metadata.decode().lstrip(":").split()
            if status not in {"A", "M"} or new_mode != "100644" or old_mode not in {"000000", "100644"}:
                raise Rejected("Only added or modified ordinary files are allowed")
            size = int(git(repo, "cat-file", "-s", new_sha).stdout)
            total += size
            if total >= MAX_BYTES:
                raise Rejected("Expanded changed-file content reaches the 20 MiB limit")
            content = git(repo, "cat-file", "blob", new_sha).stdout
            check_content(repo, name, content, old, date)
            actual.add(name)
        if actual != paths:
            raise Rejected("Patch headers and actual changed tree disagree")
    if apply and patch:
        if git(repo, "diff", "--no-ext-diff", "--quiet", "HEAD", check=False).returncode:
            raise Rejected("Apply requires an unchanged tracked worktree and index")
        git(repo, "apply", "--index", "--binary", "-", data=patch)
    return {"valid": True, "base_sha": base, "date": date, "patch_sha256": expected["patch_sha256"],
            "paths": sorted(paths), "file_count": len(paths), "expanded_bytes": total,
            "patch_bytes": len(patch), "mode": expected["mode"], "has_changes": bool(paths)}


def bundle(repo, destination, base, date, preflight=False, allow_no_change=False):
    check_base(repo, base)
    parse_date(date)
    if destination.exists():
        raise Rejected("Bundle destination already exists")
    old = tree_entries(repo, base)
    changed = set(filter(None, git(repo, "diff", "--name-only", "--no-ext-diff", "--no-renames", "-z", base).stdout.decode().split("\0")))
    changed.update(filter(None, git(repo, "ls-files", "--others", "--exclude-standard", "-z").stdout.decode().split("\0")))
    if len(changed) > MAX_FILES:
        raise Rejected("Too many changed files")
    total = 0
    with tempfile.TemporaryDirectory(prefix="evr-bundle-") as directory:
        index = Path(directory) / "index"
        git(repo, "read-tree", base, index=index)
        for name in sorted(changed):
            allowed_path(name)
            path = repo / name
            if not path.exists() or path.is_symlink():
                raise Rejected("Deletions and symlinks are forbidden")
            if path.stat().st_mode & 0o111:
                raise Rejected("Executable files are forbidden")
            content = read_regular(path)
            check_content(repo, name, content, old, date)
            total += len(content)
            if total >= MAX_BYTES:
                raise Rejected("Changed-file content reaches the 20 MiB limit")
            digest = git(repo, "hash-object", "-w", "--stdin", data=content).stdout.decode().strip()
            git(repo, "update-index", "--add", "--cacheinfo", f"100644,{digest},{name}", index=index)
        patch = git(repo, "diff", "--cached", "--binary", "--full-index", "--no-ext-diff", "--no-renames", base, index=index).stdout
    if not patch and not preflight:
        if allow_no_change:
            return {"has_changes": False, "reason": "no_material_change", "base_sha": base, "date": date}
        raise Rejected("No substantive file change; refusing an empty research bundle")
    destination.mkdir(parents=True)
    (destination / "changes.patch").write_bytes(patch)
    manifest = {"schema_version": 1, "base_sha": base, "date": date,
                "patch_sha256": hashlib.sha256(patch).hexdigest(),
                "mode": "preflight" if preflight else "research"}
    (destination / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return validate(repo, destination, base, date, preflight)


def gate(repo, mode, enabled, has_api_key, now=None):
    moment = dt.datetime.fromisoformat(now.replace("Z", "+00:00")) if now else dt.datetime.now(dt.timezone.utc)
    if moment.tzinfo is None:
        raise Rejected("Gate time must include a timezone")
    date = moment.astimezone(ZoneInfo("America/New_York")).date()
    eligible = START <= date <= END
    reason = "ready" if eligible else "outside_authorized_dates"
    if eligible and (repo / ".research" / "cloud-runs" / f"{date.isoformat()}.json").exists():
        eligible, reason = False, "already_published_today"
    if eligible and mode == "research" and enabled != "true":
        eligible, reason = False, "research_disabled"
    if eligible and mode == "research" and has_api_key != "true":
        eligible, reason = False, "api_key_missing"
    return {"eligible": eligible, "run_ai": eligible and mode == "research",
            "run_publish": eligible and mode == "research", "date": date.isoformat(),
            "base_sha": head(repo), "reason": reason, "mode": mode}


def record(repo, date, base, run_id, patch_sha256):
    parse_date(date)
    if date != current_date().isoformat():
        raise Rejected("Publication date must equal today's date in America/New_York")
    check_base(repo, base)
    if not re.fullmatch(r"[0-9]+", run_id) or not SHA256.fullmatch(patch_sha256):
        raise Rejected("Invalid publication ledger identifiers")
    destination = repo / ".research" / "cloud-runs" / f"{date}.json"
    if any(parent.is_symlink() for parent in destination.parents if parent != repo.parent):
        raise Rejected("Ledger parent must not be a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    value = {"schema_version": 1, "date": date, "base_sha": base,
             "run_id": run_id, "patch_sha256": patch_sha256,
             "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    with destination.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {"ledger": destination.relative_to(repo).as_posix(), **value}


def current_date():
    return dt.datetime.now(dt.timezone.utc).astimezone(ZoneInfo("America/New_York")).date()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("bundle", "validate", "apply", "gate", "record"):
        sub = commands.add_parser(name)
        sub.add_argument("--repo", type=Path, default=Path.cwd())
        if name in {"bundle", "validate", "apply"}:
            sub.add_argument("--output" if name == "bundle" else "--bundle", type=Path, required=True)
            sub.add_argument("--base-sha", required=True)
            sub.add_argument("--date", required=True)
            sub.add_argument("--preflight", action="store_true")
            if name == "bundle":
                sub.add_argument("--allow-no-change", action="store_true")
                sub.add_argument("--github-output", type=Path)
        elif name == "gate":
            sub.add_argument("--event", choices=("schedule", "workflow_dispatch"), default="workflow_dispatch")
            sub.add_argument("--mode", choices=("preflight", "research"), required=True)
            sub.add_argument("--enabled", default="false")
            sub.add_argument("--has-api-key", default="false")
            sub.add_argument("--now")
            sub.add_argument("--github-output", type=Path)
        else:
            sub.add_argument("--date", required=True)
            sub.add_argument("--base-sha", required=True)
            sub.add_argument("--run-id", required=True)
            sub.add_argument("--patch-sha256", required=True)
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    try:
        if args.command == "gate":
            result = gate(repo, args.mode, args.enabled, args.has_api_key, args.now)
        elif args.command == "record":
            result = record(repo, args.date, args.base_sha, args.run_id, args.patch_sha256)
        elif args.command == "bundle":
            result = bundle(repo, args.output.resolve(), args.base_sha, args.date, args.preflight, args.allow_no_change)
        else:
            result = validate(repo, args.bundle.resolve(), args.base_sha, args.date,
                              args.preflight, apply=args.command == "apply")
        if getattr(args, "github_output", None):
            with args.github_output.open("a") as output:
                for key, value in result.items():
                    encoded = json.dumps(value, separators=(",", ":")) if isinstance(value, (bool, list, dict)) else str(value)
                    output.write(f"{key}={encoded}\n")
        print(json.dumps(result, sort_keys=True))
        return 0
    except (Rejected, OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"Cloud proposal rejected: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
