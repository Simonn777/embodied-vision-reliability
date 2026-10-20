"""Fresh complete metric acquisition with persistent HTTPS, never request retry.

The unchanged RangeReader encloses connect/headers/body in its SIGALRM deadline
and checks response identity and bytes. This adapter supplies transport only.
No network, imports of old workers, or filesystem operations occur on import.
"""
import argparse
import hashlib
import http.client
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request

BASE = "experiments/0035-transformation-support/"
SELF = BASE + "metric-transport/reuse_https.py"
DECLARATION = BASE + "metric-transport/METRIC_TRANSPORT_ADMISSION_02.md"
ALLOCATION = BASE + "metric-transport/metric-allocation-02.json"
OLD_COMMIT = "784fc1a22d068dd7ce155129ac60d09d723f3f4d"
ACQUIRE_SHA = "a69768de68ac61585df7ceb077c48830a72c11dc88685472b2c0afd8f6b5341f"
FAILED_RECEIPT_SHA = "60dd51670ac4ad69b89dc5c299c086761db732264c2fa32e3a0f321b9181c5b3"
URL = "https://tri-ml-public.s3.amazonaws.com/datasets/VOST.zip"
HOST, TARGET = "tri-ml-public.s3.amazonaws.com", "/datasets/VOST.zip"
ETAG = '"656e2cc81e3dece60378b161992b3f1d-6439"'
EXPECTED = dict(schema="evr0035.metric-transport-allocation.v1", authorized=True,
    manifest_sha256="141371225d59c0b1ef483ee3b7565aace037033df072eabb2839e8d4f19bfb07",
    reservation_bytes=5000000, body_bytes=5000000, requests=1326,
    request_seconds=30, wall_seconds=1800, retries=0, expected_total=442,
    prior_balance=913590729, remaining_balance=908590729, refunds_taken=False,
    transport_mode="persistent_https_no_retry", parent_failed_receipt_sha256=FAILED_RECEIPT_SHA)
ARGUMENTS = ("repo_root", "expected_commit", "metadata_root", "manifest_relative",
             "admission_relative", "output_dir", "phase", "initial_dir",
             "initial_inspection", "initial_inspection_outer", "prediction_proof")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 64 * 1024**2,
            "bounded regular text input required")
    raw = path.read_bytes()
    require(len(raw) <= 64 * 1024**2, "input grew beyond bound")
    return raw


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    return json.loads(read(path), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


class PersistentHTTPS:
    """One active connection; server-closed connections reopen only for NEW requests."""
    def __init__(self, *, factory=None, proxies=None):
        self.factory = factory or http.client.HTTPSConnection
        configured = urllib.request.getproxies() if proxies is None else proxies
        self.proxy_keys = sorted(k for k, v in configured.items() if k != "no" and v)
        require(not self.proxy_keys, "configured proxies unsupported (keys only): " + ",".join(self.proxy_keys))
        self.connection = self.response = None
        self.opener_attempt_count = self.request_count = self.connection_count = self.connected_count = 0
        self.failed = self.closed = False

    def __call__(self, request, *, timeout):
        require(not self.closed and not self.failed, "transport closed or failed; retry forbidden")
        require(request.full_url == URL and request.get_method() == "GET" and request.data is None,
                "only exact frozen GET URL allowed")
        headers = {k.lower(): v for k, v in request.header_items()}
        require(set(headers) == {"range", "if-match", "accept-encoding"}
                and re.fullmatch(r"bytes=[0-9]+-[0-9]+", headers["range"]) is not None
                and headers["if-match"] == ETAG and headers["accept-encoding"] == "identity",
                "exact range request headers required")
        require(type(timeout) in (float, int) and math.isfinite(timeout) and 0 < timeout <= 30,
                "bounded request timeout required")
        require(self.opener_attempt_count < EXPECTED["requests"], "request count exceeded")
        require(self.response is None or self.response.isclosed(), "prior response not fully closed")
        self.opener_attempt_count += 1
        try:
            if self.connection is None or self.connection.sock is None:
                if self.connection is not None:
                    self.connection.close()
                self.connection = self.factory(HOST, port=443, timeout=timeout)
                self.connection.auto_open = 0  # No implicit reconnect/resend inside request().
                self.connection_count += 1  # Actual explicit connect() calls, including failures.
                self.connection.connect()
                self.connected_count += 1
            self.connection.timeout = timeout
            self.connection.sock.settimeout(timeout)
            self.request_count += 1  # request() invocations, not proof of remote receipt.
            self.connection.request("GET", TARGET, headers=dict(request.header_items()))
            self.response = self.connection.getresponse()
            return self.response  # Original RangeReader checks status/headers and consumes body.
        except BaseException:
            self.failed = True
            raise

    def close(self):
        try:
            if self.response is not None:
                self.response.close()
        finally:
            try:
                if self.connection is not None:
                    self.connection.close()
            finally:
                self.closed = True

    def metadata(self):
        return dict(mode="persistent_https_no_retry", no_retry=True, request_count=self.request_count,
                    opener_attempt_count=self.opener_attempt_count,
                    request_count_scope="HTTPConnection.request invocations including send failures",
                    connection_count=self.connection_count, successful_connections=self.connected_count,
                    proxy_policy="reject_configured_proxies", configured_proxy_keys=self.proxy_keys,
                    connection_closed=self.closed, request_deadline_owner="unchanged RangeReader",
                    new_reservation_bytes=5000000)


def source_binding(spec):
    root = Path(spec["repo_root"]).resolve()
    require(spec["expected_commit"] == OLD_COMMIT and spec["phase"] == "metric"
            and re.fullmatch("[a-f0-9]{40}", spec["adapter_source_commit"]) is not None,
            "exact old acquisition/new adapter commits and metric phase required")
    require(sha(read(root / (BASE + "acquire.py"))) == ACQUIRE_SHA, "old acquisition source changed")
    bindings = {}
    for name in (SELF, DECLARATION, ALLOCATION):
        raw = read(root / name)
        frozen = subprocess.check_output(["git", "show", spec["adapter_source_commit"] + ":" + name],
                                         cwd=root, timeout=15)
        require(raw == frozen, "adapter source differs from freeze: " + name)
        bindings[name] = sha(raw)
    require(read(__file__) == read(root / SELF), "executing adapter source differs")
    allocation = load(root / ALLOCATION)
    require(all(type(allocation.get(k)) is type(v) and allocation[k] == v for k, v in EXPECTED.items()),
            "exact prospective additional allocation required")
    module_spec = importlib.util.spec_from_file_location("_evr0035_metric_original", root / (BASE + "acquire.py"))
    acq = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = acq
    module_spec.loader.exec_module(acq)
    return root, acq, bindings


def parent_failure(spec):
    descriptors = spec["parent_failure"]
    require(set(descriptors) == {"receipt", "outer", "operation_spec"}, "complete failed parent descriptors")
    fingerprints, values = {}, {}
    for kind, desc in descriptors.items():
        path = Path(desc["path"])
        require(path.is_absolute(), "absolute parent path required")
        raw = read(path)
        require(sha(raw) == desc["sha256"], "failed parent input changed")
        fingerprints[str(path)] = desc["sha256"]
        values[kind] = json.loads(raw)
    receipt, outer = values["receipt"], values["outer"]
    require(descriptors["receipt"]["sha256"] == FAILED_RECEIPT_SHA
            and receipt["status"] == "failed" and receipt["phase"] == "metric"
            and len(receipt["members"]) == 45 and receipt["media_decoded"] is False,
            "exact preserved failed metric01 required")
    require(outer["status"] == "failed" and outer["exit_code"] == 1
            and outer["termination"] is None and not outer["remaining_descendant_pids"]
            and not outer["cleanup_errors"] and outer["output_sha256"] == FAILED_RECEIPT_SHA
            and outer["spec_sha256"] == descriptors["operation_spec"]["sha256"]
            and set(outer["log_sha256"]) == {"stdout.log", "stderr.log"},
            "failed parent outer/spec mismatch")
    for name, digest in outer["log_sha256"].items():
        path = Path(descriptors["outer"]["path"]).parent / name
        require(sha(read(path)) == digest, "failed parent log changed")
        fingerprints[str(path)] = digest
    return fingerprints


def execute(acq, args, pool, bindings, guard, metadata):
    """Inject only opener, added source bindings and a composed source postcheck."""
    old_phase, began, caught = acq.acquire_phase, time.monotonic(), None
    output = Path(args.output_dir)
    require(not output.exists() and not output.is_symlink(), "fresh whole-phase output required")
    def phase(*pos, **kw):
        require(kw.get("opener") is None, "unexpected existing opener")
        old_post = kw["postcheck"]
        def postcheck():
            old_post()
            guard()
        kw["bindings"] = dict(kw["bindings"], **bindings)
        kw.update(opener=pool, postcheck=postcheck)
        return old_phase(*pos, **kw)
    acq.acquire_phase = phase
    try:
        acq.run(args)
    except BaseException as exc:
        caught = exc
    finally:
        acq.acquire_phase = old_phase
        final_errors = []
        for action in (pool.close, guard):
            try:
                action()
            except BaseException as exc:
                final_errors.append(dict(type=type(exc).__name__, message=str(exc)))
        adapter = dict(metadata, **pool.metadata(), status="failed" if caught or final_errors else "completed",
                       wall_seconds=time.monotonic() - began, finalization_errors=final_errors)
        if caught:
            adapter["error"] = dict(type=type(caught).__name__, message=str(caught))
        # Old phase owns its original status/error/costs; append metadata only.
        receipt_path = output / "receipt.json"
        if receipt_path.exists():
            record = load(receipt_path)
            require("transport_adapter" not in record, "unexpected existing adapter metadata")
            record["transport_adapter"] = adapter
            temporary = output / "transport-receipt.pending"
            with temporary.open("x") as stream:
                json.dump(record, stream, indent=2, allow_nan=False)
                stream.write("\n")
            temporary.replace(receipt_path)
    if caught:
        raise caught
    require(not final_errors, "adapter finalization failed")
    return record


def run(spec_path):
    spec_raw = read(spec_path)
    spec = json.loads(spec_raw)
    require(spec.get("schema") == "evr0035.metric-transport-spec.v1", "adapter spec schema")
    root, acq, bindings = source_binding(spec)
    parent = parent_failure(spec)
    fingerprints = {str(root / name): digest for name, digest in bindings.items()}
    fingerprints.update(parent)
    fingerprints[str(Path(spec_path).absolute())] = sha(spec_raw)
    def guard():
        require(all(sha(read(p)) == digest for p, digest in fingerprints.items()),
                "adapter source/spec/failed parent changed")
    guard()
    pool = PersistentHTTPS()
    args = argparse.Namespace(**{k: spec[k] for k in ARGUMENTS})
    return execute(acq, args, pool, bindings, guard,
        dict(adapter_source_commit=spec["adapter_source_commit"], adapter_spec_sha256=sha(spec_raw),
             adapter_source_sha256=bindings, parent_failure=spec["parent_failure"],
             parent_fingerprint_sha256=parent, remaining_balance_bytes=908590729,
             full_population_restart=True, reused_members=0))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--spec", required=True)
    result = run(cli.parse_args().spec)
    print(json.dumps(dict(status=result["status"], members=len(result["members"]),
                          transport_adapter=result["transport_adapter"])))
