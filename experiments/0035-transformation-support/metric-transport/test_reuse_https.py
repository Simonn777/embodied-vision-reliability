"""Tiny manufactured transports/receipts only; no socket or original payload reads."""
import contextlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
import urllib.request
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


adapter = module("_test_metric_https", HERE / "reuse_https.py")


def request(url=adapter.URL, method="GET"):
    return urllib.request.Request(url, method=method, headers={
        "Range": "bytes=0-2", "If-Match": adapter.ETAG, "Accept-Encoding": "identity"})


class Response:
    def __init__(self, *, status=206):
        self.status, self.closed, self.body, self.reads = status, False, b"abc", 0
        self.headers = {"ETag": adapter.ETAG, "Content-Range": "bytes 0-2/100",
                        "Content-Length": "3"}

    def isclosed(self):
        return self.closed

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def read(self, count):
        self.reads += 1
        chunk, self.body = self.body[:count], self.body[count:]
        return chunk


class Connection:
    def __init__(self, host, *, port, timeout):
        self.host, self.port, self.timeout = host, port, timeout
        self.sock, self.connects, self.calls = None, 0, []
        self.reply, self.server_closes = Response(), False
        self.connect_error = self.send_error = None

    def connect(self):
        self.connects += 1
        if self.connect_error:
            raise self.connect_error
        self.sock = Mock()

    def request(self, method, target, *, headers):
        self.calls.append((method, target, headers))
        if self.send_error:
            raise self.send_error

    def getresponse(self):
        if self.server_closes:
            self.sock = None
        return self.reply

    def close(self):
        self.sock = None


class Tests(unittest.TestCase):
    def pool(self):
        made = []
        def factory(*args, **kw):
            made.append(Connection(*args, **kw))
            return made[-1]
        return adapter.PersistentHTTPS(factory=factory, proxies={}), made

    def test_reuse_and_exact_requests(self):
        pool, made = self.pool()
        pool(request(), timeout=30).close()
        made[0].reply = Response()
        pool(request(), timeout=4.5).close()
        self.assertEqual(len(made), 1)
        self.assertEqual((pool.connection_count, pool.request_count), (1, 2))
        self.assertEqual(made[0].auto_open, 0)
        self.assertEqual(made[0].calls[0][:2], ("GET", "/datasets/VOST.zip"))
        self.assertEqual(dict((k.lower(), v) for k, v in made[0].calls[0][2].items()),
                         {"range": "bytes=0-2", "if-match": adapter.ETAG, "accept-encoding": "identity"})
        made[0].sock.settimeout.assert_called_with(4.5)
        pool.close()
        self.assertTrue(pool.metadata()["connection_closed"])

    def test_server_close_reconnects_only_for_next_request(self):
        pool, made = self.pool()
        original = pool.factory
        def factory(*a, **kw):
            connection = original(*a, **kw)
            connection.server_closes = True
            return connection
        pool.factory = factory
        for _ in range(2):
            pool(request(), timeout=30).close()
        self.assertEqual((pool.connection_count, pool.request_count, len(made)), (2, 2, 2))
        self.assertEqual([len(c.calls) for c in made], [1, 1])
        pool.close()

    def test_connect_and_send_failure_never_resend(self):
        for field, expected_requests in (("connect_error", 0), ("send_error", 1)):
            pool, made = self.pool()
            original = pool.factory
            def factory(*a, **kw):
                conn = original(*a, **kw)
                setattr(conn, field, ConnectionResetError("manufactured reset"))
                return conn
            pool.factory = factory
            with self.assertRaises(ConnectionResetError):
                pool(request(), timeout=30)
            with self.assertRaisesRegex(ValueError, "retry forbidden"):
                pool(request(), timeout=30)
            self.assertEqual((pool.connection_count, pool.request_count, pool.opener_attempt_count),
                             (1, expected_requests, 1))
            self.assertEqual(len(made), 1)
            pool.close()

    def test_proxy_and_request_validation_precedes_connect(self):
        with self.assertRaisesRegex(ValueError, "https") as caught:
            adapter.PersistentHTTPS(proxies={"https": "secret-proxy-value"})
        self.assertNotIn("secret-proxy-value", str(caught.exception))
        pool, made = self.pool()
        for value, timeout in ((request("https://invalid.example/a"), 30),
                               (request(method="POST"), 30), (request(), 31)):
            with self.assertRaises(ValueError):
                pool(value, timeout=timeout)
        self.assertFalse(made)
        pool.close()

    def test_original_range_deadline_and_bad_headers_still_apply(self):
        relative = "experiments/0023-mapfree/range_inputs.py"
        root = next((p for p in HERE.parents if (p / relative).is_file()),
                    HERE.parents[1] / "embodied-vision-reliability")
        helper = module("_metric_fake_range", root / relative)
        pool, made = self.pool()
        active = []
        @contextlib.contextmanager
        def deadline(seconds):
            active.append(seconds)
            try:
                yield
            finally:
                active.pop()
        original_factory = pool.factory
        def factory(*a, **kw):
            self.assertEqual(active, [30])
            return original_factory(*a, **kw)
        pool.factory = factory
        index = helper.Index(adapter.URL, adapter.ETAG, 100, 90, {}, {})
        limits = helper.Limits(1000, 1000, 100, 10, 1800, 30)
        record = dict(requests=[], response_body_bytes=0, response_body_bytes_exact=True,
                      requested_body_upper_bound=0)
        reader = helper.RangeReader(index, limits, record, lambda: None, pool)
        with patch.object(helper, "deadline", deadline):
            self.assertEqual(reader.read(0, 2), b"abc")
            made[0].reply = Response(status=302)
            with self.assertRaisesRegex(ValueError, "HTTP identity"):
                reader.read(0, 2)
        self.assertEqual(record["response_body_bytes"], 3)
        self.assertEqual(made[0].reply.reads, 0)
        self.assertEqual([r["status"] for r in record["requests"]], [206, 302])
        pool.close()

    def test_wrapper_preserves_old_status_and_attaches_sources_on_both_outcomes(self):
        for fail in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "new"
                original = dict(status="failed" if fail else "completed", phase="metric",
                                response_body_bytes=7, members=[], error="old error" if fail else None)
                old_bindings, guard, pool = {"old": "a"}, Mock(), self.pool()[0]
                def old_phase(*pos, **kw):
                    self.assertEqual(kw["bindings"], {"old": "a", "adapter": "b"})
                    self.assertIs(kw["opener"], pool)
                    output.mkdir()
                    (output / "receipt.json").write_text(json.dumps(original))
                    if fail:
                        raise ConnectionResetError("manufactured")
                    kw["postcheck"]()
                acq = types.SimpleNamespace(acquire_phase=old_phase)
                def old_run(args):
                    return acq.acquire_phase(bindings=old_bindings, postcheck=lambda: None)
                acq.run = old_run
                invoke = lambda: adapter.execute(acq, types.SimpleNamespace(output_dir=str(output)), pool,
                                                  {"adapter": "b"}, guard, {})
                if fail:
                    with self.assertRaises(ConnectionResetError):
                        invoke()
                else:
                    invoke()
                actual = json.loads((output / "receipt.json").read_text())
                self.assertEqual({k: actual[k] for k in original}, original)
                self.assertEqual(actual["transport_adapter"]["status"], original["status"])
                self.assertTrue(actual["transport_adapter"]["connection_closed"])
                self.assertEqual(old_bindings, {"old": "a"})
                self.assertIs(acq.acquire_phase, old_phase)
                self.assertGreaterEqual(guard.call_count, 1)

    def test_final_source_failure_is_not_clean_even_if_old_phase_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "new"
            def old_run(_):
                output.mkdir()
                (output / "receipt.json").write_text('{"status":"completed"}')
            acq = types.SimpleNamespace(acquire_phase=Mock(), run=old_run)
            with self.assertRaisesRegex(ValueError, "finalization failed"):
                adapter.execute(acq, types.SimpleNamespace(output_dir=str(output)), self.pool()[0], {},
                                Mock(side_effect=ValueError("source changed")), {})
            saved = json.loads((output / "receipt.json").read_text())
            self.assertEqual(saved["status"], "completed")
            self.assertEqual(saved["transport_adapter"]["status"], "failed")
            self.assertEqual(saved["transport_adapter"]["finalization_errors"][0]["message"], "source changed")


if __name__ == "__main__":
    unittest.main()
