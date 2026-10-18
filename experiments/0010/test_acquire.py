"""Only invented byte strings, in-process fake responses and temporary files."""
import hashlib
import http.client
import io
import json
import tempfile
import unittest
import urllib.error
from email.message import Message
from pathlib import Path
from unittest.mock import patch
import acquire


class Response:
    def __init__(self, body=b'', status=200, headers=(), error=None, chunk=None):
        self.body, self.status, self.pos = body, status, 0
        self.headers = Message()
        for k, v in headers: self.headers[k] = v
        self.error, self.chunk, self.closed, self.reads = error, chunk, False, []
    def getcode(self): return self.status
    def read(self, n):
        self.reads.append(n)
        if self.error is not None and self.pos == len(self.body):
            e, self.error = self.error, None
            raise e
        take = min(n, self.chunk or n, len(self.body) - self.pos)
        b = self.body[self.pos:self.pos + take]; self.pos += take
        return b
    def close(self): self.closed = True


class Opener:
    def __init__(self, *responses): self.responses, self.calls = list(responses), []
    def open(self, req, timeout):
        self.calls.append((req.full_url, timeout, req.headers))
        response = self.responses.pop(0)
        if isinstance(response, Exception): raise response
        return response


def manifest(body=b'example', **changes):
    asset = dict(id='toy', filename='toy.bin', url='https://a.example/asset?secret=abc',
                 allowed_hosts=['a.example', 'b.example'], bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
    asset.update(changes)
    return {'body_cap_bytes': 1024 * 1024, 'assets': [asset]}


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='fixture-', dir=Path(__file__).parent)
        self.root = Path(self.tmp.name) / 'output'
    def tearDown(self): self.tmp.cleanup()
    def run_it(self, m, opener, **kw):
        # Any accidental non-injected network route fails locally.
        with patch('urllib.request.urlopen', side_effect=AssertionError('network forbidden')):
            return acquire.acquire_all(m, self.root, opener=opener, **kw)
    def failed(self, m, o, reason, **kw):
        with self.assertRaisesRegex(acquire.AcquisitionError, reason): self.run_it(m, o, **kw)
        return json.loads((self.root / 'ledger.json').read_text()) if (self.root / 'ledger.json').exists() else None
    def test_success_sha_and_exact_sentinel(self):
        r = Response(b'example', headers=[('Content-Length', '7')]); o = Opener(r)
        l = self.run_it(manifest(), o)
        self.assertEqual(l['status'], 'completed'); self.assertEqual(l['actual_body_bytes'], 7)
        self.assertEqual(r.reads, [8, 1]); self.assertTrue(r.closed)
        self.assertEqual((self.root / 'toy.bin').read_bytes(), b'example')
        self.assertFalse((self.root / 'toy.bin.partial').exists())
        self.assertTrue((self.root / 'operation.started.json').exists())
    def test_git_blob_digest(self):
        m = manifest(); a = m['assets'][0]; del a['sha256']
        a['git_blob_sha1'] = hashlib.sha1(b'blob 7\0example').hexdigest()
        self.assertEqual(self.run_it(m, Opener(Response(b'example')))['status'], 'completed')
    def test_redirect_body_count_and_private_location(self):
        o = Opener(Response(b'move', 302, [('Location', 'https://b.example/signed-secret-path?token=SECRET')]), Response(b'example'))
        l = self.run_it(manifest(), o)
        self.assertEqual(l['actual_body_bytes'], 11); self.assertEqual(len(l['requests']), 2)
        self.assertEqual(l['requests'][0]['bytes_read'], 4)
        for filename in ('ledger.json', 'events.jsonl', 'operation.started.json'):
            text = (self.root / filename).read_text()
            for secret in ('SECRET', 'signed-secret-path', 'secret=abc', 'token='): self.assertNotIn(secret, text)
    def test_relative_redirect(self):
        o = Opener(Response(b'', 307, [('Location', '/new')]), Response(b'example'))
        self.run_it(manifest(), o); self.assertEqual(o.calls[1][0], 'https://a.example/new')
    def test_maximum_four_redirects(self):
        o = Opener(*[Response(b'x', 302, [('Location', '/next')]) for _ in range(5)])
        l = self.failed(manifest(), o, 'redirect_limit')
        self.assertEqual(len(o.calls), 5); self.assertEqual(l['actual_body_bytes'], 5)
    def test_disallowed_redirect_host(self):
        o = Opener(Response(b'notice', 302, [('Location', 'https://bad.example/x')]))
        l = self.failed(manifest(), o, 'url_scope')
        self.assertEqual(len(o.calls), 1); self.assertEqual(l['actual_body_bytes'], 6)
    def test_https_downgrade(self):
        self.failed(manifest(), Opener(Response(b'', 302, [('Location', 'http://a.example/x')])), 'url_scope')
    def test_url_credentials(self):
        o = Opener(); self.failed(manifest(url='https://user:pw@a.example/x'), o, 'url_scope')
        self.assertFalse(o.calls)
    def test_error_response_count(self):
        r = Response(b'error-payload', 500); o = Opener(r)
        l = self.failed(manifest(), o, 'unexpected_http_status')
        self.assertEqual(l['actual_body_bytes'], 13); self.assertTrue(r.closed)
    def test_http_error_is_read_not_retried(self):
        h = Message(); h['Content-Length'] = '4'
        e = urllib.error.HTTPError('https://a.example/SECRET', 403, 'SECRET', h, io.BytesIO(b'nope'))
        o = Opener(e); l = self.failed(manifest(), o, 'unexpected_http_status')
        self.assertEqual(len(o.calls), 1); self.assertEqual(l['actual_body_bytes'], 4)
        self.assertNotIn('SECRET', json.dumps(l))
    def test_non_success_body_limit(self):
        r = Response(b'x' * (acquire.BODY_CAP + 10), 302, [('Location', '/next')]); o = Opener(r)
        l = self.failed(manifest(), o, 'non_success_body_limit')
        self.assertEqual(l['actual_body_bytes'], acquire.BODY_CAP)
        self.assertEqual(r.pos, acquire.BODY_CAP); self.assertEqual(len(o.calls), 1)
        self.assertFalse(l['requests'][0]['body_complete'])
    def test_total_cap_limits_reads_not_only_final_check(self):
        m = manifest(b'abc'); m['body_cap_bytes'] = 16
        r = Response(b'x' * 100, 500); o = Opener(r)
        l = self.failed(m, o, 'body_budget_exhausted')
        self.assertEqual(l['actual_body_bytes'], 16); self.assertEqual(r.reads, [16]); self.assertEqual(r.pos, 16)
    def test_all_asset_budget_preflight(self):
        m = manifest(b'abc'); m['body_cap_bytes'] = 7
        m['assets'].append(dict(m['assets'][0], id='two', filename='two.bin'))
        o = Opener(); l = self.failed(m, o, 'all_assets_budget')
        self.assertEqual(l['actual_body_bytes'], 0); self.assertFalse(o.calls)
    def test_overflow_preserved(self):
        o = Opener(Response(b'exampleX'))
        l = self.failed(manifest(), o, 'body_overflow')
        self.assertEqual(l['actual_body_bytes'], 8); self.assertEqual((self.root / 'toy.bin.partial').read_bytes(), b'exampleX')
        self.assertFalse((self.root / 'toy.bin').exists())
    def test_underflow_preserved(self):
        self.failed(manifest(), Opener(Response(b'ex')), 'body_underflow')
        self.assertEqual((self.root / 'toy.bin.partial').read_bytes(), b'ex')
    def test_digest_mismatch_preserved(self):
        l = self.failed(manifest(), Opener(Response(b'EXAMPLE')), 'body_digest_mismatch')
        self.assertEqual(l['assets'][0]['partial_sha256'], hashlib.sha256(b'EXAMPLE').hexdigest())
        self.assertEqual((self.root / 'toy.bin.partial').read_bytes(), b'EXAMPLE')
    def test_existing_partial_rejected_before_get(self):
        self.root.mkdir(); (self.root / 'toy.bin.partial').write_bytes(b'ex')
        o = Opener(); self.failed(manifest(), o, 'existing_partial')
        self.assertFalse(o.calls); self.assertEqual((self.root / 'toy.bin.partial').read_bytes(), b'ex')
    def test_existing_bad_cache_not_replaced(self):
        self.root.mkdir(); (self.root / 'toy.bin').write_bytes(b'EXAMPLE')
        o = Opener(); self.failed(manifest(), o, 'existing_target_mismatch')
        self.assertFalse(o.calls); self.assertEqual((self.root / 'toy.bin').read_bytes(), b'EXAMPLE')
    def test_verified_cache_zero_get(self):
        self.root.mkdir(); (self.root / 'toy.bin').write_bytes(b'example')
        o = Opener(); l = self.run_it(manifest(), o)
        self.assertFalse(o.calls); self.assertEqual(l['actual_body_bytes'], 0)
        self.assertEqual(l['assets'][0]['status'], 'reused_verified')
    def test_persistent_marker_blocks_second_attempt(self):
        self.failed(manifest(), Opener(Response(b'bad')), 'body_underflow')
        before = (self.root / 'ledger.json').read_bytes(); o = Opener()
        self.failed(manifest(), o, 'existing_operation')
        self.assertFalse(o.calls); self.assertEqual((self.root / 'ledger.json').read_bytes(), before)
    def test_duplicate_length_reject(self):
        r = Response(b'example', headers=[('Content-Length', '7'), ('content-length', '7')])
        l = self.failed(manifest(), Opener(r), 'duplicate_content-length')
        self.assertEqual(l['actual_body_bytes'], 0); self.assertFalse(l['requests'][0]['body_complete'])
    def test_duplicate_encoding_reject(self):
        self.failed(manifest(), Opener(Response(b'example', headers=[('Content-Encoding', 'identity')] * 2)), 'duplicate_content-encoding')
    def test_gzip_not_accepted(self):
        self.failed(manifest(), Opener(Response(b'example', headers=[('Content-Encoding', 'gzip')])), 'content_encoding')
    def test_wrong_length_reject_before_body(self):
        r = Response(b'example', headers=[('Content-Length', '8')])
        self.failed(manifest(), Opener(r), 'unexpected_content_length'); self.assertFalse(r.reads)
    def test_bad_error_headers_still_count_bounded_body(self):
        r = Response(b'abc', 500, [('Content-Length', '3'), ('Content-Length', '3')])
        l = self.failed(manifest(), Opener(r), 'duplicate_content-length')
        self.assertEqual(l['actual_body_bytes'], 3)
    def test_partial_read_exception_accounts_exposed_bytes(self):
        r = Response(b'ex', error=http.client.IncompleteRead(b'amp', 2)); o = Opener(r)
        l = self.failed(manifest(), o, 'incomplete_read')
        self.assertEqual(l['actual_body_bytes'], 5)
        self.assertEqual((self.root / 'toy.bin.partial').read_bytes(), b'ex')
        self.assertEqual(l['assets'][0]['partial_bytes'], 2)
    def test_unknown_read_failure_does_not_claim_complete_body(self):
        r = Response(b'ex', error=OSError('secret-url')); l = self.failed(manifest(), Opener(r), 'read_failure')
        self.assertEqual(l['actual_body_bytes'], 2); self.assertTrue(l['requests'][0]['unobserved_body_bytes_possible'])
        self.assertNotIn('secret-url', json.dumps(l))
    def test_open_error_single_attempt_and_redacted(self):
        o = Opener(OSError('https://a.example/?SECRET'))
        l = self.failed(manifest(), o, 'OSError'); self.assertEqual(len(o.calls), 1)
        self.assertNotIn('SECRET', json.dumps(l)); self.assertEqual(l['actual_body_bytes'], 0)
    def test_deadline_before_request(self):
        values = iter([0., 841.]); o = Opener()
        l = self.failed(manifest(), o, 'asset_deadline', clock=lambda: next(values))
        self.assertFalse(o.calls); self.assertEqual(l['actual_body_bytes'], 0)
    def test_deadline_after_request_log_before_open(self):
        values = iter([0., 0., 841.]); o = Opener()
        l = self.failed(manifest(), o, 'asset_deadline', clock=lambda: next(values))
        self.assertFalse(o.calls); self.assertFalse(l['requests'][0]['open_attempted'])
    def test_symlink_target_reject(self):
        self.root.mkdir(); (self.root / 'outside').write_bytes(b'example')
        (self.root / 'toy.bin').symlink_to(self.root / 'outside')
        o = Opener(); self.failed(manifest(), o, 'symlink'); self.assertFalse(o.calls)
    def test_atomic_publish_never_overwrites_racing_target(self):
        old_link = acquire.os.link
        def race(src, dst):
            Path(dst).write_bytes(b'other-producer')
            return old_link(src, dst)
        with patch.object(acquire.os, 'link', side_effect=race):
            self.failed(manifest(), Opener(Response(b'example')), 'FileExistsError')
        self.assertEqual((self.root / 'toy.bin').read_bytes(), b'other-producer')
        self.assertEqual((self.root / 'toy.bin.partial').read_bytes(), b'example')
    def test_no_redirect_handler_never_constructs_followup(self):
        self.assertIsNone(acquire.NoRedirect().redirect_request(None, None, 302, '', Message(), 'https://b.example'))
    def test_fragment_and_port_rejected_before_request(self):
        for url in ('https://a.example/x#secret', 'https://a.example:444/x'):
            with self.subTest(url=url):
                with self.assertRaises(acquire.AcquisitionError): acquire.validate(manifest(url=url))


if __name__ == '__main__': unittest.main(verbosity=2)
