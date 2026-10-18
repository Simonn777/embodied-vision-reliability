"""Prospective, stdlib-only asset acquisition; no implicit calls on import."""
import hashlib
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CAP = 512 * 1024 * 1024
BODY_CAP = 64 * 1024


class AcquisitionError(RuntimeError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fail(reason):
    raise AcquisitionError(reason)


def url_parts(url, hosts):
    p = urllib.parse.urlsplit(url)
    if (p.scheme != 'https' or p.hostname not in hosts or p.username is not None
            or p.password is not None or p.fragment or p.port not in (None, 443)):
        fail('url_scope')
    return p


def safe_url(url):
    return {'host': urllib.parse.urlsplit(url).hostname, 'path': '<redacted>'}


def validate(manifest):
    cap = manifest.get('body_cap_bytes', CAP)
    if type(cap) is not int or not 0 < cap <= CAP:
        fail('invalid_budget')
    assets = manifest.get('assets')
    if not isinstance(assets, list) or not assets:
        fail('invalid_assets')
    names, ids = set(), set()
    for a in assets:
        for key in ('id', 'filename'):
            if not isinstance(a.get(key), str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', a[key]):
                fail('invalid_name')
        if a['id'] in ids or a['filename'] in names:
            fail('duplicate_asset')
        ids.add(a['id']); names.add(a['filename'])
        if type(a.get('bytes')) is not int or not 0 <= a['bytes'] < cap:
            fail('invalid_size')
        hashes = [k for k in ('sha256', 'git_blob_sha1') if k in a]
        if not hashes or any(not isinstance(a[k], str) or not re.fullmatch('[0-9a-f]{%d}' % (64 if k == 'sha256' else 40), a[k]) for k in hashes):
            fail('invalid_digest')
        hosts = a.get('allowed_hosts')
        if not isinstance(hosts, list) or not hosts or any(not isinstance(h, str) or not re.fullmatch(r'[a-z0-9.-]+', h) for h in hosts):
            fail('invalid_hosts')
        url_parts(a['url'], hosts)
    return cap


def digests(size):
    s = hashlib.sha1(b'blob ' + str(size).encode() + b'\0')
    return hashlib.sha256(), s


def matches(a, hs):
    return all(a[k] == hs[i].hexdigest() for i, k in enumerate(('sha256', 'git_blob_sha1')) if k in a)


def headers(response):
    values = {}
    for key in ('Content-Length', 'Content-Encoding', 'Transfer-Encoding', 'Location'):
        v = response.headers.get_all(key, [])
        if len(v) > 1:
            fail('duplicate_' + key.lower())
        values[key] = v[0].strip() if v else None
    length = values['Content-Length']
    if length is not None and (not length.isascii() or not length.isdecimal()):
        fail('invalid_content_length')
    if values['Content-Encoding'] not in (None, 'identity'):
        fail('content_encoding')
    if values['Transfer-Encoding'] not in (None, 'chunked') or (values['Transfer-Encoding'] and length is not None):
        fail('transfer_encoding')
    return None if length is None else int(length), values['Location']


def acquire_all(manifest, root, opener=None, clock=time.monotonic):
    """One operation per root. Verified pre-existing targets may be reused.

    Returns a completed ledger or raises; marker, event log, partials and final
    failed ledger persist. Injected opener/clock are for offline fixtures only.
    """
    cap = validate(manifest)
    root = Path(root)
    if root.is_symlink():
        fail('symlink_root')
    root.mkdir(parents=True, exist_ok=True)
    reserved = {'operation.started.json', 'ledger.json', 'ledger.pending', 'events.jsonl'}
    if any(a['filename'] in reserved or a['filename'].endswith('.partial') for a in manifest['assets']):
        fail('reserved_filename')
    if any((root / n).exists() or (root / n).is_symlink() for n in reserved):
        fail('existing_operation')
    config_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    with (root / 'operation.started.json').open('x') as f:
        json.dump({'manifest_sha256': config_hash, 'started_utc': time.time()}, f)
    ledger = {'schema': 1, 'manifest_sha256': config_hash, 'status': 'started',
              'body_cap_bytes': cap, 'actual_body_bytes': 0, 'requests': [], 'assets': []}
    def save():
        with (root / 'ledger.pending').open('w') as f:
            json.dump(ledger, f, sort_keys=True, indent=2, allow_nan=False)
        os.replace(root / 'ledger.pending', root / 'ledger.json')
    def event(kind, **fields):
        with (root / 'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(kind=kind, **fields), sort_keys=True, allow_nan=False) + '\n')
    save()
    try:
        if sum(a['bytes'] + 1 for a in manifest['assets'] if not (root / a['filename']).exists()) > cap:
            fail('all_assets_budget_with_overflow_sentinels')
        opener = opener or urllib.request.build_opener(NoRedirect())
        for a in manifest['assets']:
            deadline = clock() + 840.0
            def check_time():
                if clock() >= deadline:
                    fail('asset_deadline')
            target, partial = root / a['filename'], root / (a['filename'] + '.partial')
            rec = {'id': a['id'], 'status': 'started', 'new_body_bytes': 0}
            ledger['assets'].append(rec)
            if partial.exists() or partial.is_symlink() or target.is_symlink():
                fail('existing_partial_or_symlink')
            hs = digests(a['bytes'])
            if target.exists():
                if not target.is_file() or target.stat().st_size != a['bytes']:
                    fail('existing_target_mismatch')
                with target.open('rb') as f:
                    while True:
                        check_time(); b = f.read(BODY_CAP)
                        if not b: break
                        for h in hs: h.update(b)
                check_time()
                if not matches(a, hs): fail('existing_target_mismatch')
                rec.update(status='reused_verified', sha256=hs[0].hexdigest())
                event('cache_reuse', id=a['id']); save(); continue
            if cap - ledger['actual_body_bytes'] < a['bytes'] + 1:
                fail('insufficient_budget_with_overflow_sentinel')
            url = a['url']
            for hop in range(5):
                check_time(); url_parts(url, a['allowed_hosts'])
                reqrec = dict(index=len(ledger['requests']) + 1, id=a['id'], hop=hop,
                              url=safe_url(url), bytes_read=0, body_complete=False, open_attempted=False, status='started')
                ledger['requests'].append(reqrec)
                event('request_started', **reqrec); save()
                response = None
                try:
                    request = urllib.request.Request(url, headers={'Accept-Encoding': 'identity'})
                    try:
                        check_time(); reqrec['open_attempted'] = True
                        response = opener.open(request, timeout=min(30.0, max(0.001, deadline - clock())))
                    except urllib.error.HTTPError as e:
                        response = e
                    check_time(); status = response.getcode(); reqrec['http_status'] = status
                    header_error = None
                    try: length, location = headers(response)
                    except AcquisitionError as e: header_error, length, location = e, None, None
                    def read(n):
                        check_time()
                        n = min(n, cap - ledger['actual_body_bytes'])
                        if n <= 0: fail('body_budget_exhausted')
                        read_error = None
                        try: b = response.read(n)
                        except http.client.IncompleteRead as e: b, read_error = e.partial, 'IncompleteRead'
                        except Exception as e:
                            reqrec['unobserved_body_bytes_possible'] = True
                            event('read_error', request=reqrec['index'], error=type(e).__name__, observed_partial_bytes=0)
                            raise AcquisitionError('read_failure') from None
                        reqrec['bytes_read'] += len(b); rec['new_body_bytes'] += len(b)
                        ledger['actual_body_bytes'] += len(b)
                        event('read', request=reqrec['index'], requested=n, bytes_read=len(b), total=ledger['actual_body_bytes'])
                        if read_error:
                            event('read_error', request=reqrec['index'], error=read_error, observed_partial_bytes=len(b))
                            fail('incomplete_read')
                        if len(b) > n: fail('reader_exceeded_requested_bound')
                        check_time()
                        return b
                    if status != 200:
                        while reqrec['bytes_read'] < BODY_CAP:
                            b = read(BODY_CAP - reqrec['bytes_read'])
                            if not b: reqrec['body_complete'] = True; break
                        if not reqrec['body_complete']: fail('non_success_body_limit')
                        if header_error: raise header_error
                        if length is not None and length != reqrec['bytes_read']: fail('response_length_mismatch')
                        if status not in (301, 302, 303, 307, 308): fail('unexpected_http_status')
                        if hop == 4: fail('redirect_limit')
                        if not location: fail('missing_location')
                        url = urllib.parse.urljoin(url, location)
                        url_parts(url, a['allowed_hosts']); reqrec['status'] = 'redirected'
                        continue
                    if header_error: raise header_error
                    if length is not None and length != a['bytes']: fail('unexpected_content_length')
                    written = 0
                    with partial.open('xb', buffering=0) as out:
                        try:
                            while True:
                                b = read(min(BODY_CAP, a['bytes'] - written + 1))
                                if not b: reqrec['body_complete'] = True; break
                                n = out.write(b); written += n
                                for h in hs: h.update(b[:n])
                                if n != len(b): fail('partial_write')
                                if written > a['bytes']: fail('body_overflow')
                        finally:
                            rec.update(partial_bytes=written, partial_sha256=hs[0].hexdigest())
                    if written != a['bytes']: fail('body_underflow')
                    if not matches(a, hs): fail('body_digest_mismatch')
                    check_time()
                    os.link(partial, target)  # atomic, never overwrites a racing destination
                    partial.unlink()
                    rec.update(status='downloaded_verified', sha256=hs[0].hexdigest(), partial_exists=False)
                    reqrec['status'] = 'complete'; event('asset_published', id=a['id']); break
                except Exception as e:
                    reqrec.update(status='failed', error=str(e) if isinstance(e, AcquisitionError) else type(e).__name__)
                    raise
                finally:
                    try:
                        if response is not None: response.close()
                    except Exception as e:
                        reqrec.update(status='failed', close_error=type(e).__name__)
                        event('request_finished', **reqrec); save()
                        raise AcquisitionError('response_close_failure') from None
                    event('request_finished', **reqrec); save()
            save()
        ledger['status'] = 'completed'
        return ledger
    except Exception as e:
        ledger.update(status='failed', error=str(e) if isinstance(e, AcquisitionError) else type(e).__name__)
        event('operation_failed', error=ledger['error'])
        raise AcquisitionError(ledger['error']) from None
    finally:
        ledger['finished_utc'] = time.time()
        save()
