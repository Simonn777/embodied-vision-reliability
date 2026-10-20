"""Fake-only provenance, selection, extraction and resource-monitor checks."""
import copy
import email.message
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import tarfile
import tempfile
import time
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


c = load('fake_common_test', HERE / 'common.py')
s = load('fake_supervisor_test', HERE / 'supervise.py')
a = load('fake_acquire_reader', HERE / 'acquire.py')
COMMIT = 'a' * 40


def pack(entries):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w', format=tarfile.USTAR_FORMAT) as archive:
        for name, payload in entries:
            header = tarfile.TarInfo(name)
            header.size = len(payload)
            archive.addfile(header, io.BytesIO(payload))
    return gzip.compress(buffer.getvalue(), mtime=0)


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='fake-depth-common-')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / 'repo'
        self.sources = self.repo / 'experiments/0025b-physical-depth'
        self.sources.mkdir(parents=True)
        for name in c.OWNED:
            source = HERE / name
            (self.sources / name).write_bytes(source.read_bytes() if source.exists() else b'# fake not executed\n')
        self.protocol = self.repo / 'protocols/0025b-rs-depth-semantics.md'
        self.protocol.parent.mkdir()
        self.protocol.write_text('Artificial prospective protocol, no real data.\n')
        for module, name, value in ((c, 'ROOT', self.repo), (c, 'HERE', self.sources),
                                    (c, 'PROTOCOL', self.protocol), (s, 'ROOT', self.repo),
                                    (s, 'PROTOCOL', self.protocol)):
            patcher = mock.patch.object(module, name, value)
            patcher.start(); self.addCleanup(patcher.stop)
        for target in ('urllib.request.urlopen', 'urllib.request.OpenerDirector.open', 'socket.socket.connect'):
            patcher = mock.patch(target, side_effect=AssertionError('REAL NETWORK FORBIDDEN'))
            patcher.start(); self.addCleanup(patcher.stop)
        self.bindings = {path: c.sha(self.repo / path) for path in c.owned_paths()}
        self.data = self.base / 'fake-acquisition'
        self.data.mkdir()
        self.outer_dir = self.base / 'fake-supervision'
        self.outer_dir.mkdir()
        (self.outer_dir / 'stdout.log').write_text('fake stdout only\n')
        (self.outer_dir / 'stderr.log').write_text('')
        self.outer_path = self.outer_dir / 'receipt.json'
        self.outer = dict(schema='evr0025b.supervision.v1', operation='acquire', expected_commit=COMMIT,
                          status='completed', exit_code=0, termination=None, remaining_descendant_pids=[],
                          cleanup_errors=[], hard_timeout_seconds=1800.0, group_rss_limit_kib=8*1024*1024,
                          wall_seconds=.1, peak_group_rss_kib=1000,
                          thread_environment={key: '4' for key in c.THREAD_KEYS}, source_sha256=self.bindings,
                          command=['$PYTHON', '$REPOSITORY/experiments/0025b-physical-depth/acquire.py',
                                   '--expected-commit', COMMIT, '--repo-root', '$REPOSITORY',
                                   '--protocol-relative', 'protocols/0025b-rs-depth-semantics.md',
                                   '--destination', str(self.data)],
                          stdout_sha256=c.sha(self.outer_dir/'stdout.log'), stderr_sha256=c.sha(self.outer_dir/'stderr.log'))
        self.save_outer()

    def save_outer(self, value=None):
        self.outer_path.write_text(json.dumps(self.outer if value is None else value))

    def fixture(self, count=12):
        assets, rows, indices = {}, [], {}
        for name, (scene, _) in c.ASSETS.items():
            entries = [(kind + '/' + str(i*3) + '.png', (kind + ' opaque fake ' + str(i)).encode())
                       for kind in ('rgb', 'depth') for i in range(count)]
            entries += [('groundtruth.txt', b'\xff\x00deliberately invalid pose numbers')]
            payload = pack(entries)
            (self.data / name).write_bytes(payload)
            index = a.index_archive(self.data/name, self.data/(name+'.index.json'), time.monotonic())
            indices[name] = c.read_json(self.data/(name+'.index.json'))
            assets[name] = (scene, len(payload))
            url, etag = c.ASSET_HEADERS[name]
            response = dict(content_length=len(payload), etag=etag, content_encoding='x-gzip', status=200, final_url=url)
            observed = dict(status=200, final_url=url,
                            headers=[['Content-Length', str(len(payload))], ['ETag', etag], ['Content-Encoding', 'x-gzip']])
            rows.append(dict(name=name, url=url, etag=etag, bytes=len(payload), body_bytes=len(payload),
                             status='completed', sha256=c.sha(self.data/name), index=index, response=response,
                             observed_response=observed, sha256_kind='trust-on-first-use; not a publisher digest'))
        total = sum(size for _, size in assets.values())
        patcher = mock.patch.object(c, 'ASSETS', assets)
        patcher.start(); self.addCleanup(patcher.stop)
        subset = {p: self.bindings[p] for p in ('experiments/0025b-physical-depth/acquire.py',
                                              'protocols/0025b-rs-depth-semantics.md')}
        record = dict(schema='evr0025b.acquisition.v1', status='completed', expected_commit=COMMIT,
                      assets=rows, body_bytes=total, wall_seconds=.05, source_sha256=subset,
                      thread_environment={key: '4' for key in c.THREAD_KEYS},
                      limits=dict(wall_seconds=1800, body_bytes=total, request_timeout_seconds=30,
                                  outer_descendants_rss_bytes=8*1024**3, maximum_payload_GETs=2, automatic_retries=0))
        self.save_acquisition(record)
        return record, indices

    def save_acquisition(self, record):
        (self.data/'receipt.json').write_text(json.dumps(record))

    def test_outer_exact_threads_types_sources_and_logs(self):
        c.verify_outer(self.outer_path, 'acquire', COMMIT, self.data, '--destination')
        bads = []
        for key, value in [('wall_seconds', True), ('wall_seconds', '1'), ('wall_seconds', float('nan')),
                           ('peak_group_rss_kib', 1.0), ('peak_group_rss_kib', -1), ('exit_code', False),
                           ('source_sha256', {}), ('hard_timeout_seconds', float('inf'))]:
            row = copy.deepcopy(self.outer); row[key] = value; bads.append(row)
        row = copy.deepcopy(self.outer); row['thread_environment'].pop('OMP_NUM_THREADS'); row['thread_environment']['FAKE']='4'; bads.append(row)
        row = copy.deepcopy(self.outer); row['source_sha256'].pop('experiments/0025b-physical-depth/common.py'); bads.append(row)
        row = copy.deepcopy(self.outer); row['command'][-1] = str(self.base/'other-output'); bads.append(row)
        row = copy.deepcopy(self.outer); row['command'][-3] = 'protocols/other.md'; bads.append(row)
        for i, row in enumerate(bads):
            with self.subTest(i=i), self.assertRaises(ValueError):
                self.save_outer(row)
                c.verify_outer(self.outer_path, 'acquire', COMMIT, self.data, '--destination')
        self.save_outer()
        (self.outer_dir/'stdout.log').write_text('changed')
        with self.assertRaises(ValueError):
            c.verify_outer(self.outer_path, 'acquire', COMMIT, self.data, '--destination')

    def test_acquisition_http_counts_and_provenance_fail_closed(self):
        receipt, _ = self.fixture()
        indices, bindings = c.verify_acquisition(self.data, self.outer_path, COMMIT)
        self.assertEqual(len(indices), 2)
        self.assertEqual(bindings['receipt_sha256'], c.sha(self.data/'receipt.json'))
        mutations = [lambda r: r['assets'][0].update(url='https://example.invalid/not-official'),
                     lambda r: r['assets'][0].update(etag='changed'),
                     lambda r: r['assets'][0]['observed_response']['headers'].append(['ETag', r['assets'][0]['etag']]),
                     lambda r: r['assets'][0]['observed_response']['headers'].append(['Transfer-Encoding', 'chunked']),
                     lambda r: r['assets'][0]['response'].update(status=True),
                     lambda r: r.update(body_bytes=r['body_bytes']-1),
                     lambda r: r['assets'].reverse(),
                     lambda r: r['source_sha256'].clear(),
                     lambda r: r['limits'].update(automatic_retries=False),
                     lambda r: r['limits'].update(maximum_payload_GETs=3)]
        for i, mutate in enumerate(mutations):
            changed = copy.deepcopy(receipt); mutate(changed); self.save_acquisition(changed)
            with self.subTest(i=i), self.assertRaises(ValueError):
                c.verify_acquisition(self.data, self.outer_path, COMMIT)
        self.save_acquisition(receipt)
        path = self.data/next(iter(c.ASSETS)); payload = path.read_bytes(); path.write_bytes(payload[:-1]+bytes([payload[-1]^1]))
        with self.assertRaises(ValueError):
            c.verify_acquisition(self.data, self.outer_path, COMMIT)

    def test_complete_population_saved_before_payload_and_ambiguous_paths_reject(self):
        _, indices = self.fixture()
        with mock.patch.object(tarfile, 'open', side_effect=AssertionError('no member opening for selection')):
            selected, populations = c.select_frames(indices)
        self.assertEqual(len(selected), 16)
        self.assertEqual(populations[0]['paired_stems'], list(range(0,36,3)))
        self.assertEqual(populations[0]['ranks'], [0,2,3,5,6,8,9,11])
        self.assertEqual(len(populations[0]['paired_inventory']),12)
        self.assertEqual(selected[0]['numeric_stem'], 0)
        self.assertEqual(selected[7]['numeric_stem'], 33)
        first=next(iter(indices))
        for kind in ('ambiguous', 'duplicate', 'unsafe'):
            wrong=copy.deepcopy(indices)
            for member in wrong[first]['members']:
                if member['name']=='depth/0.png':
                    member['name']={'ambiguous':'other/depth/0.png','duplicate':'depth/03.png','unsafe':'../depth/0.png'}[kind]
                    break
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                c.select_frames(wrong)

    def test_extract_only_fixed_category_and_no_decoders_or_pose_extraction(self):
        _, indices = self.fixture()
        selected, _ = c.select_frames(indices)
        with mock.patch.object(tarfile.TarFile, 'extractfile', side_effect=AssertionError('generic extract forbidden')), \
             mock.patch.object(tarfile.TarFile, 'extractall', side_effect=AssertionError('generic extract forbidden')):
            output=c.extract_selected(self.data, selected, 'rgb', self.base/'rgb-output')
        self.assertEqual(set(output),set(range(16)))
        self.assertTrue(all(Path(row['path']).read_bytes().startswith(b'rgb opaque fake') for row in output.values()))
        self.assertFalse(any('pose' in p.name or 'groundtruth' in p.name for p in (self.base/'rgb-output').iterdir()))
        with self.assertRaises(ValueError):
            c.extract_selected(self.data, selected, 'pose', self.base/'pose-output')
        wrong=copy.deepcopy(selected); wrong[0]['rgb_member']=wrong[0]['depth_member']
        with self.assertRaises(ValueError):
            c.extract_selected(self.data, wrong, 'rgb', self.base/'wrong-output')
        self.assertFalse((self.base/'wrong-output').exists())
        archive=self.data/selected[0]['archive']; archive.write_bytes(archive.read_bytes()+b'changed')
        with self.assertRaises(ValueError):
            c.extract_selected(self.data, selected, 'rgb', self.base/'tampered-output')
        self.assertEqual(list((self.base/'tampered-output').iterdir()), [])

    def fake_supervision(self, mode):
        output=self.base/('supervised-'+mode)
        command=[str(self.sources/'predict.py'),'--expected-commit',COMMIT,'--output-dir',str(self.base/'fake-child')]
        args=types.SimpleNamespace(operation='predict',expected_commit=COMMIT,output_dir=str(output),command_json=json.dumps(command))
        child=types.SimpleNamespace(pid=900001,returncode=(1 if mode=='exit' else 0))
        child.poll=lambda: None if mode in ('rss','timeout') else child.returncode
        child.wait=lambda timeout: child.returncode
        times=[0.0]
        def clock():
            if mode=='timeout': times[0]+=900.0
            return times[0]
        def launched(*args,**kwargs):
            self.assertTrue(kwargs['start_new_session'])
            self.assertEqual({k:kwargs['env'][k] for k in s.THREAD_KEYS},{k:'4' for k in s.THREAD_KEYS})
            if mode=='mutation': (self.sources/'common.py').write_text('changed during fake child')
            return child
        rss=s.RSS_LIMIT_KIB+1 if mode=='rss' else 100
        rows=[(child.pid,1,child.pid,rss)]
        if mode=='descendant': rows.append((900002,child.pid,child.pid,100))
        with mock.patch.object(s,'verify_frozen',return_value=self.bindings), \
             mock.patch.object(s.subprocess,'Popen',side_effect=launched) as popen, \
             mock.patch.object(s,'snapshot',return_value=(rows,{child.pid})), \
             mock.patch.object(s.os,'killpg') as kill, \
             mock.patch.object(s.time,'perf_counter',side_effect=clock):
            result=s.run(args)
        return result,c.read_json(output/'receipt.json'),kill.call_args_list,popen.call_count

    def test_supervisor_resource_cleanup_exit_and_source_postcheck(self):
        for mode in ('success','rss','timeout','exit','descendant','mutation'):
            with self.subTest(mode=mode):
                result,record,kills,calls=self.fake_supervision(mode)
                self.assertEqual(calls,1)
                self.assertEqual(result,0 if mode=='success' else 1)
                self.assertEqual(record['status'],'completed' if mode=='success' else 'failed')
                self.assertEqual(set(record['source_sha256']),c.owned_paths())
                if mode in ('rss','timeout','exit','descendant'):
                    self.assertTrue(any(call.args[1]==signal.SIGKILL for call in kills))
                if mode=='mutation': self.assertEqual(record['termination'],'source_changed_during_child')

    def test_supervisor_preflight_rejects_changed_commit_or_acquisition_protocol(self):
        command=[str(self.sources/'acquire.py'),'--expected-commit',COMMIT,'--repo-root',str(self.repo),
                 '--protocol-relative','protocols/other.md','--destination',str(self.data)]
        args=types.SimpleNamespace(operation='acquire',expected_commit=COMMIT,output_dir=str(self.base/'not-created'),command_json=json.dumps(command))
        with mock.patch.object(s.subprocess,'Popen',side_effect=AssertionError('must not launch')), self.assertRaises(ValueError):
            s.run(args)
        command[2]='b'*40; args.command_json=json.dumps(command)
        with mock.patch.object(s.subprocess,'Popen',side_effect=AssertionError('must not launch')), self.assertRaises(ValueError):
            s.run(args)
        self.assertFalse(Path(args.output_dir).exists())

    def test_audit_allowlist_and_binding_are_complete(self):
        self.assertEqual(s.SCRIPTS['audit'], 'audit_saved.py')
        self.assertIn('audit_saved.py', s.OWNED)
        self.assertTrue(all('audit_saved.py' in values for values in s.HELPERS.values()))
        outer=copy.deepcopy(self.outer)
        outer.update(operation='audit', command=['$PYTHON', '$REPOSITORY/experiments/0025b-physical-depth/audit_saved.py',
            '--expected-commit', COMMIT, '--output-dir', str(self.data)])
        self.save_outer(outer)
        c.verify_outer(self.outer_path, 'audit', COMMIT, self.data, '--output-dir')
        outer['source_sha256'].pop('experiments/0025b-physical-depth/audit_saved.py')
        self.save_outer(outer)
        with self.assertRaises(ValueError):
            c.verify_outer(self.outer_path, 'audit', COMMIT, self.data, '--output-dir')


if __name__ == '__main__':
    unittest.main(verbosity=2)
