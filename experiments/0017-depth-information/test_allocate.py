"""Fake HTTP and fake HDF5 metadata only; no real cache or network access."""
import contextlib
import hashlib
import http.client
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('acquire0017_artificial', HERE / 'acquire_and_allocate.py')
acq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acq)


class HardLink:
    pass


class Dataset:
    def __init__(self, shape=(8, 2), dtype='float64'):
        self.shape, self.dtype = shape, dtype

    def __getitem__(self, key):
        raise AssertionError('Dataset values must never be read')

    @property
    def attrs(self):
        raise AssertionError('Attribute values must never be read')


class Group:
    def __init__(self, entries, links=None):
        self.entries, self.links = entries, links or {}

    def keys(self):
        return list(reversed(list(self.entries)))

    def get(self, key, getlink=False):
        assert getlink is True
        return self.links.get(key, HardLink())

    def __getitem__(self, key):
        return self.entries[key]

    @property
    def attrs(self):
        raise AssertionError('Attributes forbidden')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def root_group(count=20):
    pair = {name: Dataset() for name in acq.REQUIRED_FIELDS}
    pair['camera1'] = Group({'model': Dataset((), '|O'), 'params': Dataset((4,), 'float64')})
    return Group({'pair-%02d' % i: Group(dict(pair)) for i in range(count)})


class Response:
    def __init__(self, url, data, *, status=200, headers=None, action=None):
        self.url, self.data, self.status, self.pos = url, data, status, 0
        self.headers = {'Content-Length': str(len(data)), 'Content-Encoding': 'identity'}
        if headers:
            self.headers.update(headers)
        self.action, self.read_sizes = action, []

    def geturl(self):
        return self.url

    def read(self, maximum):
        self.read_sizes.append(maximum)
        if self.action:
            return self.action(self, maximum)
        data = self.data[self.pos:self.pos + maximum]
        self.pos += len(data)
        return data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class Tests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.cache, self.receipt, self.manifest = (self.root / 'cache', self.root / 'receipt.json', self.root / 'manifest.json')
        self.protocol = self.root / 'asset-protocol.md'
        self.protocol.write_text('Artificial frozen protocol only')
        self.assets = tuple({**asset, 'bytes': 10} for asset in acq.ASSETS)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in (('PROTOCOL', self.protocol), ('ASSETS', self.assets), ('MAX_NETWORK_BYTES', 50)):
            self.stack.enter_context(patch.object(acq, name, value))
        self.calls, self.opens, self.responses = [], [], []
        self.h5 = types.SimpleNamespace(HardLink=HardLink, Group=Group, Dataset=Dataset, File=self.open_h5)

    def open_h5(self, path, mode):
        self.assertEqual(mode, 'r')
        self.assertEqual(len(self.calls), 5, 'metadata must follow all five acquisitions')
        self.opens.append(Path(path).name)
        return root_group()

    def opener(self, request, timeout):
        self.calls.append(request.full_url)
        self.assertEqual(request.get_header('Accept-encoding'), 'identity')
        self.assertEqual(timeout, 60)
        response = Response(request.full_url, b'0123456789')
        self.responses.append(response)
        return response

    def run_acquire(self, opener=None):
        return acq.acquire(self.cache, self.receipt, self.manifest, opener or self.opener, self.h5)

    def test_frozen_resource_declarations(self):
        self.assertEqual(sum(size for _, size in acq.SCENE_SIZES), 53112544)
        self.assertEqual([scene for scene, _ in acq.SCENE_SIZES], [a['scene'] for a in self.assets])
        self.assertTrue(all(a['url'] == acq.BASE_URL + a['scene'] + '_roma_moge.h5' for a in self.assets))

    def test_exact_salted_selection_is_order_independent_and_ties_lexical(self):
        keys = ['k%02d' % i for i in range(30)]
        scene = 'london_bridge'
        expected = sorted(keys, key=lambda k: (hashlib.sha256((acq.SALT + scene + '\0' + k).encode()).hexdigest(), k))[:16]
        self.assertEqual(acq.select(scene, list(reversed(keys))), expected)
        with patch.object(acq.hashlib, 'sha256', return_value=types.SimpleNamespace(hexdigest=lambda: 'f' * 64)):
            self.assertEqual(acq.select(scene, list(reversed(keys))), sorted(keys)[:16])
        for bad in (keys[:15], keys[:15] + [keys[0]]):
            with self.assertRaises(ValueError):
                acq.select(scene, bad)

    def test_full_five_fake_downloads_and_names_only_allocation(self):
        receipt, manifest = self.run_acquire()
        self.assertEqual(receipt['status'], 'completed')
        self.assertEqual(receipt['network_body_bytes'], 50)
        self.assertEqual(receipt['requested_body_upper_bound'], 50)
        self.assertTrue(receipt['network_body_bytes_are_exact'])
        self.assertEqual(self.calls, [a['url'] for a in self.assets])
        self.assertEqual(self.opens, [a['filename'] for a in self.assets])
        self.assertEqual(len(manifest['selected_pairs']), 80)
        self.assertEqual(manifest['development_forever'], [a['scene'] for a in self.assets])
        self.assertEqual(manifest['dataset_values_read'], 0)
        self.assertEqual(receipt['manifest_sha256'], acq.sha256(self.manifest))
        self.assertEqual(manifest['source_sha256'], receipt['source_sha256'])
        self.assertEqual(manifest['assets_sha256'], receipt['assets_sha256'])
        for i, row in enumerate(manifest['selected_pairs']):
            self.assertEqual(row['pair_index'], i)
            self.assertEqual(row['dataset_bytes'], 10)
            self.assertEqual(row['dataset_sha256'], hashlib.sha256(b'0123456789').hexdigest())
        for scene in manifest['scenes']:
            self.assertEqual(scene['root_keys'], ['pair-%02d' % i for i in range(20)])
            self.assertEqual(scene['selected_pair_keys'], acq.select(scene['scene'], scene['root_keys']))
            self.assertEqual(scene['pairs_metadata'][0]['datasets']['camera1/model'], {'shape': [], 'dtype': '|O'})
            self.assertEqual(scene['schema_summary']['dataset_values_read'], 0)

    def test_existing_outputs_prevent_any_network_or_overwrite(self):
        for which in ('receipt', 'manifest', 'cache'):
            with self.subTest(which=which):
                path = {'receipt': self.receipt, 'manifest': self.manifest,
                        'cache': self.cache / self.assets[-1]['filename']}[which]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'preserve')
                with self.assertRaises(FileExistsError):
                    self.run_acquire()
                self.assertEqual(path.read_bytes(), b'preserve')
                self.assertEqual(self.calls, [])
                path.unlink()

    def test_bad_http_headers_refused_before_any_body(self):
        cases = [('Content-Length', '11'), ('Content-Encoding', 'gzip'), ('Transfer-Encoding', 'chunked')]
        for field, value in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                response = Response(self.assets[0]['url'], b'0123456789', headers={field: value})
                def opener(*args, **kwargs):
                    self.calls.append('attempt')
                    return response
                folder = Path(temp)
                with self.assertRaises(ValueError):
                    acq.acquire(folder / 'cache', folder / 'receipt.json', folder / 'manifest.json', opener, self.h5)
                self.assertEqual(response.read_sizes, [])
                saved = json.loads((folder / 'receipt.json').read_text())
                self.assertEqual(saved['status'], 'failed')
                self.assertEqual(saved['network_body_bytes'], 0)
                self.assertFalse((folder / 'manifest.json').exists())
                self.assertEqual(len(saved['requests']), 1)

    def test_changed_response_url_and_non200_status_are_refused(self):
        for url, status in [('https://example.invalid/other', 200), (self.assets[0]['url'], 206)]:
            with self.subTest(url=url, status=status), tempfile.TemporaryDirectory() as temp:
                response = Response(url, b'0123456789', status=status)
                folder = Path(temp)
                with self.assertRaises(ValueError):
                    acq.acquire(folder / 'cache', folder / 'receipt.json', folder / 'manifest.json',
                                lambda *a, **k: response, self.h5)
                self.assertEqual(response.read_sizes, [])

    def test_short_body_is_preserved_and_never_retried(self):
        response = Response(self.assets[0]['url'], b'abc', headers={'Content-Length': '10'})
        calls = []
        def opener(*args, **kwargs):
            calls.append(1)
            return response
        with self.assertRaisesRegex(ValueError, 'short HTTP body'):
            self.run_acquire(opener)
        receipt = json.loads(self.receipt.read_text())
        self.assertEqual(receipt['network_body_bytes'], 3)
        self.assertEqual(receipt['requests'][0]['stored_bytes'], 3)
        self.assertEqual(receipt['requests'][0]['sha256'], hashlib.sha256(b'abc').hexdigest())
        self.assertEqual((self.cache / self.assets[0]['filename']).read_bytes(), b'abc')
        self.assertEqual(calls, [1])
        self.assertEqual(self.opens, [])
        self.assertFalse(self.manifest.exists())

    def test_incomplete_read_accounts_for_returned_partial_bytes(self):
        def fail(response, maximum):
            raise http.client.IncompleteRead(b'abc', 7)
        response = Response(self.assets[0]['url'], b'0123456789', action=fail)
        with self.assertRaisesRegex(ValueError, 'incomplete HTTP body'):
            self.run_acquire(lambda *a, **k: response)
        receipt = json.loads(self.receipt.read_text())
        self.assertEqual(receipt['network_body_bytes'], 3)
        self.assertTrue(receipt['network_body_bytes_are_exact'])
        self.assertEqual((self.cache / self.assets[0]['filename']).read_bytes(), b'abc')

    def test_interrupt_preserves_receipt_and_discloses_unknown_partial_bytes(self):
        def fail(response, maximum):
            raise KeyboardInterrupt('Artificial interruption')
        response = Response(self.assets[0]['url'], b'0123456789', action=fail)
        with self.assertRaises(KeyboardInterrupt):
            self.run_acquire(lambda *a, **k: response)
        receipt = json.loads(self.receipt.read_text())
        self.assertEqual(receipt['status'], 'failed')
        self.assertFalse(receipt['network_body_bytes_are_exact'])
        self.assertEqual(len(receipt['requests']), 1)
        self.assertFalse(self.manifest.exists())

    def test_metadata_failure_preserves_all_downloads_without_manifest(self):
        self.h5.File = lambda *args: root_group(15)
        with self.assertRaisesRegex(ValueError, 'pair count'):
            self.run_acquire()
        saved = json.loads(self.receipt.read_text())
        self.assertEqual(saved['status'], 'failed')
        self.assertEqual(saved['network_body_bytes'], 50)
        self.assertEqual(len(self.calls), 5)
        self.assertFalse(self.manifest.exists())
        for a in self.assets:
            self.assertEqual((self.cache / a['filename']).read_bytes(), b'0123456789')

    def test_external_link_and_missing_field_are_rejected_without_values(self):
        group = root_group()
        group.links['pair-00'] = object()
        self.h5.File = lambda *args: group
        with self.assertRaisesRegex(ValueError, 'external/soft'):
            acq.inspect_metadata(Path('fake-not-opened'), self.h5)
        group = root_group()
        del group.entries['pair-00'].entries['depth1']
        self.h5.File = lambda *args: group
        with self.assertRaisesRegex(ValueError, 'missing expected'):
            acq.inspect_metadata(Path('fake-not-opened'), self.h5)

    def test_mutation_during_metadata_is_detected(self):
        def altered(path, mode):
            Path(path).write_bytes(b'ABCDEFGHIJ')
            return root_group()
        self.h5.File = altered
        with self.assertRaisesRegex(ValueError, 'changed during metadata'):
            self.run_acquire()
        self.assertEqual(json.loads(self.receipt.read_text())['status'], 'failed')
        self.assertFalse(self.manifest.exists())


if __name__ == '__main__':
    unittest.main()
