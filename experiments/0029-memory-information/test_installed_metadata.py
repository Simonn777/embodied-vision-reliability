"""Manufactured metadata only; no package import or install."""
import importlib.metadata as metadata
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import importlib.util

spec = importlib.util.spec_from_file_location('evr0029_installer_metadata_test', Path(__file__).with_name('install_runtime.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
installed_metadata_path = m.installed_metadata_path


class MetadataPaths(unittest.TestCase):
    def make_distribution(self, site, entries):
        info = site / "demo-1.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text("Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n\n")
        (info / "RECORD").write_text("".join(f"{entry},,\n" for entry in entries))
        return metadata.Distribution.at(info), info / "METADATA"

    def test_own_metadata_selected_despite_vendored_distribution(self):
        with tempfile.TemporaryDirectory() as temp:
            site = Path(temp)
            own = "demo-1.0.dist-info/METADATA"
            nested = "demo/_vendor/vendor-2.0.dist-info/METADATA"
            dist, path = self.make_distribution(site, [own, nested])
            vendored = site / nested
            vendored.parent.mkdir(parents=True)
            vendored.write_text("Metadata-Version: 2.1\nName: vendor\nVersion: 2.0\n\n")
            self.assertEqual(len([p for p in dist.files if str(p).endswith(".dist-info/METADATA")]), 2)
            self.assertEqual(installed_metadata_path(dist, site), path)
            self.assertEqual([(d.metadata["Name"], d.version)
                             for d in metadata.distributions(path=[str(site)])], [("demo", "1.0")])
            self.assertTrue(vendored.is_file())

    def test_missing_or_multiple_top_level_metadata_rejected(self):
        for entries in (["demo/_vendor/vendor-2.0.dist-info/METADATA"],
                        ["demo-1.0.dist-info/METADATA", "other-1.0.dist-info/METADATA"]):
            with self.subTest(entries=entries), tempfile.TemporaryDirectory() as temp:
                dist, _ = self.make_distribution(Path(temp), entries)
                # importlib.metadata omits nonexistent RECORD targets.
                for entry in entries:
                    path = Path(temp) / entry
                    if not path.exists():
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text("Name: other\nVersion: 1.0\n")
                with self.assertRaisesRegex(ValueError, "missing/ambiguous"):
                    installed_metadata_path(dist, temp)

    def test_wrong_identity_and_outside_site_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            dist, path = self.make_distribution(Path(temp), ["demo-1.0.dist-info/METADATA"])
            fake = Mock(files=dist.files)
            fake.locate_file.side_effect = dist.locate_file
            fake.read_text.return_value = "Name: different\nVersion: 1.0\n"
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                installed_metadata_path(fake, temp)
            with self.assertRaisesRegex(ValueError, "outside direct"):
                installed_metadata_path(dist, path.parent)

    def test_symlink_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            dist, path = self.make_distribution(Path(temp), ["demo-1.0.dist-info/METADATA"])
            text = path.read_text()
            path.unlink()
            actual = Path(temp) / "external-metadata"
            actual.write_text(text)
            path.symlink_to(actual)
            with self.assertRaisesRegex(ValueError, "outside direct"):
                installed_metadata_path(dist, temp)


if __name__ == "__main__":
    unittest.main()
