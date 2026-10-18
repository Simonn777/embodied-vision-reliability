"""Offline import-gate checks: toy files and inert modules, no ML/runtime call."""
import ast
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


ADAPTER_PATH = Path(__file__).resolve().parents[1] / "scripts/tapnext_native_adapter.py"
with mock.patch.object(sys, "dont_write_bytecode", True):
    _spec = importlib.util.spec_from_file_location("adapter_gate_test_target", ADAPTER_PATH)
    adapter = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(adapter)


def tapnet_module(name):
    return name == "tapnet" or name.startswith("tapnet.")


class NativeSourceImportGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / "source"
        pins = {}
        for name in adapter.SOURCES:
            path = self.root / (name.replace(".", "/") + ".py")
            path.parent.mkdir(parents=True, exist_ok=True)
            body = "APPROVED_TOY_LEAF = " + repr(name) + "\n"
            path.write_text(body)
            pins[name] = hashlib.sha256(body.encode()).hexdigest()
        self.saved_modules = {k: v for k, v in sys.modules.items() if tapnet_module(k)}
        for name in self.saved_modules:
            del sys.modules[name]
        self.patchers = [mock.patch.object(adapter, "SOURCES", pins),
                         mock.patch.object(sys, "path", [str(self.root)] + list(sys.path)),
                         mock.patch.object(sys, "dont_write_bytecode", True)]
        for patcher in self.patchers:
            patcher.start()
        importlib.invalidate_caches()

    def tearDown(self):
        for name in list(sys.modules):
            if tapnet_module(name):
                del sys.modules[name]
        sys.modules.update(self.saved_modules)
        for patcher in reversed(self.patchers):
            patcher.stop()
        for path in list(sys.path_importer_cache):
            if path.startswith(str(self.base)):
                sys.path_importer_cache.pop(path, None)
        importlib.invalidate_caches()
        self.temp.cleanup()

    def assert_rejects_before_initializer(self, source_root=None):
        with self.assertRaises(ValueError):
            adapter.verify_source_tree(self.root if source_root is None else source_root)
        self.assertEqual(list(self.base.rglob("INITIALIZER_RAN")), [])

    def write_initializer(self, path):
        path.write_text("from pathlib import Path\n"
                        "Path(__file__).with_name('INITIALIZER_RAN').write_text('yes')\n")

    def test_exact_five_leaf_tree_imports_with_only_expected_namespaces(self):
        adapter.verify_source_tree(self.root)
        leaf = importlib.import_module("tapnet.tapnextpp.votsp2026.model")
        self.assertEqual(leaf.APPROVED_TOY_LEAF, "tapnet.tapnextpp.votsp2026.model")
        for name in ("tapnet", "tapnet.tapnextpp", "tapnet.tapnextpp.votsp2026"):
            package = sys.modules[name]
            self.assertIsNone(package.__file__)
            self.assertEqual([Path(p).resolve() for p in package.__path__],
                             [(self.root / name.replace(".", "/")).resolve()])

    def test_each_parent_initializer_rejects_without_execution(self):
        for parent in ("tapnet", "tapnet/tapnext", "tapnet/tapnextpp",
                       "tapnet/tapnextpp/votsp2026"):
            with self.subTest(parent=parent):
                path = self.root / parent / "__init__.py"
                self.write_initializer(path)
                try:
                    self.assert_rejects_before_initializer()
                finally:
                    path.unlink()

    def test_later_installed_regular_package_cannot_override_namespace(self):
        alternate = self.base / "installed"
        (alternate / "tapnet").mkdir(parents=True)
        self.write_initializer(alternate / "tapnet/__init__.py")
        sys.path.insert(1, str(alternate))
        self.assert_rejects_before_initializer()

    def test_alternate_top_level_module_rejects(self):
        alternate = self.base / "installed"
        alternate.mkdir()
        self.write_initializer(alternate / "tapnet.py")
        sys.path.insert(1, str(alternate))
        self.assert_rejects_before_initializer()

    def test_additional_namespace_location_rejects(self):
        alternate = self.base / "another"
        (alternate / "tapnet").mkdir(parents=True)
        sys.path.insert(1, str(alternate))
        self.assert_rejects_before_initializer()

    def test_preloaded_top_or_descendant_rejects(self):
        for name in ("tapnet", "tapnet.extraneous"):
            with self.subTest(name=name):
                sys.modules[name] = types.ModuleType(name)
                try:
                    self.assert_rejects_before_initializer()
                finally:
                    sys.modules.pop(name)

    def test_extra_file_or_parent_module_rejects(self):
        for relative in ("tapnet/tapnext/other.py", "tapnet/tapnextpp.py"):
            with self.subTest(relative=relative):
                path = self.root / relative
                path.write_text("UNBOUND = True\n")
                try:
                    self.assert_rejects_before_initializer()
                finally:
                    path.unlink()

    def test_extra_empty_directory_rejects(self):
        (self.root / "tapnet/empty_extra").mkdir()
        self.assert_rejects_before_initializer()

    def test_missing_leaf_rejects(self):
        (self.root / "tapnet/tapnext/pscan.py").unlink()
        self.assert_rejects_before_initializer()

    def test_modified_leaf_rejects(self):
        (self.root / "tapnet/tapnext/pscan.py").write_text("UNBOUND = True\n")
        self.assert_rejects_before_initializer()

    def test_same_hash_leaf_symlink_rejects(self):
        path = self.root / "tapnet/tapnext/pscan.py"
        outside = self.base / "outside.py"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
        self.assert_rejects_before_initializer()

    def test_same_tree_parent_symlink_rejects(self):
        path = self.root / "tapnet/tapnext"
        outside = self.base / "outside"
        path.rename(outside)
        path.symlink_to(outside, target_is_directory=True)
        self.assert_rejects_before_initializer()

    def test_source_root_symlink_rejects(self):
        linked = self.base / "linked_source"
        linked.symlink_to(self.root, target_is_directory=True)
        sys.path[0] = str(linked)
        self.assert_rejects_before_initializer(linked)


class NativeDependencyIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Execute the real guards verbatim with inert module records. Running
        # run_capability itself would require a forbidden real Linux/ML setup.
        tree = ast.parse(ADAPTER_PATH.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "run_capability")
        body = next(n for n in fn.body if isinstance(n, ast.Try)).body
        version_loop = next(n for n in body if isinstance(n, ast.For)
                            and any(isinstance(x, ast.Attribute) and x.attr == "__version__"
                                    for x in ast.walk(n)))
        origin_loop = next(n for n in body if isinstance(n, ast.For)
                           and ast.unparse(n.iter) == "(torchvision, einops, PIL)")
        cls.version_guard = compile(ast.Module(body=[version_loop], type_ignores=[]),
                                    "VERBATIM_VERSION_GUARD", "exec")
        cls.origin_guard = compile(ast.Module(body=[origin_loop], type_ignores=[]),
                                   "VERBATIM_OVERLAY_ORIGIN_GUARD", "exec")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name).resolve()
        self.overlay = self.base / "overlay"
        self.overlay.mkdir()
        self.other = self.base / "other"
        self.other.mkdir()
        self.addCleanup(self.temp.cleanup)
        self.versions = {"torch": "2.10.0+cpu", "numpy": "2.0.2",
                         "torchvision": "0.25.0+cpu", "einops": "0.8.2",
                         "Pillow": "11.3.0"}
        # Pins are invented fixture values, not a real package preflight.
        self.env = {"Path": Path, "overlay_root": self.overlay,
                    "expected_packages": self.versions}
        for variable, distribution in (("torch", "torch"), ("np", "numpy"),
                                       ("torchvision", "torchvision"), ("einops", "einops"),
                                       ("PIL", "Pillow")):
            self.env[variable] = types.SimpleNamespace(
                __file__=str(self.overlay / variable / "__init__.py"),
                __version__=self.versions[distribution])

    def test_actual_module_versions_matching_all_pins_accept(self):
        exec(self.version_guard, self.env)

    def test_any_actual_module_version_mismatch_rejects(self):
        for variable in ("torch", "np", "torchvision", "einops", "PIL"):
            with self.subTest(variable=variable):
                package = self.env[variable]
                old = package.__version__
                package.__version__ = "different-module-version"
                try:
                    with self.assertRaisesRegex(ValueError, "Imported module version differs"):
                        exec(self.version_guard, self.env)
                finally:
                    package.__version__ = old

    def test_missing_actual_module_version_fails_closed(self):
        del self.env["PIL"].__version__
        with self.assertRaises(AttributeError):
            exec(self.version_guard, self.env)

    def test_pillow_overlay_origin_accepts(self):
        exec(self.origin_guard, self.env)

    def test_pillow_external_origin_rejects(self):
        self.env["PIL"].__file__ = str(self.other / "PIL/__init__.py")
        with self.assertRaisesRegex(ValueError, "independent overlay"):
            exec(self.origin_guard, self.env)

    def test_pillow_symlink_escape_rejects(self):
        (self.overlay / "PIL").symlink_to(self.other, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "independent overlay"):
            exec(self.origin_guard, self.env)


if __name__ == "__main__":
    unittest.main()
