"""The catalog follows public exports and native exclusions, not a partial dataset list."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_robocasa365_catalog import build_catalog


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "robocasa"
        self.files = {
            "__init__.py": "from robocasa.environments.kitchen.kitchen import Kitchen\n"
                "from robocasa.environments.kitchen.atomic.example import OpenDoor, CloseDoor, AtomicTask\n"
                "from robocasa.environments.kitchen.composite.example import CompositeTask\n",
            "environments/kitchen/kitchen.py": "class KitchenEnvMeta(type):\n"
                "    def register(cls):\n        if cls.__name__ not in ['OpenDoor', 'CloseDoor']:\n            pass\n"
                "class Kitchen: pass\n",
            "environments/kitchen/atomic/example.py": "class OpenDoor(Kitchen): pass\n"
                "class CloseDoor(Kitchen): pass\nclass AtomicTask(OpenDoor): pass\n"
                "class InternalHelper(Kitchen): pass\n",
            "environments/kitchen/composite/example.py": "class CompositeTask(AtomicTask): pass\n",
            "utils/dataset_registry.py": "ATOMIC_TASK_DATASETS = OrderedDict(AtomicTask=dict())\n"
                "COMPOSITE_TASK_DATASETS = OrderedDict()\n",
        }
        for name, text in self.files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        self.manifest = self.root.parent / "manifest.json"
        self.manifest.write_text(json.dumps(dict(packages={"robocasa": {"version": "fixture"}},
            files={"robocasa/" + name: hashlib.sha256(text.encode()).hexdigest()
                   for name, text in self.files.items()})))

    def tearDown(self):
        self.temporary.cleanup()

    def test_public_exports_native_exclusions_and_inheritance_define_the_set(self):
        result = build_catalog(self.root, self.manifest, expected_count=2)
        self.assertEqual(result["task_names"], ["AtomicTask", "CompositeTask"])
        self.assertEqual(result["provenance"]["counts_by_kind"], {"atomic": 1, "composite": 1})
        self.assertEqual(result["provenance"]["public_tasks_missing_from_dataset_registry"], ["CompositeTask"])
        self.assertEqual({item["name"] for item in result["provenance"]["excluded_public_exports"]},
                         {"Kitchen", "OpenDoor", "CloseDoor"})

    def test_manifest_mismatch_fails_closed(self):
        (self.root / "__init__.py").write_text(self.files["__init__.py"] + "# altered\n")
        with self.assertRaisesRegex(ValueError, "differs from installed-package manifest"):
            build_catalog(self.root, self.manifest, expected_count=2)

    def test_count_mismatch_is_not_silently_padded_or_truncated(self):
        with self.assertRaisesRegex(ValueError, "Expected 365 public tasks; found 2"):
            build_catalog(self.root, self.manifest)


if __name__ == "__main__":
    unittest.main()
