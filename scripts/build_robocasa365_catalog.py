"""Audit the installed RoboCasa public task set without importing simulators."""
import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import warnings


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def build_catalog(package_root, manifest_path, *, source_root=None, expected_count=365):
    package_root, manifest_path = Path(package_root), Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    trees, hashes, classes = {}, {}, defaultdict(list)
    paths = {package_root / "__init__.py", package_root / "utils/dataset_registry.py",
             *(package_root / "environments").rglob("*.py")}
    for path in sorted(paths):
        relative = path.relative_to(package_root).as_posix()
        data = path.read_bytes()
        digest = _hash(data)
        expected = manifest["files"].get("robocasa/" + relative)
        if expected != digest:
            raise ValueError("Source differs from installed-package manifest: " + relative)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(data.decode("utf-8"), filename=relative)
        trees[relative], hashes[relative] = tree, digest
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes[node.name].append((relative, node))

    kitchen = trees["environments/kitchen/kitchen.py"]
    meta = next(n for n in kitchen.body if isinstance(n, ast.ClassDef) and n.name == "KitchenEnvMeta")
    exclusion_lists = [n.comparators[0] for n in ast.walk(meta) if isinstance(n, ast.Compare)
                       and len(n.ops) == 1 and isinstance(n.ops[0], ast.NotIn)
                       and isinstance(n.comparators[0], (ast.List, ast.Tuple))]
    if len(exclusion_lists) != 1:
        raise ValueError("Cannot resolve the native metaclass registration exclusions")
    excluded = set(ast.literal_eval(exclusion_lists[0]))
    if not {"OpenDoor", "CloseDoor"}.issubset(excluded):
        raise ValueError("Unexpected native registration exclusions")

    def kitchen_subclass(relative, node, visiting=()):
        key = (relative, node.name)
        if key in visiting:
            raise ValueError("Cyclic static task inheritance")
        if node.name == "Kitchen" and relative == "environments/kitchen/kitchen.py":
            return True
        for base in node.bases:
            if not isinstance(base, ast.Name):
                continue
            for parent_relative, parent in classes.get(base.id, []):
                if kitchen_subclass(parent_relative, parent, (*visiting, key)):
                    return True
        return False

    entries, omitted = {}, []
    for node in trees["__init__.py"].body:
        if not isinstance(node, ast.ImportFrom) or not (node.module or "").startswith("robocasa.environments.kitchen."):
            continue
        relative = node.module.removeprefix("robocasa.").replace(".", "/") + ".py"
        definitions = {n.name: n for n in trees[relative].body if isinstance(n, ast.ClassDef)}
        for alias in node.names:
            name = alias.name
            if name not in definitions:
                raise ValueError("Unresolvable public class: " + name)
            if name == "Kitchen" or name in excluded:
                omitted.append(dict(name=name, reason="base Kitchen environment" if name == "Kitchen"
                                    else "excluded by native KitchenEnvMeta registration"))
                continue
            definition = definitions[name]
            if not kitchen_subclass(relative, definition):
                raise ValueError("Public task is not statically derived from Kitchen: " + name)
            if name in entries:
                raise ValueError("Duplicate public task: " + name)
            kind = "atomic" if ".atomic." in node.module else "composite"
            entries[name] = dict(name=name, kind=kind, module=node.module,
                source=relative, source_sha256=hashes[relative], class_line=definition.lineno)
    if len(entries) != expected_count:
        raise ValueError(f"Expected {expected_count} public tasks; found {len(entries)}")

    datasets = {}
    for node in trees["utils/dataset_registry.py"].body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"ATOMIC_TASK_DATASETS", "COMPOSITE_TASK_DATASETS"}:
                    datasets[target.id] = [item.arg for item in node.value.keywords]
    dataset_names = {name for group in datasets.values() for name in group}
    names = sorted(entries)
    provenance = dict(
        generated_at=datetime.now(timezone.utc).isoformat(), method="static AST audit; no scene initialization",
        source_root=source_root or str(package_root.resolve()), packages=manifest["packages"],
        simulator_manifest_sha256=_hash(manifest_path.read_bytes()),
        public_task_count=len(entries), counts_by_kind=dict(Counter(item["kind"] for item in entries.values())),
        exclusion_rule="Public __init__.py kitchen class exports, excluding Kitchen and native registration exclusions",
        excluded_public_exports=omitted, registration_exclusions=sorted(excluded),
        all_task_classes_resolved=True, all_task_classes_derive_from_kitchen=True,
        audited_source_files=len(hashes), source_sha256=hashes,
        dataset_registry_counts={key: len(value) for key, value in datasets.items()},
        public_tasks_missing_from_dataset_registry=sorted(set(names) - dataset_names),
        task_names_sha256=_hash(("\n".join(names) + "\n").encode("utf-8")),
        runtime_limit="Static task availability does not verify every task's assets, scene reset, or success behavior")
    catalog = dict(schema_version=1, task_count=len(entries), task_names=names,
                   entries=[entries[name] for name in names], provenance=provenance)
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--provenance-output")
    args = parser.parse_args()
    catalog = build_catalog(args.package_root, args.manifest, source_root=args.source_root)
    for path, data in [(args.output, catalog), (args.provenance_output, catalog["provenance"])]:
        if path:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    print(json.dumps({key: catalog["provenance"][key] for key in
                      ("public_task_count", "counts_by_kind", "excluded_public_exports", "task_names_sha256")}))


if __name__ == "__main__":
    main()
