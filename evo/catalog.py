"""The fixed, initially unimplemented vocabulary of robot atomic skills.

The catalog is a vocabulary and a contract, not a set of demonstration policies.
Argument values are scene bindings supplied by a planner, never baked into code.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


CATALOG_PATH = Path(__file__).with_name("atomic_skills.json").resolve()


def _read_document():
    raw = CATALOG_PATH.read_bytes()
    document = json.loads(raw)
    skills = document["skills"]
    expected = {f"{prefix}{i:02d}" for prefix, count in (("P", 3), ("A", 26), ("C", 3))
                for i in range(1, count + 1)}
    if (document["version"] != "robocasa-human300-pac32-v1"
            or set(skills) != expected
            or len(document["families"]) != 13
            or any(spec["id"] != atom or spec["family"] not in document["families"]
                   or spec["layer"] != {"P": "perception", "A": "action", "C": "coordination"}[atom[0]]
                   or spec["execution_mode"] != "learned_code"
                   or not spec["description"].strip() for atom, spec in skills.items())):
        raise ValueError("Invalid RoboCasa Human300 P/A/C32 catalog")
    return document, hashlib.sha256(raw).hexdigest()


_DOCUMENT, CATALOG_SHA256 = _read_document()
CATALOG_VERSION = _DOCUMENT["version"]
CATALOG = _DOCUMENT["skills"]
PERCEPTION_CATALOG = {atom: spec for atom, spec in CATALOG.items() if spec["layer"] == "perception"}
ACTION_CATALOG = {atom: spec for atom, spec in CATALOG.items() if spec["layer"] == "action"}
COORDINATION_CATALOG = {atom: spec for atom, spec in CATALOG.items() if spec["layer"] == "coordination"}


def task_catalog(library):
    """Read the source before each task; deliver its overview, not just a path.

    Providers cannot read local files. Details are supplied on demand by the
    host; a changed file requires restarting so schemas and contracts agree.
    """
    document, digest = _read_document()
    if digest != CATALOG_SHA256:
        raise ValueError("Atomic catalog changed during this process; restart the runner")
    overview = {atom: {key: spec[key] for key in ("id", "name", "family", "layer", "summary", "entrypoint", "execution_mode", "execution_owner")}
                for atom, spec in document["skills"].items()}
    for atom, spec in overview.items():
        spec["stub_path"] = str(library.root / "stubs" / f"{atom}.py")
    return {"atomic_catalog": overview,
        "robot_contract": copy.deepcopy(document["robot_contract"]), "catalog_document": {
        "source_path": str(CATALOG_PATH), "sha256": digest,
        "version": CATALOG_VERSION, "loaded_by_host": True,
        "families": document["families"], "provenance": document["provenance"],
        "executable_atoms": list(CATALOG), "library_catalog_path": str(library.root / "catalog.json"),
        "implementation_directory": str(library.root / "sources"),
        "reading_order": "Read all 32 atomic_catalog interfaces and robot_contract before planning. P, A and C all generate executable code. For the selected atom read atomic_skill.description, then implementation description/code and failure feedback."}}


def skill_context(library, atom):
    """Detailed contract and real source locations for only the selected atom."""
    return {"atomic_skill": {**get_spec(atom),
        "catalog_path": str(CATALOG_PATH),
        "stub_path": str(library.root / "stubs" / f"{atom}.py"),
        "implementation": library.active(atom)}, "require_skill_description": True}


def validate_library_catalog(state):
    """Reject foreign taxonomies without relabeling or rewriting their code."""
    version = state.get("atomic_catalog_version")
    if version is not None and version != CATALOG_VERSION:
        raise ValueError("Incompatible atomic catalog; use a separate Human300 P/A/C32 library")
    atoms = set()
    for key in ("active", "versions", "lessons", "working_memory_index"):
        atoms.update(state.get(key, {}))
    records = [r for versions in state.get("versions", {}).values() for r in versions.values()]
    for key in ("candidates", "cases", "pending_cases"):
        records.extend(state.get(key, {}).values())
    for record in records:
        if record.get("atom"):
            atoms.add(record["atom"])
        atoms.update(record.get("dependencies", {}))
    if atoms - set(CATALOG) or (atoms and version is None):
        raise ValueError("Incompatible or unversioned atomic catalog; use the current P/A/C32 namespace")


def get_spec(atom: str) -> dict:
    """Return an isolated copy; unknown atomic IDs are always rejected."""
    if atom not in CATALOG:
        raise ValueError(f"Unknown atomic skill: {atom!r}; expected P01..P03, A01..A26 or C01..C03")
    return copy.deepcopy(CATALOG[atom])


def stub(atom: str) -> str:
    """Return an explicitly empty function, never a hand-authored policy."""
    spec = get_spec(atom)
    description = spec["description"] + "\nImplement using only the primitive API and published dependencies."
    return ("def skill(api, params):\n"
            + "    " + repr(description) + "\n"
            + f'    raise NotImplementedError("{atom} has not been learned")\n')



def validate_params(atom: str, params: dict, strict: bool = False) -> None:
    """Validate a binding dictionary; optional strict mode checks suggested names.

    A learned implementation may use more specific bindings. The adapter and
    replay postcondition validate actual scene semantics rather than key names.
    """
    spec = get_spec(atom)
    if not isinstance(params, dict):
        raise ValueError("params must be a dictionary")
    missing = [key for key, value in spec["parameters"].items()
               if (value["required"] or strict) and key not in params]
    if missing:
        raise ValueError(f"{atom} missing required parameter(s): {', '.join(missing)}")
