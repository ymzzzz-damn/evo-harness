#!/usr/bin/env python3
"""Recover an unpublished library into a new root; never edit the evidence.

Only libraries without active or published versions can be merged automatically.
Attempt records provide failure evidence, never successful validation or promotion.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evo.catalog import get_spec
from evo.codecheck import check_code, source_hash
from evo.library import Library, LibraryError, SCHEMA_VERSION, CASE_LIFECYCLE_VERSION


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _counts(state):
    return {"candidates": len(state["candidates"]), "cases": len(state["cases"]),
            "pending_cases": len(state.get("pending_cases", {})),
            "lessons": sum(len(items) for items in state["lessons"].values()),
            "active": len(state["active"]), "versions": len(state["versions"])}


def _insert(records, identifier, value, kind):
    if identifier in records and records[identifier] != value:
        raise ValueError(f"Conflicting {kind} ID: {identifier}")
    records.setdefault(identifier, copy.deepcopy(value))


def _merge_states(paths):
    merged = {"schema_version": SCHEMA_VERSION, "active": {}, "versions": {},
              "candidates": {}, "cases": {}, "pending_cases": {}, "case_activations": {},
              "case_lifecycle_version": CASE_LIFECYCLE_VERSION, "lessons": {}, "events": []}
    lessons, event_keys, inputs = {}, set(), []
    known = set(merged) | {"_state_sha256", "activation_revisions"}
    for path in paths:
        state = Library._read_state_file(path)
        if state["active"] or state["versions"] or state.get("case_activations"):
            raise ValueError(f"Refusing automatic merge of published skills: {path}")
        if set(state) - known:
            raise ValueError(f"Unknown state fields in {path}: {sorted(set(state) - known)}")
        inputs.append({"path": str(path), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "counts": _counts(state)})
        for identifier, candidate in state["candidates"].items():
            get_spec(candidate["atom"])
            if candidate.get("id") != identifier or candidate.get("status") == "promoted":
                raise ValueError(f"Invalid/unexpected published candidate: {identifier}")
            _insert(merged["candidates"], identifier, candidate, "candidate")
        # An unpublished source cannot supply admitted regression evidence. Old
        # v1 cases and modern pending cases are both retained only as proposals;
        # conflicting IDs still fail, including conflicts across these stores.
        for collection in (state["cases"], state.get("pending_cases", {})):
            for identifier, case in collection.items():
                get_spec(case["atom"])
                if case.get("id") != identifier:
                    raise ValueError(f"Case ID mismatch: {identifier}")
                _insert(merged["pending_cases"], identifier, case, "case")
        for atom, entries in state["lessons"].items():
            get_spec(atom)
            for lesson in entries:
                if lesson.get("atom") != atom or not isinstance(lesson.get("id"), str):
                    raise ValueError(f"Invalid lesson identity in {path}: {atom}")
                _insert(lessons, lesson["id"], lesson, "lesson")
        for atom, revision in state.get("activation_revisions", {}).items():
            _insert(merged.setdefault("activation_revisions", {}), atom, revision, "activation revision")
        for event in state["events"]:
            key = _json(event)
            if key not in event_keys:
                merged["events"].append(copy.deepcopy(event))
                event_keys.add(key)
    for lesson in lessons.values():
        merged["lessons"].setdefault(lesson["atom"], []).append(lesson)
    return merged, inputs


def _identity(value):
    keys = ("source_run", "atom", "step_id", "attempt", "sha256")
    if any(key not in value for key in keys):
        return None
    return (str(Path(value["source_run"]).expanduser().resolve()), value["atom"],
            value["step_id"], value["attempt"], value["sha256"])


def _attempts(runs):
    paths = sorted({path.resolve() for run in runs for path in run.rglob("attempts/*/record.json")})
    records = []
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        if type(record.get("success")) is not bool:
            raise ValueError(f"Attempt requires explicit boolean success: {path}")
        step = record["step"]
        get_spec(step["atom"])
        if not isinstance(step.get("id"), str) or type(record.get("attempt")) is not int or record["attempt"] < 0:
            raise ValueError(f"Invalid attempt identity: {path}")
        code = record.get("code")
        if not isinstance(code, str) or record.get("sha256") != source_hash(code):
            raise ValueError(f"Attempt code hash mismatch: {path}")
        run = path.parent.parent.parent
        identity = {"source_run": str(run), "atom": step["atom"], "step_id": step["id"],
                    "attempt": record["attempt"], "sha256": record["sha256"]}
        records.append((path, record, identity))
    return records


def recover(source_library, state_snapshots, runs, output):
    source_library = Path(source_library).expanduser().resolve()
    output = Path(output).expanduser()
    if output.exists() or output.is_symlink():
        raise ValueError(f"Output must be a new library root: {output}")
    output = output.resolve()
    if output == source_library or source_library in output.parents:
        raise ValueError("Output must not be inside the source library")
    runs = [Path(run).expanduser().resolve() for run in runs]
    for run in runs:
        if not run.is_dir():
            raise ValueError(f"Run directory does not exist: {run}")
        if run in output.parents:
            raise ValueError("Output must not be inside an evidence run directory")
    paths = [Path(path).expanduser().resolve() for path in state_snapshots]
    paths.append(source_library / "state.json")
    paths = list(dict.fromkeys(paths))
    merged, input_states = _merge_states(paths)
    records = _attempts(runs)
    sources = {}
    for path in sorted((source_library / "sources").iterdir()):
        if not path.is_file() or path.suffix != ".py":
            raise ValueError(f"Unexpected source entry: {path}")
        code = path.read_text(encoding="utf-8")
        if path.stem != source_hash(code):
            raise ValueError(f"Immutable source hash mismatch: {path}")
        sources[path.stem] = (path, code)
    for candidate in merged["candidates"].values():
        if candidate["sha256"] not in sources:
            raise ValueError(f"Missing source for candidate: {candidate['id']}")

    manifest = {"status": "complete", "source_library": str(source_library), "output": str(output),
                "output_schema_version": SCHEMA_VERSION,
                "state_inputs": input_states, "runs": [str(run) for run in runs],
                "run_inputs": [{"path": str(run),
                    "attempt_records": sum(run in path.parents for path, _, _ in records),
                    "failed_records": sum(run in path.parents and not record["success"]
                                          for path, record, _ in records)} for run in runs],
                "merged_counts": _counts(merged), "source_files_copied": len(sources),
                "attempt_records": len(records), "successful_records_skipped": 0,
                "failed_records": 0, "existing_lessons_matched": 0,
                "lessons_recovered": 0, "candidates_recovered": 0,
                "candidate_recovery_skipped": []}
    existing = Counter(_identity(lesson) for lessons in merged["lessons"].values() for lesson in lessons)
    existing.pop(None, None)
    consumed = Counter()
    candidate_hashes = {(candidate["atom"], candidate["sha256"]) for candidate in merged["candidates"].values()}
    output.parent.mkdir(parents=True, exist_ok=True)
    # mkdir is exclusive, so an existing output can never be replaced, including
    # if another process creates it after the initial preflight check.
    output.mkdir()
    try:
        library = Library(output)
        for digest, (_, code) in sources.items():
            library._atomic_write(library.sources_dir / f"{digest}.py", code)
        with library._locked():
            library._save(merged)
        for path, record, identity in records:
            if record["success"]:
                manifest["successful_records_skipped"] += 1
                continue
            manifest["failed_records"] += 1
            key = _identity(identity)
            if consumed[key] < existing[key]:
                manifest["existing_lessons_matched"] += 1
                consumed[key] += 1
            else:
                execution = copy.deepcopy(record.get("execution", {}))
                library.add_lesson(identity["atom"], {**identity,
                    "error": execution.get("error"), "verification": copy.deepcopy(record.get("verification")),
                    "execution": {key: value for key, value in execution.items() if key != "observation"},
                    "observation": execution.get("observation"), "recovered_from_record": str(path)})
                manifest["lessons_recovered"] += 1
            atom_hash = (identity["atom"], identity["sha256"])
            if atom_hash in candidate_hashes:
                continue
            checked = check_code(record["code"], identity["atom"])
            try:
                if not checked["ok"]:
                    raise ValueError(str(checked.get("errors", checked)))
                library.propose(identity["atom"], record["code"],
                    metadata={"stage": "recovered_failed_exploration", "source_run": identity["source_run"],
                              "recovered_from_record": str(path)}, dependencies=record.get("dependencies", {}))
                candidate_hashes.add(atom_hash)
                manifest["candidates_recovered"] += 1
            except (LibraryError, ValueError, SyntaxError, TypeError) as error:
                manifest["candidate_recovery_skipped"].append({"record": str(path), "reason": str(error)})
        with library._locked():
            final = library._load()
            if final["active"] or final["versions"] or final["cases"] or final["case_activations"]:
                raise ValueError("Recovery must not publish any skill or admit any regression case")
            manifest["final_counts"] = _counts(final)
            # Both fixed inodes contain a complete recovered baseline before any
            # experiment starts writing to this library.
            library._save(final)
        for candidate_id in final["candidates"]:
            library.get_candidate(candidate_id)
        library._atomic_write(output / "migration_manifest.json", _json(manifest))
        return manifest
    except BaseException:
        # Only this invocation's exclusively created destination is removed.
        shutil.rmtree(output)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-library", required=True)
    parser.add_argument("--state-snapshot", action="append", default=[])
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        manifest = recover(args.source_library, args.state_snapshot, args.run, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Recovery refused: {error}\n")
    print(_json(manifest), end="")


if __name__ == "__main__":
    main()
