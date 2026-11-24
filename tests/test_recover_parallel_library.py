"""Evidence recovery must be lossless, conservative, and leave inputs untouched."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from evo.codecheck import source_hash
from evo.library import Library, LibraryError


SPEC = importlib.util.spec_from_file_location(
    "recover_parallel_library", Path(__file__).resolve().parents[1] / "scripts/recover_parallel_library.py")
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)

CODE = 'def skill(api, params):\n    api.wait(params.get("steps", 1))\n    return {}\n'
OTHER = 'def skill(api, params):\n    api.wait(params.get("steps", 2))\n    return {}\n'
RECOVERED = 'def skill(api, params):\n    api.wait(params.get("steps", 3))\n    return {}\n'
INVALID = 'def skill(api, params):\n    import os\n    return params\n'


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "old-library"
        self.library = Library(self.source)
        self.output = self.root / "recovered-library"
        self.run = self.root / "run"
        self.exploration = self.run / "episodes/000/exploration"
        self.exploration.mkdir(parents=True)

    def record(self, name, atom="A02", code=CODE, success=False, attempt=0):
        path = self.exploration / "attempts" / name / "record.json"
        path.parent.mkdir(parents=True)
        record = {"step": {"id": "step-0", "atom": atom, "params": {"steps": 1}},
                  "attempt": attempt, "success": success, "code": code,
                  "sha256": source_hash(code), "dependencies": {},
                  "execution": {"ok": success, "error": {"message": "fixture failure"},
                                "observation": {"image": "evidence"}},
                  "verification": {"success": success}}
        path.write_text(json.dumps(record))
        return path, record

    def snapshot(self, state, name="baseline.json"):
        path = self.root / name
        path.write_text(Library._state_text(state))
        return path

    def test_union_and_counted_failure_recovery_without_inventing_success(self):
        first_path, first_record = self.record("plan00_step00_try00")
        self.record("plan01_step00_try00")  # Same identity occurs twice; match by count.
        self.record("plan00_step01_try00", atom="A04", code=RECOVERED)
        self.record("plan00_step02_try00", atom="A05", code=INVALID)
        self.record("plan00_step03_try00", atom="A07", code=CODE, success=True)
        baseline_candidate = self.library.propose("A02", CODE)
        baseline_case = self.library.register_case({"atom": "A02", "task_config": {"seed": 1}})
        self.library.add_lesson("A02", {"source_run": str(self.exploration), "atom": "A02",
            "step_id": "step-0", "attempt": 0, "sha256": first_record["sha256"]})
        baseline = self.snapshot(Library._read_state_file(self.library.state_path))
        # A branch has lost the baseline records but kept its immutable sources.
        with self.library._locked():
            state = self.library._load()
            state.update(candidates={}, cases={}, pending_cases={}, lessons={})
            self.library._save(state)
        branch_candidate = self.library.propose("A04", OTHER)
        branch_case = self.library.register_case({"atom": "A04", "task_config": {"seed": 2}})
        self.library.add_lesson("A04", "unrelated retained lesson")
        evidence = {path: path.read_bytes() for directory in (self.source, self.run)
                    for path in directory.rglob("*") if path.is_file()}
        evidence[baseline] = baseline.read_bytes()
        manifest = recovery.recover(self.source, [baseline], [self.run, self.exploration], self.output)
        restored = Library(self.output)
        state = Library._read_state_file(restored.state_path)
        self.assertEqual(manifest["attempt_records"], 5)
        self.assertEqual(manifest["failed_records"], 4)
        self.assertEqual(manifest["successful_records_skipped"], 1)
        self.assertEqual(manifest["existing_lessons_matched"], 1)
        self.assertEqual(manifest["lessons_recovered"], 3)
        self.assertEqual(manifest["candidates_recovered"], 1)
        self.assertEqual(len(manifest["candidate_recovery_skipped"]), 1)
        self.assertEqual(manifest["final_counts"], {"candidates": 3, "cases": 0, "pending_cases": 2, "lessons": 5,
                                                    "active": 0, "versions": 0})
        self.assertEqual(set(state["pending_cases"]), {baseline_case, branch_case})
        self.assertEqual(state["cases"], {})
        self.assertEqual(state["schema_version"], 2)
        for candidate in (baseline_candidate, branch_candidate):
            self.assertEqual(state["candidates"][candidate["id"]]["sha256"], candidate["sha256"])
        recovered = [candidate for candidate in state["candidates"].values()
                     if candidate["id"] not in (baseline_candidate["id"], branch_candidate["id"])]
        self.assertEqual(recovered[0]["sha256"], source_hash(RECOVERED))
        self.assertEqual(recovered[0]["status"], "candidate")
        self.assertEqual(recovered[0]["validations"], [])
        self.assertFalse(state["active"])
        self.assertFalse(state["versions"])
        self.assertTrue(all("recovered_from_record" in lesson for lesson in restored.lessons("A05")))
        self.assertEqual(Library._read_state_file(restored.backup_path), state)
        self.assertEqual(json.loads((self.output / "migration_manifest.json").read_text()), manifest)
        for path, contents in evidence.items():
            self.assertEqual(path.read_bytes(), contents, str(path))

    def test_conflicting_candidate_case_and_lesson_ids_are_rejected(self):
        candidate = self.library.propose("A02", CODE)
        case_id = self.library.register_case({"atom": "A02", "task_config": {}})
        self.library.add_lesson("A02", "original")
        state = Library._read_state_file(self.library.state_path)
        for kind in ("candidate", "case", "lesson"):
            with self.subTest(kind=kind):
                conflicting = copy.deepcopy(state)
                if kind == "candidate":
                    conflicting["candidates"][candidate["id"]]["metadata"] = {"conflict": True}
                elif kind == "case":
                    conflicting["pending_cases"][case_id]["task_config"] = {"conflict": True}
                else:
                    conflicting["lessons"]["A02"][0]["message"] = "conflict"
                snapshot = self.snapshot(conflicting, f"conflict-{kind}.json")
                with self.assertRaisesRegex(ValueError, f"Conflicting {kind}"):
                    recovery.recover(self.source, [snapshot], [], self.output)
                self.assertFalse(self.output.exists())

    def test_any_published_state_is_refused(self):
        state = Library._read_state_file(self.library.state_path)
        for field in ("active", "versions"):
            with self.subTest(field=field):
                published = copy.deepcopy(state)
                published[field] = {"A02": 1 if field == "active" else {}}
                snapshot = self.snapshot(published, f"published-{field}.json")
                with self.assertRaisesRegex(ValueError, "published skills"):
                    recovery.recover(self.source, [snapshot], [], self.output)
                self.assertFalse(self.output.exists())

    def test_input_checksum_must_match(self):
        state = Library._read_state_file(self.library.state_path)
        state["events"] = [{"tampered": True}]
        snapshot = self.root / "tampered.json"
        snapshot.write_text(json.dumps(state))
        with self.assertRaisesRegex(LibraryError, "checksum"):
            recovery.recover(self.source, [snapshot], [], self.output)
        self.assertFalse(self.output.exists())

    def test_source_hash_mismatch_is_refused_before_destination_creation(self):
        candidate = self.library.propose("A02", CODE)
        Path(candidate["source_path"]).write_text(OTHER)
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            recovery.recover(self.source, [], [], self.output)
        self.assertFalse(self.output.exists())

    def test_missing_candidate_source_is_refused(self):
        candidate = self.library.propose("A02", CODE)
        Path(candidate["source_path"]).unlink()
        with self.assertRaisesRegex(ValueError, "Missing source"):
            recovery.recover(self.source, [], [], self.output)
        self.assertFalse(self.output.exists())

    def test_attempt_hash_mismatch_is_refused(self):
        path, record = self.record("plan00_step00_try00")
        record["code"] = OTHER
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "Attempt code hash mismatch"):
            recovery.recover(self.source, [], [self.run], self.output)
        self.assertFalse(self.output.exists())

    def test_existing_destination_is_never_overwritten(self):
        self.output.mkdir()
        marker = self.output / "keep"
        marker.write_text("preserve")
        with self.assertRaisesRegex(ValueError, "new library root"):
            recovery.recover(self.source, [], [], self.output)
        self.assertEqual(marker.read_text(), "preserve")

    def test_unresolved_dependency_retains_failure_without_inventing_a_version(self):
        caller = 'def skill(api, params):\n    return api.call("A04", params)\n'
        self.record("plan00_step00_try00", code=caller)
        manifest = recovery.recover(self.source, [], [self.run], self.output)
        restored = Library(self.output)
        self.assertEqual(manifest["lessons_recovered"], 1)
        self.assertEqual(manifest["candidates_recovered"], 0)
        self.assertEqual(len(manifest["candidate_recovery_skipped"]), 1)
        self.assertEqual(restored.list_active(), [])
        self.assertEqual(len(restored.lessons("A02")), 1)

    def test_cli_accepts_repeatable_snapshot_and_run_arguments(self):
        self.record("plan00_step00_try00")
        baseline = self.snapshot(Library._read_state_file(self.library.state_path))
        import contextlib
        import io
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            recovery.main(["--source-library", str(self.source), "--state-snapshot", str(baseline),
                           "--state-snapshot", str(baseline), "--run", str(self.run),
                           "--run", str(self.run), "--output", str(self.output)])
        self.assertEqual(json.loads(stdout.getvalue())["lessons_recovered"], 1)

    def test_legacy_v1_cases_and_v2_pending_merge_as_pending_without_mutating_inputs(self):
        first = self.library.stage_case({"atom": "A02", "task_config": {"seed": 20}})
        modern = Library._read_state_file(self.library.state_path)
        legacy = copy.deepcopy(modern)
        legacy["schema_version"] = 1
        legacy["cases"] = legacy.pop("pending_cases")
        legacy.pop("case_activations")
        legacy.pop("case_lifecycle_version")
        legacy_snapshot = self.snapshot(legacy, "legacy-v1.json")
        second = self.library.stage_case({"atom": "A04", "task_config": {"seed": 21}})
        evidence = {path: path.read_bytes() for path in self.source.rglob("*") if path.is_file()}
        evidence[legacy_snapshot] = legacy_snapshot.read_bytes()
        manifest = recovery.recover(self.source, [legacy_snapshot], [], self.output)
        restored = Library(self.output)
        self.assertEqual(restored.cases("A02"), [])
        self.assertEqual(restored.cases("A04"), [])
        self.assertEqual([case["id"] for case in restored.pending_cases("A02")], [first])
        self.assertEqual([case["id"] for case in restored.pending_cases("A04")], [second])
        self.assertEqual(manifest["output_schema_version"], 2)
        for path in (restored.state_path, restored.backup_path):
            state = Library._read_state_file(path)
            self.assertEqual(state["schema_version"], 2)
            self.assertEqual(state["cases"], {})
            self.assertEqual(state["case_activations"], {})
        for path, contents in evidence.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_v1_source_library_is_read_only_and_recovered_as_v2_pending(self):
        identifier = self.library.stage_case({"atom": "A02", "task_config": {"seed": 22}})
        legacy = Library._read_state_file(self.library.state_path)
        legacy["schema_version"] = 1
        legacy["cases"] = legacy.pop("pending_cases")
        legacy.pop("case_activations")
        legacy.pop("case_lifecycle_version")
        for path in (self.library.state_path, self.library.backup_path):
            Library._write_in_place(path, Library._state_text(legacy))
        source_before = self.library.state_path.read_bytes()
        recovery.recover(self.source, [], [], self.output)
        self.assertEqual(self.library.state_path.read_bytes(), source_before)
        restored = Library(self.output)
        self.assertEqual(restored.cases("A02"), [])
        self.assertEqual([case["id"] for case in restored.pending_cases("A02")], [identifier])

    def test_conflicting_v1_case_and_v2_pending_case_id_is_rejected(self):
        identifier = self.library.stage_case({"atom": "A02", "task_config": {"seed": 23}})
        legacy = Library._read_state_file(self.library.state_path)
        legacy["schema_version"] = 1
        legacy["cases"] = legacy.pop("pending_cases")
        legacy.pop("case_activations")
        legacy.pop("case_lifecycle_version")
        legacy["cases"][identifier]["task_config"] = {"seed": 999}
        snapshot = self.snapshot(legacy, "conflicting-legacy.json")
        with self.assertRaisesRegex(ValueError, "Conflicting case"):
            recovery.recover(self.source, [snapshot], [], self.output)
        self.assertFalse(self.output.exists())

    def test_admitted_case_evidence_without_version_history_is_not_auto_merged(self):
        state = Library._read_state_file(self.library.state_path)
        state["case_activations"]["case-orphan"] = {"atom": "A02", "version": 1, "sha256": source_hash(CODE)}
        snapshot = self.snapshot(state, "orphan-published-evidence.json")
        with self.assertRaisesRegex(ValueError, "published skills"):
            recovery.recover(self.source, [snapshot], [], self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
