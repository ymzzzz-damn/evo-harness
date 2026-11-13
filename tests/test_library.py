"""Library invariants: no accidental promotion, immutable pins, and replay coverage."""

from concurrent.futures import ProcessPoolExecutor
import errno
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evo.catalog import CATALOG, stub, validate_params
from evo.codecheck import check_code
from evo.library import Library, LibraryError


CODE = 'def skill(api, params):\n    api.wait(params.get("steps", 1))\n    return {"done": True}\n'
OTHER_CODE = 'def skill(api, params):\n    api.wait(params.get("steps", 2))\n    return {"done": True}\n'


def _add_lessons(root: str, worker: int):
    library = Library(root)
    for index in range(5):
        library.add_lesson("P01", {"worker": worker, "index": index})


def _write_candidates_and_cases(root: str, worker: int):
    library = Library(root)
    written = []
    for index in range(3):
        identity = {"worker": worker, "index": index}
        case_id = library.register_case({"atom": "P01", "task_config": identity})
        candidate = library.propose("P01", CODE, metadata=identity)
        library.record_validation(candidate["id"], {
            "candidate_sha256": candidate["sha256"], "success": True,
            "validation_kind": "fresh_environment_replay",
            "cases": [{"case_id": case_id, "success": True}],
        })
        library.add_lesson("P01", identity)
        written.append((candidate["id"], case_id, identity))
    return written


def _promote_candidate(root: str, candidate_id: str):
    try:
        record = Library(root).promote(candidate_id)
        return {"version": record["version"], "candidate_id": candidate_id}
    except LibraryError as exc:
        return {"error": str(exc), "candidate_id": candidate_id}


def _publish_atom(root: str, atom: str):
    library = Library(root)
    case_id = library.register_case({"atom": atom, "task_config": {"task": atom}})
    candidate = library.propose(atom, CODE)
    library.record_validation(candidate["id"], {
        "candidate_sha256": candidate["sha256"], "success": True,
        "validation_kind": "fresh_environment_replay",
        "cases": [{"case_id": case_id, "success": True}],
    })
    return library.promote(candidate["id"])


def _probe_maintenance_lock(root: str, atom: str, attempted, acquired):
    library = Library(root)
    attempted.set()
    with library.maintenance_lock(atom):
        # Exploration/state writes must still work while this long-lived lock is
        # held. Reusing .lock here would deadlock this call.
        library.add_lesson(atom, "maintenance probe")
        acquired.set()
    return atom


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.library = Library(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def candidate(self, atom="P01", code=CODE):
        return self.library.propose(atom, code, metadata={"task": "fixture", "seed": 3})

    def register(self, atom="P01", seed=3):
        return self.library.register_case({"atom": atom, "task_config": {"task": "fixture", "seed": seed},
                                           "prefix": [], "step": {"atom": atom, "params": {"object": "object"}}})

    def validate(self, candidate, case_ids=None, success=True):
        case_ids = case_ids if case_ids is not None else [case["id"] for case in
            self.library.cases(candidate["atom"]) + self.library.pending_cases(candidate["atom"])]
        return self.library.record_validation(candidate["id"], {
            "candidate_sha256": candidate["sha256"], "success": success,
            "validation_kind": "fresh_environment_replay",
            "cases": [{"case_id": identifier, "success": success, "native_task_success": False} for identifier in case_ids],
        })

    def publish(self, atom="P01", code=CODE):
        if not self.library.cases(atom):
            self.register(atom)
        candidate = self.candidate(atom, code)
        self.validate(candidate)
        return self.library.promote(candidate["id"])

    def test_initial_catalog_is_complete_and_empty(self):
        self.assertEqual(set(CATALOG), {f"P{index:02d}" for index in range(1, 41)})
        self.assertEqual(self.library.list_active(), [])
        self.assertEqual(self.library.snapshot(), {})
        self.assertEqual(len(list(Path(self.temporary.name, "stubs").glob("*.py"))), 40)
        self.assertIsNone(self.library.active("P01"))
        with self.assertRaises(ValueError):
            self.library.propose("P01", stub("P01"))
        with self.assertRaises(ValueError):
            self.library.active("P41")

    def test_mutable_state_inodes_stay_stable_for_cached_directory_entries(self):
        state_inode = self.library.state_path.stat().st_ino
        backup_inode = self.library.backup_path.stat().st_ino
        cached_name = Path(self.temporary.name) / "cached-state-inode.json"
        os.link(self.library.state_path, cached_name)
        for index in range(5):
            self.library.add_lesson("P01", {"index": index})
        self.assertEqual(self.library.state_path.stat().st_ino, state_inode)
        self.assertEqual(self.library.backup_path.stat().st_ino, backup_inode)
        self.assertEqual(len(json.loads(cached_name.read_text())["lessons"]["P01"]), 5)

    def test_cached_negative_exists_does_not_reinitialize_existing_library(self):
        self.library.add_lesson("P01", "preserve")
        with patch.object(Path, "exists", return_value=False):
            reopened = Library(self.temporary.name)
        self.assertEqual(reopened.lessons("P01")[0]["message"], "preserve")

    def test_cached_negative_open_cannot_overwrite_existing_library(self):
        self.library.add_lesson("P01", "preserve")
        before = self.library.state_path.read_bytes()
        with patch.object(Library, "_read_state_file", side_effect=FileNotFoundError("cached ENOENT")):
            with self.assertRaises(LibraryError):
                Library(self.temporary.name)
        self.assertEqual(self.library.state_path.read_bytes(), before)

    def test_transient_io_error_does_not_restore_stale_backup(self):
        self.library.add_lesson("P01", "first")
        self.library.add_lesson("P01", "latest acknowledged commit")
        before = self.library.state_path.read_bytes()
        read = Library._read_state_file

        def fail_main(path):
            if path == self.library.state_path:
                raise OSError(errno.EIO, "simulated transient read failure")
            return read(path)

        with patch.object(Library, "_read_state_file", side_effect=fail_main):
            with self.assertRaises(OSError):
                Library(self.temporary.name)
        self.assertEqual(self.library.state_path.read_bytes(), before)
        self.assertEqual(len(Library(self.temporary.name).lessons("P01")), 2)

    def test_legacy_state_is_upgraded_in_place_with_checksums(self):
        self.library.add_lesson("P01", "legacy")
        inode = self.library.state_path.stat().st_ino
        for path in (self.library.state_path, self.library.backup_path):
            state = json.loads(path.read_text())
            state["schema_version"] = 1
            state.pop("_state_sha256", None)
            path.write_text(json.dumps(state))
        reopened = Library(self.temporary.name)
        self.assertEqual(reopened.lessons("P01")[0]["message"], "legacy")
        reopened.add_lesson("P01", "upgrade")
        self.assertEqual(reopened.state_path.stat().st_ino, inode)
        for path in (reopened.state_path, reopened.backup_path):
            self.assertEqual(len(json.loads(path.read_text())["_state_sha256"]), 64)
            self.assertEqual(json.loads(path.read_text())["schema_version"], 2)

    def test_interrupted_backup_write_keeps_current_committed_state(self):
        self.library.add_lesson("P01", "committed")
        write = Library._write_in_place

        def fail_backup(path, text):
            if path == self.library.backup_path:
                write(path, '{"interrupted":')
                raise OSError("simulated crash during backup write")
            write(path, text)

        with patch.object(Library, "_write_in_place", side_effect=fail_backup):
            with self.assertRaisesRegex(OSError, "simulated crash"):
                self.library.add_lesson("P01", "uncommitted")
        reopened = Library(self.temporary.name)
        self.assertEqual([lesson["message"] for lesson in reopened.lessons("P01")], ["committed"])
        reopened.add_lesson("P01", "continued")
        self.assertEqual(len(reopened.lessons("P01")), 2)

    def test_interrupted_main_write_recovers_previous_state_on_same_inode(self):
        self.library.add_lesson("P01", "committed")
        inode = self.library.state_path.stat().st_ino
        write = Library._write_in_place

        def fail_main(path, text):
            if path == self.library.state_path:
                write(path, text[:len(text) // 2])
                raise OSError("simulated crash during main write")
            write(path, text)

        with patch.object(Library, "_write_in_place", side_effect=fail_main):
            with self.assertRaisesRegex(OSError, "simulated crash"):
                self.library.add_lesson("P01", "uncommitted")
        reopened = Library(self.temporary.name)
        self.assertEqual(reopened.state_path.stat().st_ino, inode)
        self.assertEqual([lesson["message"] for lesson in reopened.lessons("P01")], ["committed"])

    def test_syntactically_valid_torn_main_write_is_detected_by_checksum(self):
        self.library.add_lesson("P01", "committed")
        write = Library._write_in_place

        def fail_main(path, text):
            if path == self.library.state_path:
                torn = json.loads(text)
                torn["lessons"]["P01"][0]["message"] = "mixed old and new contents"
                write(path, json.dumps(torn))
                raise OSError("simulated valid-JSON torn write")
            write(path, text)

        with patch.object(Library, "_write_in_place", side_effect=fail_main):
            with self.assertRaisesRegex(OSError, "torn write"):
                self.library.add_lesson("P01", "uncommitted")
        self.assertEqual([lesson["message"] for lesson in Library(self.temporary.name).lessons("P01")],
                         ["committed"])

    def test_missing_main_recovers_backup_instead_of_creating_empty_library(self):
        self.library.add_lesson("P01", "committed")
        self.library.add_lesson("P01", "latest")
        self.library.state_path.unlink()
        reopened = Library(self.temporary.name)
        self.assertEqual([lesson["message"] for lesson in reopened.lessons("P01")], ["committed"])

    def test_broken_main_and_backup_fail_closed(self):
        for path in (self.library.state_path, self.library.backup_path):
            path.write_text("{broken")
        with self.assertRaisesRegex(LibraryError, "state or recovery backup"):
            Library(self.temporary.name)

    def test_unvalidated_and_failed_candidates_cannot_be_promoted(self):
        case_id = self.register()
        candidate = self.candidate()
        with self.assertRaisesRegex(LibraryError, "not been validated"):
            self.library.promote(candidate["id"])
        self.validate(candidate, [case_id], success=False)
        with self.assertRaisesRegex(LibraryError, "latest validation"):
            self.library.promote(candidate["id"])
        self.assertIsNone(self.library.active("P01"))

    def test_latest_failure_invalidates_previous_success(self):
        self.register()
        candidate = self.candidate()
        self.validate(candidate)
        self.validate(candidate, success=False)
        with self.assertRaises(LibraryError):
            self.library.promote(candidate["id"])

    def test_empty_or_partial_replay_cannot_pass(self):
        first = self.register(seed=3)
        self.publish()
        second = self.register(seed=4)
        candidate = self.library.propose("P01", OTHER_CODE, metadata={"new_case_id": second})
        self.validate(candidate, [])
        with self.assertRaises(LibraryError):
            self.library.promote(candidate["id"])
        self.validate(candidate, [first])
        with self.assertRaisesRegex(LibraryError, second):
            self.library.promote(candidate["id"])
        self.validate(candidate, [second])
        with self.assertRaisesRegex(LibraryError, first):
            self.library.promote(candidate["id"])
        self.validate(candidate, [first, second])
        self.assertEqual(self.library.promote(candidate["id"])["version"], 2)

    def test_case_admitted_during_validation_must_also_pass(self):
        active = self.publish()
        candidate = self.candidate()
        self.validate(candidate)
        added = self.register(seed=5)
        validation = {"candidate_sha256": active["sha256"], "success": True,
                      "validation_kind": "fresh_environment_replay",
                      "cases": [{"case_id": identifier, "success": True}
                                for identifier in active["case_ids"] + [added]]}
        self.library.accept_case(added, "P01", active["version"], validation)
        with self.assertRaisesRegex(LibraryError, added):
            self.library.promote(candidate["id"])

    def test_forged_hash_unknown_cases_and_duplicate_cases_rejected(self):
        case_id = self.register()
        candidate = self.candidate()
        with self.assertRaisesRegex(LibraryError, "candidate_sha256"):
            self.library.record_validation(candidate["id"], {"candidate_sha256": "0" * 64, "success": True, "cases": []})
        with self.assertRaisesRegex(LibraryError, "not registered"):
            self.validate(candidate, ["unknown"])
        with self.assertRaisesRegex(LibraryError, "Duplicate"):
            self.validate(candidate, [case_id, case_id])

    def test_promotion_preserves_metadata_and_native_outcome(self):
        published = self.publish()
        self.assertEqual(published["metadata"]["seed"], 3)
        self.assertFalse(published["validation"]["cases"][0]["native_task_success"])
        self.assertEqual(self.library.active("P01")["code"], CODE)
        self.assertEqual(self.library.snapshot(), {"P01": {"version": 1, "sha256": published["sha256"]}})
        self.assertEqual(self.library.promote(published["candidate_id"])["version"], 1)
        with self.assertRaisesRegex(LibraryError, "immutable"):
            self.validate(self.library.get_candidate(published["candidate_id"]))

    def test_version_history_and_rollback_are_persistent(self):
        first = self.publish()
        second = self.publish(code=OTHER_CODE)
        self.assertEqual(second["version"], 2)
        self.assertEqual(self.library.get_version("P01", 1)["code"], CODE)
        self.library.rollback("P01", 1)
        self.assertEqual(Library(self.temporary.name).active("P01")["sha256"], first["sha256"])
        self.assertEqual(self.library.get_version("P01", 2)["sha256"], second["sha256"])
        with self.assertRaises(LibraryError):
            self.library.rollback("P01", 200)

    def test_dependency_is_locked_to_published_version(self):
        first = self.publish("P01")
        caller = 'def skill(api, params):\n    return api.call("P01", params)\n'
        second_atom = self.publish("P02", caller)
        self.publish("P01", OTHER_CODE)
        self.assertEqual(second_atom["dependencies"], {"P01": {"version": 1, "sha256": first["sha256"]}})
        self.assertEqual(self.library.active("P02")["dependencies"]["P01"]["version"], 1)
        self.assertEqual(self.library.get_version("P01", **second_atom["dependencies"]["P01"])["code"], CODE)
        with self.assertRaisesRegex(LibraryError, "hash mismatch"):
            self.library.get_version("P01", 1, "0" * 64)

    def test_unpublished_dependency_and_transitive_cycle_rejected(self):
        caller = 'def skill(api, params):\n    return api.call("P01", params)\n'
        with self.assertRaisesRegex(LibraryError, "no published"):
            self.candidate("P02", caller)
        self.publish("P01")
        self.publish("P02", caller)
        recursive = 'def skill(api, params):\n    return api.call("P02", params)\n'
        with self.assertRaisesRegex(LibraryError, "cycle"):
            self.candidate("P01", recursive)

    def test_declared_dependencies_must_match_actual_calls(self):
        record = self.publish()
        with self.assertRaisesRegex(LibraryError, "exactly match"):
            self.library.propose("P02", CODE, dependencies={"P01": {"version": 1, "sha256": record["sha256"]}})

    def test_list_dependencies_pin_active_versions(self):
        record = self.publish()
        code = 'def skill(api, params):\n    return api.call("P01", params)\n'
        candidate = self.library.propose("P02", code, dependencies=["P01"])
        self.assertEqual(candidate["dependencies"], {"P01": {"version": 1, "sha256": record["sha256"]}})
        self.assertEqual(self.library.propose("P02", CODE, dependencies=[])["dependencies"], {})
        with self.assertRaisesRegex(LibraryError, "exactly match"):
            self.library.propose("P02", code, dependencies=[])

    def test_immutable_source_tampering_is_detected(self):
        first = self.publish()
        Path(first["source_path"]).write_text(OTHER_CODE, encoding="utf-8")
        with self.assertRaisesRegex(LibraryError, "integrity"):
            self.library.active("P01")

    def test_stale_candidate_cannot_overwrite_newer_active_version(self):
        self.register()
        older = self.candidate()
        newer = self.candidate(code=OTHER_CODE)
        self.validate(older)
        self.validate(newer)
        self.library.promote(newer["id"])
        with self.assertRaisesRegex(LibraryError, "changed during validation"):
            self.library.promote(older["id"])

    def test_publish_then_rollback_still_invalidates_inflight_candidate(self):
        first = self.publish()
        inflight = self.candidate()
        self.validate(inflight)
        self.publish(code=OTHER_CODE)
        self.library.rollback("P01", first["version"])
        with self.assertRaisesRegex(LibraryError, "changed during validation"):
            self.library.promote(inflight["id"])
        fresh = self.candidate()
        self.validate(fresh)
        self.assertEqual(self.library.promote(fresh["id"])["version"], 3)

    def test_case_identity_ignores_id_and_created_timestamp(self):
        case_id = self.register()
        case = self.library.get_case(case_id)
        case["created_at"] = "later"
        case["id"] = "untrusted-id"
        self.assertEqual(self.library.register_case(case), case_id)
        self.assertEqual(len(self.library.pending_cases("P01")), 1)
        self.assertEqual(self.library.cases("P01"), [])

    def test_staged_case_only_activates_with_candidate_promotion(self):
        case_id = self.register()
        candidate = self.library.propose("P01", CODE, metadata={"new_case_id": case_id})
        self.assertEqual(self.library.cases("P01"), [])
        self.validate(candidate, [case_id])
        self.assertEqual(self.library.cases("P01"), [])
        self.assertEqual(self.library.get_case(case_id)["atom"], "P01")
        record = self.library.promote(candidate["id"])
        self.assertEqual([case["id"] for case in self.library.cases("P01")], [case_id])
        self.assertEqual(self.library.pending_cases("P01"), [])
        state = json.loads(self.library.state_path.read_text())
        self.assertEqual(state["case_activations"][case_id]["candidate_id"], record["candidate_id"])

    def test_failed_pending_case_cannot_poison_unrelated_new_candidate(self):
        failed_case = self.register(seed=20)
        failed_candidate = self.library.propose("P01", CODE, metadata={"new_case_id": failed_case})
        self.validate(failed_candidate, [failed_case], success=False)
        valid_case = self.register(seed=21)
        valid_candidate = self.library.propose("P01", OTHER_CODE, metadata={"new_case_id": valid_case})
        self.validate(valid_candidate, [valid_case])
        published = self.library.promote(valid_candidate["id"])
        self.assertEqual(published["case_ids"], [valid_case])
        self.assertEqual([case["id"] for case in self.library.pending_cases("P01")], [failed_case])
        self.assertEqual([case["id"] for case in self.library.cases("P01")], [valid_case])

    def test_unrelated_pending_case_created_during_validation_is_not_an_obligation(self):
        self.publish()
        candidate = self.candidate(code=OTHER_CODE)
        self.validate(candidate)
        untested = self.register(seed=30)
        self.assertEqual(self.library.promote(candidate["id"])["version"], 2)
        self.assertEqual([case["id"] for case in self.library.pending_cases("P01")], [untested])

    def test_new_case_requires_explicit_fresh_environment_evidence(self):
        case_id = self.register()
        candidate = self.candidate()
        report = {"candidate_sha256": candidate["sha256"], "success": True,
                  "cases": [{"case_id": case_id, "success": True}]}
        with self.assertRaisesRegex(LibraryError, "fresh_environment_replay"):
            self.library.record_validation(candidate["id"], report)
        self.assertEqual(self.library.get_candidate(candidate["id"])["validations"], [])
        self.assertEqual(self.library.cases("P01"), [])

    def test_candidate_new_case_must_belong_to_its_atom(self):
        other = self.register(atom="P02")
        with self.assertRaisesRegex(LibraryError, "new_case_id"):
            self.library.propose("P01", CODE, metadata={"new_case_id": other})
        with self.assertRaisesRegex(LibraryError, "new_case_id"):
            self.library.propose("P01", CODE, metadata={"new_case_id": "missing"})

    def test_rebinding_creates_distinct_pending_case_without_changing_regression(self):
        active = self.publish()
        old_case = self.library.get_case(active["case_ids"][0])
        edited = json.loads(json.dumps(old_case))
        edited["step"]["params"]["object"] = "different-object"
        new_id = self.library.stage_case(edited)
        self.assertNotEqual(new_id, old_case["id"])
        self.assertEqual(self.library.get_case(old_case["id"])["step"]["params"]["object"], "object")
        self.assertEqual([case["id"] for case in self.library.cases("P01")], [old_case["id"]])

    def test_interrupted_promotion_commits_neither_version_nor_new_case(self):
        active = self.publish()
        new_case = self.register(seed=7)
        candidate = self.library.propose("P01", OTHER_CODE, metadata={"new_case_id": new_case})
        self.validate(candidate)
        inode = self.library.state_path.stat().st_ino
        write = Library._write_in_place

        def interrupt_main(path, text):
            if path == self.library.state_path:
                write(path, text[:len(text) // 2])
                raise OSError("synthetic interrupted case activation")
            write(path, text)

        with patch.object(Library, "_write_in_place", side_effect=interrupt_main):
            with self.assertRaisesRegex(OSError, "interrupted case activation"):
                self.library.promote(candidate["id"])
        reopened = Library(self.temporary.name)
        self.assertEqual(reopened.state_path.stat().st_ino, inode)
        self.assertEqual(reopened.active("P01")["version"], active["version"])
        self.assertEqual([case["id"] for case in reopened.pending_cases("P01")], [new_case])
        self.assertNotIn(new_case, {case["id"] for case in reopened.cases("P01")})
        self.assertEqual(reopened.get_candidate(candidate["id"])["status"], "validated")
        self.assertEqual(reopened.promote(candidate["id"])["version"], 2)
        self.assertEqual(reopened.pending_cases("P01"), [])

    def test_reused_active_version_can_accept_case_without_creating_version(self):
        active = self.publish()
        new_case = self.register(seed=11)
        report = {"candidate_sha256": active["sha256"], "success": True,
                  "validation_kind": "fresh_environment_replay",
                  "cases": [{"case_id": identifier, "success": True} for identifier in active["case_ids"] + [new_case]]}
        accepted = self.library.accept_case(new_case, "P01", active["version"], report)
        self.assertEqual(accepted["id"], new_case)
        self.assertEqual(self.library.active("P01")["candidate_id"], active["candidate_id"])
        self.assertEqual(self.library.active("P01")["version"], 1)
        self.assertEqual(self.library.pending_cases("P01"), [])
        self.assertEqual(len(self.library.cases("P01")), 2)
        # Repeating an acknowledged admission preserves the original immutable code.
        self.library.accept_case(new_case, "P01", 1, report)
        state = json.loads(self.library.state_path.read_text())
        self.assertEqual(len(state["versions"]["P01"]), 1)
        self.assertEqual(state["case_activations"][new_case]["validation"], report)

    def test_case_admission_rejects_wrong_hash_stale_version_missing_regression_and_failure(self):
        active = self.publish()
        new_case = self.register(seed=12)
        base = {"candidate_sha256": active["sha256"], "success": True,
                "validation_kind": "fresh_environment_replay",
                "cases": [{"case_id": identifier, "success": True} for identifier in active["case_ids"] + [new_case]]}
        bad_reports = [
            {**base, "candidate_sha256": "0" * 64},
            {**base, "validation_kind": "exploration"},
            {**base, "cases": [{"case_id": new_case, "success": True}]},
            {**base, "cases": [{"case_id": identifier, "success": identifier != new_case}
                               for identifier in active["case_ids"] + [new_case]]},
        ]
        for report in bad_reports:
            with self.subTest(report=report), self.assertRaises(LibraryError):
                self.library.accept_case(new_case, "P01", 1, report)
        self.assertEqual([case["id"] for case in self.library.pending_cases("P01")], [new_case])
        candidate = self.candidate(code=OTHER_CODE)
        self.validate(candidate, active["case_ids"])
        self.library.promote(candidate["id"])
        with self.assertRaisesRegex(LibraryError, "active version changed"):
            self.library.accept_case(new_case, "P01", 1, base)

    def _write_legacy_case_state(self, state):
        state = json.loads(json.dumps(state))
        state["schema_version"] = 1
        state["cases"].update(state.pop("pending_cases", {}))
        state.pop("case_lifecycle_version", None)
        state.pop("case_activations", None)
        state.pop("_state_sha256", None)
        for path in (self.library.state_path, self.library.backup_path):
            Library._write_in_place(path, Library._state_text(state))

    def test_legacy_migration_keeps_published_fresh_cases_and_quarantines_unpublished_cases(self):
        active = self.publish()
        unused = self.register(seed=15)
        candidate = self.candidate(code=OTHER_CODE)
        self.validate(candidate, [unused])  # Even a successful unpromoted candidate is not evidence of admission.
        state = json.loads(self.library.state_path.read_text())
        self._write_legacy_case_state(state)
        inode = self.library.state_path.stat().st_ino
        migrated = Library(self.temporary.name)
        self.assertEqual(migrated.state_path.stat().st_ino, inode)
        self.assertEqual(migrated.active("P01")["sha256"], active["sha256"])
        self.assertEqual([case["id"] for case in migrated.cases("P01")], active["case_ids"])
        self.assertEqual([case["id"] for case in migrated.pending_cases("P01")], [unused])
        self.assertEqual(migrated.get_candidate(candidate["id"])["status"], "validated")
        # Recovery in the same object cannot resurrect unverified legacy membership.
        migrated.state_path.write_text("{torn")
        self.assertEqual([case["id"] for case in migrated.cases("P01")], active["case_ids"])
        self.assertEqual([case["id"] for case in migrated.pending_cases("P01")], [unused])

    def test_legacy_published_case_without_fresh_marker_is_pending_but_source_stays_active(self):
        active = self.publish()
        state = json.loads(self.library.state_path.read_text())
        del state["versions"]["P01"]["1"]["validation"]["validation_kind"]
        self._write_legacy_case_state(state)
        migrated = Library(self.temporary.name)
        self.assertEqual(migrated.active("P01")["code"], CODE)
        self.assertEqual(migrated.cases("P01"), [])
        self.assertEqual([case["id"] for case in migrated.pending_cases("P01")], active["case_ids"])
        fresh_report = {"candidate_sha256": active["sha256"], "success": True,
                        "validation_kind": "fresh_environment_replay",
                        "cases": [{"case_id": active["case_ids"][0], "success": True}]}
        migrated.accept_case(active["case_ids"][0], "P01", 1, fresh_report)
        self.assertEqual(len(migrated.cases("P01")), 1)

    def test_schema_two_blocks_legacy_reader_and_writer_on_main_and_backup(self):
        case_id = self.register()

        def old_reader(path):
            state = json.loads(path.read_text())
            if state.get("schema_version") != 1:
                raise LibraryError("Unsupported library schema_version")
            return state

        for path in (self.library.state_path, self.library.backup_path):
            with self.subTest(path=path), self.assertRaisesRegex(LibraryError, "schema_version"):
                old_reader(path)
        self.assertEqual(self.library.get_case(case_id)["id"], case_id)

    def test_schema_one_with_existing_pending_lifecycle_is_upgraded_without_reclassification(self):
        active = self.publish()
        pending = self.register(seed=55)
        state = json.loads(self.library.state_path.read_text())
        state["schema_version"] = 1
        for path in (self.library.state_path, self.library.backup_path):
            Library._write_in_place(path, Library._state_text(state))
        reopened = Library(self.temporary.name)
        self.assertEqual([case["id"] for case in reopened.cases("P01")], active["case_ids"])
        self.assertEqual([case["id"] for case in reopened.pending_cases("P01")], [pending])
        for path in (reopened.state_path, reopened.backup_path):
            self.assertEqual(json.loads(path.read_text())["schema_version"], 2)

    def test_interrupted_schema_migration_recovers_v2_backup_and_cannot_resurrect_v1(self):
        active = self.publish()
        pending = self.register(seed=56)
        self._write_legacy_case_state(json.loads(self.library.state_path.read_text()))
        state_inode, backup_inode = self.library.state_path.stat().st_ino, self.library.backup_path.stat().st_ino
        write = Library._write_in_place

        def fail_migration_main(path, text):
            if path == self.library.state_path:
                write(path, text[:len(text) // 2])
                raise OSError("synthetic migration interruption")
            write(path, text)

        with patch.object(Library, "_write_in_place", side_effect=fail_migration_main):
            with self.assertRaisesRegex(OSError, "migration interruption"):
                Library(self.temporary.name)
        self.assertEqual(json.loads(self.library.backup_path.read_text())["schema_version"], 2)
        reopened = Library(self.temporary.name)
        self.assertEqual(reopened.state_path.stat().st_ino, state_inode)
        self.assertEqual(reopened.backup_path.stat().st_ino, backup_inode)
        self.assertEqual([case["id"] for case in reopened.cases("P01")], active["case_ids"])
        self.assertEqual([case["id"] for case in reopened.pending_cases("P01")], [pending])
        self.assertEqual(json.loads(reopened.state_path.read_text())["schema_version"], 2)

    def test_v2_missing_lifecycle_is_corruption_not_a_second_legacy_migration(self):
        self.publish()
        state = json.loads(self.library.state_path.read_text())
        state.pop("case_lifecycle_version")
        Library._write_in_place(self.library.state_path, Library._state_text(state))
        # Restore from the existing valid v2 backup rather than accepting raw cases.
        reopened = Library(self.temporary.name)
        self.assertEqual(json.loads(reopened.state_path.read_text())["schema_version"], 2)
        self.assertIn("case_lifecycle_version", json.loads(reopened.state_path.read_text()))
        self.assertIsNone(reopened.active("P01"))  # backup precedes the promotion

    def test_failures_are_kept_and_returned_data_are_copies(self):
        self.library.add_lesson("P01", {"failure": "missed grasp", "attempt": 1})
        lessons = self.library.lessons("P01")
        lessons[0]["attempt"] = 8
        self.assertEqual(self.library.lessons("P01")[0]["attempt"], 1)

    def test_concurrent_writers_do_not_lose_lessons(self):
        with ProcessPoolExecutor(max_workers=3) as executor:
            list(executor.map(_add_lessons, [self.temporary.name] * 3, range(3)))
        self.assertEqual(len(self.library.lessons("P01")), 15)

    def test_concurrent_writers_preserve_candidates_cases_and_validations(self):
        with ProcessPoolExecutor(max_workers=4) as executor:
            batches = list(executor.map(_write_candidates_and_cases,
                                        [self.temporary.name] * 4, range(4)))
        records = [record for batch in batches for record in batch]
        self.assertEqual(len({candidate_id for candidate_id, _, _ in records}), 12)
        self.assertEqual({case["id"] for case in self.library.pending_cases("P01")},
                         {case_id for _, case_id, _ in records})
        self.assertEqual(self.library.cases("P01"), [])
        self.assertEqual({(lesson["worker"], lesson["index"]) for lesson in self.library.lessons("P01")},
                         {(identity["worker"], identity["index"]) for _, _, identity in records})
        for candidate_id, case_id, identity in records:
            candidate = self.library.get_candidate(candidate_id)
            self.assertEqual(candidate["metadata"], identity)
            self.assertEqual(candidate["code"], CODE)
            self.assertEqual(candidate["validations"][0]["cases"], [{"case_id": case_id, "success": True}])

    def test_concurrent_same_atom_promotion_has_one_winner(self):
        self.register()
        candidates = [self.candidate(code=CODE if index % 2 else OTHER_CODE) for index in range(4)]
        for candidate in candidates:
            self.validate(candidate)
        with ProcessPoolExecutor(max_workers=4) as executor:
            outcomes = list(executor.map(_promote_candidate, [self.temporary.name] * 4,
                                         [candidate["id"] for candidate in candidates]))
        published = [outcome for outcome in outcomes if "version" in outcome]
        rejected = [outcome for outcome in outcomes if "error" in outcome]
        self.assertEqual(len(published), 1)
        self.assertEqual(published[0]["version"], 1)
        self.assertEqual(len(rejected), 3)
        self.assertTrue(all("changed during validation" in outcome["error"] for outcome in rejected))
        active = self.library.active("P01")
        self.assertEqual(active["candidate_id"], published[0]["candidate_id"])
        # Retrying an already committed promotion must never allocate version 2.
        with ProcessPoolExecutor(max_workers=4) as executor:
            repeated = list(executor.map(_promote_candidate, [self.temporary.name] * 4,
                                         [active["candidate_id"]] * 4))
        self.assertTrue(all(outcome["version"] == 1 for outcome in repeated))

    def test_concurrent_different_atom_promotions_preserve_all_versions(self):
        atoms = ["P01", "P02", "P09", "P10"]
        with ProcessPoolExecutor(max_workers=4) as executor:
            records = list(executor.map(_publish_atom, [self.temporary.name] * 4, atoms))
        self.assertEqual({record["atom"] for record in records}, set(atoms))
        self.assertEqual(set(self.library.snapshot()), set(atoms))
        for record in records:
            self.assertEqual(record["version"], 1)
            self.assertEqual(self.library.version(record["atom"], 1)["candidate_id"], record["candidate_id"])

    def test_maintenance_lock_serializes_same_atom_but_not_other_atoms(self):
        # Spawn gives each worker an independent interpreter, matching separate
        # Pods and avoiding inherited process-local lock bookkeeping.
        context = multiprocessing.get_context("spawn")
        with context.Manager() as manager, ProcessPoolExecutor(max_workers=2, mp_context=context) as executor:
            same_attempted, same_acquired = manager.Event(), manager.Event()
            other_attempted, other_acquired = manager.Event(), manager.Event()
            with self.library.maintenance_lock("P01"):
                same = executor.submit(_probe_maintenance_lock, self.temporary.name, "P01",
                                       same_attempted, same_acquired)
                other = executor.submit(_probe_maintenance_lock, self.temporary.name, "P02",
                                        other_attempted, other_acquired)
                self.assertTrue(same_attempted.wait(10))
                self.assertTrue(other_acquired.wait(10))
                self.assertEqual(other.result(timeout=10), "P02")
                self.assertFalse(same_acquired.wait(0.1))
                self.library.add_lesson("P01", "exploration while maintenance waits")
            self.assertTrue(same_acquired.wait(10))
            self.assertEqual(same.result(timeout=10), "P01")
        self.assertEqual(len(self.library.lessons("P01")), 2)
        self.assertEqual(len(self.library.lessons("P02")), 1)

    def test_nested_maintenance_locks_are_rejected_before_deadlock(self):
        with self.library.maintenance_lock("P01"):
            for atom in ("P01", "P02"):
                with self.subTest(atom=atom), self.assertRaisesRegex(LibraryError, "Only one maintenance lock"):
                    with Library(self.temporary.name).maintenance_lock(atom):
                        self.fail("nested lock unexpectedly acquired")
        with self.library.maintenance_lock("P02"):
            self.library.add_lesson("P02", "lock released after context")


class CodeContractTests(unittest.TestCase):
    def test_accepts_parameterized_numeric_policy(self):
        code = 'def skill(api, params):\n    delta = np.clip(params["delta"], -0.1, 0.1)\n    api.move_tcp(delta)\n    return {"done": True}\n'
        self.assertTrue(check_code(code, "P01")["ok"])

    def test_rejects_files_imports_private_and_oracle_access(self):
        snippets = [
            "import os\n    return params",
            'open("/tmp/x", "w")\n    return params',
            'np.save("/tmp/x", params)\n    return params',
            'api.env.step(params)\n    return params',
            'api.check(params)\n    return params',
            'x = api.__class__\n    return params',
            'getattr(api, "env")\n    return params',
        ]
        for snippet in snippets:
            with self.subTest(snippet=snippet):
                self.assertFalse(check_code("def skill(api, params):\n    " + snippet + "\n", "P01")["ok"])

    def test_requires_runtime_params_and_literal_dependency(self):
        self.assertFalse(check_code('def skill(api, params):\n    return {"done": True}\n', "P01")["ok"])
        self.assertFalse(check_code('def skill(api, params):\n    return api.call(params["atom"], params)\n', "P01")["ok"])
        self.assertFalse(check_code('def skill(api, params):\n    return api.call("P01", params)\n', "P01")["ok"])
        self.assertEqual(check_code('def skill(api, params):\n    return api.call("P02", params)\n', "P01")["dependencies"], ["P02"])
        self.assertFalse(check_code('def skill(api, params):\n    call = api.call\n    return call("P02", params)\n', "P01")["ok"])
        self.assertFalse(check_code('def skill(api, params):\n    a = api\n    return a.check(params)\n', "P01")["ok"])

    def test_no_top_level_execution_or_default_expression(self):
        self.assertFalse(check_code("danger = 1\n" + CODE, "P01")["ok"])
        self.assertFalse(check_code('def helper(x=api.observe()):\n    return x\n' + CODE, "P01")["ok"])

    def test_execution_deadline_cannot_be_caught_as_base_exception(self):
        for expression in ("", " BaseException", " BaseExceptionGroup", " SystemExit", " KeyboardInterrupt", " GeneratorExit"):
            with self.subTest(expression=expression):
                code = ('def skill(api, params):\n    try:\n        api.wait(params["duration"])\n'
                        f'    except{expression}:\n        pass\n    return {{}}\n')
                self.assertFalse(check_code(code, "P01")["ok"])
        code = ('def skill(api, params):\n    try:\n        api.wait(params["duration"])\n'
                '    except Exception:\n        pass\n    return {}\n')
        self.assertTrue(check_code(code, "P01")["ok"])

    def test_parameter_contract_reports_missing_bindings(self):
        validate_params("P01", {"object": "apple"})
        validate_params("P02", {"moving_entity": "apple"})
        with self.assertRaisesRegex(ValueError, "target"):
            validate_params("P02", {"object": "apple"}, strict=True)


if __name__ == "__main__":
    unittest.main()
