"""Transactional, append-only source storage for validated atomic skills.

First admission requires independent replay. Regression-mode promotion covers
every formal case; an explicit comparison policy can update a nonempty atom
without replay, with a distinct admission receipt and no claimed case coverage.
Working drafts preserve incomplete work but are never active dependencies.
Code is immutable and dependencies refer to exact version/hash pairs. The library
is a local trusted-process registry; it does not authenticate validation workers.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid

from .catalog import CATALOG, CATALOG_VERSION, get_spec, stub, validate_library_catalog
from .codecheck import require_valid_code, source_hash


SCHEMA_VERSION = 2
READABLE_SCHEMA_VERSIONS = (1, SCHEMA_VERSION)
CASE_LIFECYCLE_VERSION = 1
_HELD_MAINTENANCE_LOCK = ContextVar("evo_maintenance_lock", default=None)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _clone(value):
    """Check serializability at the boundary; don't retain mutable caller data."""
    return json.loads(_json(value))


class LibraryError(ValueError):
    pass


class _StateIntegrityError(LibraryError):
    pass


class Library:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.sources_dir = self.root / "sources"
        self.sources_dir.mkdir(exist_ok=True)
        self.stubs_dir = self.root / "stubs"
        self.stubs_dir.mkdir(exist_ok=True)
        self.state_path = self.root / "state.json"
        self.backup_path = self.root / "state.backup.json"
        with self._locked():
            catalog_path = self.root / "catalog.json"
            if catalog_path.exists() and set(json.loads(catalog_path.read_text())) != set(CATALOG):
                raise LibraryError("Incompatible catalog.json; use a separate Human300 P/A/C32 library")
            # Don't trust a cached negative directory lookup when another client
            # may have initialized the shared library while we waited for .lock.
            state = self._load(allow_missing=True)
            if state is None:
                self._initialize({"schema_version": SCHEMA_VERSION, "atomic_catalog_version": CATALOG_VERSION,
                                  "active": {}, "versions": {},
                                  "candidates": {}, "cases": {}, "pending_cases": {}, "case_activations": {},
                                  "case_lifecycle_version": CASE_LIFECYCLE_VERSION, "lessons": {}, "events": []})
            elif state["schema_version"] != SCHEMA_VERSION:
                if state.get("case_lifecycle_version") is None:
                    self._migrate_case_lifecycle(state)
                else:
                    self._commit_schema_migration(state)
            if state is not None and state.get("atomic_catalog_version") is None:
                # Only a completely empty unversioned registry passed the guard.
                state["atomic_catalog_version"] = CATALOG_VERSION
                self._save(state)
            if not catalog_path.exists() or json.loads(catalog_path.read_text()) != CATALOG:
                self._atomic_write(catalog_path, _json(CATALOG))
            for atom in CATALOG:
                path = self.stubs_dir / f"{atom}.py"
                if not path.exists() or path.read_text() != stub(atom):
                    self._atomic_write(path, stub(atom))

    @contextmanager
    def _locked(self):
        with (self.root / ".lock").open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def maintenance_lock(self, atom: str):
        """Serialize replay/promotion for one atom without blocking exploration.

        All workers sharing the library must use a filesystem with cross-process
        flock support. Hold only one maintenance lock at a time; state operations
        take their own short-lived lock inside this context.
        """
        get_spec(atom)
        if _HELD_MAINTENANCE_LOCK.get() is not None:
            raise LibraryError("Only one maintenance lock may be held in a context")
        with (self.root / f".maintenance-{atom}.lock").open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            token = _HELD_MAINTENANCE_LOCK.set((str(self.root), atom))
            try:
                yield
            finally:
                _HELD_MAINTENANCE_LOCK.reset(token)
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _atomic_write(path: Path, text: str):
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _state_text(state: dict) -> str:
        payload = dict(state)
        payload.pop("_state_sha256", None)
        digest = hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()
        return _json({**payload, "_state_sha256": digest})

    @staticmethod
    def _write_in_place(path: Path, text: str):
        """Keep the inode stable across clients; flush and close before unlock.

        Replacing a mutable file on a distributed filesystem can leave another
        mount's directory-entry cache pointing at an obsolete inode, even when
        flock itself works across nodes. Reopening this stable inode relies on
        the filesystem's close-to-open consistency (open-cache must be disabled).
        """
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            handle.truncate()
            os.fsync(handle.fileno())
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _read_state_file(path: Path) -> dict:
        # Open after acquiring .lock, and close before releasing it. A retained
        # descriptor would bypass close-to-open cache invalidation on JuiceFS.
        with path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        if not isinstance(state, dict):
            raise _StateIntegrityError("Library state must be a dictionary")
        schema = state.get("schema_version")
        if type(schema) is not int or schema not in READABLE_SCHEMA_VERSIONS:
            raise LibraryError("Unsupported library schema_version")
        if (any(not isinstance(state.get(key), dict)
                for key in ("active", "versions", "candidates", "cases", "lessons"))
                or not isinstance(state.get("events"), list)):
            raise _StateIntegrityError("Incomplete library state")
        if schema == SCHEMA_VERSION and state.get("case_lifecycle_version") is None:
            raise _StateIntegrityError("Schema 2 requires explicit case lifecycle state")
        if state.get("case_lifecycle_version") is not None:
            if state["case_lifecycle_version"] != CASE_LIFECYCLE_VERSION:
                raise LibraryError("Unsupported case lifecycle version")
            if any(not isinstance(state.get(key), dict) for key in ("pending_cases", "case_activations")):
                raise _StateIntegrityError("Incomplete case lifecycle state")
            if set(state["cases"]) & set(state["pending_cases"]):
                raise _StateIntegrityError("Case is simultaneously pending and active")
        if "_state_sha256" in state:
            payload = dict(state)
            digest = payload.pop("_state_sha256")
            try:
                actual = hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()
            except (TypeError, ValueError) as exc:
                raise _StateIntegrityError("Invalid library state payload") from exc
            if digest != actual:
                raise _StateIntegrityError("Library state checksum mismatch")
        # Schema 1 is read only for migration. All normal writers require v2.
        try:
            validate_library_catalog(state)
        except ValueError as error:
            raise LibraryError(str(error)) from error
        return state

    def _load(self, allow_missing: bool = False) -> dict | None:
        # Permission/I/O errors aren't proof of a torn write. In particular, an
        # intermittent EIO must never replace a committed state with its backup.
        recoverable = (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError, _StateIntegrityError)
        try:
            return self._read_state_file(self.state_path)
        except recoverable as main_error:
            try:
                state = self._read_state_file(self.backup_path)
            except recoverable as backup_error:
                if (allow_missing and isinstance(main_error, FileNotFoundError)
                        and isinstance(backup_error, FileNotFoundError)):
                    return None
                raise LibraryError(f"Cannot read library state or recovery backup: "
                                   f"state={main_error}; backup={backup_error}") from main_error
            # A crash during an in-place state write is recoverable because the
            # preceding committed state was made durable before that write began.
            self._write_in_place(self.state_path, self._state_text(state))
            return state

    def _initialize(self, state: dict):
        try:
            descriptor = os.open(self.state_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            # O_EXCL asks the metadata service rather than trusting a stale
            # negative lookup. If reading is still stale, fail rather than reset.
            self._load()
            return
        os.close(descriptor)
        text = self._state_text(state)
        self._write_in_place(self.backup_path, text)
        self._write_in_place(self.state_path, text)

    def _save(self, state: dict):
        if state.get("schema_version") != SCHEMA_VERSION:
            raise LibraryError("Refusing to write an unmigrated library state")
        text = self._state_text(state)
        previous = self._load()
        # Never replace either mutable inode. If backup writing fails, main is
        # untouched; if main writing fails, backup contains the last valid state.
        self._write_in_place(self.backup_path, self._state_text(previous))
        self._write_in_place(self.state_path, text)

    def _commit_schema_migration(self, state: dict):
        previous_schema = state["schema_version"]
        state["schema_version"] = SCHEMA_VERSION
        state["events"].append({"event": "migrate_schema", "from_schema": previous_schema,
                                "schema_version": SCHEMA_VERSION, "at": _now()})
        text = self._state_text(state)
        # The recovery copy is upgraded first. If the main write is interrupted,
        # both new and old clients encounter v2 rather than restoring a v1 backup
        # that could let an old worker bypass the pending-case lifecycle.
        self._write_in_place(self.backup_path, text)
        self._write_in_place(self.state_path, text)

    def _migrate_case_lifecycle(self, state: dict):
        """Quarantine legacy cases without a successful published fresh replay.

        Published source/version history stays intact, including versions whose
        older evidence predates the freshness marker. Such cases must be replayed
        before they can become new regression obligations.
        """
        cases = state["cases"]
        evidence = {}
        for atom, versions in state["versions"].items():
            for record in versions.values():
                report = record.get("validation", {})
                if (report.get("validation_kind") != "fresh_environment_replay"
                        or report.get("success") is not True
                        or report.get("candidate_sha256") != record.get("sha256")):
                    continue
                results = report.get("cases", [])
                if not isinstance(results, list) or not results or any(
                        not isinstance(item, dict) or item.get("success") is not True for item in results):
                    continue
                for item in results:
                    identifier = item.get("case_id")
                    case = cases.get(identifier)
                    if case is not None and case.get("atom") == atom:
                        evidence[identifier] = {"atom": atom, "version": record["version"],
                                                "sha256": record["sha256"],
                                                "candidate_id": record.get("candidate_id"),
                                                "validation_kind": "fresh_environment_replay",
                                                "migrated_from_published_version": True}
        state["pending_cases"] = {identifier: case for identifier, case in cases.items() if identifier not in evidence}
        state["cases"] = {identifier: case for identifier, case in cases.items() if identifier in evidence}
        state["case_activations"] = evidence
        state["case_lifecycle_version"] = CASE_LIFECYCLE_VERSION
        state["events"].append({"event": "migrate_case_lifecycle", "at": _now(),
                                "retained_cases": len(evidence), "pending_cases": len(state["pending_cases"])})
        self._commit_schema_migration(state)

    def _source(self, digest: str) -> str:
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise LibraryError("Invalid source hash")
        path = self.sources_dir / f"{digest}.py"
        try:
            code = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise LibraryError(f"Missing immutable source {digest}") from exc
        if source_hash(code) != digest:
            raise LibraryError(f"Source integrity check failed: {digest}")
        return code

    def _hydrate(self, record: dict) -> dict:
        result = copy.deepcopy(record)
        result["code"] = self._source(record["sha256"])
        result["source"] = result["code"]
        result["source_path"] = str(self.sources_dir / f'{record["sha256"]}.py')
        result["description"] = record.get("description") or record.get("metadata", {}).get("description") or get_spec(record["atom"])["description"]
        return result

    @staticmethod
    def _version(state: dict, atom: str, version: int) -> dict:
        if type(version) is not int or version < 1:
            raise LibraryError("version must be a positive integer")
        try:
            return state["versions"][atom][str(version)]
        except KeyError as exc:
            raise LibraryError(f"Unknown published version: {atom}@{version}") from exc

    def _check_graph(self, state: dict, dependencies: dict, root_atom: str | None = None):
        """Validate immutable dependency closure; never silently use newer active code."""
        def visit(atom: str, pin: dict, path: tuple[str, ...]):
            get_spec(atom)
            if atom in path:
                raise LibraryError(f"Dependency cycle: {' -> '.join(path + (atom,))}")
            if not isinstance(pin, dict) or set(pin) != {"version", "sha256"}:
                raise LibraryError(f"Dependency {atom} needs exactly version and sha256")
            record = self._version(state, atom, pin["version"])
            if record["sha256"] != pin["sha256"]:
                raise LibraryError(f"Dependency hash mismatch for {atom}@{pin['version']}")
            self._source(record["sha256"])
            for child, child_pin in record["dependencies"].items():
                visit(child, child_pin, path + (atom,))
        for atom, pin in dependencies.items():
            visit(atom, pin, (root_atom,) if root_atom else ())

    def active(self, atom: str) -> dict | None:
        get_spec(atom)
        with self._locked():
            state = self._load()
            version = state["active"].get(atom)
            if version is None:
                return None
            record = self._version(state, atom, version)
            self._check_graph(state, record["dependencies"], atom)
            return self._hydrate(record)

    def version(self, atom: str, version: int, sha256: str | None = None) -> dict:
        """Resolve an exact pin for execution, even after its active version changes."""
        get_spec(atom)
        with self._locked():
            state = self._load()
            record = self._version(state, atom, version)
            if sha256 is not None and record["sha256"] != sha256:
                raise LibraryError(f"Dependency hash mismatch for {atom}@{version}")
            self._check_graph(state, record["dependencies"], atom)
            return self._hydrate(record)

    def get_version(self, atom: str, version: int, sha256: str | None = None) -> dict:
        return self.version(atom, version, sha256)

    def list_active(self) -> list[dict]:
        with self._locked():
            state = self._load()
            result = []
            for atom, version in sorted(state["active"].items()):
                record = self._version(state, atom, version)
                self._check_graph(state, record["dependencies"], atom)
                result.append(self._hydrate(record))
            return result

    def propose(self, atom: str, code: str, metadata: dict | None = None,
                dependencies: dict | list[str] | None = None) -> dict:
        get_spec(atom)
        checked = require_valid_code(code, atom)
        metadata = _clone(metadata or {})
        if not isinstance(metadata, dict):
            raise LibraryError("metadata must be a dictionary")
        if "description" in metadata and (not isinstance(metadata["description"], str) or not metadata["description"].strip()):
            raise LibraryError("Implementation description must be a nonempty string")
        with self._locked():
            state = self._load()
            new_case_id = metadata.get("new_case_id")
            if new_case_id is not None:
                case = self._find_case(state, new_case_id)
                if case is None or case["atom"] != atom:
                    raise LibraryError("metadata.new_case_id must identify a staged or active case for this atom")
            pins = {}
            if isinstance(dependencies, list):
                if any(not isinstance(value, str) for value in dependencies) or len(set(dependencies)) != len(dependencies):
                    raise LibraryError("dependencies list must contain unique atomic skill IDs")
                if set(dependencies) != set(checked["dependencies"]):
                    raise LibraryError("Declared dependencies must exactly match literal api.call dependencies")
            if dependencies is None or isinstance(dependencies, list):
                for dependency in checked["dependencies"]:
                    version = state["active"].get(dependency)
                    if version is None:
                        raise LibraryError(f"Dependency {dependency} has no published implementation")
                    record = self._version(state, dependency, version)
                    pins[dependency] = {"version": version, "sha256": record["sha256"]}
            else:
                pins = _clone(dependencies)
                if not isinstance(pins, dict):
                    raise LibraryError("dependencies must be an atom-to-pin dictionary")
            if set(pins) != set(checked["dependencies"]):
                raise LibraryError("Declared dependencies must exactly match literal api.call dependencies")
            self._check_graph(state, pins, atom)
            digest = checked["sha256"]
            source_path = self.sources_dir / f"{digest}.py"
            if source_path.exists():
                self._source(digest)
            else:
                self._atomic_write(source_path, code)
            identifier = f"{atom}-{uuid.uuid4().hex}"
            candidate = {"id": identifier, "atom": atom, "sha256": digest, "dependencies": pins,
                         "metadata": metadata, "created_at": _now(), "status": "candidate",
                         "validations": [], "base_version": state["active"].get(atom),
                         "base_activation_revision": state.get("activation_revisions", {}).get(atom, 0)}
            state["candidates"][identifier] = candidate
            self._save(state)
            return self._hydrate(candidate)

    def get_candidate(self, candidate_id: str) -> dict:
        with self._locked():
            state = self._load()
            try:
                candidate = state["candidates"][candidate_id]
            except KeyError as exc:
                raise LibraryError(f"Unknown candidate: {candidate_id}") from exc
            return self._hydrate(candidate)

    @staticmethod
    def _find_case(state: dict, identifier: str) -> dict | None:
        if not isinstance(identifier, str):
            return None
        return state["cases"].get(identifier) or state["pending_cases"].get(identifier)

    def _validate_report_cases(self, state: dict, atom: str, report: dict, digest: str,
                               dependencies=None) -> set[str]:
        if type(report.get("success")) is not bool:
            raise LibraryError("validation report requires a boolean success")
        if report.get("candidate_sha256") != digest:
            raise LibraryError("Validation must identify the exact candidate_sha256")
        cases = report.get("cases")
        if not isinstance(cases, list):
            raise LibraryError("Validation report requires cases list")
        seen = set()
        for case in cases:
            if not isinstance(case, dict) or not isinstance(case.get("case_id"), str) or type(case.get("success")) is not bool:
                raise LibraryError("Each case result requires case_id and boolean success")
            identifier = case["case_id"]
            if identifier in seen:
                raise LibraryError(f"Duplicate validation case: {identifier}")
            seen.add(identifier)
            registered = self._find_case(state, identifier)
            if registered is None or registered["atom"] != atom:
                raise LibraryError(f"Validation case is not registered for {atom}: {identifier}")
            if identifier in state["pending_cases"] and report.get("validation_kind") != "fresh_environment_replay":
                raise LibraryError("Pending cases require fresh_environment_replay evidence")
            evidence = case.get("replay_evidence", {})
            if not isinstance(evidence, dict):
                raise LibraryError("Replay evidence must be a dictionary")
            origin = evidence.get("kind", "fresh")
            if origin not in ("fresh", "cached_success"):
                raise LibraryError("Unknown replay evidence origin")
            if origin == "cached_success":
                from .replay_evidence import validate_cached_result
                try:
                    validate_cached_result(self.root, state, registered,
                        {"atom": atom, "sha256": digest, "dependencies": dependencies or {}}, report, case)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    raise LibraryError(f"Invalid cached replay evidence: {error}") from error
        if "replay_policy" in report:
            from .replay_evidence import POLICY
            cached = sum(c.get("replay_evidence", {}).get("kind") == "cached_success" for c in cases)
            if (report["replay_policy"] != POLICY or report.get("validation_kind") != "fresh_environment_replay"
                    or any(c.get("replay_evidence", {}).get("kind") not in ("fresh", "cached_success") for c in cases)
                    or report.get("replay_counts") != {"total": len(cases), "fresh": len(cases) - cached, "cached": cached}):
                raise LibraryError("Replay policy/counts must identify actual fresh and cached evidence")
        return {case["case_id"] for case in cases if case["success"] is True}

    def record_validation(self, candidate_id: str, report: dict) -> dict:
        report = _clone(report)
        if not isinstance(report, dict) or type(report.get("success")) is not bool:
            raise LibraryError("validation report requires a boolean success")
        with self._locked():
            state = self._load()
            if candidate_id not in state["candidates"]:
                raise LibraryError(f"Unknown candidate: {candidate_id}")
            candidate = state["candidates"][candidate_id]
            if candidate["status"] == "promoted":
                raise LibraryError("A promoted candidate is immutable; propose a new candidate")
            self._validate_report_cases(state, candidate["atom"], report, candidate["sha256"], candidate["dependencies"])
            cases = report["cases"]
            report["recorded_at"] = _now()
            candidate["validations"].append(report)
            candidate["status"] = "validated" if report["success"] and cases and all(case["success"] for case in cases) else "rejected"
            self._save(state)
            return copy.deepcopy(report)

    def promote(self, candidate_id: str) -> dict:
        with self._locked():
            state = self._load()
            if candidate_id not in state["candidates"]:
                raise LibraryError(f"Unknown candidate: {candidate_id}")
            candidate = state["candidates"][candidate_id]
            if candidate["status"] == "promoted":
                return self._hydrate(self._version(state, candidate["atom"], candidate["version"]))
            if not candidate["validations"]:
                raise LibraryError("Candidate has not been validated")
            report = candidate["validations"][-1]
            if report.get("success") is not True or candidate["status"] != "validated":
                raise LibraryError("The latest validation must be successful")
            atom = candidate["atom"]
            # The revision also detects publish/rollback back to the same version
            # while validation was running (an ABA change). Missing revisions are
            # zero so existing schema-v1 libraries and candidates remain readable.
            revision = state.get("activation_revisions", {}).get(atom, 0)
            if (candidate["base_version"] != state["active"].get(atom)
                    or candidate.get("base_activation_revision", 0) != revision):
                raise LibraryError("The active version changed during validation; propose and validate again")
            required = {identifier for identifier, case in state["cases"].items() if case["atom"] == atom}
            if candidate["metadata"].get("new_case_id") is not None:
                required.add(candidate["metadata"]["new_case_id"])
            passed = self._validate_report_cases(state, atom, report, candidate["sha256"], candidate["dependencies"])
            if not passed or not required.issubset(passed):
                raise LibraryError(f"Missing successful regression coverage: {sorted(required - passed) or ['at least one registered case']}")
            if any(case["success"] is not True for case in report["cases"]):
                raise LibraryError("Failed validation cases prevent promotion")
            self._source(candidate["sha256"])
            self._check_graph(state, candidate["dependencies"], atom)
            versions = state["versions"].setdefault(atom, {})
            version = max((int(value) for value in versions), default=0) + 1
            record = {"atom": atom, "version": version, "sha256": candidate["sha256"],
                      "dependencies": copy.deepcopy(candidate["dependencies"]),
                      "metadata": copy.deepcopy(candidate["metadata"]),
                      "candidate_id": candidate_id, "created_at": _now(),
                      "validation": copy.deepcopy(report), "case_ids": sorted(passed)}
            versions[str(version)] = record
            for identifier in passed & set(state["pending_cases"]):
                state["cases"][identifier] = state["pending_cases"].pop(identifier)
                state["case_activations"][identifier] = {
                    "atom": atom, "version": version, "sha256": candidate["sha256"],
                    "candidate_id": candidate_id, "validation_kind": "fresh_environment_replay", "activated_at": _now()}
            state["active"][atom] = version
            state.setdefault("activation_revisions", {})[atom] = revision + 1
            candidate["status"] = "promoted"
            candidate["version"] = version
            state["events"].append({"event": "promote", "atom": atom, "version": version, "at": _now()})
            self._save(state)
            return self._hydrate(record)

    def stage_case(self, case: dict) -> str:
        """Stage a proposed case without adding a regression obligation.

        A fresh successful replay must accompany promote() or accept_case() before
        cases() returns it. Failed/unattempted staged cases never block unrelated
        candidates. Content IDs remain stable when retrieved records are restaged.
        """
        case = _clone(case)
        if not isinstance(case, dict):
            raise LibraryError("case must be a dictionary")
        get_spec(case.get("atom"))
        case.pop("id", None)
        case.pop("created_at", None)
        identifier = "case-" + hashlib.sha256(_json(case).encode("utf-8")).hexdigest()[:24]
        with self._locked():
            state = self._load()
            if self._find_case(state, identifier) is None:
                state["pending_cases"][identifier] = {**case, "id": identifier, "created_at": _now()}
                self._save(state)
        return identifier

    def remember(self, atom: str, payload: dict, kind="attempt") -> dict:
        """Append durable per-atom learning memory, including unsuccessful code.

        This does not publish a callable, activate a case, or establish success.
        Large observations/traces live outside the transactional registry state.
        """
        get_spec(atom)
        if kind not in {"attempt", "draft"}:
            raise LibraryError("Memory kind must be attempt or draft")
        payload = _clone(payload)
        identifier = "memory-" + hashlib.sha256(_json([atom, kind, payload]).encode()).hexdigest()[:24]
        from .admission import host_success
        status = "observed_success" if kind == "attempt" and host_success(payload.get("record", {})) else "incomplete"
        item = dict(id=identifier, atom=atom, kind=kind, status=status, payload=payload, created_at=_now())
        path = self.root / "working_memory" / atom / (identifier + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked():
            state = self._load()
            index = state.setdefault("working_memory_index", {}).setdefault(atom, [])
            if not any(row["id"] == identifier for row in index):
                self._atomic_write(path, _json(item))
                index.append({k: item[k] for k in ("id", "atom", "kind", "status", "created_at")})
                index[-1]["task"] = payload.get("task_config", {}).get("task")
                index[-1]["goal"] = payload.get("step", {}).get("goal")
                self._save(state)
            else:
                item = json.loads(path.read_text())
        return item

    def memory(self, atom: str, identifier: str) -> dict:
        get_spec(atom)
        with self._locked():
            if not any(row["id"] == identifier for row in self._load().get("working_memory_index", {}).get(atom, [])):
                raise LibraryError("Unknown working memory")
            return json.loads((self.root / "working_memory" / atom / (identifier + ".json")).read_text())

    def working_memory(self, atom: str, limit=8, task=None) -> list[dict]:
        get_spec(atom)
        if type(limit) is not int or limit < 1:
            raise LibraryError("Working memory limit must be positive")
        with self._locked():
            rows = self._load().get("working_memory_index", {}).get(atom, [])
            ranked = sorted(enumerate(rows), key=lambda pair: (pair[1].get("task") == task, pair[0]), reverse=True)
            return [json.loads((self.root / "working_memory" / atom / (row["id"] + ".json")).read_text())
                    for _, row in ranked[:limit]]

    def working_catalog(self) -> list[dict]:
        with self._locked():
            return [copy.deepcopy(rows[-1]) for _, rows in sorted(
                self._load().get("working_memory_index", {}).items()) if rows]

    def publish_comparison(self, candidate_id: str, memory_id: str, rationale: str) -> dict:
        """Update a nonempty atom by comparison, with explicitly weaker evidence.

        A successful *source attempt* is required; the parameterized candidate is
        not claimed to have executed. Partial implementations remain drafts.
        """
        from .admission import code_diff, host_success, same_program, validate_comparison
        with self._locked():
            state = self._load()
            candidate = state["candidates"][candidate_id]
            atom = candidate["atom"]
            revision = state.get("activation_revisions", {}).get(atom, 0)
            if (candidate.get("base_version") is None or state["active"].get(atom) != candidate["base_version"]
                    or candidate.get("base_activation_revision", 0) != revision):
                raise LibraryError("Comparison requires the unchanged nonempty active version")
            base = self._version(state, atom, candidate["base_version"])
            if candidate["status"] == "promoted":
                raise LibraryError("A promoted candidate cannot be compared again")
            rows = state.get("working_memory_index", {}).get(atom, [])
            if not any(row["id"] == memory_id for row in rows):
                raise LibraryError("Comparison requires recorded host experience")
            memory = json.loads((self.root / "working_memory" / atom / (memory_id + ".json")).read_text())
            observed = memory["payload"].get("record", {})
            if memory["kind"] != "attempt" or not host_success(observed):
                raise LibraryError("Incomplete code cannot replace a published implementation")
            if source_hash(observed["code"]) != observed["sha256"] or observed["step"]["atom"] != atom:
                raise LibraryError("Source experience code/atom mismatch")
            old, new = self._hydrate(base), self._hydrate(candidate)
            if same_program(old, new):
                raise LibraryError("Equivalent code and pins do not require a new version")
            self._check_graph(state, candidate["dependencies"], atom)
            proof = dict(validation_kind="code_comparison", success=True, candidate_sha256=candidate["sha256"],
                decision="update", rationale=rationale, replay_verified=False, regression_verified=False, cases=[],
                base={k: base[k] for k in ("atom", "version", "sha256")},
                source_evidence=dict(atom=atom, memory_id=memory_id, host_success=True,
                    sha256=observed["sha256"], source_run=memory["payload"].get("source_run")),
                code_diff=code_diff(old["code"], new["code"]), recorded_at=_now())
            validate_comparison(proof, atom, candidate["sha256"])
            versions = state["versions"].setdefault(atom, {})
            version = max((int(v) for v in versions), default=0) + 1
            record = {"atom": atom, "version": version, "sha256": candidate["sha256"],
                      "dependencies": copy.deepcopy(candidate["dependencies"]),
                      "metadata": copy.deepcopy(candidate["metadata"]), "candidate_id": candidate_id,
                      "created_at": _now(), "validation": proof, "case_ids": []}
            versions[str(version)] = record
            state["active"][atom] = version
            state.setdefault("activation_revisions", {})[atom] = revision + 1
            candidate.update(status="promoted", version=version)
            candidate["validations"].append(proof)
            state["events"].append(dict(event="publish_comparison", atom=atom, version=version, at=_now()))
            self._save(state)
            return self._hydrate(record)

    def register_case(self, case: dict) -> str:
        """Compatibility alias: registration now stages; it never admits a case."""
        return self.stage_case(case)

    def get_case(self, case_id: str) -> dict:
        with self._locked():
            case = self._find_case(self._load(), case_id)
            if case is None:
                raise LibraryError(f"Unknown staged or active case: {case_id}")
            return copy.deepcopy(case)

    def pending_cases(self, atom: str) -> list[dict]:
        get_spec(atom)
        with self._locked():
            state = self._load()
            return [copy.deepcopy(case) for _, case in sorted(state["pending_cases"].items()) if case["atom"] == atom]

    def accept_case(self, case_id: str, atom: str, version: int, validation: dict) -> dict:
        """Admit one fresh-replayed case against the current active version.

        This avoids creating duplicate code versions for successful reuse. The
        supplied report must cover all existing regressions and this new case;
        admission and proof recording happen together under the state lock.
        """
        get_spec(atom)
        validation = _clone(validation)
        if not isinstance(validation, dict):
            raise LibraryError("validation must be a dictionary")
        with self._locked():
            state = self._load()
            record = self._version(state, atom, version)
            if state["active"].get(atom) != version:
                raise LibraryError("The active version changed during case validation")
            case = self._find_case(state, case_id)
            if case is None or case["atom"] != atom:
                raise LibraryError("Case is not registered for the active atom")
            if validation.get("validation_kind") != "fresh_environment_replay":
                raise LibraryError("Case admission requires fresh_environment_replay evidence")
            passed = self._validate_report_cases(state, atom, validation, record["sha256"], record["dependencies"])
            required = {identifier for identifier, value in state["cases"].items() if value["atom"] == atom} | {case_id}
            if (validation["success"] is not True or not required.issubset(passed)
                    or any(result["success"] is not True for result in validation["cases"])):
                raise LibraryError("Case admission requires successful new-case and complete regression coverage")
            self._source(record["sha256"])
            self._check_graph(state, record["dependencies"], atom)
            if case_id in state["pending_cases"]:
                state["cases"][case_id] = state["pending_cases"].pop(case_id)
                state["case_activations"][case_id] = {
                    "atom": atom, "version": version, "sha256": record["sha256"],
                    "validation_kind": "fresh_environment_replay", "activated_at": _now(),
                    "validation": validation}
                state["events"].append({"event": "accept_case", "case_id": case_id, "atom": atom,
                                        "version": version, "at": _now()})
                self._save(state)
            return copy.deepcopy(state["cases"][case_id])

    def cases(self, atom: str) -> list[dict]:
        get_spec(atom)
        with self._locked():
            state = self._load()
            return [copy.deepcopy(case) for _, case in sorted(state["cases"].items()) if case["atom"] == atom]

    def add_lesson(self, atom: str, lesson: dict | str) -> dict:
        get_spec(atom)
        lesson = {"message": lesson} if isinstance(lesson, str) else _clone(lesson)
        if not isinstance(lesson, dict):
            raise LibraryError("lesson must be a dictionary or string")
        record = {**lesson, "id": uuid.uuid4().hex, "atom": atom, "created_at": _now()}
        with self._locked():
            state = self._load()
            state["lessons"].setdefault(atom, []).append(record)
            self._save(state)
        return copy.deepcopy(record)

    def lessons(self, atom: str) -> list[dict]:
        get_spec(atom)
        with self._locked():
            return copy.deepcopy(self._load()["lessons"].get(atom, []))

    def rollback(self, atom: str, version: int) -> dict:
        get_spec(atom)
        with self._locked():
            state = self._load()
            record = self._version(state, atom, version)
            self._check_graph(state, record["dependencies"], atom)
            hydrated = self._hydrate(record)
            previous = state["active"].get(atom)
            state["active"][atom] = version
            revisions = state.setdefault("activation_revisions", {})
            revisions[atom] = revisions.get(atom, 0) + 1
            state["events"].append({"event": "rollback", "atom": atom, "from_version": previous,
                                    "version": version, "at": _now()})
            self._save(state)
            return hydrated

    def snapshot(self) -> dict:
        """Return active exact pins, suitable for run provenance and future replay."""
        with self._locked():
            state = self._load()
            pins = {atom: {"version": version, "sha256": self._version(state, atom, version)["sha256"]}
                    for atom, version in sorted(state["active"].items())}
            self._check_graph(state, pins)
            return pins
