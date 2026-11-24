"""Frozen global policy, task-local repair memory, at most three reset episodes.

The published dependency closure is copied into a fresh scratch registry for each
task. Exploration may repair code there, but no maintenance or publication runs.
Transport interruptions consume an attempt and remain unknown outcomes.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import traceback

from .catalog import CATALOG_VERSION, get_spec, validate_library_catalog
from .codecheck import require_valid_code, source_hash
from .admission import validate_comparison
from .io import read_json, write_json
from .library import Library, LibraryError, SCHEMA_VERSION, CASE_LIFECYCLE_VERSION
from .loops import ExplorationLoop
from .robocasa_adapter import HOST_PREDICATES
from .runtime_api import RuntimeAPI, API_DESCRIPTION


GROUPS = ("atomic_seen", "composite_seen", "composite_unseen")
MAX_EPISODES = 3


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FrozenLibrary:
    """Read-only reader; deliberately never constructs Library on the input."""

    def __init__(self, root, state_sha256):
        self.root = Path(root).expanduser().resolve()
        self.expected_sha256 = state_sha256
        if (not isinstance(state_sha256, str) or len(state_sha256) != 64
                or any(c not in "0123456789abcdef" for c in state_sha256)):
            raise ValueError("frozen_library_sha256 must pin the state.json bytes")
        raw = (self.root / "state.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != state_sha256:
            raise LibraryError("Frozen library state hash mismatch")
        state = json.loads(raw)
        try:
            validate_library_catalog(state)
        except ValueError as error:
            raise LibraryError(str(error)) from error
        if state.get("schema_version") != SCHEMA_VERSION:
            raise LibraryError("Freeze a current-schema library before evaluation")
        if state.get("_state_sha256") != json.loads(Library._state_text(state))["_state_sha256"]:
            raise LibraryError("Frozen library internal checksum mismatch")
        self.active, self.versions, self.sources, self.proofs = {}, {}, {}, {}

        def visit(atom, version, expected_hash=None, ancestors=()):
            get_spec(atom)
            if atom in ancestors:
                raise LibraryError("Frozen dependency cycle: " + " -> ".join(ancestors + (atom,)))
            record = Library._version(state, atom, version)
            if record.get("atom") != atom or record.get("version") != version:
                raise LibraryError("Frozen version identity differs from its registry key")
            digest = record.get("sha256")
            if expected_hash is not None and expected_hash != digest:
                raise LibraryError(f"Frozen dependency hash mismatch: {atom}@{version}")
            if str(version) in self.versions.get(atom, {}):
                return
            proof = record.get("validation", {})
            cases = proof.get("cases", [])
            compared = proof.get("validation_kind") == "code_comparison"
            if compared:
                validate_comparison(proof, atom, digest)
                if record.get("case_ids"):
                    raise LibraryError("Comparison updates cannot claim replay case coverage")
            elif (proof.get("validation_kind") != "fresh_environment_replay"
                    or proof.get("success") is not True or proof.get("candidate_sha256") != digest
                    or not isinstance(cases, list) or not cases
                    or any(not isinstance(c, dict) or c.get("success") is not True or not c.get("case_id")
                           for c in cases)
                    or not set(record.get("case_ids", [])).issubset({c["case_id"] for c in cases})):
                raise LibraryError(f"Frozen version lacks successful independent replay: {atom}@{version}")
            for case_result in cases:
                if case_result.get("replay_evidence", {}).get("kind") == "cached_success":
                    from .replay_evidence import validate_cached_result
                    validate_cached_result(self.root, state, state["cases"][case_result["case_id"]],
                                           record, proof, case_result)
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise LibraryError("Invalid frozen source SHA")
            code = (self.root / "sources" / f"{digest}.py").read_text()
            checked = require_valid_code(code, atom)
            if source_hash(code) != digest:
                raise LibraryError(f"Frozen source hash mismatch: {atom}@{version}")
            dependencies = record.get("dependencies", {})
            if not isinstance(dependencies, dict) or set(dependencies) != set(checked["dependencies"]):
                raise LibraryError("Frozen dependency declaration differs from executable api.call")
            # Resolve historical exact pins, even when a newer child is active.
            for child, pin in dependencies.items():
                if not isinstance(pin, dict) or set(pin) != {"version", "sha256"}:
                    raise LibraryError("Frozen dependencies require exact version/SHA pins")
                visit(child, pin["version"], pin["sha256"], ancestors + (atom,))
            self.sources[digest] = code
            self.proofs[(atom, str(version))] = {
                "validation_kind": "fresh_environment_replay", "success": True,
                "candidate_sha256": digest,
                "cases": [{"case_id": c["case_id"], "success": True} for c in cases]}
            if compared:
                self.proofs[(atom, str(version))] = {k: copy.deepcopy(proof[k]) for k in (
                    "validation_kind", "success", "candidate_sha256", "decision", "base",
                    "replay_verified", "regression_verified", "cases")}
                self.proofs[(atom, str(version))].update(
                    rationale="Comparison admission; transformed code was not independently replayed",
                    source_evidence={k: proof["source_evidence"][k] for k in ("atom", "memory_id", "sha256", "host_success")})
            # Compact frozen exports retain the policy/counts/origins without
            # importing training traces or depending on external cache objects.
            for key in ("replay_policy", "replay_counts"):
                if key in proof:
                    self.proofs[(atom, str(version))][key] = copy.deepcopy(proof[key])
            if "replay_policy" in proof:
                self.proofs[(atom, str(version))]["replay_origins"] = copy.deepcopy(
                    proof.get("replay_origins", {c["case_id"]: c.get("replay_evidence", {}).get("kind", "fresh")
                                                 for c in cases}))
            # Training metadata, cases, candidate history and lessons never enter scratch.
            self.versions.setdefault(atom, {})[str(version)] = {
                "atom": atom, "version": version, "sha256": digest,
                "dependencies": copy.deepcopy(dependencies), "metadata": {},
                "description": record.get("description") or record.get("metadata", {}).get("description") or get_spec(atom)["description"]}

        if not isinstance(state.get("active"), dict):
            raise LibraryError("Frozen active pins must be a dictionary")
        for atom, version in sorted(state["active"].items()):
            visit(atom, version)
            self.active[atom] = {"version": version, "sha256": self.versions[atom][str(version)]["sha256"]}
        self.verify()

    def verify(self):
        if _digest(self.root / "state.json") != self.expected_sha256:
            raise LibraryError("Frozen library changed during evaluation")
        for digest in self.sources:
            if _digest(self.root / "sources" / f"{digest}.py") != digest:
                raise LibraryError("Frozen published source changed during evaluation: " + digest)

    def scratch(self, directory):
        directory = Path(directory).resolve()
        if directory == self.root or self.root in directory.parents:
            raise ValueError("Scratch must be outside the frozen library")
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "sources").mkdir()
        for digest, code in self.sources.items():
            (directory / "sources" / f"{digest}.py").write_text(code)
        state = {"schema_version": SCHEMA_VERSION, "atomic_catalog_version": CATALOG_VERSION,
                 "case_lifecycle_version": CASE_LIFECYCLE_VERSION,
                 "active": {a: p["version"] for a, p in self.active.items()},
                 "versions": copy.deepcopy(self.versions), "candidates": {}, "cases": {},
                 "pending_cases": {}, "case_activations": {}, "lessons": {}, "events": []}
        for name in ("state.json", "state.backup.json"):
            (directory / name).write_text(Library._state_text(state))
        return TrialLibrary(directory)

    def receipt(self):
        return {"path": str(self.root), "state_sha256": self.expected_sha256,
                "active": copy.deepcopy(self.active),
                "dependency_versions": {a: {v: r["sha256"] for v, r in versions.items()}
                                        for a, versions in self.versions.items()},
                "sourcehashes": {f"sources/{digest}.py": digest for digest in self.sources},
                "training_lessons_candidates_cases_imported": False}


class TrialLibrary(Library):
    """Only proposals and task-local failure lessons may mutate this registry."""

    def promote(self, *args, **kwargs):
        raise LibraryError("Publication is disabled during policy evaluation")

    rollback = promote
    accept_case = promote
    record_validation = promote
    publish_comparison = promote
    remember = promote
    stage_case = promote
    register_case = promote


def freeze_active_library(source_library, destination):
    """Freeze verified published pins/closure; read source without locks or writes.

    The caller must stop training and finish maintenance before calling this.
    A concurrent source change fails the hash checks instead of selecting a mix.
    Only compact replay certificates survive, never training-case contents.
    """
    root = Path(source_library).expanduser().resolve()
    frozen = FrozenLibrary(root, _digest(root / "state.json"))
    destination = Path(destination).expanduser().resolve()
    if destination == root or root in destination.parents:
        raise ValueError("Frozen output must be outside its source library")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "sources").mkdir()
    versions = copy.deepcopy(frozen.versions)
    for atom, records in versions.items():
        for version, record in records.items():
            record["validation"] = copy.deepcopy(frozen.proofs[(atom, version)])
            record["case_ids"] = [c["case_id"] for c in record["validation"]["cases"]]
    state = {"schema_version": SCHEMA_VERSION, "atomic_catalog_version": CATALOG_VERSION,
                 "case_lifecycle_version": CASE_LIFECYCLE_VERSION,
             "active": {a: pin["version"] for a, pin in frozen.active.items()}, "versions": versions,
             "candidates": {}, "cases": {}, "pending_cases": {}, "case_activations": {},
             "lessons": {}, "events": []}
    for digest, code in frozen.sources.items():
        (destination / "sources" / f"{digest}.py").write_text(code)
    for name in ("state.json", "state.backup.json"):
        (destination / name).write_text(Library._state_text(state))
    frozen.verify()
    exported = FrozenLibrary(destination, _digest(destination / "state.json"))
    receipt = {**exported.receipt(), "source_library": str(root),
               "source_state_sha256": frozen.expected_sha256,
               "replay_certificate_scope": "Case IDs/verdicts only; no training examples or memos"}
    write_json(destination / "freeze_receipt.json", receipt)
    for path in destination.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    return receipt


def _without_paths_and_time(value):
    if isinstance(value, dict):
        return {k: _without_paths_and_time(v) for k, v in value.items()
                if not k.endswith("path") and k not in {"images", "time_s", "timestamp", "wall_seconds"}}
    if isinstance(value, list):
        return [_without_paths_and_time(v) for v in value]
    if isinstance(value, float):
        return round(value, 12)
    return value


def initial_scene_evidence(initial, task):
    """An observation fingerprint is evidence of reset, not hidden scene truth."""
    problems, cameras = [], initial.get("cameras", {})
    if (initial.get("task") != task["task"] or initial.get("seed") != task.get("seed", 1)
            or initial.get("step") != 0 or initial.get("terminal") is not False
            or not isinstance(initial.get("horizon"), int) or initial["horizon"] <= 0):
        problems.append("initial task/seed/step/horizon/terminal does not match")
    if task.get("horizon") is not None and initial.get("horizon") != task["horizon"]:
        problems.append("initial horizon differs from frozen task")
    if not isinstance(cameras, dict) or len(cameras) != 3:
        problems.append("initial observation requires three RGB-D cameras")
        cameras = cameras if isinstance(cameras, dict) else {}
    payload = _without_paths_and_time(initial)
    files = {}
    for camera, data in cameras.items():
        files[camera] = {}
        for key in ("image_path", "depth_path"):
            try:
                files[camera][key.removesuffix("_path") + "_sha256"] = _digest(data[key])
            except (OSError, KeyError, TypeError) as error:
                problems.append(f"missing/unreadable {camera}.{key}: {type(error).__name__}")
    payload["camera_file_hashes"] = files
    payload["task_scene_contract"] = {k: task[k] for k in
        ("task", "seed", "scene_split", "layout_ids", "style_ids", "layout_and_style_ids", "horizon", "camera_size") if k in task}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return {"valid_native_initial": not problems, "problems": problems, "fingerprint": digest,
            "scope": "Public initial observation, robot/camera calibration and RGB-D bytes; excludes paths/time, not hidden state",
            "camera_hashes": files, "task_scene_contract": payload["task_scene_contract"]}


def classify_attempt(result, initial_evidence):
    """Keep provider/init interruption separate from an ordinary native failure."""
    stats = result.get("stats", {})
    native = stats.get("native_success")
    reasons = []
    if not initial_evidence.get("valid_native_initial"):
        reasons.append("invalid_or_missing_initial")
    if initial_evidence.get("matches_first_scene") is False:
        reasons.append("initial_scene_mismatch")
    if type(native) is not bool:
        reasons.append("missing_native_final_verdict")
    if result.get("status") == "already_successful_at_reset":
        reasons.append("already_successful_at_reset")
    if result.get("stats_error"):
        reasons.append("native_stats_error")
    # A genuine native success remains a success even if generated code throws
    # after achieving it. Whole-task scoring is not a function return value.
    if not reasons and native is True:
        return {"valid_outcome": True, "success": True, "outcome": "success", "invalid_reasons": []}
    if result.get("error") or result.get("close_error") or result.get("status") == "error":
        reasons.append("interrupted_episode")
    if result.get("status") not in {"success", "step_failed", "plan_complete_task_failed"}:
        reasons.append("unrecognized_or_incomplete_outcome")
    if reasons:
        return {"valid_outcome": False, "success": None, "outcome": "invalid", "invalid_reasons": reasons}
    return {"valid_outcome": True, "success": False, "outcome": "native_failure", "invalid_reasons": []}


def trial_outcome(attempts, settled=True):
    successes = [a["episode_attempt"] for a in attempts if a.get("success") is True]
    invalid = [a["episode_attempt"] for a in attempts if not a.get("valid_outcome")]
    within = True if successes else False if len(attempts) == MAX_EPISODES and not invalid else None
    return {"first_attempt": attempts[0].get("success") if attempts else None,
            "success_within3": within, "first_success_attempt": min(successes) if successes else None,
            "invalid_attempts": invalid, "physical_attempts": len(attempts),
            "valid_outcomes": sum(bool(a.get("valid_outcome")) for a in attempts),
            "trial_outcome": "success" if within else "failure" if within is False else "unknown" if settled else "running"}


def run_policy_evaluation(config, output):
    # Lazy lookup preserves the GPU-tracked adapter wrapper installed by worker.
    from . import cli

    options = config.get("policy_evaluation", {})
    if options.get("max_episodes", MAX_EPISODES) != MAX_EPISODES:
        raise ValueError("Policy evaluation uses a total cap of three physical attempts")
    tasks = config.get("tasks") or [config["task"]]
    if not tasks or len({t["task"] for t in tasks}) != len(tasks):
        raise ValueError("Policy evaluation requires distinct task IDs")
    for task in tasks:
        if task.get("seed", 1) != 1 or task.get("scene_split") != "pretrain":
            raise ValueError("Policy evaluation requires fixed seed1 / pretrain scene per task")
        if task.get("evaluation_group") not in GROUPS:
            raise ValueError("Every task requires an evaluation_group")
    frozen_root = config["frozen_library"]
    if isinstance(frozen_root, dict):
        expected = frozen_root.get("state_sha256")
        frozen_root = frozen_root["path"]
    else:
        expected = config.get("frozen_library_sha256")
    if config.get("library") and Path(config["library"]).resolve() != Path(frozen_root).resolve():
        raise ValueError("library must identify the same read-only frozen input")
    frozen = FrozenLibrary(frozen_root, expected)
    output = Path(output).resolve()
    if output == frozen.root or frozen.root in output.parents:
        raise ValueError("Evaluation output must be outside the frozen library")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    write_json(output / "frozen_library.json", frozen.receipt())
    report = {"mode": "evaluate_policy", "status": "running", "task_status": "running",
              "maintenance_status": "disabled", "maintenance_trigger": "disabled",
              "promoted_versions": 0, "accepted_case_count": 0, "maintenance_runs": [],
              "episodes": [], "total_tasks": len(tasks), "total_episode_attempts": 0,
              "library_before": copy.deepcopy(frozen.active), "frozen_library": frozen.receipt(),
              "protocol": {"max_physical_attempts_per_task": MAX_EPISODES,
                           "invalid_attempts_consume_budget": True, "scene_policy": "fixed seed1/pretrain with observable reset fingerprint",
                           "conversation_policy": "new Agent per episode; bounded task-local history",
                           "library_policy": "frozen published dependency closure; isolated trial scratch; no maintenance/publication",
                           "budget_claim": "Config and native benchmark horizon; no claim of exact original RSI budget"}}

    def persist():
        report["total_episode_attempts"] = sum(len(t["episode_attempts"]) for t in report["episodes"])
        report["native_successes"] = sum(t.get("success_within3") is True for t in report["episodes"])
        report["program_successes"] = sum(any(a.get("status") == "success" for a in t["episode_attempts"])
                                          for t in report["episodes"])
        write_json(output / "summary.json", report)

    try:
        for index, task in enumerate(tasks):
            episode = output / f"{index:03d}_{task['task']}_seed1"
            scratch = frozen.scratch(episode / "scratch_library")
            record = {"task": copy.deepcopy(task), "exploration": None, "episode_attempts": [],
                      "maintenance": None, "maintenance_runs": [], "maintenance_status": "disabled",
                      "retry_status": "running", "scratch_library": str(scratch.root),
                      "initial_scenes": [], **trial_outcome([], settled=False)}
            report["episodes"].append(record)
            history, first_fingerprint = [], None
            for number in range(1, MAX_EPISODES + 1):
                frozen.verify()
                if scratch.snapshot() != frozen.active:
                    raise LibraryError("Scratch active pins changed during evaluation")
                run = episode / ("exploration" if number == 1 else f"retries/episode_{number:04d}/exploration")
                report.update(current_task_index=index, current_episode_attempt=number)
                record["retry_status"] = "running"
                persist()
                scene = {}

                def factory(task_config, directory):
                    nonlocal first_fingerprint
                    adapter = cli.adapter_factory(task_config, directory)
                    try:
                        initial = adapter.observe()
                        write_json(Path(directory).parent / "initial.json", initial)
                        scene.update(initial_scene_evidence(initial, task))
                        scene["matches_first_scene"] = (scene["fingerprint"] == first_fingerprint
                                                        if first_fingerprint is not None else None)
                        if scene["valid_native_initial"] and first_fingerprint is None:
                            first_fingerprint = scene["fingerprint"]
                        write_json(Path(directory).parent / "scene_fingerprint.json", scene)
                        if not scene["valid_native_initial"]:
                            raise ValueError("Invalid native initial observation")
                        if scene["matches_first_scene"] is False:
                            raise ValueError("Reset observation differs from first task scene")
                        return adapter
                    except Exception:
                        adapter.close()
                        raise

                try:
                    agent = cli.create_agent(config["agent"], episode / "model" / f"episode_{number:04d}")
                    exploration_options = {**cli._loop_options(config), "record_working_memory": False}
                    loop = ExplorationLoop(scratch, agent, factory, RuntimeAPI, API_DESCRIPTION,
                                           HOST_PREDICATES, **exploration_options)
                    result = loop.run(copy.deepcopy(task), run, retry_context={
                        "episode_attempt": number, "reset_to_initial_state": True,
                        "previous_episodes": copy.deepcopy(history[-3:])})
                except Exception as error:
                    result = {"status": "error", "task_config": copy.deepcopy(task),
                              "native_success": False, "completed_steps": [], "failures": [], "pending": [],
                              "error": {"type": type(error).__name__, "message": str(error),
                                        "traceback": traceback.format_exc()}}
                    write_json(run / "task.json", task)
                    write_json(run / "summary.json", result)
                evidence = scene or {"valid_native_initial": False, "problems": ["initial observation not created"]}
                verdict = classify_attempt(result, evidence)
                attempt = {**cli._attempt_summary(result, run, number), **verdict,
                           "initial_scene": copy.deepcopy(evidence), "error": copy.deepcopy(result.get("error")),
                           "close_error": copy.deepcopy(result.get("close_error")),
                           "native_success_observed": result.get("stats", {}).get("native_success"),
                           "reported_native_success": result.get("native_success"),
                           "native_success": result.get("stats", {}).get("native_success"),
                           "program_status": result.get("status")}
                record["episode_attempts"].append(attempt)
                record["initial_scenes"].append(copy.deepcopy(evidence))
                record["exploration"] = result
                record.update(trial_outcome(record["episode_attempts"], settled=number == MAX_EPISODES or verdict["success"] is True))
                frozen.verify()
                if scratch.snapshot() != frozen.active:
                    raise LibraryError("Scratch active pins changed during evaluation")
                if verdict["success"] is True:
                    record["retry_status"] = "success"
                    persist()
                    break
                record["retry_status"] = "retrying" if number < MAX_EPISODES else "budget_exhausted"
                history.append(cli._episode_history(run, result, number))
                persist()
            record["task_status"] = record["trial_outcome"]
            persist()
        report["status"] = "complete"
        report["task_status"] = "complete" if all(t["success_within3"] is True for t in report["episodes"]) else "tasks_incomplete"
        report["evaluation_status"] = ("complete_with_unknown_trials" if any(t["success_within3"] is None for t in report["episodes"])
                                       else "complete")
        report["library_after"] = copy.deepcopy(frozen.active)
        frozen.verify()
        report["library_unchanged"] = True
    except Exception as error:
        report.update(status="error", error={"type": type(error).__name__, "message": str(error),
                                              "traceback": traceback.format_exc()})
        report["protocol_integrity_error"] = isinstance(error, LibraryError)
        try:
            frozen.verify()
            report["library_unchanged"] = True
        except (LibraryError, OSError):
            report["library_unchanged"] = False
    finally:
        persist()
    return report
