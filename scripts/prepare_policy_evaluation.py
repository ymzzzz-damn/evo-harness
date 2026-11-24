#!/usr/bin/env python3
"""Audit a drained training queue and prepare, but never submit, frozen evaluation."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import re
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_RUNTIME = "/mnt/auwomo-data/yumuzhen/physical-rsi"
TRAIN_TASKS = 300
TERMINAL_WORKERS = {"succeeded", "failed", "interrupted", "cancelled"}
TERMINAL_CAMPAIGNS = {"complete", "tasks_incomplete", "validation_failed", "error"}
TERMINAL_EPISODES = {"success", "step_failed", "plan_complete_task_failed", "error"}


class HandoffNotReady(ValueError):
    def __init__(self, message, audit=None):
        super().__init__(message)
        self.audit = audit or {}


def _read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise HandoffNotReady(f"Expected an object: {path}")
    return value


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    return path == root or root in path.parents


def verify_source_snapshot(training_run):
    """Check every frozen source byte, allowing only generated Python bytecode."""
    run = Path(training_run).resolve()
    source = run / "source"
    expected = _read(run / "source_sha256.json")
    actual = {}
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise HandoffNotReady(f"Frozen source is a symlink: {relative}")
        if path.is_file():
            actual[str(relative)] = _sha(path)
    if not expected or actual != expected:
        raise HandoffNotReady("Training source snapshot differs from source_sha256.json")
    return {"root": str(source), "files": len(actual),
            "manifest_sha256": _sha(run / "source_sha256.json"), "all_files_match": True}


def _rgbd_evidence(initial, task, exploration):
    if (initial.get("task") != task["task"] or initial.get("seed") != task.get("seed", 0)
            or initial.get("step") != 0 or initial.get("terminal") is not False
            or type(initial.get("horizon")) is not int or initial["horizon"] <= 0):
        raise HandoffNotReady("Initial task/seed/step/terminal/horizon is invalid")
    if task.get("horizon") is not None and initial["horizon"] != task["horizon"]:
        raise HandoffNotReady("Initial horizon differs from the frozen task")
    cameras = initial.get("cameras")
    if not isinstance(cameras, dict) or len(cameras) != 3:
        raise HandoffNotReady("Initial observation needs three RGB-D cameras")
    if task.get("camera_size") != [512, 512]:
        raise HandoffNotReady("Training requires 512 by 512 RGB-D")
    hashes = {}
    paths = set()
    for name, camera in cameras.items():
        if not isinstance(camera, dict) or (camera.get("width"), camera.get("height")) != (512, 512):
            raise HandoffNotReady(f"Invalid initial camera dimensions: {name}")
        for kind in ("image_path", "depth_path"):
            path = Path(camera.get(kind, "")).resolve()
            if not _inside(path, exploration) or not path.is_file() or path in paths:
                raise HandoffNotReady(f"Missing, external or duplicate RGB-D artifact: {name}.{kind}")
            paths.add(path)
            with path.open("rb") as stream:
                if kind == "image_path":
                    header = stream.read(24)
                    if (len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n"
                            or header[12:16] != b"IHDR" or struct.unpack(">II", header[16:24]) != (512, 512)):
                        raise HandoffNotReady(f"Invalid 512 RGB PNG: {path}")
                else:
                    magic = stream.read(8)
                    if magic[:6] != b"\x93NUMPY" or len(magic) != 8 or magic[6] not in (1, 2, 3):
                        raise HandoffNotReady(f"Invalid depth NPY: {path}")
                    size_format = "<H" if magic[6] == 1 else "<I"
                    size = struct.unpack(size_format, stream.read(struct.calcsize(size_format)))[0]
                    if size > 65536:
                        raise HandoffNotReady(f"Oversized depth NPY header: {path}")
                    meta = ast.literal_eval(stream.read(size).decode("latin1"))
                    dtype = meta.get("descr", "")
                    if meta.get("shape") != (512, 512) or dtype not in ("<f4", ">f4", "<f8", ">f8", "=f4", "=f8"):
                        raise HandoffNotReady(f"Invalid depth dimensions/type: {path}")
                    if path.stat().st_size != stream.tell() + 512 * 512 * int(dtype[-1]):
                        raise HandoffNotReady(f"Truncated or extra depth data: {path}")
            hashes[str(path)] = _sha(path)
    return hashes


def audit_training(training_run):
    """Do not infer native episodes or native successes from worker exit codes."""
    run = Path(training_run).resolve()
    plan, config, receipt = (_read(run / name) for name in
                              ("parallel_plan.json", "config.json", "parallel_receipt.json"))
    workers = plan.get("workers", [])
    if (len(workers) != TRAIN_TASKS or len({w["task"]["task"] for w in workers}) != TRAIN_TASKS
            or len({w["index"] for w in workers}) != TRAIN_TASKS
            or config.get("tasks") != [w["task"] for w in workers]
            or plan.get("mode", "run") != "run" or config.get("mode", "run") != "run"
            or Path(plan.get("run", "")).resolve() != run
            or Path(plan.get("source", "")).resolve() != run / "source"):
        raise HandoffNotReady("Training plan must contain exactly 300 unique frozen tasks")
    counts = receipt.get("counts", {})
    rows = receipt.get("workers", [])
    if (counts.get("pending") != 0 or counts.get("running") != 0
            or counts.get("planned") != TRAIN_TASKS or counts.get("finished") != TRAIN_TASKS
            or len(rows) != TRAIN_TASKS or len({r.get("index") for r in rows}) != TRAIN_TASKS
            or receipt.get("status") not in {"complete", "complete_with_failures", "workers_failed"}
            or any(r.get("status") not in TERMINAL_WORKERS for r in rows)):
        raise HandoffNotReady("Training queue has pending/running or nonterminal workers")
    by_index = {r["index"]: r for r in rows}
    classifications, evidence, problems = [], {}, []
    totals = Counter(planned=TRAIN_TASKS)
    for name in ("parallel_plan.json", "config.json", "parallel_receipt.json", "source_sha256.json"):
        evidence[str(run / name)] = _sha(run / name)
    for worker in workers:
        row = {"index": worker["index"], "task": worker["task"], "worker_status": by_index[worker["index"]].get("status")}
        try:
            directory = Path(worker["run"]).resolve()
            if directory == run or not _inside(directory, run / "workers"):
                raise HandoffNotReady("Worker directory is outside the frozen training run")
            ledger = _read(directory / "task_receipt.json")
            saved = by_index[worker["index"]]
            for key in ("index", "task", "run"):
                if saved.get(key) != worker.get(key) or ledger.get(key) != worker.get(key):
                    raise HandoffNotReady("Worker receipt identity differs from its frozen plan")
            if ledger.get("status") not in TERMINAL_WORKERS or ledger["status"] != saved["status"]:
                raise HandoffNotReady("Worker task receipt is not terminal/consistent")
            campaign = _read(directory / "campaign/summary.json")
            if campaign.get("status") not in TERMINAL_CAMPAIGNS or campaign.get("maintenance_status") == "running":
                raise HandoffNotReady("Campaign or maintenance is not terminal")
            episodes = campaign.get("episodes", [])
            if len(episodes) != 1 or episodes[0].get("task") != worker["task"]:
                raise HandoffNotReady("Campaign task differs from the frozen worker task")
            if len(episodes[0].get("episode_attempts", [])) != 1:
                raise HandoffNotReady("Training must retain exactly one exploration attempt")
            if any(r.get("status") == "running" for r in campaign.get("maintenance_runs", [])):
                raise HandoffNotReady("Maintenance ledger still has a running attempt")
            explorations = sorted(set((directory / "campaign").glob("*/exploration"))
                                  | set((directory / "campaign").glob("*/retries/*/exploration")))
            if len(explorations) != 1:
                raise HandoffNotReady("Training must have exactly one exploration directory")
            exploration = explorations[0]
            if not (exploration / "initial.json").is_file():
                raise HandoffNotReady("Missing initial observation; root must selectively resolve this task")
            initial, final = (_read(exploration / name) for name in ("initial.json", "summary.json"))
            if _read(exploration / "task.json") != worker["task"] or final.get("task_config") != worker["task"]:
                raise HandoffNotReady("Exploration task/summary differs from the frozen task")
            rgbd = _rgbd_evidence(initial, worker["task"], exploration)
            if final.get("status") not in TERMINAL_EPISODES:
                raise HandoffNotReady("Exploration lacks a final native episode summary")
            reported_native = final.get("native_success")
            observed_native = final.get("stats", {}).get("native_success")
            native = observed_native if type(observed_native) is bool else None
            interrupted = final["status"] == "error" or ledger["status"] in {"interrupted", "cancelled"}
            if interrupted:
                outcome = "interrupted_after_initial"
            elif native is None:
                raise HandoffNotReady("Final summary lacks a boolean native verdict in host stats")
            elif final["status"] == "success" and native:
                outcome = "native_success"
            elif not native:
                outcome = "native_failure"
            else:
                outcome = "native_success_program_incomplete"
            row.update(outcome=outcome, initial=str(exploration / "initial.json"),
                       final_summary=str(exploration / "summary.json"),
                       exploration_status=final["status"], program_status=final["status"], native_success=native,
                       observed_native_success=native, reported_native_success=reported_native,
                       error=final.get("error"))
            totals["valid_initial_episodes"] += 1
            totals[outcome] += 1
            totals["observed_native_success" if native is True else
                   "observed_native_failure" if native is False else "unknown_native_verdict"] += 1
            for path in (directory / "task_receipt.json", directory / "campaign/summary.json",
                         exploration / "task.json", exploration / "initial.json", exploration / "summary.json"):
                evidence[str(path)] = _sha(path)
            evidence.update(rgbd)
        except (OSError, ValueError, KeyError, TypeError, struct.error) as error:
            row.update(outcome="blocked", reason=str(error))
            totals["blocked_tasks"] += 1
            problems.append({"index": row["index"], "task": worker["task"]["task"], "reason": str(error)})
        classifications.append(row)
    audit = {"counts": dict(totals), "tasks": classifications, "problems": problems,
             "evidence_sha256": evidence, "exit_code_is_episode_evidence": False}
    if problems:
        raise HandoffNotReady("Training evidence is incomplete or invalid; no automatic evaluation handoff", audit)
    return audit


@contextmanager
def _source_paths(source):
    before = sys.path[:]
    sys.path[:0] = [str(source), str(source / "scripts")]
    try:
        yield
    finally:
        sys.path[:] = before


def _frozen_policy(source):
    """An isolated package name prevents use of a newer cached canonical evo."""
    name = "_evo_handoff_" + hashlib.sha256(str(source).encode()).hexdigest()[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, source / "evo/__init__.py",
                                                     submodule_search_locations=[str(source / "evo")])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return importlib.import_module(name + ".policy_evaluation")


def _frozen_launcher(source):
    spec = importlib.util.spec_from_file_location("_eval_launcher_" + hashlib.sha256(str(source).encode()).hexdigest()[:16],
                                                 source / "scripts/launch_gpu_parallel_p6000.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_frozen_memory(path, receipt):
    if Path(receipt.get("path", "")).resolve() != path or _sha(path / "state.json") != receipt.get("state_sha256"):
        raise HandoffNotReady("Frozen library receipt path/state hash differs")
    state = _read(path / "state.json")
    allowed_state = {"schema_version", "atomic_catalog_version", "case_lifecycle_version", "active", "versions", "candidates", "cases",
                     "pending_cases", "case_activations", "lessons", "events", "_state_sha256"}
    if set(state) - allowed_state:
        raise HandoffNotReady("Unexpected training context in frozen state")
    for key in ("candidates", "cases", "pending_cases", "case_activations", "lessons", "events"):
        if state.get(key) != ([] if key == "events" else {}):
            raise HandoffNotReady(f"Training memory leaked into evaluation snapshot: {key}")
    if (path / "state.backup.json").read_bytes() != (path / "state.json").read_bytes():
        raise HandoffNotReady("Frozen state backup differs from the pinned state")
    hashes = set()
    records = set()
    for atom, versions in state.get("versions", {}).items():
        for version, item in versions.items():
            records.add((atom, str(version)))
            if (set(item) - {"atom", "version", "sha256", "dependencies", "metadata", "description", "validation", "case_ids"}
                    or item.get("metadata") != {}):
                raise HandoffNotReady("Training metadata leaked into a frozen version")
            if "description" in item and (not isinstance(item["description"], str) or not item["description"].strip()):
                raise HandoffNotReady("Invalid frozen function description")
            proof = item.get("validation", {})
            compared = proof.get("validation_kind") == "code_comparison"
            allowed_proof = {"validation_kind", "success", "candidate_sha256", "cases",
                             "replay_policy", "replay_counts", "replay_origins"}
            if compared:
                allowed_proof.update({"decision", "base", "replay_verified", "regression_verified",
                                      "rationale", "source_evidence"})
            if set(proof) - allowed_proof:
                raise HandoffNotReady("Training context leaked into a replay certificate")
            if any(set(c) != {"case_id", "success"} or c["success"] is not True for c in proof.get("cases", [])):
                raise HandoffNotReady("Frozen replay certificate contains training examples")
            digest = item["sha256"]
            if compared:
                # Use the validator shipped with this exact handoff source; a
                # comparison certificate must never be relabeled as a replay.
                try:
                    _frozen_policy(Path(__file__).resolve().parents[1]).validate_comparison(proof, atom, digest)
                except ValueError as error:
                    raise HandoffNotReady("Invalid comparison certificate: " + str(error)) from error
                if item.get("case_ids") or set(proof["source_evidence"]) != {"atom", "memory_id", "sha256", "host_success"}:
                    raise HandoffNotReady("Training context or replay claims leaked into a comparison certificate")
            elif (proof.get("success") is not True or proof.get("validation_kind") != "fresh_environment_replay"
                    or proof.get("candidate_sha256") != digest or not proof.get("cases")):
                raise HandoffNotReady("Frozen version has no independent successful replay certificate")
            if _sha(path / "sources" / (digest + ".py")) != digest:
                raise HandoffNotReady("Frozen source hash mismatch")
            hashes.add(digest)
    needed = set()

    def visit(atom, version, expected=None, ancestors=()):
        key = (atom, str(version))
        if atom in ancestors or key not in records:
            raise HandoffNotReady("Invalid frozen dependency closure")
        record = state["versions"][atom][str(version)]
        if expected is not None and record["sha256"] != expected:
            raise HandoffNotReady("Frozen dependency source SHA differs from its exact pin")
        if key in needed:
            return
        needed.add(key)
        for child, pin in record.get("dependencies", {}).items():
            if not isinstance(pin, dict) or set(pin) != {"version", "sha256"}:
                raise HandoffNotReady("Frozen dependency is not an exact version/SHA pin")
            visit(child, pin["version"], pin["sha256"], ancestors + (atom,))

    for atom, version in state.get("active", {}).items():
        visit(atom, version)
    if needed != records:
        raise HandoffNotReady("Frozen snapshot contains versions outside the active dependency closure")
    actual = set()
    for file in path.rglob("*"):
        if not file.is_file():
            continue
        rel = file.relative_to(path)
        if len(rel.parts) == 2 and rel.parts[0] == "sources" and file.suffix == ".py":
            actual.add(file.stem)
        elif str(rel) not in {"state.json", "state.backup.json", "freeze_receipt.json"}:
            raise HandoffNotReady(f"Unexpected file in frozen evaluation library: {rel}")
    if actual != hashes:
        raise HandoffNotReady("Frozen snapshot has extra or missing source files")
    if receipt.get("training_lessons_candidates_cases_imported") is not False:
        raise HandoffNotReady("Frozen library lacks a clean-memory receipt")
    return {"training_memory_empty": True, "source_files": len(actual),
            "state_sha256": receipt["state_sha256"]}


def prepare_evaluation(training_run, template, name, project=None, runtime=DEFAULT_RUNTIME):
    run, template = Path(training_run).expanduser().resolve(), Path(template).expanduser().resolve()
    project = Path(project).expanduser().resolve() if project is not None else run.parent.parent
    runtime = Path(runtime).expanduser().resolve()
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,44}[a-z0-9])?", name):
        raise HandoffNotReady("Invalid evaluation run name")
    config = _read(run / "config.json")
    expected = config.get("experiment_metadata", {}).get("evaluation_template_sha256")
    if not expected or _sha(template) != expected:
        raise HandoffNotReady("Evaluation template SHA differs from the training contract")
    source_audit = verify_source_snapshot(run)
    eval_run, frozen = project / "runs" / name, run / "evaluation_frozen_library"
    destination = run / "evaluation_preparation_receipt.json"
    identity = {"training_run": str(run), "template": str(template), "template_sha256": expected,
                "name": name, "project": str(project), "runtime": str(runtime),
                "training_config_sha256": _sha(run / "config.json"),
                "training_source_manifest_sha256": source_audit["manifest_sha256"]}
    if destination.exists():
        previous = _read(destination)
        if previous.get("status") != "prepared" or previous.get("identity") != identity:
            raise HandoffNotReady("Existing preparation receipt does not match this handoff")
        for path, digest in previous.get("artifact_sha256", {}).items():
            if _sha(path) != digest:
                raise HandoffNotReady("Prepared evaluation artifact changed: " + path)
        for path, digest in previous.get("training_audit", {}).get("evidence_sha256", {}).items():
            if _sha(path) != digest:
                raise HandoffNotReady("Training evidence changed after handoff: " + path)
        verify_source_snapshot(eval_run)
        if _read(eval_run / "source_sha256.json") != _read(run / "source_sha256.json"):
            raise HandoffNotReady("Prepared evaluation sources differ from the training framework")
        _check_frozen_memory(frozen, previous["frozen_library"])
        return previous
    if eval_run.exists() or frozen.exists() or (run / "evaluation_config.json").exists() or (run / "handoff_ready.json").exists():
        raise HandoffNotReady("Refuse to overwrite an existing evaluation run/frozen directory/config")
    audit = audit_training(run)
    evaluation = _read(template)
    tasks = evaluation.get("tasks", [])
    if len(tasks) != 50 or len({t["task"] for t in tasks}) != 50:
        raise HandoffNotReady("Evaluation template must contain 50 unique tasks")
    if evaluation.get("retry_experiences"):
        raise HandoffNotReady("Evaluation template cannot import training retry experiences")
    source_library = Path(config["library"]).resolve()
    if Path(_read(run / "parallel_plan.json")["shared_library"]).resolve() != source_library:
        raise HandoffNotReady("Training library differs between config and frozen plan")
    with _source_paths(run / "source"):
        freezer = _frozen_policy(run / "source")
        frozen_receipt = freezer.freeze_active_library(source_library, frozen)
        clean = _check_frozen_memory(frozen, frozen_receipt)
        verify_source_snapshot(run)
        evaluation.update(mode="evaluate_policy", library=str(frozen), frozen_library=str(frozen),
                          frozen_library_sha256=frozen_receipt["state_sha256"])
        evaluation.pop("evaluation_handoff", None)
        evaluation.pop("retry_experiences", None)
        evaluation_path = run / "evaluation_config.json"
        _write(evaluation_path, evaluation)
        launcher = _frozen_launcher(run / "source")
        plan, manifest = launcher.prepare(evaluation_path, name, project, runtime, source_root=run / "source",
            cpu_request="64", cpu_limit="128", memory_request="256Gi", memory_limit="384Gi",
            max_parallel=64, preflight=False)
    if (Path(plan["run"]).resolve() != eval_run or Path(plan["source_origin"]).resolve() != run / "source"
            or plan.get("mode") != "evaluate_policy"):
        raise HandoffNotReady("Prepared evaluation plan is not pinned to training sources")
    if _read(eval_run / "source_sha256.json") != _read(run / "source_sha256.json"):
        raise HandoffNotReady("Evaluation framework snapshot differs from the training framework")
    verify_source_snapshot(eval_run)
    verify_source_snapshot(run)
    artifacts = [evaluation_path, eval_run / "config.json", eval_run / "parallel_plan.json", Path(manifest),
                 eval_run / "source_sha256.json", frozen / "state.json", frozen / "state.backup.json", frozen / "freeze_receipt.json"]
    artifacts.extend(sorted((frozen / "sources").glob("*.py")))
    receipt = {"status": "prepared", "identity": identity, "job_created": False,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(), "training_source": source_audit,
        "training_audit": audit, "frozen_library": frozen_receipt, "frozen_memory_audit": clean,
        "evaluation_run": str(eval_run), "job": plan["job"], "plan": str(eval_run / "parallel_plan.json"),
        "manifest": str(manifest), "evaluation_config": str(evaluation_path),
        "artifact_sha256": {str(p): _sha(p) for p in artifacts}}
    _write(destination, receipt)
    ready = {"status": "ready_for_submission", "job_created": False,
        "preparation_receipt": str(destination), "preparation_receipt_sha256": _sha(destination),
        "training_counts": audit["counts"], "template_sha256": expected,
        "training_source_manifest_sha256": source_audit["manifest_sha256"],
        "frozen_library_sha256": frozen_receipt["state_sha256"],
        "evaluation_config_sha256": _sha(evaluation_path), "job": plan["job"],
        "run": str(eval_run), "plan": receipt["plan"], "manifest": str(manifest)}
    _write(run / "handoff_ready.json", ready)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--project")
    parser.add_argument("--runtime", default=DEFAULT_RUNTIME)
    args = parser.parse_args(argv)
    try:
        receipt = prepare_evaluation(args.training_run, args.template, args.name, args.project, args.runtime)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "handoff_blocked", "error": str(error),
                          "training_counts": getattr(error, "audit", {}).get("counts")}), flush=True)
        return 2
    print(json.dumps({"status": receipt["status"], "job_created": False, "job": receipt["job"],
                      "run": receipt["evaluation_run"], "training_counts": receipt["training_audit"]["counts"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
