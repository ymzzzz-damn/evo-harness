#!/usr/bin/env python3
"""Summarize one finite task queue and export its active, validated code tools.

Run on demand or once after the queue drains, never on the supervisor's poll loop.
Task success requires the host's native verdict and a successful program. Worker
exit codes and exploration invocations are separate from native episode evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evo.catalog import get_spec
from evo.codecheck import source_hash
from evo.library import Library
from evo.replay_evidence import validate_cached_result
from evo.admission import validate_comparison


TERMINAL_WORKERS = {"succeeded", "failed", "interrupted", "cancelled"}
TERMINAL_CAMPAIGNS = {"complete", "tasks_incomplete", "validation_failed", "error"}
TERMINAL_EPISODES = {"success", "step_failed", "plan_complete_task_failed", "error"}


def _read(path, issues):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError) as error:
        issues.append({"path": str(path), "error": str(error)})
        return None


def _inside(path, roots):
    if not isinstance(path, str) or not path:
        return False
    path = Path(path).expanduser().resolve()
    return any(path == root or root in path.parents for root in roots)


def _maintenance_summary(campaign_root, campaign, task_record, issues):
    """Prefer the immediate-maintenance ledger, which may outlive an interrupted episode."""
    runs = task_record.get("maintenance_runs", campaign.get("maintenance_runs", []))
    ledger = campaign_root / "maintenance_runs.json"
    if ledger.is_file():
        try:
            saved = json.loads(ledger.read_text(encoding="utf-8"))
            if not isinstance(saved, list) or any(not isinstance(item, dict) for item in saved):
                raise ValueError("maintenance ledger must be a list of objects")
            runs = saved
        except (OSError, ValueError) as error:
            issues.append({"path": str(ledger), "error": str(error)})
    runs = runs if isinstance(runs, list) else []
    previous = task_record.get("maintenance") or {}
    promoted, accepted, rejected = [], [], []
    completed, dispatched = set(), set()
    unique_runs = {}
    for run in runs:
        key = run.get("output") or json.dumps([run.get("episode_attempt"), run.get("experience_paths", [])])
        unique_runs[key] = run
    runs = list(unique_runs.values())
    for run in runs:
        outcome = run.get("result") or {}
        promoted.extend(outcome.get("promoted", []))
        accepted.extend(outcome.get("accepted_cases", []))
        rejected.extend(outcome.get("rejected", []))
        paths = {str(Path(path).expanduser().resolve()) for path in run.get("experience_paths", [])}
        dispatched.update(paths)
        if run.get("status") == "complete" and not outcome.get("rejected"):
            completed.update(paths)
    if runs:
        status = ("running" if any(run.get("status") == "running" for run in runs) else
                  "failed" if rejected or any(run.get("status") != "complete" for run in runs) else "complete")
    else:
        status = task_record.get("maintenance_status", campaign.get("maintenance_status", previous.get("status")))
        promoted, accepted, rejected = (previous.get("promoted", []), previous.get("accepted_cases", []),
                                         previous.get("rejected", []))
    return {"status": status, "runs": runs, "promoted": promoted, "accepted_cases": accepted,
            "rejected_count": len(rejected), "dispatched": dispatched, "completed": completed}


def _task_summary(worker, receipt, issues):
    root = Path(worker["run"]).resolve()
    campaign_root = root / "campaign"
    campaign = _read(campaign_root / "summary.json", issues) or {}
    episodes = campaign.get("episodes", [])
    matching = [episode for episode in episodes if episode.get("task") == worker["task"]]
    task_record = matching[0] if len(matching) == 1 else {}
    if episodes and len(matching) != 1:
        issues.append({"path": str(campaign_root / "summary.json"), "error": "planned task does not match one campaign task"})
    directories = set(campaign_root.glob("*/exploration")) | set(campaign_root.glob("*/retries/*/exploration"))
    attempts = task_record.get("episode_attempts", [])
    for attempt in attempts:
        if attempt.get("summary_path"):
            directory = Path(attempt["summary_path"]).resolve().parent
            if root not in directory.parents:
                issues.append({"path": str(directory), "error": "episode summary outside planned worker root"})
            else:
                directories.add(directory)
    directories = sorted(directories, key=lambda path: ("retries" in path.parts, str(path)))
    result = task_record.get("exploration") or {}
    result_path = None
    native_episodes = 0
    for directory in directories:
        current = _read(directory / "summary.json", issues)
        if current is None:
            current = _read(directory / "progress.json", issues)
        task_config = current.get("task_config") if current else _read(directory / "task.json", issues)
        if task_config != worker["task"]:
            issues.append({"path": str(directory), "error": "exploration task differs from planned task"})
            continue
        initial = _read(directory / "initial.json", issues)
        if initial is not None:
            if (("task" in initial and initial["task"] != worker["task"]["task"])
                    or ("seed" in initial and initial["seed"] != worker["task"].get("seed", 0))):
                issues.append({"path": str(directory / "initial.json"), "error": "native task/seed differs from planned task"})
            else:
                native_episodes += 1
        if current is not None:
            result, result_path = current, str(directory / "summary.json")
    invocations = max(len(attempts), len(directories), int(bool(result)))
    worker_status = receipt.get("status", "pending")
    worker_attempted = bool(invocations or receipt.get("pid") or (campaign_root / "config.json").is_file()
                     or worker_status in {"starting", "running", "succeeded", "failed", "interrupted"})
    settled = worker_status in TERMINAL_WORKERS or campaign.get("status") in TERMINAL_CAMPAIGNS
    native_success = result.get("native_success") is True
    program_success = result.get("status") == "success"
    episode_outcome_recorded = result.get("status") in TERMINAL_EPISODES
    native_episode_attempted = native_episodes == 1 and episode_outcome_recorded and invocations == 1
    if native_success and program_success and not native_episodes:
        issues.append({"path": str(root), "error": "reported success lacks a native initial-observation artifact"})
    if native_success and program_success and native_episode_attempted:
        outcome = "success"
    elif not settled:
        outcome = "running" if worker_attempted else "pending"
    else:
        outcome = "failed" if native_episode_attempted else "no_episode"
    completed = result.get("completed_steps", [])
    maintenance = _maintenance_summary(campaign_root, campaign, task_record, issues)
    pending = {str(Path(path).expanduser().resolve()) for path in result.get("pending", [])}
    return {"index": worker["index"], "task": copy.deepcopy(worker["task"]), "run": str(root),
            "worker_status": worker_status, "returncode": receipt.get("returncode"),
            "attempted": native_episode_attempted, "worker_attempted": worker_attempted,
            "settled": settled, "outcome": outcome, "episode_outcome_recorded": episode_outcome_recorded,
            "exploration_invocations": invocations, "native_episodes_confirmed": native_episodes,
            "native_episode_evidence": "exploration/initial.json after reset and initial observation",
            "episode_limit_violated": invocations > 1, "exploration_status": result.get("status"),
            "reported_native_success": native_success, "program_success": program_success,
            "initially_satisfied_reset": result.get("status") == "already_successful_at_reset",
            "summary_path": result_path, "stop_reason": copy.deepcopy(result.get("stop_reason")),
            "error": copy.deepcopy(result.get("error")), "stats": copy.deepcopy(result.get("stats", {})),
            "executed_successful_atomic_steps": sum(item.get("success") is True and not item.get("skipped") for item in completed),
            "task_status": task_record.get("task_status", campaign.get("task_status")),
            "pending_experiences_total": len(pending),
            "pending_unpublished_experiences": len(pending - maintenance["completed"]),
            "maintained_experiences": len(maintenance["dispatched"]),
            "maintenance_status": maintenance["status"], "maintenance_runs_count": len(maintenance["runs"]),
            "maintenance_promoted": copy.deepcopy(maintenance["promoted"]),
            "maintenance_accepted_cases": copy.deepcopy(maintenance["accepted_cases"]),
            "maintenance_rejected_count": maintenance["rejected_count"]}


def _version_keys(state):
    return {(atom, int(version)): record["sha256"] for atom, versions in state.get("versions", {}).items()
            for version, record in versions.items()}


def _tool_report(library_root, baseline_path, worker_roots, issues):
    report = {"library": str(library_root) if library_root else None,
              "baseline_state": str(baseline_path) if baseline_path else None,
              "available": False, "published_versions": [], "active_validated_tools": [],
              "baseline_versions": [], "new_versions": [], "campaign_new_versions": [],
              "exported_code": [], "case_activations": []}
    baseline = None
    if baseline_path:
        try:
            baseline = Library._read_state_file(Path(baseline_path))
        except (OSError, ValueError) as error:
            issues.append({"path": str(baseline_path), "error": str(error)})
    report["baseline_available"] = baseline is not None
    baseline_versions = _version_keys(baseline or {})
    report["baseline_counts"] = {"published_versions": len(baseline_versions),
                                  "active_atoms": len((baseline or {}).get("active", {}))}
    if not library_root:
        return report
    library_root = Path(library_root).resolve()
    try:
        # Read-only: do not construct Library, initialize files, or recover state.
        with (library_root / ".lock").open("r") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
            try:
                state = Library._read_state_file(library_root / "state.json")
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    except (OSError, ValueError) as error:
        issues.append({"path": str(library_root / "state.json"), "error": str(error)})
        return report
    report["available"] = True
    report["case_lifecycle_version"] = state.get("case_lifecycle_version")

    def validate(atom, version, expected=None, stack=()):
        if atom in stack:
            raise ValueError("dependency cycle in published record")
        record = state["versions"][atom][str(version)]
        digest = record["sha256"]
        if expected is not None and expected != digest:
            raise ValueError("dependency hash differs from exact published pin")
        source = library_root / "sources" / f"{digest}.py"
        if source_hash(source.read_text(encoding="utf-8")) != digest:
            raise ValueError("immutable source hash mismatch")
        validation = record.get("validation", {})
        cases = validation.get("cases", [])
        case_ids = [case.get("case_id") for case in cases]
        if validation.get("validation_kind") == "code_comparison":
            validate_comparison(validation, atom, digest)
            if record.get("case_ids"):
                raise ValueError("Comparison admission cannot claim replay case coverage")
        elif (validation.get("success") is not True or validation.get("validation_kind") != "fresh_environment_replay"
                or validation.get("candidate_sha256") != digest or not cases
                or len(set(case_ids)) != len(case_ids) or set(record.get("case_ids", [])) != set(case_ids)
                or any(case.get("success") is not True for case in cases)
                or any(state["cases"].get(identifier, {}).get("atom") != atom for identifier in case_ids)):
            raise ValueError("published record lacks successful validation coverage")
        for case_result in cases:
            if case_result.get("replay_evidence", {}).get("kind") == "cached_success":
                validate_cached_result(library_root, state, state["cases"][case_result["case_id"]],
                                       record, validation, case_result)
        for child, pin in record.get("dependencies", {}).items():
            validate(child, pin["version"], pin["sha256"], stack + (atom,))
        return source

    # Reuse admission adds fresh coverage to an existing version, without
    # modifying its immutable publication report or creating a new code version.
    reuse_coverage = {}
    baseline_cases = set((baseline or {}).get("cases", {}))
    for identifier, activation in sorted(state.get("case_activations", {}).items()):
        case = state["cases"].get(identifier)
        item = {"case_id": identifier, "atom": activation.get("atom"), "version": activation.get("version"),
                "sha256": activation.get("sha256"), "validated": False,
                "via_reuse": isinstance(activation.get("validation"), dict),
                "source_run": case.get("source_run") if case else None,
                "new_since_baseline": identifier not in baseline_cases if baseline is not None else None}
        item["from_this_campaign"] = _inside(item["source_run"], worker_roots)
        try:
            atom, version = activation["atom"], activation["version"]
            if case is None or case.get("atom") != atom or activation.get("validation_kind") != "fresh_environment_replay":
                raise ValueError("case activation lacks matching formal case and fresh replay kind")
            validate(atom, version, activation["sha256"])
            evidence = activation.get("validation", state["versions"][atom][str(version)]["validation"])
            results = evidence.get("cases", [])
            ids = [result.get("case_id") for result in results]
            if (evidence.get("success") is not True or evidence.get("validation_kind") != "fresh_environment_replay"
                    or evidence.get("candidate_sha256") != activation["sha256"] or identifier not in ids
                    or len(ids) != len(set(ids)) or any(result.get("success") is not True for result in results)
                    or any(state["cases"].get(case_id, {}).get("atom") != atom for case_id in ids)):
                raise ValueError("case activation lacks successful matching replay evidence")
            for result in results:
                if result.get("replay_evidence", {}).get("kind") == "cached_success":
                    validate_cached_result(library_root, state, state["cases"][result["case_id"]],
                                           state["versions"][atom][str(version)], evidence, result)
            item["validated"] = True
            if item["via_reuse"]:
                proof = next(result for result in results if result["case_id"] == identifier)
                reuse_coverage.setdefault((atom, int(version)), []).append((case, proof))
        except (OSError, ValueError, KeyError, TypeError) as error:
            item["integrity_error"] = str(error)
            issues.append({"path": str(library_root / "state.json"), "case_id": identifier, "error": str(error)})
        report["case_activations"].append(item)

    for atom, versions in sorted(state["versions"].items()):
        spec = get_spec(atom)
        for version, record in sorted(versions.items(), key=lambda item: int(item[0])):
            key = (atom, int(version))
            previous = baseline_versions.get(key)
            entry = {"atom": atom, "version": int(version), "sha256": record["sha256"],
                     "active": state["active"].get(atom) == int(version),
                     "baseline": previous == record["sha256"] if baseline is not None else None,
                     "new_since_baseline": key not in baseline_versions if baseline is not None else None,
                     "source_run": record.get("metadata", {}).get("source_run"),
                     "candidate_id": record.get("candidate_id"), "created_at": record.get("created_at"),
                     "metadata": copy.deepcopy(record.get("metadata", {})),
                     "dependencies": copy.deepcopy(record.get("dependencies", {})),
                     "dependency_sources": {child: {**pin, "source_path": str(library_root / "sources" / f"{pin['sha256']}.py")}
                                            for child, pin in record.get("dependencies", {}).items()},
                     "entrypoint": spec["entrypoint"], "name": spec["name"], "description": spec["description"],
                     "parameter_contract": {"catalog_suggestions": spec["parameters"],
                         "note": "Scene bindings are runtime params; replay coverage lists the tested bindings."},
                     "validation_kind": record.get("validation", {}).get("validation_kind"),
                     "replay_policy": record.get("validation", {}).get("replay_policy", "full_fresh"),
                     "replay_counts": record.get("validation", {}).get("replay_counts"),
                     "coverage": [], "validated": False,
                     "source_path": str(library_root / "sources" / f"{record['sha256']}.py")}
            entry["from_this_campaign"] = _inside(entry["source_run"], worker_roots)
            coverage = [(state["cases"].get(value.get("case_id"), {}), value, "publication")
                        for value in record.get("validation", {}).get("cases", [])]
            original_case_ids = {value.get("case_id") for value in record.get("validation", {}).get("cases", [])}
            coverage.extend((case, result, "reuse_acceptance") for case, result in reuse_coverage.get(key, [])
                            if result["case_id"] not in original_case_ids)
            for case, case_result, origin in coverage:
                case_id = case_result.get("case_id")
                entry["coverage"].append({"case_id": case_id, "success": case_result.get("success"),
                    "evidence_origin": origin,
                    "replay_origin": case_result.get("replay_evidence", {}).get("kind", "fresh"),
                    "native_success": case_result.get("native_success", case_result.get("native_task_success")),
                    "task_config": copy.deepcopy(case.get("task_config")),
                    "params": copy.deepcopy(case.get("step", {}).get("params", {})),
                    "require_native_success": case.get("require_native_success", False),
                    "source_run": case.get("source_run")})
            try:
                validate(atom, int(version))
                if previous is not None and previous != record["sha256"]:
                    raise ValueError("published version hash changed relative to baseline")
                entry["validated"] = True
                entry["replay_verified"] = entry["validation_kind"] == "fresh_environment_replay"
            except (OSError, ValueError, KeyError, TypeError) as error:
                entry["integrity_error"] = str(error)
                issues.append({"path": entry["source_path"], "error": str(error)})
            report["published_versions"].append(entry)
            if entry["baseline"]:
                report["baseline_versions"].append(entry)
            if entry["new_since_baseline"]:
                report["new_versions"].append(entry)
                if entry["from_this_campaign"]:
                    report["campaign_new_versions"].append(entry)
            if entry["active"] and entry["validated"]:
                report["active_validated_tools"].append(entry)
    report["counts"] = {"published_versions": len(report["published_versions"]),
        "active_validated_tools": len(report["active_validated_tools"]),
        "baseline_versions_retained": len(report["baseline_versions"]),
        "new_versions_since_baseline": len(report["new_versions"]) if baseline is not None else None,
        "new_versions_from_this_campaign": len(report["campaign_new_versions"]) if baseline is not None else None,
        "new_validated_versions_from_this_campaign": sum(entry["validated"] for entry in report["campaign_new_versions"]) if baseline is not None else None,
        "published_versions_with_integrity_errors": sum(not entry["validated"] for entry in report["published_versions"]),
        "unvalidated_candidates": sum(candidate.get("status") == "candidate" for candidate in state["candidates"].values()),
        "candidate_statuses": dict(Counter(candidate.get("status", "unknown") for candidate in state["candidates"].values())),
        "failure_lessons": sum(len(entries) for entries in state["lessons"].values()),
        "formal_regression_cases": len(state["cases"]), "pending_cases": len(state.get("pending_cases", {})),
        "comparison_admitted_versions": sum(e["validated"] and e["validation_kind"] == "code_comparison"
                                            for e in report["published_versions"]),
        "working_memory_records": sum(len(rows) for rows in state.get("working_memory_index", {}).values()),
        "working_drafts": sum(row["kind"] == "draft" for rows in state.get("working_memory_index", {}).values() for row in rows),
        "case_activation_records": len(report["case_activations"]),
        "validated_case_activations": sum(item["validated"] for item in report["case_activations"]),
        "case_activation_integrity_errors": sum(not item["validated"] for item in report["case_activations"]),
        "validated_reuse_case_admissions": sum(item["validated"] and item["via_reuse"] for item in report["case_activations"]),
        "new_formal_cases_since_baseline": len(set(state["cases"]) - baseline_cases) if baseline is not None else None,
        "new_validated_cases_from_this_campaign": sum(item["validated"] and item["new_since_baseline"]
            and item["from_this_campaign"] for item in report["case_activations"]) if baseline is not None else None,
        "pending_cases_from_this_campaign": sum(_inside(case.get("source_run"), worker_roots)
            for case in state.get("pending_cases", {}).values())}
    return report


def summarize_campaign(plan_path, receipt_path=None, baseline_state=None):
    plan_path = Path(plan_path).resolve()
    issues = []
    plan = _read(plan_path, issues)
    if plan is None:
        raise ValueError(f"Cannot read campaign plan: {plan_path}")
    run = Path(plan["run"]).resolve()
    receipt_path = Path(receipt_path) if receipt_path else run / "parallel_receipt.json"
    receipt = _read(receipt_path, issues) or {}
    ledger = {}
    for worker in receipt.get("workers", []):
        if worker["index"] in ledger:
            raise ValueError("Duplicate task index in receipt")
        ledger[worker["index"]] = worker
    if len({worker["index"] for worker in plan["workers"]}) != len(plan["workers"]):
        raise ValueError("Duplicate task index in plan")
    tasks = [_task_summary(worker, ledger.get(worker["index"], {}), issues) for worker in plan["workers"]]
    config = _read(run / "config.json", issues) or {}
    if baseline_state is None:
        baseline_state = config.get("experiment_metadata", {}).get("library_baseline_state")
    tools = _tool_report(plan.get("shared_library"), baseline_state,
                         [Path(worker["run"]).resolve() for worker in plan["workers"]], issues)
    counts = {"planned_tasks": len(tasks), "attempted_tasks": sum(task["attempted"] for task in tasks),
              "worker_attempted_tasks": sum(task["worker_attempted"] for task in tasks),
              "settled_tasks": sum(task["settled"] for task in tasks),
              "successful_tasks": sum(task["outcome"] == "success" for task in tasks),
              "failed_tasks": sum(task["outcome"] == "failed" for task in tasks),
              "tasks_without_native_episode": sum(task["outcome"] == "no_episode" for task in tasks),
              "pending_tasks": sum(task["outcome"] == "pending" for task in tasks),
              "running_tasks": sum(task["outcome"] == "running" for task in tasks),
              "exploration_invocations": sum(task["exploration_invocations"] for task in tasks),
              "native_episodes_confirmed": sum(task["native_episodes_confirmed"] for task in tasks),
              "episode_limit_violations": sum(task["episode_limit_violated"] for task in tasks)}
    all_settled = bool(tasks) and all(task["settled"] for task in tasks)
    all_workers_attempted = all_settled and all(task["worker_attempted"] for task in tasks)
    all_attempted = all_settled and all(task["attempted"] and not task["episode_limit_violated"] for task in tasks)
    return {"status": "all_attempted" if all_attempted else "needs_episode_recovery" if all_settled else "incomplete", "job": plan.get("job"),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(), "plan": str(plan_path),
            "receipt": str(receipt_path), "supervisor_status": receipt.get("status"),
            "all_tasks_attempted": all_attempted, "all_native_episodes_attempted": all_attempted,
            "all_workers_attempted": all_workers_attempted, "all_workers_settled": all_settled,
            "all_tasks_succeeded": bool(tasks) and all(task["outcome"] == "success" for task in tasks),
            "episode_policy": config.get("episode_retry"), "counts": counts,
            "tasks": tasks, "tools": tools, "issues": issues}


def write_summary(result, summary_path, tool_report_path):
    """Write final artifacts, exporting source only for intact active versions."""
    summary_path, tool_report_path = Path(summary_path).resolve(), Path(tool_report_path).resolve()
    for path in (summary_path, tool_report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    code_dir = tool_report_path.parent / "code_tools"
    tools = result["tools"]
    for entry in tools["active_validated_tools"]:
        code = Path(entry["source_path"]).read_text(encoding="utf-8")
        if source_hash(code) != entry["sha256"]:
            raise ValueError(f"Tool source changed during export: {entry['source_path']}")
        code_dir.mkdir(exist_ok=True)
        path = code_dir / f"{entry['atom']}_v{entry['version']}.py"
        if path.exists() and source_hash(path.read_text(encoding="utf-8")) != entry["sha256"]:
            raise ValueError(f"Refusing to replace a different code export: {path}")
        if not path.exists():
            Library._atomic_write(path, code)
        entry["exported_source_path"] = str(path)
        export = {key: copy.deepcopy(entry[key]) for key in
                  ("atom", "version", "sha256", "source_path", "exported_source_path", "entrypoint", "baseline", "new_since_baseline")}
        if export not in tools["exported_code"]:
            tools["exported_code"].append(export)
    if code_dir.is_dir():
        Library._atomic_write(code_dir / "manifest.json", json.dumps(tools["exported_code"], ensure_ascii=False, indent=2) + "\n")
    Library._atomic_write(tool_report_path, json.dumps(tools, ensure_ascii=False, indent=2) + "\n")
    Library._atomic_write(summary_path, json.dumps(result, ensure_ascii=False, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--receipt")
    parser.add_argument("--baseline-state")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tool-report", required=True)
    args = parser.parse_args(argv)
    result = summarize_campaign(args.plan, args.receipt, args.baseline_state)
    write_summary(result, args.output, args.tool_report)
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "output": str(Path(args.output).resolve()), "tool_report": str(Path(args.tool_report).resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
