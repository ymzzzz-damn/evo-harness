#!/usr/bin/env python3
"""Audit task-level first-episode and within-three success without retry selection."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evo.io import write_json
from evo.policy_evaluation import GROUPS, MAX_EPISODES, classify_attempt, initial_scene_evidence, trial_outcome


def _read(path, issues, required=False):
    path = Path(path)
    if not path.exists() and not required:
        return None
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError) as error:
        issues.append({"path": str(path), "error": str(error)})
        return None


def _task_row(worker, receipt, issues, expected_frozen_sha=None):
    root = Path(worker["run"]).resolve()
    campaign_root, task = root / "campaign", worker["task"]
    campaign = _read(campaign_root / "summary.json", issues) or {}
    local_receipt = _read(root / "task_receipt.json", issues) or receipt or {}
    matching = [t for t in campaign.get("episodes", []) if t.get("task") == task]
    record = matching[0] if len(matching) == 1 else {}
    violation = []
    if campaign and campaign.get("mode") != "evaluate_policy":
        violation.append("campaign is not evaluate_policy")
    if campaign.get("episodes") and len(matching) != 1:
        violation.append("campaign task does not uniquely match planned task")
    if campaign.get("library_unchanged") is False:
        violation.append("frozen library integrity failed")
    if campaign.get("protocol_integrity_error"):
        violation.append("policy evaluation integrity failed")
    if campaign and expected_frozen_sha and campaign.get("frozen_library", {}).get("state_sha256") != expected_frozen_sha:
        violation.append("worker frozen library differs from planned state SHA")
    settled = (campaign.get("status") in {"complete", "error"}
               or local_receipt.get("status") in {"succeeded", "failed", "interrupted", "cancelled"})
    discovered = {}
    for directory in list(campaign_root.glob("*/exploration")) + list(campaign_root.glob("*/retries/*/exploration")):
        directory = directory.resolve()
        if campaign_root not in directory.parents:
            violation.append("physical attempt directory escapes planned worker")
            continue
        try:
            number = int(directory.parent.name.removeprefix("episode_")) if "retries" in directory.parts else 1
        except ValueError:
            violation.append("invalid physical attempt directory name")
            continue
        if number < 1:
            violation.append("physical attempt number must be positive")
            continue
        if number in discovered and directory != discovered[number]:
            violation.append("multiple exploration directories for one physical attempt")
        discovered[number] = directory
    ledger = {}
    for attempt in record.get("episode_attempts", []):
        number = attempt.get("episode_attempt")
        if type(number) is not int or number < 1 or number in ledger:
            violation.append("invalid/duplicate physical attempt number")
            continue
        ledger[number] = attempt
        if attempt.get("summary_path"):
            directory = Path(attempt["summary_path"]).resolve().parent
            if campaign_root not in directory.parents:
                violation.append("attempt evidence escapes planned worker")
            elif number in discovered and discovered[number] != directory:
                violation.append("ledger and physical attempt directory disagree")
            else:
                discovered[number] = directory
    if campaign.get("current_episode_attempt") and not discovered and not ledger:
        # A process can die before its first observation or final attempt record.
        discovered[1] = campaign_root / "missing_initial_attempt" / "exploration"
    maximum = max(set(discovered) | set(ledger), default=0)
    if maximum > MAX_EPISODES:
        violation.append("more than three physical attempts")
    attempts, first_fingerprint = [], None
    for number in range(1, maximum + 1):
        directory = discovered.get(number)
        initial = _read(directory / "initial.json", issues) if directory else None
        result = _read(directory / "summary.json", issues) if directory else None
        finished = result is not None or number in ledger
        if result is None and directory:
            result = _read(directory / "progress.json", issues)
        result = result or {"status": "missing_attempt_evidence", "task_config": task}
        if result.get("task_config") != task:
            violation.append(f"attempt {number} task configuration differs")
        scene = initial_scene_evidence(initial, task) if initial else {
            "valid_native_initial": False, "problems": ["initial observation absent"]}
        if scene.get("valid_native_initial"):
            scene["matches_first_scene"] = (scene["fingerprint"] == first_fingerprint
                                             if first_fingerprint is not None else None)
            first_fingerprint = first_fingerprint or scene["fingerprint"]
        if not finished and not settled and number == maximum:
            verdict = {"valid_outcome": False, "success": None, "outcome": "running", "invalid_reasons": []}
        else:
            verdict = classify_attempt(result, scene)
        attempts.append({"episode_attempt": number, "status": result.get("status"),
                         "summary_path": str(directory / "summary.json") if directory else None,
                         **verdict, "initial_scene": scene, "error": copy.deepcopy(result.get("error")),
                         "stats": {k: result.get("stats", {})[k] for k in
                                   ("steps", "horizon", "native_success", "terminal") if k in result.get("stats", {})}})
    outcome = trial_outcome(attempts[:MAX_EPISODES], settled=settled)
    if any(a.get("success") is True for a in attempts[:-1]):
        violation.append("episodes were started after a native success")
    if violation:
        issues.extend({"path": str(root), "error": message} for message in violation)
        outcome.update(first_attempt=None, success_within3=None, trial_outcome="unknown")
    invalid = [a for a in attempts if a["outcome"] == "invalid"]
    return {"index": worker.get("index"), "task": copy.deepcopy(task), "run": str(root),
            "evaluation_group": task.get("evaluation_group"),
            "worker_status": local_receipt.get("status", "pending"), "settled": settled,
            **outcome, "episode_attempts": attempts, "invalid_attempt_count": len(invalid),
            "provider_interruptions": sum((a.get("error") or {}).get("type") == "ProviderError" for a in invalid),
            "native_initial_episodes": sum(a["initial_scene"].get("valid_native_initial", False) for a in attempts),
            "protocol_violations": violation}


def _metric(rows, field, denominator):
    successes = sum(row.get(field) is True for row in rows)
    failures = sum(row.get(field) is False for row in rows)
    unknown = max(0, denominator - successes - failures)
    return {"denominator_tasks": denominator, "observed_successes": successes,
            "known_failures": failures, "unknown_or_pending_tasks": unknown,
            "observed_success_rate": successes / denominator if denominator else None,
            "rate_is_lower_bound": unknown > 0,
            "exact_success_rate": successes / denominator if denominator and unknown == 0 else None}


def summarize_evaluation(plan_path, receipt_path=None):
    plan_path, issues = Path(plan_path).resolve(), []
    plan = _read(plan_path, issues, required=True)
    if plan is None:
        return {"status": "inspection_error", "issues": issues, "tasks": []}
    config_path = Path(plan.get("run", plan_path.parent)) / "config.json"
    config = _read(config_path, issues) or {}
    expected = config.get("policy_evaluation", {}).get("expected_tasks", 50)
    workers = plan.get("workers", [])
    if len(workers) != expected:
        issues.append({"path": str(plan_path), "error": f"planned task count {len(workers)} differs from expected {expected}"})
    names = [w["task"]["task"] for w in workers]
    if len(set(names)) != len(names):
        issues.append({"path": str(plan_path), "error": "duplicate evaluation task IDs"})
    receipt = _read(receipt_path or plan_path.parent / "parallel_receipt.json", issues) or {}
    by_index = {r.get("index"): r for r in receipt.get("workers", []) if isinstance(r, dict)}
    frozen_config = config.get("frozen_library")
    expected_sha = (frozen_config.get("state_sha256") if isinstance(frozen_config, dict)
                    else config.get("frozen_library_sha256"))
    rows = [_task_row(worker, by_index.get(worker.get("index")), issues, expected_sha) for worker in workers]
    groups = {}
    group_counts = Counter(row["evaluation_group"] for row in rows)
    expected_groups = dict(zip(GROUPS, (18, 16, 16))) if expected == 50 else group_counts
    mapping = config.get("policy_evaluation", {}).get("task_groups", {})
    for row in rows:
        task = row["task"]
        if row["evaluation_group"] not in GROUPS or (task["task"] in mapping and mapping[task["task"]] != row["evaluation_group"]):
            issues.append({"path": row["run"], "error": "evaluation group missing or inconsistent"})
        if task.get("seed") != 1 or task.get("scene_split") != "pretrain":
            issues.append({"path": row["run"], "error": "task does not use fixed seed1/pretrain scene"})
    for group in GROUPS:
        selected = [row for row in rows if row["evaluation_group"] == group]
        denominator = expected_groups.get(group, 0)
        if len(selected) != denominator:
            issues.append({"path": str(plan_path), "error": f"group {group} has {len(selected)}, expected {denominator}"})
        groups[group] = {"planned_tasks": len(selected),
                         "first_episode": _metric(selected, "first_attempt", denominator),
                         "within_three": _metric(selected, "success_within3", denominator),
                         "invalid_attempts": sum(row["invalid_attempt_count"] for row in selected)}
    settled = sum(row["settled"] for row in rows)
    invalid = sum(row["invalid_attempt_count"] for row in rows)
    return {"status": "invalid_protocol" if issues else "complete" if settled == expected else "running",
            "mode": "evaluate_policy", "plan": str(plan_path),
            "metric_definition": "First episode and any success within at most3 same-scene episodes, task denominator; invalid episodes consume cap and stay unknown",
            "first_episode": _metric(rows, "first_attempt", expected),
            "within_three": _metric(rows, "success_within3", expected),
            "first_success_histogram": dict(Counter(str(row["first_success_attempt"]) for row in rows if row["first_success_attempt"] is not None)),
            "groups": groups,
            "counts": {"planned_tasks": len(rows), "expected_tasks": expected, "settled_tasks": settled,
                       "physical_attempts": sum(len(row["episode_attempts"]) for row in rows),
                       "native_initial_episodes": sum(row["native_initial_episodes"] for row in rows),
                       "invalid_attempts": invalid,
                       "provider_interruptions": sum(row["provider_interruptions"] for row in rows),
                       "unknown_trials": sum(row["success_within3"] is None for row in rows),
                       "known_failed_trials": sum(row["success_within3"] is False for row in rows)},
            "tasks": rows, "issues": issues}


def write_summary(result, output_path):
    write_json(output_path, result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--receipt")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = summarize_evaluation(args.plan, args.receipt)
    output = args.output or str(Path(args.plan).parent / "evaluation_summary.json")
    write_summary(result, output)
    print(json.dumps({"status": result["status"], "counts": result.get("counts"), "output": output}))
    return int(result["status"] in {"invalid_protocol", "inspection_error"})


if __name__ == "__main__":
    raise SystemExit(main())
