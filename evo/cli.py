"""CLI and task-stream orchestration. One library is inherited across tasks."""
import argparse
import copy
from pathlib import Path
import json
import sys

from .agent import Agent, CommandProvider, HTTPProvider, ScriptedProvider
from .catalog import CATALOG, ACTION_CATALOG, PERCEPTION_CATALOG, COORDINATION_CATALOG, CATALOG_VERSION
from .io import read_json, write_json
from .library import Library
from .loops import ExplorationLoop, MaintenanceLoop, SkillExecutor, validate_plan
from .runtime_api import RuntimeAPI, API_DESCRIPTION
from .robocasa_adapter import RoboCasaAdapter, HOST_PREDICATES


def adapter_factory(config, output):
    if config.get("backend", "robocasa") != "robocasa":
        raise ValueError("The CLI benchmark backend must be robocasa")
    kwargs = {k:config[k] for k in ("layout_ids", "style_ids", "camera_size", "scene_split", "horizon") if k in config}
    return RoboCasaAdapter(config["task"], seed=config.get("seed", 0), output_dir=output, **kwargs)


def create_agent(config, output):
    kind = config.get("provider", "http")
    if kind == "http":
        options = {k:config[k] for k in ("model", "base_url", "api_key_env", "timeout_s", "max_retries",
                                        "api", "reasoning_effort") if k in config}
        provider = HTTPProvider(**options)
    elif kind == "command":
        provider = CommandProvider(config["command"], timeout_s=config.get("timeout_s", 180))
    elif kind == "scripted":
        provider = ScriptedProvider(read_json(config["responses"]))
    else:
        raise ValueError("Unknown agent provider: " + kind)
    return Agent(provider, log_dir=output, max_attempts=config.get("format_attempts", 2))


def _loop_options(config):
    values = config.get("exploration", {})
    options = {k:values[k] for k in ("max_attempts", "max_replans", "code_timeout_s", "max_plan_segments",
                                    "plan_one_step", "record_working_memory") if k in values}
    if config.get("maintenance", {}).get("admission_policy") == "compare_existing":
        options.setdefault("plan_one_step", True)
        options.setdefault("record_working_memory", True)
    return options


def _episode_retry_options(config):
    """Absence preserves one-episode behavior; null max_episodes means unlimited."""
    enabled = "episode_retry" in config
    options = config.get("episode_retry", {})
    if not isinstance(options, dict):
        raise ValueError("episode_retry must be an object")
    maximum = options.get("max_episodes", 1)
    history = options.get("history_episodes", 3)
    if maximum is not None and (type(maximum) is not int or maximum < 1):
        raise ValueError("episode_retry.max_episodes must be null or a positive integer")
    if type(history) is not int or history < 1:
        raise ValueError("episode_retry.history_episodes must be a positive integer")
    return enabled, maximum, history


def _compact_attempt(record):
    """Keep executable prior code and verdicts; never present old sensor images as current."""
    execution = record.get("execution") or {}
    step = record.get("step") or {"id": record.get("step_id"), "atom": record.get("atom"),
                                  "params": record.get("params", {})}
    result = {"step": copy.deepcopy(step), "code": record.get("code", ""),
              "dependencies": copy.deepcopy(record.get("dependencies", {})),
              "error": copy.deepcopy(record.get("error", execution.get("error"))),
              "verification": copy.deepcopy(record.get("verification", {})),
              "success": bool(record.get("success", False)), "skipped": bool(record.get("skipped", False))}
    if record.get("sha256"):
        result["sha256"] = record["sha256"]
    for key in ("executed", "failure_feedback", "description"):
        if key in record:
            result[key] = copy.deepcopy(record[key])
    result["execution"] = {key: copy.deepcopy(execution[key]) for key in
                           ("ok", "result", "error", "elapsed_s") if key in execution}
    for key in ("before", "after"):
        if isinstance(execution.get(key), dict):
            result["execution"][key] = {k:v for k,v in execution[key].items()
                if k in {"steps", "horizon", "terminal", "native_success", "reward"}}
    return result


def _episode_history(directory, result=None, episode_attempt=None):
    """Build bounded model memory from existing exploration artifacts."""
    directory = Path(directory).expanduser().resolve()
    if result is None:
        summary = directory / "summary.json"
        if not summary.is_file():
            summary = directory / "progress.json"
        if not summary.is_file():
            raise ValueError(f"Retry experience has no summary.json or progress.json: {directory}")
        result = read_json(summary)
    task_config = result.get("task_config")
    if task_config is None and (directory / "task.json").is_file():
        task_config = read_json(directory / "task.json")
    failures = []
    for path in sorted((directory / "attempts").glob("*/record.json"), reverse=True):
        record = read_json(path)
        if record.get("success") is not True:
            failures.append(_compact_attempt(record))
            if len(failures) == 3:
                break
    failures.reverse()
    if not failures:
        history = result.get("attempt_history", result.get("failures", []))
        failures = [_compact_attempt(record) for record in history if record.get("success") is not True][-3:]
    return {"episode_attempt": episode_attempt, "source_directory": str(directory),
            "summary_path": str(directory / "summary.json"), "task_config": copy.deepcopy(task_config),
            "status": result.get("status"), "native_success": bool(result.get("native_success", False)),
            "error": copy.deepcopy(result.get("error")), "stop_reason": result.get("stop_reason"),
            "stats": {key: value for key, value in result.get("stats", {}).items()
                      if key in {"steps", "horizon", "terminal", "native_success", "reward"}},
            "observations_are_stale": True,
            "completed_steps": [_compact_attempt(record) for record in result.get("completed_steps", [])],
            "recent_failures": failures}


def _initial_retry_history(config, task, history_limit):
    paths = config.get("retry_experiences", [])
    if isinstance(paths, str):
        paths = [paths]
    if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
        raise ValueError("retry_experiences must be an exploration directory or list of directories")
    history = []
    for path in paths:
        entry = _episode_history(path)
        if entry["task_config"] == task:
            history.append(entry)
    return history[-history_limit:]


def _attempt_summary(result, directory, number):
    return {"episode_attempt": number, "status": result.get("status"),
            "native_success": bool(result.get("native_success", False)),
            "summary_path": str(Path(directory).resolve() / "summary.json"),
            "stats": {key: value for key, value in result.get("stats", {}).items()
                      if key in {"steps", "horizon", "terminal", "native_success", "reward"}},
            "completed_steps": len(result.get("completed_steps", [])),
            "failures_count": len(result.get("failures", [])), "pending_count": len(result.get("pending", []))}


def _campaign_event(event, **values):
    # Cluster stdout/stderr remains observable when shared-volume readers have
    # cached an old inode of an atomically replaced summary.json.
    print(json.dumps(dict(event=event, **values), ensure_ascii=False), file=sys.stderr, flush=True)


def run_config(config, output):
    retry_enabled, max_episodes, history_limit = _episode_retry_options(config)
    comparison_policy = config.get("maintenance", {}).get("admission_policy") == "compare_existing"
    maintenance_trigger = config.get("maintenance_trigger", "episode" if comparison_policy else "atomic")
    if maintenance_trigger not in {"atomic", "episode"}:
        raise ValueError("maintenance_trigger must be atomic or episode")
    maintenance_enabled = config.get("maintain_after_task", True)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    library = Library(config.get("library", "library"))
    write_json(output / "library_before.json", library.snapshot())
    agent = create_agent(config["agent"], output / "model")
    common = dict(library=library, agent=agent, adapter_factory=adapter_factory,
                  runtime_api=RuntimeAPI, api_description=API_DESCRIPTION,
                  supported_predicates=HOST_PREDICATES)
    exploration = ExplorationLoop(**common, **_loop_options(config))
    maintenance = MaintenanceLoop(**common, **config.get("maintenance", {}))
    tasks = config.get("tasks") or [config["task"]]
    report = dict(status="running", task_status="running", episodes=[], library_before=library.snapshot(),
                  maintenance_trigger=maintenance_trigger, maintenance_runs=[], promoted_versions=0, accepted_case_count=0,
                  maintenance_status="not_needed" if maintenance_enabled else "disabled")
    handled_experiences = set()

    def update_maintenance_totals():
        """Preserve the legacy aggregate while retaining every independent run."""
        for record in report["episodes"]:
            runs = record["maintenance_runs"]
            if not runs:
                record["maintenance_status"] = "not_needed" if maintenance_enabled else "disabled"
                continue
            aggregate = {"status": "complete", "proposals": [], "promoted": [], "rejected": [],
                         "accepted_cases": [], "drafts": [], "kept": [], "retained_failures": []}
            for run in runs:
                outcome = run.get("result") or {}
                for key in ("proposals", "promoted", "rejected", "accepted_cases", "drafts", "kept", "retained_failures"):
                    aggregate[key].extend(outcome.get(key, []))
            running = any(run["status"] == "running" for run in runs)
            failed = any(run["status"] not in {"running", "complete"} or
                         (run.get("result") or {}).get("rejected") for run in runs)
            record["maintenance_status"] = "running" if running else "failed" if failed else "complete"
            aggregate["status"] = ("running" if running else "some_candidates_rejected" if failed else "complete")
            record["maintenance"] = aggregate
        statuses = [record["maintenance_status"] for record in report["episodes"]]
        report["maintenance_status"] = ("disabled" if not maintenance_enabled else
            "running" if "running" in statuses else "failed" if "failed" in statuses else
            "complete" if "complete" in statuses else "not_needed")
        report["promoted_versions"] = sum(len(record["maintenance"]["promoted"])
            for record in report["episodes"] if record["maintenance"])
        report["accepted_case_count"] = sum(len(record["maintenance"]["accepted_cases"])
            for record in report["episodes"] if record["maintenance"])
        report["draft_count"] = sum(len(record["maintenance"]["drafts"])
            for record in report["episodes"] if record["maintenance"])
        report["kept_count"] = sum(len(record["maintenance"]["kept"])
            for record in report["episodes"] if record["maintenance"])
        report["retained_failure_count"] = sum(len(record["maintenance"]["retained_failures"])
            for record in report["episodes"] if record["maintenance"])

    try:
        for index, task in enumerate(tasks):
            episode = output / f"{index:03d}_{task['task']}_seed{task.get('seed', 0)}"
            history = _initial_retry_history(config, task, history_limit)
            task_record = dict(task=copy.deepcopy(task), exploration=None, maintenance=None,
                               maintenance_runs=[], maintenance_status="not_needed" if maintenance_enabled else "disabled",
                               episode_attempts=[], retry_enabled=retry_enabled, retry_status="running")
            report["episodes"].append(task_record)
            attempt = 0

            def maintain_pending(paths, trigger):
                """Failure-isolated synchronous callback; the exploration env stays alive."""
                if not maintenance_enabled:
                    return
                for raw_path in paths:
                    path = str(Path(raw_path).expanduser().resolve())
                    if path in handled_experiences:
                        continue
                    # Mark before dispatch so a repeated callback or episode-end
                    # sweep cannot perform the same maintenance operation twice.
                    handled_experiences.add(path)
                    directory = episode / "maintenance" / f"run_{len(task_record['maintenance_runs']) + 1:04d}"
                    entry = dict(task_index=index, episode_attempt=attempt, trigger=trigger,
                                 experience_paths=[path], output=str(directory), status="running", result=None)
                    task_record["maintenance_runs"].append(entry)
                    report["maintenance_runs"].append(entry)
                    update_maintenance_totals()
                    write_json(episode / "maintenance_runs.json", task_record["maintenance_runs"])
                    write_json(output / "maintenance_runs.json", report["maintenance_runs"])
                    write_json(output / "summary.json", report)
                    _campaign_event("maintenance_start", task=task["task"], episode_attempt=attempt,
                                    trigger=trigger, experience=path, output=str(directory))
                    try:
                        outcome = maintenance.run([path], directory)
                        if (not isinstance(outcome, dict) or not isinstance(outcome.get("status"), str)
                                or any(not isinstance(outcome.get(key), list) for key in ("proposals", "promoted", "rejected"))):
                            raise ValueError("Maintenance result must contain status and proposals/promoted/rejected lists")
                    except Exception as error:
                        # A failed replay/model request/library write must not
                        # discard subsequent exploration or prevent a reset.
                        failure = {"type": type(error).__name__, "message": str(error)}
                        outcome = dict(status="error", proposals=[], promoted=[],
                                       rejected=[{"experience": path, "error": failure}], error=failure)
                        write_json(directory / "summary.json", outcome)
                    entry["result"] = outcome
                    entry["status"] = outcome["status"]
                    update_maintenance_totals()
                    write_json(episode / "maintenance_runs.json", task_record["maintenance_runs"])
                    write_json(output / "maintenance_runs.json", report["maintenance_runs"])
                    write_json(output / "summary.json", report)
                    _campaign_event("maintenance_end", task=task["task"], episode_attempt=attempt,
                                    trigger=trigger, status=outcome["status"], promoted=len(outcome["promoted"]))

            while True:
                attempt += 1
                run_directory = (episode / "exploration" if attempt == 1 else
                                 episode / "retries" / f"episode_{attempt:04d}" / "exploration")
                report["current_task_index"] = index
                report["current_episode_attempt"] = attempt
                task_record["retry_status"] = "running"
                write_json(output / "summary.json", report)
                _campaign_event("episode_start", task=task["task"], seed=task.get("seed", 0),
                                episode_attempt=attempt, history_episodes=len(history), output=str(run_directory))
                retry_context = {"episode_attempt": attempt, "reset_to_initial_state": True,
                                 "previous_episodes": copy.deepcopy(history[-history_limit:])}
                run_options = {"retry_context": retry_context}
                if maintenance_enabled and maintenance_trigger == "atomic":
                    run_options["on_atomic_success"] = lambda path: maintain_pending([path], "atomic")
                result = exploration.run(copy.deepcopy(task), run_directory, **run_options)
                task_record["exploration"] = result
                task_record["episode_attempts"].append(_attempt_summary(result, run_directory, attempt))
                # Episode mode also maintains verified partial work from failed
                # episodes. Atomic mode sweeps only paths not already dispatched.
                paths = result.get("pending", []) + (result.get("incomplete", []) if comparison_policy else [])
                maintain_pending(paths, "episode" if maintenance_trigger == "episode" else "episode_pending")
                succeeded = result.get("status") == "success" and result.get("native_success") is True
                retryable = result.get("status") in {"step_failed", "plan_complete_task_failed"}
                has_budget = max_episodes is None or attempt < max_episodes
                will_retry = not succeeded and retryable and has_budget
                task_record["retry_status"] = ("success" if succeeded else "retrying" if will_retry else
                                                "budget_exhausted" if retryable else "not_retryable")
                report["total_episode_attempts"] = sum(len(item["episode_attempts"]) for item in report["episodes"])
                write_json(output / "summary.json", report)
                _campaign_event("episode_end", task=task["task"], seed=task.get("seed", 0),
                                episode_attempt=attempt, status=result.get("status"),
                                native_success=result.get("native_success"),
                                steps=result.get("stats", {}).get("steps"), next_action=task_record["retry_status"])
                if not will_retry:
                    break
                history.append(_episode_history(run_directory, result, attempt))
                history = history[-history_limit:]
            write_json(output / "summary.json", report)
        task_success = all(e["exploration"]["native_success"] and e["exploration"]["status"] == "success"
                           for e in report["episodes"])
        update_maintenance_totals()
        validation_success = report["maintenance_status"] not in {"failed", "running"}
        report["task_status"] = "complete" if task_success else "tasks_incomplete"
        report["status"] = ("complete" if validation_success else "validation_failed") if task_success else "tasks_incomplete"
        report["library_after"] = library.snapshot()
        report["native_successes"] = sum(e["exploration"]["native_success"] for e in report["episodes"])
        report["program_successes"] = sum(e["exploration"]["status"] == "success" for e in report["episodes"])
        report["total_tasks"] = len(report["episodes"])
    except Exception as error:
        import traceback
        report.update(status="error", task_status="error", error={"type":type(error).__name__, "message":str(error),
                                              "traceback":traceback.format_exc()})
    finally:
        write_json(output / "summary.json", report)
    return report


def evaluate_frozen(config, output):
    """Use exact library pins and a supplied plan; no repair or library writeback."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    library = Library(config["library"])
    frozen = library.snapshot()
    write_json(output / "frozen_library.json", frozen)
    reports = []
    for index, item in enumerate(config["evaluations"]):
        directory = output / f"{index:03d}"
        adapter = None
        report = dict(task=item["task"], success=False, steps=[])
        try:
            adapter = adapter_factory(item["task"], directory / "simulation")
            if adapter.native_success():
                raise ValueError("Evaluation task already successful at reset")
            plan = validate_plan(item["plan"], HOST_PREDICATES)
            executor = SkillExecutor(adapter, library, RuntimeAPI, config.get("code_timeout_s", 120))
            for step in plan["steps"]:
                before_check = adapter.evaluate(step["postcondition"])
                if before_check.get("known", True) and before_check["success"]:
                    report["steps"].append(dict(step=step, skipped=True,
                        execution=dict(ok=True, skipped=True), verification=before_check))
                    continue
                if step["atom"] not in frozen:
                    raise ValueError("Unimplemented frozen skill: " + step["atom"])
                pin = frozen[step["atom"]]
                source = library.get_version(step["atom"], pin["version"], pin["sha256"])
                result = executor.run(source["code"], step["params"], source["dependencies"])
                check = adapter.evaluate(step["postcondition"])
                report["steps"].append(dict(step=step, execution=result, verification=check))
                if not result["ok"] or not check.get("known", True) or not check["success"]:
                    break
            report["success"] = bool(adapter.native_success()) and len(report["steps"]) == len(plan["steps"]) and all(
                s["execution"]["ok"] and s["verification"].get("known", True) and s["verification"]["success"] for s in report["steps"])
            report["stats"] = adapter.stats()
        except Exception as error:
            report["success"] = False
            report["error"] = {"type":type(error).__name__, "message":str(error)}
        finally:
            if adapter:
                try:
                    adapter.close()
                except Exception as error:
                    report["success"] = False
                    report["close_error"] = {"type":type(error).__name__, "message":str(error)}
            write_json(directory / "report.json", report)
            reports.append(report)
    summary = dict(status="complete" if all(r["success"] for r in reports) else "evaluation_failed",
                   evaluations=reports, native_successes=sum(r["success"] for r in reports),
                   frozen=frozen, library_unchanged=library.snapshot() == frozen)
    write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "inspect"):
        sub = commands.add_parser(name)
        sub.add_argument("--library", required=True)
    commands.add_parser("catalog")
    sub = commands.add_parser("rollback")
    sub.add_argument("--library", required=True)
    sub.add_argument("atom", choices=CATALOG)
    sub.add_argument("version", type=int)
    for name in ("run", "explore", "maintain", "evaluate", "evaluate-policy"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True)
        sub.add_argument("--output", required=True)
        if name == "maintain":
            sub.add_argument("--experience", nargs="+", required=True)
    args = parser.parse_args(argv)
    if args.command == "catalog":
        result = CATALOG
    elif args.command in ("init", "inspect"):
        lib = Library(args.library)
        result = dict(root=str(lib.root), defined_atoms=len(CATALOG), executable_atoms=len(CATALOG),
                      perception_atoms=len(PERCEPTION_CATALOG), action_atoms=len(ACTION_CATALOG), coordination_atoms=len(COORDINATION_CATALOG), atomic_catalog_version=CATALOG_VERSION, active=lib.snapshot(),
                      working_memory=lib.working_catalog())
    elif args.command == "rollback":
        result = Library(args.library).rollback(args.atom, args.version)
    else:
        config = read_json(args.config)
        if args.command == "evaluate-policy":
            from .policy_evaluation import run_policy_evaluation
            result = run_policy_evaluation(config, args.output)
        elif args.command == "evaluate":
            result = evaluate_frozen(config, args.output)
        elif args.command == "maintain":
            lib = Library(config["library"])
            agent = create_agent(config["agent"], Path(args.output).with_name(Path(args.output).name+"-model"))
            result = MaintenanceLoop(lib, agent, adapter_factory, RuntimeAPI, API_DESCRIPTION,
                                     HOST_PREDICATES, **config.get("maintenance", {})).run(args.experience, args.output)
        else:
            if args.command == "explore":
                config["maintain_after_task"] = False
            result = run_config(config, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("status") in ("error", "tasks_incomplete", "validation_failed",
                                        "evaluation_failed", "some_candidates_rejected") else 0


if __name__ == "__main__":
    raise SystemExit(main())
