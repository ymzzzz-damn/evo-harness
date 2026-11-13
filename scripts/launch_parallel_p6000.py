"""Split a task stream into independent CPU Jobs sharing its original skill library.

Each child is launched by launch_p6000.py. A dry run freezes child sources and
manifests without submitting Jobs, and consumes the batch/run names just as the
single-job launcher's dry run does. Use a new batch name for a later submission.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _history_task(directory):
    """Match the task identity used by cli._initial_retry_history, without imports."""
    directory = Path(directory).expanduser().resolve()
    summary = directory / "summary.json"
    if not summary.is_file():
        summary = directory / "progress.json"
    if not summary.is_file():
        raise ValueError(f"Retry experience has no summary.json or progress.json: {directory}")
    report = json.loads(summary.read_text())
    if not isinstance(report, dict):
        raise ValueError(f"Retry experience report must be an object: {summary}")
    task = report.get("task_config")
    if task is None and (directory / "task.json").is_file():
        task = json.loads((directory / "task.json").read_text())
    if not isinstance(task, dict):
        raise ValueError(f"Retry experience has no task configuration: {directory}")
    return str(directory), task


def prepare_batch(config_path, name, project, runtime, deadline, dry_run):
    """Validate all inputs before creating a batch or invoking any launcher."""
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,40}[a-z0-9])?", name):
        raise ValueError("batch name must use lowercase letters/digits/hyphens, end in a letter/digit, and be at most 42 characters")
    if deadline < 0:
        raise ValueError("deadline must be nonnegative; use 0 for no wall-time deadline")
    project, runtime = Path(project).resolve(), Path(runtime).resolve()
    config_path = Path(config_path).expanduser().resolve()
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError("source configuration must be an object")
    if config.get("mode", "run") != "run":
        raise ValueError("parallel task launch supports run configurations only")
    tasks = config.get("tasks") or ([config["task"]] if "task" in config else [])
    if not isinstance(tasks, list) or not tasks or len(tasks) > 99:
        raise ValueError("source configuration must contain 1..99 tasks")
    if any(not isinstance(task, dict) or not isinstance(task.get("task"), str)
           or not task["task"].strip() for task in tasks):
        raise ValueError("each task must be an object with a nonempty task name")
    library = config.get("library", str(project / "library"))
    if not isinstance(library, str) or not library.strip():
        raise ValueError("library must be a nonempty shared path")
    for required in (project / "evo", project / "scripts", project / "pyproject.toml",
                     runtime / "deployment/job_template.json"):
        if not required.exists():
            raise ValueError(f"Required launcher input does not exist: {required}")
    histories = config.get("retry_experiences", [])
    if isinstance(histories, str):
        histories = [histories]
    if not isinstance(histories, list) or any(not isinstance(path, str) for path in histories):
        raise ValueError("retry_experiences must be a directory or list of directories")
    history_tasks = [_history_task(path) for path in histories]
    batch = project / "batches" / name
    if batch.exists():
        raise ValueError(f"Batch already exists; choose a new name: {batch}")
    children = []
    for index, task in enumerate(tasks, 1):
        child_name = f"{name}-t{index:02d}"
        run = project / "runs" / child_name
        if run.exists():
            raise ValueError(f"Run already exists; choose a new batch name: {run}")
        child_config = copy.deepcopy(config)
        child_config.pop("task", None)
        child_config["tasks"] = [copy.deepcopy(task)]
        child_config["library"] = library
        if "retry_experiences" in config:
            child_config["retry_experiences"] = [path for path, previous_task in history_tasks
                                                 if previous_task == task]
        children.append({"index": index, "task": copy.deepcopy(task), "name": child_name,
                         "job": "yumuzhen-" + child_name, "run": str(run),
                         "config": str(batch / "configs" / f"task_{index:02d}.json"),
                         "launcher_log": str(batch / f"task_{index:02d}_launcher.log"),
                         "status": "planned", "_configuration": child_config})
    receipt = {"batch": name, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "source_config": str(config_path), "project": str(project), "runtime": str(runtime),
               "shared_library": library, "dry_run": dry_run,
               "job_deadline_seconds": deadline or None, "status": "prepared", "tasks": children}
    return batch, receipt


def launch_batch(config_path, name, project, runtime, deadline=0, dry_run=False):
    batch, receipt = prepare_batch(config_path, name, project, runtime, deadline, dry_run)
    (batch / "configs").mkdir(parents=True)
    for child in receipt["tasks"]:
        _write_json(Path(child["config"]), child.pop("_configuration"))
    manifest = batch / "batch.json"
    _write_json(manifest, receipt)
    launcher = Path(__file__).resolve().with_name("launch_p6000.py")
    for child in receipt["tasks"]:
        command = [sys.executable, str(launcher), "--config", child["config"], "--name", child["name"],
                   "--project", receipt["project"], "--runtime", receipt["runtime"],
                   "--deadline", str(deadline)]
        if dry_run:
            command.append("--dry-run")
        child["status"] = "preparing" if dry_run else "submitting"
        _write_json(manifest, receipt)
        with Path(child["launcher_log"]).open("w") as log:
            try:
                result = subprocess.run(command, cwd=receipt["project"], stdout=log, stderr=subprocess.STDOUT)
                child["launcher_returncode"] = result.returncode
            except OSError as error:
                child["launcher_error"] = {"type": type(error).__name__, "message": str(error)}
                child["launcher_returncode"] = None
        if child["launcher_returncode"] == 0:
            child["status"] = "dry_run_prepared" if dry_run else "submitted"
        else:
            # An API timeout can follow server-side creation. Do not imply this
            # child is absent, or encourage an automatic duplicate submission.
            child["status"] = "preparation_failed" if dry_run else "submission_unconfirmed"
            receipt["status"] = "incomplete"
            _write_json(manifest, receipt)
            print(json.dumps(child, ensure_ascii=False), flush=True)
            return receipt, manifest
        _write_json(manifest, receipt)
        print(json.dumps(child, ensure_ascii=False), flush=True)
    receipt["status"] = "dry_run_prepared" if dry_run else "submitted"
    _write_json(manifest, receipt)
    return receipt, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="source multi-task run configuration")
    parser.add_argument("--name", "--batch-name", dest="name", required=True, help="unique batch name; child runs append -t01, -t02, ...")
    parser.add_argument("--project", default="/mnt/auwomo-data/yumuzhen/my/evo")
    parser.add_argument("--runtime", default="/mnt/auwomo-data/yumuzhen/physical-rsi")
    parser.add_argument("--deadline", type=int, default=0, help="seconds per Job; default 0 preserves unlimited episode retries")
    parser.add_argument("--dry-run", action="store_true", help="freeze configurations/manifests without submitting; consumes batch/run names")
    args = parser.parse_args(argv)
    try:
        receipt, manifest = launch_batch(args.config, args.name, args.project, args.runtime,
                                         args.deadline, args.dry_run)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"batch": receipt["batch"], "status": receipt["status"], "manifest": str(manifest),
                      "shared_library": receipt["shared_library"], "tasks": receipt["tasks"]},
                     ensure_ascii=False, indent=2), flush=True)
    return 0 if receipt["status"] in {"submitted", "dry_run_prepared"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
