"""Freeze a task list and launch one ordinary eight-GPU Pod with parallel workers."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def history_task(path):
    directory = Path(path).expanduser().resolve()
    report_path = directory / "summary.json"
    if not report_path.is_file():
        report_path = directory / "progress.json"
    report = json.loads(report_path.read_text())
    task = report.get("task_config")
    if task is None:
        task = json.loads((directory / "task.json").read_text())
    return str(directory), task


def prepare(config_path, name, project, runtime, *, deadline=0, preflight=False,
            cpu_request="64", cpu_limit="128", memory_request="192Gi", memory_limit="384Gi",
            max_parallel=None):
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,44}[a-z0-9])?", name):
        raise ValueError("name must be a lowercase Kubernetes name of at most 46 characters")
    if deadline < 0:
        raise ValueError("deadline must be nonnegative")
    project, runtime = Path(project).resolve(), Path(runtime).resolve()
    config_path = Path(config_path).expanduser().resolve()
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError("source configuration must be an object")
    if max_parallel is None:
        max_parallel = config.get("gpu_execution", {}).get("worker_count", 16)
    if (not isinstance(max_parallel, int) or isinstance(max_parallel, bool)
            or not 1 <= max_parallel <= 64):
        raise ValueError("max_parallel must be between one and sixty-four (at most eight workers per GPU)")
    capacity = copy.deepcopy(config.get("capacity_planning", {}))
    if capacity.get("enabled"):
        try:
            from gpu_capacity import memory_mib
        except ModuleNotFoundError:
            from scripts.gpu_capacity import memory_mib
        if max_parallel < 8:
            raise ValueError("Capacity planning requires at least eight slots")
        capacity.update(host_memory_budget_mib=memory_mib(memory_request),
                        host_memory_limit_mib=memory_mib(memory_limit), cpu_limit=float(cpu_limit))
    tasks = config.get("tasks") or ([config["task"]] if "task" in config else [])
    if config.get("mode", "run") != "run" or not isinstance(tasks, list) or not tasks:
        raise ValueError("source must be a run configuration with a nonempty task list")
    if any(not isinstance(task, dict) or not isinstance(task.get("task"), str) or not task["task"] for task in tasks):
        raise ValueError("each task must have a task name")
    library = config.get("library", str(project / "library"))
    if not isinstance(library, str) or not Path(library).is_absolute():
        raise ValueError("shared library must be an absolute path")
    histories = config.get("retry_experiences", [])
    if isinstance(histories, str):
        histories = [histories]
    if not isinstance(histories, list) or any(not isinstance(path, str) for path in histories):
        raise ValueError("retry_experiences must be a directory or list of directories")
    previous_tasks = [history_task(path) for path in histories]
    job = json.loads((runtime / "deployment/job_template.json").read_text())
    for path in (project / "evo", project / "scripts", project / "pyproject.toml"):
        if not path.exists():
            raise ValueError(f"Missing source input: {path}")
    run = project / "runs" / name
    run.mkdir(parents=True, exist_ok=False)
    snapshot = run / "source"
    for directory in ("evo", "scripts"):
        shutil.copytree(project / directory, snapshot / directory,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy(project / "pyproject.toml", snapshot / "pyproject.toml")
    hashes = {str(path.relative_to(snapshot)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(snapshot.rglob("*")) if path.is_file()}
    write(run / "source_sha256.json", hashes)
    write(run / "config.json", config)
    workers = []
    for index, task in enumerate(tasks):
        directory = run / "workers" / f"task_{index + 1:0{max(2, len(str(len(tasks))))}d}"
        directory.mkdir(parents=True)
        child = copy.deepcopy(config)
        child.pop("task", None)
        child.update(tasks=[copy.deepcopy(task)], library=library)
        if "retry_experiences" in config:
            child["retry_experiences"] = [path for path, old_task in previous_tasks if old_task == task]
        write(directory / "config.json", child)
        workers.append({"index": index, "task": task, "gpu_slot": index % 8,
                        "run": str(directory), "config": str(directory / "config.json"),
                        "log": str(directory / "worker.log")})
    plan = {"name": name, "job": "yumuzhen-" + name, "run": str(run), "source": str(snapshot),
            "source_config": str(config_path), "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "shared_library": library, "gpu_requested": 8, "preflight": preflight,
            "max_parallel": max_parallel,
            "gpu_assignment_policy": "A free dispatch slot selects slot_index % 8; workers[].gpu_slot is an initial hint only. Actual assignments are recorded in task_receipt.json and gpu_assignment.json.",
            "completion_mode": "all_attempted" if config.get("episode_retry", {}).get("max_episodes") == 1 else "all_success",
            "preflight_scene_workers": 8 if capacity.get("enabled") else min(2, len(tasks)),
            "capacity_planning": capacity,
            "low_utilization_waiver": ("User explicitly accepts low utilization for this run"
                if config.get("gpu_execution", {}).get("low_utilization_exception_explicitly_accepted_by_user") else None),
            "workers": workers}
    plan_path = run / "parallel_plan.json"
    write(plan_path, plan)
    labels = dict(owner="yumuzhen", project="atomic-evo", purpose="gpu-parallel", queue="shared")
    job["metadata"].update(name=plan["job"], labels=labels)
    job["spec"].update(backoffLimit=0, ttlSecondsAfterFinished=604800, parallelism=1, completions=1)
    if deadline:
        job["spec"]["activeDeadlineSeconds"] = deadline
    else:
        job["spec"].pop("activeDeadlineSeconds", None)
    pod = job["spec"]["template"]
    pod["metadata"]["labels"] = labels.copy()
    for key in ("nodeName", "nodeSelector", "affinity", "priorityClassName", "priority",
                "preemptionPolicy", "schedulerName"):
        pod["spec"].pop(key, None)
    pod["spec"].update(restartPolicy="Never", terminationGracePeriodSeconds=45)
    container = pod["spec"]["containers"][0]
    excluded = {"CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "ROBOENV_GPU", "MUJOCO_EGL_DEVICE_ID"}
    env = {entry["name"]: entry["value"] for entry in container.get("env", [])
           if "value" in entry and not entry["name"].startswith("RSI_") and entry["name"] not in excluded}
    env.update(PYTHONPATH=":".join([str(snapshot), str(runtime / "source/RoboEnv/src"),
        str(runtime / "source/RoboEnv/runtime/src"), str(runtime / "runtime/python")]),
        MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", NVIDIA_DRIVER_CAPABILITIES="compute,utility,graphics,display",
        EVO_RUN=str(run), EVO_RUNTIME=str(runtime), EVO_PARALLEL_PLAN=str(plan_path))
    for variable in ("LP_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[variable] = "1"
    container["env"] = [dict(name=key, value=value) for key, value in env.items()] + [
        {"name": "EVO_NODE_NAME", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}}]
    container["command"] = ["/opt/envs/dreamzero/bin/python", "-u", str(snapshot / "scripts/parallel_worker.py"),
                            "--plan", str(plan_path)]
    container["resources"] = {
        "requests": {"cpu": str(cpu_request), "memory": memory_request, "nvidia.com/gpu": "8", "ephemeral-storage": "4Gi"},
        "limits": {"cpu": str(cpu_limit), "memory": memory_limit, "nvidia.com/gpu": "8", "ephemeral-storage": "32Gi"}}
    for volume in pod["spec"]["volumes"]:
        if volume["name"] == "project":
            volume["hostPath"]["path"] = str(project)
        if volume["name"] == "shm":
            volume["emptyDir"]["sizeLimit"] = "16Gi"
    for mount in container["volumeMounts"]:
        if mount["name"] == "project":
            mount["mountPath"] = str(project)
    manifest = run / "job.json"
    write(manifest, job)
    return plan, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--project", default="/mnt/auwomo-data/yumuzhen/my/evo")
    parser.add_argument("--runtime", default="/mnt/auwomo-data/yumuzhen/physical-rsi")
    parser.add_argument("--deadline", type=int, default=0)
    parser.add_argument("--cpu-request", default="64")
    parser.add_argument("--cpu-limit", default="128")
    parser.add_argument("--memory-request", default="192Gi")
    parser.add_argument("--memory-limit", default="384Gi")
    parser.add_argument("--max-parallel", type=int, help="one to sixty-four concurrent tasks; capacity planning can reduce this ceiling")
    parser.add_argument("--preflight", action="store_true", help="eight EGL probes and two real scene probes, then exit")
    parser.add_argument("--dry-run", action="store_true", help="freeze source/config/manifests; no Job submission; consumes run name")
    args = parser.parse_args(argv)
    try:
        plan, manifest = prepare(args.config, args.name, args.project, args.runtime, deadline=args.deadline,
            preflight=args.preflight, cpu_request=args.cpu_request, cpu_limit=args.cpu_limit,
            memory_request=args.memory_request, memory_limit=args.memory_limit, max_parallel=args.max_parallel)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(dict(job=plan["job"], run=plan["run"],
        plan=str(Path(plan["run"]) / "parallel_plan.json"), tasks=len(plan["workers"]),
        max_parallel=plan["max_parallel"], completion_mode=plan["completion_mode"],
        manifest=str(manifest), dry_run=args.dry_run), ensure_ascii=False), flush=True)
    if not args.dry_run:
        subprocess.run(["kubectl", "--kubeconfig", "/root/.kube/p6000-user.conf", "-n", "dreamzero-training",
                        "--request-timeout=30s", "create", "-f", str(manifest)], check=True, timeout=40)


if __name__ == "__main__":
    main()
