"""Supervise isolated Evo workers inside one eight-GPU Pod; never create dummy load."""
import argparse
import csv
import fcntl
import importlib
import json
import os
from pathlib import Path
import runpy
import resource
import signal
import subprocess
import sys
import time
import traceback
import uuid


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def event(root, name, **values):
    record = dict(event=name, time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **values)
    with (Path(root) / "supervisor_events.jsonl").open("a") as log:
        log.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def discover_devices():
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id,memory.total",
                                   "--format=csv,noheader,nounits"], text=True, timeout=20)
    keys = ("index", "name", "uuid", "pci_bus_id", "memory_mib")
    devices = [dict(zip(keys, (value.strip() for value in row))) for row in csv.reader(raw.splitlines()) if row]
    if len(devices) != 8 or len({device.get("uuid") for device in devices}) != 8:
        raise RuntimeError("This Pod must expose exactly eight distinct NVIDIA GPU UUIDs")
    if any(not device.get("name") or not device.get("uuid", "").startswith("GPU-") for device in devices):
        raise RuntimeError("GPU model/UUID inventory is incomplete")
    return sorted(devices, key=lambda device: int(device["index"]))


def worker_environment(parent, directory, assignment):
    env = dict(parent)
    for variable in ("ROBOENV_GPU", "MUJOCO_EGL_DEVICE_ID"):
        env.pop(variable, None)
    scratch = Path(directory) / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    for variable, suffix in (("TMPDIR", "tmp"), ("ROBOCASA_TEMP_DIR", "robocasa"), ("NUMBA_CACHE_DIR", "numba")):
        path = scratch / suffix
        path.mkdir(exist_ok=True)
        env[variable] = str(path)
    env.update(EVO_RUN=str(directory), EVO_GPU_ASSIGNMENT=str(Path(directory) / "gpu_assignment.json"),
               CUDA_VISIBLE_DEVICES=assignment["uuid"], MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    return env


def child_command(plan_path, index, preflight=False, scene=False):
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--plan", str(plan_path),
               "--child", "--worker-index", str(index)]
    if preflight:
        command.append("--probe-only")
    if scene:
        command.append("--scene")
    return command


def _campaign_progress(directory):
    path = Path(directory) / "campaign/summary.json"
    if not path.is_file():
        return None
    try:
        report = json.loads(path.read_text())
        progress = {key: report[key] for key in ("status", "current_task_index", "current_episode_attempt",
                "total_episode_attempts", "native_successes", "program_successes", "promoted_versions") if key in report}
        if report.get("episodes"):
            task = report["episodes"][0]
            progress.update(task_retry_status=task.get("retry_status"),
                            completed_episode_attempts=len(task.get("episode_attempts", [])))
        return progress
    except (OSError, ValueError):
        return {"status": "summary_temporarily_unreadable"}


def _signal_process(process, signum):
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _task_record(worker):
    return {"index": worker["index"], "task": worker["task"], "run": worker["run"],
            "log": worker["log"], "status": "pending", "probe_only": False}


def _save_task(record):
    write(Path(record["run"]) / "task_receipt.json", record)


def _owned_worker_alive(record, plan_path):
    """A reused PID on another node is not proof that our old worker is alive."""
    pid = record.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        proc = Path("/proc") / str(pid)
        environment = proc.joinpath("environ").read_bytes().split(b"\0")
        command = [value.decode() for value in proc.joinpath("cmdline").read_bytes().split(b"\0") if value]
        return (f"EVO_RUN={record['run']}".encode() in environment and
                "--plan" in command and command[command.index("--plan") + 1] == str(plan_path) and
                "--worker-index" in command and
                command[command.index("--worker-index") + 1] == str(record["index"]) and
                os.getpgid(pid) == pid)
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False


def _assert_no_orphan_workers(plan, plan_path):
    """Catch a child spawned before its PID reached the durable task receipt."""
    runs = {worker["run"] for worker in plan["workers"]}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = [value.decode(errors="replace") for value in proc.joinpath("cmdline").read_bytes().split(b"\0") if value]
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if "--child" not in command or "--plan" not in command:
            continue
        try:
            if command[command.index("--plan") + 1] != str(plan_path):
                continue
            environment = proc.joinpath("environ").read_bytes().split(b"\0")
            if any(f"EVO_RUN={run}".encode() in environment for run in runs) and os.getpgid(int(proc.name)) == int(proc.name):
                raise RuntimeError("A previous worker is still alive; refuse to duplicate this campaign queue (pid " + proc.name + ")")
        except (FileNotFoundError, ProcessLookupError, IndexError):
            continue


def _restore_task(worker, plan_path=None):
    """Never replay an episode after a supervisor restart, including partial episodes."""
    record = _task_record(worker)
    path = Path(worker["run"]) / "task_receipt.json"
    if path.is_file():
        previous = json.loads(path.read_text())
        if any(previous.get(key) != record[key] for key in ("index", "task", "run")):
            raise ValueError("Task receipt does not match the frozen plan: " + str(path))
        record.update(previous)
        if plan_path is not None and _owned_worker_alive(record, plan_path):
            raise RuntimeError("A previous worker is still alive; stop that owned process before resuming: " + str(path))
    campaign = _campaign_progress(worker["run"])
    if record["status"] in {"starting", "running"} or (
            record["status"] == "pending" and (Path(worker["run"]) / "campaign").exists()):
        # A finished campaign can outlive its parent's final ledger write. Every
        # other started output is interrupted and must not replay its episode.
        terminal = campaign and campaign.get("status") not in {None, "running", "summary_temporarily_unreadable"}
        record.update(status=("succeeded" if campaign["status"] == "complete" else "failed")
                      if terminal else "interrupted", campaign=campaign,
                      recovered_at_utc=_now(), resume_policy="started_tasks_are_not_replayed")
        _save_task(record)
    return record


def _counts(records):
    counts = {"planned": len(records), "pending": 0, "running": 0, "finished": 0,
              "succeeded": 0, "failed": 0, "interrupted": 0, "cancelled": 0}
    for record in records:
        status = record["status"]
        if status == "pending":
            counts["pending"] += 1
        elif status in {"starting", "running"}:
            counts["running"] += 1
        else:
            counts["finished"] += 1
            counts[status] = counts.get(status, 0) + 1
    return counts


def supervise(plan_path, poll_seconds=1):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    root = Path(plan["run"])
    max_parallel = plan.get("max_parallel", 16)
    if not isinstance(max_parallel, int) or isinstance(max_parallel, bool) or not 1 <= max_parallel <= 64:
        raise ValueError("max_parallel must be between one and sixty-four (at most eight workers per GPU)")
    supervisor_lock = (root / ".supervisor.lock").open("a")
    try:
        fcntl.flock(supervisor_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        supervisor_lock.close()
        raise RuntimeError("Another supervisor already owns this campaign queue")
    # All children of this immutable runtime share a cache epoch. A supervisor
    # restart always creates a new epoch; old-run proofs cannot skip new checks.
    replay_cache_session = uuid.uuid4().hex
    receipt = {"status": "initializing", "job": plan["job"], "preflight": plan["preflight"],
               "replay_cache_session": replay_cache_session,
               "node": os.environ.get("EVO_NODE_NAME"), "gpu_requested": 8, "workers": [],
               "max_parallel": max_parallel, "completion_mode": plan.get("completion_mode", "all_success"),
               "low_utilization_waiver": plan.get("low_utilization_waiver")}
    receipt_path = root / "parallel_receipt.json"
    processes, cancelled, previous_handlers = [], [], {}

    def stop(signum, _frame):
        if not cancelled:
            cancelled.append(signum)

    def start(record, directory, task_index, slot_index, probe_only=False, scene=False):
        gpu_slot = slot_index % 8
        assignment = dict(devices[gpu_slot], visible_gpu_slot=gpu_slot, slot_index=slot_index,
                          worker_index=task_index, cuda_logical_device=0,
                          cuda_visible_devices=devices[gpu_slot]["uuid"])
        directory.mkdir(parents=True, exist_ok=True)
        write(directory / "gpu_assignment.json", assignment)
        record.update(status="starting", gpu_slot=gpu_slot, gpu_uuid=assignment["uuid"],
                      slot_index=slot_index, started_at_utc=_now())
        if not probe_only:
            # Claim before spawn so a crash cannot silently duplicate an episode.
            record["spawn_count"] = 1
            _save_task(record)
        log = Path(record["log"]).open("a")
        try:
            process = subprocess.Popen(child_command(plan_path, task_index, probe_only, scene),
                env=worker_environment({**os.environ, "EVO_REPLAY_CACHE_SESSION": replay_cache_session},
                                       directory, assignment), stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        except Exception as error:
            log.close()
            record.update(status="failed", ended_at_utc=_now(),
                          start_error={"type": type(error).__name__, "message": str(error)})
            if not probe_only:
                _save_task(record)
            event(root, "worker_start_failed", **record)
            return None
        record.update(status="running", pid=process.pid)
        if not probe_only:
            _save_task(record)
        item = {"process": process, "record": record, "log": log}
        processes.append(item)
        event(root, "worker_start", **record)
        return item

    def refresh(active, stop_started):
        for item in active:
            process, record = item["process"], item["record"]
            code = process.poll()
            if code is None and stop_started is not None and time.monotonic() - stop_started > 25:
                _signal_process(process, signal.SIGKILL)
            record["campaign"] = _campaign_progress(record["run"])
            proof = Path(record["run"]) / ("gpu_preflight.json" if record["probe_only"] else "gpu_binding.json")
            if proof.is_file():
                try:
                    record["gpu_evidence"] = json.loads(proof.read_text())
                except (OSError, ValueError):
                    pass
            if code is not None and record["status"] == "running":
                record.update(returncode=code, status="cancelled" if cancelled else "succeeded" if code == 0 else "failed",
                              ended_at_utc=_now())
                if not record["probe_only"]:
                    _save_task(record)
                item["log"].close()
                event(root, "worker_end", **record)
        return [item for item in active if item["record"]["status"] == "running"]

    def checkpoint():
        receipt["updated_at_utc"] = _now()
        receipt["counts"] = _counts(receipt["workers"])
        try:
            metric_workers = [{key: worker[key] for key in
                ("index", "task", "gpu_slot", "gpu_uuid", "run", "pid", "status", "campaign") if key in worker}
                for worker in (receipt.get("workers") or receipt.get("gpu_probes", []))
                if worker["status"] == "running"]
            metrics = sampler.maybe_sample(workers=metric_workers)
            if metrics is not None:
                receipt["gpu_metrics"] = metrics
        except Exception as error:
            receipt["gpu_metrics_error"] = {"type": type(error).__name__, "message": str(error)}
        write(receipt_path, receipt)

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.signal(signum, stop)
    try:
        try:
            from gpu_metrics import GPUMetricsSampler
        except ModuleNotFoundError:
            from scripts.gpu_metrics import GPUMetricsSampler
        sampler = GPUMetricsSampler(root, expected_gpu_count=8, interval_s=30,
                                    waiver=receipt["low_utilization_waiver"])
        _assert_no_orphan_workers(plan, plan_path)
        devices = discover_devices()
        receipt["devices"] = devices
        write(root / "devices.json", dict(node=receipt["node"], devices=devices, gpu_requested=8))
        receipt["status"] = "checking_gpus"
        receipt["gpu_probes"] = []
        active = []
        for index in range(8):
            worker = plan["workers"][index % len(plan["workers"])]
            directory = root / "gpu_probes" / f"gpu_{index:02d}"
            record = {"index": index, "task": worker["task"], "run": str(directory), "status": "pending",
                      "log": str(directory / "worker.log"), "probe_only": True}
            receipt["gpu_probes"].append(record)
            item = start(record, directory, index, index, True,
                         index < plan["preflight_scene_workers"]
                         if plan["preflight"] or plan.get("capacity_planning", {}).get("enabled") else False)
            if item:
                active.append(item)
        stop_started = None
        while active:
            if cancelled and stop_started is None:
                stop_started = time.monotonic()
                event(root, "stop_requested", signal=cancelled[0])
                for item in active:
                    _signal_process(item["process"], signal.SIGTERM)
            active = refresh(active, stop_started)
            checkpoint()
            if active:
                time.sleep(poll_seconds)
        probes_passed = all(record["status"] == "succeeded" for record in receipt["gpu_probes"])
        if probes_passed and not cancelled and plan.get("capacity_planning", {}).get("enabled"):
            try:
                from gpu_capacity import choose_capacity
            except ModuleNotFoundError:
                from scripts.gpu_capacity import choose_capacity
            capacity = choose_capacity(devices, [record["gpu_evidence"] for record in receipt["gpu_probes"]],
                                       max_parallel, plan["capacity_planning"])
            max_parallel = capacity["max_parallel"]
            receipt.update(max_parallel=max_parallel, capacity=capacity)
            write(root / "capacity_plan.json", capacity)
            event(root, "capacity_selected", max_parallel=max_parallel,
                  workers_per_gpu=capacity["workers_per_gpu"], host_peak_mib=capacity["peak_host_rss_mib"])
        if probes_passed and not cancelled and not plan["preflight"]:
            receipt["workers"] = [_restore_task(worker, plan_path) for worker in plan["workers"]]
            receipt["status"] = "running"
            active, stop_started = [], None
            while True:
                if cancelled and stop_started is None:
                    stop_started = time.monotonic()
                    event(root, "stop_requested", signal=cancelled[0])
                    for item in active:
                        _signal_process(item["process"], signal.SIGTERM)
                active = refresh(active, stop_started)
                if not cancelled:
                    occupied = {item["record"]["slot_index"] for item in active}
                    available = iter(slot for slot in range(max_parallel) if slot not in occupied)
                    for record in receipt["workers"]:
                        if record["status"] != "pending":
                            continue
                        slot_index = next(available, None)
                        if slot_index is None:
                            break
                        item = start(record, Path(record["run"]), record["index"], slot_index)
                        if item:
                            active.append(item)
                checkpoint()
                if not active and (cancelled or not any(record["status"] == "pending" for record in receipt["workers"])):
                    break
                time.sleep(poll_seconds)
        if cancelled:
            receipt["status"] = "cancelled"
        elif not probes_passed:
            receipt["status"] = "workers_failed"
        elif plan["preflight"]:
            receipt["status"] = "complete"
        elif all(record["status"] == "succeeded" for record in receipt["workers"]):
            receipt["status"] = "complete"
        elif receipt["completion_mode"] == "all_attempted":
            # Queue drained is distinct from all native episodes succeeding or
            # initializing. The campaign summary audits those outcomes separately.
            receipt["status"] = "complete_with_failures"
        else:
            receipt["status"] = "workers_failed"
    except Exception as error:
        receipt.update(status="error", error={"type": type(error).__name__, "message": str(error)})
        event(root, "supervisor_error", error=receipt["error"])
    finally:
        for item in processes:
            if item["process"].poll() is None:
                _signal_process(item["process"], signal.SIGTERM)
                try:
                    item["process"].wait(timeout=25)
                except subprocess.TimeoutExpired:
                    _signal_process(item["process"], signal.SIGKILL)
                    item["process"].wait(timeout=5)
                record = item["record"]
                record.update(status="interrupted", returncode=item["process"].returncode, ended_at_utc=_now())
                if not record["probe_only"]:
                    _save_task(record)
            item["log"].close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        receipt["counts"] = _counts(receipt["workers"])
        receipt["updated_at_utc"] = _now()
        write(receipt_path, receipt)
        if not plan["preflight"] and receipt["status"] in {"complete", "complete_with_failures", "workers_failed"}:
            try:
                if plan.get("mode") == "evaluate_policy":
                    try:
                        from summarize_policy_evaluation import summarize_evaluation, write_summary
                    except ModuleNotFoundError:
                        from scripts.summarize_policy_evaluation import summarize_evaluation, write_summary
                    summary = summarize_evaluation(plan_path, receipt_path=receipt_path)
                    write_summary(summary, root / "evaluation_summary.json")
                    receipt["final_summary"] = str(root / "evaluation_summary.json")
                    event(root, "evaluation_summary", status=summary.get("status"), counts=summary.get("counts", {}))
                else:
                    try:
                        from summarize_parallel_campaign import summarize_campaign, write_summary
                    except ModuleNotFoundError:
                        from scripts.summarize_parallel_campaign import summarize_campaign, write_summary
                    summary = summarize_campaign(plan_path, receipt_path=receipt_path)
                    write_summary(summary, root / "campaign_summary.json", root / "tool_report.json")
                    receipt["final_summary"] = str(root / "campaign_summary.json")
                    receipt["tool_report"] = str(root / "tool_report.json")
                    event(root, "campaign_summary", status=summary.get("status"),
                          counts=summary.get("counts", {}), tool_counts=summary.get("tools", {}).get("counts", {}))
            except Exception as error:
                receipt["summary_error"] = {"type": type(error).__name__, "message": str(error)}
                event(root, "summary_error", error=receipt["summary_error"])
            write(receipt_path, receipt)
        event(root, "supervisor_end", status=receipt["status"])
        fcntl.flock(supervisor_lock.fileno(), fcntl.LOCK_UN)
        supervisor_lock.close()
    return 0 if receipt["status"] in {"complete", "complete_with_failures"} else 1


def install_renderer_mapping():
    """Use RoboEnv UUID/EGL discovery and the validated robosuite UUID-mask fix."""
    from roboenv.provisioning import gpu
    original = gpu.configure_gpu

    def configure(backend, options=None):
        selection = original(backend, options)
        if backend == "robocasa" and selection and "robosuite" not in sys.modules:
            mask = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(selection["egl_index"])
            try:
                importlib.import_module("robosuite")
            finally:
                if mask is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = mask
        return selection

    gpu.configure_gpu = configure


def renderer_evidence(assignment):
    from OpenGL.GL import glGetError, glGetString, GL_VENDOR, GL_RENDERER, GL_VERSION
    pending = []
    for _ in range(16):
        error = int(glGetError())
        if not error:
            break
        pending.append(error)
    else:
        raise RuntimeError("OpenGL error queue did not clear")
    binding = json.loads(os.environ.get("ROBOENV_GPU", "{}"))
    vendor = (glGetString(GL_VENDOR) or b"").decode()
    renderer = (glGetString(GL_RENDERER) or b"").decode()
    version = (glGetString(GL_VERSION) or b"").decode()
    if binding.get("uuid") != assignment["uuid"] or "NVIDIA" not in vendor or not renderer:
        raise RuntimeError("Actual EGL renderer/UUID does not match the worker's NVIDIA GPU assignment")
    return {"assignment": assignment, "roboenv_binding": binding, "gl_vendor": vendor,
            "gl_renderer": renderer, "gl_version": version, "gl_errors_before_diagnostic": pending,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID")}


def verify_fresh_scene(adapter):
    """Host-only validity check: backend reset clears its cached task verdict."""
    raw_success = bool(adapter._backend.env._check_success())
    if raw_success:
        raise RuntimeError("Fresh scene already satisfies the native task predicate; refuse to score or learn it")
    return {"raw_native_success_at_reset": raw_success, "native_steps_at_reset": adapter.stats()["steps"]}


def child_main(plan_path, index, probe_only=False, scene=False):
    plan = json.loads(Path(plan_path).read_text())
    root = Path(os.environ["EVO_RUN"])
    assignment = json.loads(Path(os.environ["EVO_GPU_ASSIGNMENT"]).read_text())
    report = {"status": "initializing", "assignment": assignment, "scene_probe": scene}
    adapter = None
    try:
        install_renderer_mapping()
        if probe_only:
            from roboenv.provisioning import gpu
            gpu.configure_gpu("robocasa")
            from mujoco.egl import GLContext
            context = GLContext(32, 32)
            try:
                context.make_current()
                report["renderer"] = renderer_evidence(assignment)
            finally:
                context.free()
            if scene:
                from evo.cli import adapter_factory
                policy = plan.get("capacity_planning", {})
                def gpu_memory():
                    from scripts.gpu_metrics import query_gpus
                    return next(gpu["memory_used_mib"] for gpu in query_gpus() if gpu["uuid"] == assignment["uuid"])
                baseline_gpu = gpu_memory() if policy.get("enabled") else None
                task = policy.get("probe_task") or plan["workers"][index % len(plan["workers"])]["task"]
                adapter = adapter_factory(task, root / "simulation")
                report["reset_validity"] = verify_fresh_scene(adapter)
                initial = adapter.observe()
                before = adapter.stats()
                report["scene_renderer"] = renderer_evidence(assignment)
                adapter.wait(.05)
                after = adapter.stats()
                if len(initial.get("cameras", {})) < 3 or after["steps"] <= before["steps"]:
                    raise RuntimeError("Real scene probe must render three RGB-D cameras and advance native physics")
                report.update(task=task, before=before, after=after, images=initial["images"])
                if policy.get("enabled"):
                    time.sleep(.2)
                    one_gpu = gpu_memory()
                    primary_robot = adapter.observe()["robot"]
                    primary_steps = adapter.stats()["steps"]
                    replay = adapter_factory(policy.get("replay_probe_task", task), root / "replay_simulation")
                    try:
                        verify_fresh_scene(replay)
                        replay.observe()
                        replay.wait(.05)
                        renderer_evidence(assignment)
                        time.sleep(.2)
                        two_gpu = gpu_memory()
                        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                    finally:
                        replay.close()
                    resumed = adapter.observe()
                    if resumed["robot"] != primary_robot or adapter.stats()["steps"] != primary_steps:
                        raise RuntimeError("Independent replay changed the paused exploration state")
                    adapter.wait(.05)
                    adapter.observe()
                    renderer_evidence(assignment)
                    report["capacity"] = {"gpu_uuid": assignment["uuid"], "baseline_gpu_mib": baseline_gpu,
                        "one_scene_gpu_mib": one_gpu, "two_scene_gpu_mib": two_gpu,
                        "peak_host_rss_mib": peak_rss, "dual_environment_verified": True}
            report["status"] = "passed"
            write(root / "gpu_preflight.json", report)
            print(json.dumps(report), flush=True)
            return 0
        import evo.cli as cli
        original_factory = cli.adapter_factory

        def tracked_factory(config, output):
            current = original_factory(config, output)
            try:
                reset_validity = verify_fresh_scene(current)
                current.observe()
                evidence = renderer_evidence(assignment)
                evidence.update(simulation=str(output), native_steps=current.stats()["steps"], **reset_validity)
                write(root / "gpu_binding.json", evidence)
                with (root / "gpu_bindings.jsonl").open("a") as log:
                    log.write(json.dumps(evidence) + "\n")
                return current
            except Exception:
                current.close()
                raise

        cli.adapter_factory = tracked_factory
        runpy.run_path(str(Path(plan["source"]) / "scripts/worker.py"), run_name="__main__")
        return 0
    except SystemExit as error:
        return error.code if isinstance(error.code, int) else 1
    except Exception as error:
        report.update(status="error", error={"type": type(error).__name__, "message": str(error)},
                      traceback=traceback.format_exc())
        write(root / ("gpu_preflight.json" if probe_only else "worker_error.json"), report)
        traceback.print_exc()
        return 1
    finally:
        if adapter is not None:
            adapter.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--scene", action="store_true")
    args = parser.parse_args(argv)
    return (child_main(args.plan, args.worker_index, args.probe_only, args.scene) if args.child
            else supervise(args.plan))


if __name__ == "__main__":
    raise SystemExit(main())
