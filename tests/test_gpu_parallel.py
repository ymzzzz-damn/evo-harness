"""GPU manifest, reset validity, and real subprocess supervision without GPU access."""
from collections import Counter
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load_script("launch_gpu_parallel_p6000")
supervisor = load_script("parallel_worker")


class GPUParallelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project, self.runtime = self.root / "project", self.root / "runtime"
        for directory in (self.project / "evo", self.project / "scripts", self.runtime / "deployment"):
            directory.mkdir(parents=True)
        (self.project / "evo/__init__.py").write_text("")
        (self.project / "scripts/worker.py").write_text("raise SystemExit(0)\n")
        (self.project / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='0.0.0'\n")
        pod = {"nodeName": "forbidden-node", "nodeSelector": {"gpu": "fixed"}, "priorityClassName": "custom",
               "preemptionPolicy": "PreemptLowerPriority", "schedulerName": "custom-scheduler", "affinity": {},
               "imagePullSecrets": [{"name": "registry-secret"}], "containers": [{"name": "worker", "image": "image@sha256:fixture",
                   "env": [{"name": "MUJOCO_GL", "value": "osmesa"}, {"name": "CUDA_VISIBLE_DEVICES", "value": "stale"},
                           {"name": "ROBOENV_GPU", "value": "stale"}], "volumeMounts": [
                       {"name": "shared", "mountPath": "/mnt/shared", "readOnly": True},
                       {"name": "assets", "mountPath": "/assets", "readOnly": True},
                       {"name": "project", "mountPath": "/old-project"}, {"name": "shm", "mountPath": "/dev/shm"}]}],
               "volumes": [{"name": "project", "hostPath": {"path": "/old-project"}},
                           {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "2Gi"}}]}
        self.write(self.runtime / "deployment/job_template.json", {"apiVersion": "batch/v1", "kind": "Job",
            "metadata": {"namespace": "dreamzero-training"}, "spec": {"activeDeadlineSeconds": 7200,
            "template": {"metadata": {}, "spec": pod}}})
        self.tasks = [{"task": f"Task{index}", "seed": 0, "camera_size": [512, 512]} for index in range(16)]
        histories = []
        for index in range(4):
            path = self.root / f"history{index}"
            path.mkdir()
            self.write(path / "summary.json", {"task_config": self.tasks[index]})
            histories.append(str(path))
        self.config = {"tasks": self.tasks, "library": str(self.root / "library"),
            "agent": {"provider": "http", "model": "fixture-model", "reasoning_effort": "high"},
            "exploration": {"max_attempts": 6, "max_replans": 1, "code_timeout_s": 180},
            "maintenance": {"max_candidates": 2}, "maintain_after_task": True,
            "episode_retry": {"max_episodes": None, "history_episodes": 3}, "retry_experiences": histories}
        self.config_path = self.root / "config.json"
        self.write(self.config_path, self.config)
        self.devices = [{"index": str(index), "name": "Fixture NVIDIA GPU", "uuid": f"GPU-{index}",
                         "pci_bus_id": f"0000:{index:02d}:00.0", "memory_mib": "32000"} for index in range(8)]

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))

    def prepare(self, preflight=False):
        plan, manifest = launcher.prepare(self.config_path, "evo-gpu-test", self.project, self.runtime,
                                           deadline=0, preflight=preflight)
        return plan, manifest, Path(plan["run"]) / "parallel_plan.json"

    def test_policy_evaluation_uses_frozen_training_source_and_settles_failed_trials(self):
        import shutil
        frozen = self.root / "training_source"
        frozen.mkdir()
        for name in ("evo", "scripts"):
            shutil.copytree(self.project / name, frozen / name)
        shutil.copyfile(self.project / "pyproject.toml", frozen / "pyproject.toml")
        (frozen / "evo/__init__.py").write_text("FROZEN_TRAINING_ALGORITHM = True\n")
        (self.project / "evo/__init__.py").write_text("LATER_CANONICAL_CHANGE = True\n")
        self.config.update(mode="evaluate_policy", episode_retry={"max_episodes": 3, "history_episodes": 3})
        self.config.pop("retry_experiences", None)
        self.write(self.config_path, self.config)
        plan, _ = launcher.prepare(self.config_path, "evo-policy-eval", self.project, self.runtime,
                                   source_root=frozen)
        self.assertEqual(plan["mode"], "evaluate_policy")
        self.assertEqual(plan["completion_mode"], "all_attempted")
        self.assertEqual(plan["source_origin"], str(frozen))
        self.assertEqual((Path(plan["source"]) / "evo/__init__.py").read_text(),
                         "FROZEN_TRAINING_ALGORITHM = True\n")
        child = json.loads(Path(plan["workers"][0]["config"]).read_text())
        self.assertEqual(child["episode_retry"]["max_episodes"], 3)
        self.assertEqual(child["mode"], "evaluate_policy")

    def test_training_handoff_runs_wrapper_and_preserves_one_episode_budget(self):
        self.config.update(mode="run", episode_retry={"max_episodes": 1},
            evaluation_handoff={"template": str(self.root / "evaluation_template.json"),
                                "name": "evo-policy-eval"})
        self.config.pop("retry_experiences", None)
        self.write(self.config_path, self.config)
        plan, manifest, _ = self.prepare()
        command = json.loads(manifest.read_text())["spec"]["template"]["spec"]["containers"][0]["command"]
        self.assertEqual(command[2], str(Path(plan["source"]) / "scripts/train_and_prepare_evaluation.py"))
        self.assertEqual(plan["completion_mode"], "all_attempted")
        self.assertEqual(plan["evaluation_handoff"], self.config["evaluation_handoff"])
        child = json.loads(Path(plan["workers"][0]["config"]).read_text())
        self.assertEqual(child["episode_retry"]["max_episodes"], 1)

    def test_one_eight_gpu_pod_with_sixteen_preserved_task_configs_and_one_snapshot(self):
        plan, manifest, _ = self.prepare()
        job = json.loads(manifest.read_text())
        self.assertNotIn("activeDeadlineSeconds", job["spec"])
        self.assertEqual(job["spec"]["parallelism"], 1)
        pod = job["spec"]["template"]["spec"]
        for field in ("nodeName", "nodeSelector", "affinity", "priorityClassName", "preemptionPolicy", "schedulerName"):
            self.assertNotIn(field, pod)
        container = pod["containers"][0]
        self.assertEqual(container["resources"]["requests"]["nvidia.com/gpu"], "8")
        self.assertEqual(container["resources"]["limits"]["nvidia.com/gpu"], "8")
        self.assertEqual(container["resources"]["requests"]["cpu"], "64")
        self.assertEqual(container["resources"]["limits"]["memory"], "384Gi")
        self.assertEqual(container["image"], "image@sha256:fixture")
        self.assertEqual(pod["imagePullSecrets"], [{"name": "registry-secret"}])
        env = {value["name"]: value.get("value") for value in container["env"]}
        self.assertEqual(env["MUJOCO_GL"], "egl")
        self.assertEqual(env["PYOPENGL_PLATFORM"], "egl")
        self.assertNotIn("CUDA_VISIBLE_DEVICES", env)
        self.assertNotIn("ROBOENV_GPU", env)
        self.assertEqual(Counter(worker["gpu_slot"] for worker in plan["workers"]), dict.fromkeys(range(8), 2))
        self.assertEqual(len({worker["run"] for worker in plan["workers"]}), 16)
        self.assertTrue((Path(plan["source"]) / "evo/__init__.py").is_file())
        for index, worker in enumerate(plan["workers"]):
            child = json.loads(Path(worker["config"]).read_text())
            self.assertEqual(child["tasks"], [self.tasks[index]])
            for key in ("library", "agent", "exploration", "maintenance", "maintain_after_task", "episode_retry"):
                self.assertEqual(child[key], self.config[key])
            self.assertEqual(child["retry_experiences"], self.config["retry_experiences"][index:index + 1] if index < 4 else [])
            self.assertFalse((Path(worker["run"]) / "source").exists())

    def dummy_script(self):
        path = self.root / "dummy_worker.py"
        path.write_text('''import json, os, sys, time
from pathlib import Path
index, probe = int(sys.argv[1]), sys.argv[2] == '1'
root = Path(os.environ['EVO_RUN'])
assignment = json.loads(Path(os.environ['EVO_GPU_ASSIGNMENT']).read_text())
assert os.environ['CUDA_VISIBLE_DEVICES'] == assignment['uuid']
assert 'ROBOENV_GPU' not in os.environ and 'MUJOCO_EGL_DEVICE_ID' not in os.environ
assert os.environ['MUJOCO_GL'] == 'egl'
if probe:
    (root/'gpu_preflight.json').write_text(json.dumps({'status':'passed'}))
    raise SystemExit(5 if index == int(os.environ.get('FAIL_PROBE', '-1')) else 0)
shared = Path(os.environ['EVO_PARALLEL_PLAN']).parent
barrier = shared/'barrier'
barrier.mkdir(exist_ok=True)
(barrier/str(index)).write_text('started')
end = time.monotonic()+8
while len(list(barrier.iterdir())) < 16:
    if time.monotonic() > end:
        raise RuntimeError('workers did not run concurrently')
    time.sleep(.01)
time.sleep(.05)
(root/'campaign').mkdir()
(root/'campaign/summary.json').write_text(json.dumps({'status':'complete','current_episode_attempt':2}))
raise SystemExit(3 if index == 0 else 0)
''')
        return path

    def supervise_fixture(self, plan_path, fail_probe=None):
        dummy = self.dummy_script()
        self.spawned = []
        metrics_module = ModuleType("gpu_metrics")
        self.sampler = Mock()
        self.sampler.maybe_sample.return_value = {"fixture_sample": True}
        metrics_module.GPUMetricsSampler = Mock(return_value=self.sampler)
        env = {"EVO_PARALLEL_PLAN": str(plan_path), "ROBOENV_GPU": "inherited-stale", "MUJOCO_EGL_DEVICE_ID": "7"}
        if fail_probe is not None:
            env["FAIL_PROBE"] = str(fail_probe)
        def command(_plan, index, probe, scene):
            self.spawned.append((index, probe, scene))
            return [sys.executable, str(dummy), str(index), "1" if probe else "0"]
        with patch.object(supervisor, "discover_devices", return_value=self.devices), \
             patch.object(supervisor, "child_command", side_effect=command), patch.dict(os.environ, env), \
             patch.dict(sys.modules, {"gpu_metrics": metrics_module}), \
             contextlib.redirect_stdout(io.StringIO()):
            return supervisor.supervise(plan_path, poll_seconds=.01)

    def test_eight_probes_precede_sixteen_concurrent_workers_and_one_failure_does_not_stop_peers(self):
        plan, _, path = self.prepare()
        self.assertEqual(self.supervise_fixture(path), 1)
        receipt = json.loads((Path(plan["run"]) / "parallel_receipt.json").read_text())
        self.assertEqual(receipt["status"], "workers_failed")
        self.assertEqual(receipt["gpu_metrics"], {"fixture_sample": True})
        self.assertTrue(self.sampler.maybe_sample.called)
        self.assertEqual(len(receipt["gpu_probes"]), 8)
        self.assertTrue(all(record["status"] == "succeeded" for record in receipt["gpu_probes"]))
        self.assertEqual(len(receipt["workers"]), 16)
        self.assertEqual(receipt["workers"][0]["status"], "failed")
        self.assertTrue(all(record["status"] == "succeeded" for record in receipt["workers"][1:]))
        self.assertTrue(all(record["campaign"]["current_episode_attempt"] == 2 for record in receipt["workers"]))
        self.assertEqual(Counter(record["gpu_uuid"] for record in receipt["workers"]), dict.fromkeys((f"GPU-{i}" for i in range(8)), 2))
        events = [json.loads(line) for line in (Path(plan["run"]) / "supervisor_events.jsonl").read_text().splitlines()]
        first_real_start = next(index for index, value in enumerate(events)
                                if value["event"] == "worker_start" and not value["probe_only"])
        self.assertEqual(sum(value["event"] == "worker_end" for value in events[:first_real_start]), 8)

    def test_failed_gpu_probe_prevents_formal_workers_from_starting(self):
        plan, _, path = self.prepare()
        self.assertEqual(self.supervise_fixture(path, fail_probe=3), 1)
        receipt = json.loads((Path(plan["run"]) / "parallel_receipt.json").read_text())
        self.assertEqual(receipt["status"], "workers_failed")
        self.assertEqual(receipt["workers"], [])
        self.assertFalse((Path(plan["run"]) / "barrier").exists())

    def test_preflight_probes_all_eight_gpus_but_only_two_real_scenes_then_exits(self):
        plan, _, path = self.prepare(preflight=True)
        self.assertEqual(self.supervise_fixture(path), 0)
        self.assertEqual(len(self.spawned), 8)
        self.assertEqual(self.spawned[:2], [(0, True, True), (1, True, True)])
        self.assertTrue(all(probe and not scene for _, probe, scene in self.spawned[2:]))
        receipt = json.loads((Path(plan["run"]) / "parallel_receipt.json").read_text())
        self.assertEqual(receipt["status"], "complete")
        self.assertEqual(receipt["workers"], [])

    def test_365_tasks_freeze_one_episode_and_use_sixteen_queue_slots(self):
        self.config.update(tasks=[{"task": f"Task{index}", "seed": 0} for index in range(365)],
                           episode_retry={"max_episodes": 1}, gpu_execution={"worker_count": 16})
        self.write(self.config_path, self.config)
        plan, _, _ = self.prepare()
        self.assertEqual(len(plan["workers"]), 365)
        self.assertEqual(plan["max_parallel"], 16)
        self.assertEqual(plan["completion_mode"], "all_attempted")
        self.assertEqual(Path(plan["workers"][0]["run"]).name, "task_001")
        self.assertEqual(Path(plan["workers"][-1]["run"]).name, "task_365")
        self.assertTrue(all(json.loads(Path(worker["config"]).read_text())["episode_retry"] ==
                            {"max_episodes": 1} for worker in plan["workers"]))

    def test_queue_continues_after_failure_and_never_exceeds_two_workers_per_gpu(self):
        self.config.update(tasks=[{"task": f"Task{index}", "seed": 0} for index in range(37)],
                           episode_retry={"max_episodes": 1})
        self.write(self.config_path, self.config)
        plan, _, path = self.prepare()
        self.assertEqual(self.supervise_fixture(path), 0)
        receipt = json.loads((Path(plan["run"]) / "parallel_receipt.json").read_text())
        self.assertEqual(receipt["status"], "complete_with_failures")
        self.assertEqual(receipt["counts"], {"planned": 37, "pending": 0, "running": 0,
            "finished": 37, "succeeded": 36, "failed": 1, "interrupted": 0, "cancelled": 0})
        self.assertEqual(Counter(index for index, probe, _ in self.spawned if not probe),
                         dict.fromkeys(range(37), 1))
        live, peak = {}, 0
        events = [json.loads(line) for line in (Path(plan["run"]) / "supervisor_events.jsonl").read_text().splitlines()]
        for value in events:
            if value.get("probe_only") or value["event"] not in {"worker_start", "worker_end"}:
                continue
            if value["event"] == "worker_start":
                live[value["index"]] = value["gpu_uuid"]
                self.assertLessEqual(len(live), 16)
                self.assertTrue(all(count <= 2 for count in Counter(live.values()).values()))
                peak = max(peak, len(live))
            else:
                live.pop(value["index"])
        self.assertEqual(peak, 16)
        self.assertEqual(live, {})
        # A queue restart still probes hardware, but never repeats an ended task.
        self.assertEqual(self.supervise_fixture(path), 0)
        self.assertEqual(len(self.spawned), 8)
        self.assertTrue(all(probe for _, probe, _ in self.spawned))

    def test_restart_marks_started_episode_interrupted_and_preserves_unstarted_queue(self):
        self.config.update(tasks=[{"task": f"Task{index}", "seed": 0} for index in range(37)],
                           episode_retry={"max_episodes": 1})
        self.write(self.config_path, self.config)
        plan, _, path = self.prepare()
        first = supervisor._task_record(plan["workers"][0])
        first.update(status="running", pid=999999, spawn_count=1)
        supervisor._save_task(first)
        # A child may finish before its supervisor saves the final ledger.
        second = Path(plan["workers"][1]["run"]) / "campaign"
        second.mkdir()
        self.write(second / "summary.json", {"status": "tasks_incomplete", "total_episode_attempts": 1})
        self.assertEqual(self.supervise_fixture(path), 0)
        receipt = json.loads((Path(plan["run"]) / "parallel_receipt.json").read_text())
        self.assertEqual(receipt["workers"][0]["status"], "interrupted")
        self.assertEqual(receipt["workers"][1]["status"], "failed")
        self.assertEqual(Counter(index for index, probe, _ in self.spawned if not probe),
                         dict.fromkeys(range(2, 37), 1))
        self.assertEqual(receipt["counts"]["finished"], 37)

    def test_rejects_more_than_sixty_four_slots_before_creating_run(self):
        with self.assertRaisesRegex(ValueError, "one and sixty-four"):
            launcher.prepare(self.config_path, "evo-gpu-test", self.project, self.runtime, max_parallel=65)
        self.assertFalse((self.project / "runs/evo-gpu-test").exists())

    def test_thirty_two_workers_share_eight_gpus_and_queue_drains_once(self):
        self.config.update(tasks=[{"task": f"Task{index}", "seed": 0} for index in range(65)],
                           episode_retry={"max_episodes": 1}, gpu_execution={"worker_count": 32})
        self.write(self.config_path, self.config)
        plan, _, path = self.prepare()
        self.assertEqual(self.supervise_fixture(path), 0)
        live, peak = {}, 0
        for line in (Path(plan["run"]) / "supervisor_events.jsonl").read_text().splitlines():
            record = json.loads(line)
            if record.get("probe_only"):
                continue
            if record["event"] == "worker_start":
                live[record["index"]] = record["gpu_uuid"]
                peak = max(peak, len(live))
                self.assertLessEqual(len(live), 32)
                self.assertTrue(all(count <= 4 for count in Counter(live.values()).values()))
            elif record["event"] == "worker_end":
                live.pop(record["index"])
        self.assertEqual(peak, 32)
        self.assertEqual(Counter(index for index, probe, _ in self.spawned if not probe), dict.fromkeys(range(65), 1))
        self.assertEqual(live, {})

    def test_memory_profile_selects_dispatch_limit_before_any_task_starts(self):
        from scripts.gpu_capacity import choose_capacity
        self.config.update(tasks=[{"task": f"Task{index}", "seed": 0} for index in range(65)],
                           episode_retry={"max_episodes": 1}, gpu_execution={"worker_count": 64},
                           capacity_planning={"enabled": True})
        self.write(self.config_path, self.config)
        plan, _, path = self.prepare()
        self.assertEqual(plan["preflight_scene_workers"], 8)
        self.assertEqual(plan["capacity_planning"]["host_memory_budget_mib"], 192 * 1024)
        with patch("scripts.gpu_capacity.choose_capacity", return_value={
                "max_parallel": 24, "workers_per_gpu": 3, "peak_host_rss_mib": 2100}) as select, \
             patch.dict(sys.modules, {"gpu_capacity": sys.modules["scripts.gpu_capacity"]}):
            # The CLI and package import paths must share the same mock, even
            # after the handoff tests have put scripts/ on sys.path.
            self.assertEqual(self.supervise_fixture(path), 0)
        self.assertEqual(select.call_args.args[2], 64)
        self.assertTrue(all(scene for _, probe, scene in self.spawned if probe))
        receipt = json.loads((Path(plan["run"]) / "parallel_receipt.json").read_text())
        self.assertEqual(receipt["max_parallel"], 24)
        self.assertEqual(receipt["counts"]["finished"], 65)
        self.assertTrue((Path(plan["run"]) / "capacity_plan.json").is_file())

    def test_restart_refuses_queue_when_verified_old_worker_is_still_alive(self):
        plan, _, path = self.prepare()
        worker = plan["workers"][0]
        old = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)",
            "--plan", str(path), "--worker-index", "0"],
            env=dict(os.environ, EVO_RUN=worker["run"]), start_new_session=True)
        self.addCleanup(lambda: old.poll() is None and old.kill())
        try:
            record = supervisor._task_record(worker)
            record.update(status="running", pid=old.pid, spawn_count=1)
            supervisor._save_task(record)
            self.assertTrue(supervisor._owned_worker_alive(record, path))
            self.assertFalse(supervisor._owned_worker_alive(dict(record, index=1), path))
            self.assertEqual(self.supervise_fixture(path), 1)
            receipt = json.loads((Path(plan["run"]) / "parallel_receipt.json").read_text())
            self.assertEqual(receipt["status"], "error")
            self.assertIn("previous worker is still alive", receipt["error"]["message"])
            self.assertTrue(all(probe for _, probe, _ in self.spawned))
            self.assertIsNone(old.poll())
        finally:
            old.terminate()
            old.wait(timeout=5)

    def test_orphan_spawned_before_pid_receipt_blocks_even_gpu_probes(self):
        plan, _, path = self.prepare()
        worker = plan["workers"][0]
        old = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)",
            "--plan", str(path), "--child", "--worker-index", "0"],
            env=dict(os.environ, EVO_RUN=worker["run"]), start_new_session=True)
        try:
            record = supervisor._task_record(worker)
            record.update(status="starting", spawn_count=1)
            supervisor._save_task(record)
            self.assertEqual(self.supervise_fixture(path), 1)
            receipt = json.loads((Path(plan["run"]) / "parallel_receipt.json").read_text())
            self.assertEqual(receipt["status"], "error")
            self.assertIn("previous worker is still alive", receipt["error"]["message"])
            self.assertEqual(self.spawned, [])
            self.assertIsNone(old.poll())
        finally:
            old.terminate()
            old.wait(timeout=5)

    def test_inventory_rejects_missing_or_duplicate_gpu_uuids(self):
        rows = "\n".join(f"{index}, GPU Model, GPU-0, 0000:00:00.0, 32000" for index in range(8))
        with patch.object(supervisor.subprocess, "check_output", return_value=rows):
            with self.assertRaisesRegex(RuntimeError, "eight distinct"):
                supervisor.discover_devices()

    def test_uuid_mask_is_temporarily_numeric_only_for_robosuite_import_and_is_restored(self):
        gpu = ModuleType("roboenv.provisioning.gpu")
        gpu.configure_gpu = lambda backend, options=None: {"egl_index": 5, "uuid": "GPU-assigned"}
        provisioning = ModuleType("roboenv.provisioning")
        provisioning.gpu = gpu
        with patch.dict(sys.modules, {"roboenv": ModuleType("roboenv"), "roboenv.provisioning": provisioning,
                                      "roboenv.provisioning.gpu": gpu}), \
             patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-assigned"}):
            with patch.object(supervisor.importlib, "import_module", side_effect=lambda _: self.assertEqual(
                    os.environ["CUDA_VISIBLE_DEVICES"], "5")):
                supervisor.install_renderer_mapping()
                gpu.configure_gpu("robocasa")
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-assigned")

    def test_raw_initial_success_is_rejected_even_when_backend_cached_success_is_false(self):
        adapter = SimpleNamespace(_backend=SimpleNamespace(env=SimpleNamespace(_check_success=lambda: True)),
                                  stats=lambda: {"steps": 0, "native_success": False})
        with self.assertRaisesRegex(RuntimeError, "already satisfies"):
            supervisor.verify_fresh_scene(adapter)
        adapter._backend.env._check_success = lambda: False
        self.assertEqual(supervisor.verify_fresh_scene(adapter),
                         {"raw_native_success_at_reset": False, "native_steps_at_reset": 0})

    def test_renderer_proof_rejects_software_rendering_despite_matching_configured_uuid(self):
        gl = ModuleType("OpenGL.GL")
        gl.GL_VENDOR, gl.GL_RENDERER, gl.GL_VERSION = 1, 2, 3
        gl.glGetError = lambda: 0
        gl.glGetString = lambda field: {1: b"Mesa", 2: b"llvmpipe", 3: b"4.5"}[field]
        with patch.dict(sys.modules, {"OpenGL": ModuleType("OpenGL"), "OpenGL.GL": gl}), \
             patch.dict(os.environ, {"ROBOENV_GPU": json.dumps({"uuid": "GPU-assigned"})}):
            with self.assertRaisesRegex(RuntimeError, "Actual EGL renderer/UUID"):
                supervisor.renderer_evidence({"uuid": "GPU-assigned"})


if __name__ == "__main__":
    unittest.main()
