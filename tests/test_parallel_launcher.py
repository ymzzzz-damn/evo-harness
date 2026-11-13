"""Parallel launch isolation and partial-submission receipts, without cluster access."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "parallel_launcher", Path(__file__).resolve().parents[1] / "scripts/launch_parallel_p6000.py")
parallel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parallel)


class ParallelLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project, self.runtime = self.root / "project", self.root / "runtime"
        for directory in (self.project / "evo", self.project / "scripts", self.runtime / "deployment"):
            directory.mkdir(parents=True)
        (self.project / "evo/__init__.py").write_text("")
        (self.project / "scripts/worker.py").write_text("# fixture\n")
        (self.project / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='0.0.0'\n")
        template = {"apiVersion": "batch/v1", "kind": "Job",
                    "metadata": {"namespace": "dreamzero-training"},
                    "spec": {"activeDeadlineSeconds": 60, "template": {"metadata": {}, "spec": {
                        "containers": [{"name": "worker", "image": "fixture", "env": [],
                                        "volumeMounts": [{"name": "project", "mountPath": "/old"}]}],
                        "volumes": [{"name": "project", "hostPath": {"path": "/old"}}]}}}}
        self.write(self.runtime / "deployment/job_template.json", template)
        self.tasks = [{"task": name, "seed": 0, "camera_size": [512, 512]} for name in
                      ("TurnOnMicrowave", "TurnOffMicrowave", "TurnOnSinkFaucet", "TurnOffSinkFaucet")]
        histories = []
        for index, task in enumerate(self.tasks):
            history = self.root / f"old_{index}"
            history.mkdir()
            self.write(history / "summary.json", {"task_config": task, "status": "step_failed"})
            histories.append(str(history))
        other_seed = self.root / "other_seed"
        other_seed.mkdir()
        self.write(other_seed / "progress.json", {"status": "initializing"})
        self.write(other_seed / "task.json", dict(self.tasks[0], seed=1))
        self.config = {"tasks": self.tasks, "task": {"task": "stale_fallback"},
                       "library": str(self.root / "shared_library"),
                       "agent": {"provider": "http", "model": "fixture-model", "timeout_s": 360},
                       "exploration": {"max_attempts": 6, "max_replans": 1, "code_timeout_s": 180},
                       "maintenance": {"max_candidates": 2}, "maintain_after_task": True,
                       "episode_retry": {"max_episodes": None, "history_episodes": 3},
                       "retry_experiences": histories + [str(other_seed)]}
        self.config_path = self.root / "source.json"
        self.write(self.config_path, self.config)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))

    def launch(self, name="evo-parallel-test", dry_run=True):
        with contextlib.redirect_stdout(io.StringIO()):
            return parallel.launch_batch(self.config_path, name, self.project, self.runtime,
                                          deadline=0, dry_run=dry_run)

    def test_real_dry_run_freezes_four_isolated_cpu_jobs_and_preserves_shared_policy(self):
        receipt, manifest = self.launch()
        self.assertEqual(receipt["status"], "dry_run_prepared")
        self.assertEqual(json.loads(manifest.read_text()), receipt)
        self.assertEqual(len({child["run"] for child in receipt["tasks"]}), 4)
        self.assertEqual(len({child["job"] for child in receipt["tasks"]}), 4)
        for index, child in enumerate(receipt["tasks"]):
            with self.subTest(task=self.tasks[index]["task"]):
                actual = json.loads(Path(child["config"]).read_text())
                self.assertEqual(actual["tasks"], [self.tasks[index]])
                self.assertNotIn("task", actual)
                self.assertEqual(actual["retry_experiences"], [self.config["retry_experiences"][index]])
                for field in ("library", "agent", "exploration", "maintenance", "maintain_after_task", "episode_retry"):
                    self.assertEqual(actual[field], self.config[field])
                run = Path(child["run"])
                self.assertEqual(json.loads((run / "config.json").read_text()), actual)
                self.assertTrue((run / "source_sha256.json").is_file())
                job = json.loads((run / "job.json").read_text())
                self.assertEqual(job["metadata"]["name"], child["job"])
                self.assertNotIn("activeDeadlineSeconds", job["spec"])
                resources = job["spec"]["template"]["spec"]["containers"][0]["resources"]
                self.assertEqual(resources["requests"]["cpu"], "4")
                self.assertNotIn("nvidia.com/gpu", resources["requests"])
                self.assertEqual(child["status"], "dry_run_prepared")

    def test_all_run_collisions_are_checked_before_preparation_or_submission(self):
        (self.project / "runs/evo-collision-t04").mkdir(parents=True)
        with patch.object(parallel.subprocess, "run") as launch:
            with self.assertRaisesRegex(ValueError, "Run already exists"):
                self.launch("evo-collision", dry_run=False)
        launch.assert_not_called()
        self.assertFalse((self.project / "batches/evo-collision").exists())
        self.assertFalse((self.project / "runs/evo-collision-t01").exists())

    def test_partial_submission_stops_and_keeps_confirmed_and_unconfirmed_receipts(self):
        with patch.object(parallel.subprocess, "run", side_effect=[SimpleNamespace(returncode=0),
                                                                  SimpleNamespace(returncode=1)]) as launch:
            receipt, manifest = self.launch(dry_run=False)
        self.assertEqual(launch.call_count, 2)
        self.assertEqual(receipt["status"], "incomplete")
        self.assertEqual([child["status"] for child in receipt["tasks"]],
                         ["submitted", "submission_unconfirmed", "planned", "planned"])
        self.assertEqual(json.loads(manifest.read_text()), receipt)
        self.assertTrue(all(Path(child["config"]).is_file() for child in receipt["tasks"]))

    def test_invalid_history_is_rejected_before_any_job_is_created(self):
        self.config["retry_experiences"].append(str(self.root / "missing_history"))
        self.write(self.config_path, self.config)
        with patch.object(parallel.subprocess, "run") as launch:
            with self.assertRaisesRegex(ValueError, "Retry experience has no summary"):
                self.launch(dry_run=False)
        launch.assert_not_called()
        self.assertFalse((self.project / "batches/evo-parallel-test").exists())


if __name__ == "__main__":
    unittest.main()
