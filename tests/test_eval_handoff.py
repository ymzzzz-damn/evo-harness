"""Training-to-evaluation handoff evidence and isolation, without GPUs or APIs."""
import binascii
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zlib


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("eval_handoff", ROOT / "scripts/prepare_policy_evaluation.py")
handoff = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(handoff)
SPEC = importlib.util.spec_from_file_location("training_handoff_wrapper", ROOT / "scripts/train_and_prepare_evaluation.py")
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def png():
    def chunk(name, data):
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", binascii.crc32(name + data) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 512, 512, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress((b"\0" + b"\0" * (512 * 3)) * 512)) + chunk(b"IEND", b""))


class EvaluationHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project, self.runtime = self.root / "project", self.root / "runtime"
        self.run = self.project / "runs/train300"
        self.run.mkdir(parents=True)
        self.source = self.run / "source"
        for name in ("evo/__init__.py", "evo/policy_evaluation.py", "scripts/parallel_worker.py",
                     "scripts/launch_gpu_parallel_p6000.py", "scripts/prepare_policy_evaluation.py", "pyproject.toml"):
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if name == "scripts/prepare_policy_evaluation.py":
                shutil.copyfile(ROOT / name, path)
            else:
                path.write_text("# frozen fixture\n")
        manifest = {str(p.relative_to(self.source)): handoff._sha(p) for p in self.source.rglob("*") if p.is_file()}
        write(self.run / "source_sha256.json", manifest)
        self.template = self.root / "evaluation_template.json"
        write(self.template, {"tasks": [{"task": f"Eval{i:03d}", "seed": 1, "camera_size": [512, 512]} for i in range(50)],
                              "library": "/obsolete/library", "frozen_library": "/obsolete/frozen", "frozen_library_sha256": "old"})
        self.rgb = self.root / "fixture.png"
        self.rgb.write_bytes(png())
        self.depth = self.root / "fixture.npy"
        header = b"{'descr': '<f4', 'fortran_order': False, 'shape': (512, 512), }"
        header += b" " * ((64 - ((10 + len(header) + 1) % 64)) % 64) + b"\n"
        self.depth.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + b"\0" * (512 * 512 * 4))
        workers, records = [], []
        for i in range(300):
            task = {"task": f"Task{i:03d}", "seed": 0, "camera_size": [512, 512]}
            directory = self.run / "workers" / f"task_{i:03d}"
            exploration = directory / "campaign" / f"000_{task['task']}_seed0" / "exploration"
            exploration.mkdir(parents=True)
            cameras = {}
            for camera in ("left", "right", "wrist"):
                rgb, depth = exploration / (camera + ".png"), exploration / (camera + ".npy")
                os.link(self.rgb, rgb)
                os.link(self.depth, depth)
                cameras[camera] = {"image_path": str(rgb), "depth_path": str(depth), "width": 512, "height": 512}
            write(exploration / "initial.json", {"task": task["task"], "seed": 0, "step": 0,
                "terminal": False, "horizon": 1000, "cameras": cameras})
            write(exploration / "task.json", task)
            write(exploration / "summary.json", {"task_config": task, "status": "step_failed", "native_success": False,
                                                 "stats": {"steps": 100, "native_success": False}})
            worker = {"index": i, "task": task, "run": str(directory)}
            record = dict(worker, status="failed", returncode=1)
            write(directory / "task_receipt.json", record)
            write(directory / "campaign/summary.json", {"status": "tasks_incomplete", "maintenance_status": "not_needed",
                "episodes": [{"task": task, "episode_attempts": [{"summary_path": str(exploration / 'summary.json')}]}]})
            workers.append(worker)
            records.append(record)
        write(self.run / "config.json", {"mode": "run", "tasks": [w["task"] for w in workers],
            "library": str(self.run / "library"), "episode_retry": {"max_episodes": 1},
            "experiment_metadata": {"evaluation_template_sha256": handoff._sha(self.template)},
            "evaluation_handoff": {"template": str(self.template), "name": "eval50", "project": str(self.project), "runtime": str(self.runtime)}})
        write(self.run / "parallel_plan.json", {"mode": "run", "run": str(self.run), "source": str(self.source),
            "shared_library": str(self.run / "library"), "workers": workers})
        write(self.run / "parallel_receipt.json", {"status": "complete_with_failures", "workers": records,
            "counts": {"planned": 300, "pending": 0, "running": 0, "finished": 300, "failed": 300}})
        self.freeze = Mock(side_effect=self.fake_freeze)
        self.launch = Mock(side_effect=self.fake_launch)
        p1 = patch.object(handoff, "_frozen_policy", return_value=SimpleNamespace(freeze_active_library=self.freeze))
        p2 = patch.object(handoff, "_frozen_launcher", return_value=SimpleNamespace(prepare=self.launch))
        p1.start()
        p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)

    def fake_freeze(self, source, destination):
        destination.mkdir()
        (destination / "sources").mkdir()
        state = {"schema_version": 2, "case_lifecycle_version": 1, "active": {}, "versions": {},
                 "candidates": {}, "cases": {}, "pending_cases": {}, "case_activations": {}, "lessons": {}, "events": []}
        write(destination / "state.json", state)
        write(destination / "state.backup.json", state)
        receipt = {"path": str(destination), "state_sha256": handoff._sha(destination / "state.json"),
                   "source_library": str(source), "source_state_sha256": "training-state-sha",
                   "training_lessons_candidates_cases_imported": False}
        write(destination / "freeze_receipt.json", receipt)
        return receipt

    def fake_launch(self, config, name, project, runtime, **kwargs):
        run = project / "runs" / name
        run.mkdir()
        shutil.copytree(kwargs["source_root"], run / "source")
        shutil.copyfile(self.run / "source_sha256.json", run / "source_sha256.json")
        shutil.copyfile(config, run / "config.json")
        plan = {"run": str(run), "job": "yumuzhen-" + name, "source_origin": str(kwargs["source_root"]), "mode": "evaluate_policy"}
        write(run / "parallel_plan.json", plan)
        write(run / "job.json", {"kind": "Job"})
        return plan, run / "job.json"

    def prepare(self):
        return handoff.prepare_evaluation(self.run, self.template, "eval50", self.project, self.runtime)

    def exploration(self, index):
        return next((self.run / "workers" / f"task_{index:03d}" / "campaign").glob("*/exploration"))

    def test_complete_evidence_freezes_once_and_pins_framework_resources_and_all_library_paths(self):
        p = self.exploration(12) / "summary.json"
        summary = json.loads(p.read_text())
        summary.update(status="error", native_success=False, stats={"steps": 100, "native_success": True},
                       error={"type": "ProviderError", "message": "request interrupted"})
        write(p, summary)
        receipt = self.prepare()
        self.assertFalse(receipt["job_created"])
        self.assertEqual(receipt["training_audit"]["counts"]["valid_initial_episodes"], 300)
        self.assertEqual(receipt["training_audit"]["counts"]["native_failure"], 299)
        self.assertEqual(receipt["training_audit"]["counts"]["interrupted_after_initial"], 1)
        interrupted = receipt["training_audit"]["tasks"][12]
        self.assertIs(interrupted["observed_native_success"], True)
        self.assertIs(interrupted["reported_native_success"], False)
        self.assertEqual(receipt["training_audit"]["counts"]["observed_native_success"], 1)
        kwargs = self.launch.call_args.kwargs
        self.assertEqual(kwargs["source_root"], self.source)
        self.assertEqual((kwargs["cpu_request"], kwargs["cpu_limit"], kwargs["memory_request"], kwargs["memory_limit"], kwargs["max_parallel"]),
                         ("64", "128", "256Gi", "384Gi", 64))
        evaluation = json.loads((self.run / "evaluation_config.json").read_text())
        self.assertEqual(evaluation["mode"], "evaluate_policy")
        self.assertEqual(evaluation["library"], str(self.run / "evaluation_frozen_library"))
        self.assertEqual(evaluation["frozen_library"], evaluation["library"])
        self.assertEqual(evaluation["frozen_library_sha256"], receipt["frozen_library"]["state_sha256"])
        self.assertTrue((self.run / "handoff_ready.json").exists())
        self.assertEqual(self.prepare(), receipt)
        self.freeze.assert_called_once()
        self.launch.assert_called_once()

    def test_running_queue_is_rejected_before_freeze(self):
        p = self.run / "parallel_receipt.json"
        value = json.loads(p.read_text())
        value["counts"].update(running=1, finished=299)
        value["workers"][0]["status"] = "running"
        write(p, value)
        with self.assertRaisesRegex(handoff.HandoffNotReady, "pending/running"):
            self.prepare()
        self.freeze.assert_not_called()

    def test_missing_initial_is_not_hidden_by_exit_zero(self):
        (self.exploration(42) / "initial.json").unlink()
        with self.assertRaises(handoff.HandoffNotReady) as raised:
            self.prepare()
        self.assertEqual(raised.exception.audit["counts"]["blocked_tasks"], 1)
        self.assertIn("Missing initial", raised.exception.audit["problems"][0]["reason"])
        self.freeze.assert_not_called()

    def test_missing_final_summary_and_multiple_explorations_are_rejected(self):
        (self.exploration(7) / "summary.json").unlink()
        (self.exploration(8).parent / "retries/episode_0002/exploration").mkdir(parents=True)
        with self.assertRaises(handoff.HandoffNotReady) as raised:
            self.prepare()
        self.assertEqual(raised.exception.audit["counts"]["blocked_tasks"], 2)
        self.freeze.assert_not_called()

    def test_source_tampering_is_rejected(self):
        (self.source / "evo/__init__.py").write_text("# changed\n")
        with self.assertRaisesRegex(handoff.HandoffNotReady, "source snapshot differs"):
            self.prepare()
        self.freeze.assert_not_called()

    def test_template_tampering_is_rejected(self):
        self.template.write_text(self.template.read_text() + "\n")
        with self.assertRaisesRegex(handoff.HandoffNotReady, "template SHA"):
            self.prepare()
        self.freeze.assert_not_called()

    def test_unknown_existing_freeze_or_evaluation_run_is_preserved(self):
        frozen = self.run / "evaluation_frozen_library"
        frozen.mkdir()
        sentinel = frozen / "unowned"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(handoff.HandoffNotReady, "Refuse to overwrite"):
            self.prepare()
        self.assertEqual(sentinel.read_text(), "keep")
        self.freeze.assert_not_called()

    def test_memory_in_a_frozen_snapshot_is_rejected(self):
        def contaminated(source, destination):
            receipt = self.fake_freeze(source, destination)
            state = json.loads((destination / "state.json").read_text())
            state["lessons"] = {"A02": ["training failure"]}
            write(destination / "state.json", state)
            receipt["state_sha256"] = handoff._sha(destination / "state.json")
            return receipt
        self.freeze.side_effect = contaminated
        with self.assertRaisesRegex(handoff.HandoffNotReady, "Training memory leaked"):
            self.prepare()
        self.launch.assert_not_called()

    def test_prepared_artifact_tampering_breaks_idempotency(self):
        self.prepare()
        (self.project / "runs/eval50/job.json").write_text("{}\n")
        with self.assertRaisesRegex(handoff.HandoffNotReady, "artifact changed"):
            self.prepare()
        self.launch.assert_called_once()

    def test_wrapper_forwards_stop_to_frozen_supervisor_and_never_hands_off(self):
        sent = []
        class Process:
            def wait(process):
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
                return 1
            def poll(process):
                return None
            def send_signal(process, signum):
                sent.append(signum)
        with patch.object(wrapper.subprocess, "Popen", return_value=Process()) as popen:
            receipt = wrapper.train_and_prepare(self.run / "parallel_plan.json")
        self.assertEqual(receipt["status"], "interrupted")
        self.assertEqual(sent, [signal.SIGTERM])
        command = popen.call_args.args[0]
        self.assertEqual(command, [sys.executable, "-u", str(self.source / "scripts/parallel_worker.py"), "--plan", str(self.run / "parallel_plan.json")])
        self.freeze.assert_not_called()
        self.launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
