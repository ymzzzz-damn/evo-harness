"""Finite queue summaries distinguish scheduling, native episodes, and tools."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from evo.codecheck import source_hash
from evo.library import Library


SPEC = importlib.util.spec_from_file_location(
    "summarize_parallel_campaign", Path(__file__).resolve().parents[1] / "scripts/summarize_parallel_campaign.py")
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)

CODE = 'def skill(api, params):\n    api.wait(params.get("steps", 1))\n    return {}\n'
OTHER = 'def skill(api, params):\n    api.wait(params.get("steps", 2))\n    return {}\n'


class CampaignSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.library = Library(self.root / "library")
        self.baseline_version = self.publish("P09", CODE, "/old_campaign/exploration")
        self.baseline = self.root / "library_baseline_state.json"
        self.baseline.write_bytes(self.library.state_path.read_bytes())
        self.workers = [{"index": index, "task": {"task": f"Task{index}", "seed": 0},
                         "run": str(self.run / f"workers/task_{index:03d}")} for index in range(6)]
        self.plan = self.run / "parallel_plan.json"
        self.write(self.plan, {"run": str(self.run), "job": "finite-test", "workers": self.workers,
                               "shared_library": str(self.library.root)})
        self.write(self.run / "config.json", {"episode_retry": {"max_episodes": 1},
            "experiment_metadata": {"library_baseline_state": str(self.baseline)}})
        self.receipt = self.run / "parallel_receipt.json"

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def publish(self, atom, code, source_run):
        case_id = self.library.stage_case({"atom": atom, "task_config": {"task": atom, "seed": 0},
            "step": {"atom": atom, "params": {"object": "fixture", "target": "on"}},
            "require_native_success": True, "source_run": source_run})
        candidate = self.library.propose(atom, code, metadata={"source_run": source_run, "stage": "distill",
                                                             "new_case_id": case_id})
        cases = self.library.cases(atom)
        if case_id not in {case["id"] for case in cases}:
            cases.append(self.library.get_case(case_id))
        self.library.record_validation(candidate["id"], {"candidate_sha256": candidate["sha256"],
            "success": True, "validation_kind": "fresh_environment_replay",
            "cases": [{"case_id": case["id"], "success": True, "native_success": True}
                      for case in cases]})
        return self.library.promote(candidate["id"])

    def episode(self, index, status, native=False, initial=True, pending=None, progress=False):
        worker = self.workers[index]
        campaign = Path(worker["run"]) / "campaign"
        exploration = campaign / f"000_Task{index}_seed0/exploration"
        result = {"task_config": worker["task"], "status": status, "native_success": native,
                  "stats": {"steps": 1 if initial else 0}, "pending": pending or [],
                  "completed_steps": [{"success": True, "skipped": False}] if pending else [],
                  "failures": []}
        if status == "error":
            result["error"] = {"type": "RuntimeError", "message": "startup or backend error"}
        self.write(exploration / ("progress.json" if progress else "summary.json"), result)
        self.write(exploration / "task.json", worker["task"])
        if initial:
            self.write(exploration / "initial.json", {"instruction": "fixture", "step": 0,
                                                        "task": worker["task"]["task"], "seed": 0})
        self.write(campaign / "summary.json", {"status": "running" if progress else "complete" if native else "tasks_incomplete",
            "episodes": [{"task": worker["task"], "exploration": None if progress else result,
                          "episode_attempts": [] if progress else [{"summary_path": str(exploration / "summary.json")}],
                          "maintenance": None}]})
        return exploration

    def setup_mixed_queue(self):
        self.episode(0, "success", native=True)
        self.episode(1, "step_failed", pending=["unpublished_partial.json"])
        self.episode(2, "error", initial=False)
        self.episode(5, "initializing", progress=True)
        self.write(self.receipt, {"status": "running", "workers": [
            {"index": 0, "status": "succeeded", "returncode": 0},
            {"index": 1, "status": "failed", "returncode": 1},
            {"index": 2, "status": "failed", "returncode": 1},
            {"index": 3, "status": "failed", "returncode": 1},
            {"index": 4, "status": "pending"},
            {"index": 5, "status": "running", "pid": 123}]})

    def test_startup_failures_are_attempted_but_not_native_episodes(self):
        self.setup_mixed_queue()
        report = summary.summarize_campaign(self.plan)
        self.assertEqual([task["outcome"] for task in report["tasks"]],
                         ["success", "failed", "no_episode", "no_episode", "pending", "running"])
        self.assertEqual(report["counts"], {"planned_tasks": 6, "attempted_tasks": 2, "worker_attempted_tasks": 5, "settled_tasks": 4,
            "successful_tasks": 1, "failed_tasks": 1, "tasks_without_native_episode": 2,
            "pending_tasks": 1, "running_tasks": 1, "exploration_invocations": 4,
            "native_episodes_confirmed": 3, "episode_limit_violations": 0})
        self.assertEqual(report["tasks"][2]["exploration_invocations"], 1)
        self.assertEqual(report["tasks"][2]["native_episodes_confirmed"], 0)
        self.assertEqual(report["tasks"][3]["exploration_invocations"], 0)
        self.assertEqual(report["tasks"][1]["pending_unpublished_experiences"], 1)
        self.assertEqual(report["tools"]["counts"]["new_versions_since_baseline"], 0)
        self.assertFalse(report["all_tasks_attempted"])

    def test_queue_requires_recovery_when_workers_finish_without_native_episodes(self):
        self.setup_mixed_queue()
        receipt = json.loads(self.receipt.read_text())
        receipt["status"] = "complete_with_failures"
        receipt["workers"][4].update(status="failed", returncode=1)
        receipt["workers"][5].update(status="failed", returncode=1)
        self.episode(5, "plan_complete_task_failed")
        self.write(self.receipt, receipt)
        report = summary.summarize_campaign(self.plan)
        self.assertEqual(report["status"], "needs_episode_recovery")
        self.assertFalse(report["all_tasks_attempted"])
        self.assertTrue(report["all_workers_attempted"])
        self.assertFalse(report["all_tasks_succeeded"])
        self.assertEqual(report["counts"]["attempted_tasks"], 3)
        self.assertEqual(report["counts"]["worker_attempted_tasks"], 6)
        self.assertEqual(report["counts"]["settled_tasks"], 6)
        self.assertEqual(report["counts"]["native_episodes_confirmed"], 3)

    def test_all_valid_native_episodes_complete_even_when_tasks_fail(self):
        for index in range(6):
            self.episode(index, "success" if index == 0 else "step_failed", native=index == 0)
        self.write(self.receipt, {"status": "complete_with_failures", "workers": [
            {"index": index, "status": "succeeded" if index == 0 else "failed",
             "returncode": 0 if index == 0 else 1} for index in range(6)]})
        report = summary.summarize_campaign(self.plan)
        self.assertEqual(report["status"], "all_attempted")
        self.assertTrue(report["all_native_episodes_attempted"])
        self.assertFalse(report["all_tasks_succeeded"])
        self.assertEqual(report["counts"]["attempted_tasks"], 6)
        self.assertEqual(report["counts"]["failed_tasks"], 5)

    def test_interruption_with_initial_frame_and_no_final_verdict_needs_recovery(self):
        self.episode(0, "initializing", progress=True)
        self.write(self.receipt, {"workers": [{"index": 0, "status": "interrupted", "pid": 123}]})
        task = summary.summarize_campaign(self.plan)["tasks"][0]
        self.assertTrue(task["worker_attempted"])
        self.assertEqual(task["native_episodes_confirmed"], 1)
        self.assertFalse(task["attempted"])
        self.assertEqual(task["outcome"], "no_episode")

    def test_mismatched_task_or_seed_never_counts_as_native_episode(self):
        exploration = self.episode(0, "step_failed")
        self.write(exploration / "initial.json", {"task": "WrongTask", "seed": 0})
        self.write(self.receipt, {"workers": [{"index": 0, "status": "failed", "returncode": 1}]})
        self.assertEqual(summary.summarize_campaign(self.plan)["tasks"][0]["native_episodes_confirmed"], 0)
        self.write(exploration / "initial.json", {"task": "Task0", "seed": 99})
        self.assertEqual(summary.summarize_campaign(self.plan)["tasks"][0]["native_episodes_confirmed"], 0)

    def test_baseline_tools_are_separate_from_new_tools_and_exported_without_library_mutation(self):
        self.setup_mixed_queue()
        new = self.publish("P10", OTHER, str(Path(self.workers[0]["run"]) / "campaign/000_Task0_seed0/exploration"))
        self.library.propose("P01", CODE)  # Unvalidated source isn't a tool.
        before = {path: path.read_bytes() for path in self.library.root.rglob("*") if path.is_file()}
        report = summary.summarize_campaign(self.plan)
        tools = report["tools"]
        self.assertEqual([(entry["atom"], entry["version"]) for entry in tools["baseline_versions"]], [("P09", 1)])
        self.assertEqual([(entry["atom"], entry["version"]) for entry in tools["campaign_new_versions"]], [("P10", 1)])
        self.assertEqual(tools["counts"]["active_validated_tools"], 2)
        self.assertEqual(tools["counts"]["unvalidated_candidates"], 1)
        summary.write_summary(report, self.run / "campaign_summary.json", self.run / "tool_report.json")
        self.assertEqual(len(tools["exported_code"]), 2)
        exported = self.run / "code_tools/P10_v1.py"
        self.assertEqual(source_hash(exported.read_text()), new["sha256"])
        self.assertFalse((self.run / "code_tools/P01_v1.py").exists())
        entry = tools["campaign_new_versions"][0]
        self.assertEqual(entry["entrypoint"], "skill(api, params) -> dict")
        self.assertEqual(entry["coverage"][0]["params"]["object"], "fixture")
        self.assertTrue(Path(entry["source_path"]).is_absolute())
        self.assertEqual(json.loads((self.run / "tool_report.json").read_text())["counts"], tools["counts"])
        for path, contents in before.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_worker_exit_zero_alone_cannot_prove_success(self):
        self.write(self.receipt, {"workers": [{"index": 0, "status": "succeeded", "returncode": 0}]})
        report = summary.summarize_campaign(self.plan)
        self.assertEqual(report["tasks"][0]["outcome"], "no_episode")
        self.assertEqual(report["counts"]["successful_tasks"], 0)

    def test_native_success_with_program_failure_is_still_failed_task(self):
        self.episode(0, "step_failed", native=True)
        self.write(self.receipt, {"workers": [{"index": 0, "status": "failed", "returncode": 1}]})
        task = summary.summarize_campaign(self.plan)["tasks"][0]
        self.assertTrue(task["reported_native_success"])
        self.assertFalse(task["program_success"])
        self.assertEqual(task["outcome"], "failed")

    def test_broken_tool_source_is_reported_and_never_exported(self):
        version = self.publish("P10", OTHER, str(Path(self.workers[0]["run"]) / "campaign/exploration"))
        Path(version["source_path"]).write_text("tampered")
        report = summary.summarize_campaign(self.plan)
        self.assertEqual(report["tools"]["counts"]["published_versions_with_integrity_errors"], 1)
        self.assertEqual(report["tools"]["counts"]["new_validated_versions_from_this_campaign"], 0)
        summary.write_summary(report, self.run / "campaign_summary.json", self.run / "tool_report.json")
        self.assertFalse((self.run / "code_tools/P10_v1.py").exists())

    def test_corrupted_replay_metadata_is_not_counted_as_validated_tool(self):
        original = Library._read_state_file(self.library.state_path)
        for field in ("validation_kind", "candidate_sha256", "case_ids"):
            with self.subTest(field=field):
                state = json.loads(json.dumps(original))
                record = state["versions"]["P09"]["1"]
                if field == "case_ids":
                    record["case_ids"] = []
                else:
                    record["validation"][field] = "corrupted"
                with self.library._locked():
                    self.library._save(state)
                tools = summary.summarize_campaign(self.plan)["tools"]
                self.assertEqual(tools["counts"]["active_validated_tools"], 0)
                self.assertEqual(tools["counts"]["published_versions_with_integrity_errors"], 1)

    def test_missing_baseline_does_not_label_existing_versions_as_new(self):
        report = summary.summarize_campaign(self.plan, baseline_state=self.root / "missing.json")
        self.assertFalse(report["tools"]["baseline_available"])
        self.assertIsNone(report["tools"]["counts"]["new_versions_since_baseline"])
        self.assertIsNone(report["tools"]["published_versions"][0]["new_since_baseline"])

    def test_duplicate_plan_indices_are_rejected(self):
        plan = json.loads(self.plan.read_text())
        plan["workers"][1]["index"] = 0
        self.write(self.plan, plan)
        with self.assertRaisesRegex(ValueError, "Duplicate task index"):
            summary.summarize_campaign(self.plan)

    def test_failed_task_keeps_immediately_published_tool_and_maintenance_status(self):
        exploration = self.episode(1, "step_failed")
        experience = exploration / "pending/01_P10.json"
        self.episode(1, "step_failed", pending=[str(experience)])
        version = self.publish("P10", OTHER, str(exploration))
        campaign_root = Path(self.workers[1]["run"]) / "campaign"
        campaign = json.loads((campaign_root / "summary.json").read_text())
        campaign.update(task_status="tasks_incomplete", maintenance_status="complete")
        self.write(campaign_root / "summary.json", campaign)
        self.write(campaign_root / "maintenance_runs.json", [{
            "output": str(campaign_root / "maintenance/run_0001"), "status": "complete",
            "experience_paths": [str(experience)], "result": {"status": "complete", "rejected": [],
                "promoted": [{"atom": "P10", "version": 1, "sha256": version["sha256"]}]}}])
        self.write(self.receipt, {"workers": [{"index": 1, "status": "failed", "returncode": 1}]})
        report = summary.summarize_campaign(self.plan)
        task = report["tasks"][1]
        self.assertEqual(task["outcome"], "failed")
        self.assertEqual(task["task_status"], "tasks_incomplete")
        self.assertEqual(task["maintenance_status"], "complete")
        self.assertEqual(task["maintenance_promoted"][0]["atom"], "P10")
        self.assertEqual(task["pending_experiences_total"], 1)
        self.assertEqual(task["pending_unpublished_experiences"], 0)
        self.assertEqual(report["tools"]["counts"]["new_validated_versions_from_this_campaign"], 1)

    def test_task_success_is_counted_when_maintenance_failed_and_worker_exits_one(self):
        exploration = self.episode(0, "success", native=True)
        campaign_root = exploration.parents[1]
        campaign = json.loads((campaign_root / "summary.json").read_text())
        campaign.update(status="validation_failed", task_status="complete", maintenance_status="failed")
        campaign["episodes"][0].update(maintenance_status="failed", maintenance={
            "status": "some_candidates_rejected", "promoted": [], "rejected": [{"atom": "P10"}]})
        self.write(campaign_root / "summary.json", campaign)
        self.write(self.receipt, {"workers": [{"index": 0, "status": "failed", "returncode": 1}]})
        task = summary.summarize_campaign(self.plan)["tasks"][0]
        self.assertEqual(task["outcome"], "success")
        self.assertEqual(task["task_status"], "complete")
        self.assertEqual(task["maintenance_status"], "failed")
        self.assertEqual(task["maintenance_rejected_count"], 1)

    def test_reuse_admission_adds_coverage_without_inventing_a_new_code_version(self):
        exploration = self.episode(2, "step_failed")
        case_id = self.library.stage_case({"atom": "P09", "task_config": self.workers[2]["task"],
            "step": {"atom": "P09", "params": {"object": "another_fixture", "target": "on"}},
            "require_native_success": False, "source_run": str(exploration)})
        self.library.stage_case({"atom": "P09", "source_run": str(exploration),
                                 "step": {"params": {"unverified": True}}})
        cases = self.library.cases("P09") + [self.library.get_case(case_id)]
        proof = {"candidate_sha256": self.baseline_version["sha256"], "success": True,
                 "validation_kind": "fresh_environment_replay",
                 "cases": [{"case_id": case["id"], "success": True, "native_success": False} for case in cases]}
        self.library.accept_case(case_id, "P09", 1, proof)
        report = summary.summarize_campaign(self.plan)
        tools = report["tools"]
        self.assertEqual(tools["counts"]["published_versions"], 1)
        self.assertEqual(tools["counts"]["new_versions_since_baseline"], 0)
        self.assertEqual(tools["counts"]["formal_regression_cases"], 2)
        self.assertEqual(tools["counts"]["pending_cases"], 1)
        self.assertEqual(tools["counts"]["validated_reuse_case_admissions"], 1)
        self.assertEqual(tools["counts"]["new_validated_cases_from_this_campaign"], 1)
        self.assertEqual(tools["counts"]["pending_cases_from_this_campaign"], 1)
        coverage = tools["active_validated_tools"][0]["coverage"]
        self.assertEqual(len(coverage), 2)
        added = next(item for item in coverage if item["case_id"] == case_id)
        self.assertEqual(added["evidence_origin"], "reuse_acceptance")
        self.assertEqual(added["params"]["object"], "another_fixture")
        self.assertFalse(added["require_native_success"])

    def test_corrupted_reuse_activation_is_not_counted_as_validated_new_coverage(self):
        source = str(Path(self.workers[0]["run"]) / "campaign/exploration")
        case_id = self.library.stage_case({"atom": "P09", "source_run": source, "step": {"params": {"new": True}}})
        proof = {"candidate_sha256": self.baseline_version["sha256"], "success": True,
                 "validation_kind": "fresh_environment_replay", "cases": [
                     {"case_id": case["id"], "success": True}
                     for case in self.library.cases("P09") + [self.library.get_case(case_id)]]}
        self.library.accept_case(case_id, "P09", 1, proof)
        with self.library._locked():
            state = self.library._load()
            state["case_activations"][case_id]["validation"]["success"] = False
            self.library._save(state)
        tools = summary.summarize_campaign(self.plan)["tools"]
        self.assertEqual(tools["counts"]["validated_reuse_case_admissions"], 0)
        self.assertEqual(tools["counts"]["case_activation_integrity_errors"], 1)
        self.assertEqual(tools["counts"]["new_validated_cases_from_this_campaign"], 0)
        self.assertEqual(len(tools["active_validated_tools"][0]["coverage"]), 1)

    def test_immediate_ledger_is_read_when_campaign_summary_is_stale(self):
        exploration = self.episode(0, "step_failed", pending=["staged.json"])
        campaign_root = exploration.parents[1]
        self.write(campaign_root / "maintenance_runs.json", [{"output": "maintenance/one", "status": "error",
            "experience_paths": ["staged.json"], "result": {"status": "error", "promoted": [],
                "accepted_cases": [], "rejected": [{"error": "fresh replay failed"}]}}])
        task = summary.summarize_campaign(self.plan)["tasks"][0]
        self.assertEqual(task["maintenance_status"], "failed")
        self.assertEqual(task["maintenance_runs_count"], 1)
        self.assertEqual(task["pending_unpublished_experiences"], 1)


if __name__ == "__main__":
    unittest.main()
