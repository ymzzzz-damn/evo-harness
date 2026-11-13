"""Episode-reset orchestration tests using the explicit toy adapter from loop tests."""

import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

from evo.agent import Agent, ScriptedProvider
from evo.cli import run_config
from evo.io import read_json, write_json
from evo.library import Library
from evo.loops import MaintenanceLoop

from test_loops import BAD, CLAIM, GOOD, Factory, candidate, plan, step


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.factory = Factory()
        self.sequence = 0
        self.task = {"task": "toy_plant", "seed": 17, "goal": 2, "horizon": 1}

    def tearDown(self):
        self.temporary.cleanup()

    def run_campaign(self, fixtures, options=None, factory=None):
        self.sequence += 1
        output = self.root / f"campaign_{self.sequence}"
        self.provider = ScriptedProvider(fixtures)
        agent = Agent(self.provider, max_attempts=1)
        config = {"library": str(self.root / "library"), "agent": {}, "tasks": [self.task],
                  "exploration": {"max_attempts": 1, "max_replans": 0},
                  "maintenance": {"max_candidates": 1}, "maintain_after_task": False,
                  "episode_retry": {"max_episodes": 2, "history_episodes": 3}}
        config.update(options or {})
        if config.get("episode_retry") == "legacy":
            del config["episode_retry"]
        with patch("evo.cli.create_agent", return_value=agent), patch("evo.cli.adapter_factory", factory or self.factory):
            report = run_config(config, output)
        return report, output

    def test_horizon_failure_resets_same_task_and_seed_then_succeeds(self):
        report, output = self.run_campaign({"plan": [plan(), plan()], "repair": [candidate(BAD), candidate(GOOD)]})
        self.assertEqual(report["status"], "complete")
        task = report["episodes"][0]
        self.assertEqual([attempt["status"] for attempt in task["episode_attempts"]], ["step_failed", "success"])
        self.assertEqual(report["total_episode_attempts"], 2)
        self.assertEqual([adapter.config for adapter in self.factory.instances], [self.task, self.task])
        self.assertTrue(all(adapter.closed for adapter in self.factory.instances))
        self.assertEqual(self.factory.instances[0].position, 1)
        self.assertEqual(self.factory.instances[1].position, 2)
        expected = output / "000_toy_plant_seed17" / "retries" / "episode_0002" / "exploration"
        self.assertEqual(Path(task["episode_attempts"][1]["summary_path"]), expected / "summary.json")
        self.assertEqual(read_json(expected / "initial.json")["joint"], 0)
        self.assertEqual(read_json(output / "summary.json")["status"], "complete")

    def test_null_budget_retries_until_third_episode_with_bounded_history(self):
        fixtures = {"plan": [plan(), plan(), plan()], "repair": [candidate(BAD), candidate(BAD), candidate(GOOD)]}
        report, _ = self.run_campaign(fixtures, {"episode_retry": {"max_episodes": None, "history_episodes": 1}})
        self.assertEqual(report["status"], "complete")
        self.assertEqual(len(report["episodes"][0]["episode_attempts"]), 3)
        requests = [request for request in self.provider.requests if request["stage"] == "plan"]
        first = requests[0]["context"]["episode_retry"]
        self.assertEqual(first["previous_episodes"], [])
        self.assertTrue(first["reset_to_initial_state"])
        third = requests[2]["context"]
        self.assertEqual(third["initial_observation"]["joint"], 0)
        self.assertNotIn("completed_steps", third)
        history = third["episode_retry"]["previous_episodes"]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["episode_attempt"], 2)
        self.assertTrue(history[0]["observations_are_stale"])
        self.assertEqual(history[0]["recent_failures"][0]["code"], BAD)
        self.assertEqual(history[0]["recent_failures"][0]["step"]["params"], {"target": 2})
        self.assertFalse(history[0]["recent_failures"][0]["verification"]["success"])
        self.assertNotIn("observation", history[0]["recent_failures"][0])
        # The permanent campaign index contains paths/counts, not repeated code.
        self.assertNotIn("code", str(report["episodes"][0]["episode_attempts"]))

    def test_finite_budget_exhaustion_is_explicit(self):
        report, _ = self.run_campaign({"plan": [plan(), plan()], "repair": [candidate(BAD), candidate(BAD)]})
        self.assertEqual(report["status"], "tasks_incomplete")
        self.assertEqual(report["episodes"][0]["retry_status"], "budget_exhausted")
        self.assertEqual(len(self.factory.instances), 2)

    def test_plan_complete_without_native_success_is_retried(self):
        partial = step(target=1, predicate="joint_near")
        fixtures = {"plan": [plan(partial), plan()], "repair": [candidate(GOOD), candidate(GOOD)],
                    "distill": [candidate(GOOD)]}
        report, _ = self.run_campaign(fixtures, {"maintain_after_task": True})
        attempts = report["episodes"][0]["episode_attempts"]
        self.assertEqual([attempt["status"] for attempt in attempts], ["plan_complete_task_failed", "success"])
        self.assertEqual(attempts[0]["pending_count"], 1)
        self.assertEqual(len(Library(self.root / "library").cases("P01")), 2)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(len(report["episodes"][0]["maintenance_runs"]), 2)

    def test_native_success_with_program_error_still_requires_successful_retry(self):
        errored = ('def skill(api, params):\n    api.move_joints([params["target"]])\n'
                   '    raise RuntimeError("Synthetic post-motion program error")\n    return {}\n')
        fixtures = {"plan": [plan(), plan()], "repair": [candidate(errored), candidate(GOOD)],
                    "distill": [candidate(GOOD)]}
        report, _ = self.run_campaign(fixtures, {"maintain_after_task": True})
        attempts = report["episodes"][0]["episode_attempts"]
        self.assertEqual([attempt["native_success"] for attempt in attempts], [True, True])
        self.assertEqual([attempt["status"] for attempt in attempts], ["step_failed", "success"])
        self.assertEqual([request["stage"] for request in self.provider.requests].count("distill"), 1)
        self.assertEqual(Library(self.root / "library").active("P01")["code"], GOOD)

    def test_campaign_progress_is_written_before_starting_next_episode(self):
        snapshots = []

        def observed_factory(task, output):
            snapshots.append(read_json(self.root / "campaign_1" / "summary.json"))
            return self.factory(task, output)

        fixtures = {"plan": [plan(), plan()], "repair": [candidate(BAD), candidate(GOOD)]}
        report, _ = self.run_campaign(fixtures, factory=observed_factory)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(snapshots[0]["current_episode_attempt"], 1)
        self.assertEqual(snapshots[1]["current_episode_attempt"], 2)
        self.assertEqual(snapshots[1]["status"], "running")
        self.assertEqual(len(snapshots[1]["episodes"][0]["episode_attempts"]), 1)
        self.assertEqual(snapshots[1]["episodes"][0]["exploration"]["status"], "step_failed")

    def test_hard_environment_error_does_not_retry_forever(self):
        calls = []

        def broken_factory(task, output):
            calls.append(task)
            raise RuntimeError("Synthetic environment initialization failure")

        report, _ = self.run_campaign({}, {"episode_retry": {"max_episodes": None}}, factory=broken_factory)
        self.assertEqual(len(calls), 1)
        self.assertEqual(report["episodes"][0]["exploration"]["status"], "error")
        self.assertEqual(report["episodes"][0]["retry_status"], "not_retryable")
        self.assertFalse(self.provider.requests)

    def test_already_successful_reset_is_not_retried(self):
        report, _ = self.run_campaign({}, {"tasks": [{**self.task, "initial": 2}],
                                           "episode_retry": {"max_episodes": None}})
        self.assertEqual(len(self.factory.instances), 1)
        self.assertEqual(report["episodes"][0]["exploration"]["status"], "already_successful_at_reset")
        self.assertFalse(self.provider.requests)

    def test_failed_episode_partial_work_is_published_before_reset(self):
        pre = step(identifier="pre", target=1, predicate="joint_near")
        final = step(identifier="final", target=2, atom="P02", depends=["pre"])
        fixtures = {"plan": [plan(pre, final), plan(step(atom="P02"))],
                    "repair": [candidate(GOOD), candidate(CLAIM), candidate(GOOD)],
                    "distill": [candidate(GOOD), candidate(GOOD)]}
        report, _ = self.run_campaign(fixtures, {"tasks": [{**self.task, "horizon": 4}], "maintain_after_task": True})
        self.assertEqual(report["status"], "complete", report)
        attempts = report["episodes"][0]["episode_attempts"]
        self.assertEqual(attempts[0]["pending_count"], 1)
        library = Library(self.root / "library")
        self.assertEqual(library.active("P01")["version"], 1)
        self.assertEqual(len(library.cases("P01")), 1)
        self.assertEqual(library.active("P02")["version"], 1)
        self.assertEqual([request["stage"] for request in self.provider.requests].count("distill"), 2)
        context = [request["context"] for request in self.provider.requests if request["stage"] == "plan"][1]
        self.assertEqual(context["episode_retry"]["previous_episodes"][0]["completed_steps"][0]["code"], GOOD)
        self.assertEqual(context["library"][0]["atom"], "P01")
        self.assertNotIn("completed_steps", context)

    def test_single_failed_episode_retains_partial_published_skill(self):
        pre = step(identifier="pre", target=1, predicate="joint_near")
        final = step(identifier="final", target=2, atom="P02", depends=["pre"])
        fixtures = {"plan": [plan(pre, final)], "repair": [candidate(GOOD), candidate(CLAIM)],
                    "distill": [candidate(GOOD)]}
        report, _ = self.run_campaign(fixtures, {"tasks": [{**self.task, "horizon": 4}], "maintain_after_task": True,
                                                "episode_retry": {"max_episodes": 1}})
        self.assertEqual(report["status"], "tasks_incomplete")
        self.assertEqual(report["task_status"], "tasks_incomplete")
        self.assertEqual(report["maintenance_status"], "complete")
        self.assertEqual(report["promoted_versions"], 1)
        self.assertEqual(report["episodes"][0]["maintenance"]["promoted"][0]["atom"], "P01")
        self.assertEqual(Library(self.root / "library").active("P01")["version"], 1)

    def test_atomic_maintenance_runs_with_live_main_environment_and_deduplicates_pending(self):
        pre = step(identifier="pre", target=1, predicate="joint_near")
        final = step(identifier="final", target=2, atom="P02", depends=["pre"])
        replay_observations = []

        def observed_factory(task, output):
            if "maintenance" in Path(output).parts:
                main = self.factory.instances[0]
                replay_observations.append({"closed": main.closed, "position": main.position,
                                           "snapshot": read_json(self.root / "campaign_1" / "summary.json")})
            return self.factory(task, output)

        fixtures = {"plan": [plan(pre, final)], "repair": [candidate(GOOD), candidate(GOOD)],
                    "distill": [candidate(GOOD), candidate(GOOD)]}
        report, output = self.run_campaign(fixtures, {"tasks": [{**self.task, "horizon": 4}],
                                                      "maintain_after_task": True}, factory=observed_factory)
        self.assertEqual(report["status"], "complete", report)
        self.assertEqual([request["stage"] for request in self.provider.requests],
                         ["plan", "repair", "distill", "repair", "distill"])
        self.assertTrue(replay_observations)
        self.assertTrue(all(not item["closed"] for item in replay_observations))
        self.assertEqual(replay_observations[0]["position"], 1)
        self.assertEqual(replay_observations[0]["snapshot"]["maintenance_status"], "running")
        runs = report["maintenance_runs"]
        self.assertEqual(len(runs), 2)  # Episode-end pending sweep does not repeat either path.
        self.assertEqual({item["trigger"] for item in runs}, {"atomic"})
        self.assertEqual(len({item["output"] for item in runs}), 2)
        self.assertEqual(read_json(output / "maintenance_runs.json"), runs)
        self.assertEqual(report["promoted_versions"], 2)
        for run in runs:
            self.assertTrue((Path(run["output"]) / "summary.json").is_file())

    def test_episode_trigger_maintains_every_failed_episode_with_unlimited_retry(self):
        pre = step(identifier="pre", target=1, predicate="joint_near")
        final = step(identifier="final", target=2, atom="P02", depends=["pre"])
        fixtures = {"plan": [plan(pre, final), plan(step(atom="P02"))],
                    "repair": [candidate(GOOD), candidate(CLAIM), candidate(GOOD)],
                    "distill": [candidate(GOOD), candidate(GOOD)]}
        report, _ = self.run_campaign(fixtures, {"tasks": [{**self.task, "horizon": 4}],
            "maintain_after_task": True, "maintenance_trigger": "episode",
            "episode_retry": {"max_episodes": None}})
        self.assertEqual(report["status"], "complete", report)
        self.assertEqual([item["episode_attempt"] for item in report["maintenance_runs"]], [1, 2])
        self.assertEqual({item["trigger"] for item in report["maintenance_runs"]}, {"episode"})
        stages = [request["stage"] for request in self.provider.requests]
        self.assertEqual(stages[:5], ["plan", "repair", "repair", "distill", "plan"])
        self.assertEqual(report["promoted_versions"], 2)

    def test_maintenance_exception_does_not_interrupt_later_atoms(self):
        pre = step(identifier="pre", target=1, predicate="joint_near")
        final = step(identifier="final", target=2, atom="P02", depends=["pre"])
        fixtures = {"plan": [plan(pre, final)], "repair": [candidate(GOOD), candidate(GOOD)],
                    "distill": [candidate(GOOD)]}
        original = MaintenanceLoop.run
        calls = []

        def raise_once(loop, paths, output):
            calls.append(list(paths))
            if len(calls) == 1:
                raise RuntimeError("Synthetic maintenance initialization error")
            return original(loop, paths, output)

        with patch("evo.cli.MaintenanceLoop.run", autospec=True, side_effect=raise_once):
            report, output = self.run_campaign(fixtures, {"tasks": [{**self.task, "horizon": 4}],
                                                           "maintain_after_task": True})
        self.assertEqual(len(calls), 2)
        self.assertEqual(report["status"], "validation_failed", report)
        self.assertEqual(report["task_status"], "complete")
        self.assertEqual(report["maintenance_status"], "failed")
        self.assertEqual(report["program_successes"], 1)
        self.assertEqual(report["promoted_versions"], 1)
        self.assertEqual([item["status"] for item in report["maintenance_runs"]], ["error", "complete"])
        self.assertIn("initialization error", report["maintenance_runs"][0]["result"]["error"]["message"])
        self.assertEqual(read_json(output / "summary.json")["maintenance_status"], "failed")

    def test_maintenance_exception_does_not_block_episode_reset(self):
        pre = step(identifier="pre", target=1, predicate="joint_near")
        final = step(identifier="final", target=2, atom="P02", depends=["pre"])
        fixtures = {"plan": [plan(pre, final), plan(step(atom="P02"))],
                    "repair": [candidate(GOOD), candidate(CLAIM), candidate(GOOD)]}
        with patch("evo.cli.MaintenanceLoop.run", side_effect=RuntimeError("Synthetic replay error")) as mocked:
            report, _ = self.run_campaign(fixtures, {"tasks": [{**self.task, "horizon": 4}],
                "maintain_after_task": True, "episode_retry": {"max_episodes": None}})
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(report["total_episode_attempts"], 2)
        self.assertEqual(report["task_status"], "complete")
        self.assertEqual(report["maintenance_status"], "failed")
        self.assertEqual(report["status"], "validation_failed")

    def test_disabled_maintenance_never_invokes_callback_or_episode_fallback(self):
        with patch("evo.cli.MaintenanceLoop.run") as mocked:
            report, _ = self.run_campaign({"plan": [plan()], "repair": [candidate(GOOD)]})
        mocked.assert_not_called()
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["maintenance_status"], "disabled")
        self.assertEqual(report["maintenance_runs"], [])
        self.assertIsNone(report["episodes"][0]["maintenance"])

    def test_duplicate_success_callback_is_handled_once(self):
        path = self.root / "pending" / "one.json"

        def fake_explore(_task, _output, retry_context=None, on_atomic_success=None):
            on_atomic_success(path)
            on_atomic_success(path.parent / "." / path.name)
            return {"status": "success", "native_success": True, "pending": [str(path)]}

        maintained = {"status": "complete", "proposals": [], "promoted": [], "rejected": []}
        with patch("evo.cli.ExplorationLoop.run", side_effect=fake_explore), \
                patch("evo.cli.MaintenanceLoop.run", return_value=maintained) as mocked:
            report, _ = self.run_campaign({}, {"maintain_after_task": True})
        mocked.assert_called_once()
        self.assertEqual(len(report["maintenance_runs"]), 1)

    def test_invalid_maintenance_trigger_is_rejected_before_start(self):
        with self.assertRaises(ValueError):
            self.run_campaign({}, {"maintenance_trigger": "task_success_only"})

    def test_worker_reports_task_and_maintenance_outcomes_separately(self):
        write_json(self.root / "config.json", {"agent": {}})
        report = {"status": "validation_failed", "task_status": "complete", "maintenance_status": "failed",
                  "native_successes": 1, "program_successes": 1, "total_tasks": 1,
                  "promoted_versions": 0, "maintenance_runs": [{"status": "error"}]}
        stdout = io.StringIO()
        worker = Path(__file__).resolve().parents[1] / "scripts" / "worker.py"
        with patch.dict(os.environ, {"EVO_RUN": str(self.root)}), \
                patch("evo.cli.run_config", return_value=report), contextlib.redirect_stdout(stdout), \
                self.assertRaises(SystemExit) as exited:
            runpy.run_path(str(worker), run_name="__main__")
        self.assertEqual(exited.exception.code, 1)
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["task_status"], "complete")
        self.assertEqual(summary["maintenance_status"], "failed")
        self.assertEqual(summary["maintenance_runs_count"], 1)
        self.assertEqual(summary["program_successes"], 1)

    def test_legacy_one_episode_partial_maintenance_is_preserved(self):
        pre = step(identifier="pre", target=1, predicate="joint_near")
        final = step(identifier="final", target=2, atom="P02", depends=["pre"])
        fixtures = {"plan": [plan(pre, final)], "repair": [candidate(GOOD), candidate(CLAIM)], "distill": [candidate(GOOD)]}
        report, _ = self.run_campaign(fixtures, {"tasks": [{**self.task, "horizon": 4}], "maintain_after_task": True,
                                                "episode_retry": "legacy"})
        self.assertEqual(len(report["episodes"][0]["episode_attempts"]), 1)
        self.assertEqual(Library(self.root / "library").active("P01")["version"], 1)

    def test_maintenance_failure_does_not_reset_successful_task(self):
        fixtures = {"plan": [plan()], "repair": [candidate(GOOD)], "distill": [candidate(BAD)]}
        report, _ = self.run_campaign(fixtures, {"maintain_after_task": True,
                                                "episode_retry": {"max_episodes": None}})
        self.assertEqual(report["status"], "validation_failed")
        self.assertEqual(len(report["episodes"][0]["episode_attempts"]), 1)
        self.assertEqual([request["stage"] for request in self.provider.requests], ["plan", "repair", "distill"])

    def test_next_task_runs_after_first_task_budget_exhaustion(self):
        second = {**self.task, "seed": 18, "goal": 3}
        fixtures = {"plan": [plan(), plan(), plan(step(target=3))],
                    "repair": [candidate(BAD), candidate(BAD), candidate(GOOD)]}
        report, _ = self.run_campaign(fixtures, {"tasks": [self.task, second]})
        self.assertEqual([len(record["episode_attempts"]) for record in report["episodes"]], [2, 1])
        self.assertEqual(report["episodes"][1]["exploration"]["status"], "success")
        self.assertEqual(report["native_successes"], 1)
        self.assertEqual([adapter.config["seed"] for adapter in self.factory.instances], [17, 17, 18])
        contexts = [request["context"] for request in self.provider.requests if request["stage"] == "plan"]
        self.assertEqual(contexts[-1]["episode_retry"]["previous_episodes"], [])

    def test_external_experience_requires_matching_task_and_preserves_failed_code(self):
        matching, other = self.root / "old-matching", self.root / "old-other"
        failed = {"step": step(), "code": BAD, "dependencies": {}, "success": False,
                  "execution": {"error": {"type": "TimeoutError", "message": "horizon exhausted"}},
                  "verification": {"known": True, "success": False}}
        for directory, task in ((matching, self.task), (other, {**self.task, "seed": 999})):
            write_json(directory / "task.json", task)
            write_json(directory / "summary.json", {"task_config": task, "status": "step_failed",
                       "native_success": False, "completed_steps": [], "failures": [], "stats": {"steps": 1, "horizon": 1}})
            write_json(directory / "attempts" / "plan00_step00_try00" / "record.json", failed)
        report, _ = self.run_campaign({"plan": [plan()], "repair": [candidate(GOOD)]},
                                      {"retry_experiences": [str(matching), str(other)]})
        self.assertEqual(report["status"], "complete")
        context = self.provider.requests[0]["context"]["episode_retry"]
        self.assertEqual(len(context["previous_episodes"]), 1)
        self.assertEqual(context["previous_episodes"][0]["source_directory"], str(matching))
        self.assertEqual(context["previous_episodes"][0]["recent_failures"][0]["code"], BAD)
        repair = [request for request in self.provider.requests if request["stage"] == "repair"][0]
        self.assertEqual(repair["context"]["previous"]["code"], BAD)

    def test_invalid_retry_budget_is_rejected_before_start(self):
        for invalid in (0, -1, True, 1.5, "unlimited"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.run_campaign({}, {"episode_retry": {"max_episodes": invalid}})


if __name__ == "__main__":
    unittest.main()
