"""Protocol tests use an explicit toy plant, never claim RoboCasa task success.

The plant has one measured joint and an independent target verifier. Fixtures
exercise orchestration and promotion; real benchmark checks are separate.
"""

from pathlib import Path
import multiprocessing
import tempfile
import time
import unittest
from unittest.mock import patch

from evo.agent import Agent, ScriptedProvider
from evo.cli import evaluate_frozen, run_config
from evo.execution import SkillDeadlineExceeded, deadline, execute_skill
from evo.io import read_json
from evo.library import Library
from evo.loops import ExplorationLoop, MaintenanceLoop
from evo.runtime_api import RuntimeAPI, API_DESCRIPTION


GOOD = 'def skill(api, params):\n    api.move_joints([params["target"]])\n    return {"motion": "attempted"}\n'
BAD = 'def skill(api, params):\n    api.move_joints([params["target"] / 2])\n    return {"success": True}\n'
CLAIM = 'def skill(api, params):\n    return {"success": True, "target": params["target"]}\n'


def candidate(code):
    return {"code": code, "rationale": "Explicit unit-test fixture", "dependencies": []}


def step(identifier="move", target=2, atom="P01", predicate="native_success", depends=None):
    return {"id": identifier, "atom": atom, "params": {"target": target}, "goal": "Move measured joint to target",
            "postcondition": {"predicate": predicate, "args": {"target": [target]}}, "depends_on": depends or []}


def plan(*steps):
    return {"steps": list(steps or (step(),)), "reasoning": "Explicit unit-test plan"}


class FakeAdapter:
    """A one-joint deterministic test plant with no policy code in its verifier."""
    def __init__(self, config, output):
        self.config = config
        self.output = Path(output)
        self.position = float(config.get("initial", 0))
        self.goal = float(config.get("goal", 2))
        self.steps = 0
        self.closed = False
        self.gripper_closed = True
        self.horizon = config.get("horizon", 20)

    def observe(self):
        return {"instruction": f"Move to {self.goal}", "joint": self.position,
                "gripper_closed": self.gripper_closed,
                "step": self.steps, "horizon": self.horizon, "images": []}

    def move_joints(self, target, duration=2.0, **kwargs):
        if self.steps >= self.horizon:
            raise RuntimeError("Test plant horizon exhausted")
        self.steps += 1
        self.position = float(target[0])
        return {"position": self.position}

    def wait(self, duration=.5):
        if self.steps >= self.horizon:
            raise RuntimeError("Test plant horizon exhausted")
        self.steps += 1
        return {}

    def gripper(self, direction, duration=.5):
        if self.steps >= self.horizon:
            raise RuntimeError("Test plant horizon exhausted")
        self.steps += 1
        self.gripper_closed = direction > 0
        return {"gripper_closed": self.gripper_closed}

    def native_success(self):
        return abs(self.position - self.goal) < 1e-6

    def evaluate(self, postcondition):
        if postcondition["predicate"] == "native_success":
            success = self.native_success()
        elif postcondition["predicate"] == "joint_near":
            success = abs(self.position - postcondition["args"]["target"][0]) < 1e-6
        else:
            return {"known": False, "success": False}
        return {"known": True, "success": success, "measured": self.position}

    def stats(self):
        return {"steps": self.steps, "horizon": self.horizon, "native_success": self.native_success(),
                "terminal": self.steps >= self.horizon or self.native_success()}

    def close(self):
        self.closed = True


class Factory:
    def __init__(self):
        self.instances = []

    def __call__(self, config, output):
        adapter = FakeAdapter(config, output)
        self.instances.append(adapter)
        return adapter


def _maintain_in_process(library_root, experience_path, output, started, entered, release=None):
    """Run real replay deadlines in each process's main thread."""
    class ObservedProvider(ScriptedProvider):
        def generate(self, request):
            if request["stage"] == "distill":
                entered.set()
                if release is not None and not release.wait(10):
                    raise RuntimeError("Test did not release the first maintenance worker")
            return super().generate(request)

    library = Library(library_root)
    provider = ObservedProvider({"distill": [candidate(GOOD)]})
    loop = MaintenanceLoop(library=library, agent=Agent(provider, max_attempts=1),
                           adapter_factory=FakeAdapter, runtime_api=RuntimeAPI,
                           api_description=API_DESCRIPTION,
                           supported_predicates=["native_success", "joint_near"], max_candidates=1)
    started.set()
    loop.run([experience_path], output)


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.library = Library(self.root / "library")
        self.factory = Factory()
        self.sequence = 0

    def tearDown(self):
        self.temporary.cleanup()

    def destination(self, label):
        self.sequence += 1
        return self.root / f"{self.sequence}_{label}"

    def common(self, fixtures, library=None):
        self.provider = ScriptedProvider(fixtures)
        return {"library": library or self.library, "agent": Agent(self.provider, max_attempts=1),
                "adapter_factory": self.factory, "runtime_api": RuntimeAPI,
                "api_description": API_DESCRIPTION, "supported_predicates": ["native_success", "joint_near"]}

    def explore(self, fixtures, task=None, library=None, **options):
        return ExplorationLoop(**self.common(fixtures, library), **{"max_replans": 0, **options}).run(
            task or {"task": "toy_plant", "seed": 1, "goal": 2}, self.destination("exploration"))

    def maintain(self, paths, fixtures, **options):
        return MaintenanceLoop(**self.common(fixtures), **{"max_candidates": 1, **options}).run(paths, self.destination("maintenance"))

    def learn_first(self):
        explored = self.explore({"plan": [plan()], "repair": [candidate(GOOD)]})
        maintained = self.maintain(explored["pending"], {"distill": [candidate(GOOD)]})
        self.assertEqual(maintained["promoted"][0]["version"], 1)
        return explored, maintained

    def test_exploration_failure_is_feedback_then_repaired_without_publish(self):
        result = self.explore({"plan": [plan()], "repair": [candidate(BAD), candidate(GOOD)]})
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["native_success"])
        self.assertEqual(len(result["failures"]), 1)
        self.assertEqual(len(result["pending"]), 1)
        self.assertEqual(len(self.library.lessons("P01")), 1)
        self.assertIsNone(self.library.active("P01"))
        repairs = [request for request in self.provider.requests if request["stage"] == "repair"]
        self.assertEqual(repairs[1]["context"]["observation"]["joint"], 1)
        self.assertFalse(repairs[1]["context"]["failures"][0]["verification"]["success"])
        self.assertTrue(all(adapter.closed for adapter in self.factory.instances))

    def test_claimed_success_does_not_override_host(self):
        result = self.explore({"plan": [plan()], "repair": [candidate(CLAIM)]}, max_attempts=1)
        self.assertEqual(result["status"], "step_failed")
        self.assertFalse(result["native_success"])
        self.assertFalse(result["pending"])
        self.assertIsNone(self.library.active("P01"))

    def test_fresh_validation_rejects_recovery_that_needs_failed_attempt_state(self):
        recovery = ('def skill(api, params):\n    current = api.observe()["joint"]\n'
                    '    api.move_joints([current + params["target"] / 2])\n    return {"motion": "attempted"}\n')
        explored = self.explore({"plan": [plan()], "repair": [candidate(BAD), candidate(recovery)]})
        self.assertTrue(explored["native_success"])
        maintained = self.maintain(explored["pending"], {"distill": [candidate(recovery)]})
        self.assertFalse(maintained["promoted"])
        self.assertEqual(len(maintained["rejected"]), 1)
        self.assertIsNone(self.library.active("P01"))
        self.assertEqual(len(self.factory.instances), 2)
        self.assertEqual(self.factory.instances[0].position, 2)
        self.assertEqual(self.factory.instances[1].position, 1)
        self.assertTrue(all(adapter.closed for adapter in self.factory.instances))

    def test_distill_can_fix_fresh_replay_failure_and_publish(self):
        explored = self.explore({"plan": [plan()], "repair": [candidate(GOOD)]})
        maintained = self.maintain(explored["pending"], {"distill": [candidate(BAD), candidate(GOOD)]}, max_candidates=2)
        self.assertEqual(len(maintained["proposals"]), 2)
        self.assertEqual(len(maintained["promoted"]), 1)
        self.assertEqual(self.library.active("P01")["code"], GOOD)
        self.assertEqual(len(self.factory.instances), 3)

    def test_parallel_maintenance_serializes_same_atom_without_losing_experiences(self):
        first = self.explore({"plan": [plan()], "repair": [candidate(GOOD)]})
        second = self.explore({"plan": [plan(step(target=3))], "repair": [candidate(GOOD)]},
                              task={"task": "toy_plant", "seed": 2, "goal": 3})
        context = multiprocessing.get_context("spawn")
        first_started, first_entered, release = (context.Event() for _ in range(3))
        second_started, second_entered = (context.Event() for _ in range(2))
        first_output, second_output = self.destination("first_maintenance"), self.destination("second_maintenance")
        workers = [
            context.Process(target=_maintain_in_process, args=(str(self.library.root), first["pending"][0],
                            str(first_output), first_started, first_entered, release)),
            context.Process(target=_maintain_in_process, args=(str(self.library.root), second["pending"][0],
                            str(second_output), second_started, second_entered)),
        ]
        try:
            workers[0].start()
            self.assertTrue(first_entered.wait(10), "First worker never began distillation")
            workers[1].start()
            self.assertTrue(second_started.wait(10), "Second worker never reached maintenance")
            self.assertFalse(second_entered.wait(.2), "Same-atom distillation overlapped")
            release.set()
            for worker in workers:
                worker.join(10)
                self.assertEqual(worker.exitcode, 0)
        finally:
            release.set()
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join(10)

        reports = [read_json(path / "summary.json") for path in (first_output, second_output)]
        self.assertEqual([report["status"] for report in reports], ["complete", "complete"])
        self.assertEqual([report["promoted"][0]["version"] for report in reports], [1, 2])
        self.assertEqual([len(report["proposals"]) for report in reports], [1, 1])
        self.assertEqual([len(report["proposals"][0]["validation"]["cases"]) for report in reports], [1, 2])
        active = self.library.active("P01")
        self.assertEqual(set(active["case_ids"]), {case["id"] for case in self.library.cases("P01")})

    def test_unvalidated_case_added_during_replay_does_not_block_promotion(self):
        first = self.explore({"plan": [plan()], "repair": [candidate(GOOD)]})
        second = self.explore({"plan": [plan(step(target=3))], "repair": [candidate(GOOD)]},
                              task={"task": "toy_plant", "seed": 2, "goal": 3})
        experience = read_json(second["pending"][0])
        added_case = {key: experience[key] for key in ("atom", "task_config", "prefix", "step", "suffix",
                                                      "require_native_success", "source_run")}
        promote = self.library.promote
        added = False

        def add_obligation_before_publish(candidate_id):
            nonlocal added
            if not added:
                added = True
                self.library.register_case(added_case)
                self.library.add_lesson("P01", "Another explorer added a new binding")
            return promote(candidate_id)

        with patch.object(self.library, "promote", side_effect=add_obligation_before_publish):
            maintained = self.maintain(first["pending"], {"distill": [candidate(GOOD), candidate(GOOD)]},
                                       max_candidates=2)
        self.assertEqual(maintained["status"], "complete", maintained)
        self.assertEqual(len(maintained["proposals"]), 1)
        self.assertEqual([len(item["validation"]["cases"]) for item in maintained["proposals"]], [1])
        requests = [request["context"] for request in self.provider.requests if request["stage"] == "distill"]
        self.assertEqual([len(request["regression_cases"]) for request in requests], [0])
        self.assertEqual(len(self.library.active("P01")["case_ids"]), 1)
        self.assertEqual(len(self.library.pending_cases("P01")), 1)

    def test_upgrade_must_replay_old_and_new_bindings(self):
        self.learn_first()
        first_hash = self.library.active("P01")["sha256"]
        fresh_library = Library(self.root / "independent-exploration-library")
        new_task = {"task": "toy_plant", "seed": 2, "goal": 3}
        explored = self.explore({"plan": [plan(step(target=3))], "repair": [candidate(GOOD)]},
                                task=new_task, library=fresh_library)
        overfit = ('def skill(api, params):\n    api.move_joints([params["target"] * 0 + 3])\n'
                   '    return {"motion": "attempted"}\n')
        rejected = self.maintain(explored["pending"], {"distill": [candidate(overfit)]})
        self.assertFalse(rejected["promoted"])
        results = rejected["proposals"][0]["validation"]["cases"]
        self.assertEqual(len(results), 2)
        self.assertEqual(sorted(case["success"] for case in results), [False, True])
        self.assertEqual(self.library.active("P01")["sha256"], first_hash)
        accepted = self.maintain(explored["pending"], {"distill": [candidate(GOOD)]})
        self.assertEqual(accepted["promoted"][0]["version"], 2)
        self.assertEqual(len(self.library.active("P01")["case_ids"]), 2)

    def test_prefix_is_replayed_in_each_fresh_environment(self):
        first = step(identifier="pre", target=1, atom="P01", predicate="joint_near")
        second = step(identifier="final", target=2, atom="P02", depends=["pre"])
        use_prefix = ('def skill(api, params):\n    current = api.observe()["joint"]\n'
                      '    api.move_joints([current + params["target"] / 2])\n    return {}\n')
        explored = self.explore({"plan": [plan(first, second)], "repair": [candidate(GOOD), candidate(use_prefix)]})
        self.assertTrue(explored["native_success"])
        maintained = self.maintain([explored["pending"][1]], {"distill": [candidate(use_prefix)]})
        self.assertEqual(maintained["promoted"][0]["atom"], "P02")
        replay = maintained["proposals"][0]["validation"]["cases"][0]
        self.assertEqual(len(replay["prefix"]), 1)
        self.assertEqual(replay["stats"]["steps"], 2)

    def test_entry_trace_retains_failed_preparation_from_previous_atom(self):
        first = step(identifier="pre", target=1, predicate="joint_near")
        final = step(identifier="final", target=2, atom="P02", depends=["pre"])
        prepare = ('def skill(api, params):\n    api.gripper(-1)\n'
                   '    api.move_joints([params["target"] / 2])\n    return {}\n')
        needs_open = ('def skill(api, params):\n'
                      '    if not api.observe()["gripper_closed"]:\n'
                      '        api.move_joints([params["target"]])\n    return {}\n')
        explored = self.explore({"plan": [plan(first, final)],
            "repair": [candidate(prepare), candidate(GOOD), candidate(needs_open)]})
        experience = read_json(explored["pending"][1])
        self.assertEqual(len(experience["entry_trace"]), 2)
        self.assertFalse(experience["entry_trace"][0]["success"])
        self.assertEqual(experience["entry_observation"]["joint"], 1)
        self.assertFalse(experience["entry_observation"]["gripper_closed"])
        maintained = self.maintain([explored["pending"][1]], {"distill": [candidate(needs_open)]})
        self.assertEqual(len(maintained["promoted"]), 1)
        replay = maintained["proposals"][0]["validation"]["cases"][0]
        self.assertEqual(len(replay["prefix"]), 2)
        self.assertEqual(replay["stats"]["steps"], 4)
        request = self.provider.requests[0]["context"]
        self.assertEqual(request["observation"]["joint"], 1)

    def test_pending_entry_bindings_can_change_without_rewriting_success_evidence(self):
        increment = ('def skill(api, params):\n'
                     '    api.move_joints([api.observe()["joint"] + params["delta"]])\n    return {}\n')
        planned = step()
        planned["params"] = {"delta": 1}
        explored = self.explore({"plan": [plan(planned)],
            "repair": [candidate(increment), candidate(increment)]})
        source = read_json(explored["pending"][0])
        self.assertEqual(source["entry_observation"]["joint"], 0)
        self.assertEqual(source["successful_record"]["binding_update"]["observation"]["joint"], 1)
        maintained = self.maintain(explored["pending"], {"distill": [
            {**candidate(increment), "params": {"delta": 2}}]})
        self.assertEqual(len(maintained["promoted"]), 1)
        self.assertEqual(self.provider.requests[0]["context"]["observation"]["joint"], 0)
        case = self.library.cases("P01")[0]
        self.assertEqual(case["step"]["params"], {"delta": 2})
        self.assertEqual(case["step"]["postcondition"], planned["postcondition"])
        self.assertEqual(read_json(explored["pending"][0]), source)

    def test_trace_preserves_failed_program_whose_physical_condition_became_true(self):
        first = step(identifier="pre", target=1, predicate="joint_near")
        final = step(identifier="final", target=2, atom="P02", depends=["pre"])
        moves_then_raises = ('def skill(api, params):\n'
            '    api.move_joints([params["target"]])\n'
            '    if params.get("raise", True):\n        raise ValueError("after motion")\n    return {}\n')
        explored = self.explore({"plan": [plan(first, final)],
            "repair": [candidate(moves_then_raises), candidate(GOOD), candidate(GOOD)]})
        experience = read_json(explored["pending"][1])
        self.assertFalse(experience["entry_trace"][0]["success"])
        self.assertTrue(experience["entry_trace"][0]["verification"]["success"])
        maintained = self.maintain([explored["pending"][1]], {"distill": [candidate(GOOD)]})
        self.assertEqual(len(maintained["promoted"]), 1)

    def test_failed_prefix_case_does_not_poison_later_independent_case(self):
        pre = step(identifier="pre", target=1, predicate="joint_near")
        final = step(identifier="final", target=2, atom="P02", depends=["pre"])
        explored = self.explore({"plan": [plan(pre, final)],
            "repair": [candidate(GOOD), candidate(GOOD)]})
        experience = read_json(explored["pending"][1])
        # Simulate an old recording whose prefix is not replayable.
        experience.pop("entry_trace")
        experience["prefix"][0]["code"] = CLAIM
        from evo.io import write_json
        damaged = self.destination("legacy_experience.json")
        write_json(damaged, experience)
        rejected = self.maintain([damaged], {"distill": [candidate(GOOD)]})
        replay = rejected["proposals"][0]["validation"]["cases"][0]
        self.assertEqual(replay["failure_stage"], "prefix")
        self.assertNotIn("execution", replay)
        self.assertEqual(self.library.cases("P02"), [])
        fresh = self.explore({"plan": [plan(step(atom="P02"))], "repair": [candidate(GOOD)]})
        accepted = self.maintain(fresh["pending"], {"distill": [candidate(GOOD)]})
        self.assertEqual(len(accepted["promoted"]), 1)
        self.assertEqual(len(self.library.cases("P02")), 1)
        self.assertEqual(len(self.library.pending_cases("P02")), 1)

    def test_atomic_success_that_breaks_task_suffix_cannot_be_published(self):
        pre = step(identifier="pre", target=1, predicate="joint_near")
        final = step(identifier="final", target=2, atom="P02", depends=["pre"])
        needs_closed_gripper = ('def skill(api, params):\n'
                                '    if api.observe()["gripper_closed"]:\n'
                                '        api.move_joints([params["target"]])\n'
                                '    return {"motion": "attempted"}\n')
        explored = self.explore({"plan": [plan(pre, final)],
                                 "repair": [candidate(GOOD), candidate(needs_closed_gripper)]})
        self.assertTrue(explored["native_success"])
        continuation_path = Path(explored["pending"][0]).parent.parent / "continuations" / "01_P01.json"
        experience = read_json(continuation_path)
        self.assertTrue(experience["require_native_success"])
        self.assertEqual(len(experience["suffix"]), 1)
        # This reaches precisely the same local joint postcondition, but releases
        # the gripper needed by the rest of the recorded task trajectory.
        breaks_suffix = ('def skill(api, params):\n    api.gripper(-1)\n'
                         '    api.move_joints([params["target"]])\n    return {}\n')
        rejected = self.maintain([continuation_path], {"distill": [candidate(breaks_suffix)]})
        self.assertFalse(rejected["promoted"])
        replay = rejected["proposals"][0]["validation"]["cases"][0]
        self.assertTrue(replay["atomic_success"])
        self.assertTrue(replay["verification"]["success"])
        self.assertFalse(replay["success"])
        self.assertFalse(replay["native_success"])
        self.assertEqual(replay["error"], "Downstream task replay failed")
        self.assertEqual(len(replay["suffix"]), 1)
        self.assertIsNone(self.library.active("P01"))
        accepted = self.maintain([continuation_path], {"distill": [candidate(GOOD)]})
        self.assertEqual(accepted["promoted"][0]["atom"], "P01")

    def test_repair_can_replace_scene_bindings_without_changing_goal(self):
        planned = step(target=2)
        planned["params"] = {"target": 1, "stale_binding": "must be replaced"}
        corrected = {**candidate(GOOD), "params": {"target": 2, "offset": 0}}
        explored = self.explore({"plan": [plan(planned)], "repair": [candidate(GOOD), corrected]})
        self.assertTrue(explored["native_success"], explored)
        self.assertEqual(len(explored["failures"]), 1)
        record = explored["completed_steps"][0]
        self.assertEqual(record["step"]["params"], corrected["params"])
        self.assertEqual(record["step"]["goal"], planned["goal"])
        self.assertEqual(record["step"]["postcondition"], planned["postcondition"])
        self.assertTrue(record["binding_update"]["changed"])
        self.assertEqual(record["binding_update"]["previous_params"], planned["params"])
        self.assertEqual(record["binding_update"]["observation"]["joint"], 1)
        self.assertEqual(self.factory.instances[0].position, 2)
        self.assertEqual(record["code"], GOOD)
        experience = read_json(explored["pending"][0])
        self.assertEqual(experience["step"]["params"], corrected["params"])
        maintained = self.maintain(explored["pending"], {"distill": [candidate(GOOD)]})
        self.assertEqual(len(maintained["promoted"]), 1)
        stored_case = self.library.cases("P01")[0]
        self.assertEqual(stored_case["step"]["params"], corrected["params"])
        self.assertEqual(stored_case["step"]["postcondition"], planned["postcondition"])
        self.assertIn('params["target"]', self.library.active("P01")["code"])

    def test_suffix_verifier_exception_cannot_leave_atomic_success_promotable(self):
        pre = step(identifier="pre", target=1, predicate="joint_near")
        final = step(identifier="final", target=2, atom="P02", depends=["pre"])
        explored = self.explore({"plan": [plan(pre, final)],
                                 "repair": [candidate(GOOD), candidate(GOOD)]})
        self.assertTrue(explored["native_success"])

        class BrokenSuffixVerifier(FakeAdapter):
            def evaluate(self, postcondition):
                if postcondition["predicate"] == "native_success" and self.steps >= 2:
                    raise RuntimeError("Synthetic suffix verifier failure")
                return super().evaluate(postcondition)

        self.factory = BrokenSuffixVerifier
        continuation_path = Path(explored["pending"][0]).parent.parent / "continuations" / "01_P01.json"
        result = self.maintain([continuation_path], {"distill": [candidate(GOOD)]})
        self.assertFalse(result["promoted"])
        replay = result["proposals"][0]["validation"]["cases"][0]
        self.assertTrue(replay["atomic_success"])
        self.assertFalse(replay["success"])
        self.assertEqual(replay["error"]["type"], "RuntimeError")
        self.assertIsNone(self.library.active("P01"))

    def test_published_skill_is_reused_without_requesting_code(self):
        self.learn_first()
        result = self.explore({"plan": [plan(step(target=3))]}, task={"task": "toy_plant", "seed": 2, "goal": 3})
        self.assertTrue(result["native_success"])
        self.assertTrue(result["completed_steps"][0]["reused"])
        self.assertEqual(len(result["pending"]), 1)
        self.assertEqual([request["stage"] for request in self.provider.requests], ["plan"])

    def test_reused_active_skill_registers_new_regression_binding(self):
        self.learn_first()
        self.assertEqual(len(self.library.cases("P01")), 1)
        result = self.explore({"plan": [plan(step(target=3))]}, task={"task": "toy_plant", "seed": 2, "goal": 3})
        self.assertTrue(result["native_success"])
        self.assertEqual(len(self.library.cases("P01")), 1)
        maintained = self.maintain(result["pending"], {})
        self.assertEqual(maintained["promoted"], [])
        self.assertEqual(len(maintained["accepted_cases"]), 1)
        self.assertEqual(len(self.library.cases("P01")), 2)
        self.assertEqual(self.library.active("P01")["version"], 1)

    def test_already_satisfied_step_is_not_learned(self):
        already_done = step(identifier="already", target=0, predicate="joint_near")
        final = step(identifier="final", target=2, atom="P02", depends=["already"])
        result = self.explore({"plan": [plan(already_done, final)], "repair": [candidate(GOOD)]})
        self.assertTrue(result["native_success"], result)
        self.assertEqual(len(result["pending"]), 1)
        self.assertIsNone(self.library.active("P01"))
        repairs = [request for request in self.provider.requests if request["stage"] == "repair"]
        self.assertEqual(len(repairs), 1)
        self.assertEqual(repairs[0]["context"]["step"]["atom"], "P02")

    def test_frozen_evaluation_does_not_change_library_files(self):
        self.learn_first()
        before = {str(path.relative_to(self.library.root)): path.read_bytes()
                  for path in self.library.root.rglob("*") if path.is_file()}
        config = {"library": str(self.library.root), "evaluations": [
            {"task": {"task": "toy_plant", "seed": 22, "goal": 3}, "plan": plan(step(target=3))}]}
        with patch("evo.cli.adapter_factory", self.factory):
            result = evaluate_frozen(config, self.destination("evaluation"))
        after = {str(path.relative_to(self.library.root)): path.read_bytes()
                 for path in self.library.root.rglob("*") if path.is_file()}
        self.assertEqual(result["native_successes"], 1)
        self.assertTrue(result["library_unchanged"])
        self.assertEqual(before, after)

    def test_frozen_evaluation_skips_satisfied_unimplemented_prefix(self):
        self.learn_first()
        satisfied = step(identifier="satisfied", target=0, atom="P02", predicate="joint_near")
        final = step(identifier="final", target=2, depends=["satisfied"])
        config = {"library": str(self.library.root), "evaluations": [
            {"task": {"task": "toy_plant", "goal": 2}, "plan": plan(satisfied, final)}]}
        before = self.library.state_path.read_bytes()
        with patch("evo.cli.adapter_factory", self.factory):
            result = evaluate_frozen(config, self.destination("evaluation"))
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["evaluations"][0]["success"])
        self.assertTrue(result["evaluations"][0]["steps"][0]["skipped"])
        self.assertEqual(self.factory.instances[-1].steps, 1)
        self.assertIsNone(self.library.active("P02"))
        self.assertEqual(before, self.library.state_path.read_bytes())

    def test_frozen_final_stats_failure_cannot_report_success(self):
        self.learn_first()

        class BrokenFinalStats(FakeAdapter):
            def stats(self):
                self.stats_calls = getattr(self, "stats_calls", 0) + 1
                # SkillExecutor obtains before/after stats; the third read is
                # the evaluator's final report, after the task verdict is true.
                if self.stats_calls == 3:
                    raise RuntimeError("Synthetic final stats failure")
                return super().stats()

        config = {"library": str(self.library.root), "evaluations": [
            {"task": {"task": "toy_plant", "goal": 2}, "plan": plan()}]}
        with patch("evo.cli.adapter_factory", BrokenFinalStats):
            result = evaluate_frozen(config, self.destination("evaluation"))
        self.assertEqual(result["status"], "evaluation_failed")
        self.assertEqual(result["native_successes"], 0)
        self.assertFalse(result["evaluations"][0]["success"])
        self.assertEqual(result["evaluations"][0]["error"]["type"], "RuntimeError")

    def test_frozen_close_failure_marks_evaluation_failed_and_keeps_report(self):
        self.learn_first()

        class BrokenClose(FakeAdapter):
            def close(self):
                super().close()
                raise RuntimeError("Synthetic close failure")

        config = {"library": str(self.library.root), "evaluations": [
            {"task": {"task": "toy_plant", "goal": 2}, "plan": plan()}]}
        destination = self.destination("evaluation")
        with patch("evo.cli.adapter_factory", BrokenClose):
            result = evaluate_frozen(config, destination)
        self.assertEqual(result["status"], "evaluation_failed")
        self.assertEqual(result["native_successes"], 0)
        self.assertFalse(result["evaluations"][0]["success"])
        self.assertEqual(result["evaluations"][0]["close_error"]["type"], "RuntimeError")
        self.assertEqual(read_json(destination / "000" / "report.json"), result["evaluations"][0])
        self.assertEqual(read_json(destination / "summary.json")["status"], "evaluation_failed")

    def test_exhausted_horizon_stops_model_requests(self):
        result = self.explore({"plan": [plan()], "repair": [candidate(BAD)]},
                              task={"task": "toy_plant", "goal": 2, "horizon": 1}, max_attempts=5, max_replans=3)
        self.assertEqual(result["status"], "step_failed")
        self.assertEqual([request["stage"] for request in self.provider.requests], ["plan", "repair"])

    def test_reset_context_repairs_old_code_without_replaying_old_progress(self):
        retry = dict(episode_attempt=2, reset_to_initial_state=True, previous_episodes=[
            dict(source_run="old-episode", recent_failures=[dict(step=step(target=99), code=BAD)])])
        output = self.destination("retry")
        common = self.common({"plan": [plan()], "repair": [candidate(GOOD)]})
        result = ExplorationLoop(**common, max_attempts=1, max_replans=0).run(
            {"task":"toy_plant", "goal":2}, output, retry_context=retry)
        self.assertEqual(result["status"], "success")
        self.assertEqual(self.factory.instances[-1].steps, 1)
        request = self.provider.requests[1]["context"]
        self.assertEqual(request["previous"]["code"], BAD)
        self.assertTrue(request["previous"]["requires_fresh_grounding"])
        self.assertEqual(request["step"]["params"], {"target":2})
        self.assertEqual(request["completed_steps"], [])
        self.assertEqual(request["observation"]["joint"], 0)
        self.assertEqual(read_json(output/"retry_context.json"), retry)

    def test_empty_replan_ends_episode_as_retryable_failure(self):
        result = self.explore({"plan": [plan(), {"steps":[], "reasoning":"Restart this scene"}],
                               "repair":[candidate(BAD)]}, max_attempts=1, max_replans=1)
        self.assertEqual(result["status"], "plan_complete_task_failed")
        self.assertEqual(result["stop_reason"], "planner_returned_empty_suffix")
        self.assertFalse(result["native_success"])
        self.assertTrue(all(a.closed for a in self.factory.instances))

    def test_empty_initial_plan_feedback_survives_reset_then_validated_task_succeeds(self):
        empty = {"steps": [], "reasoning": "No safe drawer motion identified; re-ground after reset."}
        common = self.common({"plan": [empty, plan()], "repair": [candidate(GOOD)],
                              "distill": [candidate(GOOD)]})
        task = {"task": "toy_plant", "seed": 17, "goal": 2, "horizon": 1}
        config = {"library": str(self.library.root), "agent": {}, "tasks": [task],
                  "exploration": {"max_attempts": 1, "max_replans": 0},
                  "maintenance": {"max_candidates": 1}, "maintain_after_task": True,
                  "episode_retry": {"max_episodes": 2, "history_episodes": 3}}
        with patch("evo.cli.create_agent", return_value=common["agent"]), \
                patch("evo.cli.adapter_factory", self.factory):
            campaign = run_config(config, self.destination("empty_initial_campaign"))
        attempts = campaign["episodes"][0]["episode_attempts"]
        self.assertEqual([attempt["status"] for attempt in attempts], ["plan_complete_task_failed", "success"])
        first_path = Path(attempts[0]["summary_path"])
        first = read_json(first_path)
        feedback = {"kind": "planner_returned_empty_initial_plan", "summary": empty["reasoning"]}
        self.assertEqual(read_json(first_path.parent / "plan_0.json"), empty)
        self.assertEqual(first["stop_reason"], feedback)
        self.assertEqual(first["stats"]["steps"], 0)
        self.assertEqual(first["final_observation"]["joint"], 0)
        self.assertFalse(first["native_success"])
        self.assertEqual(first["failures"], [])
        self.assertEqual(first["completed_steps"], [])
        self.assertEqual(first["pending"], [])
        self.assertFalse((first_path.parent / "attempts").exists())
        self.assertEqual([request["stage"] for request in self.provider.requests],
                         ["plan", "plan", "repair", "distill"])
        retry_plan = self.provider.requests[1]["context"]
        self.assertEqual(retry_plan["initial_observation"]["joint"], 0)
        history = retry_plan["episode_retry"]["previous_episodes"][0]
        self.assertEqual(history["stop_reason"], feedback)
        self.assertEqual(history["recent_failures"], [])
        self.assertEqual([adapter.config for adapter in self.factory.instances[:2]], [task, task])
        self.assertTrue(all(adapter.closed for adapter in self.factory.instances))
        self.assertEqual(len(self.factory.instances), 3)  # Two episodes and independent replay.
        self.assertEqual(self.library.lessons("P01"), [])
        self.assertEqual(campaign["status"], "complete")
        self.assertEqual(campaign["promoted_versions"], 1)
        self.assertEqual(self.library.active("P01")["code"], GOOD)

    def test_initial_planner_provider_error_remains_nonretryable(self):
        common = self.common({"plan": []})  # ProviderError from an exhausted backend fixture.
        config = {"library": str(self.library.root), "agent": {},
                  "tasks": [{"task": "toy_plant", "seed": 17, "goal": 2}],
                  "episode_retry": {"max_episodes": 2}, "maintain_after_task": True}
        with patch("evo.cli.create_agent", return_value=common["agent"]), \
                patch("evo.cli.adapter_factory", self.factory):
            campaign = run_config(config, self.destination("provider_error_campaign"))
        episode = campaign["episodes"][0]
        self.assertEqual(episode["retry_status"], "not_retryable")
        self.assertEqual(len(episode["episode_attempts"]), 1)
        self.assertEqual(episode["exploration"]["status"], "error")
        self.assertEqual(episode["exploration"]["error"]["type"], "ProviderError")
        self.assertNotIn("stop_reason", episode["exploration"])
        self.assertIsNone(episode["maintenance"])
        self.assertEqual(self.library.list_active(), [])
        self.assertEqual(len(self.factory.instances), 1)
        self.assertTrue(self.factory.instances[0].closed)

    def test_failed_active_skill_is_repaired_before_execution_after_reset(self):
        self.learn_first()
        retry = dict(episode_attempt=2, reset_to_initial_state=True, previous_episodes=[
            dict(source_run="old-episode", recent_failures=[dict(step=step(), code=BAD, success=False)])])
        common = self.common({"plan":[plan()], "repair":[candidate(GOOD)]})
        result = ExplorationLoop(**common, max_attempts=1).run(
            {"task":"toy_plant", "goal":2}, self.destination("retry_active"), retry_context=retry)
        self.assertEqual(result["status"], "success")
        self.assertEqual([r["stage"] for r in self.provider.requests], ["plan", "repair"])
        self.assertEqual(self.provider.requests[1]["context"]["previous"]["code"], BAD)
        self.assertFalse(result["completed_steps"][0]["reused"])

    def test_exploration_close_failure_is_saved_and_cannot_report_success(self):
        output = self.destination("close_error")
        common = self.common({"plan":[plan()], "repair":[candidate(GOOD)]})
        class CloseFailure(FakeAdapter):
            def close(self):
                super().close()
                raise RuntimeError("Synthetic close failure")
        common["adapter_factory"] = CloseFailure
        result = ExplorationLoop(**common).run({"task":"toy_plant", "goal":2}, output)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["close_error"]["type"], "RuntimeError")
        self.assertEqual(read_json(output/"summary.json")["status"], "error")

    def test_replan_can_depend_on_already_completed_prefix(self):
        pre = step(identifier="pre", target=1, predicate="joint_near")
        fail = step(identifier="failed", target=2, atom="P02", depends=["pre"])
        suffix = step(identifier="replanned", target=2, atom="P02", depends=["pre"])
        result = self.explore({"plan": [plan(pre, fail), plan(suffix)],
                               "repair": [candidate(GOOD), candidate(CLAIM), candidate(GOOD)]},
                              max_attempts=1, max_replans=1)
        self.assertEqual(result["status"], "success", result)
        self.assertTrue(result["native_success"])
        self.assertEqual(result["replans"], 1)

    def test_contract_error_is_repaired_within_attempt_budget(self):
        invalid = 'def skill(api, params):\n    import os\n    return params\n'
        result = self.explore({"plan": [plan()], "repair": [candidate(invalid), candidate(GOOD)]}, max_attempts=2)
        self.assertEqual(result["status"], "success", result)
        self.assertTrue(result["native_success"])


class ExecutionBudgetTests(unittest.TestCase):
    def test_nested_deadline_preserves_total_outer_budget(self):
        started = time.monotonic()
        timed_out = False
        try:
            with deadline(.18):
                with deadline(.12):
                    time.sleep(.10)
                time.sleep(.10)
        except (TimeoutError, SkillDeadlineExceeded):
            timed_out = True
        self.assertTrue(timed_out, "Nested call restored spent outer time and exceeded total budget")
        self.assertLess(time.monotonic() - started, .24)

    def test_generated_return_must_follow_dictionary_contract(self):
        api = RuntimeAPI(FakeAdapter({}, "/unused"))
        result = execute_skill('def skill(api, params):\n    return params.get("value", 1)\n', api, {})
        self.assertFalse(result["ok"])

    def test_long_child_deadline_cannot_extend_short_parent(self):
        started = time.monotonic()
        expired = False
        try:
            with deadline(.08):
                with deadline(.5):
                    time.sleep(.15)
        except BaseException as error:
            if type(error).__name__ not in ("TimeoutError", "SkillDeadlineExceeded"):
                raise
            expired = True
        self.assertTrue(expired)
        self.assertLess(time.monotonic() - started, .14)

    def test_except_exception_does_not_swallow_deadline(self):
        code = ('def skill(api, params):\n    try:\n        while params["run"]:\n'
                '            pass\n    except Exception:\n        return {}\n    return {}\n')
        result = execute_skill(code, RuntimeAPI(FakeAdapter({}, "/unused")), {"run": True}, timeout_s=.03)
        self.assertFalse(result["ok"])
        self.assertIn(result["error"]["type"], ("SkillDeadlineExceeded", "TimeoutError"))


if __name__ == "__main__":
    unittest.main()
