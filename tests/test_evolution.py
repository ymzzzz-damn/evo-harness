"""Comparison admission and reuse of parameterized failures on a toy plant."""
from unittest.mock import patch

import pytest

from evo.agent import Agent, ResponseError, ScriptedProvider
from evo.cli import run_config, _episode_history
from evo.io import read_json
from evo.library import Library, LibraryError
from evo.loops import ExplorationLoop, MaintenanceLoop
from evo.policy_evaluation import FrozenLibrary, freeze_active_library
from evo.runtime_api import API_DESCRIPTION, RuntimeAPI
from scripts.summarize_parallel_campaign import _tool_report
from test_loops import GOOD, BAD, Factory, candidate, plan, step


PARAMETERIZED = ('def skill(api, params):\n'
                 '    api.move_joints([params["target"] * params["fraction"]])\n'
                 '    return {"motion": "attempted"}\n')


def consolidation(code=GOOD, decision="update", target=2):
    return dict(code=code, description="Toy motion: target in plant units; fraction scales travel. Unverified until host validation.", rationale="Merge measured motion behavior while keeping the parameter contract",
                dependencies=[], params={"target": target}, decision=decision,
                progress={"completed": ["Motion attempted; exact partial completion unknown"],
                          "remaining": ["Verify the goal from the new entry"],
                          "entry_requirements": ["Re-ground current pose and target after reset"]})


def failed_consolidation(target=2):
    response = consolidation(PARAMETERIZED, "draft", target)
    response["params"]["fraction"] = .5
    response["rationale"] = "target is the new scene's joint goal; fraction is the fraction of that motion to attempt"
    response["failure_analysis"] = dict(
        summary="The host goal was not reached; the measured joint stopped short of the target.",
        hypotheses=["The attempted travel may have been too short; check current measurements."],
        next_changes=["Re-ground target in the new scene and revise fraction using current feedback."])
    return response


class Harness:
    def __init__(self, root):
        self.root, self.library, self.factory, self.sequence = root, Library(root / "library"), Factory(), 0

    def path(self, kind):
        self.sequence += 1
        return self.root / f"{self.sequence}_{kind}"

    def common(self, fixtures):
        self.provider = ScriptedProvider(fixtures)
        return dict(library=self.library, agent=Agent(self.provider, max_attempts=1), adapter_factory=self.factory,
                    runtime_api=RuntimeAPI, api_description=API_DESCRIPTION,
                    supported_predicates=["native_success", "joint_near"])

    def explore(self, fixtures, goal=2, horizon=20, task="toy", retry_context=None, **options):
        return ExplorationLoop(**self.common(fixtures), **dict(max_attempts=options.pop("max_attempts", 1),
            max_replans=0, record_working_memory=True, plan_one_step=True, **options)).run(
                {"task": task, "seed": goal, "goal": goal, "horizon": horizon}, self.path("exploration"),
                retry_context=retry_context)

    def maintain(self, result, fixtures):
        return MaintenanceLoop(**self.common(fixtures), admission_policy="compare_existing", max_candidates=1).run(
            result["pending"] + result.get("incomplete", []), self.path("maintenance"))

    def learn(self, code=GOOD, target=2):
        result = self.explore({"plan": [plan(step(target=target))], "repair": [candidate(code)]})
        assert result["status"] == "success", result
        outcome = self.maintain(result, {"distill": [candidate(code)]})
        assert outcome["status"] == "complete", outcome
        return result, outcome


@pytest.fixture
def h(tmp_path):
    return Harness(tmp_path)


def test_empty_slot_always_requires_first_use_replay(h):
    _, report = h.learn()
    assert report["replay_counts"] == {"total": 1, "fresh": 1, "cached": 0}
    assert h.library.active("A02")["validation"]["validation_kind"] == "fresh_environment_replay"
    assert len(h.factory.instances) == 2


def test_failed_first_use_preserves_parameterized_draft_without_activating_it(h):
    result = h.explore({"plan": [plan()], "repair": [candidate(GOOD)]})
    report = h.maintain(result, {"distill": [candidate(BAD)]})
    assert report["drafts"] and report["rejected"]
    assert h.library.active("A02") is None
    draft = h.library.working_memory("A02")[0]
    assert draft["kind"] == "draft" and draft["payload"]["code"] == BAD
    assert not h.library.cases("A02")
    assert report["replay_counts"]["fresh"] == 1


@pytest.mark.parametrize("format_change", [False, True])
def test_existing_equivalent_code_keeps_version_without_replay_or_model(h, format_change):
    h.learn()
    result = h.explore({"plan": [plan(step(target=3))]}, goal=3)
    if format_change:
        experience = read_json(result["pending"][0])
        from evo.io import write_json, sha256
        experience["successful_record"]["code"] += "\n# equivalent formatting\n"
        experience["successful_record"]["sha256"] = sha256(experience["successful_record"]["code"])
        write_json(result["pending"][0], experience)
    before = len(h.factory.instances)
    report = h.maintain(result, {})
    assert report["kept"] and not report["promoted"]
    assert report["replay_counts"]["total"] == 0 and len(h.factory.instances) == before
    assert not h.provider.requests
    assert h.library.active("A02")["version"] == 1
    assert len(h.library.cases("A02")) == 1  # No fabricated fresh case admission.


def test_changed_existing_skill_updates_by_comparison_and_exports_honest_evidence(h):
    h.learn(BAD, target=4)  # BAD(4) reaches this first task's native goal of 2.
    result = h.explore({"plan": [plan(step(target=3))], "repair": [candidate(GOOD)]}, goal=3, max_attempts=2)
    before = len(h.factory.instances)
    report = h.maintain(result, {"consolidate": [consolidation(target=3)]})
    assert report["status"] == "complete", report
    assert len(h.factory.instances) == before and report["replay_counts"]["total"] == 0
    active = h.library.active("A02")
    assert active["version"] == 2 and active["code"] == GOOD
    assert active["validation"]["validation_kind"] == "code_comparison"
    assert active["validation"]["replay_verified"] is False and active["case_ids"] == []
    request = h.provider.requests[0]
    assert request["context"]["source_success"] and request["context"]["code_diff"]
    assert request["context"]["active"]["code"] == BAD
    issues = []
    tools = _tool_report(h.library.root, None, [], issues)
    assert not issues
    assert tools["counts"]["comparison_admitted_versions"] == 1
    assert tools["active_validated_tools"][0]["replay_verified"] is False
    receipt = freeze_active_library(h.library.root, h.root / "frozen")
    frozen = FrozenLibrary(h.root / "frozen", receipt["state_sha256"])
    assert frozen.proofs[("A02", "2")]["validation_kind"] == "code_comparison"
    assert frozen.proofs[("A02", "2")]["replay_verified"] is False
    from scripts.prepare_policy_evaluation import _check_frozen_memory, HandoffNotReady, _sha
    assert _check_frozen_memory(h.root / "frozen", receipt)["training_memory_empty"] is True
    # The handoff must accept the weaker proof honestly, while rejecting both
    # fabricated replay claims and training context added to its compact proof.
    state_path = h.root / "frozen/state.json"
    original = state_path.read_bytes()
    for field, value in (("replay_verified", True), ("source_evidence", {
            **frozen.proofs[("A02", "2")]["source_evidence"], "prompt": "training text"})):
        state = read_json(state_path)
        state["versions"]["A02"]["2"]["validation"][field] = value
        state_path.chmod(0o644)
        state_path.write_text(Library._state_text(state))
        backup = h.root / "frozen/state.backup.json"
        backup.chmod(0o644)
        backup.write_bytes(state_path.read_bytes())
        with pytest.raises(HandoffNotReady):
            _check_frozen_memory(h.root / "frozen", {**receipt, "state_sha256": _sha(state_path)})
        state_path.write_bytes(original)
        backup.write_bytes(original)
    scratch = frozen.scratch(h.root / "scratch")
    assert scratch.working_catalog() == []
    with pytest.raises(LibraryError, match="disabled"):
        scratch.publish_comparison("irrelevant", "irrelevant", "irrelevant")


def test_failed_code_is_parameterized_with_reasons_then_rebound_in_new_scene(h):
    raw = 'def skill(api, params):\n    api.move_joints([1])\n    return {"target": params["target"]}\n'
    failed = h.explore({"plan": [plan()], "repair": [candidate(raw), candidate(raw)]}, max_attempts=2)
    assert failed["status"] == "step_failed" and len(failed["attempt_memories"]) == 2
    before = len(h.factory.instances)
    report = h.maintain(failed, {"consolidate": [failed_consolidation()]})
    assert report["retained_failures"] and report["status"] == "complete"
    assert report["drafts"] and not report["promoted"]
    assert [r["stage"] for r in h.provider.requests] == ["consolidate"]
    assert h.provider.requests[0]["context"]["failure_feedback"]["kind"] == "postcondition_not_met"
    assert report["replay_counts"]["total"] == 0
    assert len(h.factory.instances) == before and h.library.active("A02") is None
    memories = h.library.working_memory("A02")
    assert len(memories) == 3 and memories[0]["kind"] == "draft"
    draft = memories[0]["payload"]
    assert draft["code"] == PARAMETERIZED and draft["code"] != raw
    assert draft["params"] == {"target": 2, "fraction": .5}
    assert draft["source_memory_id"] == failed["attempt_memories"][-1]
    assert draft["failure_analysis"] == failed_consolidation()["failure_analysis"]
    assert draft["source_success"] is False and draft["parameterized"] is True
    assert memories[1]["payload"]["progress"]["host_verified_goal"] is False
    assert memories[1]["payload"]["progress"]["completed"] == []
    rebound = {**candidate(PARAMETERIZED), "params": {"target": 3, "fraction": 1}}
    succeeded = h.explore({"plan": [plan(step(target=3))], "repair": [rebound]}, goal=3, task="another_task")
    repair = [r for r in h.provider.requests if r["stage"] == "repair"][0]["context"]
    assert repair["previous"]["code"] == PARAMETERIZED and repair["previous"]["requires_fresh_grounding"]
    assert repair["previous"]["failure_analysis"] == draft["failure_analysis"]
    assert repair["working_memory"][0]["kind"] == "draft"
    assert repair["previous"]["failure_feedback"]["kind"] == "postcondition_not_met"
    assert repair["observation"]["joint"] == 0 and repair["completed_steps"] == []
    assert succeeded["native_success"]
    assert succeeded["completed_steps"][0]["step"]["params"] == {"target": 3, "fraction": 1}
    report = h.maintain(succeeded, {"distill": [candidate(PARAMETERIZED)]})
    assert report["replay_counts"]["fresh"] == 1


def test_parameterized_failure_cannot_replace_existing_active_even_through_library_api(h):
    h.learn()
    failed = h.explore({"plan": [plan(step(target=2))]}, goal=4)
    report = h.maintain(failed, {"consolidate": [failed_consolidation()]})
    assert report["retained_failures"] and report["drafts"]
    assert report["replay_counts"]["total"] == 0
    assert h.library.active("A02")["version"] == 1
    c = h.library.propose("A02", BAD)
    with pytest.raises(LibraryError, match="Incomplete"):
        h.library.publish_comparison(c["id"], failed["attempt_memories"][0], "bad replacement")
    assert h.library.active("A02")["code"] == GOOD


def test_failed_consolidation_does_not_erase_attempts(h):
    h.learn(BAD, target=4)
    succeeded = h.explore({"plan": [plan(step(target=3))], "repair": [candidate(GOOD)]}, goal=3, max_attempts=2)
    report = h.maintain(succeeded, {})  # Successful change still needs comparison.
    assert report["errors"] and report["status"] == "some_candidates_rejected"
    memory = h.library.memory("A02", succeeded["attempt_memories"][-1])
    assert memory["payload"]["code"] == GOOD and memory["status"] == "observed_success"
    assert h.library.active("A02")["code"] == BAD


def test_failed_parameterization_keeps_raw_failure_available_for_repair(h):
    failed = h.explore({"plan": [plan()], "repair": [candidate(BAD)]})
    report = h.maintain(failed, {})
    assert report["errors"] and not report["drafts"]
    assert len(h.library.working_memory("A02")) == 1
    result = h.explore({"plan": [plan()], "repair": [candidate(GOOD)]}, task="new_task")
    repair = [r for r in h.provider.requests if r["stage"] == "repair"][0]["context"]
    assert repair["previous"]["code"] == BAD and repair["previous"]["failure_feedback"]
    assert result["native_success"]


def test_provider_interruption_is_recorded_without_claiming_an_execution(h):
    result = h.explore({"plan": [plan()], "repair": [candidate(BAD)]}, max_attempts=2)
    assert result["status"] == "error" and len(result["attempt_memories"]) == 2
    interrupted = h.library.memory("A02", result["attempt_memories"][-1])["payload"]["record"]
    assert interrupted["executed"] is False and interrupted["verification"]["known"] is False
    assert interrupted["code"] == BAD
    assert interrupted["failure_feedback"]["kind"] == "attempt_not_executed"
    assert result["incomplete"] and h.factory.instances[0].steps == 1


@pytest.mark.parametrize("next_task,with_retry_history", [("toy", False), ("another_task", False), ("toy", True)])
def test_parameterized_horizon_failure_is_used_before_active_or_raw_retry_code(h, next_task, with_retry_history):
    h.learn(BAD, target=4)
    failed = h.explore({"plan": [plan(step(target=3))]}, goal=3, horizon=1)
    assert failed["status"] == "step_failed"
    memory = h.library.memory("A02", failed["attempt_memories"][0])["payload"]
    assert memory["code"] == BAD
    assert memory["failure_feedback"]["kind"] == "horizon_exhausted"
    assert memory["failure_feedback"]["steps"] == memory["failure_feedback"]["horizon"] == 1
    assert memory["failure_feedback"]["remaining_steps"] == 0
    response = failed_consolidation(3)
    response["failure_analysis"]["summary"] = "Horizon was exhausted at step 1 before the host goal was reached."
    report = h.maintain(failed, {"consolidate": [response]})
    assert report["drafts"] and report["replay_counts"]["total"] == 0
    retry = {"previous_episodes": [_episode_history(memory["source_run"], failed)]} if with_retry_history else None
    result = h.explore({"plan": [plan(step(target=3))], "repair": [candidate(GOOD)]},
                       goal=3, horizon=1, task=next_task, retry_context=retry)
    assert result["native_success"] is True
    assert result["completed_steps"][0]["reused"] is False
    repair = [r for r in h.provider.requests if r["stage"] == "repair"][0]["context"]
    assert repair["previous"]["from_working_memory"] and repair["observation"]["joint"] == 0
    assert repair["previous"]["failure_feedback"]["kind"] == "horizon_exhausted"
    assert repair["previous"]["code"] == PARAMETERIZED
    assert repair["previous"]["failure_analysis"] == response["failure_analysis"]
    assert len(result["attempt_memories"]) == 1


def test_stale_comparison_rejected_after_active_version_changes(h):
    h.learn(BAD, target=4)
    result = h.explore({"plan": [plan(step(target=3))], "repair": [candidate(GOOD)]}, goal=3, max_attempts=2)
    stale = h.library.propose("A02", GOOD)
    report = h.maintain(result, {"consolidate": [consolidation(target=3)]})
    assert report["promoted"]
    h.library.rollback("A02", 1)
    with pytest.raises(LibraryError, match="unchanged"):
        h.library.publish_comparison(stale["id"], result["attempt_memories"][-1], "stale revision")


def test_composite_parameterizes_success_and_failure_but_replays_only_success(h):
    first = step("pick", target=1, predicate="joint_near")
    last = step("place", target=3, atom="A04", depends=["pick"])
    fixtures = [{"stage": s, "response": r} for s, r in (
        ("plan", {**plan(first), "plan_complete": False}), ("repair", candidate(GOOD)),
        ("plan", plan(last)), ("repair", candidate(BAD)),
        ("distill", candidate(GOOD)), ("consolidate", failed_consolidation(3)))]
    agent = Agent(ScriptedProvider(fixtures), max_attempts=1)
    config = dict(library=str(h.library.root), agent={}, tasks=[{"task": "toy", "seed": 1, "goal": 3, "horizon": 2}],
        exploration={"max_attempts": 1, "max_replans": 0}, maintenance={"admission_policy": "compare_existing", "max_candidates": 1},
        episode_retry={"max_episodes": 1})
    with patch("evo.cli.create_agent", return_value=agent), patch("evo.cli.adapter_factory", h.factory):
        report = run_config(config, h.path("campaign"))
    assert report["status"] == "tasks_incomplete", report
    assert report["maintenance_status"] == "complete", report
    assert report["maintenance_trigger"] == "episode"
    assert report["promoted_versions"] == 1 and report["draft_count"] == 1
    assert report["retained_failure_count"] == 1
    assert sum(r["stage"] == "consolidate" for r in agent.provider.requests) == 1
    assert h.library.active("A02") and h.library.active("A04") is None
    assert len(h.factory.instances) == 2  # One exploration, one first-use replay.
    second_plan = [r for r in agent.provider.requests if r["stage"] == "plan"][1]["context"]
    assert second_plan["instruction"] == "Move to 3.0" and second_plan["observation"]["joint"] == 1
    assert second_plan["completed_steps"][0]["step"]["id"] == "pick"
    assert second_plan["library"] == []  # Consolidation has not happened yet.
    assert second_plan["working_memory_catalog"][0]["status"] == "observed_success"


def test_one_step_contract_and_incomplete_update_decisions_are_enforced():
    agent = Agent(ScriptedProvider({"plan": [plan(step("a", predicate="joint_near"), step("b"))]}), max_attempts=1)
    with pytest.raises(ResponseError, match="one next atom"):
        agent.plan({"plan_one_step": True})
    agent = Agent(ScriptedProvider({"consolidate": [consolidation(decision="update")]}), max_attempts=1)
    with pytest.raises(ResponseError, match="draft"):
        agent.consolidate({"source_success": False, "active": {"code": GOOD}})


@pytest.mark.parametrize("analysis", [None, {"summary": "", "hypotheses": [], "next_changes": []},
                                      {"summary": "Goal not reached", "hypotheses": "guess", "next_changes": []}])
def test_failed_parameterization_requires_structured_failure_reason(analysis):
    response = failed_consolidation()
    if analysis is None:
        response.pop("failure_analysis")
    else:
        response["failure_analysis"] = analysis
    agent = Agent(ScriptedProvider({"consolidate": [response]}), max_attempts=1)
    with pytest.raises(ResponseError, match="failure_analysis"):
        agent.consolidate({"source_success": False, "active": None})
