"""Real loop/registry tests on the deterministic toy plant, not RoboCasa claims."""
import copy
from pathlib import Path
from unittest.mock import patch

import pytest

from evo.agent import Agent, ScriptedProvider
from evo.io import read_json
from evo.library import Library, LibraryError
from evo.loops import ExplorationLoop, MaintenanceLoop
from evo.policy_evaluation import freeze_active_library, FrozenLibrary
from evo.replay_evidence import ReplayEvidence, contract
from evo.runtime_api import RuntimeAPI, API_DESCRIPTION
from test_loops import GOOD, BAD, Factory, candidate, plan, step


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setenv("EVO_REPLAY_CACHE_SESSION", "test-session")
    library, factory = Library(tmp_path / "library"), Factory()
    provider = Agent(ScriptedProvider({"distill": [candidate(GOOD)]}), max_attempts=1)
    common = dict(library=library, agent=provider, adapter_factory=factory, runtime_api=RuntimeAPI,
                  api_description=API_DESCRIPTION, supported_predicates=["native_success", "joint_near"])
    loop = MaintenanceLoop(**common, reuse_replay_evidence=True, max_candidates=1)
    sequence = 0

    def stage(goal=2, atom="A02"):
        identifier = library.stage_case(dict(atom=atom, task_config={"task": "toy", "goal": goal, "seed": goal},
            prefix=[], step=step(target=goal, atom=atom), source_run="test"))
        return library.get_case(identifier)

    def validate(program, cases, selected_loop=None):
        nonlocal sequence
        sequence += 1
        selected_loop = selected_loop or loop
        results = selected_loop.replay_many(cases, program, tmp_path / f"replay_{sequence}")
        report = dict(candidate_sha256=program["sha256"], validation_kind="fresh_environment_replay",
                      success=all(c["success"] for c in results), cases=results)
        selected_loop._validation_evidence(report)
        return report

    def publish(atom="A02", code=GOOD, dependencies=None):
        case = stage(atom=atom)
        program = library.propose(atom, code, dependencies=dependencies or [], metadata={"new_case_id": case["id"]})
        report = validate(program, library.cases(atom) or [case])
        library.record_validation(program["id"], report)
        return library.promote(program["id"]), case

    return tmp_path, library, factory, common, loop, stage, validate, publish


def test_six_reuse_admissions_need_six_fresh_replays_instead_of_twenty_one(harness):
    root, library, factory, common, loop, stage, validate, publish = harness
    calls = 0
    for index in range(6):
        goal = index + 2
        provider = Agent(ScriptedProvider({"plan": [plan(step(target=goal))],
                                          "repair": [candidate(GOOD)]}), max_attempts=1)
        explored = ExplorationLoop(**{**common, "agent": provider}).run(
            {"task": "toy", "goal": goal, "seed": index}, root / f"explore_{index}")
        before = len(factory.instances)
        result = loop.run(explored["pending"], root / f"maintain_{index}")
        calls += len(factory.instances) - before
        assert result["status"] == "complete", result
        assert result["replay_counts"] == {"total": index + 1, "fresh": 1, "cached": index}
    assert calls == 6
    assert loop.replay_counts == {"total": 21, "fresh": 6, "cached": 15}
    assert len(library.cases("A02")) == 6
    assert library.active("A02")["version"] == 1


def test_pending_always_fresh_and_cannot_be_admitted_using_cached_proof(harness):
    root, library, factory, common, loop, stage, validate, publish = harness
    case = stage()
    program = library.propose("A02", GOOD)
    for _ in range(2):
        report = validate(program, [case])
        assert report["replay_counts"] == {"total": 1, "fresh": 1, "cached": 0}
    cache = ReplayEvidence(library.root, loop.last_replay_context)
    cached = cache.lookup(contract(case, program, library, cache.context_sha256))
    report["cases"] = [cached]
    with pytest.raises(LibraryError, match="Pending cases"):
        library.record_validation(program["id"], report)
    assert not library.cases("A02")
    assert len(factory.instances) == 2


def test_changed_candidate_replays_every_case_and_failure_prevents_publish(harness):
    root, library, factory, common, loop, stage, validate, publish = harness
    active, old = publish()
    new = stage(3)
    bad = library.propose("A02", BAD, metadata={"new_case_id": new["id"]})
    report = validate(bad, [old, new])
    assert report["replay_counts"] == {"total": 2, "fresh": 2, "cached": 0}
    library.record_validation(bad["id"], report)
    with pytest.raises(LibraryError):
        library.promote(bad["id"])
    assert library.active("A02")["version"] == active["version"]
    assert len(library.cases("A02")) == 1
    # Failure certificates cannot turn a second attempt into a cache hit.
    assert validate(bad, [old, new])["replay_counts"]["fresh"] == 2


@pytest.mark.parametrize("field", ["params", "predicate", "horizon", "seed", "entry_trace", "native", "suffix"])
def test_case_contract_change_invalidates_old_proof(harness, field):
    root, library, factory, common, loop, stage, validate, publish = harness
    active, old = publish()
    changed = copy.deepcopy(old)
    if field == "params":
        changed["step"]["params"]["extra"] = 1
    elif field == "predicate":
        changed["step"]["postcondition"]["predicate"] = "joint_near"
    elif field in ("horizon", "seed"):
        changed["task_config"][field] = 18
    elif field == "entry_trace":
        changed["entry_trace"] = []
    elif field == "native":
        changed["require_native_success"] = True
    else:
        changed["suffix"] = []
    report = validate(active, [changed])
    assert report["replay_counts"]["fresh"] == 1


@pytest.mark.parametrize("change", ["session", "timeout", "predicates", "gpu", "framework"])
def test_execution_context_change_invalidates_old_proof(harness, change, monkeypatch):
    root, library, factory, common, loop, stage, validate, publish = harness
    active, old = publish()
    if change == "session":
        loop.replay_session = "new-session"
    elif change == "timeout":
        loop.timeout += 1
    elif change == "predicates":
        loop.predicates = ["native_success"]
    elif change == "gpu":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-new")
    else:
        from evo import replay_evidence
        original = replay_evidence.context
        monkeypatch.setattr(replay_evidence, "context", lambda *a: {**original(*a), "framework": "new"})
    assert validate(active, [old])["replay_counts"]["fresh"] == 1


def test_same_session_worker_reuses_but_disabled_mode_replays(harness):
    root, library, factory, common, loop, stage, validate, publish = harness
    active, old = publish()
    other = MaintenanceLoop(**common, reuse_replay_evidence=True)
    assert validate(active, [old], other)["replay_counts"]["cached"] == 1
    disabled = MaintenanceLoop(**common)
    before = len(factory.instances)
    report = validate(active, [old], disabled)
    assert len(factory.instances) == before + 1
    assert "replay_policy" not in report
    assert "replay_evidence" not in report["cases"][0]


@pytest.mark.parametrize("damage", ["missing", "corrupt", "wrong_shape", "changed_result", "changed_provenance"])
def test_bad_certificate_falls_back_to_fresh_and_cannot_validate(harness, damage):
    root, library, factory, common, loop, stage, validate, publish = harness
    active, old = publish()
    report = validate(active, [old])
    reference = report["cases"][0]["replay_evidence"]
    path = library.root / "replay_evidence/objects" / (reference["certificate_sha256"] + ".json")
    if damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.write_text("broken")
    elif damage == "wrong_shape":
        path.write_text("[]")
    elif damage == "changed_provenance":
        reference["original_report"] = "another_report"
    else:
        report["cases"][0]["native_success"] = False
    program = library.propose("A02", GOOD)
    with pytest.raises(LibraryError, match="cached replay"):
        library.record_validation(program["id"], report)
    if damage not in ("changed_result", "changed_provenance"):
        assert validate(active, [old])["replay_counts"]["fresh"] == 1


def test_dependency_pin_change_invalidates_identical_parent_source(harness):
    root, library, factory, common, loop, stage, validate, publish = harness
    child, _ = publish("A04")
    source = 'def skill(api, params):\n    return api.call("A04", params)\n'
    parent, old = publish("A02", source, ["A04"])
    assert validate(parent, [old])["replay_counts"]["cached"] == 1
    new_child, _ = publish("A04")
    assert new_child["sha256"] == child["sha256"] and new_child["version"] != child["version"]
    changed = library.propose("A02", source, dependencies=["A04"])
    assert validate(changed, [old])["replay_counts"]["fresh"] == 1


def test_mixed_coverage_is_ordered_and_new_failure_blocks_admission(harness):
    root, library, factory, common, loop, stage, validate, publish = harness
    active, old = publish()
    new = stage(3)
    new["step"]["params"]["target"] = 1
    new = library.get_case(library.stage_case(new))
    report = validate(active, [new, old])
    assert [c["case_id"] for c in report["cases"]] == [new["id"], old["id"]]
    assert report["replay_counts"] == {"total": 2, "fresh": 1, "cached": 1}
    with pytest.raises(LibraryError, match="complete regression coverage"):
        library.accept_case(new["id"], "A02", active["version"], report)
    assert len(library.cases("A02")) == 1


def test_parallel_dispatch_receives_only_misses_in_original_order(harness):
    root, library, factory, common, loop, stage, validate, publish = harness
    active, old = publish()
    loop.replay_parallelism = 2
    def dispatch(cases, candidate, output, **kwargs):
        return [loop.replay(c, candidate, Path(output) / c["id"]) for c in cases]
    with patch("evo.replay_dispatch.dispatch_replays", side_effect=dispatch) as dispatched:
        validate(active, [old])  # Warm the distinct parallel execution context.
        new = stage(3)
        report = validate(active, [old, new])
        assert dispatched.call_args.args[0] == [new]
        assert report["replay_counts"] == {"total": 2, "fresh": 1, "cached": 1}


def test_cache_write_failure_does_not_block_a_successful_fresh_validation(harness):
    root, library, factory, common, loop, stage, validate, publish = harness
    with patch.object(ReplayEvidence, "remember", side_effect=OSError("disk unavailable")):
        active, old = publish()
    assert active["validation"]["success"] is True
    assert "disk unavailable" in active["validation"]["cases"][0]["replay_evidence"]["cache_unavailable"]


def test_mixed_evidence_promotes_and_exports_without_training_cache(harness):
    root, library, factory, common, loop, stage, validate, publish = harness
    active, old = publish()
    new = stage(3)
    program = library.propose("A02", GOOD, metadata={"new_case_id": new["id"]})
    report = validate(program, [old, new])
    library.record_validation(program["id"], report)
    library.promote(program["id"])
    receipt = freeze_active_library(library.root, root / "frozen")
    frozen = FrozenLibrary(root / "frozen", receipt["state_sha256"])
    proof = frozen.proofs[("A02", "2")]
    assert proof["replay_counts"] == {"total": 2, "fresh": 1, "cached": 1}
    assert proof["replay_origins"][old["id"]] == "cached_success"
    assert not (root / "frozen/replay_evidence").exists()
    assert read_json(root / "frozen/state.json")["cases"] == {}
