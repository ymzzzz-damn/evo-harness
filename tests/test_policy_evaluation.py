"""Policy trials: native resets, task-local memory, immutable pins and coverage."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from evo.agent import Agent, ProviderError, ScriptedProvider
from evo.io import read_json, write_json
from evo.library import Library, LibraryError
from evo.policy_evaluation import FrozenLibrary, freeze_active_library, run_policy_evaluation
from test_loops import BAD, GOOD, FakeAdapter, candidate, plan, step


class NativeFake(FakeAdapter):
    def __init__(self, config, output, scene="same"):
        super().__init__(config, output)
        self.output.mkdir(parents=True)
        self.camera_files = {}
        for name in ("left", "right", "wrist"):
            rgb, depth = self.output / f"{name}.png", self.output / f"{name}.npy"
            rgb.write_bytes((config["task"] + scene + name).encode())
            depth.write_bytes((scene + "depth" + name).encode())
            self.camera_files[name] = {"image_path": str(rgb), "depth_path": str(depth),
                                       "width": 512, "height": 512, "fx": 250, "fy": 250}

    def observe(self):
        return {**super().observe(), "task": self.config["task"], "seed": self.config["seed"],
                "terminal": self.steps >= self.horizon or self.native_success(),
                "robot": {"joint": self.position, "time_s": self.steps / 20},
                "cameras": self.camera_files}


class NativeFactory:
    def __init__(self, changed_after_first=False, init_error=False):
        self.instances, self.changed_after_first, self.init_error = [], changed_after_first, init_error

    def __call__(self, task, output):
        if self.init_error:
            raise RuntimeError("fixture EGL initialization error")
        scene = "changed" if self.changed_after_first and self.instances else "same"
        adapter = NativeFake(task, output, scene)
        self.instances.append(adapter)
        return adapter


def task(name="toy", group="atomic_seen"):
    return {"task": name, "seed": 1, "scene_split": "pretrain", "evaluation_group": group,
            "horizon": 1, "goal": 2, "camera_size": [512, 512]}


def publish(library, atom, code):
    identifier = library.register_case({"atom": atom, "task_config": {"task": "TRAINING_SECRET", "v": len(library.cases(atom))}})
    c = library.propose(atom, code, metadata={"rationale": "TRAINING_SECRET"})
    case_ids = {i["id"] for i in library.cases(atom)} | {identifier}
    library.record_validation(c["id"], {"candidate_sha256": c["sha256"], "success": True,
        "validation_kind": "fresh_environment_replay", "cases": [{"case_id": i, "success": True} for i in sorted(case_ids)]})
    return library.promote(c["id"])


def base_config(tmp_path, tasks=None, publish_good=False):
    library = Library(tmp_path / "training")
    if publish_good:
        publish(library, "A02", GOOD)
    library.add_lesson("A02", "TRAINING_SECRET")
    library.propose("A02", BAD, metadata={"rationale": "TRAINING_SECRET"})
    library.register_case({"atom": "A02", "task_config": {"task": "TRAINING_SECRET_pending"}})
    receipt = freeze_active_library(library.root, tmp_path / "frozen")
    return {"mode": "evaluate_policy", "library": receipt["path"], "frozen_library": receipt["path"],
            "frozen_library_sha256": receipt["state_sha256"], "agent": {}, "tasks": tasks or [task()],
            "exploration": {"max_attempts": 1, "max_replans": 0, "max_plan_segments": 40},
            "policy_evaluation": {"expected_tasks": len(tasks or [task()])}}


def agents(monkeypatch, fixtures):
    providers, instances = [], []
    queue = iter(fixtures)

    def create(config, output):
        fixture = next(queue)
        provider = fixture if hasattr(fixture, "generate") else ScriptedProvider(fixture)
        agent = Agent(provider, log_dir=output, max_attempts=1)
        providers.append(provider)
        instances.append(agent)
        return agent

    monkeypatch.setattr("evo.cli.create_agent", create)
    return providers, instances


def run(monkeypatch, config, output, fixtures, factory=None):
    providers, instances = agents(monkeypatch, fixtures)
    factory = factory or NativeFactory()
    monkeypatch.setattr("evo.cli.adapter_factory", factory)
    return run_policy_evaluation(config, output), providers, instances, factory


class RepairTransportError(ScriptedProvider):
    def __init__(self):
        super().__init__({"plan": [plan()]})

    def generate(self, request):
        if request["stage"] == "repair":
            self.requests.append(request)
            raise ProviderError("fixture bounded connection/timeout retries")
        return super().generate(request)


def test_same_scene_resets_new_agent_history_and_early_stop(tmp_path, monkeypatch):
    config = base_config(tmp_path)
    result, providers, instances, factory = run(monkeypatch, config, tmp_path / "eval", [
        {"plan": [plan()], "repair": [candidate(BAD)]},
        {"plan": [plan()], "repair": [candidate(GOOD)]}])
    trial = result["episodes"][0]
    assert result["status"] == "complete" and result["total_episode_attempts"] == 2
    assert trial["first_attempt"] is False and trial["success_within3"] is True
    assert trial["first_success_attempt"] == 2
    assert len({id(a) for a in instances}) == 2
    assert [a.position for a in factory.instances] == [1, 2]
    assert all(a.closed and a.config == config["tasks"][0] for a in factory.instances)
    second = providers[1].requests[0]["context"]
    assert second["initial_observation"]["joint"] == 0
    history = second["episode_retry"]["previous_episodes"]
    assert len(history) == 1 and history[0]["recent_failures"][0]["code"] == BAD
    assert history[0]["observations_are_stale"] is True
    assert trial["initial_scenes"][1]["matches_first_scene"] is True


def test_three_native_failures_are_known_failure_and_cap_is_total(tmp_path, monkeypatch):
    config = base_config(tmp_path)
    fixture = {"plan": [plan()], "repair": [candidate(BAD)]}
    result, _, _, factory = run(monkeypatch, config, tmp_path / "eval", [fixture] * 3)
    trial = result["episodes"][0]
    assert len(factory.instances) == 3 and trial["physical_attempts"] == 3
    assert trial["success_within3"] is False and trial["invalid_attempts"] == []
    assert all(a["outcome"] == "native_failure" for a in trial["episode_attempts"])


def test_provider_interruptions_consume_all_three_attempts_and_stay_unknown(tmp_path, monkeypatch):
    config = base_config(tmp_path)
    result, _, _, factory = run(monkeypatch, config, tmp_path / "eval", [RepairTransportError() for _ in range(3)])
    trial = result["episodes"][0]
    assert len(factory.instances) == 3 and trial["invalid_attempts"] == [1, 2, 3]
    assert trial["first_attempt"] is None and trial["success_within3"] is None
    assert all(a["initial_scene"]["valid_native_initial"] for a in trial["episode_attempts"])
    assert all(a["stats"]["steps"] == 0 for a in trial["episode_attempts"])
    assert result["evaluation_status"] == "complete_with_unknown_trials"


def test_initialization_errors_are_retained_and_bounded(tmp_path, monkeypatch):
    config = base_config(tmp_path)
    result, providers, _, _ = run(monkeypatch, config, tmp_path / "eval", [{}, {}, {}], NativeFactory(init_error=True))
    trial = result["episodes"][0]
    assert trial["physical_attempts"] == 3 and trial["success_within3"] is None
    assert all(not p.requests for p in providers)
    assert all(not a["initial_scene"]["valid_native_initial"] for a in trial["episode_attempts"])
    assert all(Path(a["summary_path"]).is_file() for a in trial["episode_attempts"])


def test_task_memory_is_local_and_training_private_records_are_absent(tmp_path, monkeypatch):
    config = base_config(tmp_path, [task("a"), task("b", "composite_unseen")])
    fail = {"plan": [plan()], "repair": [candidate(BAD)]}
    result, providers, _, _ = run(monkeypatch, config, tmp_path / "eval", [fail] * 6)
    assert result["total_episode_attempts"] == 6
    first_repair = [r for r in providers[0].requests if r["stage"] == "repair"][0]["context"]
    later_repair = [r for r in providers[1].requests if r["stage"] == "repair"][0]["context"]
    next_task = [r for r in providers[3].requests if r["stage"] == "repair"][0]["context"]
    assert first_repair["lessons"] == [] and len(later_repair["lessons"]) == 1
    assert next_task["lessons"] == [] and next_task["episode_retry"]["previous_episodes"] == []
    assert "TRAINING_SECRET" not in json.dumps([p.requests for p in providers])
    assert result["episodes"][0]["scratch_library"] != result["episodes"][1]["scratch_library"]
    for trial in result["episodes"]:
        state = read_json(Path(trial["scratch_library"]) / "state.json")
        assert state["active"] == {} and len(state["candidates"]) == 3
        assert state["cases"] == state["pending_cases"] == {}


def test_frozen_historical_dependency_pins_execute_without_newer_active_child(tmp_path, monkeypatch):
    library = Library(tmp_path / "training")
    old = publish(library, "A02", GOOD)
    publish(library, "A04", 'def skill(api, params):\n    return api.call("A02", params)\n')
    newer = publish(library, "A02", BAD)
    receipt = freeze_active_library(library.root, tmp_path / "frozen")
    assert receipt["dependency_versions"]["A02"] == {"1": old["sha256"], "2": newer["sha256"]}
    config = {"library": receipt["path"], "frozen_library": receipt["path"],
              "frozen_library_sha256": receipt["state_sha256"], "agent": {}, "tasks": [task()],
              "exploration": {"max_attempts": 1, "max_replans": 0}}
    before = {str(p): p.read_bytes() for p in Path(receipt["path"]).rglob("*") if p.is_file()}
    result, providers, _, _ = run(monkeypatch, config, tmp_path / "eval", [{"plan": [plan(step(atom="A04"))]}])
    assert result["episodes"][0]["first_attempt"] is True
    assert result["library_unchanged"] is True
    assert [r["stage"] for r in providers[0].requests] == ["plan"]
    assert before == {str(p): p.read_bytes() for p in Path(receipt["path"]).rglob("*") if p.is_file()}


def test_freeze_drops_unpublished_records_and_rejects_self_reported_validation(tmp_path):
    config = base_config(tmp_path, publish_good=True)
    frozen_root = Path(config["frozen_library"])
    state = read_json(frozen_root / "state.json")
    assert state["candidates"] == state["lessons"] == state["pending_cases"] == state["cases"] == {}
    assert "TRAINING_SECRET" not in json.dumps(state)
    state["versions"]["A02"]["1"]["validation"]["validation_kind"] = "self_reported"
    (frozen_root / "state.json").chmod(0o644)
    (frozen_root / "state.json").write_text(Library._state_text(state))
    with pytest.raises(LibraryError, match="independent replay"):
        FrozenLibrary(frozen_root, hashlib.sha256((frozen_root / "state.json").read_bytes()).hexdigest())


def test_frozen_hashes_are_checked_before_model_and_publication_is_disabled(tmp_path, monkeypatch):
    config = base_config(tmp_path, publish_good=True)
    frozen_root = Path(config["frozen_library"])
    frozen = FrozenLibrary(frozen_root, config["frozen_library_sha256"])
    scratch = frozen.scratch(tmp_path / "scratch")
    with pytest.raises(LibraryError, match="Publication"):
        scratch.promote("anything")
    digest = next(iter(frozen.sources))
    (frozen_root / "sources" / f"{digest}.py").chmod(0o644)
    (frozen_root / "sources" / f"{digest}.py").write_text(BAD)
    monkeypatch.setattr("evo.cli.create_agent", lambda *a, **k: pytest.fail("must fail before creating model agent"))
    with pytest.raises(LibraryError, match="source hash"):
        run_policy_evaluation(config, tmp_path / "eval")


def test_reset_fingerprint_mismatch_is_rejected_before_new_model_request(tmp_path, monkeypatch):
    config = base_config(tmp_path)
    result, providers, _, factory = run(monkeypatch, config, tmp_path / "eval", [
        {"plan": [plan()], "repair": [candidate(BAD)]}, {}, {}], NativeFactory(changed_after_first=True))
    trial = result["episodes"][0]
    assert len(factory.instances) == 3 and all(a.closed for a in factory.instances)
    assert trial["success_within3"] is None and trial["invalid_attempts"] == [2, 3]
    assert trial["initial_scenes"][1]["matches_first_scene"] is False
    assert not providers[1].requests and not providers[2].requests


def test_native_success_is_environment_verdict_even_if_code_raises_afterward(tmp_path, monkeypatch):
    config = base_config(tmp_path)
    code = ('def skill(api, params):\n    api.move_joints([params["target"]])\n'
            '    raise RuntimeError("after native success")\n    return {}\n')
    result, _, _, factory = run(monkeypatch, config, tmp_path / "eval", [{"plan": [plan()], "repair": [candidate(code)]}])
    assert len(factory.instances) == 1
    attempt = result["episodes"][0]["episode_attempts"][0]
    assert attempt["program_status"] == "step_failed" and attempt["success"] is True
    assert attempt["native_success"] is True


def test_base_drift_after_episode_is_recorded_and_stops_further_episodes(tmp_path, monkeypatch):
    config = base_config(tmp_path)

    class MutatingProvider(ScriptedProvider):
        def generate(self, request):
            response = super().generate(request)
            if request["stage"] == "plan":
                path = Path(config["frozen_library"]) / "state.json"
                path.chmod(0o644)
                path.write_bytes(path.read_bytes() + b"\n")
            return response

    provider = MutatingProvider({"plan": [plan()], "repair": [candidate(GOOD)]})
    result, _, _, factory = run(monkeypatch, config, tmp_path / "eval", [provider])
    assert len(factory.instances) == 1 and result["total_episode_attempts"] == 1
    assert result["status"] == "error" and result["protocol_integrity_error"] is True
    assert result["library_unchanged"] is False


def summary_module():
    path = Path(__file__).parents[1] / "scripts/summarize_policy_evaluation.py"
    spec = importlib.util.spec_from_file_location("summary_policy_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_preserves_task_denominator_and_unknown_coverage(tmp_path, monkeypatch):
    workers = []
    for index, name in enumerate(("good", "provider_interrupted")):
        root = tmp_path / name
        root.mkdir()
        config = base_config(root)
        config["tasks"] = [task(name, "atomic_seen" if index == 0 else "composite_unseen")]
        fixtures = ([{"plan": [plan()], "repair": [candidate(GOOD)]}] if index == 0
                    else [RepairTransportError() for _ in range(3)])
        run(monkeypatch, config, root / "campaign", fixtures)
        write_json(root / "task_receipt.json", {"status": "succeeded"})
        workers.append({"index": index, "run": str(root), "task": config["tasks"][0]})
    write_json(tmp_path / "config.json", {"policy_evaluation": {"expected_tasks": 2}})
    plan_path = tmp_path / "parallel_plan.json"
    write_json(plan_path, {"run": str(tmp_path), "workers": workers})
    module = summary_module()
    result = module.summarize_evaluation(plan_path)
    assert result["status"] == "complete", result["issues"]
    assert result["first_episode"]["denominator_tasks"] == 2
    assert result["first_episode"]["observed_success_rate"] == .5
    assert result["first_episode"]["exact_success_rate"] is None
    assert result["within_three"]["observed_success_rate"] == .5
    assert result["within_three"]["unknown_or_pending_tasks"] == 1
    assert result["counts"]["invalid_attempts"] == 3 and result["counts"]["physical_attempts"] == 4
    assert result["counts"]["provider_interruptions"] == 3
    module.write_summary(result, tmp_path / "evaluation_summary.json")
    assert read_json(tmp_path / "evaluation_summary.json")["within_three"] == result["within_three"]


def test_summary_50task_pending_groups_and_fixed_denominators(tmp_path):
    workers = []
    for group, count in (("atomic_seen", 18), ("composite_seen", 16), ("composite_unseen", 16)):
        for index in range(count):
            name = f"{group}_{index}"
            workers.append({"index": len(workers), "run": str(tmp_path / name), "task": task(name, group)})
    write_json(tmp_path / "config.json", {})
    plan_path = tmp_path / "parallel_plan.json"
    write_json(plan_path, {"run": str(tmp_path), "workers": workers})
    result = summary_module().summarize_evaluation(plan_path)
    assert result["status"] == "running" and result["issues"] == []
    assert result["first_episode"]["denominator_tasks"] == 50
    assert result["within_three"]["unknown_or_pending_tasks"] == 50
    assert [result["groups"][g]["within_three"]["denominator_tasks"] for g in
            ("atomic_seen", "composite_seen", "composite_unseen")] == [18, 16, 16]
