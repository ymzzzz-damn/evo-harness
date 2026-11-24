"""The control/verdict boundary is tested without installing a simulator."""
from contextlib import nullcontext
import json
import sys
from types import SimpleNamespace, ModuleType

import numpy as np
import pytest

from evo.robocasa_adapter import EpisodeFinished, RoboCasaAdapter
from evo.runtime_api import RuntimeAPI


class Backend:
    def __init__(self, success_at=None):
        self.count = 0
        self.success_at = success_at
        self.commands = []
        self.control_interval_s = .05

    def get_step_budget(self):
        return {"used": self.count}

    def task_completed(self):
        return self.success_at is not None and self.count >= self.success_at

    def step(self, commands):
        self.commands.append(commands)

    def tick(self):
        self.count += 1


def adapter(tmp_path, success_at=None, horizon=10):
    item = RoboCasaAdapter.__new__(RoboCasaAdapter)
    item._backend = Backend(success_at)
    item._closed = False
    item.horizon = horizon
    item._events = tmp_path / "events.jsonl"
    item._joint_ids = {"right": list(range(7)), "torso": [0]}
    item._limits = {"right": np.array([[-2, 2]]*7), "torso": np.array([[0, .4]])}
    item._robot_observation = lambda: {"gripper": {"aperture_m": .04}}
    item._joint_positions = lambda part: np.array([.1]*7 if part == "right" else [.1])
    return item


def test_step_uses_native_ticks_and_stops_at_horizon(tmp_path):
    item = adapter(tmp_path, horizon=3)
    result = item.step({"arm": [.1]*7}, repeat=3)
    assert result["steps"] == 3
    assert result["terminal"]
    assert item._backend.count == 3
    assert len(item._backend.commands) == 1
    assert "native_success" not in result
    with pytest.raises(EpisodeFinished):
        item.step({})


def test_native_success_stops_actions_without_rechecking_predicate(tmp_path):
    item = adapter(tmp_path, success_at=2)
    result = item.step({}, repeat=6)
    assert result["steps"] == 2
    assert result["terminal"]
    assert item.native_success()
    assert item.evaluate({"predicate": "native_success"})["success"]
    assert item._backend.count == 2
    with pytest.raises(EpisodeFinished):
        item.step({})


@pytest.mark.parametrize("action", [
    {"arm": [10]*7}, {"arm": [0]*6}, {"torso": [float("nan")]},
    {"gripper": 2}, {"base": [1, 0, 0]}, {"qpos": [0]},
])
def test_invalid_commands_have_no_effect(tmp_path, action):
    item = adapter(tmp_path)
    with pytest.raises(ValueError):
        item.step(action)
    assert item.steps == 0
    assert item._backend.commands == []


def test_host_evaluation_ignores_generated_success_claims(tmp_path):
    item = adapter(tmp_path)
    assert not item.evaluate({"predicate": "native_success", "success": True})["success"]
    assert not item.evaluate({"predicate": "generated_result", "value": True})["known"]
    result = item.evaluate({"predicate": "joint_near", "part": "arm", "target": [.1]*7})
    assert result["known"] and result["success"]
    aperture = item.evaluate({"predicate": "gripper_aperture", "min": .03, "max": .05})
    assert aperture["success"] and "not an object-grasp" in aperture["reason"]


def test_runtime_has_no_verdict_or_native_environment_api(tmp_path):
    runtime = RuntimeAPI(adapter(tmp_path), lambda skill, values: (skill, values))
    assert runtime.call("A05", {"duration": 1}) == ("A05", {"duration": 1})
    for forbidden in ("native_success", "evaluate", "sim", "env", "_backend", "adapter"):
        assert not hasattr(runtime, forbidden)


def test_unbound_dependency_is_explicit(tmp_path):
    with pytest.raises(ValueError, match="No committed"):
        RuntimeAPI(adapter(tmp_path)).call("A02", {})


def test_deproject_uses_depth_and_camera_coordinates(tmp_path):
    item = adapter(tmp_path)
    file = tmp_path / "depth.npy"
    np.save(file, np.full((3, 3), 2.))
    transform = np.eye(4)
    transform[:3, 3] = [1, 2, 3]
    item.observe = lambda: {"cameras": {"robot0_agentview_right": {
        "depth_path": str(file), "width": 3, "height": 3, "fx": 2, "fy": 2,
        "cx": 1, "cy": 1, "T_base_camera": transform.tolist()}}}
    assert item.deproject("camera_right", 1, 1) == [1, 2, 5]
    with pytest.raises(ValueError, match="outside"):
        item.deproject("camera_right", -1, 0)
    np.save(file, np.zeros((3, 3)))
    with pytest.raises(ValueError, match="valid measured depth"):
        item.deproject("camera_right", 1, 1)


def test_serial_chain_is_pure_math():
    pytest.importorskip("scipy")
    from evo.robocasa_adapter import SerialChain
    chain = SerialChain([{"origin": [0, 0, 1, 0, 0, 0, 1], "joint": 0,
                          "axis": [0, 0, 1], "anchor": [0, 0, 0], "reference": 0}], [2])
    assert np.allclose(chain.forward([.2])[:3, 3], [0, 0, 1.2])


@pytest.mark.parametrize("success_at,horizon,expected", [(2, 100, 2), (None, 3, 3)])
def test_long_joint_action_stops_cleanly_on_native_terminal(tmp_path, success_at, horizon, expected):
    item = adapter(tmp_path, success_at, horizon)
    # A physical terminal condition must not be converted into a later execution
    # exception by attempting the remainder of the trajectory or settle wait.
    result = item.move_joints([.2]*7, duration=4., settle=2.)
    assert item.steps == expected
    assert result["terminal"]
    assert not result["ok"]  # measured joints never reached this target
    with pytest.raises(EpisodeFinished):
        item.move_joints([.2]*7)


@pytest.mark.parametrize("tolerance", [0, -1, float("nan"), float("inf")])
def test_joint_tolerance_is_validated_before_motion(tmp_path, tolerance):
    item = adapter(tmp_path)
    with pytest.raises(ValueError):
        item.move_joints([.2]*7, tolerance=tolerance)
    assert item.steps == 0


def tcp_adapter(tmp_path, success_at=None, horizon=100):
    pytest.importorskip("scipy")
    item = adapter(tmp_path, success_at, horizon)
    def measured():
        transform = np.eye(4)
        if item.steps:
            transform[0, 3] = .0055
        return transform
    item._tcp_transform = measured
    item._ik = lambda position, rotation, seed: np.asarray(seed)
    return item


def test_tcp_arrival_is_truthful_at_default_four_mm_tolerance(tmp_path):
    item = tcp_adapter(tmp_path)
    result = item.move_tcp([.01, 0, 0])
    assert result["error_m"] == pytest.approx(.0045)
    assert not result["ok"]
    assert not result["terminal"]
    assert not item.evaluate({"predicate": "tcp_near", "position": [.01, 0, 0], "tolerance": .004})["success"]
    assert item.evaluate({"predicate": "tcp_near", "position": [.01, 0, 0], "tolerance": .005})["success"]


@pytest.mark.parametrize("success_at,horizon,expected", [(2, 100, 2), (None, 3, 3)])
def test_long_tcp_action_stops_cleanly_on_native_terminal(tmp_path, success_at, horizon, expected):
    item = tcp_adapter(tmp_path, success_at, horizon)
    result = item.move_tcp([.01, 0, 0], duration=4., settle=2.)
    assert item.steps == expected
    assert result["terminal"]
    assert not result["ok"]
    with pytest.raises(EpisodeFinished):
        item.move_tcp([.01, 0, 0])


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_tcp_duration_is_validated_before_motion(tmp_path, duration):
    item = tcp_adapter(tmp_path)
    with pytest.raises(ValueError):
        item.move_tcp([.01, 0, 0], duration=duration)
    assert item.steps == 0


@pytest.mark.parametrize("failure_stage", ["reset", "geometry"])
def test_constructor_closes_native_backend_when_initialization_fails(tmp_path, monkeypatch, failure_stage):
    import evo.robocasa_adapter as module
    class InitializingBackend(Backend):
        horizon = 1000
        closed = False
        def __init__(self, *args, **kwargs):
            super().__init__()
        def reset(self, seed):
            if failure_stage == "reset":
                raise RuntimeError("reset failed")
        def goal_text(self):
            return "test instruction"
        def close(self):
            self.closed = True
    native = InitializingBackend()
    backend_module = ModuleType("roboenv_runtime.adapters.robocasa.backend")
    backend_module.RoboCasaEnv = lambda *args, **kwargs: native
    monkeypatch.setitem(sys.modules, backend_module.__name__, backend_module)
    monkeypatch.setattr(module, "_renderer_compatibility", nullcontext)
    def fail_geometry(self):
        raise RuntimeError("geometry failed")
    monkeypatch.setattr(module.RoboCasaAdapter, "_build_robot_geometry", fail_geometry)
    with pytest.raises(RuntimeError, match="failed"):
        module.RoboCasaAdapter("TestTask", output_dir=tmp_path)
    assert native.closed


@pytest.fixture
def constructor_backend(monkeypatch):
    """Capture the real adapter constructor boundary without building a scene."""
    import evo.robocasa_adapter as module
    calls = []

    class InitializingBackend(Backend):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.horizon = 1000
            self.env = SimpleNamespace(horizon=1000)
            self.closed = False
            calls.append((args, kwargs, self))

        def reset(self, seed):
            self.budgets_at_reset = (self.horizon, self.env.horizon,
                                     self.configured_max_steps, self._max_steps)

        def goal_text(self):
            return "test instruction"

        def close(self):
            self.closed = True

    backend_module = ModuleType("roboenv_runtime.adapters.robocasa.backend")
    backend_module.RoboCasaEnv = InitializingBackend
    monkeypatch.setitem(sys.modules, backend_module.__name__, backend_module)
    monkeypatch.setattr(module, "_renderer_compatibility", nullcontext)
    monkeypatch.setattr(module.RoboCasaAdapter, "_build_robot_geometry",
                        lambda self: setattr(self, "_fk_validation", {"passed": True}))
    return calls


def test_constructor_applies_official_pretrain_scene_and_task_budget(tmp_path, constructor_backend):
    item = RoboCasaAdapter("TestTask", seed=3, output_dir=tmp_path,
                           scene_split="pretrain", horizon=7200,
                           camera_size=(512, 512))
    args, kwargs, native = constructor_backend[0]
    assert args == ("TestTask",)
    assert kwargs["layout_ids"] == kwargs["style_ids"] == -2
    assert kwargs["obj_instance_split"] == "pretrain"
    assert kwargs["max_steps"] == 7200
    assert kwargs["camera_size"] == (512, 512)
    assert kwargs["stop_on_success"] is True
    assert native.budgets_at_reset == (7200, 7200, 7200, 7200)
    assert item.horizon == native.horizon == native.env.horizon == 7200
    initialize = json.loads((tmp_path / "control_events.jsonl").read_text())
    assert initialize["scene_split"] == "pretrain"
    assert initialize["horizon"] == 7200


def test_constructor_accepts_matching_pretrain_group_ids(tmp_path, constructor_backend):
    RoboCasaAdapter("TestTask", output_dir=tmp_path, scene_split="pretrain",
                    layout_ids=-2, style_ids=-2)
    _, kwargs, _ = constructor_backend[0]
    assert kwargs["layout_ids"] == kwargs["style_ids"] == -2
    assert "max_steps" not in kwargs


def test_constructor_without_protocol_options_preserves_legacy_inputs(tmp_path, constructor_backend):
    item = RoboCasaAdapter("TestTask", output_dir=tmp_path,
                           layout_ids=[1], style_ids=[2])
    _, kwargs, native = constructor_backend[0]
    assert kwargs["layout_ids"] == [1]
    assert kwargs["style_ids"] == [2]
    assert "obj_instance_split" not in kwargs and "max_steps" not in kwargs
    assert item.horizon == 1000
    assert native.budgets_at_reset == (1000, 1000, 1000, 1000)


@pytest.mark.parametrize("options", [
    {"scene_split": "target"}, {"scene_split": "all"}, {"scene_split": False},
    {"horizon": 0}, {"horizon": -1}, {"horizon": True},
    {"horizon": 300.0}, {"horizon": "300"}, {"horizon": float("nan")},
    {"scene_split": "pretrain", "layout_ids": 1},
    {"scene_split": "pretrain", "style_ids": -3},
    {"scene_split": "pretrain", "layout_ids": [11, 12]},
])
def test_constructor_rejects_invalid_or_conflicting_protocol_before_build(
        tmp_path, constructor_backend, options):
    with pytest.raises(ValueError):
        RoboCasaAdapter("TestTask", output_dir=tmp_path, **options)
    assert constructor_backend == []


@pytest.fixture
def semantic_adapter(tmp_path, monkeypatch):
    item = adapter(tmp_path)
    utilities = SimpleNamespace()
    utils_module = ModuleType("robocasa.utils")
    utils_module.object_utils = utilities
    monkeypatch.setitem(sys.modules, "robocasa", ModuleType("robocasa"))
    monkeypatch.setitem(sys.modules, "robocasa.utils", utils_module)
    item._backend.env = SimpleNamespace(objects={}, fixtures={}, fixture_refs={})
    return item, utilities


def test_host_grasp_uses_opposing_finger_native_test(semantic_adapter):
    item, _ = semantic_adapter
    env = item._backend.env
    gripper = object()
    env.objects["cup"] = SimpleNamespace(contact_geoms=["cup_collision"])
    env.robots = [SimpleNamespace(gripper={"right": gripper})]
    calls = []
    def grasp(actual_gripper, geoms):
        calls.append((actual_gripper, geoms))
        return True
    env._check_grasp = grasp
    assert item.evaluate({"predicate": "object_grasped", "object": "cup"})["success"]
    assert calls == [(gripper, ["cup_collision"])]
    assert not item.evaluate({"predicate": "object_grasped", "object": "unknown"})["known"]


def test_native_containment_signatures_and_unknown_object(semantic_adapter):
    item, utilities = semantic_adapter
    env = item._backend.env
    fixture = SimpleNamespace(name="cabinet")
    env.fixtures["cabinet"] = fixture
    calls = []
    def in_receptacle(actual_env, obj, receptacle):
        calls.append((actual_env, obj, receptacle))
        return True
    def in_fixture(actual_env, obj, actual_fixture):
        calls.append((actual_env, obj, actual_fixture))
        return True
    utilities.check_obj_in_receptacle = in_receptacle
    utilities.obj_inside_of = in_fixture
    assert item.evaluate({"predicate": "object_in_receptacle", "object": "cup", "receptacle": "bowl"})["success"]
    assert item.evaluate({"predicate": "object_in_fixture", "object": "cup", "fixture": "cabinet"})["success"]
    assert calls == [(env, "cup", "bowl"), (env, "cup", fixture)]
    assert not item.evaluate({"predicate": "object_in_fixture", "object": "cup", "fixture": "missing"})["known"]


def test_fixture_state_supports_native_env_and_slot_selector(semantic_adapter):
    item, _ = semantic_adapter
    env = item._backend.env
    calls = []
    class Toaster:
        name = "toaster"
        def get_state(self, env, slot_pair=None):
            calls.append((env, slot_pair))
            return {"turned_on": np.bool_(True)} if slot_pair is not None else {0: {"turned_on": True}}
    env.fixtures["toaster"] = Toaster()
    verdict = item.evaluate({"predicate": "fixture_state", "fixture": "toaster",
                            "field": "turned_on", "value": True, "slot_pair": 0})
    assert verdict["known"] and verdict["success"]
    assert calls == [(env, 0)]
    assert not item.evaluate({"predicate": "fixture_state", "fixture": "toaster",
                            "field": "turned_on", "value": True})["known"]


def test_fixture_state_no_env_and_invalid_field_are_explicit(semantic_adapter):
    item, _ = semantic_adapter
    class Microwave:
        name = "microwave"
        def get_state(self):
            return {"turned_on": False, "timer": 0.0}
    item._backend.env.fixtures["microwave"] = Microwave()
    base = {"predicate": "fixture_state", "fixture": "microwave", "field": "turned_on", "value": False}
    assert item.evaluate(base)["success"]
    for extra in ({"field": "timer"}, {"field": "absent"}, {"slot_pair": 0}, {"value": 0}):
        assert not item.evaluate({**base, **extra})["known"]


def test_fixture_joint_uses_native_normalization_and_validates_ownership(semantic_adapter):
    item, _ = semantic_adapter
    env = item._backend.env
    calls = []
    class Cabinet:
        name = "cabinet"
        joints = ["cabinet_door"]
        def get_joint_state(self, actual_env, names):
            calls.append((actual_env, names))
            return {"cabinet_door": .95}
    env.fixtures["cabinet"] = Cabinet()
    base = {"predicate": "fixture_joint", "fixture": "cabinet", "joint": "cabinet_door", "min": .9, "max": 1.}
    assert item.evaluate(base)["success"]
    assert calls == [(env, ["cabinet_door"])]
    for extra in ({"joint": "unrelated_joint"}, {"min": -1.}, {"max": float("inf")}):
        assert not item.evaluate({**base, **extra})["known"]
    assert len(calls) == 1


def test_native_fixture_aliases_unwrap_region_tuples_without_exposing_geometry(semantic_adapter):
    item, _ = semantic_adapter
    counter = SimpleNamespace(name="counter_1", joints=[], pos=[5., 8., 1.])
    cabinet = SimpleNamespace(name="cabinet_1", joints=["door"], pos=[6., 8., 1.])
    env = item._backend.env
    env.fixtures = {"counter_1": counter, "cabinet_1": cabinet}
    env.fixture_refs = {"counter": (counter, True), "cab": cabinet,
                        "invalid": (SimpleNamespace(name="other"), False)}
    catalog = item._semantic_catalog()
    assert catalog["fixture_aliases"] == {"counter": "counter_1", "cab": "cabinet_1"}
    assert all(set(fixture) == {"id", "type", "joints"} for fixture in catalog["fixtures"])
    assert item._fixture("counter") is counter
    assert item._fixture("cab") is cabinet
    with pytest.raises(KeyError):
        item._fixture("invalid")


def test_base_odometry_uses_episode_origin_and_not_world_coordinates(tmp_path):
    from scipy.spatial.transform import Rotation
    item = adapter(tmp_path)
    origin = np.eye(4)
    origin[:3, :3] = Rotation.from_euler("z", np.pi/2).as_matrix()
    origin[:3, 3] = [5., -2., .1]
    current = origin.copy()
    current[:3, 3] += [0., .3, 0.]
    current[:3, :3] = origin[:3, :3] @ Rotation.from_euler("z", .2).as_matrix()
    item._initial_base_transform = origin
    item._world_base_transform = lambda: current
    observed = item._base_observation()
    assert np.allclose(observed["position"], [.3, 0., 0.])
    assert observed["yaw"] == pytest.approx(.2)
    assert observed["frame"] == "episode_initial_base"
    assert observed["control_velocity_frame"] == "native_backend"
    assert observed["control_units"] == "normalized_native_velocity"
    verdict = item.evaluate({"predicate": "base_near", "args": {"position": [.3, 0], "yaw": .2}})
    assert verdict["known"] and verdict["success"]
    assert not item.evaluate({"predicate": "base_near", "position": [5., -1.7]})["success"]


def test_base_arrival_heading_wrap_and_optional_orientation(tmp_path):
    item = adapter(tmp_path)
    item._base_observation = lambda: {"position": [.2, -.1, 0], "yaw": np.pi-.01}
    condition = {"predicate": "base_near", "position": [.2, -.1], "yaw": -np.pi+.01, "yaw_tolerance": .03}
    assert item.evaluate(condition)["success"]
    assert not item.evaluate({**condition, "yaw_tolerance": .01})["success"]
    assert item.evaluate({"predicate": "base_near", "position": [.2, -.1]})["success"]
    for invalid in ({"position": [float("nan"), 0]}, {"position_tolerance": -1},
                    {"yaw_tolerance": float("inf")}, {"yaw": float("nan")}, {"frame": "world"}):
        with pytest.raises(ValueError):
            item.evaluate({**condition, **invalid})


@pytest.fixture
def support_adapter(semantic_adapter, monkeypatch):
    item, _ = semantic_adapter
    env = item._backend.env
    obj = SimpleNamespace(root_body="object_body", contact_geoms=["object_collision"])
    fixture = SimpleNamespace(name="counter", contact_geoms=["counter_collision"])
    env.objects["obj"], env.fixtures["counter"] = obj, fixture
    env.robots = [SimpleNamespace(gripper={"right": object()})]
    state = {"gripper_contact": False, "linear": np.zeros(3), "angular": np.zeros(3), "force": .981}
    env.check_contact = lambda gripper, object_: state["gripper_contact"]
    contact = SimpleNamespace(geom1=2, geom2=1, frame=np.array([0., 0., 1., 1., 0., 0., 0., 1., 0.]))
    data = SimpleNamespace(ncon=1, contact=[contact])
    model = SimpleNamespace(opt=SimpleNamespace(gravity=np.array([0., 0., -9.81])), body_subtreemass=np.array([.1]))
    names = {"object_collision": 1, "counter_collision": 2, "other_collision": 3}
    item._backend.sim = SimpleNamespace(
        model=SimpleNamespace(_model=model, body_name2id=lambda name: 0, geom_name2id=names.__getitem__),
        data=SimpleNamespace(_data=data, get_body_xvelp=lambda name: state["linear"],
                             get_body_xvelr=lambda name: state["angular"]))
    module = ModuleType("mujoco")
    def contact_force(actual_model, actual_data, index, result):
        assert actual_model is model and actual_data is data
        result[0] = state["force"]
    module.mj_contactForce = contact_force
    monkeypatch.setitem(sys.modules, "mujoco", module)
    return item, state, contact, data


def test_released_object_support_requires_loaded_upward_contact_in_either_geom_order(support_adapter):
    item, state, contact, _ = support_adapter
    condition = {"predicate": "object_on_fixture", "object": "obj", "fixture": "counter"}
    result = item.evaluate(condition)
    assert result["known"] and result["success"]
    assert result["evidence"] == {"step": 0}  # No object pose, force or velocity reaches the policy.
    contact.geom1, contact.geom2 = contact.geom2, contact.geom1
    contact.frame[:3] *= -1
    assert item.evaluate(condition)["success"]
    state["force"] = .1  # A lightly touching fixture does not carry the object's weight.
    assert not item.evaluate(condition)["success"]


@pytest.mark.parametrize("failure", ["side", "ceiling", "zero_force", "hovering", "wrong_fixture", "held", "sliding", "spinning"])
def test_surface_contact_does_not_misclassify_nonplacements(support_adapter, failure):
    item, state, contact, data = support_adapter
    if failure == "side":
        contact.frame[:3] = [1., 0., 0.]
    elif failure == "ceiling":
        contact.frame[:3] = [0., 0., -1.]
    elif failure == "zero_force":
        state["force"] = 0.
    elif failure == "hovering":
        data.ncon = 0
    elif failure == "wrong_fixture":
        contact.geom1 = 3
    elif failure == "held":
        state["gripper_contact"] = True
    elif failure == "sliding":
        state["linear"] = np.array([.04, 0, 0])
    elif failure == "spinning":
        state["angular"] = np.array([0, 0, .3])
    result = item.evaluate({"predicate": "object_on_fixture", "object": "obj", "fixture": "counter"})
    assert result["known"] and not result["success"]


def test_support_predicate_reports_unknown_ids_and_invalid_measurements(support_adapter):
    item, state, _, _ = support_adapter
    condition = {"predicate": "object_on_fixture", "object": "obj", "fixture": "counter"}
    assert not item.evaluate({**condition, "object": "missing"})["known"]
    assert not item.evaluate({**condition, "fixture": "missing"})["known"]
    state["linear"] = np.array([float("nan"), 0, 0])
    assert not item.evaluate(condition)["known"]


@pytest.mark.parametrize("mode", ["success", "horizon", "native_stop"])
def test_observe_reports_lifecycle_without_success_score(tmp_path, mode):
    item = adapter(tmp_path, success_at=0 if mode == "success" else None)
    item.task_name, item.seed, item.instruction = "Test", 0, "test"
    item._backend._render = False
    item._semantic_catalog = lambda: {"object_ids": []}
    if mode == "horizon":
        item._backend.count = item.horizon
    if mode == "native_stop":
        item._backend._stopped = True
    result = item.observe()
    assert result["terminal"]
    assert result["control_interval_s"] == .05
    assert "native_success" not in result and "reward" not in result
    with pytest.raises(EpisodeFinished):
        item.step({})
