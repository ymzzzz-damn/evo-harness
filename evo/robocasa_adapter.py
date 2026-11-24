"""RoboCasa's native physics and verdicts behind a small sensor/control surface.

No atomic skills are imported. Model introspection is restricted to the robot's
kinematic chain, joint limits and cameras. Object/fixture truth never enters the
agent observation. The optional RoboEnv installation owns the native controller
and its genuine ``env.step``; it is loaded only when this adapter is constructed.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import json
import math
import os
from pathlib import Path
import time


HOST_PREDICATES = {
    "native_success": "args:{}; native task verdict; use only on the final plan step.",
    "joint_near": "args:{part:'arm'|'torso',target:[7 radians or 1 metre],tolerance:positive number=.02}; measured joint arrival.",
    "tcp_near": "args:{position:[x,y,z],tolerance:positive metres=.01}; measured TCP position in current robot-base frame.",
    "base_near": "args:{position:[x,y],yaw?:radians,position_tolerance:positive metres=.1,yaw_tolerance:positive radians=.15,frame:'episode_initial_base'}; measured robot.base odometry arrival, with optional wrapped heading error. Bind the destination from observations; this does not identify a fixture or prove collision-free access. move_base uses native normalized velocity channels, not SI velocities; exact orientation mapping depends on the backend. Measure actual motion through robot.base odometry.",
    "gripper_aperture": "args:{min:metres,max:metres}; measured opening interval; NEVER proves object grasp.",
    "object_grasped": "args:{object:semantic_catalog.object_ids entry}; native opposing finger contact test; not a sustained stability guarantee.",
    "object_in_receptacle": "args:{object:object_id,receptacle:object_id}; native contact and horizontal containment in a movable container.",
    "object_in_fixture": "args:{object:object_id,fixture:fixture_id}; native full object containment in a fixture.",
    "object_on_fixture": "args:{object:object_id,fixture:fixture_id}; released object with direct upward loaded support from that fixture (>=half its weight), speed<=.03 m/s and angular speed<=.2 rad/s, and no robot gripper contact. Current settled support only: does not prove sustained stability, a particular burner, left/right placement, or containment. Side contact and hovering do not qualify.",
    "fixture_state": "args:{fixture:fixture_id,field:existing boolean state field (e.g. turned_on),value:boolean,slot_pair?:nonnegative integer,rack_level?:nonnegative integer}; selectors apply only to native getters supporting them; unknown fields produce known=false.",
    "fixture_joint": "args:{fixture:fixture_id,joint:fixture.joints entry,min:number,max:number}; native normalized joint interval, 0<=min<=max<=1.",
}


class EpisodeFinished(RuntimeError):
    """The official horizon or native success has ended the episode."""


def _json(value):
    if hasattr(value, "tolist"):
        return _json(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        # Preserve rejected nonfinite input as diagnostic text; don't let its
        # audit record replace a truthful known=false verdict with a JSON error.
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    return value


def _vector(value, size, name):
    import numpy as np
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} requires {size} finite numbers")
    return result


@contextmanager
def _renderer_compatibility():
    """OSMesa doesn't need the upstream CUDA/EGL device discovery."""
    from roboenv.provisioning import gpu
    original = gpu.configure_gpu
    if os.environ.get("MUJOCO_GL") == "osmesa":
        def configure(backend, options=None):
            return None if backend == "robocasa" else original(backend, options)
        gpu.configure_gpu = configure
    try:
        yield
    finally:
        gpu.configure_gpu = original


class SerialChain:
    """Pure FK over the robot's exported fixed geometry, no simulation writes."""
    def __init__(self, entries, joint_types):
        import numpy as np
        from scipy.spatial.transform import Rotation
        self.entries = []
        for entry in entries:
            origin = np.eye(4)
            origin[:3, :3] = Rotation.from_quat(entry["origin"][3:]).as_matrix()
            origin[:3, 3] = entry["origin"][:3]
            item = {**entry, "transform": origin}
            if "joint" in entry:
                axis = np.asarray(entry["axis"], dtype=float)
                x, y, z = axis
                skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
                item.update(axis=axis, anchor=np.asarray(entry["anchor"]), skew=skew,
                            square=skew @ skew, kind=joint_types[entry["joint"]])
            self.entries.append(item)

    def forward(self, q):
        import numpy as np
        result = np.eye(4)
        for entry in self.entries:
            result = result @ entry["transform"]
            if "joint" not in entry:
                continue
            offset = q[entry["joint"]] - entry["reference"]
            if entry["kind"] == 2:  # MuJoCo slide joint
                result[:3, 3] += result[:3, :3] @ (offset * entry["axis"])
            else:
                rotation = np.eye(3) + math.sin(offset) * entry["skew"] + (1-math.cos(offset)) * entry["square"]
                result[:3, 3] += result[:3, :3] @ (entry["anchor"] - rotation @ entry["anchor"])
                result[:3, :3] = result[:3, :3] @ rotation
        return result


class RoboCasaAdapter:
    def __init__(self, task_name, seed=0, output_dir=None, *, layout_ids=None,
                 style_ids=None, enable_render=True, camera_size=(256, 256),
                 scene_split=None, horizon=None):
        import numpy as np
        if scene_split not in (None, "pretrain"):
            raise ValueError("scene_split must be null or 'pretrain'")
        if horizon is not None and (type(horizon) is not int or horizon <= 0):
            raise ValueError("horizon must be a positive integer or null")
        backend_options = {}
        if scene_split == "pretrain":
            # Same mapping as official robocasa.utils.env_utils.create_env:
            # independently sample train layouts/styles 11..60 and pretrain objects.
            for name, value in (("layout_ids", layout_ids), ("style_ids", style_ids)):
                if value is not None and not (type(value) is int and value == -2):
                    raise ValueError(f"{name} conflicts with scene_split='pretrain'")
            layout_ids = style_ids = -2
            backend_options["obj_instance_split"] = "pretrain"
        if horizon is not None:
            # The caller freezes the official per-task dataset-registry horizon.
            # RoboEnv accepts max_steps, while native env.horizon is set below.
            backend_options["max_steps"] = horizon
        self.task_name, self.seed = str(task_name), int(seed)
        self.scene_split = scene_split
        self.output_dir = Path(output_dir or "runs/robocasa")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._captures = self.output_dir / "observations"
        self._captures.mkdir(exist_ok=True)
        self._capture_index = 0
        self._started = time.monotonic()
        self._closed = False
        self._events = self.output_dir / "control_events.jsonl"
        self._frame_cache = {}
        from roboenv_runtime.adapters.robocasa.backend import RoboCasaEnv
        with _renderer_compatibility():
            self._backend = RoboCasaEnv(self.task_name, seed=self.seed,
                layout_ids=layout_ids, style_ids=style_ids, enable_render=enable_render,
                camera_size=tuple(camera_size), stop_on_success=True, **backend_options)
        try:
            # Explicit official task budgets must govern native and wrapper stops.
            # Omitting horizon preserves the legacy native-environment behavior.
            self.horizon = int(self._backend.horizon) if horizon is None else horizon
            if horizon is not None:
                self._backend.horizon = self._backend.env.horizon = self.horizon
            self._backend.configured_max_steps = self._backend._max_steps = self.horizon
            self._backend.reset(seed=self.seed)
            self.instruction = self._backend.goal_text()
            self._build_robot_geometry()
        except Exception:
            self._backend.close()
            self._closed = True
            raise
        self.initial_native_success = bool(self._backend.task_completed())
        self._record("initialize", dict(task=self.task_name, seed=self.seed,
                     horizon=self.horizon, instruction=self.instruction,
                     scene_split=self.scene_split,
                     fk_validation=self._fk_validation))

    def _record(self, operation, values):
        entry = {"operation": operation, "step": self.steps, **_json(values)}
        with self._events.open("a") as stream:
            stream.write(json.dumps(entry, allow_nan=False) + "\n")

    @property
    def steps(self):
        return int(self._backend.get_step_budget()["used"])

    def _build_robot_geometry(self):
        import numpy as np
        from roboenv_runtime.adapters.robocasa.geometry import chain
        from roboenv_runtime.bindings.backends.robocasa.gripper_geometry import geometry
        native = self._backend
        robot, sim = native._robot, native.sim
        model, data = sim.model._model, sim.data._data
        self._base_id = sim.model.body_name2id(robot.robot_model.base.root_body)
        self._initial_base_transform = self._world_base_transform()
        self._flange_id = sim.model.body_name2id(robot.robot_model.eef_name["right"])
        self._tcp_id = sim.model.site_name2id(robot.gripper["right"].important_sites["grip_site"])
        self._joint_ids = {part: np.asarray(robot.part_controllers[part].joint_index, dtype=int)
                           for part in ("right", "torso", "right_gripper")}
        self._limits = {part: model.jnt_range[ids].copy() for part, ids in self._joint_ids.items()}
        arm_base = int(model.body_parentid[model.jnt_bodyid[self._joint_ids["right"][0]]])
        self._chains = {}
        for part, tip, base in (("right", self._flange_id, arm_base), ("torso", arm_base, self._base_id)):
            ids = self._joint_ids[part]
            self._chains[part] = SerialChain(chain(model, ids, tip, base)["chain"], model.jnt_type[ids])
        flange_r = data.xmat[self._flange_id].reshape(3, 3)
        self._flange_to_tcp = np.eye(4)
        self._flange_to_tcp[:3, :3] = flange_r.T @ data.site_xmat[self._tcp_id].reshape(3, 3)
        self._flange_to_tcp[:3, 3] = flange_r.T @ (data.site_xpos[self._tcp_id] - data.xpos[self._flange_id])
        self._gripper_facts, self._aperture_model = geometry(robot.gripper["right"], sim, self._joint_ids["right_gripper"])
        self._home = np.asarray(robot.init_qpos, dtype=float).tolist()
        measured = self._tcp_transform()
        computed = self._fk(self._joint_positions("right"))
        error = float(np.max(np.abs(computed-measured)))
        self._fk_validation = {"max_transform_error": error, "passed": error < 1e-6}
        if error >= 1e-6:
            raise RuntimeError(f"Robot FK does not match measured TCP: {error}")

    def _joint_positions(self, part):
        native = self._backend
        model, data = native.sim.model._model, native.sim.data._data
        return data.qpos[model.jnt_qposadr[self._joint_ids[part]]].copy()

    def _tcp_transform(self):
        import numpy as np
        data = self._backend.sim.data._data
        base_r = data.xmat[self._base_id].reshape(3, 3)
        result = np.eye(4)
        result[:3, :3] = base_r.T @ data.site_xmat[self._tcp_id].reshape(3, 3)
        result[:3, 3] = base_r.T @ (data.site_xpos[self._tcp_id]-data.xpos[self._base_id])
        return result

    def _world_base_transform(self):
        """Read only the robot's own base pose; never a fixture destination."""
        import numpy as np
        data = self._backend.sim.data._data
        result = np.eye(4)
        result[:3, :3] = data.xmat[self._base_id].reshape(3, 3)
        result[:3, 3] = data.xpos[self._base_id]
        return result

    def _base_observation(self):
        """Robot localization relative to its pose at this episode's reset."""
        import numpy as np
        from scipy.spatial.transform import Rotation
        initial, current = self._initial_base_transform, self._world_base_transform()
        rotation = initial[:3, :3].T @ current[:3, :3]
        position = initial[:3, :3].T @ (current[:3, 3]-initial[:3, 3])
        return {"position": position.tolist(), "rotation": rotation.tolist(),
                "quaternion_xyzw": Rotation.from_matrix(rotation).as_quat().tolist(),
                "yaw": float(np.arctan2(rotation[1, 0], rotation[0, 0])),
                "frame": "episode_initial_base",
                "control_velocity_frame": "native_backend",
                "control_units": "normalized_native_velocity"}

    def _fk(self, arm_q):
        return self._chains["torso"].forward(self._joint_positions("torso")) @ self._chains["right"].forward(arm_q) @ self._flange_to_tcp

    def _robot_observation(self):
        import numpy as np
        from scipy.spatial.transform import Rotation
        model, data = self._backend.sim.model._model, self._backend.sim.data._data
        parts = {}
        for part in ("right", "torso"):
            ids = self._joint_ids[part]
            parts["arm" if part == "right" else part] = {
                "positions": self._joint_positions(part).tolist(),
                "velocities": data.qvel[model.jnt_dofadr[ids]].tolist(),
                "limits": self._limits[part].tolist()}
        transform = self._tcp_transform()
        aperture = self._aperture_model["offset_m"] + np.dot(self._aperture_model["coefficients"], self._joint_positions("right_gripper"))
        return {**parts, "base": self._base_observation(), "home_arm": self._home, "tcp": {
            "position": transform[:3, 3].tolist(),
            "rotation": transform[:3, :3].tolist(),
            "quaternion_xyzw": Rotation.from_matrix(transform[:3, :3]).as_quat().tolist(),
            "frame": "robot_base"}, "gripper": {"aperture_m": max(0., float(aperture)),
            "limits_m": self._gripper_facts["aperture_m"]}, "time_s": float(data.time)}

    def _camera_calibration(self, camera):
        import numpy as np
        native = self._backend
        model, data = native.sim.model._model, native.sim.data._data
        index = native.sim.model.camera_name2id(camera)
        h, w = native._camera_size
        focal = float(h / (2*np.tan(np.deg2rad(model.cam_fovy[index])/2)))
        base_r = data.xmat[self._base_id].reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :3] = base_r.T @ data.cam_xmat[index].reshape(3, 3) @ np.diag([1, -1, -1])
        transform[:3, 3] = base_r.T @ (data.cam_xpos[index]-data.xpos[self._base_id])
        return {"width": w, "height": h, "fx": focal, "fy": focal,
                "cx": (w-1)/2, "cy": (h-1)/2, "T_base_camera": transform.tolist(),
                "depth_units": "metres", "depth_kind": "optical_axis_z"}

    def observe(self):
        import numpy as np
        from PIL import Image
        if self._closed:
            raise RuntimeError("Adapter is closed")
        cameras = {}
        if self._backend._render:
            for camera in self._backend._camera_names:
                key = (self.steps, camera)
                if key not in self._frame_cache:
                    rgb, depth = self._backend.render_rgbd(camera)
                    stem = self._captures / f"{self._capture_index:05d}_{self.steps:05d}_{camera}"
                    self._capture_index += 1
                    Image.fromarray(rgb).save(str(stem)+".png")
                    np.save(str(stem)+".depth.npy", depth, allow_pickle=False)
                    self._frame_cache[key] = {"image_path": str(stem)+".png",
                        "depth_path": str(stem)+".depth.npy", **self._camera_calibration(camera)}
                cameras[camera] = copy.deepcopy(self._frame_cache[key])
        return {"task": self.task_name, "seed": self.seed, "instruction": self.instruction,
                "step": self.steps, "horizon": self.horizon, "terminal": self._terminal(),
                "control_interval_s": float(self._backend.control_interval_s),
                "robot": self._robot_observation(), "cameras": cameras,
                "semantic_catalog": self._semantic_catalog(),
                "images": [{"path": value["image_path"], "camera": key, "modality": "rgb"}
                           for key, value in cameras.items()],
                "available_controls": ["step", "move_joints", "move_tcp", "move_base", "gripper", "wait", "deproject", "read_camera"]}

    def _terminal(self):
        """Episode lifecycle signal, independent of a requested pose's arrival."""
        return bool(getattr(self._backend, "_stopped", False)) or self.steps >= self.horizon or self.native_success()

    def _semantic_catalog(self):
        """Symbolic benchmark IDs for grounding; no poses, states or dimensions."""
        env = self._backend.env
        return {"object_ids": sorted(env.objects), "fixtures": [
            {"id": name, "type": type(fixture).__name__, "joints": list(getattr(fixture, "joints", None) or [])}
            for name, fixture in sorted(env.fixtures.items())],
            "fixture_aliases": self._fixture_aliases()}

    def _fixture_aliases(self):
        """Native role names only; recent RoboCasa stores (fixture, region flag)."""
        env = self._backend.env
        identifiers = {id(fixture): name for name, fixture in env.fixtures.items()}
        aliases = {}
        for name, value in getattr(env, "fixture_refs", {}).items():
            fixture = value[0] if isinstance(value, tuple) and value else value
            if id(fixture) in identifiers:
                aliases[name] = identifiers[id(fixture)]
        return aliases

    def _guard(self):
        if self._closed:
            raise RuntimeError("Adapter is closed")
        if self.steps >= self.horizon:
            raise EpisodeFinished("Native benchmark horizon exhausted")
        if self.native_success():
            raise EpisodeFinished("Native benchmark task succeeded")
        if getattr(self._backend, "_stopped", False):
            raise EpisodeFinished("Native environment stopped the episode")

    def step(self, action, repeat=1):
        """Validated low-level commands; each tick executes the native env.step."""
        import numpy as np
        self._guard()
        if not isinstance(action, dict) or not set(action) <= {"arm", "torso", "gripper", "base"}:
            raise ValueError("action supports only arm, torso, gripper and base")
        if type(repeat) is not int or not 1 <= repeat <= self.horizon:
            raise ValueError("repeat must be a positive bounded integer")
        commands = {}
        for key, value in action.items():
            if key in ("arm", "torso"):
                part = "right" if key == "arm" else key
                command = _vector(value, len(self._joint_ids[part]), key)
                limits = self._limits[part]
                if (command < limits[:, 0]).any() or (command > limits[:, 1]).any():
                    raise ValueError(f"{key} target exceeds robot joint limits")
            elif key == "base":
                part, command = "base", _vector(value, 3, "base")
                if (np.abs(command) > np.array([.5, .5, 1.])).any():
                    raise ValueError("normalized native base command exceeds x/y limits of 0.5 or yaw limit of 1")
            else:
                part, command = "right_gripper", _vector([value], 1, "gripper")
                if abs(command[0]) > 1:
                    raise ValueError("gripper direction must be in [-1, 1]")
            commands[part] = command
        before = self.steps
        self._backend.step(commands)
        for _ in range(repeat):
            if self._terminal():
                break
            self._backend.tick()
        self._record("step", {"action": action, "requested_repeat": repeat, "step_before": before, "steps_executed": self.steps-before})
        # Control feedback deliberately omits benchmark truth.
        return {"steps": self.steps-before, "step": self.steps, "robot": self._robot_observation(),
                "terminal": self._terminal()}

    def wait(self, duration=0.5):
        duration = float(duration)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be finite and positive")
        return self.step({}, repeat=min(self.horizon, max(1, math.ceil(duration/self._backend.control_interval_s))))

    def move_joints(self, target, duration=2.0, *, part="arm", settle=.3, tolerance=.01):
        import numpy as np
        self._guard()
        if part not in ("arm", "torso"):
            raise ValueError("part must be arm or torso")
        native_part = "right" if part == "arm" else part
        start = self._joint_positions(native_part)
        goal = _vector(target, len(start), "target")
        bounds = self._limits[native_part]
        if (goal < bounds[:, 0]).any() or (goal > bounds[:, 1]).any():
            raise ValueError("target exceeds robot joint limits")
        duration, settle, tolerance = float(duration), float(settle), float(tolerance)
        if not math.isfinite(duration+settle+tolerance) or duration <= 0 or settle < 0 or tolerance <= 0:
            raise ValueError("positive duration/tolerance and nonnegative settle required")
        # Quintic rest-to-rest references; cap arm velocity at 0.8 rad/s, lift at 0.12 m/s.
        duration = max(duration, 1.875*float(np.max(np.abs(goal-start)))/(.8 if part == "arm" else .12))
        count = max(1, math.ceil(duration/self._backend.control_interval_s))
        for i in range(1, count+1):
            if self._terminal():
                break
            f = i/count
            f = f**3*(10+f*(-15+6*f))
            # Measured limit joints can drift a few micrometres beyond bounds;
            # clamp references rather than rejecting a valid inward motion.
            self.step({part: np.clip(start+f*(goal-start), bounds[:, 0], bounds[:, 1]).tolist()})
        if settle and not self._terminal():
            self.wait(settle)
        actual = self._joint_positions(native_part)
        error = float(np.max(np.abs(actual-goal)))
        return {"ok": error <= float(tolerance), "positions": actual.tolist(), "error": error,
                "terminal": self._terminal()}

    def gripper(self, direction, duration=.5):
        duration = float(duration)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be positive")
        result = self.step({"gripper": float(direction)}, min(self.horizon, max(1, math.ceil(duration/self._backend.control_interval_s))))
        # Stop integrating the gripper direction, retaining its native preload.
        self._backend.stop_part("right_gripper")
        return result

    def move_base(self, velocity, duration=.5):
        duration = float(duration)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be positive")
        try:
            return self.step({"base": velocity}, min(self.horizon, max(1, math.ceil(duration/self._backend.control_interval_s))))
        finally:
            self._backend.stop_part("base")

    def _ik(self, position, rotation, seed):
        import numpy as np
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
        limits = self._limits["right"]
        def residual(q):
            transform = self._fk(q)
            return np.r_[position-transform[:3, 3], Rotation.from_matrix(rotation @ transform[:3, :3].T).as_rotvec()]
        fit = least_squares(residual, np.clip(seed, limits[:, 0]+1e-7, limits[:, 1]-1e-7),
                            bounds=(limits[:, 0], limits[:, 1]), max_nfev=100,
                            ftol=1e-8, xtol=1e-8, gtol=1e-8)
        error = residual(fit.x)
        if np.linalg.norm(error[:3]) > .001 or np.linalg.norm(error[3:]) > .005:
            raise ValueError("IK target unreachable from this posture; change posture or target")
        return fit.x

    def move_tcp(self, position, rotation=None, *, speed=.10, duration=None, settle=.3, tolerance=.004):
        import numpy as np
        from scipy.spatial.transform import Rotation, Slerp
        self._guard()
        target = _vector(position, 3, "position")
        start = self._tcp_transform()
        target_r = start[:3, :3] if rotation is None else np.asarray(rotation, dtype=float)
        if target_r.shape != (3, 3) or not np.isfinite(target_r).all() or not np.allclose(target_r.T@target_r, np.eye(3), atol=1e-5) or np.linalg.det(target_r) < 0:
            raise ValueError("rotation must be a proper 3x3 matrix")
        speed, settle, tolerance = float(speed), float(settle), float(tolerance)
        if not math.isfinite(speed+settle+tolerance) or speed <= 0 or settle < 0 or tolerance <= 0:
            raise ValueError("positive speed/tolerance and nonnegative settle required")
        angle = float((Rotation.from_matrix(start[:3, :3]).inv()*Rotation.from_matrix(target_r)).magnitude())
        minimum = max(.6, 1.875*float(np.linalg.norm(target-start[:3, 3]))/speed, 1.875*angle/.8)
        if duration is None:
            duration = minimum
        else:
            duration = float(duration)
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("duration must be positive")
            duration = max(minimum, duration)
        count = max(2, math.ceil(duration/self._backend.control_interval_s))
        slerp = Slerp([0, 1], Rotation.from_matrix(np.array([start[:3, :3], target_r])))
        q = self._joint_positions("right")
        # Validate the whole Cartesian path before beginning its motion.
        path = []
        for i in range(1, count+1):
            f = i/count
            f = f**3*(10+f*(-15+6*f))
            next_q = self._ik(start[:3, 3]+f*(target-start[:3, 3]), slerp(f).as_matrix(), q)
            if np.max(np.abs(next_q-q)) > .65:
                raise ValueError("IK path has a joint branch jump; use an intermediate posture")
            path.append(next_q.tolist())
            q = next_q
        for q in path:
            if self._terminal():
                break
            self.step({"arm": q})
        if settle and not self._terminal():
            self.wait(settle)
        actual = self._tcp_transform()
        error = float(np.linalg.norm(target-actual[:3, 3]))
        rotation_error = float((Rotation.from_matrix(actual[:3, :3]).inv()*Rotation.from_matrix(target_r)).magnitude())
        return {"ok": error <= tolerance and rotation_error <= .05,
                "position": actual[:3, 3].tolist(), "error_m": error, "rotation_error_rad": rotation_error,
                "terminal": self._terminal()}

    def read_camera(self, camera):
        """Return detached arrays from the current public RGB-D frame only."""
        import numpy as np
        from PIL import Image
        aliases = {"camera_left": "robot0_agentview_left", "camera_right": "robot0_agentview_right", "camera_wrist": "robot0_eye_in_hand"}
        camera = aliases.get(camera, camera)
        observation = self.observe()
        if camera not in observation["cameras"]:
            raise ValueError("Camera is not enabled")
        metadata = observation["cameras"][camera]
        with Image.open(metadata["image_path"]) as frame:
            rgb = np.array(frame.convert("RGB"), copy=True)
        depth = np.load(metadata["depth_path"], allow_pickle=False).copy()
        return {"rgb": rgb, "depth": depth, "step": observation["step"], "camera": camera,
                "calibration": copy.deepcopy({k: v for k, v in metadata.items() if k not in ("image_path", "depth_path")})}

    def deproject(self, camera, u, v, radius=1):
        import numpy as np
        aliases = {"camera_left": "robot0_agentview_left", "camera_right": "robot0_agentview_right", "camera_wrist": "robot0_eye_in_hand"}
        camera = aliases.get(camera, camera)
        observation = self.observe()
        if camera not in observation["cameras"]:
            raise ValueError("Camera is not enabled")
        if type(radius) is not int or not 0 <= radius <= 10:
            raise ValueError("radius must be an integer between 0 and 10")
        metadata = observation["cameras"][camera]
        if not math.isfinite(float(u)+float(v)) or not (0 <= u < metadata["width"] and 0 <= v < metadata["height"]):
            raise ValueError("Pixel lies outside image")
        depth = np.load(metadata["depth_path"], allow_pickle=False)
        x, y = int(round(u)), int(round(v))
        patch = depth[max(0, y-radius):min(metadata["height"], y+radius+1), max(0, x-radius):min(metadata["width"], x+radius+1)]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if not valid.size:
            raise ValueError("Pixel has no valid measured depth")
        z = float(np.median(valid))
        point = np.array([(u-metadata["cx"])*z/metadata["fx"], (v-metadata["cy"])*z/metadata["fy"], z, 1.])
        return (np.asarray(metadata["T_base_camera"]) @ point)[:3].tolist()

    def native_success(self):
        # This is a cached verdict from native reward; don't re-evaluate predicates
        # on reads because some benchmark tasks count history inside the predicate.
        return bool(self._backend.task_completed())

    def evaluate(self, postcondition):
        import numpy as np
        predicate = postcondition.get("predicate", postcondition.get("type"))
        values = postcondition.get("args", postcondition)
        result = {"success": False, "known": True, "reason": "", "evidence": {"step": self.steps}}
        if predicate == "native_success":
            result.update(success=self.native_success(), reason="RoboCasa native reward verdict")
        elif predicate == "joint_near":
            part = values.get("part", "arm")
            if part not in ("arm", "torso"):
                raise ValueError("joint_near part must be arm or torso")
            actual = self._joint_positions("right" if part == "arm" else part)
            target = _vector(values["target"], len(actual), "target")
            error = float(np.max(np.abs(actual-target)))
            tolerance = float(values.get("tolerance", .02))
            if not math.isfinite(tolerance) or tolerance <= 0:
                raise ValueError("tolerance must be finite and positive")
            result.update(success=error <= tolerance, reason="Measured joint arrival")
            result["evidence"].update(actual=actual.tolist(), error=error, tolerance=tolerance)
        elif predicate == "tcp_near":
            actual = self._tcp_transform()[:3, 3]
            error = float(np.linalg.norm(actual-_vector(values["position"], 3, "position")))
            tolerance = float(values.get("tolerance", .01))
            if not math.isfinite(tolerance) or tolerance <= 0:
                raise ValueError("tolerance must be finite and positive")
            result.update(success=error <= tolerance, reason="Measured TCP arrival")
            result["evidence"].update(actual=actual.tolist(), error_m=error, tolerance=tolerance)
        elif predicate == "gripper_aperture":
            actual = self._robot_observation()["gripper"]["aperture_m"]
            low, high = float(values["min"]), float(values["max"])
            if not math.isfinite(low+high) or low < 0 or low > high:
                raise ValueError("Invalid aperture interval")
            result.update(success=low <= actual <= high, reason="Measured gripper opening; not an object-grasp assertion")
            result["evidence"].update(aperture_m=actual)
        elif predicate == "base_near":
            if values.get("frame", "episode_initial_base") != "episode_initial_base":
                raise ValueError("base_near targets use the episode_initial_base frame")
            actual = self._base_observation()
            target = _vector(values["position"], 2, "position")
            position_error = float(np.linalg.norm(np.asarray(actual["position"][:2])-target))
            position_tolerance = float(values.get("position_tolerance", .1))
            yaw_tolerance = float(values.get("yaw_tolerance", .15))
            if not (math.isfinite(position_tolerance) and position_tolerance > 0
                    and math.isfinite(yaw_tolerance) and 0 < yaw_tolerance <= math.pi):
                raise ValueError("base_near tolerances must be finite and positive; yaw tolerance <= pi")
            yaw_error = None
            if "yaw" in values:
                yaw = float(values["yaw"])
                if not math.isfinite(yaw):
                    raise ValueError("base_near yaw must be finite")
                yaw_error = abs(math.atan2(math.sin(actual["yaw"]-yaw), math.cos(actual["yaw"]-yaw)))
            result.update(success=position_error <= position_tolerance and (yaw_error is None or yaw_error <= yaw_tolerance),
                          reason="Measured robot base arrival in episode odometry; semantic destination grounding is separate")
            result["evidence"].update(position=actual["position"][:2], yaw=actual["yaw"],
                                      position_error_m=position_error, yaw_error_rad=yaw_error,
                                      frame="episode_initial_base")
        elif predicate in {"object_grasped", "object_in_receptacle", "object_in_fixture", "object_on_fixture", "fixture_state", "fixture_joint"}:
            # These predicates stay owner-side. Only a verdict/reason is returned,
            # never simulator positions, fixture geometry or mutable references.
            try:
                success, reason = self._semantic_predicate(predicate, values)
                result.update(success=bool(success), reason=reason)
            except (KeyError, AttributeError, ValueError, TypeError, AssertionError, IndexError) as error:
                result.update(known=False, reason=f"Predicate unavailable: {error}")
        else:
            result.update(known=False, reason=f"Unsupported host predicate: {predicate}")
        self._record("evaluate", {"postcondition": postcondition, "verdict": result})
        return result

    def _fixture(self, name):
        env = self._backend.env
        if name in env.fixtures:
            return env.fixtures[name]
        aliases = self._fixture_aliases()
        if name in aliases:
            return env.fixtures[aliases[name]]
        item = getattr(env, str(name), None)
        if item is not None and any(item is value for value in env.fixtures.values()):
            return item
        matches = [value for value in env.fixtures.values() if type(value).__name__.lower() == str(name).lower()]
        if len(matches) == 1:
            return matches[0]
        raise KeyError(f"Unknown or ambiguous fixture {name!r}")

    def _semantic_predicate(self, predicate, values):
        from robocasa.utils import object_utils
        env = self._backend.env
        if predicate == "object_grasped":
            obj = env.objects[values["object"]]
            return env._check_grasp(env.robots[0].gripper["right"], obj.contact_geoms), "Native opposing finger contact test; sustained stability is not evaluated"
        if predicate == "object_in_receptacle":
            return object_utils.check_obj_in_receptacle(env, values["object"], values["receptacle"]), "Native object/receptacle contact and horizontal containment"
        fixture = self._fixture(values["fixture"])
        if predicate == "object_in_fixture":
            return object_utils.obj_inside_of(env, values["object"], fixture), "Native object/fixture containment"
        if predicate == "object_on_fixture":
            return self._object_on_fixture(values["object"], fixture), "Released, slow object with direct upward load-bearing fixture contact; current support, not sustained or relative-placement proof"
        if predicate == "fixture_state":
            import inspect
            import numpy as np
            # Native getters are not uniform: e.g. microwave needs no env,
            # kettle needs env, toaster additionally accepts a slot_pair.
            # Inspect once; don't retry a getter after an internal TypeError.
            parameters = inspect.signature(fixture.get_state).parameters
            kwargs = {"env": env} if "env" in parameters else {}
            for selector in ("slot_pair", "rack_level"):
                if selector in values:
                    if selector not in parameters or type(values[selector]) is not int or values[selector] < 0:
                        raise ValueError(f"Invalid or unsupported fixture state selector {selector}")
                    kwargs[selector] = values[selector]
            state = fixture.get_state(**kwargs)
            field, expected = values["field"], values["value"]
            if type(expected) is not bool or not isinstance(state.get(field), (bool, np.bool_)):
                raise ValueError("fixture_state requires an existing boolean state field and boolean value")
            return bool(state[field]) == expected, "Native fixture boolean state"
        joint = values["joint"]
        if joint not in fixture.joints:
            raise ValueError("Joint does not belong to selected fixture")
        low, high = float(values["min"]), float(values["max"])
        if not math.isfinite(low+high) or not 0 <= low <= high <= 1:
            raise ValueError("Fixture normalized joint interval must lie in [0,1]")
        position = float(fixture.get_joint_state(env, [joint])[joint])
        return low <= position <= high, "Native normalized fixture joint interval"

    def _object_on_fixture(self, object_name, fixture):
        """Host-only contact/velocity evidence, without exporting scene geometry.

        Mere collision is insufficient: a side wall or a zero-force margin
        contact cannot support an object. MuJoCo's contact normal points from
        geom1 to geom2, so its sign is reversed when the object is geom1.
        A positive result is a current physical state, not a time-window claim.
        """
        import mujoco
        import numpy as np
        env, native = self._backend.env, self._backend
        obj = env.objects[object_name]
        for robot in env.robots:
            grippers = robot.gripper.values() if isinstance(robot.gripper, dict) else [robot.gripper]
            if any(env.check_contact(gripper, obj) for gripper in grippers):
                return False
        linear = np.asarray(native.sim.data.get_body_xvelp(obj.root_body), dtype=float)
        angular = np.asarray(native.sim.data.get_body_xvelr(obj.root_body), dtype=float)
        if not np.isfinite(linear).all() or not np.isfinite(angular).all():
            raise ValueError("Object velocity unavailable")
        if np.linalg.norm(linear) > .03 or np.linalg.norm(angular) > .2:
            return False
        model, data = native.sim.model._model, native.sim.data._data
        body = native.sim.model.body_name2id(obj.root_body)
        gravity = np.asarray(model.opt.gravity, dtype=float)
        gravity_norm = float(np.linalg.norm(gravity))
        weight = float(model.body_subtreemass[body])*gravity_norm
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("Object weight unavailable for support verification")
        up = -gravity/gravity_norm
        obj_geoms = {native.sim.model.geom_name2id(name) for name in obj.contact_geoms}
        fixture_geoms = {native.sim.model.geom_name2id(name) for name in fixture.contact_geoms}
        if not obj_geoms or not fixture_geoms:
            raise ValueError("Object or fixture has no contact geometry")
        support = 0.
        force = np.zeros(6)
        for index in range(data.ncon):
            contact = data.contact[index]
            if contact.geom1 in fixture_geoms and contact.geom2 in obj_geoms:
                sign = 1.
            elif contact.geom2 in fixture_geoms and contact.geom1 in obj_geoms:
                sign = -1.
            else:
                continue
            upward = float(np.dot(sign*np.asarray(contact.frame[:3]), up))
            if upward < .5:
                continue
            force[:] = 0
            mujoco.mj_contactForce(model, data, index, force)
            if not np.isfinite(force).all():
                raise ValueError("Contact force unavailable")
            support += max(0., float(force[0]))*upward
        return support >= .5*weight

    def stats(self):
        return {"task": self.task_name, "seed": self.seed, "steps": self.steps,
                "horizon": self.horizon, "native_success": self.native_success(),
                "terminal": self._terminal(),
                "reward": float(self.native_success()), "wall_seconds": time.monotonic()-self._started,
                "fk_validation": self._fk_validation, "control_log": str(self._events),
                "observation_policy": "RGB-D and robot measurements; no scene ground truth"}

    def close(self):
        if not self._closed:
            self._record("close", self.stats())
            self._backend.close()
            self._closed = True
