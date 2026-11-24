"""The only control methods exposed to generated atomic-skill programs."""
from __future__ import annotations

API_DESCRIPTION = """You implement skill(api, params) using these primitive methods only:
api.observe() -> instruction, step/horizon, terminal, control_interval_s,
  robot arm/torso measured joints and limits,
  robot.home_arm, robot.tcp {position, rotation, quaternion_xyzw}, gripper.aperture_m,
  robot.base {position, rotation, quaternion_xyzw, yaw, frame:'episode_initial_base',
  control_velocity_frame:'native_backend', control_units:'normalized_native_velocity'}
  is measured robot odometry in metres/radians, with
  initial position/yaw zero. It does not locate any fixture or semantic destination.
  cameras[name] {image_path, depth_path, fx,fy,cx,cy,T_base_camera}. Image paths
  refer to RGB observations; depth is numpy .npy in metres. No object/fixture truth.
  semantic_catalog lists benchmark object IDs, fixture IDs/types/joint names and
  aliases, without object/fixture poses or states. Use these IDs in plan predicates.
api.read_camera(camera) -> {rgb: HxWx3 uint8 numpy array, depth: HxW numpy array
  in metres, calibration: camera intrinsics and T_base_camera, step: int, camera: str}.
  Reads only a currently enabled camera observation. Arrays are independent copies;
  edits do not change sensors or simulation. No semantic recognizer or hidden labels.
  Use np for image/depth processing; return compact JSON summaries, not raw arrays.
api.step(action, repeat=1): action optional keys arm=[7 absolute radians],
  torso=[height metres], gripper=-1(open)/0(hold)/1(close), base=[ux,uy,uw]
  is a normalized native velocity request in the installed backend's command convention,
  bounded by [.5,.5,1] in absolute value. Base commands are NOT m/s or rad/s:
  the native controller applies its own scaling. Use measured odometry feedback
  to determine actual motion. Unspecified joints keep their references.
api.move_joints(target, duration=2.0, part='arm'|'torso', settle=.3, tolerance=.01)
api.move_tcp(position, rotation=None, speed=.10, duration=None, settle=.3,
  tolerance=.004): generic IK motion, current robot-base metres, proper 3x3
  rotation matrix. Local TCP +z points out of the gripper. All movement is physical;
  a false arrival result may mean contact, unreachable geometry or insufficient settling.
Timing: duration is a lower bound, not a fixed execution time. The controller uses
  a quintic rest-to-rest path whose peak speed is 1.875 times distance/duration.
  move_tcp uses at least max(.6, duration if supplied, 1.875*distance/speed,
  1.875*rotation_angle_radians/.8) seconds, then settle. move_joints uses at least
  max(duration, 1.875*max_joint_distance/max_velocity), where max_velocity is
  .8 rad/s for arm or .12 m/s for torso. Round trajectory and settle separately
  up to whole control frames. Read control_interval_s from observe(); it is .05 s
  (20 Hz) in the current PandaOmron configuration. Budget enough frames for the
  actual raised duration and settling, not merely distance/speed.
api.move_base(velocity, duration=.5): bounded normalized native velocity request then stop.
  robot.base.position/yaw and base_near use the FIXED episode-initial-base frame.
  Command orientation mapping depends on the installed native controller/backend;
  do not assume its channels equal odometry axes, especially after turning.
  Use bounded commands and measured arrival feedback; a target error in metres
  is not itself a normalized velocity. Ground semantic destinations visually.
api.gripper(direction, duration=.5): -1 opens, +1 closes; retains preload afterward.
api.wait(duration=.5): advance physics while retaining references.
api.deproject(camera,u,v,radius=1): RGB-D pixel to current robot-base [x,y,z].
  Camera names robot0_agentview_left, robot0_agentview_right, robot0_eye_in_hand;
  aliases camera_left, camera_right, camera_wrist also accepted. Invalid depth errors.
api.call(skill_id, params): call an explicitly bound version of a previously
  validated skill. Only dependencies selected for this attempt are callable.
Lifecycle: observe() and motion results include terminal. When terminal is true,
  stop all further control and return diagnostic data for independent host
  evaluation. It indicates an episode has ended, not a success score. A terminal
  move may have ok=false because its requested pose was not reached; preserve
  that measured arrival result rather than replacing it with a claimed verdict.
  Another motion call after terminal raises EpisodeFinished.
Use only the arguments in params for task-specific coordinates, pixels, durations,
and directions. Do not embed observed coordinates or seed identities into code.
Do not access simulator internals. No pick/place/press/turn prebuilt skill exists.
Return structured data and diagnostics. P/C output fields can be inputs to later
functions through api.call or next-plan params; self-reported status is not proof.
The host independently evaluates declared data contracts or physical postconditions
and the unchanged benchmark reward.
"""


class RuntimeAPI:
    """Narrow facade; the runner also validates AST and executes with a time bound.

    This is an API boundary, not an operating-system security sandbox. The host
    must not execute untrusted code outside its configured restricted runner.
    """
    __slots__ = ("__adapter", "__call_skill")

    def __init__(self, adapter, call_skill=None):
        self.__adapter = adapter
        self.__call_skill = call_skill

    def observe(self):
        return self.__adapter.observe()

    def read_camera(self, camera):
        return self.__adapter.read_camera(camera)

    def step(self, action, repeat=1):
        return self.__adapter.step(action, repeat)

    def move_joints(self, target, duration=2.0, **kwargs):
        return self.__adapter.move_joints(target, duration, **kwargs)

    def move_tcp(self, position, rotation=None, **kwargs):
        return self.__adapter.move_tcp(position, rotation, **kwargs)

    def move_base(self, velocity, duration=.5):
        return self.__adapter.move_base(velocity, duration)

    def gripper(self, direction, duration=.5):
        return self.__adapter.gripper(direction, duration)

    def wait(self, duration=.5):
        return self.__adapter.wait(duration)

    def deproject(self, camera, u, v, radius=1):
        return self.__adapter.deproject(camera, u, v, radius)

    def call(self, skill_id, params):
        if self.__call_skill is None:
            raise ValueError("No committed skill dependency is bound")
        return self.__call_skill(skill_id, params)
