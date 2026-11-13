"""Verify native horizon exhaustion followed by same-seed world recreation.

Run in the prepared RoboCasa worker with EVO_RUN pointing at an isolated run
directory. This performs no task policy and never claims benchmark success.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import weakref

import numpy as np
from PIL import Image

from evo.io import write_json
from evo.robocasa_adapter import RoboCasaAdapter


def process_resources():
    """Diagnostics only: allocators may retain pages after native resources free."""
    result = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                result["resident_kib"] = int(line.split()[1])
                break
        result["open_file_descriptors"] = len(list(Path("/proc/self/fd").iterdir()))
    except OSError:
        pass
    return result


def compare_initial_observations(first, second):
    """Compare saved sensors and robot feedback, never privileged scene poses."""
    measurements = {}
    for name, path in {
        "arm_positions": ("arm", "positions"),
        "arm_velocities": ("arm", "velocities"),
        "torso_positions": ("torso", "positions"),
        "torso_velocities": ("torso", "velocities"),
        "tcp_position": ("tcp", "position"),
        "tcp_rotation": ("tcp", "rotation"),
        "tcp_quaternion_xyzw": ("tcp", "quaternion_xyzw"),
        "gripper_aperture": ("gripper", "aperture_m"),
    }.items():
        a = np.asarray(first["robot"][path[0]][path[1]], dtype=float)
        b = np.asarray(second["robot"][path[0]][path[1]], dtype=float)
        same_shape = a.shape == b.shape
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        measurements[name] = {
            "same_shape": same_shape,
            "finite": finite,
            "max_absolute_difference": float(np.max(np.abs(a-b))) if same_shape and finite else None,
            "equal_at_1e_6": bool(same_shape and finite and np.allclose(a, b, atol=1e-6, rtol=0)),
        }
    cameras = {}
    for name in sorted(set(first["cameras"]) & set(second["cameras"])):
        a, b = first["cameras"][name], second["cameras"][name]
        with Image.open(a["image_path"]) as image:
            rgb_a = np.asarray(image).astype(np.int16)
        with Image.open(b["image_path"]) as image:
            rgb_b = np.asarray(image).astype(np.int16)
        depth_a = np.load(a["depth_path"], allow_pickle=False)
        depth_b = np.load(b["depth_path"], allow_pickle=False)
        rgb_shape_matches, depth_shape_matches = rgb_a.shape == rgb_b.shape, depth_a.shape == depth_b.shape
        cameras[name] = {
            "first_rgb": a["image_path"], "second_rgb": b["image_path"],
            "first_depth": a["depth_path"], "second_depth": b["depth_path"],
            "first_rgb_sha256": hashlib.sha256(Path(a["image_path"]).read_bytes()).hexdigest(),
            "second_rgb_sha256": hashlib.sha256(Path(b["image_path"]).read_bytes()).hexdigest(),
            "rgb_shape_matches": rgb_shape_matches,
            "rgb_pixels_identical": bool(rgb_shape_matches and np.array_equal(rgb_a, rgb_b)),
            "rgb_max_absolute_difference": int(np.max(np.abs(rgb_a-rgb_b))) if rgb_shape_matches else None,
            "depth_shape_matches": depth_shape_matches,
            "depth_identical": bool(depth_shape_matches and np.array_equal(depth_a, depth_b)),
            "depth_max_absolute_difference_m": float(np.max(np.abs(depth_a-depth_b))) if depth_shape_matches else None,
            "camera_transform_max_absolute_difference": float(np.max(np.abs(np.asarray(a["T_base_camera"])-np.asarray(b["T_base_camera"])))),
        }
    return {
        "task_matches": first["task"] == second["task"],
        "seed_matches": first["seed"] == second["seed"],
        "instruction_matches": first["instruction"] == second["instruction"],
        "camera_names_match": sorted(first["cameras"]) == sorted(second["cameras"]),
        "robot_measurements": measurements,
        "robot_matches_at_1e_6": all(value["equal_at_1e_6"] for value in measurements.values()),
        "cameras": cameras,
        "all_rgb_pixels_identical": all(value["rgb_pixels_identical"] for value in cameras.values()),
        "all_depth_arrays_identical": all(value["depth_identical"] for value in cameras.values()),
        "image_comparison_is_diagnostic": True,
    }


def main():
    root = Path(os.environ["EVO_RUN"])
    root.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running", "benchmark_success_claimed": False,
        "scope": "One native horizon exhaustion and one same-seed fresh recreation; not a long-run leak test.",
        "task": "TurnOnMicrowave", "seed": 0,
        "resources_before": process_resources(), "episodes": [], "cleanup": [],
    }
    started = time.monotonic()
    adapter = None
    active_episode = None

    def checkpoint(stage):
        report["stage"] = stage
        report["wall_seconds"] = time.monotonic()-started
        write_json(root/"reset_smoke.json", report)

    def close_current():
        nonlocal adapter
        if adapter is None:
            return
        reference = weakref.ref(adapter)
        adapter.close()
        # Owner-side lifecycle audit only. Native close is expected to free MjSim
        # and set env.sim=None; nothing here writes simulator state.
        released = adapter._backend.env.sim is None
        adapter = None
        gc.collect()
        cleanup = {"episode": active_episode, "close_returned": True,
                   "native_sim_released": released, "adapter_collected": reference() is None,
                   "process_resources": process_resources()}
        report["cleanup"].append(cleanup)
        if not released or not cleanup["adapter_collected"]:
            raise AssertionError("Native simulation or adapter was retained after close")

    checkpoint("initializing_first_episode")
    try:
        active_episode = 1
        adapter = RoboCasaAdapter("TurnOnMicrowave", seed=0, output_dir=root/"episode_1")
        first_initial = adapter.observe()
        write_json(root/"episode_1"/"initial.json", first_initial)
        before = adapter.stats()
        report["episodes"].append({"episode": 1, "initial": before,
                                   "initial_observation": str(root/"episode_1"/"initial.json")})
        if before["steps"] != 0 or before["horizon"] != 1000 or before["terminal"] or before["native_success"]:
            raise AssertionError("First episode did not start at the untouched native 1000-step budget")
        checkpoint("exhausting_native_horizon")
        control = adapter.step({}, repeat=before["horizon"])
        terminal = adapter.stats()
        terminal_observation = adapter.observe()
        write_json(root/"episode_1"/"terminal.json", terminal_observation)
        report["episodes"][0].update(terminal=terminal, control=control,
            terminal_observation=str(root/"episode_1"/"terminal.json"))
        if terminal["steps"] != 1000 or terminal["horizon"] != 1000 or not terminal["terminal"] or terminal["native_success"]:
            raise AssertionError("Expected native horizon exhaustion without benchmark task success")
        close_current()
        checkpoint("first_episode_closed")

        active_episode = 2
        adapter = RoboCasaAdapter("TurnOnMicrowave", seed=0, output_dir=root/"episode_2")
        second_initial = adapter.observe()
        write_json(root/"episode_2"/"initial.json", second_initial)
        second_before = adapter.stats()
        report["episodes"].append({"episode": 2, "initial": second_before,
                                   "initial_observation": str(root/"episode_2"/"initial.json")})
        if second_before["steps"] != 0 or second_before["horizon"] != 1000 or second_before["terminal"] or second_before["native_success"]:
            raise AssertionError("New same-seed adapter did not provide a fresh active native budget")
        comparison = compare_initial_observations(first_initial, second_initial)
        report["same_seed_comparison"] = comparison
        if not all(comparison[key] for key in ("task_matches", "seed_matches", "instruction_matches", "camera_names_match", "robot_matches_at_1e_6")):
            raise AssertionError("Same-seed recreated episode differed in its task identity or initial robot measurements")
        checkpoint("second_episode_active")
        # Five actual native control frames, with observable gripper movement.
        control = adapter.gripper(-1, duration=5*second_initial["control_interval_s"])
        second_after = adapter.stats()
        second_observation = adapter.observe()
        write_json(root/"episode_2"/"after_control.json", second_observation)
        aperture_change = second_observation["robot"]["gripper"]["aperture_m"]-second_initial["robot"]["gripper"]["aperture_m"]
        report["episodes"][1].update(after_control=second_after, control=control,
            gripper_aperture_change_m=aperture_change,
            after_control_observation=str(root/"episode_2"/"after_control.json"))
        if second_after["steps"] != 5 or second_after["terminal"] or second_after["native_success"] or aperture_change <= 1e-6:
            raise AssertionError("Fresh episode did not execute five active control frames and open its gripper")
        close_current()
        report["status"] = "passed"
        checkpoint("complete")
    except Exception as error:
        report["status"] = "error"
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        if adapter is not None:
            try:
                close_current()
            except Exception as error:
                report["status"] = "error"
                report["cleanup_error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        report["resources_after"] = process_resources()
        checkpoint(report["stage"])
        print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
