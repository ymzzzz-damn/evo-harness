"""Bounded native RoboCasa control smoke; it does not solve an atomic skill.

Run inside the prepared RoboCasa interpreter with ``python -m evo.adapter_smoke``.
EVO_RUN points to an isolated evidence directory.
"""
import json
import os
from pathlib import Path
import traceback

from .robocasa_adapter import RoboCasaAdapter


def main():
    root = Path(os.environ["EVO_RUN"])
    report = {"status": "error", "benchmark_success_claimed": False}
    adapter = None
    try:
        adapter = RoboCasaAdapter("TurnOnMicrowave", seed=0, output_dir=root/"control-smoke")
        initial = adapter.observe()
        report["initial"] = adapter.stats()
        report["home"] = adapter.move_joints(initial["robot"]["home_arm"], duration=4., settle=.5, tolerance=.04)
        report["torso"] = adapter.move_joints([.05], part="torso", duration=1., settle=.4)
        observation = adapter.observe()
        camera = observation["cameras"]["robot0_agentview_right"]
        report["deprojection"] = adapter.deproject("camera_right", camera["cx"], camera["cy"])
        target = observation["robot"]["tcp"]["position"][:]
        target[2] += .01
        report["tcp"] = adapter.move_tcp(target, speed=.04, settle=.5, tolerance=.01)
        report["host_arrival"] = adapter.evaluate({"predicate": "tcp_near", "position": target, "tolerance": .01})
        report["host_fixture"] = adapter.evaluate({"predicate": "fixture_state", "fixture": "microwave", "field": "turned_on", "value": False})
        report["final"] = adapter.stats()
        report["images"] = adapter.observe()["images"]
        if not report["home"]["ok"] or not report["torso"]["ok"] or not report["tcp"]["ok"]:
            raise AssertionError("A measured low-level arrival missed tolerance")
        if not report["host_arrival"]["success"] or not report["host_fixture"]["success"]:
            raise AssertionError("Independent host control check failed")
        report["status"] = "passed"
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        if adapter:
            adapter.close()
        (root/"adapter_control_smoke.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
