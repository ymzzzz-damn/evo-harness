"""Physical initialization and control checks; this does not claim task success."""
from pathlib import Path
import os
import traceback
from evo.io import write_json
from evo.robocasa_adapter import RoboCasaAdapter

root = Path(os.environ["EVO_RUN"])
adapter = None
report = {"status":"error", "benchmark_success_claimed":False}
try:
    adapter = RoboCasaAdapter("TurnOnMicrowave", seed=0, output_dir=root/"preflight")
    initial = adapter.observe()
    write_json(root/"initial.json", initial)
    before = adapter.stats()
    control = adapter.wait(.25)
    observation = adapter.observe()
    report.update(status="passed", before=before, after=adapter.stats(), control=control,
                  initial_success=adapter.initial_native_success, native_success=adapter.native_success(),
                  images=observation["images"])
    if adapter.stats()["steps"] <= before["steps"]:
        raise AssertionError("Native physics did not advance")
except Exception as error:
    report.update(status="error", error={"type":type(error).__name__, "message":str(error), "traceback":traceback.format_exc()})
finally:
    if adapter:
        adapter.close()
    write_json(root/"preflight.json", report)
    print(report, flush=True)
if report["status"] != "passed":
    raise SystemExit(1)
