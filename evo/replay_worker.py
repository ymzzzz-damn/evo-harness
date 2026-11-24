"""One fresh replay in a new interpreter; never creates a mutable Library."""
import argparse
import ctypes
import hashlib
import os
from pathlib import Path
import signal
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--token-fd", type=int, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args(argv)
    # Replay children inherit the task worker's process group. Also terminate if
    # that exact parent dies alone, including the race before prctl is installed.
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Replay process parent-death protection requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot install replay parent-death signal")
    if os.getppid() != args.parent_pid:
        raise RuntimeError("Replay owner exited before child initialization")
    from .io import read_json, write_json
    from .replay_dispatch import FrozenVersionResolver, import_ref
    raw = Path(args.input).read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.input_sha256:
        raise ValueError("Replay input changed before child start")
    payload = read_json(args.input)
    output = Path(payload["output"])
    try:
        # Keep the inherited descriptor open until interpreter exit. Closing it
        # early could expose a slot while GL/environment resources remain alive.
        os.fstat(args.token_fd)
        module_dir = Path(__file__).resolve().parent
        for name, digest in payload["source_sha256"].items():
            if name not in {"loops.py", "replay_dispatch.py", "replay_worker.py"} or hashlib.sha256((module_dir / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Replay implementation differs from parent snapshot")
        assignment = payload["assignment"]
        if os.environ.get("CUDA_VISIBLE_DEVICES") != assignment["uuid"]:
            raise ValueError("Replay UUID environment does not match its leased assignment")
        for key in ("ROBOENV_GPU", "MUJOCO_EGL_DEVICE_ID"):
            os.environ.pop(key, None)
        os.environ.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
        refs = payload["refs"]
        import_ref(refs["renderer_mapping_ref"])()
        factory = import_ref(refs["factory_ref"])
        runtime = import_ref(refs["runtime_ref"])
        renderer = import_ref(refs["renderer_evidence_ref"])
        fresh = import_ref(refs["fresh_scene_ref"])
        resolver = FrozenVersionResolver(payload["dependency_records"])
        def checked_factory(config, directory):
            adapter = factory(config, directory)
            try:
                initial = fresh(adapter)
                if initial.get("raw_native_success_at_reset") is not False or initial.get("native_steps_at_reset") != 0:
                    raise ValueError("Replay lacks a valid false/step0 fresh reset")
                adapter.observe()
                binding = renderer(assignment)
                if (binding.get("assignment", {}).get("uuid") != assignment["uuid"]
                        or binding.get("roboenv_binding", {}).get("uuid") != assignment["uuid"]
                        or "NVIDIA" not in binding.get("gl_vendor", "") or not binding.get("gl_renderer")):
                    raise ValueError("Actual replay GPU renderer proof mismatches assignment")
                write_json(output / "gpu_binding.json", dict(binding, **initial,
                    simulation=str(directory), child_pid=os.getpid(), replay_case_id=payload["case"]["id"]))
                return adapter
            except BaseException:
                adapter.close()
                raise
        from .loops import MaintenanceLoop
        loop = MaintenanceLoop(resolver, None, checked_factory, runtime, {},
            payload["supported_predicates"], code_timeout_s=payload["code_timeout_s"])
        loop.replay(payload["case"], payload["candidate"], output)
        return 0
    except Exception as error:
        from .replay_dispatch import _failure
        result = _failure(payload["case"]["id"], "child_initialization", error)
        write_json(output / "report.json", result)
        return 1


if __name__ == "__main__":
    sys.exit(main())
