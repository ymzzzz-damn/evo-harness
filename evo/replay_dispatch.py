"""Bounded process-only replay scheduling; no simulator is used by its threads."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from .io import read_json, write_json, sha256


def import_ref(reference):
    if not isinstance(reference, str) or reference.count(":") != 1:
        raise ValueError("An importable module:qualname reference is required")
    module, name = reference.split(":")
    if not module or module == "__main__" or not name or any(not p.isidentifier() for p in name.split(".")):
        raise ValueError("Reference must identify an importable top-level callable")
    value = importlib.import_module(module)
    for part in name.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise ValueError("Referenced object is not callable")
    return value


def callable_ref(value):
    module, name = getattr(value, "__module__", None), getattr(value, "__qualname__", None)
    if not module or module == "__main__" or not name or "<" in name:
        raise ValueError("Parallel replay needs an explicit factory_ref/runtime_ref for local callables")
    return module + ":" + name


class FrozenVersionResolver:
    """Resolve only parent-supplied exact immutable pins, never open library state."""
    def __init__(self, records):
        self.records = {}
        for record in records:
            if type(record.get("version")) is not int or record["version"] < 1:
                raise ValueError("Frozen dependency version must be a positive integer")
            key = (record["atom"], record["version"])
            if key in self.records or sha256(record["code"]) != record["sha256"]:
                raise ValueError("Duplicate or corrupt frozen dependency source")
            self.records[key] = copy.deepcopy(record)
        def visit(key, path):
            if key in path:
                raise ValueError("Cyclic frozen dependency graph")
            record = self.records[key]
            for atom, pin in record.get("dependencies", {}).items():
                child = (atom, pin["version"])
                if child not in self.records or self.records[child]["sha256"] != pin["sha256"]:
                    raise ValueError("Missing or mismatched frozen dependency pin")
                visit(child, path + (key,))
        for key in self.records:
            visit(key, ())

    def get_version(self, atom, version, sha256=None):
        record = self.records.get((atom, version))
        if record is None or sha256 is not None and record["sha256"] != sha256:
            raise ValueError("Exact dependency version is not in this replay's frozen resolver")
        return copy.deepcopy(record)


def freeze_dependencies(case, candidate, library):
    """Materialize only programs the unchanged replay loop can actually execute."""
    records = {}
    def visit(atom, pin, path=()):
        if not isinstance(pin, dict) or set(pin) != {"version", "sha256"} or type(pin["version"]) is not int:
            raise ValueError("Replay dependencies require exact version and SHA")
        key = (atom, pin["version"])
        if key in path:
            raise ValueError("Cyclic replay dependency graph")
        if key in records:
            if records[key]["sha256"] != pin["sha256"]:
                raise ValueError("Conflicting SHA for the same dependency version")
            return
        record = library.get_version(atom, pin["version"])
        if record["sha256"] != pin["sha256"] or sha256(record["code"]) != pin["sha256"]:
            raise ValueError("Replay dependency source SHA mismatch")
        records[key] = {k:copy.deepcopy(record[k]) for k in ("atom", "version", "sha256", "code", "dependencies")}
        for child, lock in record["dependencies"].items():
            visit(child, lock, path + (key,))
    sources = [candidate]
    trace = case.get("entry_trace")
    for prior in trace if trace is not None else case.get("prefix", []):
        if not prior.get("skipped"):
            sources.append(candidate if trace is None and prior["step"]["atom"] == candidate["atom"] else prior)
    for following in case.get("suffix", []):
        if not following.get("skipped"):
            sources.append(candidate if following["step"]["atom"] == candidate["atom"] else following)
    for source in sources:
        if source.get("sha256") is not None and sha256(source["code"]) != source["sha256"]:
            raise ValueError("Replay program SHA mismatch")
        for atom, pin in source.get("dependencies", {}).items():
            visit(atom, pin)
    frozen = [records[key] for key in sorted(records)]
    FrozenVersionResolver(frozen)
    return frozen


def _resources(resources, library):
    resources = copy.deepcopy(resources)
    root = Path(resources.get("token_root", "")).expanduser()
    if not root.is_absolute():
        raise ValueError("replay_resources.token_root must be an explicit absolute own-run directory")
    root = root.resolve()
    slots = resources.get("slots_per_gpu")
    if type(slots) is not int or not 1 <= slots <= 2:
        raise ValueError("slots_per_gpu must be an integer in 1..2")
    if bool(resources.get("gpus")) == bool(resources.get("inventory_path")):
        raise ValueError("Supply exactly one explicit GPU inventory_path or fixed gpus list")
    inventory = None
    if resources.get("inventory_path"):
        path = Path(resources["inventory_path"]).expanduser()
        if not path.is_absolute() or path.resolve().parent != root.parent:
            raise ValueError("GPU inventory and replay token directory must belong to the same explicit run")
        raw = path.read_bytes()
        inventory = {"path":str(path.resolve()), "sha256":hashlib.sha256(raw).hexdigest()}
        data = json.loads(raw)
        gpus = data.get("devices")
        if data.get("gpu_requested") != 8 or not isinstance(gpus, list) or len(gpus) != 8:
            raise ValueError("Own-run supervisor inventory must declare eight GPU devices")
    else:
        gpus = resources["gpus"]
    if not isinstance(gpus, list) or not 1 <= len(gpus) <= 8:
        raise ValueError("Explicit GPU inventory must contain 1..8 GPU devices")
    if any(not isinstance(g, dict) or not isinstance(g.get("uuid"), str)
           or re.fullmatch(r"GPU-[A-Za-z0-9-]+", g["uuid"]) is None for g in gpus):
        raise ValueError("Every replay GPU must identify its NVIDIA UUID")
    if len({g["uuid"] for g in gpus}) != len(gpus):
        raise ValueError("Replay GPU UUIDs must be unique")
    if inventory and any(not g.get("name") for g in gpus):
        raise ValueError("Supervisor GPU inventory lacks GPU model names")
    gpus = sorted(gpus, key=lambda g:g["uuid"])
    declaration = {"protocol_version":1, "gpus":gpus, "slots_per_gpu":slots,
                   "inventory":inventory, "library_root":str(Path(library.root).resolve())}
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".declaration.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        path = root / "declaration.json"
        if path.exists():
            if read_json(path) != declaration:
                raise ValueError("Replay token declaration differs from another worker's resource inventory")
        else:
            write_json(path, declaration)
    return root, gpus, slots


@contextmanager
def replay_slot(root, gpus, slots, offset=0):
    """The fd is inherited by the child, so parent death cannot free a live slot."""
    # Prefer one case on every GPU before the second slot on any GPU.
    tokens = [(gpu, slot) for slot in range(slots) for gpu in gpus]
    tokens = tokens[offset % len(tokens):] + tokens[:offset % len(tokens)]
    started = time.monotonic()
    while True:
        for gpu, slot in tokens:
            path = root / (gpu["uuid"] + "_" + str(slot) + ".lock")
            handle = path.open("a+")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            try:
                yield handle, dict(gpu, replay_slot=slot, cuda_logical_device=0), time.monotonic()-started, path
            finally:
                # Close, don't LOCK_UN: a surviving child retains the inherited
                # open-file-description lock until it exits and closes its fd.
                handle.close()
            return
        time.sleep(.05)


def _failure(case_id, stage, error):
    return {"case_id":case_id, "success":False, "native_success":False, "prefix":[],
            "failure_stage":stage, "error":{"type":type(error).__name__, "message":str(error)}}


def validate_replay_configuration(resources, library, adapter_factory, runtime_api):
    """Reject bad scheduling config before any exploration/model call is made."""
    if not isinstance(resources, dict):
        raise ValueError("Parallel replay resources must be a dictionary")
    allowed = {"token_root", "slots_per_gpu", "inventory_path", "gpus", "factory_ref", "runtime_ref",
               "renderer_mapping_ref", "renderer_evidence_ref", "fresh_scene_ref"}
    if set(resources) - allowed:
        raise ValueError("Unknown replay resource options: " + str(sorted(set(resources) - allowed)))
    normalized = copy.deepcopy(resources)
    defaults = {"factory_ref":lambda:callable_ref(adapter_factory),
                "runtime_ref":lambda:callable_ref(runtime_api),
                "renderer_mapping_ref":lambda:"scripts.parallel_worker:install_renderer_mapping",
                "renderer_evidence_ref":lambda:"scripts.parallel_worker:renderer_evidence",
                "fresh_scene_ref":lambda:"scripts.parallel_worker:verify_fresh_scene"}
    for key, default in defaults.items():
        normalized[key] = normalized.get(key) or default()
        import_ref(normalized[key])  # Resolve callable only; never execute a factory.
    _resources(normalized, library)
    return normalized


def dispatch_replays(cases, candidate, output, *, library, adapter_factory, runtime_api,
                     supported_predicates, code_timeout_s, parallelism, resources):
    if type(parallelism) is not int or not 2 <= parallelism <= 16:
        raise ValueError("Parallel replay dispatch requires parallelism in 2..16")
    refs = {"factory_ref":resources.get("factory_ref") or callable_ref(adapter_factory),
            "runtime_ref":resources.get("runtime_ref") or callable_ref(runtime_api),
            "renderer_mapping_ref":resources.get("renderer_mapping_ref", "scripts.parallel_worker:install_renderer_mapping"),
            "renderer_evidence_ref":resources.get("renderer_evidence_ref", "scripts.parallel_worker:renderer_evidence"),
            "fresh_scene_ref":resources.get("fresh_scene_ref", "scripts.parallel_worker:verify_fresh_scene")}
    output = Path(output)
    identifiers = [c["id"] for c in cases]
    if len(set(identifiers)) != len(identifiers) or any(not isinstance(i, str) or i in {".", ".."}
            or re.fullmatch(r"[A-Za-z0-9_.-]+", i) is None for i in identifiers):
        raise ValueError("Replay case IDs must be unique safe directory names")
    if any((output / identifier).exists() for identifier in identifiers):
        raise ValueError("Refuse to duplicate or overwrite an existing replay case directory")
    root, gpus, slots = _resources(resources, library)
    output.mkdir(parents=True, exist_ok=True)
    claimed = []
    try:
        for identifier in identifiers:
            directory = output / identifier
            directory.mkdir(exist_ok=False)
            claimed.append(directory)
    except BaseException:
        for directory in claimed:
            directory.rmdir()  # Only our still-empty directories; no child started.
        raise
    module_root = Path(__file__).resolve().parent.parent
    source_sha256 = {name:hashlib.sha256((module_root / "evo" / name).read_bytes()).hexdigest()
                     for name in ("loops.py", "replay_dispatch.py", "replay_worker.py")}
    def run_one(index, case):
        directory = output / case["id"]
        dispatch = {"case_id":case["id"], "queued_at_utc":datetime.now(timezone.utc).isoformat(),
                    "parent_pid":os.getpid(), "source_sha256":source_sha256, "attempts":1}
        started = time.monotonic()
        try:
            records = freeze_dependencies(case, candidate, library)
            with replay_slot(root, gpus, slots, index) as (handle, assignment, waited, slot_path):
                dispatch.update(queue_wait_s=waited, assignment=assignment, token_path=str(slot_path))
                # Active records contain full historic certificates. Replay uses
                # only this program/pin contract, not those large old reports.
                program = {key:candidate[key] for key in ("atom", "sha256", "code", "dependencies")}
                if "version" in candidate:
                    program["version"] = candidate["version"]
                payload = {"case":case, "candidate":program, "dependency_records":records,
                    "library_root_audit_only":str(library.root), "refs":refs,
                    "code_timeout_s":code_timeout_s, "supported_predicates":supported_predicates,
                    "assignment":assignment, "output":str(directory.resolve()), "source_sha256":source_sha256}
                input_path = directory / "dispatch_input.json"
                write_json(input_path, payload)
                env = dict(os.environ)
                for key in ("ROBOENV_GPU", "MUJOCO_EGL_DEVICE_ID"):
                    env.pop(key, None)
                scratch = directory / "scratch"
                for key, suffix in (("TMPDIR", "tmp"), ("ROBOCASA_TEMP_DIR", "robocasa"), ("NUMBA_CACHE_DIR", "numba")):
                    path = scratch / suffix
                    path.mkdir(parents=True, exist_ok=True)
                    env[key] = str(path.resolve())
                env.update(CUDA_VISIBLE_DEVICES=assignment["uuid"], MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
                    EVO_RUN=str(directory.resolve()), EVO_GPU_ASSIGNMENT=str((directory / "gpu_assignment.json").resolve()))
                paths = [str(module_root)] + [str(Path(p).resolve()) for p in sys.path if p and Path(p).is_dir()]
                env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths + env.get("PYTHONPATH", "").split(os.pathsep)))
                write_json(directory / "gpu_assignment.json", assignment)
                command = [sys.executable, "-m", "evo.replay_worker", "--input", str(input_path.resolve()),
                           "--input-sha256", hashlib.sha256(input_path.read_bytes()).hexdigest(),
                           "--token-fd", str(handle.fileno()), "--parent-pid", str(os.getpid())]
                child_started = time.monotonic()
                with (directory / "child_stdout.log").open("w") as stdout, (directory / "child_stderr.log").open("w") as stderr:
                    process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr,
                        pass_fds=(handle.fileno(),))
                    try:
                        dispatch.update(child_pid=process.pid, started_at_utc=datetime.now(timezone.utc).isoformat())
                        write_json(directory / "dispatch.json", dict(dispatch, status="running"))
                        returncode = process.wait()  # No new case wall-clock deadline.
                    except BaseException:
                        process.terminate()
                        try:
                            process.wait(timeout=5)  # Cancellation cleanup only.
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        raise
                dispatch.update(returncode=returncode, child_wall_s=time.monotonic()-child_started)
                result = read_json(directory / "report.json")
                if returncode != 0:
                    if result.get("case_id") == case["id"] and result.get("success") is False:
                        result["dispatch_error"] = {"type":"ChildProcessError", "returncode":returncode}
                    else:
                        raise RuntimeError("Replay child exited with code " + str(returncode))
                if result.get("case_id") != case["id"] or type(result.get("success")) is not bool:
                    raise ValueError("Replay child returned a mismatched or malformed report")
                if result["success"] and (result.get("execution", {}).get("sha256") != candidate["sha256"]
                        or not (directory / "gpu_binding.json").is_file()):
                    raise ValueError("Successful replay lacks exact target SHA or actual GPU binding proof")
                dispatch["status"] = "complete" if returncode == 0 else "error"
        except Exception as error:
            result = _failure(case["id"], "dispatch", error)
            dispatch.update(status="error", error=result["error"])
            if not (directory / "report.json").exists():
                write_json(directory / "report.json", result)
        dispatch.update(finished_at_utc=datetime.now(timezone.utc).isoformat(), wall_s=time.monotonic()-started)
        write_json(directory / "dispatch.json", dispatch)
        write_json(directory / "dispatch_result.json", result)
        return result
    batch_started = time.monotonic()
    # Threads only wait for resource tokens and subprocesses; no GL/control runs
    # in them. Executor.map yields original case order and never fails fast.
    with ThreadPoolExecutor(max_workers=min(parallelism, len(cases) or 1)) as pool:
        results = list(pool.map(lambda item:run_one(*item), enumerate(cases)))
    write_json(output / "replay_dispatch_summary.json", {"case_ids":identifiers,
        "parallelism":parallelism, "slots_per_gpu":slots, "gpu_uuids":[g["uuid"] for g in gpus],
        "wall_s":time.monotonic()-batch_started, "results_in_original_order":True,
        "all_cases_joined":len(results)==len(cases), "successful_cases":sum(r["success"] for r in results)})
    return results
