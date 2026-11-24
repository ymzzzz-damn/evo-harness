"""Optional, same-session reuse of successful *formal-case* replay evidence.

The simulation/runtime/assets must remain immutable within a session. This is an
optimization of repeatability checks, not evidence of reliability across trials.
Like Library validation reports, certificates are produced by trusted host code;
hashes detect corruption and mismatched contracts, not malicious host workers.
"""
from datetime import datetime, timezone
import copy
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import re
import sys

from .io import read_json, write_json
from .replay_dispatch import freeze_dependencies


POLICY = "same_session_success_v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _callable_identity(value):
    target = value if inspect.isfunction(value) or inspect.isclass(value) else type(value)
    path = inspect.getsourcefile(target)
    if not path:
        raise ValueError("Replay caching requires inspectable adapter and runtime sources")
    return {"name": target.__module__ + ":" + target.__qualname__,
            "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def context(session, factory, runtime_api, predicates, timeout, parallelism, resources):
    """Recheck host sources/settings on every batch; never reuse across restarts.

    Serial GPU workers keep separate evidence by GPU UUID. Parallel dispatch is
    scoped to its declared GPU pool, so changing that pool also invalidates it.
    External packages/assets are fixed for the session (no live runtime edits).
    """
    packages = {}
    for name in ("numpy", "scipy", "mujoco", "robosuite", "robocasa", "roboenv"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"policy": POLICY, "session": session,
            "framework": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path(__file__).parent.glob("*.py"))},
            "factory": _callable_identity(factory), "runtime": _callable_identity(runtime_api),
            "python": sys.version, "platform": platform.platform(), "packages": packages,
            "environment": {key: os.environ.get(key) for key in (
                "EVO_RUNTIME", "MUJOCO_GL", "PYOPENGL_PLATFORM", "CUDA_VISIBLE_DEVICES",
                "NVIDIA_VISIBLE_DEVICES", "MUJOCO_EGL_DEVICE_ID", "ROBOENV_GPU",
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "predicates": predicates, "code_timeout_s": timeout,
            "parallelism": parallelism, "resources": resources}


def contract(case, candidate, resolver, context_sha256):
    case_content = {k: v for k, v in case.items() if k not in ("id", "created_at")}
    # This includes prefix/suffix programs, failed entry preparations, target
    # bindings, predicates, task/seed/horizon and the complete dependency closure.
    closure = freeze_dependencies(case, candidate, resolver)
    return {"policy": POLICY, "context_sha256": context_sha256,
            "case_id": case["id"], "case_sha256": digest(case_content),
            "program": {"atom": candidate["atom"], "sha256": candidate["sha256"],
                        "dependencies": candidate.get("dependencies", {})},
            "closure_sha256": digest(closure)}


def _object_path(root, identifier):
    if not isinstance(identifier, str) or re.fullmatch(r"[0-9a-f]{64}", identifier) is None:
        raise ValueError("Invalid replay certificate digest")
    return Path(root) / "replay_evidence" / "objects" / (identifier + ".json")


def _successful(result):
    return (isinstance(result, dict) and result.get("success") is True
            and isinstance(result.get("execution"), dict) and result["execution"].get("ok") is True
            and isinstance(result.get("verification"), dict)
            and result["verification"].get("known", True) is True
            and result["verification"].get("success") is True
            and not result.get("error") and not result.get("close_error"))


def load_certificate(root, reference, expected):
    certificate = read_json(_object_path(root, reference["certificate_sha256"]))
    if (not isinstance(certificate, dict) or digest(certificate) != reference["certificate_sha256"]
            or certificate.get("contract") != expected
            or reference.get("contract_sha256") != digest(expected)
            or not _successful(certificate.get("result", {}))
            or certificate["result"].get("case_id") != expected["case_id"]
            or "replay_evidence" in certificate["result"]):
        raise ValueError("Replay certificate does not match successful fresh evidence")
    return certificate


class ReplayEvidence:
    def __init__(self, root, execution_context):
        self.root = Path(root)
        self.context = execution_context
        self.context_sha256 = digest(execution_context)

    def lookup(self, expected):
        try:
            reference = read_json(self.root / "replay_evidence" / "index" / (digest(expected) + ".json"))
            certificate = load_certificate(self.root, reference, expected)
            result = copy.deepcopy(certificate["result"])
            result["replay_evidence"] = {**reference, "kind": "cached_success",
                "original_report": certificate["report_path"], "verified_at": certificate["verified_at"]}
            return result
        except (OSError, ValueError, KeyError, TypeError):
            # Missing/corrupt evidence is a cache miss, never successful coverage.
            return None

    def remember(self, expected, result, report_path):
        if not _successful(result) or "replay_evidence" in result:
            return
        certificate = {"contract": expected, "result": result,
                       "report_path": str(Path(report_path).resolve()),
                       "verified_at": datetime.now(timezone.utc).isoformat()}
        certificate_sha256 = digest(certificate)
        # Immutable objects keep old validation references valid if a later fresh
        # replay updates the lookup index for the same execution contract.
        write_json(_object_path(self.root, certificate_sha256), certificate)
        write_json(self.root / "replay_evidence" / "index" / (digest(expected) + ".json"),
                   {"certificate_sha256": certificate_sha256, "contract_sha256": digest(expected)})


def validate_cached_result(root, state, registered, candidate, report, result):
    """Called inside Library's transaction: resolve from its state, never relock."""
    if registered["id"] not in state["cases"]:
        raise ValueError("Pending cases require a new fresh replay, not cached evidence")
    if report.get("replay_policy") != POLICY:
        raise ValueError("Cached evidence requires an explicit replay policy")
    execution_context = report.get("replay_context", {})
    if execution_context.get("policy") != POLICY or not execution_context.get("session"):
        raise ValueError("Cached evidence lacks its execution context")

    class StateResolver:
        def get_version(self, atom, version):
            record = state["versions"][atom][str(version)]
            code = (Path(root) / "sources" / (record["sha256"] + ".py")).read_text()
            return {**record, "code": code}

    code = (Path(root) / "sources" / (candidate["sha256"] + ".py")).read_text()
    expected = contract(registered, {**candidate, "code": code}, StateResolver(), digest(execution_context))
    certificate = load_certificate(root, result["replay_evidence"], expected)
    if {k: v for k, v in result.items() if k != "replay_evidence"} != certificate["result"]:
        raise ValueError("Cached replay result differs from its original certificate")
    reference = result["replay_evidence"]
    if (reference.get("original_report") != certificate["report_path"]
            or reference.get("verified_at") != certificate["verified_at"]):
        raise ValueError("Cached replay provenance differs from its certificate")
