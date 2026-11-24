"""Model-neutral planning, single-atom repair and distillation adapters.

Integration::

    agent = Agent(HTTPProvider(model="your-vision-model"), log_dir=run_dir / "agent")
    plan = agent.plan(context)
    candidate = agent.repair({**context, "step": step, "attempts": attempts})
    reusable = agent.distill({**context, "step": step, "attempts": verified_attempts})

``context`` is a JSON object supplied by the runner. It normally includes
instruction, initial_observation, observation, api_catalog, supported_predicates,
published_skills, budgets and attempt feedback. Plans may return an optional
plan_complete boolean: false marks a verifiable prefix needing continuation;
true is a planning expectation, never a host success verdict. Legacy responses
omitting it remain valid. For suffix replanning set phase="replan"; after a
verified prefix set phase="continue_after_verified_prefix". Both use current
observation, refreshed library and completed_steps=[{"id": ...}, ...] from this
episode only, whose IDs remain available as dependencies. Each observation may have
images=[{"path": "/absolute/camera.png", "camera": "agentview", "modality": "rgb"}].
Depth arrays remain public sensor artifacts for api.deproject; attach a PNG depth
preview to let a vision model inspect depth visually.

All providers implement generate(request: dict) -> dict. CommandProvider sends
one UTF-8 JSON object on stdin and expects one JSON object on stdout; wrappers may
use the supplied instructions, response_schema and attached_images. No shell is
invoked. HTTPProvider uses the broadly compatible /chat/completions endpoint and
reads credentials only from an environment variable. ScriptedProvider consumes
explicit fixture responses and cannot autonomously solve a simulation task.

These adapters never execute generated robot code and do not decide success.
The bounded executor, independent verifier, and library regression gate belong
to the host. Provider errors are sanitized; keys/headers/raw backend errors are
not written to artifacts.
"""

from __future__ import annotations

import ast
import base64
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import tempfile
import time
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid

from .prompts import (ATOM_NAMES, CANDIDATE_SCHEMA, DISTILL_REBIND_SCHEMA, PLAN_SCHEMA,
                      REPAIR_SCHEMA, CONSOLIDATE_SCHEMA, FAILURE_CONSOLIDATE_SCHEMA, instructions)


class AgentError(RuntimeError):
    """Base class for bounded, sanitized model-adapter failures."""


class ProviderError(AgentError):
    """Transport, credentials, timeout, or fixture exhaustion failure."""


class ResponseError(AgentError):
    """The model did not meet the JSON or stage contract."""


class Provider(Protocol):
    def generate(self, request: dict[str, Any]) -> dict[str, Any]: ...


def _reject_constant(_value):
    raise ValueError("nonfinite numbers are not JSON")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def parse_json(value: str | bytes) -> dict:
    """Parse one finite JSON object; reject fences, duplicates and trailing text."""
    try:
        result = json.loads(value, parse_constant=_reject_constant, object_pairs_hook=_pairs)
    except (ValueError, UnicodeError, TypeError, RecursionError):
        raise ResponseError("provider response is not a strict JSON object") from None
    if not isinstance(result, dict):
        raise ResponseError("provider response must be a JSON object")
    return result


def _copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False),
                      parse_constant=_reject_constant)


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be finite and positive")


def _count(value, name, minimum=1, maximum=100):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")


def _strings(value):
    return (isinstance(value, list) and all(isinstance(item, str) and item.strip()
                                          for item in value)
            and len(set(value)) == len(value))


def _completed_ids(context):
    steps = context.get("completed_steps", context.get("completed", []))
    if not isinstance(steps, list):
        raise ValueError("completed_steps must be a list")
    ids = [(step.get("id", step.get("step", {}).get("id")) if isinstance(step, dict) else step)
           for step in steps]
    if not _strings(ids):
        raise ValueError("completed_steps must contain unique nonempty IDs")
    return set(ids)


def validate_plan(response: dict, context: dict | None = None) -> dict:
    """Validate a full plan or verifiable prefix without deciding task success.

    Optional ``plan_complete`` is preserved, not defaulted for legacy responses.
    Explicitly unfinished prefixes cannot use the whole-task native scorer as
    the postcondition of a locally completed atom.
    """
    context = {} if context is None else context
    required_fields = {"steps", "reasoning"}
    if (not required_fields <= set(response)
            or not set(response) <= required_fields | {"plan_complete"}
            or not isinstance(response["reasoning"], str)):
        raise ResponseError("plan requires steps and reasoning, with optional plan_complete")
    if "plan_complete" in response and type(response["plan_complete"]) is not bool:
        raise ResponseError("plan_complete must be a boolean when provided")
    steps = response["steps"]
    if not isinstance(steps, list) or len(steps) > 100:
        raise ResponseError("plan steps must be a list of at most 100 steps")
    if context.get("plan_one_step") and len(steps) > 1:
        raise ResponseError("This run requests exactly one next atom; plan remaining work after its verification")
    seen = _completed_ids(context)
    supported = context.get("supported_predicates")
    if isinstance(supported, dict):
        supported = set(supported)
    elif isinstance(supported, list):
        supported = {item["name"] if isinstance(item, dict) else item for item in supported}
    elif supported is not None:
        raise ValueError("supported_predicates must be a list or dictionary")
    required = {"id", "atom", "params", "goal", "postcondition", "depends_on"}
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or set(step) != required:
            raise ResponseError("every plan step must contain exactly the required fields")
        if (not isinstance(step["id"], str) or not step["id"].strip()
                or len(step["id"]) > 128 or step["id"] in seen):
            raise ResponseError("step IDs must be nonempty and cannot repeat a completed step")
        if not isinstance(step["atom"], str) or step["atom"] not in ATOM_NAMES:
            raise ResponseError("step atom must be one of P01..P03, A01..A26 or C01..C03")
        if not isinstance(step["params"], dict) or not isinstance(step["goal"], str) or not step["goal"].strip():
            raise ResponseError("step params must be an object and goal must be nonempty")
        if not _strings(step["depends_on"]) or not set(step["depends_on"]) <= seen:
            raise ResponseError("step dependencies must reference earlier or completed steps")
        condition = step["postcondition"]
        if (not isinstance(condition, dict) or set(condition) != {"predicate", "args"}
                or not isinstance(condition["predicate"], str)
                or not condition["predicate"].strip() or not isinstance(condition["args"], dict)):
            raise ResponseError("postcondition requires predicate and args")
        predicate = condition["predicate"]
        if supported is not None and predicate not in supported:
            raise ResponseError("postcondition is not supported by the host verifier")
        if predicate == "native_success" and index != len(steps) - 1:
            raise ResponseError("native_success can verify only the final step")
        if predicate == "native_success" and response.get("plan_complete") is False:
            raise ResponseError("an unfinished prefix cannot use native_success as local verification")
        seen.add(step["id"])
    try:
        return _copy(response)
    except (TypeError, ValueError, RecursionError):
        raise ResponseError("plan must be finite JSON data") from None


def validate_candidate(response: dict, *, allow_params=False, require_description=False) -> dict:
    """Check shape/signature and host-authorized complete runtime bindings.

    The host still audits the code body. ``allow_params`` must only be enabled
    for repair or a new pending case, never immutable established cases.
    """
    required = {"code", "rationale", "dependencies"}
    if require_description:
        required.add("description")
    allowed = required | {"description"} | ({"params"} if allow_params else set())
    if "description" in response and (not isinstance(response["description"], str) or not response["description"].strip()):
        raise ResponseError("description must document the function with a nonempty string")
    if not required <= set(response) or not set(response) <= allowed:
        raise ResponseError("candidate requires code, rationale, dependencies and a description when requested; optional params require host authorization")
    if "params" in response and not isinstance(response["params"], dict):
        raise ResponseError("repair params must be a complete JSON object")
    if (not isinstance(response["code"], str) or not response["code"].strip()
            or len(response["code"].encode("utf-8")) > 200_000
            or not isinstance(response["rationale"], str) or not _strings(response["dependencies"])):
        raise ResponseError("candidate has invalid code, rationale or dependencies")
    try:
        module = ast.parse(response["code"])
    except (SyntaxError, ValueError, RecursionError):
        raise ResponseError("candidate code is not valid Python") from None
    if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
        raise ResponseError("candidate must contain only def skill(api, params)")
    function = module.body[0]
    args = function.args
    if (function.name != "skill" or function.decorator_list or function.returns
            or args.posonlyargs or args.vararg or args.kwarg or args.kwonlyargs or args.defaults
            or [arg.arg for arg in args.args] != ["api", "params"]
            or any(arg.annotation is not None for arg in args.args)):
        raise ResponseError("candidate signature must be exactly def skill(api, params)")
    try:
        return _copy(response)
    except (TypeError, ValueError, RecursionError):
        raise ResponseError("candidate and repair params must be finite JSON data") from None


def _attached_images(context: dict, max_images: int) -> list[dict]:
    images, seen = [], set()
    # The newest evidence goes first. History stays intact in textual context,
    # but old image paths are not silently presented as current observations.
    groups = [context]
    groups.extend(context[key] for key in ("observation", "current_observation", "initial_observation")
                  if isinstance(context.get(key), dict))
    for group in groups:
        records = group.get("images", [])
        if not isinstance(records, list):
            raise ValueError("observation images must be a list")
        for raw in records:
            record = {"path": raw} if isinstance(raw, str) else raw
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                raise ValueError("image references require a path string")
            path = Path(record["path"]).expanduser().resolve()
            if str(path) in seen:
                continue
            if len(images) >= max_images:
                raise ValueError("too many attached images; select camera views explicitly")
            if not path.is_file() or not 0 < path.stat().st_size <= 20_000_000:
                raise ValueError("image path is missing, empty, or exceeds 20 MB")
            data = path.read_bytes()
            if data.startswith(b"\x89PNG\r\n\x1a\n"):
                mime = "image/png"
            elif data.startswith(b"\xff\xd8\xff"):
                mime = "image/jpeg"
            elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
                mime = "image/webp"
            else:
                raise ValueError("attached images must be PNG, JPEG, or WebP; use depth previews")
            digest = hashlib.sha256(data).hexdigest()
            if record.get("sha256", digest) != digest:
                raise ValueError("attached image changed since observation")
            images.append({**record, "path": str(path), "sha256": digest,
                           "mime_type": mime, "image_index": len(images)})
            seen.add(str(path))
    return images


def _redact(value):
    """Keep supplied context/response, but remove accidental credential fields."""
    secret_fields = {"api_key", "authorization", "access_token", "refresh_token", "password", "secret"}
    secrets = [value for name, value in os.environ.items() if len(value) >= 8
               and re.search(r"(?:^|_)(?:API_KEY|TOKEN|SECRET|PASSWORD)$", name)]
    def visit(item):
        if isinstance(item, dict):
            return {key: "[REDACTED]" if key.lower() in secret_fields else visit(child)
                    for key, child in item.items()}
        if isinstance(item, list):
            return [visit(child) for child in item]
        if isinstance(item, str):
            for secret in secrets:
                item = item.replace(secret, "[REDACTED]")
        return item
    return visit(value)


def _write(path: Path, value):
    path.write_text(json.dumps(_redact(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _numeric_usage(value):
    """Retain token accounting numbers, never provider reasoning/text metadata."""
    if type(value) in (int, float) and math.isfinite(value) and value >= 0:
        return value
    if isinstance(value, dict):
        kept = {key: _numeric_usage(child) for key, child in value.items()
                if isinstance(key, str)}
        return {key: child for key, child in kept.items() if child is not None} or None
    return None


def validate_consolidation(response, context):
    required = {"code", "rationale", "dependencies", "params", "decision", "progress"}
    if context.get("source_success") is not True:
        if response.get("decision") != "draft":
            raise ResponseError("Unsuccessful experience can only produce a draft")
        required.add("failure_analysis")
    if context.get("require_skill_description") or "description" in response:
        required.add("description")
    if set(response) != required or response.get("decision") not in {"keep", "update", "draft"}:
        raise ResponseError("consolidate requires " + ", ".join(sorted(required)))
    validate_candidate({k: response[k] for k in ("code", "rationale", "dependencies", "params", "description") if k in response},
                       allow_params=True, require_description=context.get("require_skill_description", False))
    progress = response["progress"]
    if (not isinstance(progress, dict) or set(progress) != {"completed", "remaining", "entry_requirements"}
            or any(not isinstance(v, list) or any(not isinstance(s, str) or not s.strip() for s in v)
                   for v in progress.values())):
        raise ResponseError("progress requires completed/remaining/entry_requirements string lists")
    if "failure_analysis" in required:
        analysis = response["failure_analysis"]
        if (not isinstance(analysis, dict) or set(analysis) != {"summary", "hypotheses", "next_changes"}
                or not isinstance(analysis["summary"], str) or not analysis["summary"].strip()
                or any(not isinstance(analysis[k], list)
                       or any(not isinstance(s, str) or not s.strip() for s in analysis[k])
                       for k in ("hypotheses", "next_changes"))):
            raise ResponseError("failure_analysis requires a nonempty summary and hypotheses/next_changes string lists")
    if response["decision"] == "update" and not context.get("active"):
        raise ResponseError("An empty capability must pass first-use replay")
    return _copy(response)


class Agent:
    """Stage contracts plus optional complete request/response run artifacts.

    max_attempts bounds contract-repair requests, independently of the transport's
    finite retry count. Rejected JSON contracts are supplied as feedback for the
    next request. Execution failures belong in runner-provided attempt history.
    """

    def __init__(self, provider: Provider, log_dir=None, *, max_attempts=2, max_images=12,
                 max_request_bytes=32_000_000):
        _count(max_attempts, "max_attempts", maximum=5)
        _count(max_images, "max_images", maximum=32)
        _count(max_request_bytes, "max_request_bytes", maximum=256_000_000)
        self.provider, self.max_attempts, self.max_images = provider, max_attempts, max_images
        self.max_request_bytes = max_request_bytes
        self.log_dir = None if log_dir is None else Path(log_dir)

    def plan(self, context: dict) -> dict:
        return self._request("plan", context)

    def repair(self, context: dict) -> dict:
        return self._request("repair", context)

    def distill(self, context: dict) -> dict:
        return self._request("distill", context)

    def consolidate(self, context: dict) -> dict:
        return self._request("consolidate", context)

    def _request(self, stage: str, context: dict) -> dict:
        if not isinstance(context, dict):
            raise ValueError("agent context must be a JSON object")
        context = _copy(context)
        allow_params = stage == "repair" or (stage == "distill" and
            context.get("allow_pending_case_rebind") is True and isinstance(context.get("pending_case"), dict))
        schema = {"plan": PLAN_SCHEMA, "repair": REPAIR_SCHEMA,
                  "distill": DISTILL_REBIND_SCHEMA if allow_params else CANDIDATE_SCHEMA,
                  "consolidate": (CONSOLIDATE_SCHEMA if context.get("source_success") is True
                                  else FAILURE_CONSOLIDATE_SCHEMA)}[stage]
        if stage != "plan" and context.get("require_skill_description"):
            schema = _copy(schema)
            schema["required"].append("description")
        request = {"stage": stage, "instructions": instructions(stage), "context": context,
                   "response_schema": schema,
                   "attached_images": _attached_images(context, self.max_images)}
        if context.get("plan_one_step"):
            request["instructions"] += "\nPlan only ONE next atom from the instruction and current observation; do not return a multi-step prefix."
        if context.get("record_working_memory"):
            request["instructions"] += ("\nworking_memory includes parameterized failed functions, raw attempts and their failure feedback, not current physical progress. "
                "Repair the previous parameterized function using failure_feedback, failure_analysis and current observations. "
                "Bind its parameters to the new scene and restore prerequisites; never copy old scene bindings blindly. "
                "horizon_exhausted means the action budget ended before this atom succeeded, not proof the whole approach is invalid. "
                "Successful and failed implementations are parameterized at the episode boundary. "
                "Failed drafts and model cause hypotheses are unverified repair input, never published api.call targets.")
        directory = None
        if self.log_dir is not None:
            directory = self.log_dir / (stage + "_" + uuid.uuid4().hex[:16])
            directory.mkdir(parents=True)
        last_error = None
        for attempt in range(1, self.max_attempts + 1):
            if len(json.dumps(request, ensure_ascii=False).encode("utf-8")) > self.max_request_bytes:
                raise ValueError("complete agent context exceeds the request limit; no history was dropped")
            started = time.monotonic()
            if directory:
                _write(directory / f"request_{attempt}.json", request)
            metadata = {"stage": stage, "attempt": attempt, "provider": type(self.provider).__name__,
                        "provider_usage": None, "provider_elapsed_s": None}
            provider_started = None
            provider_elapsed = None
            try:
                provider_started = time.monotonic()
                response = self.provider.generate(_copy(request))
                provider_elapsed = time.monotonic() - provider_started
                if not isinstance(response, dict):
                    raise ResponseError("provider must return a JSON object")
                if directory:
                    _write(directory / f"response_{attempt}.json", response)
                result = (validate_plan(response, context) if stage == "plan" else
                          validate_consolidation(response, context) if stage == "consolidate" else
                          validate_candidate(response, allow_params=allow_params,
                                             require_description=context.get("require_skill_description", False)))
                metadata["status"] = "completed"
                if directory:
                    _write(directory / "response.json", result)
                return result
            except ResponseError as error:
                last_error = error
                metadata.update(status="invalid_response", error=str(error))
                request["contract_feedback"] = {"error": str(error), "attempt": attempt}
            except ProviderError:
                metadata.update(status="provider_error")
                raise
            finally:
                if provider_started is not None:
                    if provider_elapsed is None:
                        provider_elapsed = time.monotonic() - provider_started
                    metadata["provider_elapsed_s"] = round(provider_elapsed, 6)
                    metadata["provider_usage"] = _copy(getattr(self.provider, "last_usage", None))
                metadata["elapsed_s"] = round(time.monotonic() - started, 6)
                if directory:
                    _write(directory / f"call_{attempt}.json", metadata)
        raise ResponseError(f"{stage} failed its response contract after {self.max_attempts} attempts: {last_error}")


class ScriptedProvider:
    """Consume deterministic fixtures, explicitly separate from autonomous runs.

    Accept {stage: [response, ...]} or an ordered list of responses. An ordered
    item may be {"stage": "plan", "response": {...}} to assert request ordering.
    Fixtures are consumed once; exhaustion is an error, never a fallback skill.
    ``requests`` contains defensive copies for testing the two-loop protocol.
    """

    def __init__(self, responses):
        fixtures = _copy(responses)
        if isinstance(fixtures, dict):
            self._stages = {stage: deque(value if isinstance(value, list) else [value])
                            for stage, value in fixtures.items()}
            self._ordered = None
        elif isinstance(fixtures, list):
            self._stages, self._ordered = {}, deque(fixtures)
        else:
            raise ValueError("fixtures must be a stage mapping or response list")
        self.requests = []

    def generate(self, request: dict) -> dict:
        self.requests.append(_copy(request))
        queue = self._ordered if self._ordered is not None else self._stages.get(request["stage"])
        if not queue:
            raise ProviderError("scripted fixture exhausted for stage " + request["stage"])
        result = queue.popleft()
        if isinstance(result, dict) and set(result) == {"stage", "response"}:
            if result["stage"] != request["stage"]:
                raise ProviderError("scripted fixture stage does not match request")
            result = result["response"]
        return _copy(result)


class CommandProvider:
    """Run a trusted model wrapper with JSON stdin/stdout and a finite timeout.

    The wrapper inherits the caller's environment for authentication, but neither
    its environment, command arguments, stderr, nor malformed raw output is
    copied into run logs. Supply credentials via environment, never arguments.
    """

    def __init__(self, command: list[str] | str, *, timeout_s=120, max_response_bytes=2_000_000):
        _positive(timeout_s, "timeout_s")
        _count(max_response_bytes, "max_response_bytes", maximum=64_000_000)
        self.command = shlex.split(command) if isinstance(command, str) else list(command)
        if not self.command or any(not isinstance(part, str) or not part for part in self.command):
            raise ValueError("command must contain nonempty string arguments")
        self.timeout_s, self.max_response_bytes = timeout_s, max_response_bytes

    def generate(self, request: dict) -> dict:
        payload = json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf-8")
        process = None
        with tempfile.TemporaryFile() as output:
            try:
                process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=output,
                                           stderr=subprocess.DEVNULL, start_new_session=True)
                process.communicate(payload, timeout=self.timeout_s)
                if process.returncode:
                    raise ProviderError(f"command provider failed with exit status {process.returncode}; diagnostics omitted")
                output.seek(0)
                raw = output.read(self.max_response_bytes + 1)
                if len(raw) > self.max_response_bytes:
                    raise ResponseError("command response exceeds the configured size limit")
                return parse_json(raw)
            except subprocess.TimeoutExpired:
                raise ProviderError("command provider exceeded its timeout") from None
            except OSError:
                raise ProviderError("command provider could not start or access its output") from None
            finally:
                if process is not None and process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.communicate()


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward an authorization header to a redirect target.


class HTTPProvider:
    """OpenAI-compatible JSON + vision HTTP transport (stdlib only).

    base_url defaults to OPENAI_BASE_URL or https://api.openai.com/v1. model
    defaults to EVO_MODEL or OPENAI_MODEL; no implicit model substitution occurs.
    api="responses" selects /responses with input_image and reasoning.effort;
    the default api="chat_completions" selects /chat/completions instead.
    Set api_key_env to an empty string for an explicitly unauthenticated local
    service. JSON mode is enabled by default, and can be disabled for compatible
    servers that lack it; the exact same strict local contract still applies.
    408/409/429/5xx and transient network failures have bounded retries. Other
    HTTP errors, malformed JSON, and refusals are explicit failures. No API key,
    authorization header or raw backend error is retained in artifacts.
    """

    def __init__(self, model=None, *, base_url=None, api_key_env="OPENAI_API_KEY", api="chat_completions",
                 timeout_s=120, max_retries=2, max_response_bytes=2_000_000,
                 max_request_bytes=64_000_000, json_mode=True, max_tokens=None, reasoning_effort=None):
        self.model = model or os.environ.get("EVO_MODEL") or os.environ.get("OPENAI_MODEL")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("set model explicitly or supply EVO_MODEL/OPENAI_MODEL")
        endpoint = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        url = urlsplit(endpoint)
        if (url.scheme not in {"https", "http"} or not url.hostname or url.username
                or url.password or url.query or url.fragment):
            raise ValueError("base_url must be an HTTP(S) endpoint without embedded credentials")
        if api not in {"chat_completions", "responses"}:
            raise ValueError("api must be chat_completions or responses")
        self.api = api
        suffix = "/responses" if api == "responses" else "/chat/completions"
        self.endpoint = endpoint if endpoint.endswith(suffix) else endpoint + suffix
        if not isinstance(api_key_env, str) or (api_key_env and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env)):
            raise ValueError("api_key_env must be an environment variable name or empty")
        _positive(timeout_s, "timeout_s")
        _count(max_retries, "max_retries", minimum=0, maximum=5)
        _count(max_response_bytes, "max_response_bytes", maximum=64_000_000)
        _count(max_request_bytes, "max_request_bytes", maximum=256_000_000)
        if max_tokens is not None:
            _count(max_tokens, "max_tokens", maximum=1_000_000)
        if reasoning_effort is not None and reasoning_effort not in {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}:
            raise ValueError("unsupported reasoning_effort")
        self.api_key_env, self.timeout_s, self.max_retries = api_key_env, timeout_s, max_retries
        self.max_response_bytes, self.max_request_bytes = max_response_bytes, max_request_bytes
        self.json_mode, self.max_tokens = bool(json_mode), max_tokens
        self.reasoning_effort = reasoning_effort
        self._opener = build_opener(_NoRedirect())
        self.last_usage = None

    def _body(self, request):
        text_request = {key: value for key, value in request.items() if key != "instructions"}
        # Some Responses-compatible gateways check JSON mode against input
        # messages only, rather than also considering top-level instructions.
        parts = [{"type": "text", "text": "Return one JSON object only.\n" +
                  json.dumps(text_request, ensure_ascii=False, allow_nan=False)}]
        for item in request.get("attached_images", []):
            data = Path(item["path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ProviderError("attached observation image changed before model request")
            parts.append({"type": "image_url", "image_url": {
                "url": "data:" + item["mime_type"] + ";base64," + base64.b64encode(data).decode("ascii")}})
        if self.api == "responses":
            response_parts = [({"type": "input_text", "text": part["text"]} if part["type"] == "text"
                               else {"type": "input_image", "image_url": part["image_url"]["url"]})
                              for part in parts]
            body = {"model": self.model, "instructions": request["instructions"],
                    "input": [{"role": "user", "content": response_parts}],
                    "stream": False, "store": False}
            if self.json_mode:
                body["text"] = {"format": {"type": "json_object"}}
            if self.max_tokens is not None:
                body["max_output_tokens"] = self.max_tokens
            if self.reasoning_effort is not None:
                body["reasoning"] = {"effort": self.reasoning_effort}
            return body
        body = {"model": self.model, "messages": [
            {"role": "system", "content": request["instructions"]},
            {"role": "user", "content": parts}], "stream": False}
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        return body

    def _content(self, envelope):
        if self.api == "responses":
            if envelope.get("status") not in (None, "completed") or envelope.get("error"):
                raise ResponseError("model response failed or did not complete")
            output = envelope.get("output")
            if not isinstance(output, list):
                raise ResponseError("HTTP response lacks Responses output items")
            texts = []
            for item in output:
                if not isinstance(item, dict):
                    raise ResponseError("invalid Responses output item")
                if item.get("type") == "reasoning":
                    continue
                if item.get("type") != "message" or item.get("role") != "assistant":
                    raise ResponseError("model invoked a tool or returned an unsupported output item")
                if item.get("status") not in (None, "completed") or not isinstance(item.get("content"), list):
                    raise ResponseError("model output message is incomplete")
                for part in item["content"]:
                    if not isinstance(part, dict) or part.get("type") != "output_text" or not isinstance(part.get("text"), str):
                        raise ResponseError("model refused or returned a nontext response")
                    texts.append(part["text"])
            return "".join(texts)
        try:
            choice = envelope["choices"][0]
            message = choice["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError):
            raise ResponseError("HTTP response lacks a chat completion message") from None
        if (message.get("refusal") or message.get("tool_calls")
                or choice.get("finish_reason") not in (None, "stop")):
            raise ResponseError("model refused, invoked a tool, or did not finish its JSON response")
        if not isinstance(content, str):
            raise ResponseError("chat completion content must be a JSON string")
        return content

    def generate(self, request: dict) -> dict:
        self.last_usage = {"usage": None, "http_attempts": 0}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise ProviderError("HTTP provider credential environment variable is unset")
            headers["Authorization"] = "Bearer " + key
        try:
            payload = json.dumps(self._body(request), ensure_ascii=False, allow_nan=False).encode("utf-8")
        except OSError:
            raise ProviderError("HTTP provider cannot read an attached observation image") from None
        if len(payload) > self.max_request_bytes:
            raise ProviderError("HTTP request including images exceeds the configured size limit")
        for attempt in range(self.max_retries + 1):
            self.last_usage["http_attempts"] = attempt + 1
            try:
                with self._opener.open(Request(self.endpoint, payload, headers, method="POST"),
                                       timeout=self.timeout_s) as response:
                    raw = response.read(self.max_response_bytes + 1)
                if len(raw) > self.max_response_bytes:
                    raise ResponseError("HTTP response exceeds the configured size limit")
                envelope = parse_json(raw)
                self.last_usage["usage"] = _numeric_usage(envelope.get("usage"))
                return parse_json(self._content(envelope))
            except HTTPError as error:
                status = error.code
                error.close()
                if attempt == self.max_retries or status not in {408, 409, 429} and status < 500:
                    raise ProviderError(f"HTTP provider failed with status {status}; backend details omitted") from None
            except (URLError, TimeoutError, ConnectionError, OSError):
                if attempt == self.max_retries:
                    raise ProviderError("HTTP provider failed after bounded connection/timeout retries") from None
            time.sleep(min(0.25 * 2 ** attempt, 2.0))
        raise ProviderError("HTTP provider exhausted its retry budget")


# Descriptive alias for callers that prefer the full transport name.
OpenAICompatibleProvider = HTTPProvider
