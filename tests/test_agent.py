"""Contract, transport, evidence and secret-handling tests (no model billing)."""

import base64
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from evo.agent import (Agent, CommandProvider, HTTPProvider, ProviderError,
                       ResponseError, ScriptedProvider, parse_json,
                       validate_candidate, validate_plan)
from evo.prompts import ATOM_NAMES, PLAN_SCHEMA


def plan():
    return {"reasoning": "Close the observed door.", "steps": [{
        "id": "close", "atom": "A10", "params": {"target": "door", "fraction": 0.0},
        "goal": "Close the door", "postcondition": {"predicate": "native_success", "args": {}},
        "depends_on": []}]}


def candidate():
    return {"code": "def skill(api, params):\n    api.wait(params['ticks'])\n",
            "rationale": "Bounded control fixture; not a RoboCasa policy.", "dependencies": []}


def prefix_plan():
    result = plan()
    result["plan_complete"] = False
    result["reasoning"] = "Verify closing first; ground the remaining task from the next observation."
    result["steps"][0]["postcondition"] = {
        "predicate": "fixture_joint",
        "args": {"fixture": "door", "joint": "hinge", "min": 0.0, "max": 0.05}}
    return result


@contextmanager
def server(replies):
    requests = []
    queue = list(replies)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append({"path": self.path, "body": json.loads(raw),
                             "has_auth": self.headers.get("Authorization") == "Bearer test-key-private"})
            status, response = queue.pop(0)
            data = json.dumps(response).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/v1", requests
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.join(timeout=2)


def chat_envelope(response):
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(response)}}]}


def responses_envelope(response):
    return {"status": "completed", "output": [
        {"type": "reasoning", "summary": []},
        {"type": "message", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": json.dumps(response)}]}]}


class AgentTests(unittest.TestCase):
    def test_taxonomy_complete_and_empty_by_construction(self):
        self.assertEqual(set(ATOM_NAMES), {f"{prefix}{number:02d}" for prefix, count in (("P", 3), ("A", 26), ("C", 3)) for number in range(1, count + 1)})
        provider = ScriptedProvider({"plan": [plan()]})
        agent = Agent(provider)
        self.assertEqual(agent.plan({})["steps"][0]["atom"], "A10")
        self.assertIn("empty atom is a contract", provider.requests[0]["instructions"])

    def test_strict_json_rejects_fences_duplicates_nonfinite_and_trailing_text(self):
        for invalid in ('```json\n{}\n```', '{"x":1,"x":2}', '{"x":NaN}',
                        '{"x":Infinity}', '{} {}', '[]', ''):
            with self.subTest(invalid=invalid), self.assertRaises(ResponseError):
                parse_json(invalid)

    def test_plan_dependencies_and_host_verification(self):
        valid = plan()
        valid["steps"].insert(0, {
            "id": "approach", "atom": "A01", "params": {}, "goal": "Approach",
            "postcondition": {"predicate": "base_near", "args": {}}, "depends_on": []})
        valid["steps"][1]["depends_on"] = ["approach"]
        context = {"supported_predicates": ["base_near", "native_success"]}
        self.assertEqual(validate_plan(valid, context), valid)
        for mutate in (
            lambda p: p["steps"][0].update(atom="A27"),
            lambda p: p["steps"][0].update(depends_on=["close"]),
            lambda p: p["steps"][0].update(postcondition={"predicate": "native_success", "args": {}}),
            lambda p: p["steps"][0].update(postcondition={"predicate": "self_report", "args": {}}),
            lambda p: p["steps"][1].update(id="approach"),
        ):
            invalid = json.loads(json.dumps(valid))
            mutate(invalid)
            with self.assertRaises(ResponseError):
                validate_plan(invalid, context)

    def test_suffix_replan_preserves_completed_ids(self):
        suffix = plan()
        suffix["steps"][0]["depends_on"] = ["approach"]
        context = {"phase": "replan", "completed_steps": [{"id": "approach"}]}
        self.assertEqual(validate_plan(suffix, context), suffix)
        self.assertEqual(validate_plan(suffix, {"completed_steps": [{"step": {"id": "approach"}}]}), suffix)
        suffix["steps"][0]["id"] = "approach"
        with self.assertRaises(ResponseError):
            validate_plan(suffix, context)

    def test_verifiable_prefix_is_allowed_without_whole_task_postcondition(self):
        prefix = prefix_plan()
        provider = ScriptedProvider({"plan": [prefix]})
        context = {"instruction": "Close the door, then complete the remaining composite task.",
                   "supported_predicates": {"fixture_joint": "Host fixture joint interval"}}
        self.assertEqual(Agent(provider).plan(context), prefix)
        schema = provider.requests[0]["response_schema"]
        self.assertEqual(schema["properties"]["plan_complete"]["type"], "boolean")
        self.assertNotIn("plan_complete", schema["required"])
        self.assertIn("continue_after_verified_prefix", provider.requests[0]["instructions"])
        self.assertIn("uncertainty about later atoms must not block", provider.requests[0]["instructions"].lower())

    def test_plan_complete_preserves_legacy_and_explicit_complete_responses(self):
        legacy = plan()
        self.assertEqual(validate_plan(legacy), legacy)
        self.assertNotIn("plan_complete", validate_plan(legacy))
        complete = {**legacy, "plan_complete": True}
        self.assertEqual(validate_plan(complete), complete)
        # The host can find native success after a final local predicate too.
        local_last = {**prefix_plan(), "plan_complete": True}
        self.assertEqual(validate_plan(local_last), local_last)
        self.assertEqual(PLAN_SCHEMA["required"], ["steps", "reasoning"])

    def test_plan_complete_rejects_nonboolean_and_unknown_top_level_fields(self):
        for value in (None, 0, 1, "false", [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(ResponseError, "boolean"):
                validate_plan({**plan(), "plan_complete": value})
        with self.assertRaises(ResponseError):
            validate_plan({**prefix_plan(), "task_success": True})

    def test_unfinished_prefix_cannot_use_native_success_or_invent_predicates(self):
        with self.assertRaisesRegex(ResponseError, "unfinished prefix"):
            validate_plan({**plan(), "plan_complete": False})
        with self.assertRaisesRegex(ResponseError, "not supported"):
            validate_plan(prefix_plan(), {"supported_predicates": ["native_success"]})

    def test_continuation_uses_current_episode_prefix_and_refreshed_context(self):
        suffix = plan()
        suffix["steps"][0]["id"] = "close_next_door"
        suffix["steps"][0]["depends_on"] = ["open_first_door"]
        suffix["steps"][0]["params"]["pixel"] = [45, 88]
        context = {
            "phase": "continue_after_verified_prefix",
            "completed_steps": [{"step": {"id": "open_first_door"}, "verified": True}],
            "observation": {"frame_id": "current_after_prefix"},
            "library": {"A10": {"version": "v2"}},
            "episode_retry": {"previous_episodes": [{"completed_steps": [{"id": "old_episode_only"}]}]},
        }
        provider = ScriptedProvider({"plan": [suffix]})
        self.assertEqual(Agent(provider).plan(context), suffix)
        self.assertEqual(provider.requests[0]["context"], context)
        with self.assertRaisesRegex(ResponseError, "cannot repeat"):
            repeated = json.loads(json.dumps(suffix))
            repeated["steps"][0]["id"] = "open_first_door"
            validate_plan(repeated, context)
        with self.assertRaisesRegex(ResponseError, "dependencies"):
            old_dependency = json.loads(json.dumps(suffix))
            old_dependency["steps"][0]["depends_on"] = ["old_episode_only"]
            validate_plan(old_dependency, context)

    def test_empty_plan_remains_legal_when_no_next_atom_is_verifiable(self):
        blocked = {"steps": [], "reasoning": "No supported predicate can verify the next atom.",
                   "plan_complete": False}
        self.assertEqual(validate_plan(blocked, {"supported_predicates": []}), blocked)

    def test_candidate_contract_does_not_execute_code(self):
        self.assertEqual(validate_candidate(candidate()), candidate())
        for code in ("api.wait(10)", "def other(api, params): pass", "def skill(api, params=bad()): pass",
                     "@bad\ndef skill(api, params): pass", "def skill(api, params): pass\nx = 1",
                     "async def skill(api, params): pass", "def skill(api, params) -> bad(): pass"):
            invalid = {**candidate(), "code": code}
            with self.subTest(code=code), self.assertRaises(ResponseError):
                validate_candidate(invalid)

    def test_repair_and_distill_are_distinct_requests(self):
        provider = ScriptedProvider([
            {"stage": "repair", "response": candidate()},
            {"stage": "distill", "response": candidate()},
        ])
        agent = Agent(provider)
        agent.repair({"attempts": [{"error": "target not reached"}]})
        agent.distill({"attempts": [{"verified": True}]})
        self.assertEqual([request["stage"] for request in provider.requests], ["repair", "distill"])
        self.assertIn("target not reached", json.dumps(provider.requests[0]))
        with self.assertRaises(ProviderError):
            agent.repair({})

    def test_repair_can_return_complete_runtime_bindings(self):
        updated = {**candidate(), "params": {"camera": "camera_left", "pixel": [71, 82], "ticks": 2}}
        provider = ScriptedProvider({"repair": [updated]})
        context = {"step": {"params": {"camera": "camera_left", "roi": [10, 20, 90, 100]}}}
        result = Agent(provider).repair(context)
        self.assertEqual(result, updated)
        self.assertNotIn("pixel", context["step"]["params"])
        schema = provider.requests[0]["response_schema"]
        self.assertIn("params", schema["properties"])
        self.assertNotIn("params", schema["required"])

    def test_repair_bindings_must_be_finite_json_object(self):
        for invalid in (None, [], "choose target", {"pixel": [float("nan"), 2]}, {"other": object()}):
            with self.subTest(params=type(invalid).__name__), self.assertRaises(ResponseError):
                validate_candidate({**candidate(), "params": invalid}, allow_params=True)

    def test_distillation_cannot_change_regression_bindings(self):
        updated = {**candidate(), "params": {"ticks": 999}}
        provider = ScriptedProvider({"distill": [updated]})
        with self.assertRaises(ResponseError):
            Agent(provider, max_attempts=1).distill({})
        self.assertNotIn("params", provider.requests[0]["response_schema"]["properties"])
        self.assertEqual(validate_candidate(candidate()), candidate())

    def test_distillation_may_rebind_only_host_authorized_pending_case(self):
        updated = {**candidate(), "params": {"ticks": 2}}
        provider = ScriptedProvider({"distill": [updated]})
        context = {"allow_pending_case_rebind": True, "pending_case": {"step": {"params": {"ticks": 1}}}}
        self.assertEqual(Agent(provider, max_attempts=1).distill(context), updated)
        self.assertEqual(context["pending_case"]["step"]["params"], {"ticks": 1})
        self.assertIn("params", provider.requests[0]["response_schema"]["properties"])

    def test_contract_retry_and_artifact_credential_redaction(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-private"}):
            provider = ScriptedProvider({"plan": [{"bad": True}, plan()]})
            result = Agent(provider, log_dir=temporary).plan({"instruction": "close", "api_key": "secret",
                                                          "note": "test-key-private", "feedback": {"x": 2}})
            self.assertEqual(result, plan())
            self.assertIn("contract_feedback", provider.requests[1])
            logs = list(Path(temporary).rglob("*.json"))
            self.assertEqual(len(logs), 7)
            saved = "\n".join(path.read_text() for path in logs)
            self.assertNotIn("test-key-private", saved)
            self.assertNotIn('"secret"', saved)
            self.assertIn('"feedback"', saved)

    def test_contract_retry_is_bounded(self):
        provider = ScriptedProvider({"plan": [{"bad": True}, {"bad": True}, plan()]})
        with self.assertRaises(ResponseError):
            Agent(provider, max_attempts=2).plan({})
        self.assertEqual(len(provider.requests), 2)

    def test_command_json_exchange(self):
        wrapper = ("import json,sys; request=json.load(sys.stdin); "
                   "assert request['stage']=='plan'; print(" + repr(json.dumps(plan())) + ")")
        agent = Agent(CommandProvider([sys.executable, "-c", wrapper]), max_attempts=1)
        self.assertEqual(agent.plan({}), plan())

    def test_command_failure_and_timeout_are_sanitized(self):
        provider = CommandProvider([sys.executable, "-c", "import sys; print('a secret',file=sys.stderr);sys.exit(7)"])
        with self.assertRaisesRegex(ProviderError, "exit status 7") as failure:
            provider.generate({})
        self.assertNotIn("a secret", str(failure.exception))
        provider = CommandProvider([sys.executable, "-c", "import time;time.sleep(10)"], timeout_s=0.05)
        with self.assertRaisesRegex(ProviderError, "timeout"):
            provider.generate({})

    def test_http_chat_vision_and_retry(self):
        png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jQ1kAAAAASUVORK5CYII=")
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"EVO_TEST_API_KEY": "test-key-private"}):
            image_path = Path(temporary) / "rgb.png"
            image_path.write_bytes(png)
            with server([(429, {"error": "try later"}), (200, chat_envelope(plan()))]) as (url, requests):
                provider = HTTPProvider("fixture-model", base_url=url, api_key_env="EVO_TEST_API_KEY", max_retries=1)
                result = Agent(provider).plan({"observation": {"images": [{"path": str(image_path), "camera": "agentview"}]}})
                self.assertEqual(result, plan())
                self.assertEqual(len(requests), 2)
                self.assertEqual(requests[0]["path"], "/v1/chat/completions")
                self.assertTrue(requests[0]["has_auth"])
                body = requests[-1]["body"]
                self.assertEqual(body["response_format"], {"type": "json_object"})
                self.assertTrue(body["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_http_responses_and_reasoning(self):
        with server([(200, responses_envelope(plan()))]) as (url, requests):
            provider = HTTPProvider("fixture-model", base_url=url, api="responses", api_key_env="",
                                    reasoning_effort="high", max_tokens=1024)
            self.assertEqual(Agent(provider).plan({}), plan())
            body = requests[0]["body"]
            self.assertEqual(requests[0]["path"], "/v1/responses")
            self.assertEqual(body["reasoning"], {"effort": "high"})
            self.assertEqual(body["text"], {"format": {"type": "json_object"}})
            self.assertEqual(body["max_output_tokens"], 1024)
            self.assertIn("JSON", body["input"][0]["content"][0]["text"])
            self.assertFalse(body["store"])

    def test_http_rejects_incomplete_output_and_tools(self):
        provider = HTTPProvider("fixture-model", api="responses", api_key_env="")
        invalid = responses_envelope(plan())
        invalid["status"] = "incomplete"
        with self.assertRaises(ResponseError):
            provider._content(invalid)
        invalid = responses_envelope(plan())
        invalid["output"].append({"type": "function_call"})
        with self.assertRaises(ResponseError):
            provider._content(invalid)

    def test_http_auth_failure_is_not_retried_or_logged_raw(self):
        with server([(401, {"error": "echo-test-key-private"})]) as (url, requests):
            provider = HTTPProvider("fixture-model", base_url=url, api_key_env="", max_retries=2)
            with self.assertRaisesRegex(ProviderError, "status 401") as failure:
                Agent(provider).plan({})
            self.assertEqual(len(requests), 1)
            self.assertNotIn("echo-test-key-private", str(failure.exception))

    def test_usage_ledger_keeps_only_numbers_and_counts_http_attempts(self):
        reply = responses_envelope(plan())
        reply["usage"] = {"input_tokens": 100, "output_tokens": 25, "total_tokens": 125,
                          "output_tokens_details": {"reasoning_tokens": 12, "reasoning_text": "do not retain"},
                          "metadata": "do not retain", "flag": True, "invalid_count": -1}
        with tempfile.TemporaryDirectory() as temporary:
            with server([(429, {"error": "busy"}), (200, reply)]) as (url, _requests):
                provider = HTTPProvider("fixture-model", api="responses", base_url=url, api_key_env="", max_retries=1)
                Agent(provider, log_dir=temporary).plan({})
                expected = {"usage": {"input_tokens": 100, "output_tokens": 25, "total_tokens": 125,
                                      "output_tokens_details": {"reasoning_tokens": 12}}, "http_attempts": 2}
                self.assertEqual(provider.last_usage, expected)
            metadata = json.loads(next(Path(temporary).rglob("call_1.json")).read_text())
            self.assertEqual(metadata["provider_usage"], expected)
            self.assertGreater(metadata["provider_elapsed_s"], 0)
            self.assertNotIn("reasoning_text", json.dumps(metadata))
            self.assertNotIn("do not retain", json.dumps(metadata))

    def test_usage_is_null_for_scripted_and_cleared_before_http_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            Agent(ScriptedProvider({"plan": [plan()]}), log_dir=temporary).plan({})
            metadata = json.loads(next(Path(temporary).rglob("call_1.json")).read_text())
            self.assertIsNone(metadata["provider_usage"])
            self.assertGreaterEqual(metadata["provider_elapsed_s"], 0)
        provider = HTTPProvider("fixture-model", api_key_env="EVO_TEST_MISSING_CREDENTIAL")
        provider.last_usage = {"usage": {"total_tokens": 999}, "http_attempts": 1}
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ProviderError):
            provider.generate({})
        self.assertEqual(provider.last_usage, {"usage": None, "http_attempts": 0})


if __name__ == "__main__":
    unittest.main()
