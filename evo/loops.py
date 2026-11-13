"""Top-down exploration and a separate replay-gated maintenance loop."""
from pathlib import Path
import copy
import time

from .catalog import CATALOG
from .execution import execute_skill
from .io import read_json, write_json, sha256


def _predicate(step):
    value = step.get("postcondition", {})
    if not isinstance(value, dict) or not isinstance(value.get("predicate"), str):
        raise ValueError("Every step requires a host postcondition {predicate,args}")
    return value


def validate_plan(plan, predicates, completed_ids=()):
    steps = plan.get("steps")
    if not isinstance(steps, list) or not 1 <= len(steps) <= 40:
        raise ValueError("A plan requires 1..40 ordered atomic steps")
    seen = set(completed_ids)
    for index, step in enumerate(steps):
        if step.get("atom") not in CATALOG or not isinstance(step.get("params"), dict):
            raise ValueError("Unknown atom or invalid params")
        if not isinstance(step.get("id"), str) or step["id"] in seen:
            raise ValueError("Step IDs must be distinct strings")
        if not set(step.get("depends_on", [])).issubset(seen):
            raise ValueError("Dependencies must reference preceding steps")
        predicate = _predicate(step)["predicate"]
        if predicate not in predicates:
            raise ValueError("Unsupported host predicate: " + predicate)
        if predicate == "native_success" and index != len(steps)-1:
            raise ValueError("native_success is only a final-step predicate")
        seen.add(step["id"])
    return plan


def _active_context(library):
    return [dict(atom=r.get("atom"), version=r["version"], sha256=r["sha256"],
                 code=r["code"], dependencies=r.get("dependencies", {}))
            for r in library.list_active()]


def _previous_episode_source(retry_context, atom):
    """Historical code is repair input only; a reset invalidates physical progress."""
    for episode in reversed((retry_context or {}).get("previous_episodes", [])):
        for records in (episode.get("recent_failures", []), episode.get("completed_steps", [])):
            for record in reversed(records):
                if not record.get("skipped") and record.get("step", {}).get("atom") == atom and record.get("code"):
                    return dict(code=record["code"], dependencies=record.get("dependencies", {}),
                                params=record["step"].get("params", {}),
                                from_previous_episode=True,
                                failed_in_previous_episode=not record.get("success", False),
                                source_run=episode.get("source_run", episode.get("source_directory")),
                                requires_fresh_grounding=True)
    return None


class SkillExecutor:
    def __init__(self, adapter, library, runtime_api, timeout_s):
        self.adapter, self.library = adapter, library
        self.runtime_api, self.timeout_s = runtime_api, timeout_s

    def run(self, code, params, dependencies=None):
        started = time.monotonic()
        stack = []
        def api_for(locks):
            def call(atom, child_params):
                if atom not in locks:
                    raise ValueError("Skill dependency not pinned: " + atom)
                if atom in stack or len(stack) >= 8:
                    raise RuntimeError("Recursive/cyclic skill dependency")
                lock = locks[atom]
                record = self.library.get_version(atom, lock["version"])
                if record["sha256"] != lock["sha256"]:
                    raise ValueError("Dependency hash mismatch")
                stack.append(atom)
                try:
                    result = execute_skill(record["code"], api_for(record.get("dependencies", {})), child_params,
                                           max(0.001, self.timeout_s-(time.monotonic()-started)))
                    if not result["ok"]:
                        raise RuntimeError("Dependency failed: " + str(result["error"]))
                    return result.get("result")
                finally:
                    stack.pop()
            return self.runtime_api(self.adapter, call_skill=call)
        before = self.adapter.stats()
        result = execute_skill(code, api_for(dependencies or {}), params, self.timeout_s)
        result["before"] = before
        result["after"] = self.adapter.stats()
        result["observation"] = self.adapter.observe()
        return result


class ExplorationLoop:
    def __init__(self, library, agent, adapter_factory, runtime_api, api_description,
                 supported_predicates, max_attempts=4, max_replans=1, code_timeout_s=120):
        self.library, self.agent, self.factory = library, agent, adapter_factory
        self.runtime_api, self.api_description = runtime_api, api_description
        self.predicates = supported_predicates
        self.max_attempts, self.max_replans, self.timeout = max_attempts, max_replans, code_timeout_s

    def run(self, task_config, output, retry_context=None, on_atomic_success=None):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "task.json", task_config)
        if retry_context is not None:
            write_json(output / "retry_context.json", retry_context)
        adapter = None
        report = dict(status="initializing", task_config=task_config, completed_steps=[],
                      failures=[], native_success=False, pending=[], replans=0,
                      execution_trace=[], maintenance_errors=[])
        try:
            adapter = self.factory(task_config, output / "simulation")
            initial = adapter.observe()
            write_json(output / "initial.json", initial)
            if adapter.native_success():
                report["status"] = "already_successful_at_reset"
                return report
            context = dict(instruction=initial.get("instruction", ""), initial_observation=initial,
                           observation=initial, atomic_catalog=CATALOG, api_catalog=self.api_description,
                           supported_predicates=self.predicates, library=_active_context(self.library))
            if retry_context is not None:
                context["episode_retry"] = copy.deepcopy(retry_context)
            plan = self.agent.plan(context)
            write_json(output / "plan_0.json", plan)
            if plan.get("steps") == []:
                # An empty model plan is protocol-valid but supplies no action
                # evidence. Preserve planning feedback for the next reset episode
                # without attributing a failure to an atom that never executed.
                report.update(status="plan_complete_task_failed",
                    stop_reason={"kind": "planner_returned_empty_initial_plan", "summary": plan.get("reasoning", "")},
                    native_success=bool(adapter.native_success()), stats=adapter.stats(),
                    final_observation=adapter.observe())
                return report
            plan = validate_plan(plan, self.predicates)
            executor = SkillExecutor(adapter, self.library, self.runtime_api, self.timeout)
            pending_steps = list(plan["steps"])
            attempt_history = []
            while pending_steps:
                step = pending_steps.pop(0)
                atom = step["atom"]
                entry_observation = adapter.observe()
                entry_trace = copy.deepcopy(report["execution_trace"])
                step_attempts = []
                before_check = adapter.evaluate(_predicate(step))
                if before_check.get("known", True) and before_check["success"]:
                    # An already true condition is not evidence that a new skill works.
                    noop = "def skill(api, params):\n    return {'already_satisfied': True, 'bindings': params}\n"
                    report["completed_steps"].append(dict(step=step, skipped=True, code=noop,
                        sha256=sha256(noop), dependencies={}, success=True, verification=before_check))
                    report["execution_trace"].append(copy.deepcopy(report["completed_steps"][-1]))
                    continue
                active = self.library.active(atom)
                historical = _previous_episode_source(retry_context, atom)
                failed_before_reset = bool(historical and historical["failed_in_previous_episode"])
                previous = historical if failed_before_reset else (active or historical)
                step_success = False
                for attempt in range(self.max_attempts):
                    # An active policy may itself have consumed the last horizon.
                    # On reset ask for a revision first instead of looping that
                    # same failing policy forever without a repair request.
                    reused = attempt == 0 and active is not None and not failed_before_reset
                    source = active if reused else None
                    binding_observation = None
                    params_before = copy.deepcopy(step["params"])
                    executed = False
                    try:
                        if not reused:
                            binding_observation = adapter.observe()
                            response = self.agent.repair(dict(context, phase="fill" if previous is None else "repair",
                                step=step, observation=binding_observation, completed_steps=report["completed_steps"],
                                previous=previous, failures=report["failures"][-8:],
                                lessons=self.library.lessons(atom)[-8:], library=_active_context(self.library)))
                            if "params" in response:
                                if not isinstance(response["params"], dict):
                                    raise ValueError("Updated bindings must be a complete params dictionary")
                                step = dict(step, params=response["params"])
                            source = dict(code=response["code"], dependencies={})
                            source = self.library.propose(atom, response["code"],
                                metadata={"stage": "exploration", "task_config": task_config,
                                          "rationale": response.get("rationale", "")},
                                dependencies=response.get("dependencies", []))
                        executed = True
                        result = executor.run(source["code"], step["params"], source.get("dependencies", {}))
                        check = adapter.evaluate(_predicate(step))
                    except (ValueError, SyntaxError, TypeError) as error:
                        source = source or dict(code="", dependencies={})
                        result = dict(ok=False, error={"type":type(error).__name__, "message":str(error)},
                                      observation=adapter.observe())
                        check = dict(success=False, known=False, reason="Code/plan contract rejected before verification")
                    success = result["ok"] and check.get("known", True) and check["success"]
                    record = dict(step=step, attempt=attempt, reused=reused, code=source["code"],
                                  sha256=sha256(source["code"]), dependencies=source.get("dependencies", {}),
                                  execution=result, verification=check, success=bool(success), executed=executed)
                    record["binding_update"] = dict(changed=step["params"] != params_before,
                        previous_params=params_before, observation=binding_observation)
                    attempt_history.append(record)
                    step_attempts.append(record)
                    if executed:
                        report["execution_trace"].append(copy.deepcopy(record))
                    attempt_dir = output / "attempts" / f"plan{report['replans']:02d}_step{len(report['completed_steps']):02d}_try{attempt:02d}"
                    write_json(attempt_dir / "record.json", record)
                    (attempt_dir / "skill.py").write_text(source["code"])
                    if success:
                        prefix = copy.deepcopy(report["completed_steps"])
                        report["completed_steps"].append(record)
                        experience = dict(atom=atom, task_config=task_config, prefix=prefix,
                                          step=step, successful_record=record,
                                          step_index=len(prefix), attempt_history=copy.deepcopy(step_attempts),
                                          entry_trace=entry_trace, entry_observation=entry_observation,
                                          entry_step=copy.deepcopy(step_attempts[0]["step"]),
                                          suffix=[], require_native_success=False, validation_scope="atomic",
                                          source_run=str(output.resolve()), initial_observation=initial)
                        if retry_context is not None:
                            experience["episode_retry"] = copy.deepcopy(retry_context)
                        path = output / "pending" / f"{len(report['completed_steps']):02d}_{atom}.json"
                        write_json(path, experience)
                        report["pending"].append(str(path.resolve()))
                        # Persist evidence before maintenance; a later task/provider
                        # failure must not lose this independently verified atom.
                        write_json(output / "progress.json", report)
                        if on_atomic_success is not None:
                            try:
                                on_atomic_success(path.resolve())
                            except Exception as error:
                                report["maintenance_errors"].append(dict(experience=str(path.resolve()),
                                    type=type(error).__name__, message=str(error)))
                                write_json(output / "progress.json", report)
                        step_success = True
                        break
                    failure = dict(atom=atom, step_id=step["id"], attempt=attempt,
                                   error=result.get("error"), verification=check, sha256=record["sha256"],
                                   execution={k:v for k,v in result.items() if k != "observation"},
                                   observation=result["observation"], source_run=str(output.resolve()))
                    report["failures"].append(failure)
                    self.library.add_lesson(atom, failure)
                    previous = source
                    write_json(output / "progress.json", report)
                    if adapter.stats().get("terminal"):
                        break
                if not step_success:
                    if report["replans"] < self.max_replans and not adapter.stats().get("terminal"):
                        report["replans"] += 1
                        remaining = self.agent.plan(dict(context, phase="replan_remaining_suffix",
                            observation=adapter.observe(), completed_steps=report["completed_steps"],
                            failures=report["failures"][-8:], failed_step=step, remaining_steps=pending_steps,
                            library=_active_context(self.library)))
                        if not remaining.get("steps"):
                            # The model may abandon the exhausted/unsalvageable
                            # episode. Let the outer loop reset, not classify this
                            # valid empty suffix as a backend or transport error.
                            write_json(output / f"plan_{report['replans']}.json", remaining)
                            report["status"] = "plan_complete_task_failed"
                            report["stop_reason"] = "planner_returned_empty_suffix"
                            break
                        remaining = validate_plan(remaining, self.predicates,
                            completed_ids=[r["step"]["id"] for r in report["completed_steps"]])
                        pending_steps = list(remaining["steps"])
                        write_json(output / f"plan_{report['replans']}.json", remaining)
                        continue
                    report["status"] = "step_failed"
                    break
            else:
                report["status"] = "success" if adapter.native_success() else "plan_complete_task_failed"
            report["native_success"] = bool(adapter.native_success())
            report["stats"] = adapter.stats()
            report["final_observation"] = adapter.observe()
            # Whole-task continuations are separate evidence. They must not mutate
            # an already consumed atomic experience into a task-success obligation.
            for path in report["pending"]:
                experience = read_json(path)
                if report["native_success"]:
                    continuation = dict(experience,
                        suffix=report["completed_steps"][experience["step_index"]+1:],
                        require_native_success=True, validation_scope="task_continuation")
                    write_json(output / "continuations" / Path(path).name, continuation)
        except Exception as error:
            import traceback
            report.update(status="error", error={"type":type(error).__name__, "message":str(error),
                                                  "traceback":traceback.format_exc()})
        finally:
            if adapter is not None:
                try:
                    if "stats" not in report:
                        report["stats"] = adapter.stats()
                except Exception as error:
                    report["stats_error"] = {"type":type(error).__name__, "message":str(error)}
                try:
                    adapter.close()
                except Exception as error:
                    report["status"] = "error"
                    report["close_error"] = {"type":type(error).__name__, "message":str(error)}
            write_json(output / "summary.json", report)
        return report


class MaintenanceLoop:
    """No source is published before independent fresh-environment replay succeeds."""
    def __init__(self, library, agent, adapter_factory, runtime_api, api_description,
                 supported_predicates, max_candidates=2, code_timeout_s=120):
        self.library, self.agent, self.factory = library, agent, adapter_factory
        self.runtime_api, self.api_description = runtime_api, api_description
        self.predicates, self.max_candidates, self.timeout = supported_predicates, max_candidates, code_timeout_s

    def replay(self, case, candidate, output):
        adapter = None
        result = dict(case_id=case["id"], success=False, native_success=False, prefix=[], failure_stage="initialization")
        try:
            adapter = self.factory(case["task_config"], Path(output) / "simulation")
            if adapter.native_success():
                result["error"] = "Task already successful at reset"
                return result
            executor = SkillExecutor(adapter, self.library, self.runtime_api, self.timeout)
            result["failure_stage"] = "prefix"
            # Every occurrence of this atomic capability uses the candidate, including
            # earlier calls in the same task. Other atoms retain their recorded source.
            trace = case.get("entry_trace")
            for prior in trace if trace is not None else case["prefix"]:
                if prior.get("skipped"):
                    check = adapter.evaluate(_predicate(prior["step"]))
                    result["prefix"].append(dict(skipped=True, verification=check))
                    if not check.get("known", True) or not check["success"]:
                        result["error"] = "Previously satisfied prefix no longer holds"
                        return result
                    continue
                # New cases record the actual entry trajectory, including failed
                # preparations. Replay those exact pinned programs to establish
                # the entry; only the target atom is the program under test.
                source = candidate if trace is None and prior["step"]["atom"] == candidate["atom"] else prior
                execution = executor.run(source["code"], prior["step"]["params"], source.get("dependencies", {}))
                check = adapter.evaluate(_predicate(prior["step"]))
                result["prefix"].append(dict(execution=execution, verification=check))
                if trace is not None and not prior["success"]:
                    expected = prior["execution"]
                    expected_check = prior["verification"]
                    if (execution["ok"] != expected["ok"]
                            or (not execution["ok"] and execution["error"]["type"] != expected["error"]["type"])
                            or check.get("known", True) != expected_check.get("known", True)
                            or check["success"] != expected_check["success"]):
                        result["error"] = "Entry preparation diverged from recorded attempt"
                        return result
                    continue
                if not execution["ok"] or not check.get("known", True) or not check["success"]:
                    result["error"] = "Prefix replay failed"
                    return result
            step = case["step"]
            result["entry_observation"] = adapter.observe()
            result["failure_stage"] = "target"
            before_check = adapter.evaluate(_predicate(step))
            if before_check.get("known", True) and before_check["success"]:
                result["error"] = "Target condition already true before candidate; no skill evidence"
                return result
            result["execution"] = executor.run(candidate["code"], step["params"], candidate.get("dependencies", {}))
            result["verification"] = adapter.evaluate(_predicate(step))
            result["success"] = bool(result["execution"]["ok"] and result["verification"].get("known", True)
                                     and result["verification"]["success"])
            result["atomic_success"] = result["success"]
            result["suffix"] = []
            if result["success"]:
                result["failure_stage"] = "suffix"
                for following in case.get("suffix", []):
                    if following.get("skipped"):
                        check = adapter.evaluate(_predicate(following["step"]))
                        execution = dict(ok=True, skipped=True)
                    else:
                        source = candidate if following["step"]["atom"] == candidate["atom"] else following
                        execution = executor.run(source["code"], following["step"]["params"], source.get("dependencies", {}))
                        check = adapter.evaluate(_predicate(following["step"]))
                    result["suffix"].append(dict(execution=execution, verification=check))
                    if not execution["ok"] or not check.get("known", True) or not check["success"]:
                        result["success"] = False
                        result["error"] = "Downstream task replay failed"
                        break
            result["native_success"] = bool(adapter.native_success())
            if case.get("require_native_success") and not result["native_success"]:
                result["success"] = False
            result["stats"] = adapter.stats()
            if result["success"]:
                result["failure_stage"] = None
        except Exception as error:
            result["success"] = False
            result["error"] = {"type":type(error).__name__, "message":str(error)}
        finally:
            if adapter:
                try:
                    adapter.close()
                except Exception as error:
                    result["success"] = False
                    result["close_error"] = {"type":type(error).__name__, "message":str(error)}
            write_json(Path(output) / "report.json", result)
        return result

    def run(self, experience_paths, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        report = dict(status="complete", proposals=[], promoted=[], rejected=[], accepted_cases=[])
        for index, path in enumerate(experience_paths):
            experience = read_json(path)
            atom = experience["atom"]
            # Distillation and replay are long-running; serialize only the same
            # atom while exploration and other atoms keep making progress.
            with self.library.maintenance_lock(atom):
                case = copy.deepcopy({k:experience[k] for k in ("atom", "task_config", "prefix", "step")})
                case.update(suffix=experience.get("suffix", []),
                            require_native_success=experience.get("require_native_success", False),
                            validation_scope=experience.get("validation_scope", "atomic"))
                if "entry_trace" in experience:
                    case["entry_trace"] = experience["entry_trace"]
                case["source_run"] = experience["source_run"]
                case_id = self.library.stage_case(case)
                def replay_cases(identifier):
                    cases = self.library.cases(atom)
                    if identifier not in {c["id"] for c in cases}:
                        cases.append(self.library.get_case(identifier))
                    return cases

                # Successful reuse contributes a new independently replayed case,
                # without manufacturing an identical new code version.
                active = self.library.active(atom)
                if (experience["successful_record"].get("reused") and active
                        and active["sha256"] == experience["successful_record"]["sha256"]):
                    validation = dict(candidate_sha256=active["sha256"],
                        validation_kind="fresh_environment_replay", cases=[])
                    for replay_case in replay_cases(case_id):
                        validation["cases"].append(self.replay(replay_case, active,
                            output / f"{index:03d}_{atom}" / "reuse" / replay_case["id"]))
                    validation["success"] = all(c["success"] for c in validation["cases"])
                    write_json(output / f"{index:03d}_{atom}" / "reuse" / "validation.json", validation)
                    if validation["success"]:
                        self.library.accept_case(case_id, atom, active["version"], validation)
                        report["accepted_cases"].append(dict(atom=atom, case_id=case_id, version=active["version"]))
                        write_json(output / "summary.json", report)
                        continue
                cases = replay_cases(case_id)
                context = dict(atomic_catalog=CATALOG, api_catalog=self.api_description,
                               supported_predicates=self.predicates, successful_experience=experience,
                               observation=experience.get("entry_observation", experience["initial_observation"]),
                               initial_observation=experience["initial_observation"],
                               library=_active_context(self.library), regression_cases=self.library.cases(atom),
                               pending_case=self.library.get_case(case_id), allow_pending_case_rebind=True,
                               lessons=self.library.lessons(atom)[-8:], previous=None, failures=[])
                accepted = False
                for trial in range(self.max_candidates):
                    candidate_dir = output / f"{index:03d}_{atom}" / f"candidate_{trial:02d}"
                    try:
                        # Exploration may register fresh obligations during a replay.
                        # A rejected promotion must retry against the latest cases.
                        context.update(regression_cases=self.library.cases(atom), library=_active_context(self.library),
                                       lessons=self.library.lessons(atom)[-8:])
                        response = self.agent.distill(context)
                        if "params" in response:
                            if not isinstance(response["params"], dict):
                                raise ValueError("New case bindings must be a complete params dictionary")
                            case["step"] = dict(case["step"], params=copy.deepcopy(response["params"]))
                            case_id = self.library.stage_case(case)
                            context["pending_case"] = self.library.get_case(case_id)
                        cases = replay_cases(case_id)
                        candidate = self.library.propose(atom, response["code"],
                            metadata={"stage":"distill", "source_run":experience["source_run"],
                                      "rationale":response.get("rationale", ""), "new_case_id":case_id},
                            dependencies=response.get("dependencies", []))
                        validation = dict(candidate_sha256=candidate["sha256"], success=False,
                                          validation_kind="fresh_environment_replay", cases=[])
                        for replay_case in cases:
                            validation["cases"].append(self.replay(replay_case, candidate,
                                candidate_dir / replay_case["id"]))
                        validation["success"] = bool(validation["cases"]) and all(c["success"] for c in validation["cases"])
                        self.library.record_validation(candidate["id"], validation)
                        write_json(candidate_dir / "validation.json", validation)
                        report["proposals"].append(dict(atom=atom, candidate_id=candidate["id"], validation=validation))
                        if validation["success"]:
                            version = self.library.promote(candidate["id"])
                            report["promoted"].append(dict(atom=atom, version=version["version"], sha256=version["sha256"]))
                            accepted = True
                            break
                        context["previous"] = candidate
                        context["failures"].append(validation)
                        for replay in validation["cases"]:
                            if replay["case_id"] == case_id and replay.get("entry_observation"):
                                context["observation"] = replay["entry_observation"]
                    except Exception as error:
                        failure = {"type":type(error).__name__, "message":str(error)}
                        context["failures"].append(failure)
                        write_json(candidate_dir / "error.json", failure)
                if not accepted:
                    report["rejected"].append(dict(atom=atom, source_run=experience["source_run"],
                                                    failures=context["failures"]))
            write_json(output / "summary.json", report)
        if report["rejected"]:
            report["status"] = "some_candidates_rejected"
        write_json(output / "summary.json", report)
        return report
