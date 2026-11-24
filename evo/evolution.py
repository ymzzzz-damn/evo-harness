"""Parameterize successes and failures, with distinct publication evidence."""
import copy
from pathlib import Path

from .admission import code_diff, host_success, same_program
from .catalog import get_spec, task_catalog, skill_context
from .io import read_json, write_json
from .loops import _active_context, _attempt_memory, _failure_feedback


def _save_draft(library, atom, experience, candidate, params, progress, rationale,
                source_memory_id=None, failure_analysis=None):
    observed = experience.get("successful_record", experience.get("source_record", {}))
    return library.remember(atom, dict(code=candidate["code"], description=candidate["description"], dependencies=candidate["dependencies"],
        candidate_id=candidate["id"], params=params, step=experience["step"],
        source_memory_id=source_memory_id or observed.get("memory_id"),
        source_success=host_success(observed), parameterized=True, replay_verified=False,
        failure_feedback=_failure_feedback(observed), failure_analysis=failure_analysis,
        task_config=experience["task_config"], source_run=experience["source_run"],
        entry_observation=experience.get("entry_observation"), entry_trace=experience.get("entry_trace", []),
        attempt_history=experience.get("attempt_history", []), observations_are_stale=True,
        progress=progress, summary_source="model_unverified", rationale=rationale,
        callable_as_published=False), kind="draft")


def consolidate_experiences(loop, experience_paths, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="complete", admission_policy="compare_existing", proposals=[], promoted=[],
                  rejected=[], accepted_cases=[], kept=[], drafts=[], retained_failures=[], errors=[],
                  replay_counts={"total": 0, "fresh": 0, "cached": 0}, replay_wall_seconds=0.0)
    for index, path in enumerate(experience_paths):
        experience = read_json(path)
        atom = experience["atom"]
        directory = output / f"{index:03d}_{atom}"
        directory.mkdir()
        try:
            observed = experience.get("successful_record", experience.get("source_record"))
            if not isinstance(observed, dict):
                raise ValueError("Experience must include a source attempt record")
            source_success = host_success(observed)
            memory_id = observed.get("memory_id")
            if memory_id is None:
                memory_id = loop.library.remember(atom, _attempt_memory(observed, experience))["id"]
            with loop.library.maintenance_lock(atom):
                active = loop.library.active(atom)
                if source_success and active is None:
                    # A draft is not a published capability. First admission must
                    # still prove the parameterized implementation by replay.
                    outcome = loop._run_replay([path], directory / "first_use", _lock_held=True)
                    for key in ("proposals", "promoted", "rejected", "accepted_cases"):
                        report[key].extend(outcome.get(key, []))
                    for key, count in outcome.get("replay_counts", {}).items():
                        report["replay_counts"][key] += count
                    report["replay_wall_seconds"] += outcome.get("replay_wall_seconds", 0)
                    if not outcome.get("promoted"):
                        for proposal in outcome.get("proposals", []):
                            candidate = loop.library.get_candidate(proposal["candidate_id"])
                            pending = loop.library.get_case(candidate["metadata"]["new_case_id"])
                            draft = _save_draft(loop.library, atom, experience, candidate, pending["step"]["params"],
                                {"completed": [], "remaining": [experience["step"]["goal"]],
                                 "entry_requirements": ["First-use replay failed; inspect validation feedback"]},
                                "Parameterization preserved for repair after failed first-use replay")
                            report["drafts"].append(dict(atom=atom, memory_id=draft["id"]))
                    continue

                if source_success and same_program(active, observed):
                    # Already parameterized active code: no model call, replay,
                    # new version or falsely activated regression case.
                    kept = dict(atom=atom, version=active["version"], memory_id=memory_id,
                                reason="equivalent_code_and_dependency_pins", replay_performed=False)
                    report["kept"].append(kept)
                    write_json(directory / "comparison.json", kept)
                    continue

                context = dict(**task_catalog(loop.library), **skill_context(loop.library, atom), api_catalog=loop.api_description,
                    supported_predicates=loop.predicates, source_success=source_success,
                    failure_feedback=_failure_feedback(observed),
                    active=active, experience=experience, observation=experience.get("entry_observation"),
                    instruction=experience.get("initial_observation", {}).get("instruction", ""),
                    initial_observation=experience.get("initial_observation"),
                    code_diff=code_diff(active["code"], observed["code"]) if active else None,
                    library=_active_context(loop.library), regression_cases=loop.library.cases(atom),
                    working_memory=loop.library.working_memory(atom, limit=4, task=experience["task_config"].get("task")),
                    record_working_memory=True, lessons=loop.library.lessons(atom)[-8:])
                response = loop.agent.consolidate(context)
                write_json(directory / "consolidation.json", response)
                if response["decision"] == "keep":
                    if not source_success or not active:
                        raise ValueError("Only a successful existing capability may be retained by comparison")
                    report["kept"].append(dict(atom=atom, version=active["version"], memory_id=memory_id,
                                               reason=response["rationale"], replay_performed=False))
                    continue
                candidate = loop.library.propose(atom, response["code"], dependencies=response["dependencies"],
                    metadata=dict(stage="consolidate", source_run=experience["source_run"],
                                  rationale=response["rationale"], source_memory_id=memory_id,
                                  description=response.get("description", get_spec(atom)["description"]),
                                  parameter_example=copy.deepcopy(response["params"])))
                if response["decision"] == "update":
                    if not source_success or not active:
                        raise ValueError("Incomplete code cannot replace active code")
                    if same_program(active, candidate):
                        report["kept"].append(dict(atom=atom, version=active["version"], memory_id=memory_id,
                                                  reason="equivalent_consolidated_code_and_pins", replay_performed=False))
                    else:
                        version = loop.library.publish_comparison(candidate["id"], memory_id, response["rationale"])
                        report["promoted"].append(dict(atom=atom, version=version["version"], sha256=version["sha256"],
                            admission_kind="code_comparison", replay_verified=False))
                        report["proposals"].append(dict(atom=atom, candidate_id=candidate["id"], validation=version["validation"]))
                        write_json(directory / "comparison.json", version["validation"])
                else:
                    draft = _save_draft(loop.library, atom, experience, candidate, response["params"],
                                        response["progress"], response["rationale"],
                                        source_memory_id=memory_id, failure_analysis=response.get("failure_analysis"))
                    report["drafts"].append(dict(atom=atom, memory_id=draft["id"], candidate_id=candidate["id"]))
                    if not source_success:
                        report["retained_failures"].append(dict(atom=atom, memory_id=draft["id"],
                            source_memory_id=memory_id, reason="parameterized_failure_for_next_repair",
                            parameterized=True, replay_performed=False))
        except Exception as error:
            failure = dict(atom=atom, experience=str(path), type=type(error).__name__, message=str(error))
            report["errors"].append(failure)
            report["rejected"].append(failure)
            write_json(directory / "error.json", failure)
        finally:
            if report["rejected"]:
                report["status"] = "some_candidates_rejected"
            write_json(output / "summary.json", report)
    return report
