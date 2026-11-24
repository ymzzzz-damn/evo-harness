"""Human300 P/A/C32 document loading, implementation docs and taxonomy isolation."""
import ast
import hashlib
import json
from pathlib import Path

import pytest

from evo.agent import Agent, ResponseError, ScriptedProvider, validate_plan
from evo.catalog import CATALOG, CATALOG_PATH, CATALOG_VERSION, get_spec, stub, task_catalog
from evo.cli import _compact_attempt
from evo.codecheck import check_code
from evo.library import Library, LibraryError
from evo.policy_evaluation import FrozenLibrary, freeze_active_library
from evo.prompts import ATOM_NAMES, PLAN_SCHEMA
from test_evolution import Harness, failed_consolidation, consolidation
from test_loops import GOOD, BAD, candidate, plan, step


def test_document_schema_and_stub_descriptions_share_one_catalog(tmp_path):
    library = Library(tmp_path / "library")
    document = json.loads(CATALOG_PATH.read_text())
    assert document["skills"] == CATALOG
    assert len(document["families"]) == 13
    assert list(ATOM_NAMES) == PLAN_SCHEMA["properties"]["steps"]["items"]["properties"]["atom"]["enum"]
    assert set(ATOM_NAMES) == set(CATALOG)
    for atom in CATALOG:
        code = (library.stubs_dir / f"{atom}.py").read_text()
        assert code == stub(atom)
        assert get_spec(atom)["description"] in ast.get_docstring(ast.parse(code).body[0])
    assert library.list_active() == []
    assert json.loads(library.state_path.read_text())["atomic_catalog_version"] == CATALOG_VERSION


def test_task_loads_overview_before_factory_and_selected_details_on_demand(tmp_path):
    h = Harness(tmp_path)
    actual = h.factory
    seen = []
    def factory(config, output):
        # The task's disk-read receipt exists before its simulation is constructed.
        receipt = json.loads((Path(output).parent / "atomic_catalog.json").read_text())
        seen.append(receipt)
        return actual(config, output)
    h.factory = factory
    result = h.explore({"plan": [plan()], "repair": [candidate(GOOD)]})
    assert result["status"] == "success"
    requests = h.provider.requests
    assert [r["stage"] for r in requests] == ["plan", "repair"]  # no extra preflight model call
    initial = requests[0]["context"]
    doc = initial["catalog_document"]
    assert doc["source_path"] == str(CATALOG_PATH)
    assert doc["sha256"] == hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest()
    assert initial["atomic_catalog"] == seen[0]["atomic_catalog"]
    assert "atomic_skill" not in initial
    assert all("description" not in spec for spec in initial["atomic_catalog"].values())
    selected = requests[1]["context"]["atomic_skill"]
    assert selected["id"] == "A02" and selected["implementation"] is None
    assert selected["description"] == get_spec("A02")["description"]
    assert Path(selected["stub_path"]).is_file()
    assert str(CATALOG_PATH) in requests[0]["instructions"]
    assert "description" in requests[1]["response_schema"]["required"]
    h.factory = actual
    maintained = h.maintain(result, {"distill": [candidate(GOOD)]})
    assert maintained["promoted"]
    assert h.provider.requests[0]["context"]["atomic_skill"]["id"] == "A02"


@pytest.mark.parametrize("stage", ["repair", "distill", "consolidate"])
def test_selected_code_generation_requires_nonempty_description(stage):
    response = consolidation() if stage == "consolidate" else candidate(GOOD)
    response.pop("description")
    provider = ScriptedProvider({stage: [response]})
    context = {"require_skill_description": True, "source_success": True, "active": {"version": 1}}
    with pytest.raises(ResponseError, match="description"):
        getattr(Agent(provider, max_attempts=1), stage)(context)
    response["description"] = "   "
    provider = ScriptedProvider({stage: [response]})
    with pytest.raises(ResponseError, match="description"):
        getattr(Agent(provider, max_attempts=1), stage)(context)


def test_description_survives_publish_comparison_freeze_and_next_plan(tmp_path):
    h = Harness(tmp_path)
    h.learn(BAD, target=4)
    active = h.library.active("A02")
    assert active["description"] == candidate(GOOD)["description"]
    from scripts.prepare_policy_evaluation import _check_frozen_memory
    first_root = tmp_path / "first-use-frozen"
    first_receipt = freeze_active_library(h.library.root, first_root)
    assert _check_frozen_memory(first_root, first_receipt)["training_memory_empty"]
    result = h.explore({"plan": [plan(step(target=3))], "repair": [candidate(GOOD)]}, goal=3, max_attempts=2)
    inventory = h.provider.requests[0]["context"]["library"][0]
    assert inventory["description"] == active["description"]
    assert "code" not in inventory and Path(inventory["source_path"]).is_file()
    response = consolidation(target=3)
    response["description"] = "Move directly to target in plant joint units; returns diagnostics. Comparison update has not been replayed."
    outcome = h.maintain(result, {"consolidate": [response]})
    assert outcome["promoted"] and outcome["replay_counts"]["total"] == 0
    active = h.library.active("A02")
    assert active["version"] == 2 and active["description"] == response["description"]
    frozen_root = tmp_path / "frozen"
    receipt = freeze_active_library(h.library.root, frozen_root)
    frozen = FrozenLibrary(frozen_root, receipt["state_sha256"])
    scratch = frozen.scratch(tmp_path / "scratch")
    assert scratch.active("A02")["description"] == active["description"]
    state = json.loads((frozen_root / "state.json").read_text())
    assert state["versions"]["A02"]["2"]["metadata"] == {}


def test_parameterized_failure_description_is_used_for_new_scene(tmp_path):
    h = Harness(tmp_path)
    result = h.explore({"plan": [plan()], "repair": [candidate(BAD)]}, horizon=1)
    response = failed_consolidation()
    response["description"] = "Unverified toy motion; target: new joint goal, fraction: travel multiplier. Re-ground before use."
    report = h.maintain(result, {"consolidate": [response]})
    assert report["drafts"]
    assert h.provider.requests[0]["context"]["atomic_skill"]["id"] == "A02"
    memory = h.library.working_memory("A02")[0]["payload"]
    assert memory["description"] == response["description"] and memory["failure_analysis"]
    assert h.library.active("A02") is None
    result = h.explore({"plan": [plan(step(target=4))], "repair": [candidate(GOOD)]}, goal=4, task="another-scene")
    request = next(r for r in h.provider.requests if r["stage"] == "repair")
    assert request["context"]["previous"]["description"] == response["description"]
    assert request["context"]["previous"]["failure_analysis"] == response["failure_analysis"]
    assert request["context"]["step"]["params"]["target"] == 4
    record = result["completed_steps"][0]
    assert _compact_attempt(record)["description"] == candidate(GOOD)["description"]


@pytest.mark.parametrize("mode", ["catalog", "state", "version"])
def test_foreign_library_is_rejected_without_rewriting_evidence(tmp_path, mode):
    library = Library(tmp_path / "old")
    if mode == "catalog":
        (library.root / "catalog.json").write_text(json.dumps({"P01": {"name": "pick"}}))
    else:
        state = json.loads(library.state_path.read_text())
        if mode == "state":
            state.pop("atomic_catalog_version")
            state["lessons"] = {"P01": []}
        else:
            state["atomic_catalog_version"] = "p40-v1"
        library.state_path.write_text(Library._state_text(state))
    evidence = {p: p.read_bytes() for p in library.root.rglob("*") if p.is_file()}
    with pytest.raises(LibraryError, match="Incompatible"):
        Library(library.root)
    assert {p: p.read_bytes() for p in evidence} == evidence
    assert set(library.stubs_dir.glob("*.py")) == {library.stubs_dir / f"{atom}.py" for atom in CATALOG}


def test_foreign_atoms_rejected_in_plan_and_dependency():
    for atom in ("P04", "R01", "A27", "C04"):
        with pytest.raises(ResponseError, match="P01..P03"):
            validate_plan(plan(step(atom=atom)))
        assert not check_code(f'def skill(api, params):\n    return api.call("{atom}", params)\n', "A02")["ok"]


def test_catalog_change_mid_process_is_not_silently_mixed(tmp_path, monkeypatch):
    import evo.catalog as module
    library = Library(tmp_path / "library")
    altered = tmp_path / "altered.json"
    document = json.loads(CATALOG_PATH.read_text())
    document["skills"]["A02"]["description"] += " Changed contract."
    altered.write_text(json.dumps(document))
    monkeypatch.setattr(module, "CATALOG_PATH", altered)
    with pytest.raises(ValueError, match="restart"):
        task_catalog(library)
