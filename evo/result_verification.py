"""Host-side checks for learned perception outputs and logical assignments.

These checks certify only the declared data contract, not general visual
recognition or a physical effect. Physical predicates remain adapter-owned.
No candidate code runs inside the verifier and no verdict API is exposed to it.
"""
import math


RESULT_PREDICATES = {
    "observed_result": "P-layer only. args:{checks:[{result_path:[str|int,...],source:'observation',path:[str|int,...],atol?:nonnegative number} | {result_path:[...],source:'deproject',camera:str,u:number,v:number,radius?:int,atol?:number}]}. Nonempty checks compare returned fields with current public measurements or host RGB-D projection. Certifies only these data fields, NOT object identity/attributes or complete geometry. No constant/model-supplied expected answer.",
    "effect_report": "P03 only. args:{result_path:[str|int,...],condition:{predicate:advertised physical predicate except native_success,args:{...}}}. Host independently checks whether the returned BOOLEAN correctly reports the local condition, including a correct false. Unknown host evidence cannot pass; this is perception accuracy for that condition, not achievement of the condition.",
    "assignment_valid": "C01 only. args:{objects:[unique object IDs],quotas:{destination:nonnegative integer,...},allowed_destinations?:{object:[destination,...]}}. Return {assignments:[{object:str,destination:str},...],remaining:[object IDs,...]}. Checks exact quotas, no duplicate/foreign objects, allowed destinations and exact remaining set. Conditional on the grounded inputs; does not verify visual attributes or physical placement.",
}


def with_result_predicates(predicates):
    if isinstance(predicates, dict):
        return {**predicates, **RESULT_PREDICATES}
    return list(dict.fromkeys([*predicates, *RESULT_PREDICATES]))


def _path(value, path):
    if not isinstance(path, list) or not 1 <= len(path) <= 16:
        raise ValueError("Result/observation paths require 1..16 keys")
    for key in path:
        if isinstance(value, dict) and isinstance(key, str) and not key.startswith("_"):
            value = value[key]
        elif isinstance(value, list) and type(key) is int and 0 <= key < len(value):
            value = value[key]
        else:
            raise ValueError("Invalid public data path")
    return value


def _matches(actual, expected, atol):
    if type(actual) in (float, int) and type(expected) in (float, int):
        return math.isfinite(actual) and math.isfinite(expected) and abs(actual - expected) <= atol
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(_matches(a, b, atol) for a, b in zip(actual, expected))
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(_matches(actual[k], expected[k], atol) for k in actual)
    return actual == expected


def validate_result_condition(step, predicates):
    condition = step["postcondition"]
    name, atom = condition["predicate"], step["atom"]
    if name not in RESULT_PREDICATES:
        return
    allowed = {"observed_result": {"P01", "P02", "P03"},
               "effect_report": {"P03"}, "assignment_valid": {"C01"}}
    if atom not in allowed[name]:
        raise ValueError(f"{name} is not a valid verifier for {atom}")
    args = condition.get("args", {})
    if name == "effect_report":
        nested = args.get("condition", {}).get("predicate")
        if nested not in predicates or nested in RESULT_PREDICATES or nested == "native_success":
            raise ValueError("effect_report requires an advertised local physical predicate")
    if name == "observed_result":
        checks = args.get("checks")
        if not isinstance(checks, list) or not 1 <= len(checks) <= 32:
            raise ValueError("observed_result requires 1..32 explicit checks")


def evaluate_step(adapter, step, execution=None):
    condition = step["postcondition"]
    name = condition["predicate"]
    if name not in RESULT_PREDICATES:
        return adapter.evaluate(condition)
    result = dict(success=False, known=False, scope="declared_output_contract", predicate=name)
    # Pure functions have no pre-existing world effect to skip. No execution,
    # no result evidence; applies identically to exploration and replay.
    if execution is None or not execution.get("ok"):
        return dict(result, reason="No successfully executed function output")
    value, args = execution.get("result"), condition.get("args", {})
    if not isinstance(value, dict):
        return dict(result, reason="Missing dictionary output")
    if value.get("status") in ("unknown", "failed", "unsupported", "budget_exhausted"):
        return dict(result, reason="Function reports unavailable output evidence")
    try:
        # Validate again for persisted/replayed cases, not just new model plans.
        predicates = getattr(adapter, "host_predicates", ())
        if name != "effect_report":
            validate_result_condition(step, predicates)
        elif step["atom"] != "P03" or args["condition"]["predicate"] in {*RESULT_PREDICATES, "native_success"}:
            raise ValueError("Invalid effect_report condition")
        if name == "observed_result":
            observation = adapter.observe()
            evidence = []
            for check in args["checks"]:
                actual = _path(value, check["result_path"])
                if check["source"] == "observation":
                    expected = _path(observation, check["path"])
                elif check["source"] == "deproject":
                    expected = adapter.deproject(check["camera"], check["u"], check["v"], check.get("radius", 1))
                else:
                    raise ValueError("Unknown measurement source")
                atol = check.get("atol", 0)
                if type(atol) not in (float, int) or not math.isfinite(atol) or atol < 0:
                    raise ValueError("Invalid measurement tolerance")
                evidence.append(dict(result_path=check["result_path"], source=check["source"],
                                     actual=actual, expected=expected, atol=atol, matched=_matches(actual, expected, atol)))
            return dict(result, known=True, success=all(e["matched"] for e in evidence), evidence=evidence,
                        reason="Declared output fields compared to current public sensor data; semantic recognition unassessed")
        if name == "effect_report":
            measured = adapter.evaluate(args["condition"])
            if not measured.get("known", True):
                return dict(result, evidence=measured, reason="Physical condition unknown")
            actual = _path(value, args["result_path"])
            return dict(result, known=True, success=type(actual) is bool and actual == measured["success"],
                        evidence=dict(reported=actual, independent=measured),
                        reason="Perception report compared to independent local condition, not physical completion")
        objects, quotas = args["objects"], args["quotas"]
        if (not isinstance(objects, list) or not objects or not all(isinstance(o, str) for o in objects)
                or len(set(objects)) != len(objects) or not isinstance(quotas, dict) or not quotas
                or not all(isinstance(k, str) and type(v) is int and v >= 0 for k, v in quotas.items())
                or sum(quotas.values()) < 1):
            raise ValueError("Invalid or empty assignment specification")
        assigned, remaining = value["assignments"], value["remaining"]
        if not isinstance(assigned, list) or not isinstance(remaining, list):
            raise ValueError("Assignments and remaining must be lists")
        allowed = args.get("allowed_destinations", {})
        if not isinstance(allowed, dict) or set(allowed) - set(objects) or any(
                not isinstance(destinations, list) or any(d not in quotas for d in destinations)
                for destinations in allowed.values()):
            raise ValueError("Invalid allowed destinations")
        used, counts, valid = set(), dict.fromkeys(quotas, 0), True
        for item in assigned:
            obj, dest = item["object"], item["destination"]
            if obj not in objects or obj in used or dest not in quotas or dest not in allowed.get(obj, quotas):
                valid = False
                break
            used.add(obj)
            counts[dest] += 1
        valid = (valid and counts == quotas and len(remaining) == len(set(remaining))
                 and set(remaining) == set(objects) - used)
        return dict(result, known=True, success=valid, evidence=dict(counts=counts, quotas=quotas, remaining=remaining),
                    reason="Assignment constraints verified conditional on supplied grounded inputs; physical placement unassessed")
    except (KeyError, ValueError, TypeError, IndexError, OverflowError) as error:
        return dict(result, reason="Output verification unavailable: " + str(error))
