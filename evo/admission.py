"""Code comparison is an admission policy, never a simulator success verdict."""
import ast
import difflib
import re


def same_program(left, right):
    """Ignore formatting/comments, but retain literals and exact dependency pins."""
    return (ast.dump(ast.parse(left["code"]), include_attributes=False)
            == ast.dump(ast.parse(right["code"]), include_attributes=False)
            and left.get("dependencies", {}) == right.get("dependencies", {}))


def code_diff(before, after):
    return "\n".join(difflib.unified_diff(before.splitlines(), after.splitlines(),
                                        fromfile="active", tofile="proposed", lineterm=""))


def host_success(record):
    return (record.get("success") is True and record.get("executed", True) is True
            and not record.get("skipped") and record.get("execution", {}).get("ok") is True
            and record.get("verification", {}).get("known", True) is True
            and record.get("verification", {}).get("success") is True)


def validate_comparison(proof, atom, digest):
    """Validate a host admission receipt without claiming fresh replay coverage."""
    base, evidence = proof.get("base", {}), proof.get("source_evidence", {})
    if (proof.get("validation_kind") != "code_comparison" or proof.get("success") is not True
            or proof.get("candidate_sha256") != digest or proof.get("decision") != "update"
            or proof.get("replay_verified") is not False or proof.get("regression_verified") is not False
            or base.get("atom") != atom or type(base.get("version")) is not int or base["version"] < 1
            or not isinstance(base.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", base["sha256"])
            or evidence.get("atom") != atom or evidence.get("host_success") is not True
            or not isinstance(evidence.get("memory_id"), str)
            or not isinstance(evidence.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", evidence["sha256"])
            or not isinstance(proof.get("rationale"), str) or not proof["rationale"].strip()
            or proof.get("cases") != []):
        raise ValueError("Invalid code-comparison admission receipt")
