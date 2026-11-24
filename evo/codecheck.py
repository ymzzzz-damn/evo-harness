"""Static contract checks for generated policies, not a security sandbox.

Execute generated code only in a trusted, resource-limited worker. AST checks
reduce accidental access and enforce the public API but do not isolate Python.
"""

from __future__ import annotations

import ast
import hashlib

from .catalog import get_spec


_FORBIDDEN_NAMES = {
    "__import__", "open", "eval", "exec", "compile", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "input", "breakpoint", "help",
    "type", "object", "super", "exit", "quit", "memoryview",
    "BaseException", "BaseExceptionGroup", "SystemExit", "KeyboardInterrupt", "GeneratorExit",
}
_FORBIDDEN_NODES = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.AsyncFunctionDef, ast.Await,
                    ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom)
API_METHODS = frozenset({"observe", "step", "move_joints", "move_tcp", "move_base", "gripper", "wait", "deproject", "read_camera", "call"})
_FORBIDDEN_ATTRIBUTES = {"load", "save", "savez", "savez_compressed", "loadtxt", "savetxt", "genfromtxt",
                         "fromfile", "tofile", "memmap", "open_memmap", "dump", "dumps", "ctypes",
                         "ctypeslib", "f2py", "testing", "test", "os", "sys", "subprocess", "pathlib"}


def source_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def check_code(code: str, atom: str | None = None) -> dict:
    """Check a parameterized entrypoint and collect literal api.call dependencies.

    Constants for distances and gains are valid; scene entity IDs and poses should
    come from params and observations. Semantic generalization is established by
    replaying parameter bindings, not by pretending a syntax check proves it.
    """
    if atom is not None:
        get_spec(atom)
    errors: list[str] = []
    dependencies: set[str] = set()
    if not isinstance(code, str):
        return {"ok": False, "errors": ["code must be a string"], "dependencies": [], "sha256": None}
    digest = source_hash(code)
    if len(code.encode("utf-8")) > 128_000:
        errors.append("source exceeds the 128 KB policy limit")
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError) as exc:
        return {"ok": False, "errors": [str(exc)], "dependencies": [], "sha256": digest}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "skill"]
    if len(functions) != 1:
        errors.append("exactly one def skill(api, params) entrypoint is required")
    else:
        fn = functions[0]
        args = fn.args
        if ([arg.arg for arg in args.args] != ["api", "params"] or args.posonlyargs
                or args.kwonlyargs or args.vararg or args.kwarg or args.defaults
                or args.kw_defaults or fn.decorator_list):
            errors.append("entrypoint must be exactly def skill(api, params), without defaults or decorators")
        if not any(isinstance(node, ast.Name) and node.id == "params" and isinstance(node.ctx, ast.Load)
                   for node in ast.walk(fn)):
            errors.append("skill must read params so scene bindings remain parameterized")
        if not any(isinstance(node, ast.Return) for node in ast.walk(fn)):
            errors.append("skill must return a dictionary; unimplemented stubs cannot be candidates")
    for node in tree.body:
        is_docstring = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        if not isinstance(node, ast.FunctionDef) and not is_docstring:
            errors.append("only function definitions and a docstring are allowed at module scope")
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            errors.append(f"{type(node).__name__} is not allowed in a skill")
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            errors.append("bare except is not allowed; it could swallow the execution deadline")
        if isinstance(node, ast.FunctionDef) and (node.decorator_list or node.args.defaults or any(node.args.kw_defaults)):
            errors.append("function decorators and default expressions are not allowed")
        if isinstance(node, ast.FunctionDef) and (node.returns is not None or any(
                arg.annotation is not None for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs)):
            errors.append("function annotations are not allowed in generated skills")
        if isinstance(node, ast.Name) and (node.id in _FORBIDDEN_NAMES or node.id.startswith("__")):
            errors.append(f"name {node.id!r} is not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            errors.append("private/dunder attribute access is not allowed")
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTRIBUTES:
            errors.append(f"attribute {node.attr!r} is not allowed in a generated skill")
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "api":
            if node.attr not in API_METHODS:
                errors.append(f"api.{node.attr} is not an exposed primitive")
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                errors.append("the primitive API cannot be modified")
            parent = parents.get(node)
            if not isinstance(parent, ast.Call) or parent.func is not node:
                errors.append("API methods must be called directly, without aliases")
        if isinstance(node, ast.Name) and node.id == "api" and isinstance(node.ctx, ast.Load):
            parent = parents.get(node)
            if not isinstance(parent, ast.Attribute) or parent.value is not node:
                errors.append("the API object cannot be aliased or passed to other functions")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            func = node.func
            if isinstance(func.value, ast.Name) and func.value.id == "api" and func.attr == "call":
                if len(node.args) != 2 or node.keywords or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
                    errors.append("dependencies must use api.call('Pxx/Axx/Cxx', params_dictionary)")
                else:
                    dependency = node.args[0].value
                    try:
                        get_spec(dependency)
                    except ValueError as exc:
                        errors.append(str(exc))
                    else:
                        if dependency == atom:
                            errors.append("a skill cannot call itself")
                        dependencies.add(dependency)
    return {"ok": not errors, "errors": sorted(set(errors)), "dependencies": sorted(dependencies), "sha256": digest}


def require_valid_code(code: str, atom: str | None = None) -> dict:
    result = check_code(code, atom)
    if not result["ok"]:
        raise ValueError("Invalid skill code: " + "; ".join(result["errors"]))
    return result
