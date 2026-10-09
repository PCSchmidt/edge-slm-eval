"""Deterministic gate between model text and execution.

A model output is only ever *data*. It becomes a `ToolCall` iff it is (1) JSON matching
`ModelOutput`, and (2) its `call` string parses with `ast` as exactly one call to a registered
function with positional literal arguments of the registered types. Nothing is evaluated
before that; `eval` / `exec` are never used.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, ValidationError

from .tools import REGISTRY


class GuardrailViolation(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code


class ModelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call: str


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: tuple[str | float, ...]


def parse_output(raw: str) -> ModelOutput:
    try:
        return ModelOutput.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise GuardrailViolation("SCHEMA", str(exc).splitlines()[0]) from exc


def _literal(node: ast.expr) -> str | float:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _literal(node.operand)
        if isinstance(inner, str):
            raise GuardrailViolation("ARG_TYPE", "unary operator on string")
        return -inner if isinstance(node.op, ast.USub) else inner
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float)):
        if isinstance(node.value, bool):
            raise GuardrailViolation("ARG_TYPE", "bool literal")
        return node.value if isinstance(node.value, str) else float(node.value)
    raise GuardrailViolation("NON_LITERAL", f"{type(node).__name__} is not a literal argument")


def check_call(call: str) -> ToolCall:
    """Validate a call string. Raises GuardrailViolation; never executes anything."""
    if len(call) > 500:
        raise GuardrailViolation("TOO_LONG", "call string over 500 characters")
    try:
        tree = ast.parse(call.strip(), mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise GuardrailViolation("SYNTAX", str(exc)) from exc
    node = tree.body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise GuardrailViolation("NOT_A_CALL", "expected a single plain function call")
    if node.func.id not in REGISTRY:
        raise GuardrailViolation("UNKNOWN_TOOL", node.func.id)
    if node.keywords:
        raise GuardrailViolation("KEYWORDS", "keyword arguments are not allowed")
    spec = REGISTRY[node.func.id]
    if len(node.args) != len(spec.arg_types):
        raise GuardrailViolation("ARITY", f"{node.func.id} takes {len(spec.arg_types)} args")
    values = tuple(_literal(a) for a in node.args)
    for value, expected in zip(values, spec.arg_types, strict=True):
        if expected is str and not isinstance(value, str):
            raise GuardrailViolation("ARG_TYPE", f"expected str, got {value!r}")
        if expected is float and isinstance(value, str):
            raise GuardrailViolation("ARG_TYPE", f"expected number, got {value!r}")
    return ToolCall(node.func.id, values)


def execute(call: ToolCall) -> object:
    """Run an already-validated call."""
    return REGISTRY[call.name].fn(*call.args)
