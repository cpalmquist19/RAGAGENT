"""Safe arithmetic for the agent's `calculate` tool.

Models slip on unit conversions (semester hours to CEUs) and running totals, so the agent
does arithmetic here instead. Only numbers, + - * / // % **, parentheses, and the
functions round/min/max/abs are allowed; anything else raises ValueError.
"""

import ast
import operator
from collections.abc import Callable
from typing import Any

MAX_EXPRESSION_CHARS = 200
MAX_EXPONENT = 10

_BINARY: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_FUNCTIONS: dict[str, Callable[..., Any]] = {"round": round, "min": min, "max": max, "abs": abs}


def evaluate(expression: str) -> float:
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise ValueError(f"expression is longer than {MAX_EXPRESSION_CHARS} characters")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"not a valid arithmetic expression ({e.msg})") from None
    try:
        return _eval(tree.body)
    except (ZeroDivisionError, OverflowError, TypeError) as e:
        raise ValueError(f"cannot evaluate: {e}") from None


def _eval(node: ast.AST) -> Any:
    # type() rather than isinstance(): bool is a subclass of int and must be rejected.
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError(f"exponents larger than {MAX_EXPONENT} are not allowed")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](*(_eval(a) for a in node.args))
    raise ValueError("only numbers, + - * / // % **, and round/min/max/abs are allowed")


def format_number(value: float) -> str:
    """6.0 -> '6'; 3.3333333 -> '3.3333'."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return str(round(value, 4))
    return str(value)


def calculate(expression: str) -> str:
    """'6 * 2/3' -> '6 * 2/3 = 4'. Raises ValueError with a message the model can act on."""
    return f"{expression.strip()} = {format_number(evaluate(expression))}"
