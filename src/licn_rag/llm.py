"""Anthropic client setup shared by the app and the eval tooling.

Reads .env: ANTHROPIC_API_KEY (used by the SDK directly), ANTHROPIC_MODEL, and
ANTHROPIC_WORKSPACE_ID. The workspace ID is sent as the `anthropic-workspace-id` header,
which selects the workspace for multi-workspace keys; a single-workspace key always runs
in its own workspace. `verify_workspace` confirms which one the API actually used.
"""

import os
from typing import Any

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

DEFAULT_MODEL = "claude-sonnet-5-5"

# Standard pricing per million tokens; the Batch API halves all of these. Cache writes
# are 1.25x input and cache reads 0.1x input. Haiku rates apply to prompts up to 100K.
PRICES = {
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_write": 2.50, "cache_read": 0.20},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_write": 0.125, "cache_read": 0.01},
}


def price(model: str) -> dict[str, float]:
    return next((v for k, v in PRICES.items() if model.startswith(k)), PRICES[DEFAULT_MODEL])


def usage_cost(model: str, usage: Any) -> float:
    """Dollar cost of one response at standard (non-batch) prices."""
    p = price(model)
    return (
        usage.input_tokens * p["input"]
        + usage.output_tokens * p["output"]
        + (usage.cache_creation_input_tokens or 0) * p["cache_write"]
        + (usage.cache_read_input_tokens or 0) * p["cache_read"]
    ) / 1_000_000


def output_schema(model: type[BaseModel]) -> dict[str, Any]:
    """A Pydantic model as a JSON schema that structured outputs accept.

    Inlines $defs, drops titles/defaults/descriptions, and marks every object strict
    (all properties required, no extras).
    """
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(x) for x in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return walk(defs[node["$ref"].rsplit("/", 1)[-1]])
        out = {}
        for key, value in node.items():
            if key in ("title", "default", "description"):
                continue
            if key == "properties":  # keys here are field names, not schema keywords
                out[key] = {name: walk(v) for name, v in value.items()}
            else:
                out[key] = walk(value)
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"])
        return out

    return walk(schema)


def first_text(message: Any) -> str:
    return next((b.text for b in message.content if b.type == "text"), "")


def load_env() -> None:
    load_dotenv(override=False)


def model_name() -> str:
    load_env()
    return os.getenv("ANTHROPIC_MODEL") or DEFAULT_MODEL


def make_client() -> anthropic.Anthropic:
    load_env()
    headers = {}
    if workspace := os.getenv("ANTHROPIC_WORKSPACE_ID"):
        headers["anthropic-workspace-id"] = workspace
    return anthropic.Anthropic(default_headers=headers)


def verify_workspace(client: anthropic.Anthropic, model: str) -> str | None:
    """Return the workspace the API resolved this key to (free count_tokens call).

    Raises if ANTHROPIC_WORKSPACE_ID is set and the API used a different workspace.
    """
    raw = client.messages.with_raw_response.count_tokens(
        model=model, messages=[{"role": "user", "content": "ping"}]
    )
    actual = raw.headers.get("anthropic-workspace-id")
    expected = os.getenv("ANTHROPIC_WORKSPACE_ID")
    if expected and actual and actual != expected:
        raise RuntimeError(f"API used workspace {actual}, but .env expects {expected}")
    return actual
