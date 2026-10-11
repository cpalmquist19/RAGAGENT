"""Eval aggregation and reporting (no API calls)."""

from typing import Any

import pytest

from licn_rag.evals.runner import aggregate, print_report


def record(**extra: Any) -> dict[str, Any]:
    base = {
        "id": "q1",
        "type": "eligibility",
        "cost_usd": 0.004,
        "latency_s": 12.0,
        "metrics": {"correct": True, "context_recall": 0.5},
    }
    return base | extra


def agent_records() -> list[dict[str, Any]]:
    trace = [
        {"tool": "seed_search", "input": {}, "added_words": 800, "error": False},
        {"tool": "get_section", "input": {}, "added_words": 0, "error": True},
    ]
    return [
        record(
            id="q1",
            context_words=1000,
            research={"turns": 3, "stop": "end_turn", "calculations": []},
            trace=trace,
        ),
        record(
            id="q2",
            context_words=3000,
            research={"turns": 3, "stop": "max_turns", "calculations": []},
            trace=trace,
        ),
    ]


def test_aggregate_old_style_records() -> None:
    s = aggregate([record()])
    assert s["context_words_p50"] is None
    assert s["tool_calls_mean"] is None
    assert s["research_turns_mean"] is None
    assert s["stopped_at_turn_limit"] is None
    assert s["overall"]["context_recall"] == 0.5


def test_aggregate_agent_records() -> None:
    s = aggregate(agent_records())
    assert s["context_words_p50"] == 2000
    assert s["tool_calls_mean"] == 1.0  # the seed search isn't a tool call the model made
    assert s["research_turns_mean"] == 3.0
    assert s["stopped_at_turn_limit"] == 1
    assert s["tool_errors"] == 2


def test_print_report_handles_old_summaries(capsys: pytest.CaptureFixture[str]) -> None:
    print_report([("old", aggregate([record()])), ("new", aggregate(agent_records()))])
    out = capsys.readouterr().out
    assert "Key sections in the model's context" in out
    assert "–" in out
