"""Answer-step replay and the dev/test split (no API calls)."""

import json
from collections import Counter
from pathlib import Path

from test_agent_pipeline import ANSWER_JSON, MODEL, QUESTION, FakeClient, message, text

from licn_rag.corpus import Corpus
from licn_rag.evals.runner import ANSWER_KEYS, SPLITS, make_splits, prompt_version, read_jsonl
from licn_rag.pipelines.replay import ReplayPipeline, load_records


def test_splits_file_matches_make_splits() -> None:
    keys = {r["id"]: r["answer"] for r in read_jsonl(ANSWER_KEYS) if "answer" in r}
    splits = make_splits(keys)
    assert json.loads(SPLITS.read_text(encoding="utf-8")) == splits
    assert not set(splits["dev"]) & set(splits["test"])
    assert set(splits["dev"]) | set(splits["test"]) == set(keys)
    # Every question type is in both halves, at roughly the dev share.
    dev_types = Counter(keys[i]["corrected_type"] for i in splits["dev"])
    all_types = Counter(k["corrected_type"] for k in keys.values())
    for t, n in all_types.items():
        assert abs(dev_types[t] - n * 0.25) <= 1


def test_prompt_version_is_stable() -> None:
    assert prompt_version() == prompt_version()
    assert len(prompt_version()) == 8


def test_replay_reuses_saved_passages_and_calculations(tmp_path: Path) -> None:
    saved = {
        "id": "q1",
        "question": QUESTION,
        "context_ids": ["LICN-030 §1(a)", "LICN-001 §2(g)"],
        "research": {"calculations": ["CEUs short: 8 - 6.5 = 1.5"]},
    }
    (tmp_path / "results.jsonl").write_text(json.dumps(saved) + "\n", encoding="utf-8")
    client = FakeClient([message([text(ANSWER_JSON)], "end_turn")])
    pipeline = ReplayPipeline(
        client,  # type: ignore[arg-type]
        MODEL,
        Corpus.load(),
        load_records(tmp_path),
    )
    result = pipeline.answer(QUESTION)
    assert result.answer is not None and result.answer.verdict == "no"
    call = client.messages.calls[0]
    passages = call["system"][-1]["text"]
    assert "[LICN-030 §1(a)]" in passages and "[LICN-001 §2(g)]" in passages
    assert "LICN-019" not in passages
    assert "8 - 6.5 = 1.5" in call["messages"][0]["content"]
    assert result.context_ids == ["LICN-001 §2(g)", "LICN-030 §1(a)"]

    missing = pipeline.answer("a question the run never saw")
    assert missing.answer is None and missing.error
