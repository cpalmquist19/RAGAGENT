"""Cross-reference extraction over the real corpus (no API calls)."""

import pytest

from licn_rag.agent.references import ReferenceGraph, ref_path, reference_graph
from licn_rag.corpus import Corpus


@pytest.fixture(scope="module")
def graph() -> ReferenceGraph:
    return reference_graph(Corpus.load())


@pytest.mark.parametrize(
    ("raw", "path"),
    [("9.a.ii", "9(a)(ii)"), ("3.d.", "3(d)"), ("(2)(a)", "2(a)"), ("2(b)", "2(b)"), ("5.", "5")],
)
def test_ref_path(raw: str, path: str) -> None:
    assert ref_path(raw) == path


def targets(graph: ReferenceGraph, citation: str) -> set[str]:
    return {r.target for r in graph.from_passage(citation)}


def test_cross_policy_reference(graph: ReferenceGraph) -> None:
    assert targets(graph, "LICN-010 §5(c)") == {"LICN-019"}


def test_cross_policy_reference_to_a_subsection(graph: ReferenceGraph) -> None:
    assert "LICN-007 §2(b)" in targets(graph, "LICN-004 §1(h)(ii)(C)")


def test_same_policy_references(graph: ReferenceGraph) -> None:
    assert {"LICN-007 §3(d)", "LICN-007 §3", "LICN-007 §4"} <= targets(graph, "LICN-007 §5(c)")
    assert "LICN-019 §2(a)" in targets(graph, "LICN-019 §5(e)")


def test_parent_section_includes_children(graph: ReferenceGraph) -> None:
    # §5's sub-items point at nine license policies.
    assert {"LICN-019", "LICN-020", "LICN-024"} <= targets(graph, "LICN-010 §5")


def test_no_reference_to_own_ancestor(graph: ReferenceGraph) -> None:
    assert "LICN-007 §5" not in targets(graph, "LICN-007 §5(b)")


def test_citing(graph: ReferenceGraph) -> None:
    sources = {r.source for r in graph.citing("LICN-019")}
    assert {"LICN-001 §2(w)", "LICN-010 §5(c)", "LICN-028 §5"} <= sources


def test_missing_policy(graph: ReferenceGraph) -> None:
    assert graph.missing_policies() == ["LICN-014"]
    assert any(not r.in_corpus for r in graph.from_passage("LICN-001 §2(u)"))
