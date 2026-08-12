from types import SimpleNamespace

from lexigo.config import Settings
from lexigo.graph import build_qa_graph


class FakeStore:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, question, top_k=None):
        return self.chunks


class FakeModel:
    def invoke(self, prompt):
        return SimpleNamespace(content="The answer is in the supplied document [chunk-1].")


def settings(**overrides):
    values = {"min_relevance_score": 0.4, "min_query_term_overlap": 0.1, "max_graph_steps": 6}
    values.update(overrides)
    return Settings(**values)


def test_good_retrieval_takes_answer_branch() -> None:
    graph = build_qa_graph(
        FakeStore(
            [
                {
                    "chunk_id": "chunk-1",
                    "source_file": "facts.md",
                    "chunk_index": 0,
                    "text": "The notice period is 60 days.",
                    "score": 0.9,
                }
            ]
        ),
        FakeModel(),
        settings(),
    )

    result = graph.invoke({"question": "What is the notice period?", "max_steps": 6})

    assert result["grounded"] is True
    assert result["citations"][0]["chunk_id"] == "chunk-1"
    assert [event["node"] for event in result["trace"]] == ["retrieve", "assess", "answer"]


def test_weak_retrieval_takes_fallback_branch() -> None:
    graph = build_qa_graph(
        FakeStore(
            [
                {
                    "chunk_id": "chunk-1",
                    "source_file": "facts.md",
                    "chunk_index": 0,
                    "text": "The notice period is 60 days.",
                    "score": 0.2,
                }
            ]
        ),
        FakeModel(),
        settings(),
    )

    result = graph.invoke({"question": "What is the rent?", "max_steps": 6})

    assert result["grounded"] is False
    assert result["citations"] == []
    assert [event["node"] for event in result["trace"]] == ["retrieve", "assess", "fallback"]

