import re
from time import sleep
from typing import Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from .config import Settings, get_settings


class GraphState(TypedDict, total=False):
    question: str
    chunks: list[dict[str, Any]]
    retrieval_ok: bool
    failure_reason: str
    answer: str
    grounded: bool
    citations: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    step_count: int
    max_steps: int


_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "by",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "may",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def _terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in _STOP_WORDS
    }


def _trace(
    state: GraphState,
    node: str,
    details: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        *state.get("trace", []),
        {
            "node": node,
            **details,
        },
    ]


def _step(state: GraphState) -> int:
    return state.get("step_count", 0) + 1


def _citation(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": chunk["chunk_id"],
        "source_file": chunk["source_file"],
        "chunk_index": chunk["chunk_index"],
        "score": round(float(chunk.get("score", 0.0)), 6),
        "excerpt": chunk["text"][:280].strip(),
    }


# def _extractive_answer(chunks: list[dict[str, Any]]) -> str:
#     if not chunks:
#         return "I cannot find that information in the provided documents."

#     text = str(chunks[0].get("text", "")).strip()

#     if not text:
#         return "I cannot find that information in the provided documents."

#     text = re.sub(r"^#+\s\*", "", text)
#     text = re.sub(r"\n{3,}", "\n\n", text).strip()

#     lines = [
#         line.strip()
#         for line in text.splitlines()
#         if line.strip()
#     ]

#     snippet = " ".join(lines[:3]) if lines else text

#     if len(snippet) < 80:
#         sentences = re.split(
#             r"(?<=[.!?])\s+",
#             text,
#         )

#         snippet = " ".join(
#             sentence.strip()
#             for sentence in sentences[:2]
#             if sentence.strip()
#         )

#     if not snippet:
#         snippet = text

#     if len(snippet) > 500:
#         snippet = snippet[:500].rstrip() + "..."

#     return snippet


def _best_chunk(
    question: str,
    chunks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    question_terms = _terms(question)

    best_chunk: dict[str, Any] | None = None
    best_score = -1.0

    for chunk in chunks:
        chunk_terms = _terms(
            str(chunk.get("text", ""))
        )

        overlap = len(
            question_terms & chunk_terms
        ) / max(len(question_terms), 1)

        score = (
            float(chunk.get("score", 0.0))
            + overlap
        )

        if score > best_score:
            best_score = score
            best_chunk = chunk

    return best_chunk


def build_qa_graph(
    store: Any,
    llm: BaseChatModel,
    settings: Settings | None = None,
):
    settings = settings or get_settings()

    def _search_with_retry(
        question: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        best_chunks: list[dict[str, Any]] = []
        best_score = -1.0

        for attempt in range(3):
            chunks = store.search(
                question,
                top_k=top_k,
            )

            top_score = (
                float(chunks[0].get("score", 0.0))
                if chunks
                else 0.0
            )

            if top_score > best_score:
                best_chunks = chunks
                best_score = top_score

            if top_score >= settings.min_relevance_score:
                return chunks

            if attempt < 2:
                sleep(1)

        return best_chunks

    def retrieve_node(state: GraphState) -> GraphState:
        step_count = _step(state)

        if step_count > state.get(
            "max_steps",
            settings.max_graph_steps,
        ):
            return {
                "step_count": step_count,
                "retrieval_ok": False,
                "failure_reason": "graph step limit reached",
                "trace": _trace(
                    state,
                    "retrieve",
                    {
                        "status": "step_limit",
                    },
                ),
            }

        chunks = _search_with_retry(
            state["question"],
            settings.top_k,
        )

        return {
            "chunks": chunks,
            "step_count": step_count,
            "trace": _trace(
                state,
                "retrieve",
                {
                    "matches": len(chunks),
                },
            ),
        }

    def assess_node(state: GraphState) -> GraphState:
        step_count = _step(state)

        chunks = state.get(
            "chunks",
            [],
        )

        question_terms = _terms(
            state["question"]
        )

        evidence_terms = _terms(
            " ".join(
                chunk.get("text", "")
                for chunk in chunks[:3]
            )
        )

        overlap = len(
            question_terms & evidence_terms
        ) / max(len(question_terms), 1)

        top_score = (
            float(chunks[0].get("score", 0.0))
            if chunks
            else 0.0
        )

        enough = bool(
            chunks
            and top_score >= settings.min_relevance_score
            and overlap >= settings.min_query_term_overlap
            and step_count
            <= state.get(
                "max_steps",
                settings.max_graph_steps,
            )
        )

        reason = (
            "relevant evidence found"
            if enough
            else "retrieved evidence is insufficient"
        )

        return {
            "retrieval_ok": enough,
            "failure_reason": ""
            if enough
            else reason,
            "step_count": step_count,
            "trace": _trace(
                state,
                "assess",
                {
                    "route": (
                        "answer"
                        if enough
                        else "fallback"
                    ),
                    "top_score": round(
                        top_score,
                        6,
                    ),
                    "term_overlap": round(
                        overlap,
                        6,
                    ),
                },
            ),
        }

    def answer_node(state: GraphState) -> GraphState:
        step_count = _step(state)

        if step_count > state.get(
            "max_steps",
            settings.max_graph_steps,
        ):
            return fallback_node(
                {
                    **state,
                    "step_count": step_count,
                }
            )

        chunks = state.get(
            "chunks",
            [],
        )

        if not chunks:
            return fallback_node(
                {
                    **state,
                    "step_count": step_count,
                    "failure_reason": (
                        "no retrieved evidence"
                    ),
                }
            )

        # Use the top 3 retrieved chunks as context
        # so the LLM can combine information from
        # multiple relevant pieces of evidence.
        context_parts = []

        for chunk in chunks[:3]:
            context_parts.append(
                f"""Source: {chunk["source_file"]}
Chunk: {chunk["chunk_index"]}

{chunk["text"]}"""
            )

        context = "\n\n---\n\n".join(
            context_parts
        )

        prompt = f"""
Answer the user's question using ONLY the retrieved
document excerpts.

QUESTION:
{state["question"]}

RETRIEVED DOCUMENT EXCERPTS:
{context}

INSTRUCTIONS:
- Answer the question directly.
- Identify all factual elements requested by the question before answering.
- Answer every part of the question supported by the excerpts.
- Include all directly relevant facts needed to satisfy the question.
- Extract the specific facts requested by the user.
- Do not merely repeat document titles or metadata.
- Do not omit relevant facts that appear in the retrieved excerpts.
- Do not use outside knowledge.
- Do not invent facts.
- If the excerpts do not contain enough information, say that the information is not available.
- Keep the answer concise and factual.
"""

        try:
            response = llm.invoke(prompt)
            answer = response.content.strip()

        except Exception as exc:
            raise RuntimeError(
                f"Answer generation failed: {exc}"
            ) from exc

        return {
            "answer": answer,
            "grounded": True,
            "citations": [
                _citation(chunk)
                for chunk in chunks[:3]
            ],
            "step_count": step_count,
            "trace": _trace(
                state,
                "answer",
                {
                    "citation_count": min(
                        len(chunks),
                        3,
                    ),
                    "llm": "cerebras",
                },
            ),
        }

    def fallback_node(state: GraphState) -> GraphState:
        step_count = (
            state.get("step_count", 0) + 1
        )

        return {
            "answer": (
                "I cannot find that information "
                "in the provided documents."
            ),
            "grounded": False,
            "citations": [],
            "step_count": step_count,
            "trace": _trace(
                state,
                "fallback",
                {
                    "reason": state.get(
                        "failure_reason",
                        "insufficient evidence",
                    ),
                },
            ),
        }

    def route_after_assessment(
        state: GraphState,
    ) -> str:
        return (
            "answer"
            if state.get(
                "retrieval_ok",
                False,
            )
            else "fallback"
        )

    graph = StateGraph(GraphState)

    graph.add_node(
        "retrieve",
        retrieve_node,
    )

    graph.add_node(
        "assess",
        assess_node,
    )

    graph.add_node(
        "answer",
        answer_node,
    )

    graph.add_node(
        "fallback",
        fallback_node,
    )

    graph.add_edge(
        START,
        "retrieve",
    )

    graph.add_edge(
        "retrieve",
        "assess",
    )

    graph.add_conditional_edges(
        "assess",
        route_after_assessment,
        {
            "answer": "answer",
            "fallback": "fallback",
        },
    )

    graph.add_edge(
        "answer",
        END,
    )

    graph.add_edge(
        "fallback",
        END,
    )

    return graph.compile()