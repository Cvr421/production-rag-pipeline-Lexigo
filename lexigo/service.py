from typing import Any
from types import SimpleNamespace

from openai import OpenAI

from .config import Settings, get_settings
from .graph import build_qa_graph
from .pinecone_store import PineconeStore


class CerebrasChatClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if not self.settings.cerebras_api_key:
            raise RuntimeError("CEREBRAS_API_KEY is not configured")
        self.client = OpenAI(
            api_key=self.settings.cerebras_api_key,
            base_url=self.settings.cerebras_base_url,
        )
    SYSTEM_PROMPT = """
You are a document-grounded question-answering assistant.

Use ONLY the document excerpts supplied by the user.

Rules:
- Answer the user's question directly.
- Answer every part of the question supported by the excerpts.
- Extract the specific facts requested.
- Do not merely repeat document titles, headings, or metadata.
- Do not omit relevant facts present in the excerpts.
- Do not use outside knowledge.
- Do not invent facts.
- If the supplied excerpts do not contain enough information, say so.
- Keep answers concise and factual.
"""
    def invoke(self, prompt: str) -> SimpleNamespace:
        try:
            response = self.client.chat.completions.create(
                model=self.settings.chat_model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.settings.chat_temperature,
            )
        except Exception as exc:
            raise RuntimeError(f"Cerebras chat request failed: {exc}") from exc
        content = response.choices[0].message.content or ""
        return SimpleNamespace(content=content)


class QAService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._graph = None

    def _get_graph(self):
        if self._graph is None:
            if not self.settings.cerebras_api_key:
                raise RuntimeError("CEREBRAS_API_KEY is not configured")
            store = PineconeStore(self.settings)
            llm = CerebrasChatClient(self.settings)
            self._graph = build_qa_graph(store, llm, self.settings)
        return self._graph

    def ask(self, question: str, include_trace: bool = False) -> dict[str, Any]:
        graph = self._get_graph()
        result = graph.invoke(
            {
                "question": question,
                "trace": [],
                "step_count": 0,
                "max_steps": self.settings.max_graph_steps,
            },
            config={"recursion_limit": self.settings.max_graph_steps + 2},
        )
        return {
            "answer": result["answer"],
            "grounded": result.get("grounded", False),
            "citations": result.get("citations", []),
            "trace": result.get("trace") if include_trace else None,
        }

