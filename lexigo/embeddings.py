from typing import Any

from pinecone import Pinecone

from .config import Settings, get_settings


class PineconeInferenceEmbeddings:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if not self.settings.pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is not configured")

    def _embed(self, inputs: list[str], input_type: str) -> list[list[float]]:
        client = Pinecone(api_key=self.settings.pinecone_api_key)
        result = client.inference.embed(
            model=self.settings.embedding_model,
            inputs=inputs,
            parameters={"input_type": input_type, "truncate": "END"},
        )
        data = getattr(result, "data", result)
        vectors: list[list[float]] = []
        for item in data:
            if isinstance(item, dict):
                values = item.get("values", [])
            else:
                values = getattr(item, "values", [])
            vectors.append([float(value) for value in values])
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, input_type="passage")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], input_type="query")[0]


def get_embedding_model(settings: Settings | None = None) -> PineconeInferenceEmbeddings:
    settings = settings or get_settings()
    return PineconeInferenceEmbeddings(settings)
