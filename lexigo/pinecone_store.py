import re
from typing import Any
from time import sleep

from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import NotFoundException

from .config import Settings, get_settings
from .documents import DocumentChunk, load_markdown_documents
from .embeddings import get_embedding_model


class PineconeStore:
    """Small adapter that keeps Pinecone concerns out of the graph nodes."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is not configured")
        self.client = Pinecone(api_key=self.settings.pinecone_api_key)
        self.index = None
        self.embeddings = get_embedding_model(self.settings)
        self._local_chunks: list[DocumentChunk] | None = None

    def _fresh_client(self) -> Pinecone:
        self.client = Pinecone(api_key=self.settings.pinecone_api_key)
        return self.client

    def ensure_index(self) -> None:
        try:
            description = self._fresh_client().describe_index(self.settings.pinecone_index_name)
        except Exception as exc:
            if "not found" not in str(exc).lower() and "404" not in str(exc):
                raise
            self._fresh_client().create_index(
                name=self.settings.pinecone_index_name,
                dimension=self.settings.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self.settings.pinecone_cloud,
                    region=self.settings.pinecone_region,
                ),
            )
            description = self._fresh_client().describe_index(self.settings.pinecone_index_name)

            for _ in range(30):
                status = (
                    description.get("status", {})
                    if isinstance(description, dict)
                    else getattr(description, "status", {})
                )
                ready = (
                    status.get("ready", False)
                    if isinstance(status, dict)
                    else getattr(status, "ready", False)
                )
                if ready:
                    break
                sleep(2)
                description = self._fresh_client().describe_index(self.settings.pinecone_index_name)

        host = (
            description.get("host")
            if isinstance(description, dict)
            else getattr(description, "host", None)
        )
        dimension = (
            description.get("dimension")
            if isinstance(description, dict)
            else getattr(description, "dimension", None)
        )
        if dimension is not None and int(dimension) != self.settings.embedding_dimension:
            raise ValueError(
                f"Pinecone index dimension is {dimension}, "
                f"expected {self.settings.embedding_dimension} "
                f"for {self.settings.embedding_model}"
            )
        self.index = (
            self._fresh_client().Index(host=host)
            if host
            else self._fresh_client().Index(self.settings.pinecone_index_name)
        )

    def _require_index(self) -> Any:
        if self.index is None:
            self.ensure_index()
        return self.index

    def _namespace_vector_count(self) -> int:
        stats = self._require_index().describe_index_stats()
        namespaces = (
            stats.get("namespaces", {})
            if isinstance(stats, dict)
            else getattr(stats, "namespaces", {})
        )
        namespace_stats = (
            namespaces.get(self.settings.pinecone_namespace)
            if isinstance(namespaces, dict)
            else getattr(namespaces, self.settings.pinecone_namespace, None)
        )
        if namespace_stats is None:
            return 0
        if isinstance(namespace_stats, dict):
            return int(namespace_stats.get("vector_count", 0) or 0)
        return int(getattr(namespace_stats, "vector_count", 0) or 0)

    def _local_corpus(self) -> list[DocumentChunk]:
        if self._local_chunks is None:
            self._local_chunks = load_markdown_documents(self.settings.corpus_dir)
        return self._local_chunks

    @staticmethod
    def _keywords(text: str) -> set[str]:
        stop_words = {
            "a", "an", "and", "are", "be", "by", "can", "did", "do", "does", "for",
            "from", "how", "in", "is", "it", "may", "of", "on", "or", "the", "to",
            "was", "what", "when", "where", "which", "who", "why", "with",
        }
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 2 and token not in stop_words
        }

    def _local_search(self, question: str, top_k: int) -> list[dict[str, Any]]:
        question_terms = self._keywords(question)
        scored: list[dict[str, Any]] = []
        for chunk in self._local_corpus():
            chunk_terms = self._keywords(chunk.text)
            overlap = len(question_terms & chunk_terms) / max(len(question_terms), 1)
            if overlap <= 0:
                continue
            scored.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source_file": chunk.source_file,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "score": round(0.25 + overlap * 0.75, 6),
                }
            )
        scored.sort(key=lambda item: (item["score"], item["chunk_index"]), reverse=True)
        return scored[:top_k]

    def wait_for_namespace_visibility(self, expected_count: int, timeout_seconds: int = 60) -> None:
        deadline = timeout_seconds
        elapsed = 0
        while elapsed < deadline:
            if self._namespace_vector_count() >= expected_count:
                return
            sleep(2)
            elapsed += 2

    def reset_namespace(self) -> None:
        self.index = None
        try:
            self._require_index().delete(
                delete_all=True,
                namespace=self.settings.pinecone_namespace,
            )
        except NotFoundException:
            return

    def upsert(self, chunks: list[DocumentChunk], batch_size: int = 100) -> int:
        self.index = None
        index = self._require_index()
        vectors = self.embeddings.embed_documents([chunk.text for chunk in chunks])
        total = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            batch_vectors = vectors[start : start + batch_size]
            payload = [
                {
                    "id": chunk.chunk_id,
                    "values": vector,
                    "metadata": {
                        "chunk_id": chunk.chunk_id,
                        "source_file": chunk.source_file,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                    },
                }
                for chunk, vector in zip(batch, batch_vectors)
            ]
            index.upsert(vectors=payload, namespace=self.settings.pinecone_namespace)
            total += len(payload)
        self.wait_for_namespace_visibility(total)
        return total

    def search(self, question: str, top_k: int | None = None) -> list[dict[str, Any]]:
        self.index = None
        index = self._require_index()
        vector = self.embeddings.embed_query(question)
        result = index.query(
            vector=vector,
            top_k=top_k or self.settings.top_k,
            include_metadata=True,
            namespace=self.settings.pinecone_namespace,
        )
        matches = (
            result.get("matches", [])
            if isinstance(result, dict)
            else getattr(result, "matches", [])
        )
        records: list[dict[str, Any]] = []
        for match in matches:
            metadata = (
                match.get("metadata", {})
                if isinstance(match, dict)
                else getattr(match, "metadata", {})
            )
            score = (
                match.get("score", 0.0)
                if isinstance(match, dict)
                else getattr(match, "score", 0.0)
            )
            records.append(
                {
                    "chunk_id": metadata.get(
                        "chunk_id",
                        match.get("id") if isinstance(match, dict) else getattr(match, "id", ""),
                    ),
                    "source_file": metadata.get("source_file", "unknown"),
                    "chunk_index": int(metadata.get("chunk_index", 0)),
                    "text": metadata.get("text", ""),
                    "score": float(score or 0.0),
                }
            )
        # if records and float(records[0].get("score", 0.0)) >= self.settings.min_relevance_score:
        return records
        # local_records = self._local_search(question, top_k or self.settings.top_k)
        # return local_records or records
