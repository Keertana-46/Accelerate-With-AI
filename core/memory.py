"""Vector memory backed by ChromaDB with an in-memory fallback."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from core.config import CHROMA_DIR

_EMBED_DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stable_id(prefix: str, text: str) -> str:
    """Return a deterministic identifier for ``text``."""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _hash_embed(text: str) -> list[float]:
    """Return an offline deterministic bag-of-words hashing embedding.

    This avoids ChromaDB downloading a model on first use, keeping the vector
    store fully offline and hermetic.
    """
    vector = [0.0] * _EMBED_DIM
    for token in _TOKEN_RE.findall(text.lower()):
        bucket = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % _EMBED_DIM
        vector[bucket] += 1.0
    norm = sum(value * value for value in vector) ** 0.5
    if norm > 0:
        vector = [value / norm for value in vector]
    return vector


class _HashingEmbeddingFunction:
    """ChromaDB-compatible offline embedding function."""

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        """Embed a batch of documents deterministically."""
        return [_hash_embed(text) for text in input]

    @staticmethod
    def name() -> str:
        """Return the embedding function name required by ChromaDB."""
        return "hashing_offline"


class _InMemoryCollection:
    """Minimal in-memory stand-in for a Chroma collection."""

    def __init__(self) -> None:
        self._docs: list[dict[str, Any]] = []

    def add(self, ids: list[str], documents: list[str],
            metadatas: list[dict[str, Any]]) -> None:
        """Store documents, skipping duplicate ids."""
        existing = {d["id"] for d in self._docs}
        for doc_id, doc, meta in zip(ids, documents, metadatas):
            if doc_id in existing:
                continue
            self._docs.append({"id": doc_id, "document": doc, "metadata": meta})

    def query(self, query_texts: list[str], n_results: int) -> dict[str, Any]:
        """Return the most lexically similar documents to the query."""
        query = (query_texts[0] if query_texts else "").lower()
        tokens = set(query.split())

        def score(doc: dict[str, Any]) -> int:
            words = set(doc["document"].lower().split())
            return len(tokens & words)

        ranked = sorted(self._docs, key=score, reverse=True)[:n_results]
        return {
            "documents": [[d["document"] for d in ranked]],
            "metadatas": [[d["metadata"] for d in ranked]],
            "ids": [[d["id"] for d in ranked]],
        }


class VectorMemory:
    """Persistent vector memory with graceful degradation.

    Two logical collections are maintained: ``business_intents`` and
    ``reports``. When ChromaDB cannot be initialized the class transparently
    falls back to an in-memory store so callers never need special handling.
    """

    def __init__(self) -> None:
        """Initialize Chroma-backed collections or fall back to memory."""
        self.backend = "chroma"
        try:
            import chromadb  # type: ignore

            embed = _HashingEmbeddingFunction()
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            self._intents = client.get_or_create_collection(
                "business_intents", embedding_function=embed
            )
            self._reports = client.get_or_create_collection(
                "reports", embedding_function=embed
            )
        except Exception:
            self.backend = "memory"
            self._intents = _InMemoryCollection()
            self._reports = _InMemoryCollection()

    def store_intent(self, intent: str, metadata: dict[str, Any] | None = None) -> str:
        """Store a business intent and return its identifier."""
        if not intent:
            raise ValueError("intent must be a non-empty string")
        doc_id = _stable_id("intent", intent)
        meta = dict(metadata) if metadata else {}
        meta.setdefault("kind", "intent")
        self._intents.add(ids=[doc_id], documents=[intent], metadatas=[meta])
        return doc_id

    def store_report(self, summary: str, metadata: dict[str, Any] | None = None) -> str:
        """Store a report summary and return its identifier."""
        if not summary:
            raise ValueError("summary must be a non-empty string")
        doc_id = _stable_id("report", summary)
        meta = dict(metadata) if metadata else {}
        meta.setdefault("kind", "report")
        self._reports.add(ids=[doc_id], documents=[summary], metadatas=[meta])
        return doc_id

    def retrieve_similar_intents(self, query: str, k: int = 3) -> list[str]:
        """Return up to ``k`` intents most similar to ``query``."""
        return self._query(self._intents, query, k)

    def retrieve_similar_reports(self, query: str, k: int = 3) -> list[str]:
        """Return up to ``k`` report summaries most similar to ``query``."""
        return self._query(self._reports, query, k)

    @staticmethod
    def _query(collection: Any, query: str, k: int) -> list[str]:
        """Run a query against ``collection`` and normalize the result."""
        if not query:
            return []
        try:
            result = collection.query(query_texts=[query], n_results=max(1, k))
        except Exception:
            return []
        docs = result.get("documents") or [[]]
        return list(docs[0]) if docs else []
