"""
ChromaDB Vector Store Service.
Handles document ingestion and semantic similarity search.
"""

import uuid
import structlog
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb import PersistentClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings

logger = structlog.get_logger(__name__)

COLLECTION_NAME = "music_knowledge"


class VectorStoreService:
    """
    Wraps ChromaDB for semantic document retrieval.

    Why ChromaDB?
    - Runs fully local with no external dependencies.
    - Supports persistent storage so embeddings survive restarts.
    - Native async-compatible via thread pool.
    """

    def __init__(self):
        persist_dir = Path(settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
        )

    async def search(
        self,
        query: str,
        n_results: int = 10,
        doc_type: str | None = None,
    ) -> list[dict]:
        """
        Semantic similarity search against the collection.
        Optionally filter by document type (e.g., 'theory', 'review', 'bio').
        """
        try:
            query_embedding = self._embedder.embed_query(query)
            where = {"doc_type": doc_type} if doc_type else None

            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, max(1, self._collection.count())),
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            output: list[dict] = []
            for i, (doc, meta, dist) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )):
                output.append({
                    "id":      str(hash(doc)),
                    "content": doc,
                    "source":  meta.get("source", "Internal Database"),
                    "score":   1.0 - dist,   # convert distance → similarity
                    "_rank":   i,
                    **meta,
                })
            return output

        except Exception as exc:
            logger.warning("vector_store.search_failed", error=str(exc), query=query[:80])
            return []

    async def ingest_text(
        self,
        text: str,
        source: str,
        doc_type: str = "general",
    ) -> int:
        """
        Chunk and embed a text document into ChromaDB.
        Returns the number of chunks added.
        """
        chunks = self._splitter.split_text(text)
        if not chunks:
            return 0

        embeddings = self._embedder.embed_documents(chunks)
        ids        = [str(uuid.uuid4()) for _ in chunks]
        metadatas  = [{"source": source, "doc_type": doc_type} for _ in chunks]

        self._collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("vector_store.ingested", source=source, chunks=len(chunks))
        return len(chunks)

    def get_collection_stats(self) -> dict:
        """Return collection statistics for monitoring."""
        return {
            "collection": COLLECTION_NAME,
            "count":      self._collection.count(),
        }
