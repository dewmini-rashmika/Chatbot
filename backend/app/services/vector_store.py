"""
ChromaDB Vector Store Service.
Handles document ingestion and semantic similarity search.
"""
import uuid
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings


class VectorStoreService:
    """
    Wraps ChromaDB for semantic document retrieval.

    Why ChromaDB?
    - Runs fully local with no external dependencies.
    - Supports persistent storage so embeddings survive restarts.
    - Native async-compatible via thread pool.
    """

    COLLECTION_NAME = "music_knowledge"

    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # Cosine similarity for text
        )
        self._embedder = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " "],
        )

    async def ingest_text(self, text: str, source: str, doc_type: str = "general") -> int:
        """
        Chunk and embed a text document into ChromaDB.
        Returns the number of chunks added.
        """
        chunks = self._splitter.split_text(text)
        if not chunks:
            return 0

        ids = [str(uuid.uuid4()) for _ in chunks]
        embeddings = self._embedder.embed_documents(chunks)
        metadatas = [{"source": source, "doc_type": doc_type, "chunk_index": i}
                     for i in range(len(chunks))]

        self._collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    async def search(self, query: str, n_results: int = 10, doc_type: str | None = None) -> list[dict]:
        """
        Semantic similarity search against the collection.
        Optionally filter by document type (e.g., 'theory', 'review', 'bio').
        """
        query_embedding = self._embedder.embed_query(query)

        where = {"doc_type": doc_type} if doc_type else None

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "id": f"chroma_{hash(doc)}",
                "content": doc,
                "source": meta.get("source", "Unknown"),
                "doc_type": meta.get("doc_type", "general"),
                "score": 1 - dist,  # Convert distance to similarity
                "retrieval_method": "vector",
            })

        return output

    def get_collection_stats(self) -> dict:
        """Return collection statistics for monitoring."""
        return {
            "collection": self.COLLECTION_NAME,
            "document_count": self._collection.count(),
            "persist_dir": settings.chroma_persist_dir,
        }
