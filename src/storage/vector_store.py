"""
Qdrant Vector Store wrapper for multimodal chunk indexing.
Supports in-memory, local persistent disk, and remote Docker instances.
"""

from typing import List, Dict, Any, Optional
import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from src.ingest.base import IngestedChunk


class QdrantStore:
    """
    Manages vector storage and hybrid metadata filtering using Qdrant.
    """

    def __init__(
        self,
        collection_name: str = "multimodal_chunks",
        vector_dim: int = 384,
        storage_path: Optional[str] = "data/qdrant_db",
        use_memory: bool = False
    ):
        self.collection_name = collection_name
        self.vector_dim = vector_dim

        # Connect to Qdrant: in-memory, local path, or remote URL
        qdrant_url = os.environ.get("QDRANT_URL")
        if qdrant_url:
            self.client = QdrantClient(url=qdrant_url)
        elif use_memory:
            self.client = QdrantClient(":memory:")
        else:
            os.makedirs(storage_path, exist_ok=True)
            self.client = QdrantClient(path=storage_path)

        self._ensure_collection()

    def _ensure_collection(self):
        """Creates collection if it does not already exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE)
            )

    def add_chunks(self, chunks: List[IngestedChunk], embeddings: List[List[float]]):
        """Inserts chunks and their corresponding embeddings into Qdrant."""
        if not chunks:
            return

        points: List[PointStruct] = []
        for chunk, emb in zip(chunks, embeddings):
            # Qdrant accepts valid UUIDs or integers as point IDs
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            points.append(
                PointStruct(
                    id=point_id,
                    vector=emb,
                    payload=chunk.to_qdrant_payload()
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        modality_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Searches nearest neighbor chunks with optional modality filtering."""
        query_filter = None
        if modality_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="modality",
                        match=MatchValue(value=modality_filter)
                    )
                ]
            )

        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter
        ).points

        results: List[Dict[str, Any]] = []
        for hit in hits:
            results.append({
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            })
        return results

    def count(self) -> int:
        """Returns the number of indexed chunks."""
        try:
            return self.client.count(collection_name=self.collection_name).count
        except Exception:
            return 0

    def clear(self):
        """Deletes and recreates the collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._ensure_collection()

    def close(self):
        """Safely closes client connections."""
        try:
            self.client.close()
        except Exception:
            pass
