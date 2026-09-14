"""
Base data contracts and loader abstractions for Multimodal Ingestion.
"""

from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from abc import ABC, abstractmethod
import uuid
from pydantic import BaseModel, Field


class Modality(str, Enum):
    TEXT = "text"
    PDF = "pdf"
    TABLE = "table"
    IMAGE = "image"
    AUDIO = "audio"


class IngestedChunk(BaseModel):
    """
    Standardized atomic chunk produced across all modalities.
    Carries textual representation, metadata, modality tag, and extracted knowledge.
    """
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    modality: Modality
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    image_path: Optional[str] = None
    embedding: Optional[List[float]] = None
    entities: List[str] = Field(default_factory=list)
    triples: List[Tuple[str, str, str]] = Field(default_factory=list)

    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Converts chunk metadata and content into a Qdrant-compatible dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "modality": self.modality.value,
            "text": self.text,
            "metadata": self.metadata,
            "image_path": self.image_path,
            "entities": self.entities,
            "triples": self.triples,
        }


class BaseLoader(ABC):
    """Abstract base class for modality-specific document loaders."""

    @abstractmethod
    def load(self, file_path: str) -> List[IngestedChunk]:
        """Loads and processes a file into a list of IngestedChunks."""
        pass

    @staticmethod
    def chunk_text(text: str, max_tokens: int = 512, overlap: float = 0.1) -> List[str]:
        """
        Splits text into chunks of roughly max_tokens words with specified overlap percentage.
        """
        words = text.split()
        if not words:
            return []

        step = max(1, int(max_tokens * (1.0 - overlap)))
        chunks = []
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + max_tokens])
            if chunk.strip():
                chunks.append(chunk)
            if i + max_tokens >= len(words):
                break
        return chunks
