"""
Data models for Multimodal Retrieval outputs.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from src.ingest.base import Modality


class RetrievedChunk(BaseModel):
    chunk_id: str
    modality: Modality
    text: str
    score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    image_path: Optional[str] = None
    image_base64: Optional[str] = None  # Base64 encoded bytes for downstream VLM
    entities: List[str] = Field(default_factory=list)

    def is_visual(self) -> bool:
        return self.modality == Modality.IMAGE and bool(self.image_base64 or self.image_path)


class MultimodalRetrievalResult(BaseModel):
    query: str
    top_nodes: List[Dict[str, Any]] = Field(default_factory=list)
    chunks: List[RetrievedChunk] = Field(default_factory=list)
    seed_entities: List[str] = Field(default_factory=list)
    execution_time_ms: float = 0.0
    algorithm: str = "Multimodal PPR"
