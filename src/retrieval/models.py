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

    def to_passage(self) -> Any:
        from src.dataset import Passage
        title = self.metadata.get("source_file") or f"[{self.modality.value.upper()}] {self.chunk_id[:8]}"
        sentences = [s.strip() for s in self.text.split("\n") if s.strip()]
        if not sentences:
            sentences = [self.text]
        return Passage(title=title, sentences=sentences, text=self.text)

    def format_citation(self) -> str:
        source = self.metadata.get("source_file") or "Document"
        if self.modality == Modality.PDF or self.modality == Modality.TABLE:
            page = self.metadata.get("page_number")
            prefix = "Table " if self.modality == Modality.TABLE else ""
            if page is not None:
                return f"[{source}, {prefix}p. {page}]"
            return f"[{source}]"
        elif self.modality == Modality.AUDIO:
            start = self.metadata.get("start_time")
            end = self.metadata.get("end_time")
            if start is not None and end is not None:
                s_m, s_s = divmod(int(start), 60)
                e_m, e_s = divmod(int(end), 60)
                return f"[{source}, {s_m:02d}:{s_s:02d} - {e_m:02d}:{e_s:02d}]"
            return f"[{source}, Audio]"
        elif self.modality == Modality.IMAGE:
            return f"[{source}, Image]"
        return f"[{source}]"


class MultimodalRetrievalResult(BaseModel):
    query: str
    top_nodes: List[Dict[str, Any]] = Field(default_factory=list)
    chunks: List[RetrievedChunk] = Field(default_factory=list)
    seed_entities: List[str] = Field(default_factory=list)
    execution_time_ms: float = 0.0
    algorithm: str = "Multimodal PPR"
