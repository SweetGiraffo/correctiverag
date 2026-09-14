"""
Text and Markdown document loader.
Chunks text into 512 tokens with 10% overlap.
"""

from typing import List
import os
from .base import BaseLoader, IngestedChunk, Modality


class TextLoader(BaseLoader):
    """Loads plain text and markdown documents."""

    def __init__(self, max_tokens: int = 512, overlap: float = 0.1):
        self.max_tokens = max_tokens
        self.overlap = overlap

    def load(self, file_path: str) -> List[IngestedChunk]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        filename = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        raw_chunks = self.chunk_text(content, max_tokens=self.max_tokens, overlap=self.overlap)
        chunks: List[IngestedChunk] = []

        for idx, text in enumerate(raw_chunks):
            chunks.append(
                IngestedChunk(
                    modality=Modality.TEXT,
                    text=text,
                    metadata={
                        "source_file": filename,
                        "file_path": os.path.abspath(file_path),
                        "chunk_index": idx,
                        "total_chunks": len(raw_chunks)
                    }
                )
            )

        return chunks
