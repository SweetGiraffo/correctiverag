"""
Unified Multimodal Ingestion Pipeline.
Coordinates loaders, entity extraction, cross-modal resolution, vector indexing in Qdrant,
and graph persistence in NetworkX.
"""

from typing import List, Dict, Any, Optional
import os
from .base import BaseLoader, IngestedChunk, Modality
from .text_loader import TextLoader
from .pdf_loader import PDFLoader
from .image_loader import ImageLoader
from .audio_loader import AudioLoader
from .entity_extractor import EntityExtractor
from .entity_resolver import EntityResolver
from src.storage.vector_store import QdrantStore
from src.storage.graph_store import MultimodalGraphStore
from src.retrieval.vector_retriever import EmbeddingModelCache
from src.config import get_current_config


class IngestionPipeline:
    """
    Orchestrates end-to-end ingestion across all supported modalities.
    """

    def __init__(
        self,
        vector_store: Optional[QdrantStore] = None,
        graph_store: Optional[MultimodalGraphStore] = None,
        similarity_threshold: float = 0.90,
        embedding_model_name: str = "all-MiniLM-L6-v2"
    ):
        self.embedding_model_name = embedding_model_name
        self.vector_store = vector_store or QdrantStore(vector_dim=384)
        self.graph_store = graph_store or MultimodalGraphStore()

        self.text_loader = TextLoader()
        self.pdf_loader = PDFLoader()
        self.image_loader = ImageLoader()
        self.audio_loader = AudioLoader()

        self.extractor = EntityExtractor()
        self.resolver = EntityResolver(
            similarity_threshold=similarity_threshold,
            model_name=self.embedding_model_name
        )

    def get_loader_for_file(self, file_path: str) -> Optional[BaseLoader]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".txt", ".md"]:
            return self.text_loader
        elif ext in [".pdf"]:
            return self.pdf_loader
        elif ext in [".png", ".jpg", ".jpeg"]:
            return self.image_loader
        elif ext in [".mp3", ".wav", ".m4a"]:
            return self.audio_loader
        return None

    def ingest_file(self, file_path: str) -> List[IngestedChunk]:
        """Ingests a single file through the multimodal pipeline."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        loader = self.get_loader_for_file(file_path)
        if not loader:
            raise ValueError(f"Unsupported file format: {file_path}")

        # Step 1: Modality-specific parsing
        chunks = loader.load(file_path)
        if not chunks:
            return []

        # Step 2: Compute vector embeddings for all chunks
        model = EmbeddingModelCache.get_model(self.embedding_model_name)
        texts_to_embed = [c.text for c in chunks]
        embeddings = model.encode(texts_to_embed, convert_to_numpy=True, normalize_embeddings=True)

        # Step 3: Entity Extraction & Cross-Modal Resolution
        graph = self.graph_store.get_graph()
        for idx, chunk in enumerate(chunks):
            extraction = self.extractor.extract(chunk.text)
            chunk.entities = [e.name for e in extraction.entities]
            chunk.triples = [(t.subject, t.relation, t.object) for t in extraction.triples]
            chunk.embedding = embeddings[idx].tolist()

            # Incrementally update NetworkX graph
            self.resolver.update_graph_with_chunk(
                graph=graph,
                chunk=chunk,
                extracted_entities=extraction.entities,
                extracted_triples=extraction.triples
            )

        # Step 4: Index chunks in Qdrant Vector Store
        self.vector_store.add_chunks(chunks, embeddings.tolist())

        # Step 5: Persist updated graph
        self.graph_store.save()

        return chunks

    def ingest_directory(self, dir_path: str) -> Dict[str, Any]:
        """Recursively ingests all supported files in a directory."""
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        total_files = 0
        total_chunks = 0
        processed_files = []

        for root, _, files in os.walk(dir_path):
            for file in files:
                fpath = os.path.join(root, file)
                if self.get_loader_for_file(fpath):
                    try:
                        chunks = self.ingest_file(fpath)
                        total_files += 1
                        total_chunks += len(chunks)
                        processed_files.append(file)
                    except Exception as e:
                        print(f"Error ingesting {file}: {e}")

        stats = self.graph_store.get_stats()
        return {
            "files_ingested_count": total_files,
            "chunks_created_count": total_chunks,
            "processed_files": processed_files,
            "qdrant_total_chunks": self.vector_store.count(),
            "graph_stats": stats
        }
