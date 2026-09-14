"""
Multimodal Ingestion Package:
- BaseLoader, IngestedChunk, Modality
- TextLoader, PDFLoader, ImageLoader, AudioLoader
- EntityExtractor, EntityResolver
- IngestionPipeline
"""

from .base import Modality, IngestedChunk, BaseLoader
from .text_loader import TextLoader
from .pdf_loader import PDFLoader
from .image_loader import ImageLoader
from .audio_loader import AudioLoader
from .entity_extractor import EntityExtractor, ExtractionResult, ExtractedEntity, ExtractedTriple
from .entity_resolver import EntityResolver, normalize_entity_name
from .pipeline import IngestionPipeline

__all__ = [
    "Modality",
    "IngestedChunk",
    "BaseLoader",
    "TextLoader",
    "PDFLoader",
    "ImageLoader",
    "AudioLoader",
    "EntityExtractor",
    "ExtractionResult",
    "ExtractedEntity",
    "ExtractedTriple",
    "EntityResolver",
    "normalize_entity_name",
    "IngestionPipeline",
]
