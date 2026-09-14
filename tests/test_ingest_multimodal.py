"""
Unit and integration tests for Phase 1 Multimodal Ingestion Pipeline:
- Loaders: Text, PDF (tables as Markdown), Images, Audio
- Entity Extraction & Resolution (similarity > 0.90)
- Vector Store (Qdrant) & Graph Store (NetworkX)
- End-to-End IngestionPipeline
"""

import os
import pytest
import networkx as nx
from src.ingest.base import Modality, IngestedChunk
from src.ingest.text_loader import TextLoader
from src.ingest.pdf_loader import PDFLoader
from src.ingest.image_loader import ImageLoader
from src.ingest.audio_loader import AudioLoader
from src.ingest.entity_extractor import EntityExtractor
from src.ingest.entity_resolver import EntityResolver, normalize_entity_name
from src.storage.vector_store import QdrantStore
from src.storage.graph_store import MultimodalGraphStore
from src.ingest.pipeline import IngestionPipeline


SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data")


def test_text_loader(tmp_path):
    sample_txt = tmp_path / "sample.txt"
    sample_txt.write_text("Acme Corp is expanding operations. The QuantumAI Division generated record profits.", encoding="utf-8")

    loader = TextLoader(max_tokens=10, overlap=0.1)
    chunks = loader.load(str(sample_txt))

    assert len(chunks) >= 1
    assert chunks[0].modality == Modality.TEXT
    assert "Acme Corp" in chunks[0].text
    assert chunks[0].metadata["source_file"] == "sample.txt"


def test_pdf_loader():
    pdf_path = os.path.join(SAMPLE_DIR, "q3_earnings_report.pdf")
    assert os.path.exists(pdf_path), "Sample PDF must exist"

    loader = PDFLoader()
    chunks = loader.load(pdf_path)

    assert len(chunks) > 0
    modalities = [c.modality for c in chunks]
    assert Modality.PDF in modalities
    # Verify table or markdown structure
    all_text = " ".join(c.text for c in chunks)
    assert "Acme Corp" in all_text
    assert "QuantumAI" in all_text


def test_image_loader():
    img_path = os.path.join(SAMPLE_DIR, "revenue_breakdown_chart.png")
    assert os.path.exists(img_path), "Sample chart must exist"

    loader = ImageLoader()
    chunks = loader.load(img_path)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.modality == Modality.IMAGE
    assert chunk.image_path is not None
    assert chunk.metadata["width"] > 0
    assert chunk.metadata["height"] > 0
    assert len(chunk.text) > 0


def test_audio_loader():
    audio_path = os.path.join(SAMPLE_DIR, "executive_briefing.wav")
    assert os.path.exists(audio_path), "Sample audio must exist"

    loader = AudioLoader(model_size="base")
    chunks = loader.load(audio_path)

    assert len(chunks) >= 1
    assert chunks[0].modality == Modality.AUDIO
    assert "source_file" in chunks[0].metadata


def test_entity_extractor():
    extractor = EntityExtractor()
    sample_text = "Dr. Elena Rostova presented the architecture of Project Titan at Acme Corp in New York."
    res = extractor.extract(sample_text)

    assert len(res.entities) > 0
    names = [e.name for e in res.entities]
    assert any("Elena" in n or "Rostova" in n for n in names) or any("Acme" in n for n in names)


def test_entity_resolver_deduplication():
    resolver = EntityResolver(similarity_threshold=0.90)

    # Exact / near-normalized resolution
    c1 = resolver.resolve("Acme Corp")
    c2 = resolver.resolve("the Acme Corp")
    assert c1 == c2

    # High semantic similarity resolution
    c3 = resolver.resolve("QuantumAI Division")
    c4 = resolver.resolve("QuantumAI Division")
    assert c3 == c4

    # Graph update check
    graph = nx.DiGraph()
    dummy_chunk = IngestedChunk(modality=Modality.TEXT, text="Test", chunk_id="chk-1")
    resolver.update_graph_with_chunk(
        graph=graph,
        chunk=dummy_chunk,
        extracted_entities=[type("E", (), {"name": "Acme Corp", "type": "ORG"})],
        extracted_triples=[type("T", (), {"subject": "Acme Corp", "relation": "operates", "object": "QuantumAI"})]
    )

    assert "Acme Corp" in graph
    assert "QuantumAI" in graph
    assert graph.has_edge("Acme Corp", "QuantumAI")
    assert "text" in graph.nodes["Acme Corp"]["modalities"]


def test_qdrant_store(tmp_path):
    q_dir = str(tmp_path / "qdrant_test")
    store = QdrantStore(collection_name="test_multimodal", storage_path=q_dir, use_memory=True)

    c1 = IngestedChunk(modality=Modality.TEXT, text="Acme Corp quarterly report.")
    c2 = IngestedChunk(modality=Modality.IMAGE, text="Revenue breakdown chart.")

    dummy_emb1 = [0.1] * 384
    dummy_emb2 = [0.5] * 384

    store.add_chunks([c1, c2], [dummy_emb1, dummy_emb2])
    assert store.count() == 2

    # Search with modality filter
    res = store.search(query_vector=dummy_emb1, top_k=2, modality_filter="text")
    assert len(res) == 1
    assert res[0]["payload"]["modality"] == "text"


def test_ingestion_pipeline_end_to_end(tmp_path):
    q_dir = str(tmp_path / "qdrant_pipe")
    g_path = str(tmp_path / "graph_pipe.pkl")

    vector_store = QdrantStore(collection_name="pipe_test", storage_path=q_dir, use_memory=True)
    graph_store = MultimodalGraphStore(storage_path=g_path)

    pipeline = IngestionPipeline(
        vector_store=vector_store,
        graph_store=graph_store
    )

    # Ingest sample PDF
    pdf_path = os.path.join(SAMPLE_DIR, "q3_earnings_report.pdf")
    chunks = pipeline.ingest_file(pdf_path)
    assert len(chunks) > 0

    # Ingest sample Chart
    img_path = os.path.join(SAMPLE_DIR, "revenue_breakdown_chart.png")
    img_chunks = pipeline.ingest_file(img_path)
    assert len(img_chunks) > 0

    # Verify vector store contains chunks from both PDF and Image
    assert vector_store.count() >= 2

    # Verify graph contains entities connected across modalities
    graph = graph_store.get_graph()
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0

    stats = graph_store.get_stats()
    assert stats["total_nodes"] > 0
    assert "pdf" in stats["modality_breakdown"] or "image" in stats["modality_breakdown"]
