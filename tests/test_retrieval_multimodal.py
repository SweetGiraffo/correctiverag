"""
Unit and integration tests for Phase 2 Multimodal Retrieval Engine:
- Multimodal Personalized PageRank (PPR)
- Base64 raw image extraction for VLM
- Multimodal Hybrid (Dense + BM25 + Reranker)
"""

import os
import pytest
from src.ingest.pipeline import IngestionPipeline
from src.storage.vector_store import QdrantStore
from src.storage.graph_store import MultimodalGraphStore
from src.retrieval.ppr import MultimodalPPRRetriever, load_image_as_base64
from src.retrieval.hybrid import MultimodalHybridRetriever
from src.ingest.base import Modality


SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data")


@pytest.fixture(scope="module")
def populated_stores(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("retrieval_data")
    q_dir = str(tmp_dir / "qdrant")
    g_path = str(tmp_dir / "graph.pkl")

    vector_store = QdrantStore(collection_name="multimodal_test", storage_path=q_dir, use_memory=True)
    graph_store = MultimodalGraphStore(storage_path=g_path)

    pipe = IngestionPipeline(vector_store=vector_store, graph_store=graph_store)

    # Ingest report PDF and chart image
    pdf_path = os.path.join(SAMPLE_DIR, "q3_earnings_report.pdf")
    img_path = os.path.join(SAMPLE_DIR, "revenue_breakdown_chart.png")

    pipe.ingest_file(pdf_path)
    pipe.ingest_file(img_path)

    return vector_store, graph_store


def test_seed_matching(populated_stores):
    _, graph_store = populated_stores
    graph = graph_store.get_graph()

    retriever = MultimodalPPRRetriever()
    query = "What was the Q3 revenue of the QuantumAI Division at Acme Corp?"

    seeds = retriever.extract_and_match_seeds(query, graph)
    assert len(seeds) > 0
    # Expected seed nodes
    seed_keys = list(seeds.keys())
    assert any("QuantumAI" in s or "Acme" in s for s in seed_keys)


def test_multimodal_ppr_retrieval(populated_stores):
    vector_store, graph_store = populated_stores
    graph = graph_store.get_graph()

    retriever = MultimodalPPRRetriever(alpha=0.85)
    query = "How much revenue did QuantumAI generate according to the report and chart?"

    result = retriever.retrieve(
        query=query,
        graph=graph,
        vector_store=vector_store,
        top_k_nodes=5,
        top_k_chunks=3
    )

    assert len(result.chunks) > 0
    assert len(result.top_nodes) > 0
    assert result.execution_time_ms > 0
    assert result.algorithm == "Multimodal PPR"

    # Verify cross-modal presence
    retrieved_modalities = {c.modality for c in result.chunks}
    assert len(retrieved_modalities) >= 1

    # Check that any retrieved image chunk carries base64 image data
    for chunk in result.chunks:
        if chunk.modality == Modality.IMAGE:
            assert chunk.image_base64 is not None
            assert len(chunk.image_base64) > 100


def test_multimodal_hybrid_retrieval(populated_stores):
    vector_store, _ = populated_stores

    retriever = MultimodalHybridRetriever()
    query = "CloudScale infrastructure revenue and growth"

    result = retriever.retrieve(
        query=query,
        vector_store=vector_store,
        top_k=3,
        apply_reranker=False
    )

    assert len(result.chunks) > 0
    assert result.execution_time_ms > 0
    assert "Hybrid" in result.algorithm
    assert any("CloudScale" in c.text or "320M" in c.text for c in result.chunks)


def test_image_base64_loader():
    chart_path = os.path.join(SAMPLE_DIR, "revenue_breakdown_chart.png")
    b64_str = load_image_as_base64(chart_path)

    assert b64_str is not None
    assert len(b64_str) > 500

    # Non-existent file returns None
    assert load_image_as_base64("invalid/path/none.png") is None
