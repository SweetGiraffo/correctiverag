"""
Unit tests for multimodal evaluation benchmark harness and report formatting.
"""

import pytest
from src.evaluation.multimodal_benchmark import (
    MultimodalBenchmarkRunner,
    MultimodalBenchmarkReport,
    ArchitectureStats,
    BENCHMARK_QUERIES
)
from src.storage.vector_store import QdrantStore
from src.storage.graph_store import MultimodalGraphStore
from src.ingest.base import Modality, IngestedChunk


def test_benchmark_queries_structure():
    assert len(BENCHMARK_QUERIES) >= 4
    for q in BENCHMARK_QUERIES:
        assert q.query != ""
        assert len(q.target_modalities) >= 2
        assert len(q.target_facts) >= 2


def test_multimodal_benchmark_report_markdown():
    stats_hybrid = ArchitectureStats(
        name="Hybrid Vector",
        modality_recall_pct=65.0,
        fact_coverage_pct=50.0,
        citation_precision_pct=0.0,
        avg_latency_ms=120.0
    )
    stats_ppr = ArchitectureStats(
        name="PPR Graph",
        modality_recall_pct=85.0,
        fact_coverage_pct=75.0,
        citation_precision_pct=0.0,
        avg_latency_ms=25.0
    )
    stats_crag = ArchitectureStats(
        name="Full CRAG Agent",
        modality_recall_pct=90.0,
        fact_coverage_pct=80.0,
        citation_precision_pct=95.0,
        avg_latency_ms=85.0
    )

    report = MultimodalBenchmarkReport(
        num_queries=1,
        hybrid_vector_baseline=stats_hybrid,
        ppr_graph_retriever=stats_ppr,
        full_crag_agent=stats_crag,
        detailed_results=[{
            "query": "Test Query",
            "description": "Test description",
            "target_modalities": ["pdf", "image"],
            "target_facts": ["fact1", "fact2"],
            "hybrid": {"modalities": ["pdf"], "facts_found": 1, "citations_count": 0, "latency_ms": 100.0},
            "ppr": {"modalities": ["pdf", "image"], "facts_found": 2, "citations_count": 0, "latency_ms": 20.0},
            "crag": {"modalities": ["pdf", "image"], "facts_found": 2, "citations_count": 2, "latency_ms": 80.0, "answer": "Test answer [doc.pdf]"}
        }]
    )

    md = report.to_markdown()
    assert "# 📊 Cross-Modal Multi-Hop RAG Benchmark Report" in md
    assert "PPR Graph" in md
    assert "85.0%" in md
    assert "Test Query" in md
