"""
Unit tests for Entity Graph Builder and Personalized PageRank retrieval.
"""

import os
import pytest
from src.dataset import load_dataset
from src.graph_builder import EntityGraphBuilder
from src.retrieval.ppr_retriever import PPRRetriever
from src.retrieval.vector_retriever import SequentialVectorRetriever


@pytest.fixture
def sample_data():
    sample_path = os.path.join("data", "2wikimultihopqa_sample.json")
    samples = load_dataset(sample_path, max_samples=5)
    return samples[0]


def test_entity_graph_builder(sample_data):
    builder = EntityGraphBuilder()
    G = builder.build_graph_for_sample(sample_data)

    assert len(G.nodes) > 0
    assert len(G.edges) > 0
    assert "entity_to_passages" in G.graph
    assert "passage_to_entities" in G.graph

    # Verify evidence nodes exist in graph
    for ev in sample_data.evidences:
        assert ev.subject in G
        assert ev.object in G
        assert G.has_edge(ev.subject, ev.object)


def test_ppr_retriever(sample_data):
    builder = EntityGraphBuilder()
    G = builder.build_graph_for_sample(sample_data)

    retriever = PPRRetriever(alpha=0.85)
    result = retriever.retrieve(
        query=sample_data.question,
        graph=G,
        passages=sample_data.context_passages,
        top_k=3
    )

    assert len(result.passages) == 3
    assert len(result.scores) == 3
    assert result.vector_search_calls == 0
    assert result.execution_time_ms > 0
    # Seed entities should be identified
    assert len(result.seed_entities) > 0


def test_vector_retriever(sample_data):
    retriever = SequentialVectorRetriever()
    result = retriever.retrieve(
        query=sample_data.question,
        passages=sample_data.context_passages,
        top_k=3,
        hops=2
    )

    assert len(result.passages) == 3
    assert len(result.scores) == 3
    assert result.vector_search_calls >= 2
    assert len(result.hop_queries) == 2
