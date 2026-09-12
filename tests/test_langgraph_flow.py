"""
Integration tests for LangGraph multi-agent pipeline and loopback flows.
"""

import os
import pytest
from src.dataset import load_dataset
from src.graph_builder import EntityGraphBuilder
from src.agent.graph import run_crag_pipeline
from src.config import update_runtime_config


@pytest.fixture
def sample_multihop():
    sample_path = os.path.join("data", "2wikimultihopqa_sample.json")
    samples = load_dataset(sample_path, max_samples=3)
    return samples[0]


def test_langgraph_standard_ppr_flow(sample_multihop):
    builder = EntityGraphBuilder()
    G = builder.build_graph_for_sample(sample_multihop)

    # Set normal thresholds
    update_runtime_config({"grader_threshold_high": 0.50, "grader_threshold_low": 0.20, "retrieval_algorithm": "ppr"})

    state = run_crag_pipeline(
        query=sample_multihop.question,
        passages=sample_multihop.context_passages,
        graph=G,
        sample_id=sample_multihop.id,
        strategy="ppr"
    )

    assert state.get("final_answer") != ""
    assert len(state.get("retrieved_passages", [])) > 0
    trace = state.get("execution_trace", [])
    assert len(trace) >= 4

    step_names = [s["step_name"] for s in trace]
    assert "Router" in step_names
    assert "Retriever" in step_names
    assert "Grader" in step_names
    assert "Generator" in step_names


def test_langgraph_ambiguous_rewrite_loop(sample_multihop):
    builder = EntityGraphBuilder()
    G = builder.build_graph_for_sample(sample_multihop)

    # Force threshold_high very high to guarantee ambiguous trigger on iteration 1
    update_runtime_config({
        "grader_threshold_high": 0.99,
        "grader_threshold_low": 0.05,
        "max_retries": 1
    })

    state = run_crag_pipeline(
        query=sample_multihop.question,
        passages=sample_multihop.context_passages,
        graph=G,
        sample_id=sample_multihop.id,
        strategy="ppr"
    )

    trace = state.get("execution_trace", [])
    step_names = [s["step_name"] for s in trace]
    # Check that Rewriter was triggered
    assert "Rewriter" in step_names
    assert state.get("retry_count", 0) >= 1
    assert state.get("final_answer") != ""


def test_langgraph_fallback_flow(sample_multihop):
    # Force threshold_low very high to trigger INCORRECT fallback
    update_runtime_config({
        "grader_threshold_high": 0.99,
        "grader_threshold_low": 0.95,
        "max_retries": 1
    })

    state = run_crag_pipeline(
        query=sample_multihop.question,
        passages=sample_multihop.context_passages,
        graph=None,
        sample_id=sample_multihop.id,
        strategy="ppr"
    )

    trace = state.get("execution_trace", [])
    step_names = [s["step_name"] for s in trace]
    assert "Fallback" in step_names
    assert state.get("final_answer") != ""
