"""
LangGraph Multi-Agent CRAG Pipeline.
Wires Router -> Retriever -> Grader -> Rewriter / Fallback -> Generator as a state machine.
"""

from typing import Dict, Any, List, Optional
import time
from datetime import datetime, timezone
import networkx as nx
from langgraph.graph import StateGraph, START, END

from src.agent.state import CRAGState, ExecutionStep
from src.config import RuntimeConfig, get_current_config
from src.graph_builder import EntityGraphBuilder
from src.retrieval.ppr_retriever import PPRRetriever
from src.retrieval.vector_retriever import SequentialVectorRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.crag.grader import CRAGRelevanceGrader, GradingDecision
from src.crag.rewriter import CRAGQueryRewriter
from src.generator.generator import AnswerGenerator


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def router_node(state: CRAGState) -> Dict[str, Any]:
    """
    Router Agent: Analyzes query complexity, verifies entity graph presence,
    and sets retrieval strategy according to runtime configuration.
    """
    config = get_current_config()
    query = state.get("query", "")
    working_query = state.get("working_query") or query
    passages = state.get("passages", [])
    graph = state.get("graph")
    trace = list(state.get("execution_trace", []))

    # Construct entity graph if not already initialized
    if graph is None and passages:
        builder = EntityGraphBuilder()
        # Build graph directly from passages
        from src.dataset import MultiHopSample
        dummy_sample = MultiHopSample(
            id=state.get("sample_id", "dynamic"),
            question=query,
            type="multi_hop",
            answer="",
            context_passages=passages
        )
        graph = builder.build_graph_for_sample(dummy_sample)

    strategy = state.get("retrieval_strategy") or config.retrieval_algorithm

    step: ExecutionStep = {
        "step_name": "Router",
        "action": f"Selected retrieval strategy '{strategy}' for query",
        "timestamp": _timestamp(),
        "details": {
            "query": query,
            "strategy": strategy,
            "graph_nodes": len(graph.nodes) if graph else 0,
            "graph_edges": len(graph.edges) if graph else 0,
            "passage_count": len(passages)
        }
    }
    trace.append(step)

    return {
        "working_query": working_query,
        "graph": graph,
        "retrieval_strategy": strategy,
        "retry_count": state.get("retry_count", 0),
        "max_retries": config.max_retries,
        "execution_trace": trace
    }


def retriever_node(state: CRAGState) -> Dict[str, Any]:
    """
    Retriever Agent: Executes PPR graph traversal, sequential vector search,
    or hybrid retrieval based on the active strategy.
    """
    config = get_current_config()
    query = state["working_query"]
    passages = state.get("passages", [])
    graph = state.get("graph")
    strategy = state.get("retrieval_strategy", "ppr")
    top_k = config.top_k_passages
    trace = list(state.get("execution_trace", []))

    retrieved_passages = []
    scores = []
    metrics = {}
    seeds = []

    if strategy == "ppr" and graph:
        retriever = PPRRetriever(alpha=config.ppr_alpha)
        result = retriever.retrieve(query=query, graph=graph, passages=passages, top_k=top_k)
        retrieved_passages = result.passages
        scores = result.scores
        seeds = result.seed_entities
        metrics = {
            "execution_time_ms": result.execution_time_ms,
            "vector_search_calls": result.vector_search_calls,
            "algorithm": result.algorithm,
            "seed_entities": seeds
        }
    elif strategy == "vector_hop":
        retriever = SequentialVectorRetriever(model_name=config.embedding_model_name)
        result = retriever.retrieve(query=query, passages=passages, top_k=top_k)
        retrieved_passages = result.passages
        scores = result.scores
        metrics = {
            "execution_time_ms": result.execution_time_ms,
            "vector_search_calls": result.vector_search_calls,
            "algorithm": result.algorithm,
            "hop_queries": result.hop_queries
        }
    else:  # hybrid
        retriever = HybridRetriever(ppr_alpha=config.ppr_alpha, model_name=config.embedding_model_name)
        result = retriever.retrieve(query=query, graph=graph if graph else nx.DiGraph(), passages=passages, top_k=top_k)
        retrieved_passages = result.passages
        scores = result.scores
        metrics = {
            "execution_time_ms": result.execution_time_ms,
            "vector_search_calls": result.vector_search_calls,
            "algorithm": result.algorithm
        }

    step: ExecutionStep = {
        "step_name": "Retriever",
        "action": f"Retrieved top-{len(retrieved_passages)} passages via {strategy.upper()}",
        "timestamp": _timestamp(),
        "details": {
            "retrieved_titles": [p.title for p in retrieved_passages],
            "scores": scores,
            "latency_ms": metrics.get("execution_time_ms", 0.0),
            "vector_calls": metrics.get("vector_search_calls", 0)
        }
    }
    trace.append(step)

    return {
        "retrieved_passages": retrieved_passages,
        "retrieval_scores": scores,
        "seed_entities": seeds,
        "retrieval_metrics": metrics,
        "execution_trace": trace
    }


def grader_node(state: CRAGState) -> Dict[str, Any]:
    """
    Grader Agent: Evaluates retrieved passage relevance against query using Sentence-Transformers.
    Emits formal classification: CORRECT, AMBIGUOUS, or INCORRECT.
    """
    config = get_current_config()
    working_query = state["working_query"]
    retrieved_passages = state.get("retrieved_passages", [])
    trace = list(state.get("execution_trace", []))

    grader = CRAGRelevanceGrader(
        threshold_high=config.grader_threshold_high,
        threshold_low=config.grader_threshold_low,
        model_name=config.embedding_model_name
    )
    grading_res = grader.grade(working_query, retrieved_passages)

    step: ExecutionStep = {
        "step_name": "Grader",
        "action": f"Relevance graded as '{grading_res.decision.value.upper()}' (Confidence: {grading_res.confidence_score:.3f})",
        "timestamp": _timestamp(),
        "details": {
            "decision": grading_res.decision.value,
            "confidence": grading_res.confidence_score,
            "threshold_high": grading_res.threshold_high,
            "threshold_low": grading_res.threshold_low,
            "reasoning": grading_res.reasoning,
            "fallback_action": grading_res.fallback_action
        }
    }
    trace.append(step)

    return {
        "grading_result": grading_res,
        "execution_trace": trace
    }


def rewriter_node(state: CRAGState) -> Dict[str, Any]:
    """
    Rewriter Agent: Triggers when retrieval is AMBIGUOUS.
    Reformulates query with bridge entities and loops back to Retriever.
    """
    working_query = state["working_query"]
    retrieved_passages = state.get("retrieved_passages", [])
    current_retry = state.get("retry_count", 0) + 1
    trace = list(state.get("execution_trace", []))

    rewriter = CRAGQueryRewriter()
    rewrite_res = rewriter.rewrite(working_query, retrieved_passages, iteration=current_retry)

    step: ExecutionStep = {
        "step_name": "Rewriter",
        "action": f"Query reformulated (Retry {current_retry}/{state.get('max_retries', 2)}): {rewrite_res.strategy}",
        "timestamp": _timestamp(),
        "details": {
            "original": working_query,
            "rewritten": rewrite_res.rewritten_query,
            "strategy": rewrite_res.strategy,
            "bridge_candidates": rewrite_res.extracted_bridge_candidates
        }
    }
    trace.append(step)

    return {
        "working_query": rewrite_res.rewritten_query,
        "retry_count": current_retry,
        "execution_trace": trace
    }


def fallback_node(state: CRAGState) -> Dict[str, Any]:
    """
    Fallback Retriever Agent: Triggers when retrieval is INCORRECT.
    Executes broad fallback search over all available context passages.
    """
    current_retry = state.get("retry_count", 0) + 1
    passages = state.get("passages", [])
    query = state["working_query"]
    trace = list(state.get("execution_trace", []))

    # Broad fallback: lexical title and keyword match
    scored = []
    q_words = set(query.lower().split())
    for p in passages:
        overlap = sum(1 for w in q_words if w in p.title.lower() or w in p.text.lower())
        scored.append((p, overlap))
    scored.sort(key=lambda x: x[1], reverse=True)
    fallback_passages = [p for p, _ in scored[:3]]

    step: ExecutionStep = {
        "step_name": "Fallback",
        "action": f"Fallback broad search triggered (Retry {current_retry}/{state.get('max_retries', 2)})",
        "timestamp": _timestamp(),
        "details": {
            "fallback_titles": [p.title for p in fallback_passages]
        }
    }
    trace.append(step)

    return {
        "retrieved_passages": fallback_passages,
        "retry_count": current_retry,
        "execution_trace": trace
    }


def generator_node(state: CRAGState) -> Dict[str, Any]:
    """
    Generator Agent: Synthesizes final answer from verified context passages.
    """
    query = state.get("query", "")
    retrieved_passages = state.get("retrieved_passages", [])
    trace = list(state.get("execution_trace", []))
    config = get_current_config()

    generator = AnswerGenerator()
    gen_result = generator.generate(query, retrieved_passages, config=config)

    step: ExecutionStep = {
        "step_name": "Generator",
        "action": f"Synthesized final answer via {gen_result['mode']}",
        "timestamp": _timestamp(),
        "details": {
            "answer": gen_result["answer"],
            "mode": gen_result["mode"],
            "passages_used": [p.title for p in retrieved_passages]
        }
    }
    trace.append(step)

    return {
        "final_answer": gen_result["answer"],
        "supporting_evidence_text": gen_result["supporting_text"],
        "execution_trace": trace
    }


def decide_post_grading(state: CRAGState) -> str:
    """
    Conditional routing edge from Grader:
    - If CORRECT -> proceed to generator
    - If retries exhausted -> proceed to generator (best effort)
    - If AMBIGUOUS -> rewrite and re-retrieve
    - If INCORRECT -> fallback broad search
    """
    grading_res = state.get("grading_result")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if not grading_res:
        return "generator"

    if grading_res.decision == GradingDecision.CORRECT:
        return "generator"

    if retry_count >= max_retries:
        return "generator"

    if grading_res.decision == GradingDecision.AMBIGUOUS:
        return "rewriter"
    elif grading_res.decision == GradingDecision.INCORRECT:
        return "fallback"

    return "generator"


def build_crag_graph() -> Any:
    """Compiles the LangGraph CRAG multi-agent state machine."""
    workflow = StateGraph(CRAGState)

    # Register nodes
    workflow.add_node("router", router_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("grader", grader_node)
    workflow.add_node("rewriter", rewriter_node)
    workflow.add_node("fallback", fallback_node)
    workflow.add_node("generator", generator_node)

    # Wire edges
    workflow.add_edge(START, "router")
    workflow.add_edge("router", "retriever")
    workflow.add_edge("retriever", "grader")

    # Conditional routing based on grading decision
    workflow.add_conditional_edges(
        "grader",
        decide_post_grading,
        {
            "generator": "generator",
            "rewriter": "rewriter",
            "fallback": "fallback"
        }
    )

    # Loopbacks
    workflow.add_edge("rewriter", "retriever")
    workflow.add_edge("fallback", "grader")
    workflow.add_edge("generator", END)

    return workflow.compile()


# Global compiled pipeline instance
crag_pipeline = build_crag_graph()


def run_crag_pipeline(
    query: str,
    passages: List[Passage],
    graph: Optional[nx.DiGraph] = None,
    sample_id: Optional[str] = None,
    strategy: Optional[str] = None
) -> CRAGState:
    """
    High-level entry point to execute the LangGraph pipeline for a given query and passages.
    """
    initial_state: CRAGState = {
        "query": query,
        "working_query": query,
        "passages": passages,
        "graph": graph,
        "sample_id": sample_id,
        "retrieval_strategy": strategy or get_current_config().retrieval_algorithm,
        "retry_count": 0,
        "max_retries": get_current_config().max_retries,
        "execution_trace": []
    }

    final_state = crag_pipeline.invoke(initial_state)
    return final_state
