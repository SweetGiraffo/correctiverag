"""
LangGraph Multi-Agent CRAG Pipeline.
Wires Router -> Retriever -> Grader -> Evaluator -> Rewriter / Fallback -> Generator as a state machine.
Supports both legacy 2Wiki text passages and cross-modal retrieval across PDFs, tables, images, and audio.
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
from src.retrieval.ppr import MultimodalPPRRetriever
from src.retrieval.hybrid import MultimodalHybridRetriever
from src.crag.grader import CRAGRelevanceGrader, GradingDecision
from src.crag.evaluator import ContextSufficiencyEvaluator, SufficiencyDecision
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
    graph_store = state.get("graph_store")
    vector_store = state.get("vector_store")
    trace = list(state.get("execution_trace", []))

    if graph is None and graph_store is not None:
        graph = graph_store.get_graph()

    # Construct entity graph from passages if dataset mode and graph not initialized
    if graph is None and passages:
        builder = EntityGraphBuilder()
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
    is_multimodal = bool(vector_store is not None)

    step: ExecutionStep = {
        "step_name": "Router",
        "action": f"Selected retrieval strategy '{strategy}' ({'Multimodal' if is_multimodal else 'Text'})",
        "timestamp": _timestamp(),
        "details": {
            "query": query,
            "strategy": strategy,
            "is_multimodal": is_multimodal,
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
    or hybrid retrieval across modalities based on the active strategy.
    """
    config = get_current_config()
    query = state["working_query"]
    passages = state.get("passages", [])
    graph = state.get("graph")
    vector_store = state.get("vector_store")
    strategy = state.get("retrieval_strategy", "ppr")
    top_k = config.top_k_passages
    trace = list(state.get("execution_trace", []))

    retrieved_passages = []
    retrieved_chunks = []
    scores = []
    metrics = {}
    seeds = []

    # Case A: Multimodal retrieval with Qdrant + NetworkX
    if vector_store is not None:
        if strategy == "ppr" and graph is not None:
            retriever = MultimodalPPRRetriever(alpha=config.ppr_alpha, model_name=config.embedding_model_name)
            result = retriever.retrieve(query=query, graph=graph, vector_store=vector_store, top_k=top_k)
            retrieved_chunks = result.chunks
            retrieved_passages = [c.to_passage() for c in result.chunks]
            scores = [c.score for c in result.chunks]
            seeds = result.seed_entities
            metrics = {
                "execution_time_ms": result.execution_time_ms,
                "algorithm": result.algorithm,
                "seed_entities": seeds
            }
        else:  # hybrid or fallback
            retriever = MultimodalHybridRetriever(model_name=config.embedding_model_name)
            result = retriever.retrieve(query=query, vector_store=vector_store, top_k=top_k)
            retrieved_chunks = result.chunks
            retrieved_passages = [c.to_passage() for c in result.chunks]
            scores = [c.score for c in result.chunks]
            seeds = result.seed_entities
            metrics = {
                "execution_time_ms": result.execution_time_ms,
                "algorithm": result.algorithm,
                "seed_entities": seeds
            }

    # Case B: Dataset passage retrieval (2WikiMultiHopQA)
    else:
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
        "action": f"Retrieved top-{len(retrieved_passages)} items via {strategy.upper()}",
        "timestamp": _timestamp(),
        "details": {
            "retrieved_titles": [p.title for p in retrieved_passages],
            "scores": scores,
            "latency_ms": metrics.get("execution_time_ms", 0.0),
            "chunk_count": len(retrieved_chunks)
        }
    }
    trace.append(step)

    return {
        "retrieved_passages": retrieved_passages,
        "retrieved_chunks": retrieved_chunks,
        "retrieval_scores": scores,
        "seed_entities": seeds,
        "retrieval_metrics": metrics,
        "execution_trace": trace
    }


def grader_node(state: CRAGState) -> Dict[str, Any]:
    """
    Grader Agent: Evaluates retrieved passage / chunk relevance against query.
    Emits formal classification: CORRECT, AMBIGUOUS, or INCORRECT.
    """
    config = get_current_config()
    working_query = state["working_query"]
    # Grade chunks if available, else passages
    items_to_grade = state.get("retrieved_chunks") or state.get("retrieved_passages", [])
    trace = list(state.get("execution_trace", []))

    grader = CRAGRelevanceGrader(
        threshold_high=config.grader_threshold_high,
        threshold_low=config.grader_threshold_low,
        model_name=config.embedding_model_name
    )
    grading_res = grader.grade(working_query, items_to_grade)

    step: ExecutionStep = {
        "step_name": "Grader",
        "action": f"Relevance graded as '{grading_res.decision.value.upper()}' (Confidence: {grading_res.confidence_score:.3f})",
        "timestamp": _timestamp(),
        "details": {
            "decision": grading_res.decision.value,
            "confidence": grading_res.confidence_score,
            "threshold_high": grading_res.threshold_high,
            "threshold_low": grading_res.threshold_low,
            "relevant_ratio": grading_res.relevant_ratio,
            "reasoning": grading_res.reasoning,
            "fallback_action": grading_res.fallback_action
        }
    }
    trace.append(step)

    return {
        "grading_result": grading_res,
        "execution_trace": trace
    }


def evaluator_node(state: CRAGState) -> Dict[str, Any]:
    """
    Context Sufficiency Evaluator Agent:
    Checks if >= 60% of retrieved chunks are relevant.
    If < 60% are relevant, triggers query reformulation loop.
    If 0 are relevant, triggers broad fallback retrieval.
    """
    config = get_current_config()
    working_query = state["working_query"]
    items = state.get("retrieved_chunks") or state.get("retrieved_passages", [])
    grading_res = state.get("grading_result")
    trace = list(state.get("execution_trace", []))

    evaluator = ContextSufficiencyEvaluator(
        sufficiency_threshold=config.sufficiency_threshold,
        min_relevance_score=config.grader_threshold_low,
        model_name=config.embedding_model_name
    )
    eval_res = evaluator.evaluate(working_query, items, grading_result=grading_res)

    step: ExecutionStep = {
        "step_name": "Evaluator",
        "action": f"Context Sufficiency: '{eval_res.decision.value.upper()}' ({eval_res.relevant_count}/{eval_res.total_chunks} relevant, {eval_res.relevant_ratio * 100:.1f}%)",
        "timestamp": _timestamp(),
        "details": {
            "decision": eval_res.decision.value,
            "is_sufficient": eval_res.is_sufficient,
            "relevant_ratio": eval_res.relevant_ratio,
            "relevant_count": eval_res.relevant_count,
            "total_chunks": eval_res.total_chunks,
            "threshold": eval_res.threshold,
            "recommendation": eval_res.recommendation,
            "reasoning": eval_res.reasoning
        }
    }
    trace.append(step)

    return {
        "evaluation_result": eval_res,
        "execution_trace": trace
    }


def rewriter_node(state: CRAGState) -> Dict[str, Any]:
    """
    Rewriter Agent: Triggers when retrieval is AMBIGUOUS or INSUFFICIENT (<60% relevant).
    Reformulates query with bridge entities and loops back to Retriever.
    """
    working_query = state["working_query"]
    items = state.get("retrieved_chunks") or state.get("retrieved_passages", [])
    current_retry = state.get("retry_count", 0) + 1
    trace = list(state.get("execution_trace", []))

    rewriter = CRAGQueryRewriter()
    rewrite_res = rewriter.rewrite(working_query, items, iteration=current_retry)

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
    Fallback Retriever Agent: Triggers when retrieval is INCORRECT or 0 relevant chunks found.
    Executes broad fallback search over all available context passages or dense Qdrant vector store.
    """
    current_retry = state.get("retry_count", 0) + 1
    passages = state.get("passages", [])
    vector_store = state.get("vector_store")
    query = state["working_query"]
    trace = list(state.get("execution_trace", []))

    fallback_passages = []
    fallback_chunks = []

    # Case A: Multimodal broad search in Qdrant
    if vector_store is not None:
        from src.retrieval.vector_retriever import EmbeddingModelCache
        model = EmbeddingModelCache.get_model(get_current_config().embedding_model_name)
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()
        hits = vector_store.search(query_vector=q_emb, limit=3)
        from src.retrieval.models import RetrievedChunk
        from src.ingest.base import Modality
        for hit in hits:
            mod_str = hit.payload.get("modality", "text")
            try:
                mod = Modality(mod_str)
            except ValueError:
                mod = Modality.TEXT
            chunk = RetrievedChunk(
                chunk_id=hit.chunk_id,
                modality=mod,
                text=hit.text,
                score=hit.score,
                metadata=hit.payload.get("metadata", {}),
                image_path=hit.payload.get("image_path"),
                entities=hit.payload.get("entities", [])
            )
            fallback_chunks.append(chunk)
            fallback_passages.append(chunk.to_passage())

    # Case B: Lexical title and keyword match over dataset passages
    else:
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
        "retrieved_chunks": fallback_chunks,
        "retry_count": current_retry,
        "execution_trace": trace
    }


def generator_node(state: CRAGState) -> Dict[str, Any]:
    """
    Generator Agent: Synthesizes final answer with citations from verified context passages.
    """
    query = state.get("query", "")
    items = state.get("retrieved_chunks") or state.get("retrieved_passages", [])
    trace = list(state.get("execution_trace", []))
    config = get_current_config()

    generator = AnswerGenerator()
    gen_result = generator.generate(query, items, config=config)

    step: ExecutionStep = {
        "step_name": "Generator",
        "action": f"Synthesized final answer via {gen_result['mode']}",
        "timestamp": _timestamp(),
        "details": {
            "answer": gen_result["answer"],
            "mode": gen_result["mode"],
            "citations": gen_result.get("citations", []),
            "passages_used": [getattr(p, "title", str(p)) for p in items]
        }
    }
    trace.append(step)

    return {
        "final_answer": gen_result["answer"],
        "supporting_evidence_text": gen_result["supporting_text"],
        "citations": gen_result.get("citations", []),
        "execution_trace": trace
    }


def decide_post_evaluation(state: CRAGState) -> str:
    """
    Conditional routing edge from Evaluator:
    - If SUFFICIENT -> proceed to generator
    - If retries exhausted -> proceed to generator (best effort)
    - If INSUFFICIENT_AMBIGUOUS -> rewrite and re-retrieve
    - If INSUFFICIENT_EMPTY -> fallback broad search
    """
    eval_res = state.get("evaluation_result")
    grading_res = state.get("grading_result")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if retry_count >= max_retries:
        return "generator"

    # Any INCORRECT grading or EMPTY context triggers fallback
    if grading_res is not None and grading_res.decision == GradingDecision.INCORRECT:
        return "fallback"
    if eval_res is not None and eval_res.decision == SufficiencyDecision.INSUFFICIENT_EMPTY:
        return "fallback"

    # Any AMBIGUOUS grading or INSUFFICIENT context (<60% ratio) triggers rewrite
    if grading_res is not None and grading_res.decision == GradingDecision.AMBIGUOUS:
        return "rewriter"
    if eval_res is not None and not eval_res.is_sufficient:
        return "rewriter"

    return "generator"


def build_crag_graph() -> Any:
    """Compiles the LangGraph CRAG multi-agent state machine."""
    workflow = StateGraph(CRAGState)

    # Register nodes
    workflow.add_node("router", router_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("grader", grader_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("rewriter", rewriter_node)
    workflow.add_node("fallback", fallback_node)
    workflow.add_node("generator", generator_node)

    # Wire edges
    workflow.add_edge(START, "router")
    workflow.add_edge("router", "retriever")
    workflow.add_edge("retriever", "grader")
    workflow.add_edge("grader", "evaluator")

    # Conditional routing based on sufficiency evaluation
    workflow.add_conditional_edges(
        "evaluator",
        decide_post_evaluation,
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
    passages: Optional[List[Passage]] = None,
    graph: Optional[nx.DiGraph] = None,
    sample_id: Optional[str] = None,
    strategy: Optional[str] = None,
    vector_store: Optional[Any] = None,
    graph_store: Optional[Any] = None
) -> CRAGState:
    """
    High-level entry point to execute the LangGraph pipeline for a given query,
    supporting both dataset passages and cross-modal storage.
    """
    initial_state: CRAGState = {
        "query": query,
        "working_query": query,
        "passages": passages or [],
        "graph": graph,
        "vector_store": vector_store,
        "graph_store": graph_store,
        "sample_id": sample_id,
        "retrieval_strategy": strategy or get_current_config().retrieval_algorithm,
        "retry_count": 0,
        "max_retries": get_current_config().max_retries,
        "execution_trace": []
    }

    final_state = crag_pipeline.invoke(initial_state)
    return final_state
