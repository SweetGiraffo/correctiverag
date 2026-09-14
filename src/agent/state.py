"""
LangGraph CRAG Pipeline State Definition.
Supports both legacy text passages and unified multimodal chunks across text, PDF tables, images, and audio.
"""

from typing import List, Dict, Any, Optional, TypedDict
import networkx as nx
from src.dataset import Passage
from src.retrieval.models import RetrievedChunk
from src.crag.grader import CRAGGradingResult
from src.crag.evaluator import ContextSufficiencyResult


class ExecutionStep(TypedDict):
    step_name: str
    action: str
    timestamp: str
    details: Dict[str, Any]


class CRAGState(TypedDict, total=False):
    # Query & Inputs
    query: str
    working_query: str
    sample_id: Optional[str]
    passages: List[Passage]
    graph: Optional[nx.DiGraph]
    vector_store: Optional[Any]
    graph_store: Optional[Any]

    # Strategy & Seeds
    retrieval_strategy: str
    seed_entities: List[str]

    # Retrieval outputs (passages and multimodal chunks)
    retrieved_passages: List[Passage]
    retrieved_chunks: List[RetrievedChunk]
    retrieval_scores: List[float]
    retrieval_metrics: Dict[str, Any]

    # CRAG Grading & Sufficiency Evaluation
    grading_result: Optional[CRAGGradingResult]
    evaluation_result: Optional[ContextSufficiencyResult]
    retry_count: int
    max_retries: int

    # Generation & Trace
    execution_trace: List[ExecutionStep]
    final_answer: str
    supporting_evidence_text: str
    citations: List[str]
