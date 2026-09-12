"""
LangGraph CRAG Pipeline State Definition.
"""

from typing import List, Dict, Any, Optional, TypedDict
import networkx as nx
from src.dataset import Passage
from src.crag.grader import CRAGGradingResult


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

    # Strategy & Seeds
    retrieval_strategy: str
    seed_entities: List[str]

    # Retrieval outputs
    retrieved_passages: List[Passage]
    retrieval_scores: List[float]
    retrieval_metrics: Dict[str, Any]

    # CRAG Evaluation
    grading_result: Optional[CRAGGradingResult]
    retry_count: int
    max_retries: int

    # Generation & Trace
    execution_trace: List[ExecutionStep]
    final_answer: str
    supporting_evidence_text: str
