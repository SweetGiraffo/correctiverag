"""
CRAG Context Sufficiency Evaluator.
Evaluates whether retrieved multi-hop context is sufficient for direct answer synthesis.
Applies the sufficiency decision boundary:
- If >= 60% of chunks are relevant: SUFFICIENT -> Proceed to generation.
- If < 60% of chunks are relevant (but > 0): INSUFFICIENT_AMBIGUOUS -> Trigger query rewriting.
- If 0 chunks are relevant: INSUFFICIENT_EMPTY -> Trigger broad fallback search.
"""

from typing import List, Dict, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field
from src.dataset import Passage
from src.retrieval.models import RetrievedChunk
from src.crag.grader import CRAGGradingResult, CRAGRelevanceGrader, GradingDecision


class SufficiencyDecision(str, Enum):
    SUFFICIENT = "sufficient"                 # >= threshold ratio (default 60%) relevant
    INSUFFICIENT_AMBIGUOUS = "ambiguous"      # > 0 but < threshold ratio relevant
    INSUFFICIENT_EMPTY = "empty"              # 0 relevant chunks


class ContextSufficiencyResult(BaseModel):
    decision: SufficiencyDecision
    is_sufficient: bool
    relevant_ratio: float
    relevant_count: int
    total_chunks: int
    threshold: float
    recommendation: str
    reasoning: str


class ContextSufficiencyEvaluator:
    """
    Evaluates chunk-level relevance distributions to determine if context
    is sufficient to synthesize an accurate multi-hop answer.
    """

    def __init__(
        self,
        sufficiency_threshold: float = 0.60,
        min_relevance_score: float = 0.35,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.sufficiency_threshold = sufficiency_threshold
        self.min_relevance_score = min_relevance_score
        self.model_name = model_name

    def evaluate(
        self,
        query: str,
        chunks: Union[List[Passage], List[RetrievedChunk], List[Any]],
        grading_result: Optional[CRAGGradingResult] = None,
        override_threshold: Optional[float] = None
    ) -> ContextSufficiencyResult:
        """
        Evaluates context sufficiency from retrieved chunks or precomputed grading results.
        """
        threshold = override_threshold if override_threshold is not None else self.sufficiency_threshold
        total_chunks = len(chunks)

        if total_chunks == 0:
            return ContextSufficiencyResult(
                decision=SufficiencyDecision.INSUFFICIENT_EMPTY,
                is_sufficient=False,
                relevant_ratio=0.0,
                relevant_count=0,
                total_chunks=0,
                threshold=threshold,
                recommendation="fallback",
                reasoning="Zero context chunks available to evaluate."
            )

        # Use precomputed grading result or grade on the fly
        if grading_result is None:
            grader = CRAGRelevanceGrader(
                threshold_high=0.65,
                threshold_low=self.min_relevance_score,
                model_name=self.model_name
            )
            grading_result = grader.grade(query, chunks)

        # Count chunks that meet or exceed min relevance
        relevant_count = 0
        min_threshold = grading_result.threshold_low

        for score in grading_result.passage_scores:
            if hasattr(score, "is_relevant") and score.is_relevant:
                relevant_count += 1
            elif score.cosine_similarity >= min_threshold:
                relevant_count += 1

        relevant_ratio = round(relevant_count / total_chunks, 4)

        if relevant_count == 0 or (grading_result and grading_result.decision == GradingDecision.INCORRECT and relevant_count == 0):
            decision = SufficiencyDecision.INSUFFICIENT_EMPTY
            is_sufficient = False
            recommendation = "fallback"
            reasoning = (
                f"No relevant chunks found ({relevant_count}/{total_chunks} relevant). "
                f"Context contains only distractors. Triggering fallback search."
            )
        elif relevant_ratio >= threshold:
            decision = SufficiencyDecision.SUFFICIENT
            is_sufficient = True
            recommendation = "proceed"
            reasoning = (
                f"Context is sufficient: {relevant_count}/{total_chunks} ({relevant_ratio * 100:.1f}%) "
                f"chunks exceed relevance threshold (required >= {threshold * 100:.0f}%)."
            )
        else:
            decision = SufficiencyDecision.INSUFFICIENT_AMBIGUOUS
            is_sufficient = False
            recommendation = "rewrite"
            reasoning = (
                f"Context is partially relevant but insufficient: {relevant_count}/{total_chunks} "
                f"({relevant_ratio * 100:.1f}%) chunks relevant, below required {threshold * 100:.0f}%. "
                f"Triggering query reformulation."
            )

        return ContextSufficiencyResult(
            decision=decision,
            is_sufficient=is_sufficient,
            relevant_ratio=relevant_ratio,
            relevant_count=relevant_count,
            total_chunks=total_chunks,
            threshold=threshold,
            recommendation=recommendation,
            reasoning=reasoning
        )
