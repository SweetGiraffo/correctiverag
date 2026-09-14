"""
Corrective RAG (CRAG) Relevance Grader.
Scores the semantic alignment between the query and retrieved context passages.
Applies a formal decision function based on configurable thresholds to classify
retrieval output as CORRECT, AMBIGUOUS, or INCORRECT.
"""

from typing import List, Dict, Optional
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field
from src.dataset import Passage
from src.retrieval.vector_retriever import EmbeddingModelCache


class GradingDecision(str, Enum):
    CORRECT = "correct"       # Score >= Upper Threshold: Confident and relevant
    AMBIGUOUS = "ambiguous"   # Lower <= Score < Upper: Partial relevance, needs query rewrite
    INCORRECT = "incorrect"   # Score < Lower Threshold: Irrelevant, trigger fallback


class PassageGradingScore(BaseModel):
    title: str
    cosine_similarity: float
    best_matching_sentence: str
    chunk_id: Optional[str] = None
    modality: str = "text"
    is_relevant: bool = False


class CRAGGradingResult(BaseModel):
    """Encapsulates the decision function output and reasoning diagnostics."""
    decision: GradingDecision
    confidence_score: float
    threshold_high: float
    threshold_low: float
    passage_scores: List[PassageGradingScore]
    reasoning: str
    fallback_action: str
    relevant_ratio: float = 0.0
    relevant_count: int = 0


class CRAGRelevanceGrader:
    """
    Evaluates retrieved passages/chunks using SentenceTransformer embeddings.
    Implements the formal CRAG decision boundary.
    """

    def __init__(
        self,
        threshold_high: float = 0.65,
        threshold_low: float = 0.35,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.model_name = model_name

    def grade(
        self,
        query: str,
        retrieved_passages: List[Any],
        override_threshold_high: Optional[float] = None,
        override_threshold_low: Optional[float] = None
    ) -> CRAGGradingResult:
        """
        Grades retrieved passages or multimodal chunks against the query.
        Returns a structured CRAGGradingResult with decision and actionable fallback.
        """
        th_high = override_threshold_high if override_threshold_high is not None else self.threshold_high
        th_low = override_threshold_low if override_threshold_low is not None else self.threshold_low

        if not retrieved_passages:
            return CRAGGradingResult(
                decision=GradingDecision.INCORRECT,
                confidence_score=0.0,
                threshold_high=th_high,
                threshold_low=th_low,
                passage_scores=[],
                reasoning="Zero passages retrieved.",
                fallback_action="trigger_fallback_broad_search",
                relevant_ratio=0.0,
                relevant_count=0
            )

        model = EmbeddingModelCache.get_model(self.model_name)

        # Encode query
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

        passage_evals: List[PassageGradingScore] = []
        sim_scores: List[float] = []

        for p in retrieved_passages:
            if isinstance(p, Passage):
                title = p.title
                candidates = [p.title] + (p.sentences if p.sentences else [p.text])
                modality = "text"
                chunk_id = None
            elif hasattr(p, "modality") and hasattr(p, "text"):
                # RetrievedChunk or IngestedChunk
                title = p.metadata.get("source_file") or f"[{getattr(p.modality, 'value', str(p.modality)).upper()}]"
                modality = getattr(p.modality, "value", str(p.modality))
                chunk_id = getattr(p, "chunk_id", None)
                lines = [s.strip() for s in p.text.split("\n") if s.strip()]
                candidates = [title] + (lines if lines else [p.text])
            else:
                title = getattr(p, "title", "Unknown")
                text = getattr(p, "text", str(p))
                sentences = getattr(p, "sentences", [text])
                modality = getattr(p, "modality", "text")
                chunk_id = getattr(p, "chunk_id", None)
                candidates = [title] + (sentences if sentences else [text])

            # We score both title/header and contents to capture fine-grained relevance
            cand_embs = model.encode(candidates, convert_to_numpy=True, normalize_embeddings=True)
            cand_sims = np.dot(cand_embs, q_emb)

            best_idx = int(np.argmax(cand_sims))
            best_sim = float(cand_sims[best_idx])
            best_sent = candidates[best_idx]
            is_rel = bool(best_sim >= th_low)

            passage_evals.append(
                PassageGradingScore(
                    title=title,
                    chunk_id=chunk_id,
                    modality=modality,
                    cosine_similarity=round(best_sim, 4),
                    best_matching_sentence=best_sent[:150],
                    is_relevant=is_rel
                )
            )
            sim_scores.append(best_sim)

        # Overall confidence: weighted blend of maximum similarity and top-k mean
        max_sim = max(sim_scores) if sim_scores else 0.0
        mean_sim = float(np.mean(sim_scores)) if sim_scores else 0.0
        confidence = float(0.7 * max_sim + 0.3 * mean_sim)
        confidence = round(confidence, 4)

        rel_count = sum(1 for s in passage_evals if s.is_relevant)
        rel_ratio = round(rel_count / len(retrieved_passages), 4)

        # Decision Function
        if confidence >= th_high:
            decision = GradingDecision.CORRECT
            reasoning = (
                f"Confidence score {confidence:.3f} exceeds upper threshold {th_high:.3f}. "
                f"Retrieved context is deemed accurate and sufficient for direct answer synthesis."
            )
            fallback_action = "proceed_to_generation"
        elif confidence >= th_low:
            decision = GradingDecision.AMBIGUOUS
            reasoning = (
                f"Confidence score {confidence:.3f} is in ambiguous range [{th_low:.3f}, {th_high:.3f}). "
                f"Retrieved passages provide partial alignment but may lack the multi-hop bridge. "
                f"Triggering query reformulation."
            )
            fallback_action = "rewrite_query_and_re_retrieve"
        else:
            decision = GradingDecision.INCORRECT
            reasoning = (
                f"Confidence score {confidence:.3f} falls below lower threshold {th_low:.3f}. "
                f"Retrieved context is largely irrelevant or distractor-heavy. "
                f"Triggering fallback broad retrieval."
            )
            fallback_action = "trigger_fallback_broad_search"

        return CRAGGradingResult(
            decision=decision,
            confidence_score=confidence,
            threshold_high=th_high,
            threshold_low=th_low,
            passage_scores=passage_evals,
            reasoning=reasoning,
            fallback_action=fallback_action,
            relevant_ratio=rel_ratio,
            relevant_count=rel_count
        )
