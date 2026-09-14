"""
CRAG Query Rewriter & Decomposer.
Reformulates and disambiguates multi-hop queries when grading outputs AMBIGUOUS or INCORRECT.
"""

from typing import List, Optional
import re
from pydantic import BaseModel
import spacy
from src.dataset import Passage


class RewriteResult(BaseModel):
    original_query: str
    rewritten_query: str
    strategy: str
    extracted_bridge_candidates: List[str]


class CRAGQueryRewriter:
    """
    Reformulates multi-hop queries to surface bridge entities and clarify intent.
    Supports rule-based decomposition and bridge expansion.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.nlp = None
        self._spacy_model_name = spacy_model

    def _ensure_nlp(self):
        if self.nlp is None:
            try:
                self.nlp = spacy.load(self._spacy_model_name, disable=["parser"])
            except Exception:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except Exception:
                    self.nlp = spacy.blank("en")

    @staticmethod
    def _get_title(item: Any) -> str:
        if hasattr(item, "title"):
            return item.title
        if hasattr(item, "metadata") and isinstance(item.metadata, dict):
            return item.metadata.get("source_file", "Document")
        return str(item)

    def rewrite(
        self,
        query: str,
        retrieved_passages: List[Any],
        iteration: int = 1
    ) -> RewriteResult:
        """
        Reformulates the query using context from the previous retrieval attempt.
        """
        self._ensure_nlp()
        extracted_bridges: List[str] = []

        # 1. Identify potential bridge entities from top passages/chunks
        for p in retrieved_passages[:3]:
            # If chunk already has extracted entities, use them
            if hasattr(p, "entities") and isinstance(p.entities, list):
                for ent_str in p.entities:
                    if len(ent_str) > 2 and ent_str.lower() not in query.lower():
                        extracted_bridges.append(ent_str)

            # Also run NER on raw text snippet
            text_snippet = getattr(p, "text", "")[:400]
            if self.nlp and text_snippet:
                doc = self.nlp(text_snippet)
                for ent in doc.ents:
                    ent_str = ent.text.strip()
                    if len(ent_str) > 2 and ent_str.lower() not in query.lower():
                        extracted_bridges.append(ent_str)

        # Deduplicate candidates while preserving order
        unique_bridges = list(dict.fromkeys(extracted_bridges))[:3]

        rewritten = query
        strategy = "entity_focus"

        # Pattern 1: Compositional pattern "Who/What is the [RELATION] of the [ROLE] of [ENTITY]?"
        # e.g. "Who is the mother of the director of film Polish-Russian War?"
        comp_match = re.search(
            r"^(Who|What|Which)\s+(is|was|were|are)\s+the\s+([\w\s]+?)\s+of\s+the\s+([\w\s]+?)\s+of\s+(.+?)\??$",
            query,
            re.IGNORECASE
        )
        if comp_match:
            wh = comp_match.group(1)
            vb = comp_match.group(2)
            rel1 = comp_match.group(3).strip()
            role2 = comp_match.group(4).strip()
            entity = comp_match.group(5).strip().rstrip("?")

            if unique_bridges:
                bridge_candidate = unique_bridges[0]
                rewritten = f"{wh} {vb} the {rel1} of {bridge_candidate}, {role2} of {entity}?"
                strategy = "compositional_bridge_injection"
            else:
                rewritten = f"{role2} of {entity} and their {rel1}"
                strategy = "compositional_decomposition"

        # Pattern 2: Comparison questions "Which [entity/film] was [attribute], X or Y?"
        elif re.search(r"\b(or|versus|vs\.?)\b", query, re.IGNORECASE):
            strategy = "comparison_splitting"
            # Highlight key entities
            if len(retrieved_passages) >= 2:
                t0 = self._get_title(retrieved_passages[0])
                t1 = self._get_title(retrieved_passages[1])
                rewritten = f"{query} [Focus: {t0} and {t1}]"
            else:
                rewritten = f"{query} [Detailed timeline and attributes]"

        # Pattern 3: General expansion with bridge candidate
        else:
            if unique_bridges:
                rewritten = f"{query} (related to {', '.join(unique_bridges[:2])})"
                strategy = "contextual_bridge_expansion"
            elif retrieved_passages:
                t0 = self._get_title(retrieved_passages[0])
                rewritten = f"{query} {t0}"
                strategy = "title_anchoring"

        return RewriteResult(
            original_query=query,
            rewritten_query=rewritten,
            strategy=strategy,
            extracted_bridge_candidates=unique_bridges
        )

