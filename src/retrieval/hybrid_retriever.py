"""
Hybrid Retriever combining Personalized PageRank with Dense Vector Search using RRF.
"""

from typing import List, Dict, Tuple, Optional
import time
import networkx as nx
from pydantic import BaseModel
from src.dataset import Passage
from .ppr_retriever import PPRRetriever, PPRRetrievalResult
from .vector_retriever import SequentialVectorRetriever, VectorRetrievalResult


class HybridRetrievalResult(BaseModel):
    passages: List[Passage]
    scores: List[float]
    ppr_result: PPRRetrievalResult
    vector_result: VectorRetrievalResult
    execution_time_ms: float
    vector_search_calls: int
    algorithm: str = "Hybrid Graph + Vector (RRF)"


class HybridRetriever:
    """
    Executes both PPR and Vector Retrieval, fusing rankings via Reciprocal Rank Fusion (RRF).
    RRF Score = 1 / (60 + rank_ppr) + 1 / (60 + rank_vector).
    """

    def __init__(
        self,
        ppr_alpha: float = 0.85,
        model_name: str = "all-MiniLM-L6-v2",
        rrf_k: int = 60
    ):
        self.ppr = PPRRetriever(alpha=ppr_alpha)
        self.vector = SequentialVectorRetriever(model_name=model_name)
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        graph: nx.DiGraph,
        passages: List[Passage],
        top_k: int = 3,
        explicit_seeds: Optional[List[str]] = None
    ) -> HybridRetrievalResult:
        start_time = time.perf_counter()

        ppr_res = self.ppr.retrieve(
            query=query,
            graph=graph,
            passages=passages,
            top_k=len(passages),
            explicit_seeds=explicit_seeds
        )
        vec_res = self.vector.retrieve(
            query=query,
            passages=passages,
            top_k=len(passages)
        )

        rrf_scores: Dict[str, float] = {}
        passages_by_title = {p.title: p for p in passages}

        # Accumulate PPR ranks
        for rank, p in enumerate(ppr_res.passages):
            rrf_scores[p.title] = rrf_scores.get(p.title, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        # Accumulate Vector ranks
        for rank, p in enumerate(vec_res.passages):
            rrf_scores[p.title] = rrf_scores.get(p.title, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        # Sort by RRF score descending
        sorted_titles = sorted(rrf_scores.keys(), key=lambda t: rrf_scores[t], reverse=True)
        top_titles = sorted_titles[:top_k]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return HybridRetrievalResult(
            passages=[passages_by_title[t] for t in top_titles],
            scores=[round(rrf_scores[t], 5) for t in top_titles],
            ppr_result=ppr_res,
            vector_result=vec_res,
            execution_time_ms=round(elapsed_ms, 2),
            vector_search_calls=vec_res.vector_search_calls
        )
