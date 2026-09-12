"""
Baseline Sequential Per-Hop Vector Search Retriever.
Simulates conventional multi-hop vector retrieval where a fresh neural vector embedding
and dense similarity search is executed at each hop.
"""

from typing import List, Dict, Tuple, Optional
import time
import numpy as np
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from src.dataset import Passage


class VectorRetrievalResult(BaseModel):
    """Encapsulates output and diagnostic metrics from sequential vector retrieval."""
    passages: List[Passage]
    scores: List[float]
    hop_queries: List[str]
    execution_time_ms: float
    vector_search_calls: int
    algorithm: str = "Sequential Per-Hop Vector Search"


class EmbeddingModelCache:
    """Cached singleton for SentenceTransformer model to prevent redundant weights loading."""
    _model: Optional[SentenceTransformer] = None
    _model_name: Optional[str] = None

    @classmethod
    def get_model(cls, model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
        if cls._model is None or cls._model_name != model_name:
            cls._model = SentenceTransformer(model_name)
            cls._model_name = model_name
        return cls._model


class SequentialVectorRetriever:
    """
    Executes sequential vector search at every hop (the baseline comparison system).
    Hop 1: Query embedding -> top candidates.
    Hop 2: Intermediate context expansion -> second query embedding -> Hop 2 candidates.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name

    def retrieve(
        self,
        query: str,
        passages: List[Passage],
        top_k: int = 3,
        hops: int = 2
    ) -> VectorRetrievalResult:
        start_time = time.perf_counter()

        if not passages:
            return VectorRetrievalResult(
                passages=[],
                scores=[],
                hop_queries=[query],
                execution_time_ms=0.0,
                vector_search_calls=0
            )

        model = EmbeddingModelCache.get_model(self.model_name)
        passage_texts = [f"{p.title}: {p.text}" for p in passages]
        # Precompute or compute passage embeddings
        passage_embeddings = model.encode(passage_texts, convert_to_numpy=True, normalize_embeddings=True)

        selected_indices = set()
        ranked_items: List[Tuple[Passage, float]] = []
        hop_queries: List[str] = []
        vector_search_calls = 0

        current_query = query
        for hop in range(hops):
            hop_queries.append(current_query)
            # Encode query for this hop
            q_emb = model.encode([current_query], convert_to_numpy=True, normalize_embeddings=True)[0]
            vector_search_calls += 1

            # Cosine similarities (dot product with normalized embeddings)
            similarities = np.dot(passage_embeddings, q_emb)

            # Mask already selected passages
            for idx in selected_indices:
                similarities[idx] = -1.0

            best_idx = int(np.argmax(similarities))
            selected_indices.add(best_idx)
            ranked_items.append((passages[best_idx], float(similarities[best_idx])))

            # Formulate next hop query using retrieved context
            if hop < hops - 1:
                retrieved_p = passages[best_idx]
                # Bridge formulation: append retrieved passage title / top sentence
                current_query = f"{query} [Context: {retrieved_p.title}]"

        # If top_k > hops, backfill with next best remaining passages from hop 1
        if len(ranked_items) < top_k and len(passages) > len(ranked_items):
            q1_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
            sims = np.dot(passage_embeddings, q1_emb)
            sorted_all_indices = np.argsort(-sims)
            for idx in sorted_all_indices:
                if idx not in selected_indices:
                    selected_indices.add(idx)
                    ranked_items.append((passages[idx], float(sims[idx])))
                    if len(ranked_items) >= top_k:
                        break

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return VectorRetrievalResult(
            passages=[p for p, _ in ranked_items[:top_k]],
            scores=[round(s, 4) for _, s in ranked_items[:top_k]],
            hop_queries=hop_queries,
            execution_time_ms=round(elapsed_ms, 2),
            vector_search_calls=vector_search_calls
        )
