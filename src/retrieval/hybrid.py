"""
Multimodal Hybrid Retriever (Dense Vector + BM25 + Reranker).
Serves as the baseline comparison and fallback retrieval strategy.
"""

from typing import List, Dict, Any, Optional
import time
import os
import numpy as np
from rank_bm25 import BM25Okapi
from src.ingest.base import Modality
from src.storage.vector_store import QdrantStore
from src.retrieval.vector_retriever import EmbeddingModelCache
from .models import RetrievedChunk, MultimodalRetrievalResult
from .ppr import load_image_as_base64


class RerankerCache:
    """Singleton cache for CrossEncoder / BGE Reranker."""
    _reranker = None
    _reranker_name = None

    @classmethod
    def get_reranker(cls, model_name: str = "BAAI/bge-reranker-base"):
        if cls._reranker is None or cls._reranker_name != model_name:
            try:
                from sentence_transformers import CrossEncoder
                cls._reranker = CrossEncoder(model_name)
                cls._reranker_name = model_name
            except Exception:
                cls._reranker = None
        return cls._reranker


class MultimodalHybridRetriever:
    """
    Executes hybrid dense vector search + BM25 keyword matching,
    followed by cross-encoder neural reranking.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = "BAAI/bge-reranker-base",
        rrf_k: int = 60
    ):
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        vector_store: QdrantStore,
        top_k: int = 5,
        apply_reranker: bool = True
    ) -> MultimodalRetrievalResult:
        start_time = time.perf_counter()

        # Step 1: Fetch all chunks from Qdrant to build BM25 corpus & dense hits
        model = EmbeddingModelCache.get_model(self.embedding_model)
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

        total_chunks_count = vector_store.count() or 10
        dense_hits = vector_store.search(query_vector=q_emb.tolist(), top_k=total_chunks_count)

        if not dense_hits:
            return MultimodalRetrievalResult(
                query=query,
                execution_time_ms=0.0,
                algorithm="Hybrid Vector + BM25"
            )

        # Step 2: Dense Ranking
        dense_rank_map: Dict[str, int] = {}
        chunks_by_id: Dict[str, Dict[str, Any]] = {}
        corpus_texts = []
        chunk_ids = []

        for rank, hit in enumerate(dense_hits):
            cid = hit["payload"]["chunk_id"]
            dense_rank_map[cid] = rank + 1
            chunks_by_id[cid] = hit["payload"]
            corpus_texts.append(hit["payload"].get("text", ""))
            chunk_ids.append(cid)

        # Step 3: BM25 Sparse Ranking
        tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)

        # Sort by BM25 score
        sorted_bm25_indices = np.argsort(-bm25_scores)
        bm25_rank_map: Dict[str, int] = {}
        for rank, idx in enumerate(sorted_bm25_indices):
            cid = chunk_ids[idx]
            bm25_rank_map[cid] = rank + 1

        # Step 4: Reciprocal Rank Fusion (RRF)
        fused_scores: Dict[str, float] = {}
        for cid in chunks_by_id:
            d_rank = dense_rank_map.get(cid, 999)
            b_rank = bm25_rank_map.get(cid, 999)
            rrf = (1.0 / (self.rrf_k + d_rank)) + (1.0 / (self.rrf_k + b_rank))
            fused_scores[cid] = rrf

        # Candidate pool for reranking
        sorted_cids = sorted(fused_scores.keys(), key=lambda c: fused_scores[c], reverse=True)[:top_k * 2]

        # Step 5: BGE Neural Reranker (if available)
        candidates_to_score = [chunks_by_id[cid] for cid in sorted_cids]
        scored_candidates: List[RetrievedChunk] = []

        reranker = RerankerCache.get_reranker(self.reranker_model) if apply_reranker else None

        if reranker is not None:
            pairs = [[query, c.get("text", "")] for c in candidates_to_score]
            try:
                rerank_scores = reranker.predict(pairs)
                for c, s in zip(candidates_to_score, rerank_scores):
                    c_modality = Modality(c.get("modality", "text"))
                    img_path = c.get("image_path")
                    img_b64 = load_image_as_base64(img_path) if c_modality == Modality.IMAGE else None

                    scored_candidates.append(
                        RetrievedChunk(
                            chunk_id=c["chunk_id"],
                            modality=c_modality,
                            text=c.get("text", ""),
                            score=round(float(s), 4),
                            metadata=c.get("metadata", {}),
                            image_path=img_path,
                            image_base64=img_b64,
                            entities=c.get("entities", [])
                        )
                    )
                scored_candidates.sort(key=lambda x: x.score, reverse=True)
            except Exception:
                reranker = None

        if reranker is None:
            # Fallback to RRF score
            for c in candidates_to_score:
                cid = c["chunk_id"]
                c_modality = Modality(c.get("modality", "text"))
                img_path = c.get("image_path")
                img_b64 = load_image_as_base64(img_path) if c_modality == Modality.IMAGE else None

                scored_candidates.append(
                    RetrievedChunk(
                        chunk_id=cid,
                        modality=c_modality,
                        text=c.get("text", ""),
                        score=round(float(fused_scores[cid]), 5),
                        metadata=c.get("metadata", {}),
                        image_path=img_path,
                        image_base64=img_b64,
                        entities=c.get("entities", [])
                    )
                )

        final_chunks = scored_candidates[:top_k]
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return MultimodalRetrievalResult(
            query=query,
            top_nodes=[],
            chunks=final_chunks,
            seed_entities=[],
            execution_time_ms=round(elapsed_ms, 2),
            algorithm="Hybrid Vector + BM25 + Reranker"
        )
