"""
Multimodal Personalized PageRank (PPR) Retrieval Engine.
Diffuses probability from query seed entities across the cross-modal entity graph,
returning top nodes and backing chunks across ALL modalities (PDF, tables, images, audio).
"""

from typing import List, Dict, Any, Optional, Set
import time
import os
import base64
import networkx as nx
import numpy as np
import re
from src.ingest.base import Modality, IngestedChunk
from src.ingest.entity_extractor import EntityExtractor
from src.storage.vector_store import QdrantStore
from src.retrieval.vector_retriever import EmbeddingModelCache
from .models import RetrievedChunk, MultimodalRetrievalResult


def normalize_entity_name(name: str) -> str:
    """Normalizes entity name by lowercasing and removing noise."""
    name = name.strip()
    name = re.sub(r"^(the|a|an)\s+", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[^\w\s-]", "", name)
    return " ".join(name.lower().split())


def load_image_as_base64(image_path: Optional[str]) -> Optional[str]:
    """Reads image from disk and converts to base64 string."""
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


class MultimodalPPRRetriever:
    """
    Retrieves cross-modal chunks by performing Personalized PageRank
    over the unified entity-relation graph.
    """

    def __init__(
        self,
        alpha: float = 0.85,
        max_iter: int = 100,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.alpha = alpha
        self.max_iter = max_iter
        self.model_name = model_name
        self.extractor = EntityExtractor()

    def extract_and_match_seeds(
        self,
        query: str,
        graph: nx.DiGraph,
        top_n: int = 5
    ) -> Dict[str, float]:
        """
        Extracts entities from the query and matches them to graph nodes
        via exact matching, normalized matching, and embedding similarity.
        """
        if len(graph) == 0:
            return {}

        seeds: Dict[str, float] = {}
        nodes = list(graph.nodes())
        norm_to_node = {normalize_entity_name(str(n)): str(n) for n in nodes}

        # 1. Extract query entities via LLM / spaCy
        extraction = self.extractor.extract(query)
        extracted_names = [e.name for e in extraction.entities]

        query_lower = query.lower()

        # 2. Check extracted names and substring matches
        for name in extracted_names:
            norm = normalize_entity_name(name)
            if norm in norm_to_node:
                seeds[norm_to_node[norm]] = 2.0
            else:
                for n in nodes:
                    if str(n).lower() in name.lower() or name.lower() in str(n).lower():
                        seeds[str(n)] = 1.5

        # Also search graph node names in query string
        for norm, orig_node in norm_to_node.items():
            if len(norm) >= 3 and norm in query_lower:
                seeds[orig_node] = max(seeds.get(orig_node, 0.0), 1.5)

        # 3. Embedding similarity matching if no seeds resolved
        if not seeds:
            model = EmbeddingModelCache.get_model(self.model_name)
            q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
            node_embs = model.encode([str(n) for n in nodes], convert_to_numpy=True, normalize_embeddings=True)
            sims = np.dot(node_embs, q_emb)

            best_indices = np.argsort(-sims)[:top_n]
            for idx in best_indices:
                if sims[idx] >= 0.40:
                    seeds[str(nodes[idx])] = float(sims[idx])

        # 4. Fallback: highest degree nodes in graph
        if not seeds and nodes:
            degrees = dict(graph.degree())
            top_deg_nodes = sorted(degrees.keys(), key=lambda k: degrees[k], reverse=True)[:top_n]
            for n in top_deg_nodes:
                seeds[str(n)] = 1.0

        return seeds

    def retrieve(
        self,
        query: str,
        graph: nx.DiGraph,
        vector_store: QdrantStore,
        top_k_nodes: int = 5,
        top_k_chunks: int = 5,
        alpha: Optional[float] = None
    ) -> MultimodalRetrievalResult:
        """
        Executes Personalized PageRank retrieval over the multimodal graph.
        Returns top-ranked nodes and backing cross-modal chunks with base64 images.
        """
        start_time = time.perf_counter()
        active_alpha = alpha if alpha is not None else self.alpha

        if len(graph) == 0:
            return MultimodalRetrievalResult(
                query=query,
                execution_time_ms=0.0,
                algorithm="Multimodal PPR"
            )

        # Step 1: Match query seed entities
        seeds = self.extract_and_match_seeds(query, graph)
        valid_seeds = {s: w for s, w in seeds.items() if s in graph}

        # Step 2: Construct personalization vector
        personalization: Dict[Any, float] = {}
        if valid_seeds:
            total_weight = sum(valid_seeds.values())
            for s, w in valid_seeds.items():
                personalization[s] = w / total_weight
        else:
            num_nodes = len(graph)
            personalization = {n: 1.0 / num_nodes for n in graph.nodes()}

        # Step 3: Compute Personalized PageRank
        try:
            ppr_scores = nx.pagerank(
                graph,
                alpha=active_alpha,
                personalization=personalization,
                max_iter=self.max_iter,
                weight="weight"
            )
        except Exception:
            try:
                ppr_scores = nx.pagerank(graph, alpha=active_alpha, personalization=personalization, max_iter=200)
            except Exception:
                ppr_scores = {n: 1.0 / len(graph) for n in graph.nodes()}

        # Step 4: Rank nodes by stationary distribution
        sorted_nodes = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)
        top_nodes_data = []
        target_chunk_ids: Set[str] = set()

        for node_name, score in sorted_nodes[:top_k_nodes]:
            node_attrs = graph.nodes[node_name]
            mods = list(node_attrs.get("modalities", []))
            top_nodes_data.append({
                "name": str(node_name),
                "score": round(score, 5),
                "modalities": mods
            })
            chunk_ids = node_attrs.get("source_chunk_ids", set())
            target_chunk_ids.update(chunk_ids)

        # Step 5: Pull backing chunks from Qdrant by chunk_id
        retrieved_chunks: List[RetrievedChunk] = []

        # Use dense search over Qdrant to get candidate chunks
        model = EmbeddingModelCache.get_model(self.model_name)
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        qdrant_hits = vector_store.search(query_vector=q_emb.tolist(), top_k=vector_store.count() or 10)

        for hit in qdrant_hits:
            payload = hit["payload"]
            cid = payload.get("chunk_id")
            c_text = payload.get("text", "")
            c_modality = Modality(payload.get("modality", "text"))
            c_meta = payload.get("metadata", {})
            c_img_path = payload.get("image_path")
            c_entities = payload.get("entities", [])

            # Compute combined score: vector similarity + graph PPR mass from constituent entities
            graph_score = sum(ppr_scores.get(e, 0.0) for e in c_entities)
            is_backing_chunk = 1.0 if cid in target_chunk_ids else 0.0
            combined_score = float(0.5 * hit["score"] + 0.3 * (graph_score * 10.0) + 0.2 * is_backing_chunk)

            # Load actual image bytes as base64 if visual
            img_b64 = load_image_as_base64(c_img_path) if c_modality == Modality.IMAGE else None

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=cid,
                    modality=c_modality,
                    text=c_text,
                    score=round(combined_score, 4),
                    metadata=c_meta,
                    image_path=c_img_path,
                    image_base64=img_b64,
                    entities=c_entities
                )
            )

        # Sort chunks descending by combined score
        retrieved_chunks.sort(key=lambda c: c.score, reverse=True)
        final_chunks = retrieved_chunks[:top_k_chunks]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return MultimodalRetrievalResult(
            query=query,
            top_nodes=top_nodes_data,
            chunks=final_chunks,
            seed_entities=list(valid_seeds.keys()),
            execution_time_ms=round(elapsed_ms, 2),
            algorithm="Multimodal PPR"
        )
