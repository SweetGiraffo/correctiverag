"""
Cross-Modal Entity Resolver.
Deduplicates entities across modalities using normalized name matching and embedding similarity (>0.90).
Incrementally updates the unified NetworkX graph.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import re
import numpy as np
import networkx as nx
from src.retrieval.vector_retriever import EmbeddingModelCache
from .base import IngestedChunk, Modality


def normalize_entity_name(name: str) -> str:
    """Normalizes entity name by lowercasing and removing noise."""
    name = name.strip()
    # Remove leading articles
    name = re.sub(r"^(the|a|an)\s+", "", name, flags=re.IGNORECASE)
    # Remove excess whitespace and punctuation
    name = re.sub(r"[^\w\s-]", "", name)
    return " ".join(name.lower().split())


class EntityResolver:
    """
    Resolves entity mentions across modalities into canonical nodes in the NetworkX graph.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.90,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name

        # Mapping: normalized_name -> canonical_name
        self.canonical_entities: Dict[str, str] = {}
        # List of canonical names and their normalized embeddings
        self.canonical_names: List[str] = []
        self.canonical_embeddings: Optional[np.ndarray] = None

    def resolve(self, raw_name: str, entity_type: str = "GENERAL") -> str:
        """
        Resolves an entity name to an existing canonical name or creates a new one.
        """
        raw_clean = raw_name.strip()
        if not raw_clean:
            return ""

        norm_name = normalize_entity_name(raw_clean)
        if not norm_name:
            return raw_clean

        # Step 1: Direct normalized lookup
        if norm_name in self.canonical_entities:
            return self.canonical_entities[norm_name]

        # Step 2: Embedding cosine similarity check (> 0.90)
        model = EmbeddingModelCache.get_model(self.model_name)
        new_emb = model.encode([raw_clean], convert_to_numpy=True, normalize_embeddings=True)[0]

        if self.canonical_embeddings is not None and len(self.canonical_names) > 0:
            sims = np.dot(self.canonical_embeddings, new_emb)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

            if best_sim >= self.similarity_threshold:
                matched_canonical = self.canonical_names[best_idx]
                self.canonical_entities[norm_name] = matched_canonical
                return matched_canonical

        # Step 3: Register as new canonical entity
        canonical_name = raw_clean
        self.canonical_entities[norm_name] = canonical_name
        self.canonical_names.append(canonical_name)

        if self.canonical_embeddings is None:
            self.canonical_embeddings = np.array([new_emb])
        else:
            self.canonical_embeddings = np.vstack([self.canonical_embeddings, new_emb])

        return canonical_name

    def update_graph_with_chunk(
        self,
        graph: nx.DiGraph,
        chunk: IngestedChunk,
        extracted_entities: List[Any],
        extracted_triples: List[Any]
    ):
        """
        Incrementally registers resolved entities and triples into the NetworkX graph.
        """
        resolved_entities_in_chunk = []

        # Add / update entity nodes
        for ent in extracted_entities:
            raw_name = ent.name if hasattr(ent, "name") else str(ent)
            ent_type = ent.type if hasattr(ent, "type") else "GENERAL"
            canon = self.resolve(raw_name, ent_type)
            if not canon:
                continue

            resolved_entities_in_chunk.append(canon)

            if canon not in graph:
                graph.add_node(
                    canon,
                    label=canon,
                    type=ent_type,
                    modalities={chunk.modality.value},
                    source_chunk_ids={chunk.chunk_id},
                    mention_count=1,
                    image_paths=[chunk.image_path] if chunk.image_path else []
                )
            else:
                node = graph.nodes[canon]
                node.setdefault("modalities", set()).add(chunk.modality.value)
                node.setdefault("source_chunk_ids", set()).add(chunk.chunk_id)
                node["mention_count"] = node.get("mention_count", 0) + 1
                if chunk.image_path and chunk.image_path not in node.setdefault("image_paths", []):
                    node["image_paths"].append(chunk.image_path)

        # Add / update relation triples
        for trip in extracted_triples:
            s_raw = trip.subject if hasattr(trip, "subject") else trip[0]
            r_raw = trip.relation if hasattr(trip, "relation") else trip[1]
            o_raw = trip.object if hasattr(trip, "object") else trip[2]

            s_canon = self.resolve(s_raw)
            o_canon = self.resolve(o_raw)

            if s_canon and o_canon and s_canon != o_canon:
                if graph.has_edge(s_canon, o_canon):
                    graph[s_canon][o_canon]["weight"] = graph[s_canon][o_canon].get("weight", 1.0) + 1.0
                else:
                    graph.add_edge(
                        s_canon,
                        o_canon,
                        relation=r_raw,
                        weight=1.5,
                        modality=chunk.modality.value
                    )

        # Add co-occurrence edges between entities co-present in the same chunk
        unique_chunk_ents = list(set(resolved_entities_in_chunk[:10]))
        for i in range(len(unique_chunk_ents)):
            for j in range(i + 1, min(i + 3, len(unique_chunk_ents))):
                e1, e2 = unique_chunk_ents[i], unique_chunk_ents[j]
                if not graph.has_edge(e1, e2):
                    graph.add_edge(e1, e2, relation="co_occurs_in", weight=0.5, modality=chunk.modality.value)
                    graph.add_edge(e2, e1, relation="co_occurs_in", weight=0.5, modality=chunk.modality.value)
