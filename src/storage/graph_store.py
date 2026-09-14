"""
NetworkX Multimodal Graph Store.
Handles persistent storage (Pickle/GraphML) and cross-modal topological indexing.
"""

from typing import Dict, Any, List, Optional
import os
import pickle
import networkx as nx


class MultimodalGraphStore:
    """
    Manages the global cross-modal entity-relation NetworkX graph.
    """

    def __init__(self, storage_path: str = "data/multimodal_graph.pkl"):
        self.storage_path = storage_path
        self.graph = nx.DiGraph()
        self.load()

    def get_graph(self) -> nx.DiGraph:
        return self.graph

    def save(self, path: Optional[str] = None):
        """Serializes graph to disk using pickle."""
        target = path or self.storage_path
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        with open(target, "wb") as f:
            pickle.dump(self.graph, f)

    def load(self, path: Optional[str] = None):
        """Loads serialized graph from disk if it exists."""
        target = path or self.storage_path
        if os.path.exists(target):
            try:
                with open(target, "rb") as f:
                    self.graph = pickle.load(f)
            except Exception:
                self.graph = nx.DiGraph()
        else:
            self.graph = nx.DiGraph()

    def clear(self):
        self.graph.clear()
        if os.path.exists(self.storage_path):
            try:
                os.remove(self.storage_path)
            except Exception:
                pass

    def get_stats(self) -> Dict[str, Any]:
        """Calculates topological statistics across modalities."""
        total_nodes = len(self.graph.nodes)
        total_edges = len(self.graph.edges)

        modality_counts: Dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            modalities = data.get("modalities", set())
            if isinstance(modalities, (set, list)):
                for m in modalities:
                    modality_counts[m] = modality_counts.get(m, 0) + 1
            else:
                modality_counts[str(modalities)] = modality_counts.get(str(modalities), 0) + 1

        density = nx.density(self.graph) if total_nodes > 1 else 0.0

        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "density": round(density, 4),
            "modality_breakdown": modality_counts
        }

    def get_subgraph(self, seeds: List[str], depth: int = 2, max_nodes: int = 40) -> Dict[str, Any]:
        """Extracts ego-subgraph formatted for PyVis / Streamlit visualization."""
        active_seeds = [s for s in seeds if s in self.graph]
        if not active_seeds:
            # Pick top-degree nodes as fallback
            active_seeds = [n for n, _ in sorted(self.graph.degree(), key=lambda x: x[1], reverse=True)[:3]]

        sub_nodes = set(active_seeds)
        for s in active_seeds:
            try:
                ego = nx.ego_graph(self.graph, s, radius=depth, undirected=True)
                sub_nodes.update(ego.nodes())
            except Exception:
                continue

        if len(sub_nodes) > max_nodes:
            degrees = dict(self.graph.degree(sub_nodes))
            sorted_nodes = sorted(degrees.keys(), key=lambda k: degrees[k], reverse=True)
            sub_nodes = set(sorted_nodes[:max_nodes])

        sub_g = self.graph.subgraph(sub_nodes)

        nodes_data = []
        for n, data in sub_g.nodes(data=True):
            mods = list(data.get("modalities", []))
            nodes_data.append({
                "id": str(n),
                "label": str(n),
                "type": data.get("type", "GENERAL"),
                "modalities": mods,
                "is_seed": n in active_seeds
            })

        edges_data = []
        for u, v, data in sub_g.edges(data=True):
            edges_data.append({
                "source": str(u),
                "target": str(v),
                "relation": data.get("relation", "relates_to"),
                "weight": data.get("weight", 1.0),
                "modality": data.get("modality", "unknown")
            })

        return {"nodes": nodes_data, "edges": edges_data}
