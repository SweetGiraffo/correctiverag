"""
Entity Graph Construction using NetworkX.
Constructs entity-relation graphs from passages, dataset evidence triples, and spaCy NER.
"""

from typing import List, Dict, Set, Any, Optional, Tuple
import networkx as nx
import spacy
from src.dataset import MultiHopSample, Passage, EvidenceTriple


class EntityGraphBuilder:
    """
    Constructs and manages NetworkX entity-relation graphs for multi-hop retrieval.
    Nodes can be entities and passages; edges represent semantic relations,
    mentions, and co-occurrences.
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

    def build_graph_for_sample(
        self,
        sample: MultiHopSample,
        include_ner: bool = True,
        connect_passages: bool = True
    ) -> nx.DiGraph:
        """
        Builds a directed, weighted NetworkX graph for a single MultiHopSample
        using its candidate passages and evidence triples.
        """
        self._ensure_nlp()
        G = nx.DiGraph()

        # Inverted index: entity -> set of passage titles
        entity_to_passages: Dict[str, Set[str]] = {}
        passage_to_entities: Dict[str, Set[str]] = {}

        for passage in sample.context_passages:
            p_title = passage.title
            passage_to_entities[p_title] = set()

            if connect_passages:
                # Add passage node
                G.add_node(
                    f"doc:{p_title}",
                    type="passage",
                    title=p_title,
                    text=passage.text[:200]
                )

            # Passage title is intrinsically an entity node
            G.add_node(p_title, type="entity", label=p_title)
            entity_to_passages.setdefault(p_title, set()).add(p_title)
            passage_to_entities[p_title].add(p_title)

            if connect_passages:
                # Bi-directional link between passage node and its title entity
                G.add_edge(f"doc:{p_title}", p_title, relation="title_of", weight=1.5)
                G.add_edge(p_title, f"doc:{p_title}", relation="documented_in", weight=1.5)

            # Extract entities from passage text via spaCy NER
            if include_ner and self.nlp:
                doc = self.nlp(passage.text)
                passage_entities = []
                for ent in doc.ents:
                    ent_text = ent.text.strip()
                    # Filter out purely numeric or single-character noise
                    if len(ent_text) > 1 and not ent_text.isdigit():
                        passage_entities.append(ent_text)
                        if ent_text not in G:
                            G.add_node(ent_text, type="entity", label=ent.label_)
                        entity_to_passages.setdefault(ent_text, set()).add(p_title)
                        passage_to_entities[p_title].add(ent_text)

                        if connect_passages:
                            G.add_edge(ent_text, f"doc:{p_title}", relation="mentions", weight=1.0)
                            G.add_edge(f"doc:{p_title}", ent_text, relation="contains", weight=1.0)

                # Connect co-occurring entities within the passage with a relation edge
                unique_ents = list(set(passage_entities[:15]))  # limit density
                for i in range(len(unique_ents)):
                    for j in range(i + 1, min(i + 4, len(unique_ents))):
                        e1, e2 = unique_ents[i], unique_ents[j]
                        if not G.has_edge(e1, e2):
                            G.add_edge(e1, e2, relation="co_occurs_with", weight=0.5)
                            G.add_edge(e2, e1, relation="co_occurs_with", weight=0.5)

        # Inject ground-truth or high-confidence evidence relations
        for ev in sample.evidences:
            s, r, o = ev.subject.strip(), ev.relation.strip(), ev.object.strip()
            if s and o:
                if s not in G:
                    G.add_node(s, type="entity", label=s)
                if o not in G:
                    G.add_node(o, type="entity", label=o)

                # Directed edge with higher weight for verified relations
                G.add_edge(s, o, relation=r, weight=2.0)
                # Reverse edge for bi-directional diffusion with slightly lower weight
                if not G.has_edge(o, s):
                    G.add_edge(o, s, relation=f"inv_{r}", weight=1.2)

        # Attach indexing metadata onto graph
        G.graph["entity_to_passages"] = {k: list(v) for k, v in entity_to_passages.items()}
        G.graph["passage_to_entities"] = {k: list(v) for k, v in passage_to_entities.items()}
        G.graph["sample_id"] = sample.id
        G.graph["question"] = sample.question

        return G

    def build_merged_graph(self, samples: List[MultiHopSample]) -> nx.DiGraph:
        """Builds a composite graph merging multiple samples."""
        composite_graph = nx.DiGraph()
        composite_entity_to_passages: Dict[str, Set[str]] = {}

        for sample in samples:
            sub_g = self.build_graph_for_sample(sample)
            composite_graph = nx.compose(composite_graph, sub_g)

            sub_e2p = sub_g.graph.get("entity_to_passages", {})
            for e, p_list in sub_e2p.items():
                composite_entity_to_passages.setdefault(e, set()).update(p_list)

        composite_graph.graph["entity_to_passages"] = {k: list(v) for k, v in composite_entity_to_passages.items()}
        return composite_graph

    @staticmethod
    def get_subgraph_for_visualization(
        G: nx.DiGraph,
        seed_nodes: List[str],
        depth: int = 2,
        max_nodes: int = 40
    ) -> Dict[str, Any]:
        """
        Extracts an ego-subgraph around seed nodes formatted for JSON / D3 / Streamlit visualization.
        """
        active_seeds = [s for s in seed_nodes if s in G]
        if not active_seeds:
            # Fallback to top degree nodes
            active_seeds = [n for n, _ in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:3]]

        subgraph_nodes = set(active_seeds)
        for s in active_seeds:
            try:
                ego = nx.ego_graph(G, s, radius=depth, undirected=True)
                subgraph_nodes.update(ego.nodes())
            except Exception:
                continue

        if len(subgraph_nodes) > max_nodes:
            # Sort by degree and prune
            degrees = dict(G.degree(subgraph_nodes))
            sorted_nodes = sorted(degrees.keys(), key=lambda k: degrees[k], reverse=True)
            subgraph_nodes = set(sorted_nodes[:max_nodes])

        sub_g = G.subgraph(subgraph_nodes)

        nodes_data = []
        for n, data in sub_g.nodes(data=True):
            is_seed = n in active_seeds
            nodes_data.append({
                "id": n,
                "label": n,
                "type": data.get("type", "entity"),
                "is_seed": is_seed
            })

        edges_data = []
        for u, v, data in sub_g.edges(data=True):
            edges_data.append({
                "source": u,
                "target": v,
                "relation": data.get("relation", "connected_to"),
                "weight": data.get("weight", 1.0)
            })

        return {"nodes": nodes_data, "edges": edges_data}
