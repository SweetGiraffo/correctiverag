"""
Personalized PageRank (PPR) Graph Retriever.
Extracts seed entities from the question, runs personalized random walks over the
entity-relation graph, and ranks passages in a single diffusion step.
"""

from typing import List, Dict, Tuple, Optional, Set
import time
import networkx as nx
import spacy
from pydantic import BaseModel, Field
from src.dataset import Passage, MultiHopSample


class PPRRetrievalResult(BaseModel):
    """Encapsulates output and diagnostic metrics from PPR retrieval."""
    passages: List[Passage]
    scores: List[float]
    seed_entities: List[str]
    ppr_node_scores: Dict[str, float] = Field(default_factory=dict)
    execution_time_ms: float
    vector_search_calls: int = 0
    algorithm: str = "Personalized PageRank"


class PPRRetriever:
    """
    Graph-based retriever using Personalized PageRank on NetworkX graphs.
    """

    def __init__(
        self,
        alpha: float = 0.85,
        max_iter: int = 100,
        spacy_model: str = "en_core_web_sm"
    ):
        self.alpha = alpha
        self.max_iter = max_iter
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

    def extract_seed_entities(self, query: str, graph: nx.DiGraph) -> List[str]:
        """
        Extracts seed entities mentioned in the query text that match nodes in the graph.
        Combines spaCy NER entity matching with graph node substring matching.
        """
        self._ensure_nlp()
        matched_seeds: Set[str] = set()
        graph_nodes = set(graph.nodes())

        # 1. Exact or substring match against known entity nodes in graph
        query_lower = query.lower()
        for node in graph_nodes:
            if not str(node).startswith("doc:"):
                # Match reasonable length entity names
                node_str = str(node)
                if len(node_str) >= 3 and node_str.lower() in query_lower:
                    matched_seeds.add(node_str)

        # 2. Extract NER entities from query
        if self.nlp:
            doc = self.nlp(query)
            for ent in doc.ents:
                ent_text = ent.text.strip()
                if ent_text in graph_nodes:
                    matched_seeds.add(ent_text)
                else:
                    # Fuzzy match against graph nodes
                    for node in graph_nodes:
                        if ent_text.lower() in str(node).lower():
                            matched_seeds.add(node)

        # 3. Fallback: if no seeds matched, use graph title entities mentioned or highest degree entities
        if not matched_seeds:
            for node, data in graph.nodes(data=True):
                if data.get("type") == "entity" and str(node).lower() in query_lower:
                    matched_seeds.add(node)

        if not matched_seeds:
            # Fallback to top degree entity nodes in the graph
            entity_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "entity"]
            if entity_nodes:
                degrees = dict(graph.degree(entity_nodes))
                top_entity = max(degrees.keys(), key=lambda k: degrees[k])
                matched_seeds.add(top_entity)

        return list(matched_seeds)

    def retrieve(
        self,
        query: str,
        graph: nx.DiGraph,
        passages: List[Passage],
        top_k: int = 3,
        explicit_seeds: Optional[List[str]] = None
    ) -> PPRRetrievalResult:
        """
        Executes Personalized PageRank retrieval over the graph.
        Returns top-k ranked passages and execution metrics.
        """
        start_time = time.perf_counter()

        if len(graph) == 0 or len(passages) == 0:
            return PPRRetrievalResult(
                passages=[],
                scores=[],
                seed_entities=[],
                execution_time_ms=0.0,
                vector_search_calls=0
            )

        # 1. Determine seed entities
        seeds = explicit_seeds if explicit_seeds else self.extract_seed_entities(query, graph)
        valid_seeds = [s for s in seeds if s in graph]

        # 2. Build personalization dict
        personalization: Dict[str, float] = {}
        if valid_seeds:
            weight = 1.0 / len(valid_seeds)
            for s in valid_seeds:
                personalization[s] = weight
        else:
            # Uniform over all nodes if no seed resolved
            num_nodes = len(graph)
            personalization = {n: 1.0 / num_nodes for n in graph.nodes()}

        # 3. Run NetworkX Personalized PageRank
        try:
            ppr_scores = nx.pagerank(
                graph,
                alpha=self.alpha,
                personalization=personalization,
                max_iter=self.max_iter,
                weight="weight"
            )
        except Exception:
            # In case of non-convergence, fallback to unweighted or simple degree
            try:
                ppr_scores = nx.pagerank(
                    graph,
                    alpha=self.alpha,
                    personalization=personalization,
                    max_iter=200
                )
            except Exception:
                ppr_scores = {n: 1.0 / len(graph) for n in graph.nodes()}

        # 4. Score passages based on direct passage node score and constituent entities
        passage_to_entities = graph.graph.get("passage_to_entities", {})
        scored_passages: List[Tuple[Passage, float]] = []

        for p in passages:
            doc_node = f"doc:{p.title}"
            score = 0.0

            # Direct passage node score
            if doc_node in ppr_scores:
                score += ppr_scores[doc_node] * 2.0

            # Title entity score
            if p.title in ppr_scores:
                score += ppr_scores[p.title] * 2.0

            # Mentioned entities scores
            ents = passage_to_entities.get(p.title, [])
            for e in ents:
                if e in ppr_scores:
                    score += ppr_scores[e]

            scored_passages.append((p, score))

        # Sort descending by score
        scored_passages.sort(key=lambda x: x[1], reverse=True)
        top_results = scored_passages[:top_k]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return PPRRetrievalResult(
            passages=[p for p, _ in top_results],
            scores=[round(s, 6) for _, s in top_results],
            seed_entities=valid_seeds,
            ppr_node_scores={k: round(v, 6) for k, v in sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)[:15]},
            execution_time_ms=round(elapsed_ms, 2),
            vector_search_calls=0
        )
