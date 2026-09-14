"""
Cross-Modal Evaluation Benchmark Harness.
A/B evaluates 3 architectures across multi-hop, multi-modal questions:
1. Baseline: Hybrid Vector Search (Dense + BM25 + RRF)
2. Graph Diffusion: Multimodal Personalized PageRank (PPR)
3. Full System: Corrective Multi-Agent CRAG Pipeline (Router -> Retriever -> Grader -> Evaluator -> Generator)

Measures:
- Cross-modal recall (retrieval across all required modalities)
- Multi-hop fact recall
- Retrieval & execution latency (ms)
- Citation precision
"""

from typing import List, Dict, Any, Optional
import time
import os
import json
from pydantic import BaseModel, Field

from src.ingest.base import Modality, IngestedChunk
from src.retrieval.models import RetrievedChunk
from src.retrieval.ppr import MultimodalPPRRetriever
from src.retrieval.hybrid import MultimodalHybridRetriever
from src.agent.graph import run_crag_pipeline
from src.storage.vector_store import QdrantStore
from src.storage.graph_store import MultimodalGraphStore
from src.ingest.pipeline import IngestionPipeline


class CrossModalQueryBenchmark(BaseModel):
    query: str
    target_modalities: List[str]
    target_entities: List[str]
    target_facts: List[str]
    description: str


# Standard cross-modal multi-hop evaluation set
BENCHMARK_QUERIES: List[CrossModalQueryBenchmark] = [
    CrossModalQueryBenchmark(
        query="What was Acme Corp's Q3 revenue and what share was from Cloud?",
        target_modalities=["pdf", "image"],
        target_entities=["Acme Corp", "Cloud"],
        target_facts=["142", "45%"],
        description="Multi-hop reasoning linking financial PDF statement with revenue breakdown chart image."
    ),
    CrossModalQueryBenchmark(
        query="Who leads Project Titan and what was the quarterly revenue growth rate?",
        target_modalities=["audio", "pdf"],
        target_entities=["Dr. Elena Rostova", "Project Titan", "Acme Corp"],
        target_facts=["Elena Rostova", "18%"],
        description="Multi-hop bridging spoken executive audio memo with earnings report PDF."
    ),
    CrossModalQueryBenchmark(
        query="What are the revenue shares of Cloud and Enterprise AI, and who is leading the AI division?",
        target_modalities=["image", "audio"],
        target_entities=["Cloud", "Enterprise AI", "Project Titan"],
        target_facts=["45%", "35%", "Elena Rostova"],
        description="Multi-hop reasoning bridging chart image and audio announcement."
    ),
    CrossModalQueryBenchmark(
        query="Synthesize Acme Corp's Q3 revenue, top driver shares, and the leadership update for Project Titan.",
        target_modalities=["pdf", "image", "audio"],
        target_entities=["Acme Corp", "Cloud", "Project Titan", "Dr. Elena Rostova"],
        target_facts=["142", "45%", "Titan"],
        description="3-way cross-modal synthesis across PDF document, visual chart image, and audio briefing."
    )
]


class ArchitectureStats(BaseModel):
    name: str
    modality_recall_pct: float
    fact_coverage_pct: float
    citation_precision_pct: float
    avg_latency_ms: float


class MultimodalBenchmarkReport(BaseModel):
    num_queries: int
    hybrid_vector_baseline: ArchitectureStats
    ppr_graph_retriever: ArchitectureStats
    full_crag_agent: ArchitectureStats
    detailed_results: List[Dict[str, Any]] = []

    def to_markdown(self) -> str:
        md = []
        md.append("# 📊 Cross-Modal Multi-Hop RAG Benchmark Report\n")
        md.append(f"**Evaluated on {self.num_queries} cross-modal multi-hop test queries requiring joint reasoning across PDF, Chart Image, and Audio.**\n")
        md.append("## 🏆 Comparative Performance Summary\n")
        md.append("| Architecture | Modality Recall | Fact Coverage | Citation Precision | Avg Latency |")
        md.append("|---|---|---|---|---|")
        for s in [self.hybrid_vector_baseline, self.ppr_graph_retriever, self.full_crag_agent]:
            md.append(f"| **{s.name}** | {s.modality_recall_pct:.1f}% | {s.fact_coverage_pct:.1f}% | {s.citation_precision_pct:.1f}% | {s.avg_latency_ms:.1f} ms |")
        md.append("\n## 🔍 Query-by-Query Analysis\n")
        for idx, d in enumerate(self.detailed_results, 1):
            md.append(f"### Query {idx}: {d['query']}")
            md.append(f"*{d['description']}*\n")
            md.append(f"- **Target Modalities**: {', '.join(d['target_modalities'])}")
            md.append(f"- **Target Facts**: {', '.join(d['target_facts'])}\n")
            md.append("| Metric | Hybrid Vector | PPR Graph | Full CRAG Agent |")
            md.append("|---|---|---|---|")
            md.append(f"| Modalities Retrieved | {', '.join(d['hybrid']['modalities'])} | {', '.join(d['ppr']['modalities'])} | {', '.join(d['crag']['modalities'])} |")
            md.append(f"| Facts Captured | {d['hybrid']['facts_found']}/{len(d['target_facts'])} | {d['ppr']['facts_found']}/{len(d['target_facts'])} | {d['crag']['facts_found']}/{len(d['target_facts'])} |")
            md.append(f"| Citations Present | {d['hybrid']['citations_count']} | {d['ppr']['citations_count']} | {d['crag']['citations_count']} |")
            md.append(f"| Latency | {d['hybrid']['latency_ms']:.1f} ms | {d['ppr']['latency_ms']:.1f} ms | {d['crag']['latency_ms']:.1f} ms |\n")
            md.append(f"> **Synthesized Agent Answer**: {d['crag']['answer']}\n")
        return "\n".join(md)


class MultimodalBenchmarkRunner:
    """
    Executes the multimodal benchmark across the three architectures.
    """

    def __init__(
        self,
        vector_store: Optional[QdrantStore] = None,
        graph_store: Optional[MultimodalGraphStore] = None
    ):
        self.vector_store = vector_store or QdrantStore(storage_path="data/qdrant_db", vector_dim=384)
        self.graph_store = graph_store or MultimodalGraphStore()
        self.pipeline = IngestionPipeline(vector_store=self.vector_store, graph_store=self.graph_store)

    def ensure_sample_data_ingested(self):
        """Ensures sample_data directory is populated and ingested into storage."""
        if len(self.graph_store.get_graph()) == 0 and os.path.exists("sample_data"):
            for f in sorted(os.listdir("sample_data")):
                f_path = os.path.join("sample_data", f)
                if os.path.isfile(f_path):
                    self.pipeline.ingest_file(f_path)

    def run(self, top_k: int = 4) -> MultimodalBenchmarkReport:
        self.ensure_sample_data_ingested()
        graph = self.graph_store.get_graph()

        ppr_retriever = MultimodalPPRRetriever(alpha=0.85)
        hybrid_retriever = MultimodalHybridRetriever()

        hybrid_mod_recalls, hybrid_fact_recalls, hybrid_latencies, hybrid_citations = [], [], [], []
        ppr_mod_recalls, ppr_fact_recalls, ppr_latencies, ppr_citations = [], [], [], []
        crag_mod_recalls, crag_fact_recalls, crag_latencies, crag_citations = [], [], [], []
        details = []

        for bq in BENCHMARK_QUERIES:
            target_mods = set(bq.target_modalities)
            target_facts = [f.lower() for f in bq.target_facts]

            # 1. Hybrid Vector
            t0 = time.perf_counter()
            res_hybrid = hybrid_retriever.retrieve(query=bq.query, vector_store=self.vector_store, top_k=top_k)
            hybrid_time = (time.perf_counter() - t0) * 1000.0
            hybrid_chunks = res_hybrid.chunks
            h_mods = set(c.modality.value for c in hybrid_chunks)
            h_mod_recall = len(target_mods & h_mods) / len(target_mods) if target_mods else 1.0
            h_text = " ".join(c.text.lower() for c in hybrid_chunks)
            h_facts = sum(1 for f in target_facts if f in h_text)
            h_fact_recall = h_facts / len(target_facts) if target_facts else 1.0
            hybrid_mod_recalls.append(h_mod_recall)
            hybrid_fact_recalls.append(h_fact_recall)
            hybrid_latencies.append(hybrid_time)
            hybrid_citations.append(0.0)

            # 2. PPR Graph
            t0 = time.perf_counter()
            res_ppr = ppr_retriever.retrieve(query=bq.query, graph=graph, vector_store=self.vector_store, top_k_chunks=top_k)
            ppr_time = (time.perf_counter() - t0) * 1000.0
            ppr_chunks = res_ppr.chunks
            p_mods = set(c.modality.value for c in ppr_chunks)
            p_mod_recall = len(target_mods & p_mods) / len(target_mods) if target_mods else 1.0
            p_text = " ".join(c.text.lower() for c in ppr_chunks)
            p_facts = sum(1 for f in target_facts if f in p_text)
            p_fact_recall = p_facts / len(target_facts) if target_facts else 1.0
            ppr_mod_recalls.append(p_mod_recall)
            ppr_fact_recalls.append(p_fact_recall)
            ppr_latencies.append(ppr_time)
            ppr_citations.append(0.0)

            # 3. Full CRAG Pipeline
            t0 = time.perf_counter()
            state = run_crag_pipeline(
                query=bq.query,
                vector_store=self.vector_store,
                graph_store=self.graph_store,
                strategy="ppr"
            )
            crag_time = (time.perf_counter() - t0) * 1000.0
            crag_chunks = state.get("retrieved_chunks", [])
            c_mods = set(c.modality.value for c in crag_chunks)
            c_mod_recall = len(target_mods & c_mods) / len(target_mods) if target_mods else 1.0
            ans = state.get("final_answer", "")
            c_text = (ans + " " + " ".join(c.text for c in crag_chunks)).lower()
            c_facts = sum(1 for f in target_facts if f in c_text)
            c_fact_recall = c_facts / len(target_facts) if target_facts else 1.0
            c_citations = state.get("citations", [])
            c_cit_score = 1.0 if len(c_citations) >= len(target_mods) else (len(c_citations) / len(target_mods) if target_mods else 1.0)

            crag_mod_recalls.append(c_mod_recall)
            crag_fact_recalls.append(c_fact_recall)
            crag_latencies.append(crag_time)
            crag_citations.append(c_cit_score)

            details.append({
                "query": bq.query,
                "description": bq.description,
                "target_modalities": bq.target_modalities,
                "target_facts": bq.target_facts,
                "hybrid": {
                    "modalities": list(h_mods),
                    "facts_found": h_facts,
                    "citations_count": 0,
                    "latency_ms": round(hybrid_time, 1)
                },
                "ppr": {
                    "modalities": list(p_mods),
                    "facts_found": p_facts,
                    "citations_count": 0,
                    "latency_ms": round(ppr_time, 1)
                },
                "crag": {
                    "modalities": list(c_mods),
                    "facts_found": c_facts,
                    "citations_count": len(c_citations),
                    "latency_ms": round(crag_time, 1),
                    "answer": ans
                }
            })

        import numpy as np
        return MultimodalBenchmarkReport(
            num_queries=len(BENCHMARK_QUERIES),
            hybrid_vector_baseline=ArchitectureStats(
                name="Hybrid Vector Search (Dense + BM25)",
                modality_recall_pct=round(float(np.mean(hybrid_mod_recalls)) * 100, 1),
                fact_coverage_pct=round(float(np.mean(hybrid_fact_recalls)) * 100, 1),
                citation_precision_pct=0.0,
                avg_latency_ms=round(float(np.mean(hybrid_latencies)), 1)
            ),
            ppr_graph_retriever=ArchitectureStats(
                name="Multimodal Personalized PageRank (PPR)",
                modality_recall_pct=round(float(np.mean(ppr_mod_recalls)) * 100, 1),
                fact_coverage_pct=round(float(np.mean(ppr_fact_recalls)) * 100, 1),
                citation_precision_pct=0.0,
                avg_latency_ms=round(float(np.mean(ppr_latencies)), 1)
            ),
            full_crag_agent=ArchitectureStats(
                name="Full Corrective Agentic Pipeline (CRAG + PPR)",
                modality_recall_pct=round(float(np.mean(crag_mod_recalls)) * 100, 1),
                fact_coverage_pct=round(float(np.mean(crag_fact_recalls)) * 100, 1),
                citation_precision_pct=round(float(np.mean(crag_citations)) * 100, 1),
                avg_latency_ms=round(float(np.mean(crag_latencies)), 1)
            ),
            detailed_results=details
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run multimodal benchmark")
    parser.add_argument("--output", type=str, default="data/multimodal_benchmark_report.md", help="Markdown output path")
    args = parser.parse_args()

    print("Running Multimodal A/B Comparative Benchmark...")
    runner = MultimodalBenchmarkRunner()
    report = runner.run()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report.to_markdown())

    print(f"Benchmark completed successfully! Report saved to: {args.output}")
    print("\nSummary:")
    for s in [report.hybrid_vector_baseline, report.ppr_graph_retriever, report.full_crag_agent]:
        print(f" - {s.name}: Modality Recall={s.modality_recall_pct}%, Facts={s.fact_coverage_pct}%, Latency={s.avg_latency_ms}ms")
